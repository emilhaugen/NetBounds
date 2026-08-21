"""Streaming utilities for selected third-order derivative difference bounds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from operator import index as as_index
from typing import Any

import torch

from .common import (
    activation_derivative_bounds,
    normalize_input_batch,
    preactivation_delta,
    validate_nonnegative_bound,
)
from .first_order import FirstOrderBoundAccumulator
from .second_order import SecondOrderBoundAccumulator


@dataclass
class ThirdOrderBoundsResult:
    """Batched result of the selected third-order bound computation."""

    # Flattened symmetric triples. For index t, triples[t] == (a, b, c),
    # with a <= b <= c.
    triples: tuple[tuple[int, int, int], ...]

    # Flattened symmetric Hessian pairs needed by ``triples``.
    pairs: tuple[tuple[int, int], ...]

    # Scalar output shape: (N, T).
    base_third_derivatives: torch.Tensor
    third_derivative_bounds: torch.Tensor

    # Optional debug/cache outputs. Kept as None unless return_cache=True.
    # Q has shape (len(model.AB.M_RANGE), L, N, width), ordered by activation derivative.
    Q: torch.Tensor | None = None
    z: torch.Tensor | None = None
    preactivation_gradients: torch.Tensor | None = None
    preactivation_error_bounds: torch.Tensor | None = None
    preactivation_hessians: torch.Tensor | None = None
    preactivation_hessian_error_bounds: torch.Tensor | None = None
    activation_hessians: torch.Tensor | None = None
    activation_hessian_error_bounds: torch.Tensor | None = None
    preactivation_third_derivatives: torch.Tensor | None = None
    preactivation_third_derivative_error_bounds: torch.Tensor | None = None
    activation_third_derivatives: torch.Tensor | None = None
    activation_third_derivative_error_bounds: torch.Tensor | None = None
    eps: float = 0.0
    n_taylor: int | None = None


def make_symmetric_triples(input_dim: int) -> tuple[tuple[int, int, int], ...]:
    """Return all zero-based symmetric third-order triples for ``input_dim``."""
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    return tuple(
        (a, b, c)
        for a in range(input_dim)
        for b in range(a, input_dim)
        for c in range(b, input_dim)
    )


def resolve_third_order_multi_indices(
    model: Any,
    multi_indices: str | Sequence[Sequence[int]],
) -> tuple[tuple[int, int, int], ...]:
    """Normalize and validate selected order-3 multi-indices.

    Custom indices are zero-based triples. Each triple is mapped to its
    symmetric representative, and duplicates are removed in first-seen order.
    """
    input_dim = model.input_dim
    if isinstance(multi_indices, str):
        if multi_indices == "full_input":
            return make_symmetric_triples(input_dim)
        if multi_indices == "spatial_only":
            return make_symmetric_triples(model.d)
        raise ValueError(
            "multi_indices must be one of 'full_input', 'spatial_only', "
            "or a nonempty sequence of index triples."
        )

    try:
        raw_triples = tuple(multi_indices)
    except TypeError as exc:
        raise ValueError(
            "multi_indices must be a nonempty sequence of index triples."
        ) from exc

    if not raw_triples:
        raise ValueError("multi_indices must not be empty.")

    triples: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for raw_triple in raw_triples:
        if isinstance(raw_triple, str | bytes):
            raise ValueError("Each multi-index must be a triple of integers.")

        try:
            triple = tuple(raw_triple)
        except TypeError as exc:
            raise ValueError(
                "Each multi-index must be a triple of integers."
            ) from exc

        if len(triple) != 3:
            raise ValueError("Each multi-index must contain exactly three entries.")

        normalized = tuple(
            sorted(_validate_coordinate(value, input_dim) for value in triple)
        )

        if normalized not in seen:
            seen.add(normalized)
            triples.append(normalized)

    return tuple(triples)


def required_pairs_for_triples(
    triples: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Return the selected Hessian pairs required by ``triples``."""
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for a, b, c in triples:
        for pair in ((a, b), (a, c), (b, c)):
            normalized = pair if pair[0] <= pair[1] else (pair[1], pair[0])
            if normalized not in seen:
                seen.add(normalized)
                pairs.append(normalized)
    return tuple(pairs)


class ThirdOrderBoundAccumulator:
    """Vectorized recurrence for selected third-derivative difference bounds."""

    def __init__(
        self,
        *,
        batch_size: int,
        input_dim: int,
        triples: tuple[tuple[int, int, int], ...],
        pair_to_column: dict[tuple[int, int], int],
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        if not triples:
            raise ValueError("triples must not be empty.")

        self.batch_size = batch_size
        self.input_dim = input_dim
        self.triples = triples
        self.num_triples = len(triples)
        self.dtype = dtype
        self.device = device

        self.a_idx = torch.tensor([a for a, _, _ in triples], dtype=torch.long, device=device)
        self.b_idx = torch.tensor([b for _, b, _ in triples], dtype=torch.long, device=device)
        self.c_idx = torch.tensor([c for _, _, c in triples], dtype=torch.long, device=device)
        self.ab_pair_idx = torch.tensor(
            [pair_to_column[(a, b)] for a, b, _ in triples],
            dtype=torch.long,
            device=device,
        )
        self.ac_pair_idx = torch.tensor(
            [pair_to_column[(a, c)] for a, _, c in triples],
            dtype=torch.long,
            device=device,
        )
        self.bc_pair_idx = torch.tensor(
            [pair_to_column[(b, c)] for _, b, c in triples],
            dtype=torch.long,
            device=device,
        )

        self.activation_third_derivative: torch.Tensor | None = None
        self.activation_third_derivative_error_bound: torch.Tensor | None = None
        self.preactivation_third_derivatives: list[torch.Tensor] = []
        self.preactivation_third_derivative_error_bounds: list[torch.Tensor] = []
        self.activation_third_derivatives: list[torch.Tensor] = []
        self.activation_third_derivative_error_bounds: list[torch.Tensor] = []

    def next_preactivation(self, W_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return base third derivatives and error bounds for selected ``z^k``."""
        W_k = W_k.to(dtype=self.dtype, device=self.device)
        out_width = W_k.shape[0]

        if self.activation_third_derivative is None:
            shape = (self.batch_size, out_width, self.num_triples)
            Z3_base = torch.zeros(shape, dtype=self.dtype, device=self.device)
            Z3_error = torch.zeros_like(Z3_base)
        else:
            Z3_base = torch.einsum(
                "ji,nit->njt",
                W_k,
                self.activation_third_derivative,
            )
            Z3_error = torch.einsum(
                "ji,nit->njt",
                W_k.abs(),
                self.activation_third_derivative_error_bound,
            )

        return Z3_base, validate_nonnegative_bound(
            "third-order preactivation error bound",
            Z3_error,
        )

    def append_layer(
        self,
        *,
        m1_k: torch.Tensor,
        m2_k: torch.Tensor,
        m3_k: torch.Tensor,
        Q1_k: torch.Tensor,
        Q2_k: torch.Tensor,
        Q3_k: torch.Tensor,
        Z1_base: torch.Tensor,
        Z1_error: torch.Tensor,
        Z2_base: torch.Tensor,
        Z2_error: torch.Tensor,
        Z3_base: torch.Tensor,
        Z3_error: torch.Tensor,
        store_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update propagated activation third-derivative data for one layer."""
        m1_k = m1_k.to(dtype=self.dtype, device=self.device)
        m2_k = m2_k.to(dtype=self.dtype, device=self.device)
        m3_k = m3_k.to(dtype=self.dtype, device=self.device)
        Q1_k = validate_nonnegative_bound(
            "first activation derivative variation bound",
            Q1_k.to(dtype=self.dtype, device=self.device),
        )
        Q2_k = validate_nonnegative_bound(
            "second activation derivative variation bound",
            Q2_k.to(dtype=self.dtype, device=self.device),
        )
        Q3_k = validate_nonnegative_bound(
            "third activation derivative variation bound",
            Q3_k.to(dtype=self.dtype, device=self.device),
        )
        Z1_error = validate_nonnegative_bound(
            "first-order preactivation error bound",
            Z1_error,
        )
        Z2_error = validate_nonnegative_bound(
            "second-order preactivation Hessian error bound",
            Z2_error,
        )
        Z3_error = validate_nonnegative_bound(
            "third-order preactivation error bound",
            Z3_error,
        )

        pa = Z1_base.index_select(dim=-1, index=self.a_idx)
        pb = Z1_base.index_select(dim=-1, index=self.b_idx)
        pc = Z1_base.index_select(dim=-1, index=self.c_idx)
        Pa = Z1_error.index_select(dim=-1, index=self.a_idx)
        Pb = Z1_error.index_select(dim=-1, index=self.b_idx)
        Pc = Z1_error.index_select(dim=-1, index=self.c_idx)

        qab = Z2_base.index_select(dim=-1, index=self.ab_pair_idx)
        qac = Z2_base.index_select(dim=-1, index=self.ac_pair_idx)
        qbc = Z2_base.index_select(dim=-1, index=self.bc_pair_idx)
        Rab = Z2_error.index_select(dim=-1, index=self.ab_pair_idx)
        Rac = Z2_error.index_select(dim=-1, index=self.ac_pair_idx)
        Rbc = Z2_error.index_select(dim=-1, index=self.bc_pair_idx)

        u0 = Z3_base
        S = Z3_error

        m1 = m1_k.unsqueeze(-1)
        m2 = m2_k.unsqueeze(-1)
        m3 = m3_k.unsqueeze(-1)
        Q1 = Q1_k.unsqueeze(-1)
        Q2 = Q2_k.unsqueeze(-1)
        Q3 = Q3_k.unsqueeze(-1)

        A3_base = (
            m3 * pa * pb * pc
            + m2 * (qab * pc + qac * pb + qbc * pa)
            + m1 * u0
        )

        pabc_error = _product_error_3(pa, Pa, pb, Pb, pc, Pc)
        pabc_radius = (pa.abs() + Pa) * (pb.abs() + Pb) * (pc.abs() + Pc)
        term_m3 = m3.abs() * pabc_error + Q3 * pabc_radius

        qab_pc_error = _product_error_2(qab, Rab, pc, Pc)
        qac_pb_error = _product_error_2(qac, Rac, pb, Pb)
        qbc_pa_error = _product_error_2(qbc, Rbc, pa, Pa)
        qab_pc_radius = (qab.abs() + Rab) * (pc.abs() + Pc)
        qac_pb_radius = (qac.abs() + Rac) * (pb.abs() + Pb)
        qbc_pa_radius = (qbc.abs() + Rbc) * (pa.abs() + Pa)
        term_m2 = (
            m2.abs() * (qab_pc_error + qac_pb_error + qbc_pa_error)
            + Q2 * (qab_pc_radius + qac_pb_radius + qbc_pa_radius)
        )

        term_m1 = m1.abs() * S + Q1 * (u0.abs() + S)
        A3_error = validate_nonnegative_bound(
            "third-order activation derivative error bound",
            term_m3 + term_m2 + term_m1,
        )

        self.activation_third_derivative = A3_base
        self.activation_third_derivative_error_bound = A3_error

        if store_cache:
            self.preactivation_third_derivatives.append(Z3_base)
            self.preactivation_third_derivative_error_bounds.append(Z3_error)
            self.activation_third_derivatives.append(A3_base)
            self.activation_third_derivative_error_bounds.append(A3_error)

        return A3_base, A3_error

    def final_bounds(self, W_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``d3f(y)`` and bounds for ``|d3f(x)-d3f(y)|``."""
        if self.activation_third_derivative is None:
            raise RuntimeError("No hidden-layer third derivative data accumulated.")

        W_out = W_out.to(dtype=self.dtype, device=self.device)
        if W_out.ndim != 2 or W_out.shape[0] != 1:
            raise ValueError("The output layer must have a single row.")

        w_out = W_out[0]
        base_third_derivatives = torch.einsum(
            "i,nit->nt",
            w_out,
            self.activation_third_derivative,
        )
        third_derivative_bounds = torch.einsum(
            "i,nit->nt",
            w_out.abs(),
            self.activation_third_derivative_error_bound,
        )
        return base_third_derivatives, validate_nonnegative_bound(
            "third derivative output bounds",
            third_derivative_bounds,
        )


def compute_third_order_bounds(
    model,
    y: torch.Tensor,
    eps: float,
    n_taylor: int | None = None,
    *,
    multi_indices,
    spatial_dim: int | None = None,
    return_cache: bool = False,
) -> ThirdOrderBoundsResult:
    """Compute selected third-derivative difference bounds.

    With ``spatial_dim=d``, activation enclosures vary only in the first
    ``d`` inputs while all requested derivative components remain available
    at the fixed trace.
    """
    y = normalize_input_batch(model, y)
    triples = resolve_third_order_multi_indices(model, multi_indices)
    pairs = required_pairs_for_triples(triples)
    pair_to_column = {pair: index for index, pair in enumerate(pairs)}

    with torch.no_grad():
        first_accumulator = FirstOrderBoundAccumulator(
            batch_size=y.shape[0],
            input_dim=model.input_dim,
            dtype=y.dtype,
            device=y.device,
        )
        second_accumulator = SecondOrderBoundAccumulator(
            batch_size=y.shape[0],
            input_dim=model.input_dim,
            pairs=pairs,
            dtype=y.dtype,
            device=y.device,
        )
        third_accumulator = ThirdOrderBoundAccumulator(
            batch_size=y.shape[0],
            input_dim=model.input_dim,
            triples=triples,
            pair_to_column=pair_to_column,
            dtype=y.dtype,
            device=y.device,
        )

        Q_layers = []
        z_layers = []
        a = y
        for layer in model.hidden_layers:
            z_k = layer(a)
            m1_k = model.AB.activation_m_prime_stable(z_k, 1)
            m2_k = model.AB.activation_m_prime_stable(z_k, 2)
            m3_k = model.AB.activation_m_prime_stable(z_k, 3)

            (
                Z1_base,
                Z1_error,
                radius_components,
            ) = first_accumulator.next_preactivation(layer.weight)
            Z2_base, Z2_error = second_accumulator.next_preactivation(layer.weight)
            Z3_base, Z3_error = third_accumulator.next_preactivation(layer.weight)

            _, delta_k = preactivation_delta(
                radius_components,
                eps,
                spatial_dim=spatial_dim,
            )
            Q_k_all = activation_derivative_bounds(
                model,
                z_k,
                delta_k,
                orders=(1, 2, 3),
                n_taylor=n_taylor,
            )
            Q1_k, Q2_k, Q3_k = Q_k_all

            first_accumulator.append_layer(
                m1_k,
                Q1_k,
                Z1_base,
                Z1_error,
                store_cache=return_cache,
            )
            second_accumulator.append_layer(
                m1_k=m1_k,
                m2_k=m2_k,
                Q1_k=Q1_k,
                Q2_k=Q2_k,
                Z1_base=Z1_base,
                Z1_error=Z1_error,
                Z2_base=Z2_base,
                Z2_error=Z2_error,
                store_cache=return_cache,
            )
            third_accumulator.append_layer(
                m1_k=m1_k,
                m2_k=m2_k,
                m3_k=m3_k,
                Q1_k=Q1_k,
                Q2_k=Q2_k,
                Q3_k=Q3_k,
                Z1_base=Z1_base,
                Z1_error=Z1_error,
                Z2_base=Z2_base,
                Z2_error=Z2_error,
                Z3_base=Z3_base,
                Z3_error=Z3_error,
                store_cache=return_cache,
            )

            if return_cache:
                Q_layers.append(Q_k_all)
                z_layers.append(z_k)

            a = model.activation(z_k)

        base_third_derivatives, third_derivative_bounds = (
            third_accumulator.final_bounds(model.output_layer.weight)
        )

        return ThirdOrderBoundsResult(
            triples=triples,
            pairs=pairs,
            base_third_derivatives=base_third_derivatives,
            third_derivative_bounds=third_derivative_bounds,
            Q=torch.stack(Q_layers, dim=1) if return_cache else None,
            z=torch.stack(z_layers) if return_cache else None,
            preactivation_gradients=(
                torch.stack(first_accumulator.preactivation_gradients)
                if return_cache
                else None
            ),
            preactivation_error_bounds=(
                torch.stack(first_accumulator.preactivation_error_bounds)
                if return_cache
                else None
            ),
            preactivation_hessians=(
                torch.stack(second_accumulator.preactivation_hessians)
                if return_cache
                else None
            ),
            preactivation_hessian_error_bounds=(
                torch.stack(second_accumulator.preactivation_hessian_error_bounds)
                if return_cache
                else None
            ),
            activation_hessians=(
                torch.stack(second_accumulator.activation_hessians)
                if return_cache
                else None
            ),
            activation_hessian_error_bounds=(
                torch.stack(second_accumulator.activation_hessian_error_bounds)
                if return_cache
                else None
            ),
            preactivation_third_derivatives=(
                torch.stack(third_accumulator.preactivation_third_derivatives)
                if return_cache
                else None
            ),
            preactivation_third_derivative_error_bounds=(
                torch.stack(
                    third_accumulator.preactivation_third_derivative_error_bounds,
                )
                if return_cache
                else None
            ),
            activation_third_derivatives=(
                torch.stack(third_accumulator.activation_third_derivatives)
                if return_cache
                else None
            ),
            activation_third_derivative_error_bounds=(
                torch.stack(
                    third_accumulator.activation_third_derivative_error_bounds,
                )
                if return_cache
                else None
            ),
            eps=float(torch.as_tensor(eps, dtype=y.dtype, device=y.device).max().item()),
            n_taylor=n_taylor,
        )


def _product_error_2(
    x0: torch.Tensor,
    X: torch.Tensor,
    y0: torch.Tensor,
    Y: torch.Tensor,
) -> torch.Tensor:
    return X * y0.abs() + x0.abs() * Y + X * Y


def _product_error_3(
    x0: torch.Tensor,
    X: torch.Tensor,
    y0: torch.Tensor,
    Y: torch.Tensor,
    z0: torch.Tensor,
    Z: torch.Tensor,
) -> torch.Tensor:
    return (
        X * y0.abs() * z0.abs()
        + x0.abs() * Y * z0.abs()
        + x0.abs() * y0.abs() * Z
        + X * Y * z0.abs()
        + X * y0.abs() * Z
        + x0.abs() * Y * Z
        + X * Y * Z
    )


def _validate_coordinate(value: object, input_dim: int) -> int:
    if isinstance(value, bool):
        raise ValueError("Multi-index entries must be integers, not bools.")
    try:
        coordinate = as_index(value)
    except TypeError as exc:
        raise ValueError("Multi-index entries must be integers.") from exc
    if not 0 <= coordinate < input_dim:
        raise ValueError(
            f"Multi-index entry {coordinate} is outside [0, {input_dim})."
        )
    return coordinate
