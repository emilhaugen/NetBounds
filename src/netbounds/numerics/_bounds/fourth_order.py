"""Streaming utilities for selected fourth-order derivative difference bounds."""

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
from .third_order import ThirdOrderBoundAccumulator, required_pairs_for_triples


@dataclass
class UpToFourthOrderBoundsResult:
    """Batched fused result for first through fourth derivative bounds."""

    pairs: tuple[tuple[int, int], ...]
    triples: tuple[tuple[int, int, int], ...]
    quartets: tuple[tuple[int, int, int, int], ...]
    base_gradient: torch.Tensor
    derivative_bounds: torch.Tensor
    base_hessian: torch.Tensor
    second_derivative_bounds: torch.Tensor
    base_third_derivatives: torch.Tensor
    third_derivative_bounds: torch.Tensor
    base_fourth_derivatives: torch.Tensor
    fourth_derivative_bounds: torch.Tensor
    eps: float = 0.0
    n_taylor: int | None = None


@dataclass
class FourthOrderBoundsResult:
    """Batched result of the selected fourth-order bound computation."""

    quartets: tuple[tuple[int, int, int, int], ...]
    triples: tuple[tuple[int, int, int], ...]
    pairs: tuple[tuple[int, int], ...]
    base_fourth_derivatives: torch.Tensor
    fourth_derivative_bounds: torch.Tensor
    Q: torch.Tensor | None = None
    z: torch.Tensor | None = None
    preactivation_fourth_derivatives: torch.Tensor | None = None
    preactivation_fourth_derivative_error_bounds: torch.Tensor | None = None
    activation_fourth_derivatives: torch.Tensor | None = None
    activation_fourth_derivative_error_bounds: torch.Tensor | None = None
    eps: float = 0.0
    n_taylor: int | None = None


def make_symmetric_quads(input_dim: int) -> tuple[tuple[int, int, int, int], ...]:
    """Return all zero-based symmetric fourth-order quartets for ``input_dim``."""
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    return tuple(
        (a, b, c, d)
        for a in range(input_dim)
        for b in range(a, input_dim)
        for c in range(b, input_dim)
        for d in range(c, input_dim)
    )


def resolve_fourth_order_multi_indices(
    model: Any,
    multi_indices: str | Sequence[Sequence[int]],
) -> tuple[tuple[int, int, int, int], ...]:
    """Normalize and validate selected order-4 multi-indices."""
    input_dim = model.input_dim
    if isinstance(multi_indices, str):
        if multi_indices == "full_input":
            return make_symmetric_quads(input_dim)
        if multi_indices == "spatial_only":
            return make_symmetric_quads(model.d)
        raise ValueError(
            "multi_indices must be one of 'full_input', 'spatial_only', "
            "or a nonempty sequence of index quartets."
        )

    try:
        raw_quartets = tuple(multi_indices)
    except TypeError as exc:
        raise ValueError(
            "multi_indices must be a nonempty sequence of index quartets."
        ) from exc

    if not raw_quartets:
        raise ValueError("multi_indices must not be empty.")

    quartets: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for raw_quartet in raw_quartets:
        if isinstance(raw_quartet, str | bytes):
            raise ValueError("Each multi-index must be a quartet of integers.")
        try:
            quartet = tuple(raw_quartet)
        except TypeError as exc:
            raise ValueError(
                "Each multi-index must be a quartet of integers."
            ) from exc
        if len(quartet) != 4:
            raise ValueError("Each multi-index must contain exactly four entries.")
        normalized = tuple(
            sorted(_validate_coordinate(value, input_dim) for value in quartet)
        )
        if normalized not in seen:
            seen.add(normalized)
            quartets.append(normalized)
    return tuple(quartets)


def required_triples_for_quads(
    quartets: tuple[tuple[int, int, int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    """Return selected third-order triples required by ``quartets``."""
    triples: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for a, b, c, d in quartets:
        for triple in ((a, b, c), (a, b, d), (a, c, d), (b, c, d)):
            normalized = tuple(sorted(triple))
            if normalized not in seen:
                seen.add(normalized)
                triples.append(normalized)
    return tuple(triples)


class FourthOrderBoundAccumulator:
    """Vectorized recurrence for selected fourth-derivative difference bounds."""

    def __init__(
        self,
        *,
        batch_size: int,
        input_dim: int,
        quartets: tuple[tuple[int, int, int, int], ...],
        pair_to_column: dict[tuple[int, int], int],
        triple_to_column: dict[tuple[int, int, int], int],
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        if not quartets:
            raise ValueError("quartets must not be empty.")
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.quartets = quartets
        self.num_quartets = len(quartets)
        self.dtype = dtype
        self.device = device
        self.idx = [
            torch.tensor([quartet[pos] for quartet in quartets], dtype=torch.long, device=device)
            for pos in range(4)
        ]
        pair_keys = [
            ((0, 1), (2, 3)),
            ((0, 2), (1, 3)),
            ((0, 3), (1, 2)),
        ]
        self.pair_pair_idx = []
        for left, right in pair_keys:
            left_cols = []
            right_cols = []
            for quartet in quartets:
                lp = tuple(sorted((quartet[left[0]], quartet[left[1]])))
                rp = tuple(sorted((quartet[right[0]], quartet[right[1]])))
                left_cols.append(pair_to_column[lp])
                right_cols.append(pair_to_column[rp])
            self.pair_pair_idx.append(
                (
                    torch.tensor(left_cols, dtype=torch.long, device=device),
                    torch.tensor(right_cols, dtype=torch.long, device=device),
                )
            )
        self.pair_single_idx = []
        for pair_pos, singles in (
            ((0, 1), (2, 3)),
            ((0, 2), (1, 3)),
            ((0, 3), (1, 2)),
            ((1, 2), (0, 3)),
            ((1, 3), (0, 2)),
            ((2, 3), (0, 1)),
        ):
            pair_cols = []
            single_left = []
            single_right = []
            for quartet in quartets:
                pair = tuple(sorted((quartet[pair_pos[0]], quartet[pair_pos[1]])))
                pair_cols.append(pair_to_column[pair])
                single_left.append(quartet[singles[0]])
                single_right.append(quartet[singles[1]])
            self.pair_single_idx.append(
                (
                    torch.tensor(pair_cols, dtype=torch.long, device=device),
                    torch.tensor(single_left, dtype=torch.long, device=device),
                    torch.tensor(single_right, dtype=torch.long, device=device),
                )
            )
        self.triple_single_idx = []
        for triple_pos, single_pos in (
            ((0, 1, 2), 3),
            ((0, 1, 3), 2),
            ((0, 2, 3), 1),
            ((1, 2, 3), 0),
        ):
            triple_cols = []
            single_cols = []
            for quartet in quartets:
                triple = tuple(sorted(quartet[pos] for pos in triple_pos))
                triple_cols.append(triple_to_column[triple])
                single_cols.append(quartet[single_pos])
            self.triple_single_idx.append(
                (
                    torch.tensor(triple_cols, dtype=torch.long, device=device),
                    torch.tensor(single_cols, dtype=torch.long, device=device),
                )
            )

        self.activation_fourth_derivative: torch.Tensor | None = None
        self.activation_fourth_derivative_error_bound: torch.Tensor | None = None
        self.preactivation_fourth_derivatives: list[torch.Tensor] = []
        self.preactivation_fourth_derivative_error_bounds: list[torch.Tensor] = []
        self.activation_fourth_derivatives: list[torch.Tensor] = []
        self.activation_fourth_derivative_error_bounds: list[torch.Tensor] = []

    def next_preactivation(self, W_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        W_k = W_k.to(dtype=self.dtype, device=self.device)
        out_width = W_k.shape[0]
        if self.activation_fourth_derivative is None:
            shape = (self.batch_size, out_width, self.num_quartets)
            Z4_base = torch.zeros(shape, dtype=self.dtype, device=self.device)
            Z4_error = torch.zeros_like(Z4_base)
        else:
            Z4_base = torch.einsum("ji,niq->njq", W_k, self.activation_fourth_derivative)
            Z4_error = torch.einsum(
                "ji,niq->njq",
                W_k.abs(),
                self.activation_fourth_derivative_error_bound,
            )
        return Z4_base, validate_nonnegative_bound(
            "fourth-order preactivation error bound",
            Z4_error,
        )

    def append_layer(
        self,
        *,
        m1_k: torch.Tensor,
        m2_k: torch.Tensor,
        m3_k: torch.Tensor,
        m4_k: torch.Tensor,
        Q1_k: torch.Tensor,
        Q2_k: torch.Tensor,
        Q3_k: torch.Tensor,
        Q4_k: torch.Tensor,
        Z1_base: torch.Tensor,
        Z1_error: torch.Tensor,
        Z2_base: torch.Tensor,
        Z2_error: torch.Tensor,
        Z3_base: torch.Tensor,
        Z3_error: torch.Tensor,
        Z4_base: torch.Tensor,
        Z4_error: torch.Tensor,
        store_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m1 = m1_k.to(dtype=self.dtype, device=self.device).unsqueeze(-1)
        m2 = m2_k.to(dtype=self.dtype, device=self.device).unsqueeze(-1)
        m3 = m3_k.to(dtype=self.dtype, device=self.device).unsqueeze(-1)
        m4 = m4_k.to(dtype=self.dtype, device=self.device).unsqueeze(-1)
        Q1 = validate_nonnegative_bound("Q1", Q1_k.to(dtype=self.dtype, device=self.device)).unsqueeze(-1)
        Q2 = validate_nonnegative_bound("Q2", Q2_k.to(dtype=self.dtype, device=self.device)).unsqueeze(-1)
        Q3 = validate_nonnegative_bound("Q3", Q3_k.to(dtype=self.dtype, device=self.device)).unsqueeze(-1)
        Q4 = validate_nonnegative_bound("Q4", Q4_k.to(dtype=self.dtype, device=self.device)).unsqueeze(-1)
        Z1_error = validate_nonnegative_bound("first-order preactivation error bound", Z1_error)
        Z2_error = validate_nonnegative_bound("second-order preactivation error bound", Z2_error)
        Z3_error = validate_nonnegative_bound("third-order preactivation error bound", Z3_error)
        Z4_error = validate_nonnegative_bound("fourth-order preactivation error bound", Z4_error)

        singles = [(Z1_base.index_select(-1, idx), Z1_error.index_select(-1, idx)) for idx in self.idx]

        A4_base = m4 * _product_base(singles)
        p4_error = _product_error_many(singles)
        p4_radius = _product_radius(singles)
        A4_error = m4.abs() * p4_error + Q4 * p4_radius

        m3_base_sum = torch.zeros_like(Z4_base)
        m3_error_sum = torch.zeros_like(Z4_base)
        m3_radius_sum = torch.zeros_like(Z4_base)
        for pair_idx, single_i, single_j in self.pair_single_idx:
            factors = [
                (Z2_base.index_select(-1, pair_idx), Z2_error.index_select(-1, pair_idx)),
                (Z1_base.index_select(-1, single_i), Z1_error.index_select(-1, single_i)),
                (Z1_base.index_select(-1, single_j), Z1_error.index_select(-1, single_j)),
            ]
            m3_base_sum = m3_base_sum + _product_base(factors)
            m3_error_sum = m3_error_sum + _product_error_many(factors)
            m3_radius_sum = m3_radius_sum + _product_radius(factors)
        A4_base = A4_base + m3 * m3_base_sum
        A4_error = A4_error + m3.abs() * m3_error_sum + Q3 * m3_radius_sum

        m2_base_sum = torch.zeros_like(Z4_base)
        m2_error_sum = torch.zeros_like(Z4_base)
        m2_radius_sum = torch.zeros_like(Z4_base)
        for left_idx, right_idx in self.pair_pair_idx:
            factors = [
                (Z2_base.index_select(-1, left_idx), Z2_error.index_select(-1, left_idx)),
                (Z2_base.index_select(-1, right_idx), Z2_error.index_select(-1, right_idx)),
            ]
            m2_base_sum = m2_base_sum + _product_base(factors)
            m2_error_sum = m2_error_sum + _product_error_many(factors)
            m2_radius_sum = m2_radius_sum + _product_radius(factors)
        for triple_idx, single_idx in self.triple_single_idx:
            factors = [
                (Z3_base.index_select(-1, triple_idx), Z3_error.index_select(-1, triple_idx)),
                (Z1_base.index_select(-1, single_idx), Z1_error.index_select(-1, single_idx)),
            ]
            m2_base_sum = m2_base_sum + _product_base(factors)
            m2_error_sum = m2_error_sum + _product_error_many(factors)
            m2_radius_sum = m2_radius_sum + _product_radius(factors)
        A4_base = A4_base + m2 * m2_base_sum
        A4_error = A4_error + m2.abs() * m2_error_sum + Q2 * m2_radius_sum

        A4_base = A4_base + m1 * Z4_base
        A4_error = validate_nonnegative_bound(
            "fourth-order activation derivative error bound",
            A4_error + m1.abs() * Z4_error + Q1 * (Z4_base.abs() + Z4_error),
        )

        self.activation_fourth_derivative = A4_base
        self.activation_fourth_derivative_error_bound = A4_error
        if store_cache:
            self.preactivation_fourth_derivatives.append(Z4_base)
            self.preactivation_fourth_derivative_error_bounds.append(Z4_error)
            self.activation_fourth_derivatives.append(A4_base)
            self.activation_fourth_derivative_error_bounds.append(A4_error)
        return A4_base, A4_error

    def final_bounds(self, W_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.activation_fourth_derivative is None:
            raise RuntimeError("No hidden-layer fourth derivative data accumulated.")
        W_out = W_out.to(dtype=self.dtype, device=self.device)
        if W_out.ndim != 2 or W_out.shape[0] != 1:
            raise ValueError("The output layer must have a single row.")
        w_out = W_out[0]
        base_fourth_derivatives = torch.einsum(
            "i,niq->nq",
            w_out,
            self.activation_fourth_derivative,
        )
        fourth_derivative_bounds = torch.einsum(
            "i,niq->nq",
            w_out.abs(),
            self.activation_fourth_derivative_error_bound,
        )
        return base_fourth_derivatives, validate_nonnegative_bound(
            "fourth derivative output bounds",
            fourth_derivative_bounds,
        )


def compute_fourth_order_bounds(
    model,
    y: torch.Tensor,
    eps: float,
    n_taylor: int | None = None,
    *,
    multi_indices,
    spatial_dim: int | None = None,
    return_cache: bool = False,
) -> FourthOrderBoundsResult:
    """Compute selected fourth-derivative difference bounds.

    ``spatial_dim`` optionally fixes all trailing input coordinates during the
    activation enclosure while preserving requested derivative components.
    """
    y = normalize_input_batch(model, y)
    quartets = resolve_fourth_order_multi_indices(model, multi_indices)
    triples = required_triples_for_quads(quartets)
    pairs = required_pairs_for_triples(triples)
    pair_to_column = {pair: index for index, pair in enumerate(pairs)}
    triple_to_column = {triple: index for index, triple in enumerate(triples)}

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
        fourth_accumulator = FourthOrderBoundAccumulator(
            batch_size=y.shape[0],
            input_dim=model.input_dim,
            quartets=quartets,
            pair_to_column=pair_to_column,
            triple_to_column=triple_to_column,
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
            m4_k = model.AB.activation_m_prime_stable(z_k, 4)

            Z1_base, Z1_error, radius_components = first_accumulator.next_preactivation(layer.weight)
            Z2_base, Z2_error = second_accumulator.next_preactivation(layer.weight)
            Z3_base, Z3_error = third_accumulator.next_preactivation(layer.weight)
            Z4_base, Z4_error = fourth_accumulator.next_preactivation(layer.weight)

            _, delta_k = preactivation_delta(
                radius_components,
                eps,
                spatial_dim=spatial_dim,
            )
            Q1_k, Q2_k, Q3_k, Q4_k = activation_derivative_bounds(
                model,
                z_k,
                delta_k,
                orders=(1, 2, 3, 4),
                n_taylor=n_taylor,
            )

            first_accumulator.append_layer(m1_k, Q1_k, Z1_base, Z1_error, store_cache=return_cache)
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
            fourth_accumulator.append_layer(
                m1_k=m1_k,
                m2_k=m2_k,
                m3_k=m3_k,
                m4_k=m4_k,
                Q1_k=Q1_k,
                Q2_k=Q2_k,
                Q3_k=Q3_k,
                Q4_k=Q4_k,
                Z1_base=Z1_base,
                Z1_error=Z1_error,
                Z2_base=Z2_base,
                Z2_error=Z2_error,
                Z3_base=Z3_base,
                Z3_error=Z3_error,
                Z4_base=Z4_base,
                Z4_error=Z4_error,
                store_cache=return_cache,
            )
            if return_cache:
                Q_layers.append(torch.stack((Q1_k, Q2_k, Q3_k, Q4_k)))
                z_layers.append(z_k)
            a = model.activation(z_k)

        base_gradient, derivative_bounds = first_accumulator.final_bounds(
            model.output_layer.weight,
        )
        base_hessian, second_derivative_bounds = second_accumulator.final_bounds(
            model.output_layer.weight,
        )
        base_third_derivatives, third_derivative_bounds = third_accumulator.final_bounds(
            model.output_layer.weight,
        )
        base_fourth_derivatives, fourth_derivative_bounds = fourth_accumulator.final_bounds(
            model.output_layer.weight,
        )
        eps_value = float(torch.as_tensor(eps, dtype=y.dtype, device=y.device).max().item())
        if return_cache == "up_to_fourth":
            return UpToFourthOrderBoundsResult(
                pairs=pairs,
                triples=triples,
                quartets=quartets,
                base_gradient=base_gradient,
                derivative_bounds=derivative_bounds,
                base_hessian=base_hessian,
                second_derivative_bounds=second_derivative_bounds,
                base_third_derivatives=base_third_derivatives,
                third_derivative_bounds=third_derivative_bounds,
                base_fourth_derivatives=base_fourth_derivatives,
                fourth_derivative_bounds=fourth_derivative_bounds,
                eps=eps_value,
                n_taylor=n_taylor,
            )
        return FourthOrderBoundsResult(
            quartets=quartets,
            triples=triples,
            pairs=pairs,
            base_fourth_derivatives=base_fourth_derivatives,
            fourth_derivative_bounds=fourth_derivative_bounds,
            Q=torch.stack(Q_layers, dim=1) if return_cache else None,
            z=torch.stack(z_layers) if return_cache else None,
            preactivation_fourth_derivatives=(
                torch.stack(fourth_accumulator.preactivation_fourth_derivatives)
                if return_cache
                else None
            ),
            preactivation_fourth_derivative_error_bounds=(
                torch.stack(fourth_accumulator.preactivation_fourth_derivative_error_bounds)
                if return_cache
                else None
            ),
            activation_fourth_derivatives=(
                torch.stack(fourth_accumulator.activation_fourth_derivatives)
                if return_cache
                else None
            ),
            activation_fourth_derivative_error_bounds=(
                torch.stack(fourth_accumulator.activation_fourth_derivative_error_bounds)
                if return_cache
                else None
            ),
            eps=eps_value,
            n_taylor=n_taylor,
        )


def _product_base(factors: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    out = torch.ones_like(factors[0][0])
    for base, _ in factors:
        out = out * base
    return out


def _product_radius(factors: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    out = torch.ones_like(factors[0][0])
    for base, error in factors:
        out = out * (base.abs() + error)
    return out


def _product_error_many(factors: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    base_abs = torch.ones_like(factors[0][0])
    for base, _ in factors:
        base_abs = base_abs * base.abs()
    return validate_nonnegative_bound(
        "product error bound",
        _product_radius(factors) - base_abs,
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


def compute_up_to_fourth_order_bounds(
    model,
    y: torch.Tensor,
    eps: float,
    n_taylor: int | None = None,
    *,
    multi_indices,
    spatial_dim: int | None = None,
) -> UpToFourthOrderBoundsResult:
    """Compute first through fourth derivative bounds in one hidden-layer sweep."""
    result = compute_fourth_order_bounds(
        model,
        y,
        eps,
        n_taylor,
        multi_indices=multi_indices,
        spatial_dim=spatial_dim,
        return_cache="up_to_fourth",
    )
    if not isinstance(result, UpToFourthOrderBoundsResult):
        raise TypeError("expected fused up-to-fourth-order result")
    return result

@dataclass
class WaveOperatorFourthCombinationsResult:
    """Fourth-derivative combinations A_qr = F_ttqr - c^2 sum_i F_iiqr."""

    pairs: tuple[tuple[int, int], ...]
    quartets: tuple[tuple[int, int, int, int], ...]
    base_combinations: torch.Tensor
    combination_bounds: torch.Tensor
    source: FourthOrderBoundsResult | UpToFourthOrderBoundsResult


def wave_operator_fourth_quartets(input_dim: int, spatial_dim: int) -> tuple[tuple[int, int, int, int], ...]:
    """Quartets needed for all Wave combinations F_ttqr - c^2 sum_i F_iiqr."""
    time_idx = input_dim - 1
    pairs = tuple((q, r) for q in range(input_dim) for r in range(q, input_dim))
    seen: set[tuple[int, int, int, int]] = set()
    out: list[tuple[int, int, int, int]] = []
    for q, r in pairs:
        for quartet in [tuple(sorted((time_idx, time_idx, q, r)))] + [tuple(sorted((i, i, q, r))) for i in range(spatial_dim)]:
            if quartet not in seen:
                seen.add(quartet)
                out.append(quartet)
    return tuple(out)


def compute_wave_operator_fourth_combinations(
    model,
    y: torch.Tensor,
    eps: float,
    n_taylor: int | None = None,
    *,
    c2: float = 1.0,
) -> WaveOperatorFourthCombinationsResult:
    """Compute vectorized Wave operator fourth-derivative combinations.

    This is the first operator-specific implementation layer: it computes only
    the quartets needed for A_qr and assembles the ten combinations in one
    matrix multiply. The current radius remains the rigorous triangle radius;
    tighter correlated propagation would need a deeper recurrence change.
    """
    quartets = wave_operator_fourth_quartets(model.input_dim, model.d)
    fourth = compute_fourth_order_bounds(model, y, eps, n_taylor, multi_indices=quartets)
    q_to_col = {q: i for i, q in enumerate(fourth.quartets)}
    pairs = tuple((q, r) for q in range(model.input_dim) for r in range(q, model.input_dim))
    coeff = torch.zeros((len(pairs), len(fourth.quartets)), dtype=y.dtype, device=y.device)
    time_idx = model.input_dim - 1
    for row, (q, r) in enumerate(pairs):
        coeff[row, q_to_col[tuple(sorted((time_idx, time_idx, q, r)))]] += 1.0
        for i in range(model.d):
            coeff[row, q_to_col[tuple(sorted((i, i, q, r)))]] -= c2
    base = fourth.base_fourth_derivatives @ coeff.T
    bounds = fourth.fourth_derivative_bounds @ coeff.abs().T
    return WaveOperatorFourthCombinationsResult(
        pairs=pairs,
        quartets=fourth.quartets,
        base_combinations=base,
        combination_bounds=validate_nonnegative_bound("wave operator fourth combination bounds", bounds),
        source=fourth,
    )
