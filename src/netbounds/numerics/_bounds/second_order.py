"""Streaming utilities for selected second-order derivative difference bounds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from operator import index as as_index
from typing import Any

import torch

from .common import (
    activation_derivative_bounds,
    hidden_preactivations_and_m,
    normalize_input_batch,
    preactivation_delta,
    validate_nonnegative_bound,
)
from .first_order import FirstOrderBoundAccumulator


@dataclass
class SecondOrderBoundsResult:
    """Batched result of the selected second-order bound computation."""

    # Flattened symmetric pairs. For index h, pairs[h] == (i, j), i <= j.
    pairs: tuple[tuple[int, int], ...]

    # Scalar output shape: (N, H). Vector output shape: (N, out_dim, H).
    base_hessian: torch.Tensor
    second_derivative_bounds: torch.Tensor

    # Optional debug/cache outputs. Kept as None unless return_cache=True.
    # Q has shape (2, L, N, width), ordered by activation derivative m=1,2.
    Q: torch.Tensor | None = None
    z: torch.Tensor | None = None
    preactivation_gradients: torch.Tensor | None = None
    preactivation_error_bounds: torch.Tensor | None = None
    preactivation_hessians: torch.Tensor | None = None
    preactivation_hessian_error_bounds: torch.Tensor | None = None
    activation_hessians: torch.Tensor | None = None
    activation_hessian_error_bounds: torch.Tensor | None = None
    eps: float = 0.0
    n_taylor: int | None = None


def make_symmetric_pairs(input_dim: int) -> tuple[tuple[int, int], ...]:
    """Return all zero-based symmetric Hessian pairs for ``input_dim``."""
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    return tuple((i, j) for i in range(input_dim) for j in range(i, input_dim))


def resolve_second_order_multi_indices(
    model: Any,
    multi_indices: str | Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    """Normalize and validate selected order-2 multi-indices.

    Custom indices are zero-based pairs. Reversed pairs are mapped to their
    symmetric representative, and duplicates are removed in first-seen order.
    """
    input_dim = model.input_dim
    if isinstance(multi_indices, str):
        if multi_indices == "full_input":
            return make_symmetric_pairs(input_dim)
        if multi_indices == "spatial_only":
            return tuple((i, j) for i in range(model.d) for j in range(i, model.d))
        if multi_indices == "time_mixed":
            time_idx = input_dim - 1
            return tuple((i, time_idx) for i in range(model.d))
        raise ValueError(
            "multi_indices must be one of 'full_input', 'spatial_only', "
            "'time_mixed', or a nonempty sequence of index pairs."
        )

    try:
        raw_pairs = tuple(multi_indices)
    except TypeError as exc:
        raise ValueError(
            "multi_indices must be a nonempty sequence of index pairs."
        ) from exc

    if not raw_pairs:
        raise ValueError("multi_indices must not be empty.")

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for raw_pair in raw_pairs:
        if isinstance(raw_pair, str | bytes):
            raise ValueError("Each multi-index must be a pair of integers.")

        try:
            pair = tuple(raw_pair)
        except TypeError as exc:
            raise ValueError("Each multi-index must be a pair of integers.") from exc

        if len(pair) != 2:
            raise ValueError("Each multi-index must contain exactly two entries.")

        i = _validate_coordinate(pair[0], input_dim)
        j = _validate_coordinate(pair[1], input_dim)
        normalized = (i, j) if i <= j else (j, i)

        if normalized not in seen:
            seen.add(normalized)
            pairs.append(normalized)

    return tuple(pairs)


class SecondOrderBoundAccumulator:
    """Vectorized recurrence for selected Hessian-pair difference bounds."""

    def __init__(
        self,
        *,
        batch_size: int,
        input_dim: int,
        pairs: tuple[tuple[int, int], ...],
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        if not pairs:
            raise ValueError("pairs must not be empty.")

        self.batch_size = batch_size
        self.input_dim = input_dim
        self.pairs = pairs
        self.num_pairs = len(pairs)
        self.dtype = dtype
        self.device = device
        self.left_idx = torch.tensor(
            [i for i, _ in pairs],
            dtype=torch.long,
            device=device,
        )
        self.right_idx = torch.tensor(
            [j for _, j in pairs],
            dtype=torch.long,
            device=device,
        )

        self.activation_hessian: torch.Tensor | None = None
        self.activation_hessian_error_bound: torch.Tensor | None = None
        self.preactivation_hessians: list[torch.Tensor] = []
        self.preactivation_hessian_error_bounds: list[torch.Tensor] = []
        self.activation_hessians: list[torch.Tensor] = []
        self.activation_hessian_error_bounds: list[torch.Tensor] = []

    def next_preactivation(self, W_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return base Hessians and error bounds for selected pairs of ``z^k``."""
        W_k = W_k.to(dtype=self.dtype, device=self.device)
        out_width = W_k.shape[0]

        if self.activation_hessian is None:
            shape = (self.batch_size, out_width, self.num_pairs)
            Z2_base = torch.zeros(shape, dtype=self.dtype, device=self.device)
            Z2_error = torch.zeros_like(Z2_base)
        else:
            Z2_base = torch.einsum("ji,nih->njh", W_k, self.activation_hessian)
            Z2_error = torch.einsum(
                "ji,nih->njh",
                W_k.abs(),
                self.activation_hessian_error_bound,
            )

        return Z2_base, Z2_error

    def append_layer(
        self,
        *,
        m1_k: torch.Tensor,
        m2_k: torch.Tensor,
        Q1_k: torch.Tensor,
        Q2_k: torch.Tensor,
        Z1_base: torch.Tensor,
        Z1_error: torch.Tensor,
        Z2_base: torch.Tensor,
        Z2_error: torch.Tensor,
        store_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update propagated activation-Hessian data for one hidden layer."""
        m1_k = m1_k.to(dtype=self.dtype, device=self.device)
        m2_k = m2_k.to(dtype=self.dtype, device=self.device)
        Q1_k = Q1_k.to(dtype=self.dtype, device=self.device)
        Q2_k = Q2_k.to(dtype=self.dtype, device=self.device)

        pl = Z1_base.index_select(dim=-1, index=self.left_idx)
        pr = Z1_base.index_select(dim=-1, index=self.right_idx)
        Pl = Z1_error.index_select(dim=-1, index=self.left_idx)
        Pr = Z1_error.index_select(dim=-1, index=self.right_idx)
        q0 = Z2_base
        R = Z2_error

        m1 = m1_k.unsqueeze(-1)
        m2 = m2_k.unsqueeze(-1)
        Q1 = Q1_k.unsqueeze(-1)
        Q2 = Q2_k.unsqueeze(-1)

        A2_base = m2 * pl * pr + m1 * q0

        term_m2 = (
            m2.abs() * (pl.abs() * Pr + pr.abs() * Pl + Pl * Pr)
            + Q2 * (pl.abs() + Pl) * (pr.abs() + Pr)
        )
        term_m1 = m1.abs() * R + Q1 * (q0.abs() + R)
        A2_error = validate_nonnegative_bound(
            "second-order activation Hessian error bound",
            term_m2 + term_m1,
        )

        self.activation_hessian = A2_base
        self.activation_hessian_error_bound = A2_error

        if store_cache:
            self.preactivation_hessians.append(Z2_base)
            self.preactivation_hessian_error_bounds.append(Z2_error)
            self.activation_hessians.append(A2_base)
            self.activation_hessian_error_bounds.append(A2_error)

        return A2_base, A2_error

    def final_bounds(self, W_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``d2f(y)`` and bounds for ``|d2f(x)-d2f(y)|``."""
        if self.activation_hessian is None:
            raise RuntimeError("No hidden-layer Hessian data have been accumulated.")

        W_out = W_out.to(dtype=self.dtype, device=self.device)
        if W_out.ndim != 2 or W_out.shape[0] != 1:
            raise ValueError("The output layer must have a single row.")

        w_out = W_out[0]
        base_hessian = torch.einsum("i,nih->nh", w_out, self.activation_hessian)
        hessian_bounds = torch.einsum(
            "i,nih->nh",
            w_out.abs(),
            self.activation_hessian_error_bound,
        )
        return base_hessian, hessian_bounds


def compute_second_order_bounds(
    model,
    y: torch.Tensor,
    eps: float,
    n_taylor: int | None = None,
    *,
    multi_indices,
    spatial_dim: int | None = None,
    return_cache: bool = False,
) -> SecondOrderBoundsResult:
    """Compute selected second-derivative difference bounds.

    With ``spatial_dim=d``, only the first ``d`` input coordinates vary in
    activation enclosures.  Derivatives involving the remaining coordinates
    are still evaluated at the fixed input trace.
    """
    y = normalize_input_batch(model, y)
    pairs = resolve_second_order_multi_indices(model, multi_indices)

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

        z_values, m_by_order = hidden_preactivations_and_m(
            model,
            y,
            orders=(1, 2),
        )
        m1_values = m_by_order[1]
        m2_values = m_by_order[2]

        Q_layers = []
        z_layers = []
        for layer, z_k, m1_k, m2_k in zip(
            model.hidden_layers,
            z_values,
            m1_values,
            m2_values,
        ):
            (
                Z1_base,
                Z1_error,
                radius_components,
            ) = first_accumulator.next_preactivation(layer.weight)
            Z2_base, Z2_error = second_accumulator.next_preactivation(
                layer.weight,
            )

            _, delta_k = preactivation_delta(
                radius_components,
                eps,
                spatial_dim=spatial_dim,
            )
            Q_k_all = activation_derivative_bounds(
                model,
                z_k,
                delta_k,
                orders=(1, 2),
                n_taylor=n_taylor,
            )
            Q1_k, Q2_k = Q_k_all

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

            if return_cache:
                Q_layers.append(torch.stack((Q1_k, Q2_k)))
                z_layers.append(z_k)

        base_hessian, second_derivative_bounds = second_accumulator.final_bounds(
            model.output_layer.weight,
        )

        return SecondOrderBoundsResult(
            pairs=pairs,
            base_hessian=base_hessian,
            second_derivative_bounds=second_derivative_bounds,
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
                torch.stack(
                    second_accumulator.preactivation_hessian_error_bounds,
                )
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
            eps=float(torch.as_tensor(eps, dtype=y.dtype, device=y.device).max().item()),
            n_taylor=n_taylor,
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
