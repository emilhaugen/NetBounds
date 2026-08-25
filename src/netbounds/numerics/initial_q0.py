"""Reached 1D kernel extracted from producer revision 2add56070e88dc16def9accbed953fa854f22d39."""

from __future__ import annotations

import math

from collections.abc import Callable

from dataclasses import dataclass

from typing import Protocol

import torch

from ._bounds import (
    compute_second_order_bounds,
    compute_value_and_first_derivative_bounds_over_spatial_box,
)

from .model import TanhNetwork as RigPINN_tanh

from ._initial_support.boundary import (
    product_boundary_derivative_modulus,
    product_boundary_derivative_value,
    require_spatial_product_boundary,
)

from ._initial_support.modulus_data import (
    ModulusFunctionCertificate,
    ScalarLipschitzFunctionCertificate,
)

from ._initial_support.moment import linear_gradient_remainder_moment_integral

from ._initial_support.pde_helpers import (
    _boundary_derivative_center,
    _boundary_derivative_sup,
    _multi_choose,
    _sub_counts,
)

from ._initial_support.grids import (
    UniformBoxGrid,
    UniformUnitBoxGridSpec,
    iter_uniform_grid_batches,
)

from ._initial_support.quadrature import (
    L2QuadratureAccumulator,
    ResidualL2BoundResult,
    as_cell_vector,
    require_nonnegative,
)

class ScalarTaylorFunctionCertificate(ModulusFunctionCertificate, Protocol):
    """Scalar field values, exact gradient centers, and Hessian bounds."""

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Return exact gradients with shape [M, d]."""
        ...

    def hessian_l1_bound(self, centers: torch.Tensor, radius: float) -> torch.Tensor:
        """Return sum_ij sup |partial_ij field| over each cell, shape [M]."""
        ...

    def hessian_abs_bound(self, centers: torch.Tensor, radius: float) -> torch.Tensor:
        """Return entrywise Hessian absolute bounds with shape [M,d,d]."""
        ...

@dataclass
class TaylorInitialDatum:
    """Scalar datum for Taylor-cell quadrature using analytic derivatives."""

    value_fn: Callable[[torch.Tensor], torch.Tensor]
    gradient_fn: Callable[[torch.Tensor], torch.Tensor]
    hessian_l1_bound_value: float | torch.Tensor | Callable[[torch.Tensor, float], torch.Tensor]
    hessian_abs_bound_value: (
        float | torch.Tensor | Callable[[torch.Tensor, float], torch.Tensor] | None
    ) = None

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_fn(x)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        values = self.gradient_fn(x)
        if values.ndim != 2 or values.shape[0] != x.shape[0] or values.shape[1] != x.shape[1]:
            raise ValueError("scalar datum gradient must have shape [M, d].")
        return values

    def hessian_l1_bound(self, centers: torch.Tensor, radius: float) -> torch.Tensor:
        values = self.value(centers)
        if callable(self.hessian_l1_bound_value):
            bound = self.hessian_l1_bound_value(centers, radius)
        else:
            bound = self.hessian_l1_bound_value
        bound = torch.as_tensor(bound, dtype=centers.dtype, device=centers.device)
        if bound.ndim == 0:
            bound = bound.expand(centers.shape[0])
        bound = as_cell_vector(
            "hessian_l1_bound",
            bound,
            length=centers.shape[0],
            dtype=centers.dtype,
            device=centers.device,
        )
        return require_nonnegative("hessian_l1_bound", bound)

    def hessian_abs_bound(self, centers: torch.Tensor, radius: float) -> torch.Tensor:
        values = self.value(centers)
        if self.hessian_abs_bound_value is None:
            l1 = self.hessian_l1_bound(centers, radius)
            return l1[:, None, None].expand(-1, centers.shape[1], centers.shape[1])
        source = self.hessian_abs_bound_value
        bound = source(centers, radius) if callable(source) else source
        bound = torch.as_tensor(bound, dtype=centers.dtype, device=centers.device)
        dimension = centers.shape[1]
        if bound.ndim == 0:
            bound = bound.expand(centers.shape[0], dimension, dimension)
        elif bound.shape == (dimension, dimension):
            bound = bound.unsqueeze(0).expand(centers.shape[0], -1, -1)
        if bound.shape != (centers.shape[0], dimension, dimension):
            raise ValueError("hessian_abs_bound_value must broadcast to shape [M,d,d]")
        return require_nonnegative("hessian_abs_bound", bound)

    def variation_bound(self, centers: torch.Tensor, radius: float) -> torch.Tensor:
        grad_l1 = self.gradient(centers).abs().sum(dim=-1)
        return require_nonnegative(
            "taylor_initial_variation_bound",
            float(radius) * grad_l1 + 0.5 * float(radius) ** 2 * self.hessian_l1_bound(centers, radius),
        )

@dataclass
class InitialResidualL2BoundResult:
    l2_bound: torch.Tensor
    l2_squared_bound: torch.Tensor
    midpoint_l2: torch.Tensor | None
    midpoint_l2_squared: torch.Tensor | None
    bound_to_midpoint_ratio: torch.Tensor | None
    cell_l2_squared_bounds: torch.Tensor | None
    phi_center_abs_bounds: torch.Tensor | None
    eta_g: torch.Tensor | None
    eta_F: torch.Tensor | None
    eta_B: torch.Tensor | None
    eta_v: torch.Tensor | None
    eta_phi: torch.Tensor | None
    F_center: torch.Tensor | None
    B_center: torch.Tensor | None
    g_center: torch.Tensor | None
    centers: torch.Tensor | None
    spatial_eps: float
    cell_volume: torch.Tensor
    certified: bool = True
    notes: tuple[str, ...] = ()
    generic_result: ResidualL2BoundResult | None = None

def _spacetime_initial_centers(
    centers: torch.Tensor,
    model: RigPINN_tanh,
) -> torch.Tensor:
    y = torch.zeros(
        centers.shape[0],
        model.input_dim,
        dtype=centers.dtype,
        device=centers.device,
    )
    y[:, : model.d] = centers
    y[:, model.input_dim - 1] = 0.0
    return y

def _index_column(indices: tuple[tuple[int, ...], ...], key: tuple[int, ...]) -> int:
    normalized = tuple(sorted(key))
    try:
        return indices.index(normalized)
    except ValueError as exc:
        raise KeyError(f"missing derivative multi-index {normalized}") from exc

def _spatial_counts_to_key(counts: tuple[int, ...]) -> tuple[int, ...]:
    key: list[int] = []
    for axis, count in enumerate(counts):
        key.extend([axis] * count)
    return tuple(key)

def _network_derivative_center(*, counts: tuple[int, ...], first, second) -> torch.Tensor:
    order = sum(counts)
    if order == 0:
        return first.base_value.squeeze(-1)
    key = _spatial_counts_to_key(counts)
    if order == 1:
        return first.base_gradient[:, key[0]]
    if order == 2:
        return second.base_hessian[:, _index_column(second.pairs, key)]
    raise ValueError("only derivatives up to order 2 are supported")

def _network_derivative_sup(
    *,
    counts: tuple[int, ...],
    spatial_eps: float,
    first,
    second,
) -> torch.Tensor:
    order = sum(counts)
    if order == 0:
        spatial_grad_sup = require_nonnegative(
            "spatial_grad_sup",
            first.base_gradient[:, : len(counts)].abs() + first.derivative_bounds[:, : len(counts)],
        )
        return first.base_value.squeeze(-1).abs() + float(spatial_eps) * spatial_grad_sup.sum(dim=-1)
    key = _spatial_counts_to_key(counts)
    if order == 1:
        axis = key[0]
        return first.base_gradient[:, axis].abs() + first.derivative_bounds[:, axis]
    if order == 2:
        col = _index_column(second.pairs, key)
        return second.base_hessian[:, col].abs() + second.second_derivative_bounds[:, col]
    raise ValueError("only derivatives up to order 2 are supported")

def _bf_derivative_center(
    *,
    spatial: torch.Tensor,
    counts: tuple[int, ...],
    first,
    second,
    input_dim: int,
) -> torch.Tensor:
    total = torch.zeros(spatial.shape[0], dtype=spatial.dtype, device=spatial.device)
    full_counts = counts + (0,) * (input_dim - len(counts))
    for beta_spatial in _sub_counts(counts):
        beta_full = beta_spatial + (0,) * (input_dim - len(beta_spatial))
        gamma_spatial = tuple(a - b for a, b in zip(counts, beta_spatial))
        total = total + float(_multi_choose(full_counts, beta_full)) * _boundary_derivative_center(
            spatial,
            beta_full,
        ) * _network_derivative_center(counts=gamma_spatial, first=first, second=second)
    return total

def _bf_derivative_sup(
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    counts: tuple[int, ...],
    spatial_eps: float,
    first,
    second,
    input_dim: int,
) -> torch.Tensor:
    total = torch.zeros(lower.shape[0], dtype=lower.dtype, device=lower.device)
    full_counts = counts + (0,) * (input_dim - len(counts))
    for beta_spatial in _sub_counts(counts):
        beta_full = beta_spatial + (0,) * (input_dim - len(beta_spatial))
        gamma_spatial = tuple(a - b for a, b in zip(counts, beta_spatial))
        total = total + float(_multi_choose(full_counts, beta_full)) * _boundary_derivative_sup(
            lower,
            upper,
            beta_full,
        ) * _network_derivative_sup(
            counts=gamma_spatial,
            spatial_eps=spatial_eps,
            first=first,
            second=second,
        )
    return require_nonnegative("bf_derivative_sup", total)


def _bf_derivative_paper_sup(
    *,
    spatial: torch.Tensor,
    counts: tuple[int, ...],
    spatial_eps: float,
    first,
    second,
) -> torch.Tensor:
    total = torch.zeros(
        spatial.shape[0], dtype=spatial.dtype, device=spatial.device
    )
    for beta in _sub_counts(counts):
        gamma = tuple(a - b for a, b in zip(counts, beta))
        boundary_sup = (
            product_boundary_derivative_value(spatial, beta).abs()
            + product_boundary_derivative_modulus(spatial, spatial_eps, beta)
        )
        total = total + float(_multi_choose(counts, beta)) * boundary_sup * (
            _network_derivative_sup(
                counts=gamma,
                spatial_eps=spatial_eps,
                first=first,
                second=second,
            )
        )
    return require_nonnegative("paper_bf_derivative_sup", total)


def _initial_displacement_moment_cell_integrals(
    *,
    model: RigPINN_tanh,
    centers: torch.Tensor,
    spatial_eps: float,
    initial_g: ScalarLipschitzFunctionCertificate,
    n_taylor: int | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Literal appendix Q0 cell integral for ``g-BF``."""

    M, d = centers.shape
    dtype, device = centers.dtype, centers.device
    spacetime_centers = _spacetime_initial_centers(centers, model)
    first = compute_value_and_first_derivative_bounds_over_spatial_box(
        model, spacetime_centers, spatial_eps, n_taylor
    )
    lower = centers - float(spatial_eps)
    upper = centers + float(spatial_eps)
    zero_counts = (0,) * d
    approx_center = _bf_derivative_center(
        spatial=centers,
        counts=zero_counts,
        first=first,
        second=None,
        input_dim=model.input_dim,
    )
    approx_first_abs = torch.empty(M, d, dtype=dtype, device=device)
    for q in range(d):
        counts_q = tuple(int(axis == q) for axis in range(d))
        approx_first_abs[:, q] = _bf_derivative_paper_sup(
            spatial=centers,
            counts=counts_q,
            spatial_eps=spatial_eps,
            first=first,
            second=None,
        )

    g_center = as_cell_vector(
        "g_center", initial_g.value(centers), length=M, dtype=dtype, device=device
    )
    data_lipschitz = as_cell_vector(
        "initial_g_lipschitz",
        initial_g.lipschitz_bound(centers),
        length=M,
        dtype=dtype,
        device=device,
    )
    data_lipschitz = require_nonnegative("initial_g_lipschitz", data_lipschitz)
    first_derivative_abs_bounds = require_nonnegative(
        "initial_displacement_first_derivative_abs_bounds",
        data_lipschitz[:, None] + approx_first_abs,
    )
    residual_center = g_center - approx_center
    cell_volume = torch.as_tensor(
        (2.0 * float(spatial_eps)) ** d, dtype=dtype, device=device
    )
    moments = linear_gradient_remainder_moment_integral(
        residual_center=residual_center,
        first_derivative_abs_bounds=first_derivative_abs_bounds,
        spatial_eps=spatial_eps,
        cell_volume=cell_volume,
    )
    diagnostics = {
        "phi_center_abs_bounds": residual_center.abs(),
        "eta_g": data_lipschitz,
        "eta_F": approx_first_abs,
        "eta_v": approx_first_abs,
        "eta_phi": first_derivative_abs_bounds.sum(dim=-1),
        "F_center": first.base_value.squeeze(-1),
        "B_center": _boundary_derivative_center(centers, (0,) * model.input_dim),
        "g_center": g_center,
        "approx_center": approx_center,
        "first_derivative_abs_bounds": first_derivative_abs_bounds,
        "moment_constant_l2_squared": moments["int_p2"],
        "moment_cross_correction": moments["cross_correction"],
        "moment_remainder_l2_squared": moments["remainder_l2_squared"],
    }
    return moments["selected"], residual_center.abs(), diagnostics

def _network_time_derivative_center(
    *,
    counts: tuple[int, ...],
    time_idx: int,
    first,
    second,
    third,
) -> torch.Tensor:
    """Return the center value of a spatial derivative of ``F_t``."""
    order = sum(counts)
    key = tuple(sorted(_spatial_counts_to_key(counts) + (time_idx,)))
    if order == 0:
        return first.base_gradient[:, time_idx]
    if order == 1:
        return second.base_hessian[:, _index_column(second.pairs, key)]
    if order == 2:
        return third.base_third_derivatives[:, _index_column(third.triples, key)]
    raise ValueError("only spatial derivatives of F_t up to order 2 are supported")

def _network_time_derivative_sup(
    *,
    counts: tuple[int, ...],
    time_idx: int,
    first,
    second,
    third,
) -> torch.Tensor:
    """Return a certified cell supremum for a spatial derivative of ``F_t``."""
    order = sum(counts)
    key = tuple(sorted(_spatial_counts_to_key(counts) + (time_idx,)))
    if order == 0:
        return first.base_gradient[:, time_idx].abs() + first.derivative_bounds[:, time_idx]
    if order == 1:
        col = _index_column(second.pairs, key)
        return second.base_hessian[:, col].abs() + second.second_derivative_bounds[:, col]
    if order == 2:
        col = _index_column(third.triples, key)
        return third.base_third_derivatives[:, col].abs() + third.third_derivative_bounds[:, col]
    raise ValueError("only spatial derivatives of F_t up to order 2 are supported")

def _bft_derivative_center(
    *,
    spatial: torch.Tensor,
    counts: tuple[int, ...],
    time_idx: int,
    first,
    second,
    third,
    input_dim: int,
) -> torch.Tensor:
    """Return a spatial derivative center of ``B(x) F_t(x, 0)``."""
    total = torch.zeros(spatial.shape[0], dtype=spatial.dtype, device=spatial.device)
    full_counts = counts + (0,) * (input_dim - len(counts))
    for beta_spatial in _sub_counts(counts):
        beta_full = beta_spatial + (0,) * (input_dim - len(beta_spatial))
        gamma_spatial = tuple(a - b for a, b in zip(counts, beta_spatial))
        total = total + float(_multi_choose(full_counts, beta_full)) * _boundary_derivative_center(
            spatial,
            beta_full,
        ) * _network_time_derivative_center(
            counts=gamma_spatial,
            time_idx=time_idx,
            first=first,
            second=second,
            third=third,
        )
    return total

def _bft_derivative_sup(
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    counts: tuple[int, ...],
    time_idx: int,
    first,
    second,
    third,
    input_dim: int,
) -> torch.Tensor:
    """Return a certified cell supremum for a spatial derivative of ``B F_t``."""
    total = torch.zeros(lower.shape[0], dtype=lower.dtype, device=lower.device)
    full_counts = counts + (0,) * (input_dim - len(counts))
    for beta_spatial in _sub_counts(counts):
        beta_full = beta_spatial + (0,) * (input_dim - len(beta_spatial))
        gamma_spatial = tuple(a - b for a, b in zip(counts, beta_spatial))
        total = total + float(_multi_choose(full_counts, beta_full)) * _boundary_derivative_sup(
            lower,
            upper,
            beta_full,
        ) * _network_time_derivative_sup(
            counts=gamma_spatial,
            time_idx=time_idx,
            first=first,
            second=second,
            third=third,
        )
    return require_nonnegative("bft_derivative_sup", total)


def _bft_derivative_paper_sup(
    *,
    spatial: torch.Tensor,
    counts: tuple[int, ...],
    spatial_eps: float,
    time_idx: int,
    first,
    second,
    third,
) -> torch.Tensor:
    total = torch.zeros(
        spatial.shape[0], dtype=spatial.dtype, device=spatial.device
    )
    for beta in _sub_counts(counts):
        gamma = tuple(a - b for a, b in zip(counts, beta))
        boundary_sup = (
            product_boundary_derivative_value(spatial, beta).abs()
            + product_boundary_derivative_modulus(spatial, spatial_eps, beta)
        )
        total = total + float(_multi_choose(counts, beta)) * boundary_sup * (
            _network_time_derivative_sup(
                counts=gamma,
                time_idx=time_idx,
                first=first,
                second=second,
                third=third,
            )
        )
    return require_nonnegative("paper_bft_derivative_sup", total)


def _initial_velocity_moment_cell_integrals(
    *,
    model: RigPINN_tanh,
    centers: torch.Tensor,
    spatial_eps: float,
    initial_velocity: ScalarLipschitzFunctionCertificate,
    n_taylor: int | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Literal appendix Q0 cell integral for ``h-BF_t``."""

    M, d = centers.shape
    dtype, device = centers.dtype, centers.device
    time_idx = model.input_dim - 1
    spacetime_centers = _spacetime_initial_centers(centers, model)
    first = compute_value_and_first_derivative_bounds_over_spatial_box(
        model, spacetime_centers, spatial_eps, n_taylor
    )
    mixed_pairs = tuple((q, time_idx) for q in range(d))
    second = compute_second_order_bounds(
        model,
        spacetime_centers,
        spatial_eps,
        n_taylor,
        multi_indices=mixed_pairs,
        spatial_dim=model.d,
    )
    zero_counts = (0,) * d
    approx_center = _bft_derivative_center(
        spatial=centers,
        counts=zero_counts,
        time_idx=time_idx,
        first=first,
        second=second,
        third=None,
        input_dim=model.input_dim,
    )
    lower = centers - float(spatial_eps)
    upper = centers + float(spatial_eps)
    approx_first_abs = torch.empty(M, d, dtype=dtype, device=device)
    for q in range(d):
        counts_q = tuple(int(axis == q) for axis in range(d))
        approx_first_abs[:, q] = _bft_derivative_paper_sup(
            spatial=centers,
            counts=counts_q,
            spatial_eps=spatial_eps,
            time_idx=time_idx,
            first=first,
            second=second,
            third=None,
        )

    velocity_center = as_cell_vector(
        "initial_velocity_center",
        initial_velocity.value(centers),
        length=M,
        dtype=dtype,
        device=device,
    )
    data_lipschitz = as_cell_vector(
        "initial_velocity_lipschitz",
        initial_velocity.lipschitz_bound(centers),
        length=M,
        dtype=dtype,
        device=device,
    )
    data_lipschitz = require_nonnegative("initial_velocity_lipschitz", data_lipschitz)
    first_derivative_abs_bounds = require_nonnegative(
        "initial_velocity_first_derivative_abs_bounds",
        data_lipschitz[:, None] + approx_first_abs,
    )
    residual_center = velocity_center - approx_center
    cell_volume = torch.as_tensor(
        (2.0 * float(spatial_eps)) ** d, dtype=dtype, device=device
    )
    moments = linear_gradient_remainder_moment_integral(
        residual_center=residual_center,
        first_derivative_abs_bounds=first_derivative_abs_bounds,
        spatial_eps=spatial_eps,
        cell_volume=cell_volume,
    )
    diagnostics = {
        "phi_center_abs_bounds": residual_center.abs(),
        "eta_g": data_lipschitz,
        "eta_F": approx_first_abs,
        "eta_v": approx_first_abs,
        "eta_phi": first_derivative_abs_bounds.sum(dim=-1),
        "F_center": first.base_gradient[:, time_idx],
        "B_center": _boundary_derivative_center(centers, (0,) * model.input_dim),
        "g_center": velocity_center,
        "approx_center": approx_center,
        "first_derivative_abs_bounds": first_derivative_abs_bounds,
        "moment_constant_l2_squared": moments["int_p2"],
        "moment_cross_correction": moments["cross_correction"],
        "moment_remainder_l2_squared": moments["remainder_l2_squared"],
    }
    return moments["selected"], residual_center.abs(), diagnostics

def _run_initial_moment_residual_l2_uniform(
    *,
    model: RigPINN_tanh,
    grid: UniformBoxGrid | UniformUnitBoxGridSpec,
    datum: ScalarLipschitzFunctionCertificate,
    cell_integral_fn,
    name: str,
    n_taylor: int | None = None,
    safety_factor: float = 1.0,
    rounding_slack: float = 0.0,
    batch_size: int | None = None,
    store_diagnostics: bool = True,
    progress_callback=None,
) -> InitialResidualL2BoundResult:
    """Stream literal centered-moment Q0 squared cell integrals."""

    if not math.isfinite(float(safety_factor)) or safety_factor < 1.0:
        raise ValueError("safety_factor must be finite and at least 1.")
    if not math.isfinite(float(rounding_slack)) or rounding_slack < 0.0:
        raise ValueError("rounding_slack must be finite and nonnegative.")
    require_spatial_product_boundary(model.boundary_function, model.d)
    grid.validate()
    if grid.d != model.d:
        raise ValueError("grid.d must equal model.d.")
    dtype = model.input_layer.weight.dtype
    device = model.input_layer.weight.device
    spatial_eps = float(grid.spatial_eps)
    cell_volume = torch.as_tensor(
        (2.0 * spatial_eps) ** model.d, dtype=dtype, device=device
    )
    accumulator = L2QuadratureAccumulator(
        cell_volume=cell_volume,
        dtype=dtype,
        device=device,
        store_diagnostics=store_diagnostics,
    )
    with torch.no_grad():
        processed_boxes = 0
        for batch_grid in iter_uniform_grid_batches(grid, batch_size):
            centers = batch_grid.centers.to(dtype=dtype, device=device)
            cell_l2_squared, midpoint_residual, diagnostics = cell_integral_fn(
                model=model,
                centers=centers,
                spatial_eps=spatial_eps,
                datum=datum,
                n_taylor=n_taylor,
            )
            cell_residual_bound = (
                float(safety_factor)
                * torch.sqrt((cell_l2_squared / cell_volume).clamp_min(0.0))
                + float(rounding_slack)
            )
            accumulator.add_cell_bounds(
                cell_residual_bound,
                midpoint_residual=midpoint_residual,
                centers=centers,
                diagnostics=diagnostics,
            )
            processed_boxes += centers.shape[0]
            if progress_callback is not None:
                progress_callback(processed_boxes)
    generic_result = accumulator.result(
        name=name,
        notes=(
            "q0_constant_cell_coordinatewise_linear_derivative_envelope",
            "analytic_centered_rectangular_cross_and_second_moments",
            "eq:cell-polynomial-envelopes",
            "eq:q0-cell-moment-correction",
        ),
    )
    diagnostics = generic_result.diagnostics
    return InitialResidualL2BoundResult(
        l2_bound=generic_result.l2_bound,
        l2_squared_bound=generic_result.l2_squared_bound,
        midpoint_l2=generic_result.midpoint_l2,
        midpoint_l2_squared=generic_result.midpoint_l2_squared,
        bound_to_midpoint_ratio=generic_result.bound_to_midpoint_ratio,
        cell_l2_squared_bounds=generic_result.cell_l2_squared_bounds,
        phi_center_abs_bounds=diagnostics.get("phi_center_abs_bounds"),
        eta_g=diagnostics.get("eta_g"),
        eta_F=diagnostics.get("eta_F"),
        eta_B=diagnostics.get("eta_B"),
        eta_v=diagnostics.get("eta_v"),
        eta_phi=diagnostics.get("eta_phi"),
        F_center=diagnostics.get("F_center"),
        B_center=diagnostics.get("B_center"),
        g_center=diagnostics.get("g_center"),
        centers=generic_result.centers,
        spatial_eps=spatial_eps,
        cell_volume=generic_result.cell_volume,
        certified=generic_result.certified,
        notes=generic_result.notes,
        generic_result=generic_result,
    )

def bound_initial_displacement_residual_l2_moment_uniform(
    *,
    model: RigPINN_tanh,
    grid: UniformBoxGrid | UniformUnitBoxGridSpec,
    initial_g: ScalarLipschitzFunctionCertificate,
    n_taylor: int | None = None,
    safety_factor: float = 1.0,
    rounding_slack: float = 0.0,
    batch_size: int | None = None,
    store_diagnostics: bool = True,
    progress_callback=None,
) -> InitialResidualL2BoundResult:
    def cell_integral_fn(**kwargs):
        datum = kwargs.pop("datum")
        return _initial_displacement_moment_cell_integrals(initial_g=datum, **kwargs)

    return _run_initial_moment_residual_l2_uniform(
        model=model,
        grid=grid,
        datum=initial_g,
        cell_integral_fn=cell_integral_fn,
        name="initial_displacement",
        n_taylor=n_taylor,
        safety_factor=safety_factor,
        rounding_slack=rounding_slack,
        batch_size=batch_size,
        store_diagnostics=store_diagnostics,
        progress_callback=progress_callback,
    )

def bound_initial_velocity_residual_l2_moment_uniform(
    *,
    model: RigPINN_tanh,
    grid: UniformBoxGrid | UniformUnitBoxGridSpec,
    initial_velocity: ScalarLipschitzFunctionCertificate,
    n_taylor: int | None = None,
    safety_factor: float = 1.0,
    rounding_slack: float = 0.0,
    batch_size: int | None = None,
    store_diagnostics: bool = True,
    progress_callback=None,
) -> InitialResidualL2BoundResult:
    def cell_integral_fn(**kwargs):
        datum = kwargs.pop("datum")
        return _initial_velocity_moment_cell_integrals(
            initial_velocity=datum, **kwargs
        )

    return _run_initial_moment_residual_l2_uniform(
        model=model,
        grid=grid,
        datum=initial_velocity,
        cell_integral_fn=cell_integral_fn,
        name="initial_velocity",
        n_taylor=n_taylor,
        safety_factor=safety_factor,
        rounding_slack=rounding_slack,
        batch_size=batch_size,
        store_diagnostics=store_diagnostics,
        progress_callback=progress_callback,
    )
