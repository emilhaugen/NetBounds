"""Reached 1D kernel extracted from producer revision 1003b5448260462e888dede918904d8391fd2a78."""

from __future__ import annotations

import math

from collections.abc import Callable

from dataclasses import dataclass

from typing import Protocol

import torch

from ._bounds import (
    compute_second_order_bounds,
    compute_third_order_bounds,
    compute_value_and_first_derivative_bounds_over_spatial_box,
)

from .model import TanhNetwork as RigPINN_tanh

from ._initial_support.boundary import (
    product_boundary_derivative_modulus,
    product_boundary_derivative_value,
    require_spatial_product_boundary,
)

from ._initial_support.modulus_data import ModulusFunctionCertificate

from ._initial_support.moment import quadratic_hessian_remainder_moment_integral

from ._initial_support.pde_helpers import (
    _boundary_derivative_center,
    _boundary_derivative_sup,
    _centered_affine_abs_integral,
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


def _initial_displacement_taylor_cell_integrals(
    *,
    model: RigPINN_tanh,
    centers: torch.Tensor,
    spatial_eps: float,
    initial_g: ScalarTaylorFunctionCertificate,
    n_taylor: int | None,
    affine_abs_integral: str,
    remainder_integration: str = "corner",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    if remainder_integration not in ("corner", "quadratic_moment"):
        raise ValueError("remainder_integration must be 'corner' or 'quadratic_moment'")
    M = centers.shape[0]
    dtype = centers.dtype
    device = centers.device
    d = model.d
    spacetime_centers = _spacetime_initial_centers(centers, model)
    second_pairs = tuple((i, j) for i in range(d) for j in range(i, d))
    first = compute_value_and_first_derivative_bounds_over_spatial_box(model, spacetime_centers, spatial_eps, n_taylor)
    second = compute_second_order_bounds(
        model,
        spacetime_centers,
        spatial_eps,
        n_taylor,
        multi_indices=second_pairs,
        spatial_dim=model.d,
    )

    g_center = as_cell_vector(
        "g_center",
        initial_g.value(centers),
        length=M,
        dtype=dtype,
        device=device,
    )
    grad_g_center = torch.as_tensor(initial_g.gradient(centers), dtype=dtype, device=device)
    if grad_g_center.shape != (M, d):
        raise ValueError("initial_g.gradient must have shape [M, d].")
    g_hessian_l1 = as_cell_vector(
        "initial_g_hessian_l1_bound",
        initial_g.hessian_l1_bound(centers, spatial_eps),
        length=M,
        dtype=dtype,
        device=device,
    )
    g_hessian_abs: torch.Tensor | None = None
    if remainder_integration == "quadratic_moment":
        g_hessian_abs = torch.as_tensor(
            initial_g.hessian_abs_bound(centers, spatial_eps),
            dtype=dtype,
            device=device,
        )
        if g_hessian_abs.shape != (M, d, d):
            raise ValueError("initial_g.hessian_abs_bound must have shape [M,d,d]")
        g_hessian_abs = require_nonnegative(
            "initial_g_hessian_abs_bound", g_hessian_abs
        )

    approx_center = _bf_derivative_center(
        spatial=centers,
        counts=(0,) * d,
        first=first,
        second=second,
        input_dim=model.input_dim,
    )
    approx_grad_center = torch.zeros(M, d, dtype=dtype, device=device)
    approx_hessian_abs = torch.zeros(M, d, d, dtype=dtype, device=device)
    lower = centers - float(spatial_eps)
    upper = centers + float(spatial_eps)
    for i in range(d):
        counts_i = tuple(1 if axis == i else 0 for axis in range(d))
        approx_grad_center[:, i] = _bf_derivative_center(
            spatial=centers,
            counts=counts_i,
            first=first,
            second=second,
            input_dim=model.input_dim,
        )
        for j in range(d):
            counts_ij = list(counts_i)
            counts_ij[j] += 1
            approx_hessian_abs[:, i, j] = _bf_derivative_paper_sup(
                spatial=centers,
                counts=tuple(counts_ij),
                spatial_eps=spatial_eps,
                first=first,
                second=second,
            )

    approx_hessian_l1 = approx_hessian_abs.sum(dim=(-2, -1))
    residual_center = g_center - approx_center
    residual_gradient = grad_g_center - approx_grad_center
    rho = require_nonnegative(
        "initial_displacement_taylor_remainder",
        0.5 * float(spatial_eps) ** 2 * (g_hessian_l1 + approx_hessian_l1),
    )
    cell_volume = torch.as_tensor((2.0 * float(spatial_eps)) ** d, dtype=dtype, device=device)
    int_p2 = cell_volume * (residual_center.square() + (float(spatial_eps) ** 2 / 3.0) * residual_gradient.square().sum(dim=-1))
    if affine_abs_integral == "cauchy_schwarz":
        int_abs_p = torch.sqrt((cell_volume * int_p2).clamp_min(0.0))
    elif affine_abs_integral == "exact":
        int_abs_p = _centered_affine_abs_integral(residual_center, residual_gradient, float(spatial_eps))
    else:
        raise ValueError("affine_abs_integral must be 'cauchy_schwarz' or 'exact'.")
    corner_cell_l2_squared = require_nonnegative(
        "initial_displacement_taylor_cell_integral",
        int_p2 + 2.0 * rho * int_abs_p + cell_volume * rho.square(),
    )
    moment_diagnostics: dict[str, torch.Tensor] = {}
    if remainder_integration == "quadratic_moment":
        if g_hessian_abs is None:
            raise RuntimeError("data Hessian bounds were not constructed")
        hessian_abs_bounds = require_nonnegative(
            "initial_displacement_residual_hessian_abs_bounds",
            g_hessian_abs + approx_hessian_abs,
        )
        moments = quadratic_hessian_remainder_moment_integral(
            residual_center=residual_center,
            residual_gradient=residual_gradient,
            hessian_abs_bounds=hessian_abs_bounds,
            spatial_eps=spatial_eps,
            cell_volume=cell_volume,
        )
        cell_l2_squared_bounds = moments["selected"]
        moment_diagnostics = {
            "hessian_abs_bounds": hessian_abs_bounds,
            "moment_affine_l2_squared": moments["int_p2"],
            "moment_cross_correction": moments["cross_correction"],
            "moment_remainder_l2_squared": moments["remainder_l2_squared"],
            "corner_configured_l2_squared": corner_cell_l2_squared,
        }
    else:
        cell_l2_squared_bounds = corner_cell_l2_squared
    diagnostics = {
        "phi_center_abs_bounds": residual_center.abs(),
        "eta_g": g_hessian_l1,
        "eta_F": approx_hessian_l1,
        "eta_v": approx_hessian_l1,
        "eta_phi": rho,
        "F_center": first.base_value.squeeze(-1),
        "B_center": _boundary_derivative_center(centers, (0,) * model.input_dim),
        "g_center": g_center,
        "approx_center": approx_center,
        "grad_g_center": grad_g_center,
        "approx_grad_center": approx_grad_center,
        **moment_diagnostics,
    }
    return cell_l2_squared_bounds, residual_center.abs(), diagnostics

def bound_initial_displacement_residual_l2_taylor_uniform(
    *,
    model: RigPINN_tanh,
    grid: UniformBoxGrid | UniformUnitBoxGridSpec,
    initial_g: ScalarTaylorFunctionCertificate,
    n_taylor: int | None = None,
    safety_factor: float = 1.0,
    rounding_slack: float = 0.0,
    batch_size: int | None = None,
    store_diagnostics: bool = True,
    affine_abs_integral: str = "exact",
    progress_callback=None,
    remainder_integration: str = "corner",
) -> InitialResidualL2BoundResult:
    if not math.isfinite(float(safety_factor)) or safety_factor < 1.0:
        raise ValueError("safety_factor must be finite and at least 1.")
    if not math.isfinite(float(rounding_slack)) or rounding_slack < 0.0:
        raise ValueError("rounding_slack must be finite and nonnegative.")
    if affine_abs_integral not in ("cauchy_schwarz", "exact"):
        raise ValueError("affine_abs_integral must be 'cauchy_schwarz' or 'exact'.")
    if remainder_integration not in ("corner", "quadratic_moment"):
        raise ValueError("remainder_integration must be 'corner' or 'quadratic_moment'")
    require_spatial_product_boundary(model.boundary_function, model.d)
    grid.validate()
    if grid.d != model.d:
        raise ValueError("grid.d must equal model.d.")
    dtype = model.input_layer.weight.dtype
    device = model.input_layer.weight.device
    spatial_eps = float(grid.spatial_eps)
    cell_volume = torch.as_tensor((2.0 * spatial_eps) ** model.d, dtype=dtype, device=device)
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
            cell_l2_squared_bounds, midpoint_residual, diagnostics = _initial_displacement_taylor_cell_integrals(
                model=model,
                centers=centers,
                spatial_eps=spatial_eps,
                initial_g=initial_g,
                n_taylor=n_taylor,
                affine_abs_integral=affine_abs_integral,
                remainder_integration=remainder_integration,
            )
            if safety_factor != 1.0:
                cell_l2_squared_bounds = float(safety_factor) ** 2 * cell_l2_squared_bounds
            if rounding_slack != 0.0:
                cell_l2_squared_bounds = cell_l2_squared_bounds + cell_volume * float(rounding_slack) ** 2
            cell_residual_bound = torch.sqrt((cell_l2_squared_bounds / cell_volume).clamp_min(0.0))
            accumulator.add_cell_bounds(
                cell_residual_bound,
                midpoint_residual=midpoint_residual,
                centers=centers,
                diagnostics=diagnostics,
            )
            processed_boxes += centers.shape[0]
            if progress_callback is not None:
                progress_callback(processed_boxes)
    notes = ["first_order_taylor_cell_hessian_remainder_second_order_derivative_bounds"]
    if remainder_integration == "quadratic_moment":
        notes.extend(
            (
                "ima_section2_quadratic_hessian_remainder_moments",
                "analytic_centered_rectangular_cross_and_squared_remainder",
            )
        )
    generic_result = accumulator.result(
        name="initial_displacement",
        notes=tuple(notes),
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

def bound_initial_displacement_residual_l2_moment_taylor_uniform(
    *,
    model: RigPINN_tanh,
    grid: UniformBoxGrid | UniformUnitBoxGridSpec,
    initial_g: ScalarTaylorFunctionCertificate,
    n_taylor: int | None = None,
    safety_factor: float = 1.0,
    rounding_slack: float = 0.0,
    batch_size: int | None = None,
    store_diagnostics: bool = True,
    progress_callback=None,
) -> InitialResidualL2BoundResult:
    """Literal Section 2/4 affine-Q1 quadratic-remainder moment bound."""

    return bound_initial_displacement_residual_l2_taylor_uniform(
        model=model,
        grid=grid,
        initial_g=initial_g,
        n_taylor=n_taylor,
        safety_factor=safety_factor,
        rounding_slack=rounding_slack,
        batch_size=batch_size,
        store_diagnostics=store_diagnostics,
        affine_abs_integral="cauchy_schwarz",
        progress_callback=progress_callback,
        remainder_integration="quadratic_moment",
    )

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


def _initial_velocity_taylor_cell_integrals(
    *,
    model: RigPINN_tanh,
    centers: torch.Tensor,
    spatial_eps: float,
    initial_velocity: ScalarTaylorFunctionCertificate,
    n_taylor: int | None,
    affine_abs_integral: str,
    remainder_integration: str = "corner",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Certified Q1 cell integrals for ``h - B F_t`` on the initial slice."""
    if remainder_integration not in ("corner", "quadratic_moment"):
        raise ValueError("remainder_integration must be 'corner' or 'quadratic_moment'")
    M = centers.shape[0]
    dtype = centers.dtype
    device = centers.device
    d = model.d
    time_idx = model.input_dim - 1
    spacetime_centers = _spacetime_initial_centers(centers, model)
    mixed_pairs = tuple((axis, time_idx) for axis in range(d))
    mixed_third = tuple(
        (i, j, time_idx)
        for i in range(d)
        for j in range(i, d)
    )
    first = compute_value_and_first_derivative_bounds_over_spatial_box(
        model,
        spacetime_centers,
        spatial_eps,
        n_taylor,
    )
    second = compute_second_order_bounds(
        model,
        spacetime_centers,
        spatial_eps,
        n_taylor,
        multi_indices=mixed_pairs,
        spatial_dim=model.d,
    )
    third = compute_third_order_bounds(
        model,
        spacetime_centers,
        spatial_eps,
        n_taylor,
        multi_indices=mixed_third,
        spatial_dim=model.d,
    )

    velocity_center = as_cell_vector(
        "initial_velocity_center",
        initial_velocity.value(centers),
        length=M,
        dtype=dtype,
        device=device,
    )
    grad_velocity_center = torch.as_tensor(
        initial_velocity.gradient(centers),
        dtype=dtype,
        device=device,
    )
    if grad_velocity_center.shape != (M, d):
        raise ValueError("initial_velocity.gradient must have shape [M, d].")
    velocity_hessian_l1 = as_cell_vector(
        "initial_velocity_hessian_l1_bound",
        initial_velocity.hessian_l1_bound(centers, spatial_eps),
        length=M,
        dtype=dtype,
        device=device,
    )
    velocity_hessian_abs: torch.Tensor | None = None
    if remainder_integration == "quadratic_moment":
        velocity_hessian_abs = torch.as_tensor(
            initial_velocity.hessian_abs_bound(centers, spatial_eps),
            dtype=dtype,
            device=device,
        )
        if velocity_hessian_abs.shape != (M, d, d):
            raise ValueError("initial_velocity.hessian_abs_bound must have shape [M,d,d]")
        velocity_hessian_abs = require_nonnegative(
            "initial_velocity_hessian_abs_bound", velocity_hessian_abs
        )

    approx_center = _bft_derivative_center(
        spatial=centers,
        counts=(0,) * d,
        time_idx=time_idx,
        first=first,
        second=second,
        third=third,
        input_dim=model.input_dim,
    )
    approx_grad_center = torch.zeros(M, d, dtype=dtype, device=device)
    approx_hessian_abs = torch.zeros(M, d, d, dtype=dtype, device=device)
    lower = centers - float(spatial_eps)
    upper = centers + float(spatial_eps)
    for i in range(d):
        counts_i = tuple(1 if axis == i else 0 for axis in range(d))
        approx_grad_center[:, i] = _bft_derivative_center(
            spatial=centers,
            counts=counts_i,
            time_idx=time_idx,
            first=first,
            second=second,
            third=third,
            input_dim=model.input_dim,
        )
        for j in range(d):
            counts_ij = list(counts_i)
            counts_ij[j] += 1
            approx_hessian_abs[:, i, j] = _bft_derivative_paper_sup(
                spatial=centers,
                counts=tuple(counts_ij),
                spatial_eps=spatial_eps,
                time_idx=time_idx,
                first=first,
                second=second,
                third=third,
            )

    approx_hessian_l1 = approx_hessian_abs.sum(dim=(-2, -1))
    residual_center = velocity_center - approx_center
    residual_gradient = grad_velocity_center - approx_grad_center
    rho = require_nonnegative(
        "initial_velocity_taylor_remainder",
        0.5 * float(spatial_eps) ** 2 * (velocity_hessian_l1 + approx_hessian_l1),
    )
    cell_volume = torch.as_tensor(
        (2.0 * float(spatial_eps)) ** d,
        dtype=dtype,
        device=device,
    )
    int_p2 = cell_volume * (
        residual_center.square()
        + (float(spatial_eps) ** 2 / 3.0) * residual_gradient.square().sum(dim=-1)
    )
    if affine_abs_integral == "cauchy_schwarz":
        int_abs_p = torch.sqrt((cell_volume * int_p2).clamp_min(0.0))
    elif affine_abs_integral == "exact":
        int_abs_p = _centered_affine_abs_integral(
            residual_center,
            residual_gradient,
            float(spatial_eps),
        )
    else:
        raise ValueError("affine_abs_integral must be 'cauchy_schwarz' or 'exact'.")
    corner_cell_l2_squared = require_nonnegative(
        "initial_velocity_taylor_cell_integral",
        int_p2 + 2.0 * rho * int_abs_p + cell_volume * rho.square(),
    )
    moment_diagnostics: dict[str, torch.Tensor] = {}
    if remainder_integration == "quadratic_moment":
        if velocity_hessian_abs is None:
            raise RuntimeError("velocity Hessian bounds were not constructed")
        hessian_abs_bounds = require_nonnegative(
            "initial_velocity_residual_hessian_abs_bounds",
            velocity_hessian_abs + approx_hessian_abs,
        )
        moments = quadratic_hessian_remainder_moment_integral(
            residual_center=residual_center,
            residual_gradient=residual_gradient,
            hessian_abs_bounds=hessian_abs_bounds,
            spatial_eps=spatial_eps,
            cell_volume=cell_volume,
        )
        cell_l2_squared_bounds = moments["selected"]
        moment_diagnostics = {
            "hessian_abs_bounds": hessian_abs_bounds,
            "moment_affine_l2_squared": moments["int_p2"],
            "moment_cross_correction": moments["cross_correction"],
            "moment_remainder_l2_squared": moments["remainder_l2_squared"],
            "corner_configured_l2_squared": corner_cell_l2_squared,
        }
    else:
        cell_l2_squared_bounds = corner_cell_l2_squared
    diagnostics = {
        "phi_center_abs_bounds": residual_center.abs(),
        "eta_g": velocity_hessian_l1,
        "eta_F": approx_hessian_l1,
        "eta_v": approx_hessian_l1,
        "eta_phi": rho,
        "F_center": first.base_gradient[:, time_idx],
        "B_center": _boundary_derivative_center(centers, (0,) * model.input_dim),
        "g_center": velocity_center,
        "approx_center": approx_center,
        "grad_g_center": grad_velocity_center,
        "approx_grad_center": approx_grad_center,
        **moment_diagnostics,
    }
    return cell_l2_squared_bounds, residual_center.abs(), diagnostics

def bound_initial_velocity_residual_l2_taylor_uniform(
    *,
    model: RigPINN_tanh,
    grid: UniformBoxGrid | UniformUnitBoxGridSpec,
    initial_velocity: ScalarTaylorFunctionCertificate,
    n_taylor: int | None = None,
    safety_factor: float = 1.0,
    rounding_slack: float = 0.0,
    batch_size: int | None = None,
    store_diagnostics: bool = True,
    affine_abs_integral: str = "cauchy_schwarz",
    progress_callback=None,
    remainder_integration: str = "corner",
) -> InitialResidualL2BoundResult:
    """Bound the scalar initial-velocity residual with affine Q1 cells."""
    if not math.isfinite(float(safety_factor)) or safety_factor < 1.0:
        raise ValueError("safety_factor must be finite and at least 1.")
    if not math.isfinite(float(rounding_slack)) or rounding_slack < 0.0:
        raise ValueError("rounding_slack must be finite and nonnegative.")
    if affine_abs_integral not in ("cauchy_schwarz", "exact"):
        raise ValueError("affine_abs_integral must be 'cauchy_schwarz' or 'exact'.")
    if remainder_integration not in ("corner", "quadratic_moment"):
        raise ValueError("remainder_integration must be 'corner' or 'quadratic_moment'")
    require_spatial_product_boundary(model.boundary_function, model.d)
    grid.validate()
    if grid.d != model.d:
        raise ValueError("grid.d must equal model.d.")
    dtype = model.input_layer.weight.dtype
    device = model.input_layer.weight.device
    if affine_abs_integral == "exact" and dtype != torch.float64:
        raise ValueError(
            "exact affine-absolute integration requires float64; "
            "use cauchy_schwarz for float32 certificates"
        )
    spatial_eps = float(grid.spatial_eps)
    cell_volume = torch.as_tensor(
        (2.0 * spatial_eps) ** model.d,
        dtype=dtype,
        device=device,
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
            cell_l2_squared_bounds, midpoint_residual, diagnostics = (
                _initial_velocity_taylor_cell_integrals(
                    model=model,
                    centers=centers,
                    spatial_eps=spatial_eps,
                    initial_velocity=initial_velocity,
                    n_taylor=n_taylor,
                    affine_abs_integral=affine_abs_integral,
                    remainder_integration=remainder_integration,
                )
            )
            if safety_factor != 1.0:
                cell_l2_squared_bounds = float(safety_factor) ** 2 * cell_l2_squared_bounds
            cell_residual_bound = torch.sqrt(
                (cell_l2_squared_bounds / cell_volume).clamp_min(0.0)
            )
            if rounding_slack != 0.0:
                cell_residual_bound = cell_residual_bound + float(rounding_slack)
            accumulator.add_cell_bounds(
                cell_residual_bound,
                midpoint_residual=midpoint_residual,
                centers=centers,
                diagnostics=diagnostics,
            )
            processed_boxes += centers.shape[0]
            if progress_callback is not None:
                progress_callback(processed_boxes)
    notes = ["first_order_taylor_cell_hessian_remainder_third_order_derivative_bounds"]
    if remainder_integration == "quadratic_moment":
        notes.extend(
            (
                "ima_section2_quadratic_hessian_remainder_moments",
                "analytic_centered_rectangular_cross_and_squared_remainder",
            )
        )
    generic_result = accumulator.result(
        name="initial_velocity",
        notes=tuple(notes),
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

def bound_initial_velocity_residual_l2_moment_taylor_uniform(
    *,
    model: RigPINN_tanh,
    grid: UniformBoxGrid | UniformUnitBoxGridSpec,
    initial_velocity: ScalarTaylorFunctionCertificate,
    n_taylor: int | None = None,
    safety_factor: float = 1.0,
    rounding_slack: float = 0.0,
    batch_size: int | None = None,
    store_diagnostics: bool = True,
    progress_callback=None,
) -> InitialResidualL2BoundResult:
    """Literal Section 2/4 velocity affine-Q1 moment bound."""

    return bound_initial_velocity_residual_l2_taylor_uniform(
        model=model,
        grid=grid,
        initial_velocity=initial_velocity,
        n_taylor=n_taylor,
        safety_factor=safety_factor,
        rounding_slack=rounding_slack,
        batch_size=batch_size,
        store_diagnostics=store_diagnostics,
        affine_abs_integral="cauchy_schwarz",
        progress_callback=progress_callback,
        remainder_integration="quadratic_moment",
    )
