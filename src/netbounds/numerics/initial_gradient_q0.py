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

from ._initial_support.boundary import require_spatial_product_boundary

from ._initial_support.grids import (
    UniformBoxGrid,
    UniformUnitBoxGridSpec,
    iter_uniform_grid_batches,
)

from ._initial_support.moment import linear_gradient_remainder_moment_integral

from ._initial_support.pde_helpers import (
    _boundary_derivative_center,
    _boundary_derivative_sup,
    _multi_choose,
    _sub_counts,
)

from ._initial_support.quadrature import (
    L2QuadratureAccumulator,
    ResidualL2BoundResult,
    require_nonnegative,
)

class VectorModulusFunctionCertificate(Protocol):
    """Vector field values with certified componentwise box variations."""

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Return values with shape [M, d]."""
        ...

    def variation_bound(self, centers: torch.Tensor, radius: float) -> torch.Tensor:
        """Return componentwise variation bounds with shape [M, d]."""
        ...

@dataclass
class VectorLipschitzInitialDatum:
    """Vector datum with componentwise l1-gradient Lipschitz bounds.

    ``component_lipschitz_l1`` may be a scalar applied to all components or a
    length-d tensor/list.  The variation bound is ``L_i * radius`` for each
    component, matching the l-infinity cell radius convention used by the
    uniform grids.
    """

    value_fn: Callable[[torch.Tensor], torch.Tensor]
    component_lipschitz_l1: float | torch.Tensor

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_fn(x)

    def variation_bound(self, centers: torch.Tensor, radius: float) -> torch.Tensor:
        values = self.value(centers)
        if values.ndim != 2:
            raise ValueError("vector datum values must have shape [M, d].")
        lipschitz = torch.as_tensor(
            self.component_lipschitz_l1,
            dtype=centers.dtype,
            device=centers.device,
        )
        if lipschitz.ndim == 0:
            lipschitz = lipschitz.expand(values.shape[1])
        if lipschitz.shape != (values.shape[1],):
            raise ValueError("component_lipschitz_l1 must be scalar or have shape [d].")
        return torch.full_like(values, float(radius)) * lipschitz.unsqueeze(0)

    def component_lipschitz_bound(self, centers: torch.Tensor) -> torch.Tensor:
        values = self.value(centers)
        if values.ndim != 2:
            raise ValueError("vector datum values must have shape [M, d].")
        lipschitz = torch.as_tensor(
            self.component_lipschitz_l1,
            dtype=centers.dtype,
            device=centers.device,
        )
        if lipschitz.ndim == 0:
            lipschitz = lipschitz.expand(values.shape[1])
        if lipschitz.shape != (values.shape[1],):
            raise ValueError("component_lipschitz_l1 must be scalar or have shape [d].")
        return lipschitz.unsqueeze(0).expand(centers.shape[0], -1)

class VectorTaylorFunctionCertificate(VectorModulusFunctionCertificate, Protocol):
    """Vector field values, exact Jacobian centers, and Jacobian moduli."""

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Return component Jacobians with shape [M, d, d]."""
        ...

    def hessian_l1_bound(self, centers: torch.Tensor, radius: float) -> torch.Tensor:
        """Return componentwise sum_ij sup |partial_ij field_a|, shape [M, d]."""
        ...

    def jacobian_variation_bound(
        self,
        centers: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        """Return componentwise Jacobian moduli with shape [M, d, d]."""
        ...

    def component_hessian_abs_bound(
        self,
        centers: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        """Return second-derivative bounds for each component, shape [M,d,d,d]."""
        ...

@dataclass
class VectorTaylorInitialDatum:
    """Vector datum for Taylor-cell quadrature using analytic derivatives."""

    value_fn: Callable[[torch.Tensor], torch.Tensor]
    jacobian_fn: Callable[[torch.Tensor], torch.Tensor]
    component_hessian_l1_bound: float | torch.Tensor | Callable[[torch.Tensor, float], torch.Tensor]
    jacobian_variation_bound_value: (
        float | torch.Tensor | Callable[[torch.Tensor, float], torch.Tensor] | None
    ) = None
    component_hessian_abs_bound_value: (
        float | torch.Tensor | Callable[[torch.Tensor, float], torch.Tensor] | None
    ) = None

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_fn(x)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        values = self.jacobian_fn(x)
        if values.ndim != 3 or values.shape[0] != x.shape[0] or values.shape[1] != values.shape[2]:
            raise ValueError("vector datum jacobian must have shape [M, d, d].")
        return values

    def hessian_l1_bound(self, centers: torch.Tensor, radius: float) -> torch.Tensor:
        values = self.value(centers)
        if values.ndim != 2:
            raise ValueError("vector datum values must have shape [M, d].")
        if callable(self.component_hessian_l1_bound):
            bound = self.component_hessian_l1_bound(centers, radius)
        else:
            bound = self.component_hessian_l1_bound
        bound = torch.as_tensor(bound, dtype=centers.dtype, device=centers.device)
        if bound.ndim == 0:
            bound = bound.expand(values.shape[0], values.shape[1])
        elif bound.ndim == 1:
            if bound.shape != (values.shape[1],):
                raise ValueError("component_hessian_l1_bound vector must have shape [d].")
            bound = bound.unsqueeze(0).expand(values.shape[0], values.shape[1])
        if bound.shape != values.shape:
            raise ValueError("component_hessian_l1_bound must broadcast to shape [M, d].")
        return require_nonnegative("component_hessian_l1_bound", bound)

    def jacobian_variation_bound(
        self,
        centers: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        """Return ``omega_g^(e_j+e_l)(radius)`` for every ``j,l``.

        If no entrywise modulus is supplied, the component Hessian l1 bound
        gives a conservative compatibility fallback. Production campaigns
        should provide the manuscript's analytic entrywise data modulus.
        """
        values = self.value(centers)
        if values.ndim != 2:
            raise ValueError("vector datum values must have shape [M, d].")
        M, d = values.shape
        source = self.jacobian_variation_bound_value
        if source is None:
            bound = (
                float(radius)
                * self.hessian_l1_bound(centers, radius).unsqueeze(-1)
            ).expand(M, d, d)
        else:
            bound = source(centers, radius) if callable(source) else source
            bound = torch.as_tensor(bound, dtype=centers.dtype, device=centers.device)
            if bound.ndim == 0:
                bound = bound.expand(M, d, d)
            elif bound.shape == (d, d):
                bound = bound.unsqueeze(0).expand(M, d, d)
        if bound.shape != (M, d, d):
            raise ValueError("jacobian_variation_bound_value must broadcast to shape [M, d, d].")
        return require_nonnegative("jacobian_variation_bound", bound)

    def component_hessian_abs_bound(
        self,
        centers: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        values = self.value(centers)
        M, d = values.shape
        source = self.component_hessian_abs_bound_value
        if source is None:
            component_l1 = self.hessian_l1_bound(centers, radius)
            return component_l1[:, :, None, None].expand(M, d, d, d)
        bound = source(centers, radius) if callable(source) else source
        bound = torch.as_tensor(bound, dtype=centers.dtype, device=centers.device)
        if bound.ndim == 0:
            bound = bound.expand(M, d, d, d)
        elif bound.shape == (d, d, d):
            bound = bound.unsqueeze(0).expand(M, d, d, d)
        if bound.shape != (M, d, d, d):
            raise ValueError(
                "component_hessian_abs_bound_value must broadcast to shape [M,d,d,d]"
            )
        return require_nonnegative("component_hessian_abs_bound", bound)

    def variation_bound(self, centers: torch.Tensor, radius: float) -> torch.Tensor:
        # Compatibility with the older envelope quadrature path.
        return float(radius) * self.hessian_l1_bound(centers, radius)

@dataclass
class InitialGradientResidualL2BoundResult:
    l2_bound: torch.Tensor
    l2_squared_bound: torch.Tensor
    midpoint_l2: torch.Tensor | None
    midpoint_l2_squared: torch.Tensor | None
    bound_to_midpoint_ratio: torch.Tensor | None
    cell_l2_squared_bounds: torch.Tensor | None
    phi_center_norm_bounds: torch.Tensor | None
    eta_grad_g: torch.Tensor | None
    eta_F: torch.Tensor | None
    eta_B: torch.Tensor | None
    eta_grad_B: torch.Tensor | None
    eta_grad_v: torch.Tensor | None
    eta_phi_components: torch.Tensor | None
    F_center: torch.Tensor | None
    B_center: torch.Tensor | None
    grad_B_center: torch.Tensor | None
    grad_g_center: torch.Tensor | None
    approx_grad_center: torch.Tensor | None
    centers: torch.Tensor | None
    spatial_eps: float
    cell_volume: torch.Tensor
    certified: bool = True
    notes: tuple[str, ...] = ()
    generic_result: ResidualL2BoundResult | None = None

def _spacetime_initial_centers(centers: torch.Tensor, model: RigPINN_tanh) -> torch.Tensor:
    y = torch.zeros(
        centers.shape[0],
        model.input_dim,
        dtype=centers.dtype,
        device=centers.device,
    )
    y[:, : model.d] = centers
    y[:, model.input_dim - 1] = 0.0
    return y

def _as_cell_matrix(
    name: str,
    values: torch.Tensor,
    *,
    length: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=dtype, device=device)
    if values.shape == (length, width):
        return values
    if values.shape == (length, width, 1):
        return values.squeeze(-1)
    raise ValueError(f"{name} must have shape [{length}, {width}].")

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

def _network_derivative_center(
    *,
    counts: tuple[int, ...],
    first,
    second,
    third,
) -> torch.Tensor:
    order = sum(counts)
    if order == 0:
        return first.base_value.squeeze(-1)
    key = _spatial_counts_to_key(counts)
    if order == 1:
        return first.base_gradient[:, key[0]]
    if order == 2:
        return second.base_hessian[:, _index_column(second.pairs, key)]
    if order == 3:
        return third.base_third_derivatives[:, _index_column(third.triples, key)]
    raise ValueError("only derivatives up to order 3 are supported")

def _network_derivative_sup(
    *,
    counts: tuple[int, ...],
    spatial_eps: float,
    first,
    second,
    third,
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
    if order == 3:
        col = _index_column(third.triples, key)
        return third.base_third_derivatives[:, col].abs() + third.third_derivative_bounds[:, col]
    raise ValueError("only derivatives up to order 3 are supported")

def _bf_derivative_center(
    *,
    spatial: torch.Tensor,
    counts: tuple[int, ...],
    first,
    second,
    third,
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
        ) * _network_derivative_center(
            counts=gamma_spatial,
            first=first,
            second=second,
            third=third,
        )
    return total

def _bf_derivative_sup(
    *,
    lower: torch.Tensor,
    upper: torch.Tensor,
    counts: tuple[int, ...],
    spatial_eps: float,
    first,
    second,
    third,
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
            third=third,
        )
    return require_nonnegative("bf_derivative_sup", total)

def _displacement_gradient_moment_cell_bounds(
    *,
    model: RigPINN_tanh,
    centers: torch.Tensor,
    spatial_eps: float,
    initial_grad: VectorLipschitzInitialDatum,
    n_taylor: int | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Literal componentwise appendix Q0 moment bound for ``grad(g-BF)``."""

    M, d = centers.shape
    dtype, device = centers.dtype, centers.device
    spacetime_centers = _spacetime_initial_centers(centers, model)
    first = compute_value_and_first_derivative_bounds_over_spatial_box(
        model, spacetime_centers, spatial_eps, n_taylor
    )
    second_pairs = tuple((i, q) for i in range(d) for q in range(i, d))
    second = compute_second_order_bounds(
        model,
        spacetime_centers,
        spatial_eps,
        n_taylor,
        multi_indices=second_pairs,
        spatial_dim=model.d,
    )
    lower = centers - float(spatial_eps)
    upper = centers + float(spatial_eps)
    approx_grad_center = torch.empty(M, d, dtype=dtype, device=device)
    approx_second_abs = torch.empty(M, d, d, dtype=dtype, device=device)
    for j in range(d):
        counts_j = tuple(int(axis == j) for axis in range(d))
        approx_grad_center[:, j] = _bf_derivative_center(
            spatial=centers,
            counts=counts_j,
            first=first,
            second=second,
            third=None,
            input_dim=model.input_dim,
        )
        for q in range(d):
            counts_jq = list(counts_j)
            counts_jq[q] += 1
            approx_second_abs[:, j, q] = _bf_derivative_sup(
                lower=lower,
                upper=upper,
                counts=tuple(counts_jq),
                spatial_eps=spatial_eps,
                first=first,
                second=second,
                third=None,
                input_dim=model.input_dim,
            )

    grad_g_center = _as_cell_matrix(
        "grad_g_center",
        initial_grad.value(centers),
        length=M,
        width=d,
        dtype=dtype,
        device=device,
    )
    component_lipschitz = _as_cell_matrix(
        "initial_gradient_component_lipschitz",
        initial_grad.component_lipschitz_bound(centers),
        length=M,
        width=d,
        dtype=dtype,
        device=device,
    )
    component_lipschitz = require_nonnegative(
        "initial_gradient_component_lipschitz", component_lipschitz
    )
    data_first_abs = torch.minimum(
        component_lipschitz[:, :, None], component_lipschitz[:, None, :]
    )
    first_derivative_abs_bounds = require_nonnegative(
        "initial_gradient_first_derivative_abs_bounds",
        data_first_abs + approx_second_abs,
    )
    residual_center_components = grad_g_center - approx_grad_center
    midpoint_residual = torch.linalg.vector_norm(residual_center_components, dim=-1)
    cell_volume = torch.as_tensor(
        (2.0 * float(spatial_eps)) ** d, dtype=dtype, device=device
    )
    component_integrals = torch.empty(M, d, dtype=dtype, device=device)
    component_cross = torch.empty_like(component_integrals)
    component_remainder = torch.empty_like(component_integrals)
    for j in range(d):
        moments = linear_gradient_remainder_moment_integral(
            residual_center=residual_center_components[:, j],
            first_derivative_abs_bounds=first_derivative_abs_bounds[:, j, :],
            spatial_eps=spatial_eps,
            cell_volume=cell_volume,
        )
        component_integrals[:, j] = moments["selected"]
        component_cross[:, j] = moments["cross_correction"]
        component_remainder[:, j] = moments["remainder_l2_squared"]
    cell_l2_squared = require_nonnegative(
        "initial_gradient_q0_moment_cell_l2_squared",
        component_integrals.sum(dim=-1),
    )
    diagnostics = {
        "phi_center_norm_bounds": midpoint_residual,
        "eta_grad_g": component_lipschitz,
        "eta_grad_v": approx_second_abs.sum(dim=-1),
        "eta_phi_components": first_derivative_abs_bounds.sum(dim=-1),
        "F_center": first.base_value.squeeze(-1),
        "B_center": _boundary_derivative_center(centers, (0,) * model.input_dim),
        "grad_g_center": grad_g_center,
        "approx_grad_center": approx_grad_center,
        "first_derivative_abs_bounds": first_derivative_abs_bounds,
        "moment_component_l2_squared": component_integrals,
        "moment_component_cross_correction": component_cross,
        "moment_component_remainder_l2_squared": component_remainder,
    }
    return torch.sqrt((cell_l2_squared / cell_volume).clamp_min(0.0)), diagnostics

def _run_initial_gradient_residual_l2_uniform(
    *,
    model: RigPINN_tanh,
    grid: UniformBoxGrid | UniformUnitBoxGridSpec,
    datum: VectorModulusFunctionCertificate,
    cell_bound_fn,
    name: str,
    n_taylor: int | None = None,
    safety_factor: float = 1.0,
    rounding_slack: float = 0.0,
    batch_size: int | None = None,
    store_diagnostics: bool = True,
    progress_callback=None,
) -> InitialGradientResidualL2BoundResult:
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
            cell_residual_bound, diagnostics = cell_bound_fn(
                model=model,
                centers=centers,
                spatial_eps=spatial_eps,
                datum=datum,
                n_taylor=n_taylor,
            )
            if safety_factor != 1.0:
                cell_residual_bound = float(safety_factor) * cell_residual_bound
            if rounding_slack != 0.0:
                cell_residual_bound = cell_residual_bound + float(rounding_slack)
            cell_residual_bound = require_nonnegative("cell_residual_bound", cell_residual_bound)
            accumulator.add_cell_bounds(
                cell_residual_bound,
                midpoint_residual=diagnostics["phi_center_norm_bounds"],
                centers=centers,
                diagnostics=diagnostics,
            )
            processed_boxes += centers.shape[0]
            if progress_callback is not None:
                progress_callback(processed_boxes)

    generic_result = accumulator.result(name=name)
    diagnostics = generic_result.diagnostics
    return InitialGradientResidualL2BoundResult(
        l2_bound=generic_result.l2_bound,
        l2_squared_bound=generic_result.l2_squared_bound,
        midpoint_l2=generic_result.midpoint_l2,
        midpoint_l2_squared=generic_result.midpoint_l2_squared,
        bound_to_midpoint_ratio=generic_result.bound_to_midpoint_ratio,
        cell_l2_squared_bounds=generic_result.cell_l2_squared_bounds,
        phi_center_norm_bounds=diagnostics.get("phi_center_norm_bounds"),
        eta_grad_g=diagnostics.get("eta_grad_g"),
        eta_F=diagnostics.get("eta_F"),
        eta_B=diagnostics.get("eta_B"),
        eta_grad_B=diagnostics.get("eta_grad_B"),
        eta_grad_v=diagnostics.get("eta_grad_v"),
        eta_phi_components=diagnostics.get("eta_phi_components"),
        F_center=diagnostics.get("F_center"),
        B_center=diagnostics.get("B_center"),
        grad_B_center=diagnostics.get("grad_B_center"),
        grad_g_center=diagnostics.get("grad_g_center"),
        approx_grad_center=diagnostics.get("approx_grad_center"),
        centers=generic_result.centers,
        spatial_eps=spatial_eps,
        cell_volume=generic_result.cell_volume,
        certified=generic_result.certified,
        notes=generic_result.notes,
        generic_result=generic_result,
    )

def bound_initial_displacement_gradient_residual_l2_moment_uniform(
    *,
    model: RigPINN_tanh,
    grid: UniformBoxGrid | UniformUnitBoxGridSpec,
    initial_grad: VectorLipschitzInitialDatum,
    n_taylor: int | None = None,
    safety_factor: float = 1.0,
    rounding_slack: float = 0.0,
    batch_size: int | None = None,
    store_diagnostics: bool = True,
    progress_callback=None,
) -> InitialGradientResidualL2BoundResult:
    """Literal componentwise centered-moment Q0 initial-gradient bound."""

    def cell_bound_fn(**kwargs):
        datum = kwargs.pop("datum")
        return _displacement_gradient_moment_cell_bounds(
            initial_grad=datum, **kwargs
        )

    result = _run_initial_gradient_residual_l2_uniform(
        model=model,
        grid=grid,
        datum=initial_grad,
        cell_bound_fn=cell_bound_fn,
        name="initial_displacement_gradient",
        n_taylor=n_taylor,
        safety_factor=safety_factor,
        rounding_slack=rounding_slack,
        batch_size=batch_size,
        store_diagnostics=store_diagnostics,
        progress_callback=progress_callback,
    )
    moment_notes = (
        "q0_constant_cell_coordinatewise_linear_derivative_envelope",
        "component_integrals_summed_before_global_square_root",
        "eq:q0-cell-moment-correction",
        "eq:appendix-initial-gradient-first-coefficients",
    )
    result.notes = result.notes + moment_notes
    if result.generic_result is not None:
        result.generic_result.notes = result.generic_result.notes + moment_notes
    return result
