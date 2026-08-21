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

from ._initial_support.grids import (
    UniformBoxGrid,
    UniformUnitBoxGridSpec,
    iter_uniform_grid_batches,
)

from ._initial_support.moment import quadratic_hessian_remainder_moment_integral

from ._initial_support.pde_helpers import (
    _boundary_derivative_sup,
    _centered_affine_abs_integral,
    _multi_choose,
    _sub_counts,
)

from ._initial_support.quadrature import (
    L2QuadratureAccumulator,
    ResidualL2BoundResult,
    as_cell_vector,
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

def _initial_gradient_second_variation_product_bound(
    *,
    omega_hess_g: torch.Tensor,
    F_center: torch.Tensor,
    grad_F_center: torch.Tensor,
    hess_F_center: torch.Tensor,
    eta_F: torch.Tensor,
    eta_grad_F: torch.Tensor,
    eta_hess_F: torch.Tensor,
    B_center: torch.Tensor,
    grad_B_center: torch.Tensor,
    hess_B_center: torch.Tensor,
    omega_B: torch.Tensor,
    omega_grad_B: torch.Tensor,
    omega_hess_B: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Implement ``eq:initial-gradient-second-variation-appendix``.

    All matrix-valued tensors use axes ``[cell, j, ell]``. The returned pair
    contains the full ``rho^j_{ell,k}`` and its boundary-network contribution;
    their difference is the supplied data modulus ``omega_hess_g``.
    """
    abs_grad_B_plus_modulus = grad_B_center.abs() + omega_grad_B
    rho_bf = (
        (hess_B_center.abs() + omega_hess_B) * eta_F[:, None, None]
        + F_center.abs()[:, None, None] * omega_hess_B
        + abs_grad_B_plus_modulus[:, :, None] * eta_grad_F[:, None, :]
        + grad_F_center.abs()[:, None, :] * omega_grad_B[:, :, None]
        + abs_grad_B_plus_modulus[:, None, :] * eta_grad_F[:, :, None]
        + grad_F_center.abs()[:, :, None] * omega_grad_B[:, None, :]
        + (B_center.abs() + omega_B)[:, None, None] * eta_hess_F
        + hess_F_center.abs() * omega_B[:, None, None]
    )
    rho_bf = require_nonnegative(
        "initial_gradient_rho_boundary_network_j_ell",
        rho_bf,
    )
    rho = require_nonnegative(
        "initial_gradient_rho_j_ell",
        omega_hess_g + rho_bf,
    )
    return rho, rho_bf

def _displacement_gradient_taylor_cell_integrals(
    *,
    model: RigPINN_tanh,
    centers: torch.Tensor,
    spatial_eps: float,
    initial_grad: VectorTaylorFunctionCertificate,
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
    third_triples = tuple(
        (i, j, q)
        for i in range(d)
        for j in range(i, d)
        for q in range(j, d)
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
        multi_indices=second_pairs,
        spatial_dim=model.d,
    )
    third = compute_third_order_bounds(
        model,
        spacetime_centers,
        spatial_eps,
        n_taylor,
        multi_indices=third_triples,
        spatial_dim=model.d,
    )

    grad_g_center = _as_cell_matrix(
        "grad_g_center",
        initial_grad.value(centers),
        length=M,
        width=d,
        dtype=dtype,
        device=device,
    )
    hess_g_center = torch.as_tensor(
        initial_grad.jacobian(centers),
        dtype=dtype,
        device=device,
    )
    if hess_g_center.shape != (M, d, d):
        raise ValueError("initial_grad.jacobian must have shape [M, d, d].")
    omega_hess_g = torch.as_tensor(
        initial_grad.jacobian_variation_bound(centers, spatial_eps),
        dtype=dtype,
        device=device,
    )
    if omega_hess_g.shape != (M, d, d):
        raise ValueError(
            "initial_grad.jacobian_variation_bound must have shape [M, d, d]."
        )
    omega_hess_g = require_nonnegative("initial_gradient_omega_hess_g", omega_hess_g)

    zero_counts = (0,) * d
    F_center = as_cell_vector(
        "F_center",
        first.base_value,
        length=M,
        dtype=dtype,
        device=device,
    )
    grad_F_center = torch.empty(M, d, dtype=dtype, device=device)
    hess_F_center = torch.empty(M, d, d, dtype=dtype, device=device)
    eta_F = torch.zeros(M, dtype=dtype, device=device)
    eta_grad_F = torch.zeros(M, d, dtype=dtype, device=device)
    eta_hess_F = torch.zeros(M, d, d, dtype=dtype, device=device)

    for q in range(d):
        counts_q = tuple(int(axis == q) for axis in range(d))
        grad_F_center[:, q] = _network_derivative_center(
            counts=counts_q,
            first=first,
            second=second,
            third=third,
        )
        eta_F = eta_F + float(spatial_eps) * _network_derivative_sup(
            counts=counts_q,
            spatial_eps=spatial_eps,
            first=first,
            second=second,
            third=third,
        )

    for j in range(d):
        for q in range(d):
            counts_j_q = [0] * d
            counts_j_q[j] += 1
            counts_j_q[q] += 1
            eta_grad_F[:, j] = eta_grad_F[:, j] + float(
                spatial_eps
            ) * _network_derivative_sup(
                counts=tuple(counts_j_q),
                spatial_eps=spatial_eps,
                first=first,
                second=second,
                third=third,
            )
        for ell in range(d):
            counts_j_ell = [0] * d
            counts_j_ell[j] += 1
            counts_j_ell[ell] += 1
            counts_j_ell_tuple = tuple(counts_j_ell)
            hess_F_center[:, j, ell] = _network_derivative_center(
                counts=counts_j_ell_tuple,
                first=first,
                second=second,
                third=third,
            )
            for q in range(d):
                counts_j_ell_q = list(counts_j_ell_tuple)
                counts_j_ell_q[q] += 1
                eta_hess_F[:, j, ell] = eta_hess_F[:, j, ell] + float(
                    spatial_eps
                ) * _network_derivative_sup(
                    counts=tuple(counts_j_ell_q),
                    spatial_eps=spatial_eps,
                    first=first,
                    second=second,
                    third=third,
                )

    eta_F = require_nonnegative("initial_gradient_eta_F", eta_F)
    eta_grad_F = require_nonnegative("initial_gradient_eta_grad_F", eta_grad_F)
    eta_hess_F = require_nonnegative("initial_gradient_eta_hess_F", eta_hess_F)

    B_center = product_boundary_derivative_value(centers, zero_counts)
    omega_B = product_boundary_derivative_modulus(
        centers,
        spatial_eps,
        zero_counts,
    )
    grad_B_center = torch.empty(M, d, dtype=dtype, device=device)
    hess_B_center = torch.empty(M, d, d, dtype=dtype, device=device)
    omega_grad_B = torch.empty(M, d, dtype=dtype, device=device)
    omega_hess_B = torch.empty(M, d, d, dtype=dtype, device=device)
    for j in range(d):
        counts_j = tuple(int(axis == j) for axis in range(d))
        grad_B_center[:, j] = product_boundary_derivative_value(centers, counts_j)
        omega_grad_B[:, j] = product_boundary_derivative_modulus(
            centers,
            spatial_eps,
            counts_j,
        )
        for ell in range(d):
            counts_j_ell = [0] * d
            counts_j_ell[j] += 1
            counts_j_ell[ell] += 1
            counts_j_ell_tuple = tuple(counts_j_ell)
            hess_B_center[:, j, ell] = product_boundary_derivative_value(
                centers,
                counts_j_ell_tuple,
            )
            omega_hess_B[:, j, ell] = product_boundary_derivative_modulus(
                centers,
                spatial_eps,
                counts_j_ell_tuple,
            )

    approx_grad_center = (
        grad_B_center * F_center[:, None]
        + B_center[:, None] * grad_F_center
    )
    approx_hessian_center = (
        hess_B_center * F_center[:, None, None]
        + grad_B_center[:, :, None] * grad_F_center[:, None, :]
        + grad_B_center[:, None, :] * grad_F_center[:, :, None]
        + B_center[:, None, None] * hess_F_center
    )
    residual_center_components = grad_g_center - approx_grad_center
    residual_jacobian = hess_g_center - approx_hessian_center
    midpoint_residual = torch.linalg.vector_norm(residual_center_components, dim=-1)

    rho_j_ell, rho_boundary_network = _initial_gradient_second_variation_product_bound(
        omega_hess_g=omega_hess_g,
        F_center=F_center,
        grad_F_center=grad_F_center,
        hess_F_center=hess_F_center,
        eta_F=eta_F,
        eta_grad_F=eta_grad_F,
        eta_hess_F=eta_hess_F,
        B_center=B_center,
        grad_B_center=grad_B_center,
        hess_B_center=hess_B_center,
        omega_B=omega_B,
        omega_grad_B=omega_grad_B,
        omega_hess_B=omega_hess_B,
    )
    residual_component_hessian_abs: torch.Tensor | None = None
    if remainder_integration == "quadratic_moment":
        data_component_hessian_abs = torch.as_tensor(
            initial_grad.component_hessian_abs_bound(centers, spatial_eps),
            dtype=dtype,
            device=device,
        )
        if data_component_hessian_abs.shape != (M, d, d, d):
            raise ValueError(
                "initial_grad.component_hessian_abs_bound must have shape [M,d,d,d]"
            )
        data_component_hessian_abs = require_nonnegative(
            "initial_gradient_data_component_hessian_abs",
            data_component_hessian_abs,
        )
        approx_component_hessian_abs = torch.zeros(
            M, d, d, d, dtype=dtype, device=device
        )
        lower = centers - float(spatial_eps)
        upper = centers + float(spatial_eps)
        for j in range(d):
            for q in range(d):
                for ell in range(d):
                    counts = [0] * d
                    counts[j] += 1
                    counts[q] += 1
                    counts[ell] += 1
                    approx_component_hessian_abs[:, j, q, ell] = _bf_derivative_sup(
                        lower=lower,
                        upper=upper,
                        counts=tuple(counts),
                        spatial_eps=spatial_eps,
                        first=first,
                        second=second,
                        third=third,
                        input_dim=model.input_dim,
                    )
        residual_component_hessian_abs = require_nonnegative(
            "initial_gradient_residual_component_hessian_abs",
            data_component_hessian_abs + approx_component_hessian_abs,
        )
    delta_data = require_nonnegative(
        "initial_gradient_delta1_data_components",
        float(spatial_eps) * omega_hess_g.sum(dim=-1),
    )
    delta_boundary_network = require_nonnegative(
        "initial_gradient_delta1_boundary_network_components",
        float(spatial_eps) * rho_boundary_network.sum(dim=-1),
    )
    delta1_components = require_nonnegative(
        "initial_gradient_delta1_components",
        float(spatial_eps) * rho_j_ell.sum(dim=-1),
    )

    cell_volume = torch.as_tensor(
        (2.0 * float(spatial_eps)) ** d,
        dtype=dtype,
        device=device,
    )
    int_components = torch.zeros(M, d, dtype=dtype, device=device)
    for j in range(d):
        offset = residual_center_components[:, j]
        slope = residual_jacobian[:, j, :]
        int_p2 = cell_volume * (
            offset.square()
            + (float(spatial_eps) ** 2 / 3.0) * slope.square().sum(dim=-1)
        )
        if remainder_integration == "quadratic_moment":
            if residual_component_hessian_abs is None:
                raise RuntimeError("component Hessian bounds were not constructed")
            moments = quadratic_hessian_remainder_moment_integral(
                residual_center=offset,
                residual_gradient=slope,
                hessian_abs_bounds=residual_component_hessian_abs[:, j, :, :],
                spatial_eps=spatial_eps,
                cell_volume=cell_volume,
            )
            int_components[:, j] = moments["selected"]
            continue
        if affine_abs_integral == "cauchy_schwarz":
            int_abs_p = torch.sqrt((cell_volume * int_p2).clamp_min(0.0))
        elif affine_abs_integral == "exact":
            int_abs_p = _centered_affine_abs_integral(
                offset,
                slope,
                float(spatial_eps),
            )
        else:
            raise ValueError(
                "affine_abs_integral must be 'cauchy_schwarz' or 'exact'."
            )
        int_components[:, j] = require_nonnegative(
            "initial_gradient_taylor_component_integral",
            int_p2
            + 2.0 * delta1_components[:, j] * int_abs_p
            + cell_volume * delta1_components[:, j].square(),
        )
    cell_l2_squared_bounds = require_nonnegative(
        "initial_gradient_taylor_cell_integral",
        int_components.sum(dim=-1),
    )
    diagnostics = {
        "phi_center_norm_bounds": midpoint_residual,
        "eta_phi_components": delta1_components,
        "eta_grad_v": delta_boundary_network,
        "eta_grad_g": delta_data,
        "eta_F": eta_F,
        "eta_B": omega_B,
        "eta_grad_B": omega_grad_B,
        "F_center": F_center,
        "B_center": B_center,
        "grad_B_center": grad_B_center,
        "grad_g_center": grad_g_center,
        "approx_grad_center": approx_grad_center,
        "initial_gradient_rho_j_ell": rho_j_ell,
        "initial_gradient_rho_boundary_network_j_ell": rho_boundary_network,
        "initial_gradient_omega_hess_g": omega_hess_g,
        "initial_gradient_eta_grad_F": eta_grad_F,
        "initial_gradient_eta_hess_F": eta_hess_F,
        "initial_gradient_hess_g_center": hess_g_center,
        "initial_gradient_hess_F_center": hess_F_center,
        "initial_gradient_hess_B_center": hess_B_center,
        "initial_gradient_omega_hess_B": omega_hess_B,
        "initial_gradient_approx_hessian_center": approx_hessian_center,
        "initial_gradient_residual_jacobian_center": residual_jacobian,
        "initial_gradient_delta1_components": delta1_components,
        **(
            {
                "initial_gradient_residual_component_hessian_abs": (
                    residual_component_hessian_abs
                )
            }
            if residual_component_hessian_abs is not None
            else {}
        ),
    }
    return cell_l2_squared_bounds, midpoint_residual, diagnostics

def bound_initial_displacement_gradient_residual_l2_taylor_uniform(
    *,
    model: RigPINN_tanh,
    grid: UniformBoxGrid | UniformUnitBoxGridSpec,
    initial_grad: VectorTaylorFunctionCertificate,
    n_taylor: int | None = None,
    safety_factor: float = 1.0,
    rounding_slack: float = 0.0,
    batch_size: int | None = None,
    store_diagnostics: bool = True,
    affine_abs_integral: str = "exact",
    progress_callback=None,
    remainder_integration: str = "corner",
) -> InitialGradientResidualL2BoundResult:
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
            cell_l2_squared_bounds, midpoint_residual, diagnostics = _displacement_gradient_taylor_cell_integrals(
                model=model,
                centers=centers,
                spatial_eps=spatial_eps,
                initial_grad=initial_grad,
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
    notes = [
        "q1_affine_component_cell",
        "hessian_variation_center_plus_product_bound",
        "network_spatial_derivative_bounds_through_order_3",
        "eq:initial-gradient-cell-model-errors",
        "eq:initial-gradient-second-variation-appendix",
    ]
    if remainder_integration == "quadratic_moment":
        notes.extend(
            (
                "ima_section2_componentwise_quadratic_hessian_remainder_moments",
                "eq:appendix-initial-gradient-second-coefficients",
            )
        )
    generic_result = accumulator.result(
        name="initial_displacement_gradient",
        notes=tuple(notes),
    )

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

def bound_initial_displacement_gradient_residual_l2_moment_taylor_uniform(
    *,
    model: RigPINN_tanh,
    grid: UniformBoxGrid | UniformUnitBoxGridSpec,
    initial_grad: VectorTaylorFunctionCertificate,
    n_taylor: int | None = None,
    safety_factor: float = 1.0,
    rounding_slack: float = 0.0,
    batch_size: int | None = None,
    store_diagnostics: bool = True,
    progress_callback=None,
) -> InitialGradientResidualL2BoundResult:
    """Literal componentwise Section 2/4 gradient moment certificate."""

    return bound_initial_displacement_gradient_residual_l2_taylor_uniform(
        model=model,
        grid=grid,
        initial_grad=initial_grad,
        n_taylor=n_taylor,
        safety_factor=safety_factor,
        rounding_slack=rounding_slack,
        batch_size=batch_size,
        store_diagnostics=store_diagnostics,
        affine_abs_integral="cauchy_schwarz",
        progress_callback=progress_callback,
        remainder_integration="quadratic_moment",
    )
