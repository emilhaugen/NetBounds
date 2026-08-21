"""Analytic centered-box moment corrections for certified Taylor cells."""

from __future__ import annotations

import math

import torch

from .quadrature import require_nonnegative


def linear_gradient_remainder_moment_integral(
    *,
    residual_center: torch.Tensor,
    first_derivative_abs_bounds: torch.Tensor,
    spatial_eps: float,
    cell_volume: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Integrate a constant-Q0 cell plus its linear derivative envelope.

    If ``|e(c+z)| <= |a| + sum_q b_q |z_q|`` on the centered box
    ``[-eps, eps]^d``, this returns each term in the integrated square and
    their sum. Thus ``M_e_q=V eps/2``, ``M_2e_q=V eps^2/3``, and
    ``M_e_q+e_l=V eps^2/4`` for ``q != l``.
    """

    if residual_center.ndim != 1:
        raise ValueError("residual_center must have shape [M]")
    if first_derivative_abs_bounds.ndim != 2:
        raise ValueError("first_derivative_abs_bounds must have shape [M,d]")
    cells, dimension = first_derivative_abs_bounds.shape
    if residual_center.shape != (cells,):
        raise ValueError("residual center and derivative-bound cell counts differ")
    if dimension <= 0:
        raise ValueError("the spatial dimension must be positive")
    if not math.isfinite(float(spatial_eps)) or spatial_eps <= 0.0:
        raise ValueError("spatial_eps must be finite and positive")
    bounds = require_nonnegative(
        "linear_moment_first_derivative_abs_bounds", first_derivative_abs_bounds
    )
    volume = torch.as_tensor(
        cell_volume,
        dtype=residual_center.dtype,
        device=residual_center.device,
    )
    if volume.numel() != 1 or not bool(torch.isfinite(volume).all()) or not bool((volume > 0).all()):
        raise ValueError("cell_volume must be a finite positive scalar")

    eps = float(spatial_eps)
    int_p2 = require_nonnegative(
        "linear_moment_constant_l2_squared", volume * residual_center.square()
    )
    cross_correction = require_nonnegative(
        "linear_moment_twice_constant_remainder_cross",
        2.0 * residual_center.abs() * volume * (eps / 2.0) * bounds.sum(dim=-1),
    )
    normalized_second_moments = torch.full(
        (dimension, dimension),
        eps**2 / 4.0,
        dtype=residual_center.dtype,
        device=residual_center.device,
    )
    normalized_second_moments.diagonal().fill_(eps**2 / 3.0)
    remainder_l2_squared = require_nonnegative(
        "linear_moment_remainder_l2_squared",
        volume
        * torch.einsum(
            "mq,ml,ql->m", bounds, bounds, normalized_second_moments
        ),
    )
    selected = require_nonnegative(
        "linear_moment_cell_l2_squared",
        int_p2 + cross_correction + remainder_l2_squared,
    )
    return {
        "int_p2": int_p2,
        "cross_correction": cross_correction,
        "remainder_l2_squared": remainder_l2_squared,
        "selected": selected,
    }


def _centered_isotropic_moment(
    *,
    dimension: int,
    spatial_eps: float,
    cell_volume: torch.Tensor,
    indices: tuple[int, ...],
) -> torch.Tensor:
    """Return the centered-box moment M_alpha for unit coordinate indices."""

    counts = [0] * dimension
    for index in indices:
        if index < 0 or index >= dimension:
            raise ValueError("moment index is outside the spatial dimension")
        counts[index] += 1
    denominator = 1
    for count in counts:
        denominator *= count + 1
    return cell_volume * float(spatial_eps) ** len(indices) / float(denominator)


def quadratic_hessian_remainder_moment_integral(
    *,
    residual_center: torch.Tensor,
    residual_gradient: torch.Tensor,
    hessian_abs_bounds: torch.Tensor,
    spatial_eps: float,
    cell_volume: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Evaluate the literal affine-Q1 moment bound from IMA Section 2.

    On a centered spatial cell, let

    ``P(z) = residual_center + residual_gradient . z``

    and

    ``D(z) = 1/2 sum_(q,l) hessian_abs_bounds[q,l] |z_q z_l|``.

    This returns ``int P^2``, ``2 int A D``, ``int D^2``, and their sum,
    where ``A=|residual_center|+sum_i |residual_gradient_i||z_i|``.  Every
    product is integrated analytically using centered rectangular moments.
    No cancellation-prone affine-absolute corner formula is used.
    """

    if residual_center.ndim != 1:
        raise ValueError("residual_center must have shape [M]")
    if residual_gradient.ndim != 2:
        raise ValueError("residual_gradient must have shape [M,d]")
    if hessian_abs_bounds.ndim != 3:
        raise ValueError("hessian_abs_bounds must have shape [M,d,d]")
    cells, dimension = residual_gradient.shape
    if residual_center.shape != (cells,):
        raise ValueError("residual center and gradient cell counts differ")
    if hessian_abs_bounds.shape != (cells, dimension, dimension):
        raise ValueError("hessian_abs_bounds must match [M,d,d]")
    if not math.isfinite(float(spatial_eps)) or spatial_eps <= 0.0:
        raise ValueError("spatial_eps must be finite and positive")
    hessian_abs_bounds = require_nonnegative(
        "quadratic_moment_hessian_abs_bounds", hessian_abs_bounds
    )

    eps = float(spatial_eps)
    int_p2 = cell_volume * (
        residual_center.square()
        + (eps**2 / 3.0) * residual_gradient.square().sum(dim=-1)
    )
    int_p2 = require_nonnegative("quadratic_moment_affine_l2_squared", int_p2)

    m2 = torch.empty(
        dimension, dimension, dtype=residual_center.dtype, device=residual_center.device
    )
    m3 = torch.empty(
        dimension,
        dimension,
        dimension,
        dtype=residual_center.dtype,
        device=residual_center.device,
    )
    m4 = torch.empty(
        dimension,
        dimension,
        dimension,
        dimension,
        dtype=residual_center.dtype,
        device=residual_center.device,
    )
    for q in range(dimension):
        for ell in range(dimension):
            m2[q, ell] = _centered_isotropic_moment(
                dimension=dimension,
                spatial_eps=eps,
                cell_volume=cell_volume,
                indices=(q, ell),
            )
            for i in range(dimension):
                m3[q, ell, i] = _centered_isotropic_moment(
                    dimension=dimension,
                    spatial_eps=eps,
                    cell_volume=cell_volume,
                    indices=(q, ell, i),
                )
                for m in range(dimension):
                    for n in range(dimension):
                        m4[q, ell, m, n] = _centered_isotropic_moment(
                            dimension=dimension,
                            spatial_eps=eps,
                            cell_volume=cell_volume,
                            indices=(q, ell, m, n),
                        )

    affine_envelope_moments = (
        residual_center.abs()[:, None, None] * m2[None, :, :]
        + torch.einsum("mi,qli->mql", residual_gradient.abs(), m3)
    )
    cross_correction = torch.einsum(
        "mql,mql->m", hessian_abs_bounds, affine_envelope_moments
    )
    cross_correction = require_nonnegative(
        "quadratic_moment_twice_affine_remainder_cross", cross_correction
    )
    remainder_l2_squared = 0.25 * torch.einsum(
        "mql,mun,qlun->m", hessian_abs_bounds, hessian_abs_bounds, m4
    )
    remainder_l2_squared = require_nonnegative(
        "quadratic_moment_remainder_l2_squared", remainder_l2_squared
    )
    selected = require_nonnegative(
        "quadratic_moment_cell_l2_squared",
        int_p2 + cross_correction + remainder_l2_squared,
    )
    return {
        "int_p2": int_p2,
        "cross_correction": cross_correction,
        "remainder_l2_squared": remainder_l2_squared,
        "selected": selected,
    }
