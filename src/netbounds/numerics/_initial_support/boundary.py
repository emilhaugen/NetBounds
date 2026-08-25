"""Boundary-factor utilities for residual bounds."""

from __future__ import annotations

import math

import torch

from ..model import SpatialBoundaryFunction
from .quadrature import require_nonnegative


def _containment_tol(dtype: torch.dtype, base_tol: float) -> float:
    if torch.empty((), dtype=dtype).is_floating_point():
        return max(float(base_tol), 8.0 * torch.finfo(dtype).eps)
    return float(base_tol)


def require_spatial_product_boundary(boundary_function: object, d: int) -> None:
    if not isinstance(boundary_function, SpatialBoundaryFunction):
        raise ValueError(
            "This bound is certified only for SpatialBoundaryFunction.",
        )
    if boundary_function.d != d:
        raise ValueError("boundary_function.d must match model.d.")


def product_boundary_value(centers: torch.Tensor) -> torch.Tensor:
    if centers.ndim != 2:
        raise ValueError("centers must have shape [M, d].")
    if not centers.is_floating_point():
        raise TypeError("centers must have a floating-point dtype.")
    if not torch.isfinite(centers).all():
        raise ValueError("centers must be finite.")
    return (centers * (1.0 - centers)).prod(dim=-1)


def product_boundary_derivative_value(
    centers: torch.Tensor,
    counts: tuple[int, ...],
) -> torch.Tensor:
    """Evaluate ``partial**counts B`` for ``B=prod_i x_i(1-x_i)``.

    The multi-index contains spatial derivative counts and must have one entry
    per spatial coordinate. This is the point-value part of
    ``eq:explicit-B-moduli`` in the IMA manuscript.
    """
    if centers.ndim != 2:
        raise ValueError("centers must have shape [M, d].")
    if len(counts) != centers.shape[1] or any(count < 0 for count in counts):
        raise ValueError("counts must be a nonnegative spatial multi-index of length d.")
    if not centers.is_floating_point():
        raise TypeError("centers must have a floating-point dtype.")
    if not torch.isfinite(centers).all():
        raise ValueError("centers must be finite.")

    result = torch.ones(centers.shape[0], dtype=centers.dtype, device=centers.device)
    for axis, count in enumerate(counts):
        if count == 0:
            factor = centers[:, axis] * (1.0 - centers[:, axis])
        elif count == 1:
            factor = 1.0 - 2.0 * centers[:, axis]
        elif count == 2:
            factor = torch.full_like(centers[:, axis], -2.0)
        else:
            factor = torch.zeros_like(centers[:, axis])
        result = result * factor
    return result


def product_boundary_derivative_modulus(
    centers: torch.Tensor,
    spatial_eps: float,
    counts: tuple[int, ...],
) -> torch.Tensor:
    """Return the explicit global modulus ``omega_B**counts(spatial_eps)``.

    This implements exactly ``eq:explicit-B-moduli``:
    ``r * sum_q prod_i b[counts_i + 1_{i=q}]``, where
    ``b_0=1/4``, ``b_1=1``, ``b_2=2``, and ``b_m=0`` for ``m>=3``.
    The value is repeated over cells for componentwise tensor algebra.
    """
    if centers.ndim != 2:
        raise ValueError("centers must have shape [M, d].")
    if len(counts) != centers.shape[1] or any(count < 0 for count in counts):
        raise ValueError("counts must be a nonnegative spatial multi-index of length d.")
    if not centers.is_floating_point():
        raise TypeError("centers must have a floating-point dtype.")
    if not torch.isfinite(centers).all():
        raise ValueError("centers must be finite.")
    if not math.isfinite(float(spatial_eps)) or spatial_eps <= 0:
        raise ValueError("spatial_eps must be positive and finite.")

    b = (0.25, 1.0, 2.0)
    coefficient = 0.0
    for differentiated_axis in range(centers.shape[1]):
        term = 1.0
        for axis, count in enumerate(counts):
            order = count + int(axis == differentiated_axis)
            term *= b[order] if order < len(b) else 0.0
        coefficient += term
    return require_nonnegative(
        "product_boundary_derivative_modulus",
        torch.full(
            (centers.shape[0],),
            float(spatial_eps) * coefficient,
            dtype=centers.dtype,
            device=centers.device,
        ),
    )


def product_boundary_value_and_variation(
    centers: torch.Tensor,
    spatial_eps: float,
    *,
    tol: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if centers.ndim != 2:
        raise ValueError("centers must have shape [M, d].")
    if not centers.is_floating_point():
        raise TypeError("centers must have a floating-point dtype.")
    if not math.isfinite(float(spatial_eps)) or spatial_eps <= 0:
        raise ValueError("spatial_eps must be positive and finite.")
    if not torch.isfinite(centers).all():
        raise ValueError("centers must be finite.")

    effective_tol = _containment_tol(centers.dtype, tol)
    lower = centers - spatial_eps
    upper = centers + spatial_eps
    if (lower < -effective_tol).any() or (upper > 1.0 + effective_tol).any():
        raise ValueError("Uniform boxes must be contained in [0,1]^d.")

    s_lower = lower * (1.0 - lower)
    s_upper = upper * (1.0 - upper)
    s_min = torch.minimum(s_lower, s_upper)
    crosses_half = (lower <= 0.5) & (0.5 <= upper)
    s_max_endpoint = torch.maximum(s_lower, s_upper)
    s_max = torch.where(
        crosses_half,
        torch.full_like(s_max_endpoint, 0.25),
        s_max_endpoint,
    )

    B_lower = s_min.prod(dim=-1)
    B_upper = s_max.prod(dim=-1)
    B_center = product_boundary_value(centers)
    eta_B = torch.maximum(
        (B_center - B_lower).abs(),
        (B_upper - B_center).abs(),
    )
    return B_center, require_nonnegative("eta_B", eta_B)


boundary_product_center_and_variation_uniform = product_boundary_value_and_variation
