"""Private analytic helpers reached by the 1D initial-data kernels."""

from __future__ import annotations

import itertools

import math

import torch

from .quadrature import require_nonnegative

def _centered_affine_abs_integral(
    offset: torch.Tensor,
    slope: torch.Tensor,
    eps: float,
    *,
    zero_tol: float = 1e-30,
) -> torch.Tensor:
    """Exact ``int_{[-eps,eps]^n} |offset + slope dot z| dz``.

    Uses the truncated-power formula for the positive part of an affine
    function over an axis-aligned box, then ``|p| = 2 p_+ - p``. Axes with
    zero slope are integrated out exactly, which also avoids division by zero.
    The implementation is vectorized by zero-slope mask; dimensions here are
    the small space-time dimensions used by the PDE certificate.
    """
    if offset.ndim != 1:
        raise ValueError("offset must have shape [M].")
    if slope.ndim != 2 or slope.shape[0] != offset.shape[0]:
        raise ValueError("slope must have shape [M, n].")
    if eps < 0.0 or not math.isfinite(float(eps)):
        raise ValueError("eps must be finite and nonnegative.")

    M, n = slope.shape
    dtype = offset.dtype
    device = offset.device
    out = torch.empty(M, dtype=dtype, device=device)
    if M == 0:
        return out
    if eps == 0.0:
        return offset.abs()

    active = slope.abs() > float(zero_tol)
    full_volume = (2.0 * float(eps)) ** n

    for mask_bits in itertools.product((False, True), repeat=n):
        mask_tensor = torch.as_tensor(mask_bits, dtype=torch.bool, device=device)
        rows = (active == mask_tensor).all(dim=1)
        if not bool(rows.any()):
            continue
        active_indices = [i for i, bit in enumerate(mask_bits) if bit]
        k = len(active_indices)
        row_offset = offset[rows]
        if k == 0:
            out[rows] = full_volume * row_offset.abs()
            continue

        row_slope = slope[rows][:, active_indices]
        inactive_factor = (2.0 * float(eps)) ** (n - k)
        positive_integral = torch.zeros_like(row_offset)
        for corner in itertools.product((-float(eps), float(eps)), repeat=k):
            corner_tensor = torch.as_tensor(corner, dtype=dtype, device=device)
            value = row_offset + (row_slope * corner_tensor).sum(dim=-1)
            upper_count = sum(1 for value_ in corner if value_ > 0.0)
            sign = -1.0 if ((k - upper_count) % 2) else 1.0
            positive_integral = positive_integral + sign * value.clamp_min(0.0).pow(k + 1)

        denom = math.factorial(k + 1) * row_slope.prod(dim=-1)
        positive_integral = inactive_factor * positive_integral / denom
        signed_integral = full_volume * row_offset
        abs_integral = 2.0 * positive_integral - signed_integral
        out[rows] = abs_integral.clamp_min(0.0)

    return out

def _sub_counts(counts: tuple[int, ...]):
    return itertools.product(*(range(count + 1) for count in counts))

def _multi_choose(alpha: tuple[int, ...], beta: tuple[int, ...]) -> int:
    coefficient = 1
    for a, b in zip(alpha, beta):
        coefficient *= math.comb(a, b)
    return coefficient

def _boundary_derivative_center(
    spatial: torch.Tensor,
    counts: tuple[int, ...],
) -> torch.Tensor:
    d = spatial.shape[1]
    if any(counts[d:]):
        return torch.zeros(spatial.shape[0], dtype=spatial.dtype, device=spatial.device)
    factors = []
    for axis, count in enumerate(counts[:d]):
        x = spatial[:, axis]
        if count == 0:
            factors.append(x * (1.0 - x))
        elif count == 1:
            factors.append(1.0 - 2.0 * x)
        elif count == 2:
            factors.append(torch.full_like(x, -2.0))
        else:
            return torch.zeros(spatial.shape[0], dtype=spatial.dtype, device=spatial.device)
    out = factors[0]
    for factor in factors[1:]:
        out = out * factor
    return out

def _boundary_derivative_sup(
    lower: torch.Tensor,
    upper: torch.Tensor,
    counts: tuple[int, ...],
) -> torch.Tensor:
    d = lower.shape[1]
    if any(counts[d:]):
        return torch.zeros(lower.shape[0], dtype=lower.dtype, device=lower.device)
    factors = []
    for axis, count in enumerate(counts[:d]):
        if count == 0:
            factors.append(_s_sup_interval(lower[:, axis], upper[:, axis]))
        elif count == 1:
            factors.append(
                torch.maximum(
                    (1.0 - 2.0 * lower[:, axis]).abs(),
                    (1.0 - 2.0 * upper[:, axis]).abs(),
                )
            )
        elif count == 2:
            factors.append(torch.full_like(lower[:, axis], 2.0))
        else:
            return torch.zeros(lower.shape[0], dtype=lower.dtype, device=lower.device)
    out = factors[0]
    for factor in factors[1:]:
        out = out * factor
    return require_nonnegative("boundary_derivative_sup", out)

def _s_sup_interval(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    lower_value = lower * (1.0 - lower)
    upper_value = upper * (1.0 - upper)
    return torch.where(
        (lower <= 0.5) & (0.5 <= upper),
        torch.full_like(lower, 0.25),
        torch.maximum(lower_value, upper_value),
    )
