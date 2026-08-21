"""Analytic helpers reached by the 1D PDE Q1 trace at producer revision 53bb374."""

from __future__ import annotations

import itertools

import torch

from ._initial_support.quadrature import require_nonnegative

def _product_interval(lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if lower.ndim != 2 or upper.shape != lower.shape:
        raise ValueError("interval endpoints must have shape [M, K].")
    if lower.shape[1] == 0:
        ones = torch.ones(lower.shape[0], dtype=lower.dtype, device=lower.device)
        return ones, ones

    products = []
    for choice in itertools.product((0, 1), repeat=lower.shape[1]):
        factors = [upper[:, j] if bit else lower[:, j] for j, bit in enumerate(choice)]
        products.append(torch.stack(factors, dim=-1).prod(dim=-1))
    stacked = torch.stack(products, dim=0)
    return stacked.min(dim=0).values, stacked.max(dim=0).values

def _s_interval(
    x: torch.Tensor,
    eps: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower = x - eps
    upper = x + eps
    s_lower = lower * (1.0 - lower)
    s_upper = upper * (1.0 - upper)
    crosses_half = (lower <= 0.5) & (0.5 <= upper)
    s_max = torch.where(
        crosses_half,
        torch.full_like(s_upper, 0.25),
        torch.maximum(s_lower, s_upper),
    )
    return torch.minimum(s_lower, s_upper), s_max

def _d_interval(
    x: torch.Tensor,
    eps: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower = x - eps
    upper = x + eps
    return 1.0 - 2.0 * upper, 1.0 - 2.0 * lower

def _interval_variation(
    center: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    return require_nonnegative(
        "interval_variation",
        torch.maximum((center - lower).abs(), (upper - center).abs()),
    )

def _boundary_center_and_variations(
    y: torch.Tensor,
    eps: float | torch.Tensor,
    d: int,
) -> dict[str, torch.Tensor | list[torch.Tensor]]:
    x = y[:, :d]
    eps_tensor = torch.as_tensor(eps, dtype=x.dtype, device=x.device)
    if eps_tensor.ndim == 0:
        eps_tensor = eps_tensor.repeat(d)
    if eps_tensor.shape != (d,):
        raise ValueError("spatial radius must be scalar or have shape (d,)")
    s = x * (1.0 - x)
    dx = 1.0 - 2.0 * x
    s_lower, s_upper = _s_interval(x, eps_tensor)
    d_lower, d_upper = _d_interval(x, eps_tensor)

    B = s.prod(dim=-1)
    B_lower, B_upper = _product_interval(s_lower, s_upper)
    eta_B = _interval_variation(B, B_lower, B_upper)

    B_i = []
    eta_B_i = []
    B_ii = []
    eta_B_ii = []
    for i in range(d):
        others = [j for j in range(d) if j != i]
        prod_others = s[:, others].prod(dim=-1) if others else torch.ones_like(B)

        Bi = dx[:, i] * prod_others
        Bi_factor_lower = torch.cat(
            [d_lower[:, i : i + 1], s_lower[:, others]],
            dim=1,
        )
        Bi_factor_upper = torch.cat(
            [d_upper[:, i : i + 1], s_upper[:, others]],
            dim=1,
        )
        Bi_lower, Bi_upper = _product_interval(Bi_factor_lower, Bi_factor_upper)

        Bii = -2.0 * prod_others
        prod_lower, prod_upper = _product_interval(s_lower[:, others], s_upper[:, others])
        Bii_lower = -2.0 * prod_upper
        Bii_upper = -2.0 * prod_lower

        B_i.append(Bi)
        eta_B_i.append(_interval_variation(Bi, Bi_lower, Bi_upper))
        B_ii.append(Bii)
        eta_B_ii.append(_interval_variation(Bii, Bii_lower, Bii_upper))

    return {
        "B": B,
        "B_i": B_i,
        "B_ii": B_ii,
        "eta_B": eta_B,
        "eta_B_i": eta_B_i,
        "eta_B_ii": eta_B_ii,
    }

def _product_variation(
    a: torch.Tensor,
    eta_a: torch.Tensor,
    b: torch.Tensor,
    eta_b: torch.Tensor,
) -> torch.Tensor:
    return require_nonnegative(
        "product_variation",
        a.abs() * eta_b + b.abs() * eta_a + eta_a * eta_b,
    )

def _column(
    values: torch.Tensor,
    labels: tuple[tuple[int, ...], ...],
    key: tuple[int, ...],
) -> torch.Tensor:
    return values[:, labels.index(tuple(sorted(key)))]

def _add_counts(base: tuple[int, ...], indices: tuple[int, ...]) -> tuple[int, ...]:
    values = list(base)
    for index in indices:
        values[index] += 1
    return tuple(values)

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
