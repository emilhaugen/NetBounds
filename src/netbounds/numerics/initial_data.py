"""Reached paper initial data from producer revisions 2add560 and 1003b544.

The separable datum is $g(x)=\\prod_{j=1}^d \\sin(\\pi x_j)$ with the analytic
gradient, Hessian, and envelope bounds recorded by the producers for every
spatial dimension.  The wave initial velocity datum is identically zero.
Each constructor takes the spatial dimension; the default ``d=1`` preserves
the original one-dimensional behavior exactly.
"""

from __future__ import annotations

import math

import torch

from . import initial_gradient_q0, initial_gradient_q1, initial_q1
from ._initial_support.modulus_data import LipschitzFunction


def sine_value(x: torch.Tensor) -> torch.Tensor:
    r"""Return $g(x)=\prod_j\sin(\pi x_j)$ on batched inputs of shape [M, d]."""

    return torch.sin(math.pi * x).prod(dim=-1)


def sine_gradient(x: torch.Tensor) -> torch.Tensor:
    r"""Analytic gradient of $\prod_j\sin(\pi x_j)$, shape [M, d]."""

    d = x.shape[1]
    factors = torch.sin(math.pi * x)
    components = []
    for i in range(d):
        if d == 1:
            prod_other = torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
        else:
            mask = torch.ones(d, dtype=torch.bool, device=x.device)
            mask[i] = False
            prod_other = factors[:, mask].prod(dim=-1)
        components.append(math.pi * torch.cos(math.pi * x[:, i]) * prod_other)
    return torch.stack(components, dim=-1)


def sine_hessian(x: torch.Tensor) -> torch.Tensor:
    """Exact Hessian of the separable sine, axes [M, j, ell]."""

    d = x.shape[1]
    values = torch.empty(x.shape[0], d, d, dtype=x.dtype, device=x.device)
    for j in range(d):
        for ell in range(d):
            counts = [0] * d
            counts[j] += 1
            counts[ell] += 1
            factors = [
                (math.pi**count)
                * torch.sin(math.pi * x[:, axis] + 0.5 * math.pi * count)
                for axis, count in enumerate(counts)
            ]
            value = factors[0]
            for factor in factors[1:]:
                value = value * factor
            values[:, j, ell] = value
    return values


def sine_hessian_l1_bound(x: torch.Tensor, radius: float) -> torch.Tensor:
    r"""$\sum_{ij}\sup_C|\partial_{ij}g| = d^2\pi^2$, shape [M]."""

    del radius
    return torch.full(
        (x.shape[0],),
        (x.shape[1] ** 2) * math.pi**2,
        dtype=x.dtype,
        device=x.device,
    )


def sine_hessian_abs_bound(x: torch.Tensor, radius: float) -> torch.Tensor:
    r"""Entrywise Hessian envelope $\mathsf G^{e_q+e_\ell}=\pi^2$."""

    del radius
    d = x.shape[1]
    return torch.full(
        (x.shape[0], d, d),
        math.pi**2,
        dtype=x.dtype,
        device=x.device,
    )


def sine_gradient_third_abs_bound(x: torch.Tensor, radius: float) -> torch.Tensor:
    r"""Entrywise third-derivative envelope $\mathsf G^{e_j+e_q+e_\ell}=\pi^3$."""

    del radius
    d = x.shape[1]
    return torch.full(
        (x.shape[0], d, d, d),
        math.pi**3,
        dtype=x.dtype,
        device=x.device,
    )


def sine_gradient_jacobian_modulus(x: torch.Tensor, radius: float) -> torch.Tensor:
    r"""Literal displayed modulus $\min(2\pi^2, d\pi^3 r)$, shape [M, d, d]."""

    d = x.shape[1]
    value = min(2.0 * math.pi**2, d * math.pi**3 * float(radius))
    return torch.full(
        (x.shape[0], d, d),
        value,
        dtype=x.dtype,
        device=x.device,
    )


def zero_value(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


def zero_gradient(x: torch.Tensor) -> torch.Tensor:
    return torch.zeros(x.shape[0], x.shape[1], dtype=x.dtype, device=x.device)


def displacement_q0(d: int = 1) -> LipschitzFunction:
    del d
    return LipschitzFunction(sine_value, L=math.pi)


def displacement_q1(d: int = 1) -> initial_q1.TaylorInitialDatum:
    return initial_q1.TaylorInitialDatum(
        value_fn=sine_value,
        gradient_fn=sine_gradient,
        hessian_l1_bound_value=sine_hessian_l1_bound,
        hessian_abs_bound_value=sine_hessian_abs_bound,
    )


def displacement_gradient_q0(d: int = 1) -> initial_gradient_q0.VectorLipschitzInitialDatum:
    return initial_gradient_q0.VectorLipschitzInitialDatum(
        value_fn=sine_gradient,
        component_lipschitz_l1=math.pi**2,
    )


def displacement_gradient_q1(d: int = 1) -> initial_gradient_q1.VectorTaylorInitialDatum:
    return initial_gradient_q1.VectorTaylorInitialDatum(
        value_fn=sine_gradient,
        jacobian_fn=sine_hessian,
        component_hessian_l1_bound=float(d) ** 2 * math.pi**3,
        jacobian_variation_bound_value=sine_gradient_jacobian_modulus,
        component_hessian_abs_bound_value=sine_gradient_third_abs_bound,
    )


def velocity_q0(d: int = 1) -> LipschitzFunction:
    del d
    return LipschitzFunction(zero_value, L=0.0)


def velocity_q1(d: int = 1) -> initial_q1.TaylorInitialDatum:
    del d
    return initial_q1.TaylorInitialDatum(
        value_fn=zero_value,
        gradient_fn=zero_gradient,
        hessian_l1_bound_value=0.0,
        hessian_abs_bound_value=0.0,
    )
