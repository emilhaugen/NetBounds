"""Shared layer-loop helpers for derivative-bound computations."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def normalize_input_batch(model, y: torch.Tensor) -> torch.Tensor:
    """Return ``y`` as a batched tensor on the model's dtype and device."""
    if y.ndim == 1:
        y = y.unsqueeze(0)
    return y.to(dtype=model.input_layer.weight.dtype, device=model.input_layer.weight.device)


def hidden_preactivations_and_m(
    model,
    y: torch.Tensor,
    *,
    orders: Iterable[int] = (1,),
) -> tuple[list[torch.Tensor], dict[int, list[torch.Tensor]]]:
    """Return hidden preactivations and activation derivatives by order."""
    order_tuple = tuple(orders)
    if not order_tuple:
        raise ValueError("orders must not be empty.")

    z_values = []
    m_by_order: dict[int, list[torch.Tensor]] = {order: [] for order in order_tuple}
    a = y
    for layer in model.hidden_layers:
        z = layer(a)
        z_values.append(z)
        for order in order_tuple:
            m_by_order[order].append(model.AB.activation_m_prime_stable(z, order))
        a = model.activation(z)
    return z_values, m_by_order


def preactivation_delta(
    radius_components: torch.Tensor,
    eps: float | torch.Tensor,
    *,
    spatial_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert coordinate radius components into activation argument radii.

    ``eps`` may be a scalar half-width or a per-coordinate half-width vector.
    The vector case certifies rectangular boxes by using
    ``sum_q eps_q |partial_q z|`` instead of a single isotropic radius.
    """
    components = (
        radius_components
        if spatial_dim is None
        else radius_components[..., :spatial_dim]
    )
    E = components.sum(dim=-1)
    eps_tensor = torch.as_tensor(
        eps,
        dtype=radius_components.dtype,
        device=radius_components.device,
    )
    if eps_tensor.ndim == 0:
        return E, eps_tensor * E
    if eps_tensor.ndim != 1:
        raise ValueError("eps must be scalar or a one-dimensional tensor.")
    if spatial_dim is not None:
        eps_tensor = eps_tensor[:spatial_dim]
    if eps_tensor.shape != (components.shape[-1],):
        raise ValueError("eps vector length must match the number of active coordinates.")
    if (eps_tensor < 0).any():
        raise ValueError("eps must be nonnegative.")
    delta = (components * eps_tensor).sum(dim=-1)
    return E, delta


def activation_derivative_bounds(
    model,
    z: torch.Tensor,
    delta: torch.Tensor,
    *,
    orders: Iterable[int],
    n_taylor: int | None,
) -> torch.Tensor:
    """Return stacked activation variation bounds for the requested orders.

    The activation-bound lemma uses the same centre ``z`` and radius ``delta``
    for every requested derivative order.  This routine therefore computes the
    required exact tanh derivatives once and reuses them to assemble all
    requested ``Q`` vectors.
    """
    order_tuple = tuple(orders)
    if not order_tuple:
        raise ValueError("orders must not be empty.")
    for order in order_tuple:
        if order not in model.AB.M_RANGE:
            raise ValueError(f"order must be in {model.AB.M_RANGE}, got {order}.")
    if delta.shape != z.shape:
        raise ValueError(f"delta shape {delta.shape} must match z shape {z.shape}.")
    if (delta < 0).any():
        raise ValueError("delta must be non-negative everywhere.")

    z = _as_activation_compute_dtype(model, z)
    delta = _as_activation_compute_dtype(model, delta)

    effective_taylor_orders = {
        order: _effective_taylor_order(model, order, n_taylor)
        for order in order_tuple
    }
    derivative_orders = sorted(
        {
            order + i
            for order, n in effective_taylor_orders.items()
            for i in range(1, n + 1)
        }
    )
    derivative_abs = {
        derivative_order: model.AB.activation_m_prime_stable(
            z,
            derivative_order,
        ).abs()
        for derivative_order in derivative_orders
    }

    exponent = -2.0 * torch.clamp(z.abs() - delta, min=0.0)
    bounds = []
    for order in order_tuple:
        n = effective_taylor_orders[order]
        result = torch.zeros_like(z)
        delta_power = delta.clone()
        for i in range(1, n + 1):
            result = result + (
                delta_power / math.factorial(i)
            ) * derivative_abs[order + i]
            delta_power = delta_power * delta

        tail_order = order + n + 1
        remainder_coefficient = (
            math.factorial(tail_order)
            / math.factorial(n + 1)
            * 2 ** (tail_order + 1)
        )
        bounds.append(result + delta_power * remainder_coefficient * torch.exp(exponent))

    return torch.stack(bounds)


def validate_nonnegative_bound(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Return ``tensor`` after rejecting invalid bound values.

    Bound expressions should be assembled from nonnegative terms.  A negative
    entry is therefore a bug, not something to hide by clamping.
    """
    finite_mask = torch.isfinite(tensor)
    if not finite_mask.all():
        first_bad = tensor[~finite_mask].flatten()[0].item()
        raise FloatingPointError(
            f"{name} contains non-finite bound entries; first bad value is "
            f"{first_bad}."
        )

    negative_mask = tensor < 0
    if negative_mask.any():
        min_value = tensor.min().item()
        raise FloatingPointError(
            f"{name} contains negative bound entries; minimum is {min_value}."
        )

    return tensor


def _as_activation_compute_dtype(model, x: torch.Tensor) -> torch.Tensor:
    dtype = getattr(model.AB, "dtype", None)
    if dtype is None or x.dtype == dtype:
        return x
    return x.to(dtype=dtype)


def _effective_taylor_order(
    model,
    order: int,
    n_taylor: int | None,
) -> int:
    if n_taylor is None:
        n = (
            model.AB.n_taylor_default
            if model.AB.n_taylor_default is not None
            else model.AB.MAX_DERIVATIVE - order
        )
    else:
        n = min(n_taylor, model.AB.MAX_DERIVATIVE - order)

    if not (1 <= n <= model.AB.MAX_DERIVATIVE - order):
        raise ValueError(
            f"n_taylor={n} out of range for order={order}; "
            f"must be in [1, {model.AB.MAX_DERIVATIVE - order}]."
        )
    return n
