"""First-order derivative-difference bounds from the TeX Algorithm 1."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .common import (
    activation_derivative_bounds,
    hidden_preactivations_and_m,
    normalize_input_batch,
    preactivation_delta,
)


@dataclass
class FirstOrderBoundsResult:
    """Batched result of the first-order alternating bound computation."""

    # Shape: (len(model.AB.M_RANGE), L, batch, width), ordered by derivative order.
    Q: torch.Tensor
    E: torch.Tensor
    delta: torch.Tensor
    preactivation_radii: torch.Tensor
    z: torch.Tensor
    m: torch.Tensor
    preactivation_gradients: torch.Tensor
    preactivation_error_bounds: torch.Tensor
    activation_gradients: torch.Tensor
    activation_error_bounds: torch.Tensor
    base_gradient: torch.Tensor
    derivative_bounds: torch.Tensor
    eps: float
    n_taylor: int | None


@dataclass
class FirstOrderValueDerivativeBoundsResult:
    """Memory-light first-order result for value and derivative bounds."""

    base_value: torch.Tensor
    base_gradient: torch.Tensor
    derivative_bounds: torch.Tensor
    eps: float
    n_taylor: int | None


class FirstOrderBoundAccumulator:
    """Vectorized recurrence for first derivative difference bounds.

    The weight convention matches ``torch.nn.Linear.weight`` and the paper
    notation: hidden weights have shape ``(out_width, in_width)``.  All
    derivative tensors are batched and carry every input coordinate at once,
    with shape ``(N, width, input_dim)``.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        input_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.dtype = dtype
        self.device = device
        self.activation_gradient: torch.Tensor | None = None
        self.activation_error_bound: torch.Tensor | None = None
        self.preactivation_gradients: list[torch.Tensor] = []
        self.preactivation_error_bounds: list[torch.Tensor] = []
        self.activation_gradients: list[torch.Tensor] = []
        self.activation_error_bounds: list[torch.Tensor] = []

    def next_preactivation(
        self,
        W_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return base gradients, error bounds, and radius components for ``z^k``."""
        W_k = W_k.to(dtype=self.dtype, device=self.device)

        if self.activation_gradient is None:
            Z_base = W_k.unsqueeze(0).expand(self.batch_size, -1, -1)
            Z_error = torch.zeros_like(Z_base)
        else:
            Z_base = torch.einsum("ji,nid->njd", W_k, self.activation_gradient)
            Z_error = torch.einsum(
                "ji,nid->njd",
                W_k.abs(),
                self.activation_error_bound,
            )

        radius_components = Z_base.abs() + Z_error
        return Z_base, Z_error, radius_components

    def append_layer(
        self,
        m_k: torch.Tensor,
        Q_k: torch.Tensor,
        Z_base: torch.Tensor,
        Z_error: torch.Tensor,
        *,
        store_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update the propagated activation-gradient data for one layer."""
        m_k = m_k.to(dtype=self.dtype, device=self.device)
        Q_k = Q_k.to(dtype=self.dtype, device=self.device)

        G_k = m_k.unsqueeze(-1) * Z_base
        B_k = (
            Q_k.unsqueeze(-1) * (Z_base.abs() + Z_error)
            + m_k.abs().unsqueeze(-1) * Z_error
        )

        self.activation_gradient = G_k
        self.activation_error_bound = B_k
        if store_cache:
            self.preactivation_gradients.append(Z_base)
            self.preactivation_error_bounds.append(Z_error)
            self.activation_gradients.append(G_k)
            self.activation_error_bounds.append(B_k)
        return G_k, B_k

    def final_bounds(self, W_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``df(y)`` and a bound for ``|df(x)-df(y)|``."""
        W_out = W_out.to(dtype=self.dtype, device=self.device)
        if W_out.ndim != 2 or W_out.shape[0] != 1:
            raise ValueError("The output layer must have a single row.")

        w_out = W_out[0]
        base_gradient = torch.einsum(
            "i,nid->nd",
            w_out,
            self.activation_gradient,
        )
        derivative_bounds = torch.einsum(
            "i,nid->nd",
            w_out.abs(),
            self.activation_error_bound,
        )
        return base_gradient, derivative_bounds


def compute_first_order_bounds(
    model,
    y: torch.Tensor,
    eps: float,
    n_taylor: int | None = None,
) -> FirstOrderBoundsResult:
    """Compute first-derivative difference bounds over full input boxes."""
    return _compute_first_order_bounds(
        model,
        y,
        eps,
        n_taylor=n_taylor,
        spatial_dim=None,
    )


def compute_first_order_bounds_over_spatial_box(
    model,
    y: torch.Tensor,
    spatial_eps: float,
    n_taylor: int | None = None,
) -> FirstOrderBoundsResult:
    """Compute first-derivative difference bounds over spatial boxes."""
    return _compute_first_order_bounds(
        model,
        y,
        spatial_eps,
        n_taylor=n_taylor,
        spatial_dim=model.d,
    )


def compute_value_and_first_derivative_bounds_over_spatial_box(
    model,
    y: torch.Tensor,
    spatial_eps: float,
    n_taylor: int | None = None,
) -> FirstOrderValueDerivativeBoundsResult:
    """Return value, gradient, and derivative bounds over spatial boxes."""
    y = normalize_input_batch(model, y)

    with torch.no_grad():
        accumulator = FirstOrderBoundAccumulator(
            batch_size=y.shape[0],
            input_dim=model.input_dim,
            dtype=y.dtype,
            device=y.device,
        )

        a = y
        for layer in model.hidden_layers:
            z_k = layer(a)
            m_k = model.AB.activation_m_prime_stable(z_k, 1)
            (
                Z_base,
                Z_error,
                radius_components,
            ) = accumulator.next_preactivation(layer.weight)
            _, delta_k = preactivation_delta(
                radius_components,
                spatial_eps,
                spatial_dim=model.d,
            )
            Q_k = activation_derivative_bounds(
                model,
                z_k,
                delta_k,
                orders=(1,),
                n_taylor=n_taylor,
            )[0]
            accumulator.append_layer(
                m_k,
                Q_k,
                Z_base,
                Z_error,
                store_cache=False,
            )
            a = model.activation(z_k)

        base_value = model.output_layer(a)
        base_gradient, derivative_bounds = accumulator.final_bounds(
            model.output_layer.weight,
        )

        return FirstOrderValueDerivativeBoundsResult(
            base_value=base_value,
            base_gradient=base_gradient,
            derivative_bounds=derivative_bounds,
            eps=float(spatial_eps),
            n_taylor=n_taylor,
        )


def _compute_first_order_bounds(
    model,
    y: torch.Tensor,
    eps: float,
    *,
    n_taylor: int | None,
    spatial_dim: int | None,
) -> FirstOrderBoundsResult:
    y = normalize_input_batch(model, y)

    with torch.no_grad():
        z_values, m_by_order = hidden_preactivations_and_m(model, y, orders=(1,))
        m_values = m_by_order[1]

        Q_layers = []
        E_layers = []
        delta_layers = []
        accumulator = FirstOrderBoundAccumulator(
            batch_size=y.shape[0],
            input_dim=model.input_dim,
            dtype=y.dtype,
            device=y.device,
        )

        for layer, z_k, m_k in zip(model.hidden_layers, z_values, m_values):
            (
                Z_base,
                Z_error,
                radius_components,
            ) = accumulator.next_preactivation(layer.weight)
            E_k, delta_k = preactivation_delta(
                radius_components,
                eps,
                spatial_dim=spatial_dim,
            )
            Q_k_all = activation_derivative_bounds(
                model,
                z_k,
                delta_k,
                orders=model.AB.M_RANGE,
                n_taylor=n_taylor,
            )
            accumulator.append_layer(m_k, Q_k_all[0], Z_base, Z_error)
            E_layers.append(E_k)
            delta_layers.append(delta_k)
            Q_layers.append(Q_k_all)

        base_gradient, derivative_bounds = accumulator.final_bounds(
            model.output_layer.weight,
        )

        return FirstOrderBoundsResult(
            Q=torch.stack(Q_layers, dim=1),
            E=torch.stack(E_layers),
            delta=torch.stack(delta_layers),
            preactivation_radii=torch.stack(delta_layers),
            z=torch.stack(z_values),
            m=torch.stack(m_values),
            preactivation_gradients=torch.stack(
                accumulator.preactivation_gradients,
            ),
            preactivation_error_bounds=torch.stack(
                accumulator.preactivation_error_bounds,
            ),
            activation_gradients=torch.stack(accumulator.activation_gradients),
            activation_error_bounds=torch.stack(
                accumulator.activation_error_bounds,
            ),
            base_gradient=base_gradient,
            derivative_bounds=derivative_bounds,
            eps=float(torch.as_tensor(eps, dtype=y.dtype, device=y.device).max().item()),
            n_taylor=n_taylor,
        )
