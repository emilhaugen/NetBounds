"""Shared L2 quadrature accumulation utilities for residual bounds."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ResidualL2BoundResult:
    name: str
    l2_bound: torch.Tensor
    l2_squared_bound: torch.Tensor
    cell_l2_squared_bounds: torch.Tensor | None
    centers: torch.Tensor | None
    cell_volume: torch.Tensor
    midpoint_l2: torch.Tensor | None = None
    midpoint_l2_squared: torch.Tensor | None = None
    bound_to_midpoint_ratio: torch.Tensor | None = None
    certified: bool = True
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


L2QuadratureResult = ResidualL2BoundResult


def as_cell_vector(
    name: str,
    value: torch.Tensor,
    *,
    length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Normalize per-cell values to shape [M] and reject unsafe broadcasting shapes."""
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=dtype, device=device)
    else:
        value = value.to(dtype=dtype, device=device)

    if value.ndim == 0 and length == 1:
        value = value.reshape(1)
    elif value.ndim == 2 and value.shape[1] == 1:
        value = value.squeeze(-1)
    elif value.ndim != 1:
        raise ValueError(f"{name} must have shape [M] or [M, 1].")

    if value.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}].")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite.")
    return value


def require_nonnegative(
    name: str,
    value: torch.Tensor,
    *,
    tol: float = 0.0,
) -> torch.Tensor:
    """Validate a bound is nonnegative, allowing only explicit tiny roundoff."""
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite.")
    if (value < -tol).any():
        min_value = value.min().item()
        raise ValueError(f"{name} must be nonnegative; min={min_value:.6e}.")
    if tol > 0.0:
        return value.clamp_min(0.0)
    return value


class L2QuadratureAccumulator:
    """Streaming accumulator for cellwise L2 upper bounds."""

    def __init__(
        self,
        *,
        cell_volume: torch.Tensor | float,
        dtype: torch.dtype,
        device: torch.device | str,
        store_diagnostics: bool = False,
    ):
        self.cell_volume = torch.as_tensor(cell_volume, dtype=dtype, device=device)
        if self.cell_volume.ndim != 0:
            raise ValueError("cell_volume must be scalar.")
        if not torch.isfinite(self.cell_volume) or self.cell_volume <= 0:
            raise ValueError("cell_volume must be positive and finite.")

        self.dtype = dtype
        self.device = device
        self.store_diagnostics = store_diagnostics
        self._l2_squared_bound = torch.zeros((), dtype=dtype, device=device)
        self._midpoint_l2_squared = torch.zeros((), dtype=dtype, device=device)
        self._has_midpoint_residuals = False
        self._cell_l2_squared_bounds: list[torch.Tensor] = []
        self._centers: list[torch.Tensor] = []
        self._diagnostics: dict[str, list[torch.Tensor]] = {}

    def add_cell_bounds(
        self,
        cell_residual_bound: torch.Tensor,
        *,
        midpoint_residual: torch.Tensor | None = None,
        centers: torch.Tensor | None = None,
        diagnostics: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        cell_residual_bound = as_cell_vector(
            "cell_residual_bound",
            cell_residual_bound,
            length=cell_residual_bound.numel(),
            dtype=self.dtype,
            device=self.device,
        )
        cell_residual_bound = require_nonnegative(
            "cell_residual_bound",
            cell_residual_bound,
        )
        cell_l2_squared_bounds = self.cell_volume * cell_residual_bound.square()
        self._l2_squared_bound = (
            self._l2_squared_bound + cell_l2_squared_bounds.sum()
        )

        if midpoint_residual is not None:
            midpoint_residual = as_cell_vector(
                "midpoint_residual",
                midpoint_residual,
                length=cell_residual_bound.numel(),
                dtype=self.dtype,
                device=self.device,
            )
            midpoint_l2_squared = self.cell_volume * midpoint_residual.square()
            self._midpoint_l2_squared = (
                self._midpoint_l2_squared + midpoint_l2_squared.sum()
            )
            self._has_midpoint_residuals = True

        if self.store_diagnostics:
            self._cell_l2_squared_bounds.append(cell_l2_squared_bounds.detach())
            if centers is not None:
                self._centers.append(centers.to(dtype=self.dtype, device=self.device).detach())
            if diagnostics is not None:
                for key, tensor in diagnostics.items():
                    self._diagnostics.setdefault(key, []).append(tensor.detach())

        return cell_l2_squared_bounds

    def result(
        self,
        name: str,
        *,
        certified: bool = True,
        notes: tuple[str, ...] = (),
    ) -> ResidualL2BoundResult:
        cell_l2_squared_bounds = (
            torch.cat(self._cell_l2_squared_bounds)
            if self.store_diagnostics and self._cell_l2_squared_bounds
            else None
        )
        centers = (
            torch.cat(self._centers)
            if self.store_diagnostics and self._centers
            else None
        )
        diagnostics = {
            key: torch.cat(chunks)
            for key, chunks in self._diagnostics.items()
        }
        l2_bound = torch.sqrt(self._l2_squared_bound)
        midpoint_l2_squared = None
        midpoint_l2 = None
        bound_to_midpoint_ratio = None
        if self._has_midpoint_residuals:
            midpoint_l2_squared = self._midpoint_l2_squared
            midpoint_l2 = torch.sqrt(midpoint_l2_squared)
            if midpoint_l2.item() > 0.0:
                bound_to_midpoint_ratio = l2_bound / midpoint_l2
            else:
                bound_to_midpoint_ratio = torch.as_tensor(
                    torch.inf,
                    dtype=self.dtype,
                    device=self.device,
                )
        return ResidualL2BoundResult(
            name=name,
            l2_bound=l2_bound,
            l2_squared_bound=self._l2_squared_bound,
            cell_l2_squared_bounds=cell_l2_squared_bounds,
            centers=centers,
            cell_volume=self.cell_volume,
            midpoint_l2=midpoint_l2,
            midpoint_l2_squared=midpoint_l2_squared,
            bound_to_midpoint_ratio=bound_to_midpoint_ratio,
            certified=certified,
            diagnostics=diagnostics,
            notes=notes,
        )
