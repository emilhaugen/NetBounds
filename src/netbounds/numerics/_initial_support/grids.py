"""Spatial grids for residual quadrature."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import torch


def _containment_tol(dtype: torch.dtype, base_tol: float) -> float:
    if torch.empty((), dtype=dtype).is_floating_point():
        return max(float(base_tol), 8.0 * torch.finfo(dtype).eps)
    return float(base_tol)


@dataclass(frozen=True)
class UniformBoxGrid:
    centers: torch.Tensor
    spatial_eps: float
    d: int

    def validate(self, *, tol: float = 1e-12) -> None:
        if not isinstance(self.centers, torch.Tensor):
            raise TypeError("centers must be a torch.Tensor.")
        if self.centers.ndim != 2:
            raise ValueError("centers must have shape [M, d].")
        if not isinstance(self.d, int) or self.d <= 0:
            raise ValueError("d must be a positive integer.")
        if self.centers.shape[1] != self.d:
            raise ValueError("centers.shape[1] must equal d.")
        if not self.centers.is_floating_point():
            raise TypeError("centers must have a floating-point dtype.")
        if not math.isfinite(float(self.spatial_eps)) or self.spatial_eps <= 0:
            raise ValueError("spatial_eps must be positive and finite.")
        if not torch.isfinite(self.centers).all():
            raise ValueError("centers must be finite.")

        effective_tol = _containment_tol(self.centers.dtype, tol)
        lower = self.centers - self.spatial_eps
        upper = self.centers + self.spatial_eps
        if (lower < -effective_tol).any() or (upper > 1.0 + effective_tol).any():
            raise ValueError("Uniform boxes must be contained in [0,1]^d.")

    @property
    def volume(self) -> torch.Tensor:
        return torch.as_tensor(
            (2.0 * self.spatial_eps) ** self.d,
            dtype=self.centers.dtype,
            device=self.centers.device,
        )

    @property
    def num_boxes(self) -> int:
        return self.centers.shape[0]


@dataclass(frozen=True)
class UniformUnitBoxGridSpec:
    d: int
    cells_per_dim: int
    dtype: torch.dtype = torch.float64
    device: torch.device | str = "cpu"

    def validate(self) -> None:
        if not isinstance(self.d, int) or self.d <= 0:
            raise ValueError("d must be a positive integer.")
        if not isinstance(self.cells_per_dim, int) or self.cells_per_dim <= 0:
            raise ValueError("cells_per_dim must be a positive integer.")
        if not torch.empty((), dtype=self.dtype).is_floating_point():
            raise TypeError("dtype must be floating point.")

    @property
    def spatial_eps(self) -> float:
        return 1.0 / (2.0 * self.cells_per_dim)

    @property
    def num_boxes(self) -> int:
        return self.cells_per_dim**self.d

    @property
    def volume(self) -> torch.Tensor:
        return torch.as_tensor(
            (2.0 * self.spatial_eps) ** self.d,
            dtype=self.dtype,
            device=self.device,
        )


def make_uniform_unit_box_grid(
    d: int,
    cells_per_dim: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> UniformBoxGrid:
    spec = UniformUnitBoxGridSpec(
        d=d,
        cells_per_dim=cells_per_dim,
        dtype=dtype,
        device=device,
    )
    spec.validate()
    coords_1d = (
        torch.arange(cells_per_dim, dtype=dtype, device=device) + 0.5
    ) / cells_per_dim
    mesh = torch.meshgrid(*(coords_1d for _ in range(d)), indexing="ij")
    centers = torch.stack([component.reshape(-1) for component in mesh], dim=-1)
    grid = UniformBoxGrid(centers=centers, spatial_eps=spec.spatial_eps, d=d)
    grid.validate()
    return grid


def make_uniform_unit_box_grid_spec(
    d: int,
    cells_per_dim: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> UniformUnitBoxGridSpec:
    spec = UniformUnitBoxGridSpec(
        d=d,
        cells_per_dim=cells_per_dim,
        dtype=dtype,
        device=device,
    )
    spec.validate()
    return spec


def _centers_from_linear_indices(
    linear_indices: torch.Tensor,
    spec: UniformUnitBoxGridSpec,
) -> torch.Tensor:
    coordinate_columns = []
    remaining = linear_indices
    cells_per_dim = spec.cells_per_dim
    for axis in range(spec.d):
        stride = cells_per_dim ** (spec.d - axis - 1)
        coordinate = torch.div(remaining, stride, rounding_mode="floor")
        remaining = remaining.remainder(stride)
        coordinate_columns.append(coordinate)

    coordinates = torch.stack(coordinate_columns, dim=-1).to(dtype=spec.dtype)
    return (coordinates + 0.5) / cells_per_dim


def iter_uniform_unit_box_grid_batches(
    spec: UniformUnitBoxGridSpec,
    batch_size: int | None = None,
) -> Iterator[UniformBoxGrid]:
    """Generate uniform grid center batches without materializing the full grid."""
    spec.validate()
    if batch_size is None:
        batch_size = spec.num_boxes
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    for start in range(0, spec.num_boxes, batch_size):
        stop = min(start + batch_size, spec.num_boxes)
        linear_indices = torch.arange(
            start,
            stop,
            dtype=torch.long,
            device=spec.device,
        )
        centers = _centers_from_linear_indices(linear_indices, spec)
        yield UniformBoxGrid(
            centers=centers,
            spatial_eps=spec.spatial_eps,
            d=spec.d,
        )


def iter_uniform_grid_batches(
    grid: UniformBoxGrid | UniformUnitBoxGridSpec,
    batch_size: int | None,
) -> Iterator[UniformBoxGrid]:
    """Yield batches from a materialized grid or a streaming uniform-grid spec."""
    if isinstance(grid, UniformUnitBoxGridSpec):
        yield from iter_uniform_unit_box_grid_batches(grid, batch_size)
        return

    grid.validate()
    if batch_size is None:
        yield grid
        return
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer or None.")

    for start in range(0, grid.num_boxes, batch_size):
        stop = min(start + batch_size, grid.num_boxes)
        yield UniformBoxGrid(
            centers=grid.centers[start:stop],
            spatial_eps=grid.spatial_eps,
            d=grid.d,
        )
