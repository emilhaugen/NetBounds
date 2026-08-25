"""Scalar PDE data with user-supplied modulus-of-continuity certificates.

``GlobalModulusFunction`` is the general case: the user supplies both the
value function and the modulus. ``HolderFunction`` and ``LipschitzFunction``
are convenience special cases with the standard Holder and Lipschitz moduli only requiring constants.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch


class ModulusFunctionCertificate(Protocol):
    """User-supplied scalar field values together with certified box variations."""

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Return field values at centers, with shape [M] or [M, 1]."""

    def variation_bound(
        self,
        centers: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        """Return certified local variation bounds, shape [M] or [M, 1]."""


class ScalarLipschitzFunctionCertificate(ModulusFunctionCertificate, Protocol):
    r"""Scalar values with an explicit global $\ell_\infty$ Lipschitz constant."""

    def lipschitz_bound(self, centers: torch.Tensor) -> torch.Tensor:
        """Return per-cell Lipschitz constants with shape [M] or [M, 1]."""


ModulusInitialDatumCertificate = ModulusFunctionCertificate


# Three implementations of the interface above.
@dataclass
class GlobalModulusFunction:
    value_fn: Callable[[torch.Tensor], torch.Tensor]
    omega: Callable[[torch.Tensor], torch.Tensor]

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_fn(x)

    def variation_bound(
        self,
        centers: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        rho = torch.full(
            (centers.shape[0],),
            float(radius),
            dtype=centers.dtype,
            device=centers.device,
        )
        return self.omega(rho)


@dataclass
class HolderFunction:
    value_fn: Callable[[torch.Tensor], torch.Tensor]
    K: float | torch.Tensor
    alpha: float

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_fn(x)

    def variation_bound(
        self,
        centers: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")
        K = torch.as_tensor(self.K, dtype=centers.dtype, device=centers.device)
        rho = torch.full(
            (centers.shape[0],),
            float(radius),
            dtype=centers.dtype,
            device=centers.device,
        )
        return K * rho.pow(self.alpha)


@dataclass
class LipschitzFunction:
    value_fn: Callable[[torch.Tensor], torch.Tensor]
    L: float | torch.Tensor

    def value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_fn(x)

    def variation_bound(
        self,
        centers: torch.Tensor,
        radius: float,
    ) -> torch.Tensor:
        L = torch.as_tensor(self.L, dtype=centers.dtype, device=centers.device)
        rho = torch.full(
            (centers.shape[0],),
            float(radius),
            dtype=centers.dtype,
            device=centers.device,
        )
        return L * rho

    def lipschitz_bound(self, centers: torch.Tensor) -> torch.Tensor:
        L = torch.as_tensor(self.L, dtype=centers.dtype, device=centers.device)
        if L.ndim == 0:
            return L.expand(centers.shape[0])
        if L.shape == (centers.shape[0],):
            return L
        raise ValueError("L must be scalar or have shape [M].")


GlobalModulusInitialDatum = GlobalModulusFunction
HolderInitialDatum = HolderFunction
LipschitzInitialDatum = LipschitzFunction
