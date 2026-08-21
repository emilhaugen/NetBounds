from __future__ import annotations

from typing import Any, cast

import torch

from netbounds.numerics._pde_helpers import _boundary_center_and_variations
from netbounds.numerics.pde import _boundary_grouped_coeffs, _boundary_p_first_coeffs


def _assert_tree_close(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert left.keys() == right.keys()
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, list):
            assert isinstance(b, list) and len(a) == len(b)
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, rtol=0.0, atol=0.0)
        else:
            torch.testing.assert_close(a, b, rtol=0.0, atol=0.0)


def test_isotropic_scalar_and_vector_radii_are_identical() -> None:
    centers = torch.tensor(
        [[0.31, 0.47, 0.63, 0.52], [0.72, 0.38, 0.56, 0.28]],
        dtype=torch.float64,
    )
    scalar = 0.004
    vector = torch.full((3,), scalar, dtype=torch.float64)
    _assert_tree_close(
        _boundary_center_and_variations(centers, scalar, 3),
        _boundary_center_and_variations(centers, vector, 3),
    )
    _assert_tree_close(
        _boundary_grouped_coeffs(centers, scalar, 3),
        _boundary_grouped_coeffs(centers, vector, 3),
    )
    _assert_tree_close(
        _boundary_p_first_coeffs(centers, scalar, 3),
        _boundary_p_first_coeffs(centers, vector, 3),
    )


def test_anisotropic_p_and_first_derivative_variations_contain_samples() -> None:
    center = torch.tensor([[0.37, 0.61, 0.43, 0.53]], dtype=torch.float64)
    eps = torch.tensor([0.0025, 0.005, 0.005], dtype=torch.float64)
    coefficients = cast(dict[str, Any], _boundary_p_first_coeffs(center, eps, 3))
    generator = torch.Generator().manual_seed(97)
    offsets = (
        2.0 * torch.rand((2048, 3), generator=generator, dtype=torch.float64)
        - 1.0
    ) * eps
    x = center[:, :3] + offsets
    s = x * (1.0 - x)
    d1 = 1.0 - 2.0 * x
    for i in range(3):
        others = [j for j in range(3) if j != i]
        sampled_p = s[:, others].prod(dim=-1)
        deviation = (sampled_p - coefficients["P"][i][0]).abs().max()
        assert deviation <= coefficients["eta_P"][i][0] * (1.0 + 1e-12)
        for q in range(3):
            if q == i:
                sampled_pq = torch.zeros_like(sampled_p)
            else:
                factors = [j for j in range(3) if j != i and j != q]
                sampled_pq = d1[:, q]
                if factors:
                    sampled_pq = sampled_pq * s[:, factors].prod(dim=-1)
            deviation_q = (sampled_pq - coefficients["P_q"][i][q][0]).abs().max()
            assert deviation_q <= coefficients["eta_P_q"][i][q][0] * (1.0 + 1e-12)
