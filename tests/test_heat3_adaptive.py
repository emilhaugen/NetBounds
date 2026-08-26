from __future__ import annotations

import pytest

pytestmark = pytest.mark.torch

torch = pytest.importorskip("torch")

from netbounds.numerics.pde_adaptive import certify_heat3_adaptive_block


def test_adaptive_block_preserves_partition_and_accepted_sums() -> None:
    calls = 0

    def evaluator(centers, eps_vec, kernel_batch_size):
        nonlocal calls
        calls += 1
        count = centers.shape[0]
        base = torch.ones(count, dtype=torch.float32)
        legacy = base.clone()
        if calls == 1:
            # Split exactly the first root. The second root is accepted at depth 0.
            legacy[0] = 2.0
        values = {
            "bound_l2_squared": base * 0.8,
            "moment_minkowski_l2_squared": base * 0.9,
            "moment_cross_l2_squared": base * 0.85,
            "moment_remainder_l2_squared": base * 0.2,
            "moment_cross_integral": base * 0.1,
            "affine_l2_squared": base * 0.5,
            "midpoint_l2_squared": base * 0.25,
            "rho": base * 0.3,
            "rho_hyper": base * 0.2,
            "rho_boundary": base * 0.1,
            "legacy_bound_l2_squared": legacy,
        }
        return values

    centers = torch.tensor(
        [[0.25, 0.25, 0.25, 0.25], [0.75, 0.75, 0.75, 0.75]],
        dtype=torch.float32,
    )
    eps = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
    result = certify_heat3_adaptive_block(
        root_centers=centers,
        root_eps=eps,
        split_excess_density=15.0,
        max_depth=1,
        kernel_batch_size=8,
        evaluator=evaluator,
    )

    assert result["root_count"] == 2
    assert result["leaf_counts_by_depth"] == [1, 16]
    assert result["split_counts_by_depth"] == [1]
    assert result["adaptive_leaf_count"] == 17
    assert result["kernel_cells_evaluated"] == 18
    assert result["partition_units"] == 32
    assert result["sums"]["bound_l2_squared"] == pytest.approx(17 * 0.8)
    assert "legacy_bound_l2_squared" not in result["sums"]


def test_adaptive_block_rejects_bad_metric_ordering() -> None:
    def evaluator(centers, eps_vec, kernel_batch_size):
        count = centers.shape[0]
        base = torch.ones(count, dtype=torch.float32)
        values = {
            "bound_l2_squared": base * 0.4,
            "moment_minkowski_l2_squared": base * 0.3,
            "moment_cross_l2_squared": base * 0.5,
            "moment_remainder_l2_squared": base * 0.2,
            "moment_cross_integral": base * 0.1,
            "affine_l2_squared": base * 0.2,
            "midpoint_l2_squared": base * 0.1,
            "rho": base * 0.3,
            "rho_hyper": base * 0.2,
            "rho_boundary": base * 0.1,
            "legacy_bound_l2_squared": base * 0.6,
        }
        return values

    with pytest.raises(FloatingPointError, match="cell ordering"):
        certify_heat3_adaptive_block(
            root_centers=torch.full((1, 4), 0.5),
            root_eps=torch.full((4,), 0.25),
            split_excess_density=1.0,
            max_depth=0,
            kernel_batch_size=1,
            evaluator=evaluator,
        )
