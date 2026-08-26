"""Fixed Heat-3D adaptive quadrature used by the current paper.

It exposes only the fixed adaptive calculation used by the current Heat-3D
table entry.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import torch

from .pde import heat_hyper_taylor_rect_batch

PUBLIC_SUM_KEYS = (
    "bound_l2_squared",
    "moment_minkowski_l2_squared",
    "moment_cross_l2_squared",
    "moment_remainder_l2_squared",
    "moment_cross_integral",
    "affine_l2_squared",
    "midpoint_l2_squared",
    "rho",
    "rho_hyper",
    "rho_boundary",
)
_EVALUATOR_KEYS = (*PUBLIC_SUM_KEYS, "partition_bound_l2_squared")

HeatEvaluator = Callable[
    [torch.Tensor, torch.Tensor, int], dict[str, torch.Tensor]
]


def child_geometry(
    centers: torch.Tensor, eps_vec: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split every spatial and temporal axis, yielding 16 dyadic children."""

    if centers.ndim != 2 or centers.shape[1] != 4:
        raise ValueError("centers must have shape (batch, 4)")
    if eps_vec.shape != (4,):
        raise ValueError("eps_vec must have shape (4,)")
    signs = torch.cartesian_prod(
        *[
            torch.tensor([-1.0, 1.0], dtype=centers.dtype, device=centers.device)
            for _ in range(4)
        ]
    )
    if signs.shape != (16, 4):
        raise AssertionError(f"all-axis split generated shape {tuple(signs.shape)}")
    child_eps = eps_vec / 2.0
    children = (centers[:, None, :] + signs[None, :, :] * child_eps).reshape(
        -1, 4
    )
    return children, child_eps


def evaluate_heat3_cells(
    model: torch.nn.Module,
    centers: torch.Tensor,
    eps_vec: torch.Tensor,
    kernel_batch_size: int,
) -> dict[str, torch.Tensor]:
    """Evaluate appendix-direct and split-policy quantities per adaptive cell."""

    chunks: dict[str, list[torch.Tensor]] = {key: [] for key in _EVALUATOR_KEYS}
    cell_volume = float(torch.prod(2.0 * eps_vec).item())
    with torch.inference_mode():
        for start in range(0, centers.shape[0], kernel_batch_size):
            y = centers[start : start + kernel_batch_size]
            residual, rho, diagnostics = heat_hyper_taylor_rect_batch(
                model=model,
                y=y,
                eps_vec=eps_vec,
                alpha=0.1,
                n_taylor=None,
            )
            values = {
                "bound_l2_squared": diagnostics[
                    "hyper_taylor_moment_best_bound_l2sq"
                ],
                "moment_minkowski_l2_squared": diagnostics[
                    "hyper_taylor_moment_bound_l2sq"
                ],
                "moment_cross_l2_squared": diagnostics[
                    "hyper_taylor_moment_cross_bound_l2sq"
                ],
                "moment_remainder_l2_squared": diagnostics[
                    "hyper_taylor_moment_remainder_l2sq"
                ],
                "moment_cross_integral": diagnostics[
                    "hyper_taylor_moment_cross_integral"
                ],
                # This auxiliary certificate chooses adaptive cells. It is not
                # included in the final table value.
                "partition_bound_l2_squared": diagnostics["hyper_taylor_bound_l2sq"],
                "affine_l2_squared": diagnostics["hyper_taylor_affine_l2sq"],
                "midpoint_l2_squared": residual.square() * cell_volume,
                "rho": rho,
                "rho_hyper": diagnostics["hyper_taylor_rho_hyper"],
                "rho_boundary": diagnostics["hyper_taylor_rho_boundary"],
            }
            for key, value in values.items():
                chunks[key].append(value.detach())
    return {key: torch.cat(parts) for key, parts in chunks.items()}


def partition_units(leaf_counts_by_depth: list[int], max_depth: int) -> int:
    if len(leaf_counts_by_depth) != max_depth + 1:
        raise ValueError("leaf-count vector length does not match max_depth")
    return sum(
        int(count) * 16 ** (max_depth - depth)
        for depth, count in enumerate(leaf_counts_by_depth)
    )


def validate_adaptive_counts(
    *,
    root_count: int,
    leaf_counts_by_depth: list[int],
    split_counts_by_depth: list[int],
    max_depth: int,
    kernel_cells_evaluated: int,
) -> None:
    if (
        root_count < 0
        or len(leaf_counts_by_depth) != max_depth + 1
        or len(split_counts_by_depth) != max_depth
    ):
        raise ValueError("invalid adaptive count-vector shape")
    if any(value < 0 for value in leaf_counts_by_depth + split_counts_by_depth):
        raise ValueError("negative adaptive count")
    for depth in range(max_depth):
        available = (
            root_count if depth == 0 else 16 * split_counts_by_depth[depth - 1]
        )
        if leaf_counts_by_depth[depth] + split_counts_by_depth[depth] != available:
            raise ValueError(f"adaptive conservation mismatch at depth {depth}")
    terminal = root_count if max_depth == 0 else 16 * split_counts_by_depth[-1]
    if leaf_counts_by_depth[max_depth] != terminal:
        raise ValueError("adaptive terminal-depth conservation mismatch")
    expected_evaluated = root_count + 16 * sum(split_counts_by_depth)
    if kernel_cells_evaluated != expected_evaluated:
        raise ValueError("adaptive kernel evaluation count mismatch")
    if partition_units(leaf_counts_by_depth, max_depth) != root_count * 16**max_depth:
        raise ValueError("adaptive partition identity mismatch")


def _validate_metrics(metrics: dict[str, torch.Tensor], count: int) -> None:
    if set(metrics) != set(_EVALUATOR_KEYS):
        raise ValueError(f"evaluator returned wrong keys: {sorted(metrics)}")
    for key, value in metrics.items():
        if value.shape != (count,):
            raise ValueError(f"{key} has shape {tuple(value.shape)}, expected {(count,)}")
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"non-finite {key} in adaptive cell batch")
        if not bool((value >= 0).all().item()):
            raise FloatingPointError(f"negative {key} in adaptive cell batch")
    scale = torch.maximum(
        metrics["partition_bound_l2_squared"],
        torch.maximum(
            metrics["moment_minkowski_l2_squared"],
            torch.maximum(
                metrics["moment_cross_l2_squared"],
                torch.maximum(
                    metrics["bound_l2_squared"],
                    torch.maximum(
                        metrics["affine_l2_squared"],
                        metrics["midpoint_l2_squared"],
                    ),
                ),
            ),
        ),
    ).clamp_min(torch.finfo(metrics["bound_l2_squared"].dtype).tiny)
    tolerance = 64.0 * torch.finfo(scale.dtype).eps * scale
    for upper, lower in (
        ("moment_minkowski_l2_squared", "bound_l2_squared"),
        ("moment_cross_l2_squared", "bound_l2_squared"),
        ("partition_bound_l2_squared", "bound_l2_squared"),
        ("bound_l2_squared", "affine_l2_squared"),
        ("partition_bound_l2_squared", "affine_l2_squared"),
        ("affine_l2_squared", "midpoint_l2_squared"),
    ):
        if not bool((metrics[upper] + tolerance >= metrics[lower]).all().item()):
            raise FloatingPointError(f"cell ordering failure: {upper} < {lower}")


def certify_heat3_adaptive_block(
    *,
    root_centers: torch.Tensor,
    root_eps: torch.Tensor,
    split_excess_density: float,
    max_depth: int,
    kernel_batch_size: int,
    model: torch.nn.Module | None = None,
    evaluator: HeatEvaluator | None = None,
) -> dict[str, Any]:
    """Certify a complete root block using breadth-first dyadic refinement."""

    if root_centers.ndim != 2 or root_centers.shape[1] != 4:
        raise ValueError("root_centers must have shape (n, 4)")
    if root_eps.shape != (4,):
        raise ValueError("root_eps must have shape (4,)")
    if not math.isfinite(split_excess_density) or split_excess_density < 0.0:
        raise ValueError("split_excess_density must be finite and nonnegative")
    if max_depth < 0 or kernel_batch_size <= 0:
        raise ValueError("invalid max_depth or kernel_batch_size")
    if evaluator is None:
        if model is None:
            raise ValueError("model is required when evaluator is not supplied")
        evaluate: HeatEvaluator = lambda centers, eps, batch: evaluate_heat3_cells(
            model, centers, eps, batch
        )
    else:
        evaluate = evaluator

    root_count = int(root_centers.shape[0])
    current = root_centers
    current_eps = root_eps
    sums = {key: 0.0 for key in PUBLIC_SUM_KEYS}
    leaf_counts = [0] * (max_depth + 1)
    split_counts = [0] * max_depth
    kernel_cells_evaluated = 0
    rho_max = 0.0

    for depth in range(max_depth + 1):
        count = int(current.shape[0])
        if count == 0:
            break
        metrics = evaluate(current, current_eps, kernel_batch_size)
        _validate_metrics(metrics, count)
        kernel_cells_evaluated += count
        cell_volume = float(torch.prod(2.0 * current_eps).item())
        excess_density = (
            metrics["partition_bound_l2_squared"]
            - metrics["midpoint_l2_squared"]
        ).clamp_min(0.0) / cell_volume
        split = (
            excess_density > split_excess_density
            if depth < max_depth
            else torch.zeros(count, dtype=torch.bool, device=current.device)
        )
        accept = ~split
        accepted_count = int(accept.sum().item())
        leaf_counts[depth] = accepted_count
        if accepted_count:
            for key in PUBLIC_SUM_KEYS:
                sums[key] += float(metrics[key][accept].to(torch.float64).sum().item())
            rho_max = max(rho_max, float(metrics["rho"][accept].max().item()))
        if depth == max_depth:
            break
        split_count = int(split.sum().item())
        split_counts[depth] = split_count
        if split_count == 0:
            break
        current, current_eps = child_geometry(current[split], current_eps)

    validate_adaptive_counts(
        root_count=root_count,
        leaf_counts_by_depth=leaf_counts,
        split_counts_by_depth=split_counts,
        max_depth=max_depth,
        kernel_cells_evaluated=kernel_cells_evaluated,
    )
    return {
        "root_count": root_count,
        "sums": sums,
        "leaf_counts_by_depth": leaf_counts,
        "split_counts_by_depth": split_counts,
        "adaptive_leaf_count": sum(leaf_counts),
        "kernel_cells_evaluated": kernel_cells_evaluated,
        "partition_units": partition_units(leaf_counts, max_depth),
        "rho_max": rho_max,
    }
