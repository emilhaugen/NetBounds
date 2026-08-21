"""Fixed reproductions from reached traces at 2add560, 1003b544, and 53bb374.

The trace and source-hash evidence is retained in
``provenance/checkpoint-reproduction-1d.json``. This module intentionally
contains only the twelve selected 1D cases, never campaign or training APIs.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import torch

from netbounds.data import VerificationError, load_catalog, repository_root, verify_file

from . import initial_gradient_q0, initial_gradient_q1, initial_q0, initial_q1
from ._initial_support.grids import UniformUnitBoxGridSpec
from .initial_data import (
    displacement_gradient_q0,
    displacement_gradient_q1,
    displacement_q0,
    displacement_q1,
    velocity_q0,
    velocity_q1,
)
from .model import TanhNetwork, load_checkpoint
from .pde import heat_hyper_taylor_rect_batch, wave_hyper_taylor_rect_batch
from .cases import CASES, Case


def _checked_model(case: Case, root: Path, device: torch.device) -> TanhNetwork:
    """Load only a checkpoint declared as bundled by the portable catalog."""

    catalog = load_catalog(root)
    matches = [entry for entry in catalog["checkpoints"] if entry.get("path") == case.checkpoint]
    if len(matches) != 1:
        raise VerificationError(f"catalog must declare exactly one checkpoint at {case.checkpoint}")
    entry = matches[0]
    if entry.get("bundled") is not True:
        raise VerificationError(f"{case.checkpoint}: fixed reproduction checkpoint is not bundled")
    if entry.get("sha256") != case.checkpoint_sha256:
        raise VerificationError(f"{case.checkpoint}: catalog SHA-256 disagrees with fixed case")
    checkpoint = verify_file(root, entry)
    storage_dtype = torch.float32 if case.storage_dtype == "float32" else torch.float64
    return load_checkpoint(
        checkpoint,
        width=case.width,
        d=case.d,
        L=case.L,
        nested=case.nested,
        storage_dtype=storage_dtype,
        device=device,
    )


def _float(value: torch.Tensor | float) -> float:
    return float(value if isinstance(value, float) else value.item())


def _require_initial_invariants(
    result: dict[str, float | int], *, num_boxes: int
) -> None:
    """Reject malformed initial-condition results before they reach comparison."""

    values = {
        name: float(value)
        for name, value in result.items()
        if name != "num_boxes"
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise FloatingPointError("initial-condition result contains a non-finite value")
    if int(result["num_boxes"]) != num_boxes:
        raise FloatingPointError("initial-condition result has an unexpected cell count")
    if not (values["eps"] > 0.0 and values["cell_volume"] > 0.0):
        raise FloatingPointError("initial-condition geometry is not positive")
    if not (
        values["l2_squared_bound"] >= values["midpoint_l2_squared"] >= 0.0
        and values["l2_bound"] >= values["midpoint_l2"] >= 0.0
        and values["bound_to_midpoint_ratio"] >= 1.0
    ):
        raise FloatingPointError("initial-condition result violates bound ordering")


def _initial_result(case: Case, model: TanhNetwork, batch_size: int) -> dict[str, float | int]:
    grid = UniformUnitBoxGridSpec(
        d=case.d,
        cells_per_dim=case.cells_per_dim,
        dtype=torch.float32,
        device=model.input_layer.weight.device,
    )
    common = {
        "model": model,
        "grid": grid,
        "batch_size": batch_size,
        "store_diagnostics": False,
    }
    if case.quantity == "displacement" and case.rule == "q0":
        result = initial_q0.bound_initial_displacement_residual_l2_moment_uniform(
            initial_g=displacement_q0(case.d), **common
        )
    elif case.quantity == "displacement" and case.rule == "q1":
        result = initial_q1.bound_initial_displacement_residual_l2_moment_taylor_uniform(
            initial_g=displacement_q1(case.d), **common
        )
    elif case.quantity == "gradient" and case.rule == "q0":
        result = initial_gradient_q0.bound_initial_displacement_gradient_residual_l2_moment_uniform(
            initial_grad=displacement_gradient_q0(case.d), **common
        )
    elif case.quantity == "gradient" and case.rule == "q1":
        result = initial_gradient_q1.bound_initial_displacement_gradient_residual_l2_moment_taylor_uniform(
            initial_grad=displacement_gradient_q1(case.d), **common
        )
    elif case.quantity == "velocity" and case.rule == "q0":
        result = initial_q0.bound_initial_velocity_residual_l2_moment_uniform(
            initial_velocity=velocity_q0(case.d), **common
        )
    elif case.quantity == "velocity" and case.rule == "q1":
        result = initial_q1.bound_initial_velocity_residual_l2_moment_taylor_uniform(
            initial_velocity=velocity_q1(case.d), **common
        )
    else:  # guarded by the fixed case registry
        raise AssertionError(f"unsupported initial case: {case.name}")

    numerical: dict[str, float | int] = {
        "l2_bound": _float(result.l2_bound),
        "l2_squared_bound": _float(result.l2_squared_bound),
        "midpoint_l2": _float(result.midpoint_l2),
        "midpoint_l2_squared": _float(result.midpoint_l2_squared),
        "bound_to_midpoint_ratio": _float(result.bound_to_midpoint_ratio),
        "cell_volume": _float(result.cell_volume),
        "eps": float(result.spatial_eps),
        "num_boxes": case.cells_per_dim**case.d,
    }
    _require_initial_invariants(numerical, num_boxes=case.cells_per_dim**case.d)
    return numerical


def _decode_centers(
    start: int,
    stop: int,
    *,
    spatial_cells: int,
    time_cells: int,
    device: torch.device,
) -> torch.Tensor:
    """Decode the historical grid ordering, with time as the fastest axis."""

    flat = torch.arange(start, stop, dtype=torch.int64, device=device)
    time_index = flat.remainder(time_cells)
    space_index = torch.div(flat, time_cells, rounding_mode="floor")
    return torch.stack(
        (
            (space_index.to(torch.float32) + 0.5) / float(spatial_cells),
            (time_index.to(torch.float32) + 0.5) / float(time_cells),
        ),
        dim=-1,
    )


_PDE_SUM_KEYS = (
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


def _pde_result(
    case: Case,
    model: TanhNetwork,
    batch_size: int,
    progress: Callable[[int, int], None] | None,
) -> dict[str, object]:
    device = model.input_layer.weight.device
    spatial_cells = time_cells = 500
    total = spatial_cells * time_cells
    eps = torch.tensor(
        [1.0 / (2.0 * spatial_cells), 1.0 / (2.0 * time_cells)],
        dtype=torch.float32,
        device=device,
    )
    kernel_volume = float(torch.prod(2.0 * eps).item())
    sums = {key: 0.0 for key in _PDE_SUM_KEYS}
    rho_max = 0.0

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            centers = _decode_centers(
                start,
                stop,
                spatial_cells=spatial_cells,
                time_cells=time_cells,
                device=device,
            )
            if case.equation == "Heat":
                residual, rho, diagnostics = heat_hyper_taylor_rect_batch(
                    model=model,
                    y=centers,
                    eps_vec=eps,
                    alpha=0.1,
                    n_taylor=None,
                )
            else:
                residual, rho, diagnostics = wave_hyper_taylor_rect_batch(
                    model=model,
                    y=centers,
                    eps_vec=eps,
                    c2=1.0,
                    n_taylor=None,
                )
            tensors = {
                "bound_l2_squared": diagnostics[
                    "hyper_taylor_centered_moment_best_bound_l2sq"
                ],
                "moment_minkowski_l2_squared": diagnostics[
                    "hyper_taylor_centered_moment_bound_l2sq"
                ],
                "moment_cross_l2_squared": diagnostics[
                    "hyper_taylor_centered_moment_cross_bound_l2sq"
                ],
                "moment_remainder_l2_squared": diagnostics[
                    "hyper_taylor_centered_moment_remainder_l2sq"
                ],
                "moment_cross_integral": diagnostics[
                    "hyper_taylor_centered_moment_cross_integral"
                ],
                "affine_l2_squared": diagnostics["hyper_taylor_affine_l2sq"],
                "midpoint_l2_squared": residual.square() * kernel_volume,
                "rho": rho,
                "rho_hyper": diagnostics["hyper_taylor_rho_hyper"],
                "rho_boundary": diagnostics["hyper_taylor_rho_boundary"],
            }
            for name, tensor in tensors.items():
                if not bool(torch.isfinite(tensor).all().item()):
                    raise FloatingPointError(f"non-finite {name}")
                if not bool((tensor >= 0).all().item()):
                    raise FloatingPointError(f"negative {name}")
                sums[name] += float(tensor.to(torch.float64).sum().item())
            rho_max = max(rho_max, float(rho.max().item()))
            if progress is not None:
                progress(stop, total)

    if not (
        sums["moment_minkowski_l2_squared"] >= sums["bound_l2_squared"]
        and sums["moment_cross_l2_squared"] >= sums["bound_l2_squared"]
        and sums["bound_l2_squared"] >= sums["affine_l2_squared"]
        >= sums["midpoint_l2_squared"]
        >= 0.0
    ):
        raise FloatingPointError("invalid PDE bound ordering")

    result: dict[str, object] = {
        "sums": sums,
        "l2_squared_bound": sums["bound_l2_squared"],
        "l2_bound": math.sqrt(sums["bound_l2_squared"]),
        "moment_minkowski_l2_squared": sums["moment_minkowski_l2_squared"],
        "moment_minkowski_l2": math.sqrt(sums["moment_minkowski_l2_squared"]),
        "moment_cross_l2_squared": sums["moment_cross_l2_squared"],
        "moment_cross_l2": math.sqrt(sums["moment_cross_l2_squared"]),
        "moment_remainder_l2_squared": sums["moment_remainder_l2_squared"],
        "moment_remainder_l2": math.sqrt(sums["moment_remainder_l2_squared"]),
        "moment_cross_integral": sums["moment_cross_integral"],
        "affine_l2_squared": sums["affine_l2_squared"],
        "affine_l2": math.sqrt(sums["affine_l2_squared"]),
        "midpoint_l2_squared": sums["midpoint_l2_squared"],
        "midpoint_l2": math.sqrt(sums["midpoint_l2_squared"]),
        "bound_to_affine_ratio": math.sqrt(
            sums["bound_l2_squared"] / sums["affine_l2_squared"]
        ),
        "bound_to_midpoint_ratio": math.sqrt(
            sums["bound_l2_squared"] / sums["midpoint_l2_squared"]
        ),
        "cell_radii": [float(eps[0].item()), float(eps[1].item())],
        "num_boxes": total,
        "processed_boxes": total,
        "diagnostics": {
            "rho_sum": sums["rho"],
            "rho_hyper_sum": sums["rho_hyper"],
            "rho_boundary_sum": sums["rho_boundary"],
            "mean_rho": sums["rho"] / total,
            "mean_rho_hyper": sums["rho_hyper"] / total,
            "mean_rho_boundary": sums["rho_boundary"] / total,
            "max_rho": rho_max,
        },
    }
    return result


def _select_device(requested: str | None, *, default: str) -> torch.device:
    raw = default if requested is None else requested
    selected = torch.device(raw)
    if selected.type == "cpu":
        if selected.index is not None:
            raise ValueError("CPU execution must use the canonical device name 'cpu'")
        return selected
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return selected
    raise ValueError("fixed paper cases support only CPU or CUDA execution")


def _canonical_device_name(device: torch.device) -> str:
    if device.type == "cpu":
        return "cpu"
    return f"cuda:{device.index}" if device.index is not None else "cuda"


def reproduce(
    case_name: str,
    *,
    root: Path | None = None,
    device: str | None = None,
    batch_size: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Recompute one fixed paper case from its hash-pinned checkpoint.

    Both initial-condition and PDE cases prefer CUDA when available,
    because their retained authority was generated on CUDA.  The returned
    JSON-shaped object deliberately contains only identity, execution,
    and semantically compared numerical fields.
    """

    try:
        case = CASES[case_name]
    except KeyError as error:
        raise ValueError(f"unknown reproduction case: {case_name}") from error
    checkout = repository_root(root)
    if case.quantity == "pde":
        default = "cuda" if torch.cuda.is_available() else "cpu"
        selected = _select_device(device, default=default)
        if batch_size is not None and batch_size != 4096:
            raise ValueError("PDE paper cases require the recorded batch_size=4096")
        selected_batch = 4096
    else:
        default = "cuda" if torch.cuda.is_available() else "cpu"
        selected = _select_device(device, default=default)
        selected_batch = 65536 if batch_size is None else batch_size
    if (
        not isinstance(selected_batch, int)
        or isinstance(selected_batch, bool)
        or selected_batch <= 0
    ):
        raise ValueError("batch_size must be positive")
    model = _checked_model(case, checkout, selected)
    if case.quantity == "pde" and case.d > 1:
        try:
            from .pde_multidim import reproduce_pde_multidim
        except ImportError as error:  # fail closed, never a silent fallback
            raise RuntimeError(
                "multi-dimensional PDE closure is not integrated in this checkout"
            ) from error
        numerical = reproduce_pde_multidim(case.name, model, selected_batch, progress)
    else:
        numerical = (
            _pde_result(case, model, selected_batch, progress)
            if case.quantity == "pde"
            else _initial_result(case, model, selected_batch)
        )
    return {
        "schema_version": 1,
        "kind": "netbounds_paper_checkpoint_reproduction",
        "case": case.name,
        "authority": case.authority,
        "identity": {
            "equation": case.equation,
            "quantity": case.quantity,
            "quadrature_rule": case.rule.upper(),
            "architecture": {"layers": case.L, "width": case.width},
            "checkpoint": case.checkpoint,
            "checkpoint_sha256": case.checkpoint_sha256,
            "dtype": "float32",
            "grid": list(case.grid),
        },
        "execution": {
            "device": _canonical_device_name(selected),
            "batch_size": selected_batch,
            "accumulation_dtype": "float64" if case.quantity == "pde" else "float32",
        },
        "numerical": numerical,
    }
