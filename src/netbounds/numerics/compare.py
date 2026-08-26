"""Field-aware comparison of all fixed recomputations with retained authority."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

from netbounds.data import load_catalog, read_hashed_json, repository_root
from netbounds.tables import load_initial_pairs, load_pde_rows

from .cases import CASES, Case, case_names


CUDA_INITIAL_TOLERANCE = {"rtol": 5e-5, "atol": 5e-12}
CPU_INITIAL_TOLERANCE = {"rtol": 1.5e-4, "atol": 5e-12}
PDE_CUDA_TOLERANCE = {"rtol": 5e-7, "atol": 5e-14}
PDE_CPU_TOLERANCE = {"rtol": 5e-5, "atol": 5e-12}

_INITIAL_NUMERICAL_FIELDS = (
    "l2_bound",
    "l2_squared_bound",
    "midpoint_l2",
    "midpoint_l2_squared",
    "bound_to_midpoint_ratio",
    "cell_volume",
    "eps",
    "num_boxes",
)

_PDE_SUM_FIELDS = (
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

_PDE_NUMERICAL_FIELDS = (
    "sums",
    "l2_squared_bound",
    "l2_bound",
    "moment_minkowski_l2_squared",
    "moment_minkowski_l2",
    "moment_cross_l2_squared",
    "moment_cross_l2",
    "moment_remainder_l2_squared",
    "moment_remainder_l2",
    "moment_cross_integral",
    "affine_l2_squared",
    "affine_l2",
    "midpoint_l2_squared",
    "midpoint_l2",
    "bound_to_affine_ratio",
    "bound_to_midpoint_ratio",
    "cell_radii",
    "num_boxes",
    "processed_boxes",
    "diagnostics",
)

_PDE_DIAGNOSTIC_FIELDS = (
    "rho_sum",
    "rho_hyper_sum",
    "rho_boundary_sum",
    "mean_rho",
    "mean_rho_hyper",
    "mean_rho_boundary",
    "max_rho",
)

_PDE_ADAPTIVE_NUMERICAL_FIELDS = (
    "sums",
    "l2_squared_bound",
    "l2_bound",
    "moment_minkowski_l2_squared",
    "moment_minkowski_l2",
    "moment_cross_l2_squared",
    "moment_cross_l2",
    "moment_remainder_l2_squared",
    "moment_remainder_l2",
    "moment_cross_integral",
    "affine_l2_squared",
    "affine_l2",
    "midpoint_l2_squared",
    "midpoint_l2",
    "bound_to_affine_ratio",
    "bound_to_midpoint_ratio",
    "num_base_roots",
    "processed_base_roots",
    "root_block_size",
    "max_depth",
    "adaptive_leaf_count",
    "leaf_counts_by_depth",
    "split_counts_by_depth",
    "kernel_cells_evaluated",
    "partition_units_at_max_depth",
    "diagnostics",
)

_PDE_ADAPTIVE_DIAGNOSTIC_FIELDS = (
    "rho_sum",
    "rho_hyper_sum",
    "rho_boundary_sum",
    "max_rho",
)


def _catalog_entries(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        if {"path", "sha256", "size"} <= value.keys():
            yield value
        for child in value.values():
            yield from _catalog_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from _catalog_entries(child)


def _authority(root: Path, relative: str) -> dict[str, Any]:
    matches = [
        entry
        for entry in _catalog_entries(load_catalog(root))
        if entry.get("path") == relative
    ]
    if not matches:
        raise ValueError(f"catalog has no entry for {relative}")
    declarations = {(entry.get("sha256"), entry.get("size")) for entry in matches}
    if len(declarations) != 1:
        raise ValueError(f"catalog has conflicting entries for {relative}")
    _, payload = read_hashed_json(root, matches[0])
    return payload


def _validate_authority_case(root: Path, catalog: Mapping[str, Any], case: Case) -> None:
    """Apply the table authority validators before numerical comparison."""

    if case.quantity == "pde":
        matches = [
            row
            for row in load_pde_rows(root, catalog)
            if row.path.relative_to(root).as_posix() == case.authority
        ]
    else:
        matches = [
            pair
            for pair in load_initial_pairs(root, catalog)
            if (
                (pair.q0_path if case.rule == "q0" else pair.q1_path)
                .relative_to(root)
                .as_posix()
                == case.authority
            )
        ]
    if len(matches) != 1:
        raise ValueError(f"authority validation did not resolve exactly one row for {case.name}")


def _get(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"missing semantic field: {path}")
        current = current[key]
    return current


def _leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _leaves(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _leaves(child, f"{prefix}.{index}")
    else:
        yield prefix, value


def _authority_value(authority: Mapping[str, Any], path: str) -> Any:
    current: Any = authority
    for component in path.split("."):
        if component.isdigit() and isinstance(current, list):
            current = current[int(component)]
        elif isinstance(current, Mapping) and component in current:
            current = current[component]
        else:
            raise ValueError(f"authority is missing numerical field: {path}")
    return current


def _require_exact_keys(value: Mapping[str, Any], expected: tuple[str, ...], name: str) -> None:
    actual = set(value)
    required = set(expected)
    if actual != required:
        raise ValueError(
            f"{name} fields differ: missing={sorted(required - actual)}, "
            f"unknown={sorted(actual - required)}"
        )


def _require_finite_number(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite numeric scalar")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _validate_reproduction_shape(
    reproduction: Mapping[str, Any], case: Case
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    _require_exact_keys(
        reproduction,
        ("schema_version", "kind", "case", "authority", "identity", "execution", "numerical"),
        "reproduction",
    )
    identity = reproduction["identity"]
    execution = reproduction["execution"]
    numerical = reproduction["numerical"]
    if not all(isinstance(item, Mapping) for item in (identity, execution, numerical)):
        raise ValueError("reproduction identity, execution, and numerical fields must be objects")
    assert isinstance(identity, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(numerical, Mapping)
    _require_exact_keys(
        identity,
        ("equation", "quantity", "quadrature_rule", "architecture", "checkpoint", "checkpoint_sha256", "dtype", "grid"),
        "identity",
    )
    _require_exact_keys(execution, ("device", "batch_size", "accumulation_dtype"), "execution")
    if not isinstance(execution["device"], str):
        raise ValueError("execution.device must be a string")
    if isinstance(execution["batch_size"], bool) or not isinstance(execution["batch_size"], int) or execution["batch_size"] <= 0:
        raise ValueError("execution.batch_size must be a positive integer")
    expected_accumulation = "float64" if case.quantity == "pde" else "float32"
    if execution["accumulation_dtype"] != expected_accumulation:
        raise ValueError(f"execution.accumulation_dtype must be {expected_accumulation}")
    if case.quantity == "pde" and execution["batch_size"] != case.batch_size:
        raise ValueError(
            f"PDE execution.batch_size must be the recorded value {case.batch_size}"
        )

    if case.quantity != "pde":
        _require_exact_keys(numerical, _INITIAL_NUMERICAL_FIELDS, "initial numerical")
        for field in _INITIAL_NUMERICAL_FIELDS:
            if field == "num_boxes":
                expected_boxes = case.cells_per_dim**case.d
                if numerical[field] != expected_boxes or isinstance(numerical[field], bool):
                    raise ValueError(f"initial numerical.num_boxes must be {expected_boxes}")
            else:
                _require_finite_number(numerical[field], f"initial numerical.{field}")
        return identity, execution, numerical

    if case.name == "heat3-pde-q1":
        _require_exact_keys(numerical, _PDE_ADAPTIVE_NUMERICAL_FIELDS, "PDE adaptive numerical")
        sums = numerical["sums"]
        diagnostics = numerical["diagnostics"]
        if not isinstance(sums, Mapping) or not isinstance(diagnostics, Mapping):
            raise ValueError("PDE sums and diagnostics must be objects")
        _require_exact_keys(sums, _PDE_SUM_FIELDS, "PDE numerical.sums")
        _require_exact_keys(
            diagnostics,
            _PDE_ADAPTIVE_DIAGNOSTIC_FIELDS,
            "PDE adaptive numerical.diagnostics",
        )
        for field, value in sums.items():
            _require_finite_number(value, f"PDE numerical.sums.{field}")
        for field, value in diagnostics.items():
            _require_finite_number(value, f"PDE numerical.diagnostics.{field}")
        for field in (
            "l2_squared_bound",
            "l2_bound",
            "moment_minkowski_l2_squared",
            "moment_minkowski_l2",
            "moment_cross_l2_squared",
            "moment_cross_l2",
            "moment_remainder_l2_squared",
            "moment_remainder_l2",
            "moment_cross_integral",
            "affine_l2_squared",
            "affine_l2",
            "midpoint_l2_squared",
            "midpoint_l2",
            "bound_to_affine_ratio",
            "bound_to_midpoint_ratio",
        ):
            _require_finite_number(numerical[field], f"PDE numerical.{field}")
        for field in (
            "num_base_roots",
            "processed_base_roots",
            "root_block_size",
            "max_depth",
            "adaptive_leaf_count",
            "kernel_cells_evaluated",
            "partition_units_at_max_depth",
        ):
            value = numerical[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"PDE numerical.{field} must be a positive integer")
        leaves = numerical["leaf_counts_by_depth"]
        splits = numerical["split_counts_by_depth"]
        if (
            not isinstance(leaves, list)
            or len(leaves) != 3
            or not all(isinstance(value, int) and value >= 0 for value in leaves)
            or not isinstance(splits, list)
            or len(splits) != 2
            or not all(isinstance(value, int) and value >= 0 for value in splits)
        ):
            raise ValueError("PDE adaptive depth counts are malformed")
        return identity, execution, numerical

    _require_exact_keys(numerical, _PDE_NUMERICAL_FIELDS, "PDE numerical")
    sums = numerical["sums"]
    diagnostics = numerical["diagnostics"]
    radii = numerical["cell_radii"]
    if not isinstance(sums, Mapping) or not isinstance(diagnostics, Mapping):
        raise ValueError("PDE sums and diagnostics must be objects")
    _require_exact_keys(sums, _PDE_SUM_FIELDS, "PDE numerical.sums")
    _require_exact_keys(diagnostics, _PDE_DIAGNOSTIC_FIELDS, "PDE numerical.diagnostics")
    if not isinstance(radii, list) or len(radii) != case.d + 1:
        raise ValueError(
            f"PDE numerical.cell_radii must contain exactly {case.d + 1} values"
        )
    for field, value in sums.items():
        _require_finite_number(value, f"PDE numerical.sums.{field}")
    for field, value in diagnostics.items():
        _require_finite_number(value, f"PDE numerical.diagnostics.{field}")
    for index, value in enumerate(radii):
        _require_finite_number(value, f"PDE numerical.cell_radii.{index}")
    for field in set(_PDE_NUMERICAL_FIELDS) - {"sums", "diagnostics", "cell_radii", "num_boxes", "processed_boxes"}:
        _require_finite_number(numerical[field], f"PDE numerical.{field}")
    expected_boxes = 1
    for cells in case.grid:
        expected_boxes *= cells
    for field in ("num_boxes", "processed_boxes"):
        if numerical[field] != expected_boxes or isinstance(numerical[field], bool):
            raise ValueError(f"PDE numerical.{field} must be {expected_boxes}")
    return identity, execution, numerical


def _exact_checks(case_name: str, authority: Mapping[str, Any]) -> dict[str, Any]:
    case = CASES[case_name]
    if case.quantity == "pde":
        checkpoint_field = "checkpoint_sha256" if case.d == 3 else "weights_sha256"
        expected: dict[str, Any] = {
            "equation": case.equation,
            "quadrature_rule": case.rule.upper(),
            checkpoint_field: case.checkpoint_sha256,
            "dimension": case.d,
            "grid": list(case.grid),
            "accumulation_dtype": "float64",
            "ordinary_float32_not_outward_rounded": True,
        }
        if case.name == "heat3-pde-q1":
            expected.update(
                {
                    "num_base_roots": 500_000_000,
                    "processed_base_roots": 500_000_000,
                    "all_base_roots_enumerated": True,
                    "all_adaptive_leaves_enumerated": True,
                    "max_depth": 2,
                }
            )
        else:
            expected_boxes = 1
            for cells in case.grid:
                expected_boxes *= cells
            expected.update(
                {
                    "num_boxes": expected_boxes,
                    "processed_boxes": expected_boxes,
                    "all_cells_enumerated": True,
                }
            )
    else:
        expected = {
            "equation": case.equation,
            "quadrature_rule": case.rule.upper(),
            "weights_sha256": case.checkpoint_sha256,
            "num_boxes": case.cells_per_dim**case.d,
            "grid_dimension": case.d,
            "cells_per_dim": case.cells_per_dim,
            "dtype": "float32",
            "residual_type": (
                "initial_displacement_gradient"
                if case.quantity == "gradient"
                else f"initial_{case.quantity}"
            ),
            "nn_envelope_geometry": "spatial_trace_t0",
            "nn_envelope_time_eps": 0.0,
        }
    failures = {
        path: {"expected": value, "actual": _get(authority, path)}
        for path, value in expected.items()
        if _get(authority, path) != value
    }
    return failures


def compare_reproduction(
    reproduction: Mapping[str, Any], *, root: Path | None = None
) -> dict[str, Any]:
    """Compare every retained result leaf represented by a reproduction payload."""

    if reproduction.get("schema_version") != 1 or reproduction.get("kind") != (
        "netbounds_paper_checkpoint_reproduction"
    ):
        raise ValueError("unsupported reproduction payload")
    case_name = str(reproduction.get("case", ""))
    if case_name not in CASES:
        raise ValueError(f"unknown reproduction case: {case_name}")
    if case_name not in case_names():
        raise ValueError(
            f"reproduction case is declared but not yet public: {case_name}"
        )
    case = CASES[case_name]
    if reproduction.get("authority") != case.authority:
        raise ValueError("reproduction authority path does not match the case registry")
    identity, execution, numerical = _validate_reproduction_shape(reproduction, case)
    checkout = repository_root(root)
    catalog = load_catalog(checkout)
    _validate_authority_case(checkout, catalog, case)
    authority = _authority(checkout, case.authority)

    expected_identity = {
        "equation": case.equation,
        "quantity": case.quantity,
        "quadrature_rule": case.rule.upper(),
        "architecture": {"layers": case.L, "width": case.width},
        "checkpoint": case.checkpoint,
        "checkpoint_sha256": case.checkpoint_sha256,
        "dtype": "float32",
        "grid": list(case.grid),
    }
    identity_failures = {
        key: {"expected": value, "actual": identity.get(key)}
        for key, value in expected_identity.items()
        if identity.get(key) != value
    }
    authority_failures = _exact_checks(case_name, authority)

    device = str(execution.get("device", ""))
    if case.quantity == "pde":
        if case.d == 3 and not device.startswith("cuda"):
            raise ValueError("3D PDE comparison requires CUDA execution")
        if device.startswith("cuda"):
            tolerance = PDE_CUDA_TOLERANCE
            tolerance_class = "pde_cuda_float32_float64_accumulation"
        elif device == "cpu":
            tolerance = PDE_CPU_TOLERANCE
            tolerance_class = "pde_cpu_portability"
        else:
            raise ValueError(f"unsupported PDE device: {device}")
        compared = list(_leaves(numerical))
    else:
        if device.startswith("cuda"):
            tolerance = CUDA_INITIAL_TOLERANCE
            tolerance_class = "initial_cuda_float32"
        elif device == "cpu":
            if case.d == 3:
                raise ValueError("3D initial-data comparison requires CUDA execution")
            tolerance = (
                {"rtol": 3e-4, "atol": 5e-12}
                if case.d == 2
                else CPU_INITIAL_TOLERANCE
            )
            tolerance_class = f"initial_cpu_portability_d{case.d}"
        else:
            raise ValueError(f"unsupported initial-condition device: {device}")
        compared = [(field, numerical.get(field)) for field in _INITIAL_NUMERICAL_FIELDS]

    numerical_failures: dict[str, dict[str, Any]] = {}
    max_relative_error = 0.0
    for path, actual in compared:
        expected = _authority_value(authority, path)
        if isinstance(expected, bool) or isinstance(actual, bool):
            passed = actual is expected
            relative = 0.0 if passed else math.inf
        elif isinstance(expected, int) and isinstance(actual, int):
            passed = actual == expected
            relative = 0.0 if passed else math.inf
        elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            actual_float = float(actual)
            expected_float = float(expected)
            passed = math.isfinite(actual_float) and math.isclose(
                actual_float,
                expected_float,
                rel_tol=tolerance["rtol"],
                abs_tol=tolerance["atol"],
            )
            denominator = max(abs(expected_float), tolerance["atol"])
            relative = abs(actual_float - expected_float) / denominator
        else:
            passed = actual == expected
            relative = 0.0 if passed else math.inf
        max_relative_error = max(max_relative_error, relative)
        if not passed:
            numerical_failures[path] = {
                "expected": expected,
                "actual": actual,
                "relative_error": relative,
            }

    failures = {
        "identity": identity_failures,
        "authority": authority_failures,
        "numerical": numerical_failures,
    }
    passed = not any(failures.values())
    return {
        "schema_version": 1,
        "kind": "netbounds_paper_semantic_comparison",
        "case": case_name,
        "passed": passed,
        "authority": case.authority,
        "tolerance_class": tolerance_class,
        "rtol": tolerance["rtol"],
        "atol": tolerance["atol"],
        "compared_numerical_leaves": len(compared),
        "max_relative_error": max_relative_error,
        "failures": failures,
    }


def compare_file(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    """Load one reproduction JSON and compare it fail-closed."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return compare_reproduction(value, root=root)
