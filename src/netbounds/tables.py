"""Validate retained artifacts and render the five paper tables.

This module intentionally contains only the policy needed by the five current
numerical tables. It has no training, checkpoint loading, or GPU dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Mapping

from .data import VerificationError, read_hashed_json, safe_path


INITIAL_TABLE_PATHS = {
    "displacement": "paper/tables/initial_displacement_residuals_q0_q1_master.tex",
    "gradient": "paper/tables/initial_displacement_gradient_residuals_q0_q1_master.tex",
    "velocity": "paper/tables/initial_velocity_residuals_q0_q1_master.tex",
}
PDE_TABLE_PATH = "paper/tables/pde_residuals_q1_master.tex"
ENERGY_TABLE_PATH = "paper/tables/energy_estimate_q1_master.tex"

INITIAL_CAPTIONS = {
    "displacement": (
        "\\caption{Upper bounds for the initial displacement residual norm, "
        "as constructed in \\cref{sec:initial-displacement-quad}.}"
    ),
    "gradient": (
        "\\caption{Upper bounds for the initial displacement gradient residual norm, "
        "as constructed in \\cref{sec:initial-displacement-gradient-quad}.}"
    ),
    "velocity": (
        "\\caption{Upper bounds for the initial-velocity residual norm, as constructed "
        "in \\cref{sec:initial-velocity-quad}.}"
    ),
}
PDE_CAPTION = (
    "\\caption{Upper bounds for the PDE residual norms, as constructed in "
    "\\cref{sec:pde-residual-quad}.}"
)
ENERGY_CAPTION = (
    "\\caption{Energy-estimate bounds obtained by inserting the bounds from "
    "\\cref{tab:initial-displacement-residuals-q0-q1,"
    "tab:initial-displacement-gradient-residuals-q0-q1,"
    "tab:wave-initial-velocity-residuals-q0-q1,tab:pde-residuals-q1-master} into "
    "\\cref{eq:heat-residual-estimate-l2,eq:heat-residual-estimate-h1,"
    "eq:wave-residual-estimate}.}"
)

TASK_ORDER = (
    "heat1_displacement", "heat2_displacement", "heat3_displacement",
    "wave1_displacement", "wave2_displacement", "wave3_displacement",
    "heat1_gradient", "heat2_gradient", "heat3_gradient",
    "wave1_gradient", "wave2_gradient", "wave3_gradient",
    "wave1_velocity", "wave2_velocity", "wave3_velocity",
)
FAMILY_COUNTS = {"displacement": 6, "gradient": 6, "velocity": 3}
FAMILY_SCHEMA = {
    "displacement": {
        "residual_type": "initial_displacement",
        "selector": "initial_displacement_quadrature",
        "q0_method": "ima_initial_q0_linear_remainder_moment_v2",
        "q1_method": "ima_initial_q1_quadratic_remainder_moment_v2",
        "q0_order": 1,
        "q1_order": 2,
        "q0_cell": "constant",
        "q1_cell": "affine_taylor",
        "q0_remainder": "coordinatewise_linear_derivative_envelope",
        "q1_remainder": "quadratic_hessian_envelope",
        "q0_formula": "eq:initial-displacement-first-coefficients",
        "q1_formula": "eq:initial-displacement-second-coefficients",
    },
    "gradient": {
        "residual_type": "initial_displacement_gradient",
        "selector": "initial_gradient_quadrature",
        "q0_method": "ima_current_paper_initial_gradient_q0_centered_moment_v2",
        "q1_method": "ima_current_paper_initial_gradient_q1_centered_moment_v2",
        "q0_order": 2,
        "q1_order": 3,
        "q0_cell": "constant",
        "q1_cell": "componentwise_affine_taylor",
        "q0_remainder": "coordinatewise_linear_derivative_envelope",
        "q1_remainder": "componentwise_quadratic_hessian_envelope",
        "q0_formula": "eq:appendix-initial-gradient-first-coefficients",
        "q1_formula": "eq:appendix-initial-gradient-second-coefficients",
    },
    "velocity": {
        "residual_type": "initial_velocity",
        "selector": "initial_velocity_quadrature",
        "q0_method": "ima_current_paper_initial_velocity_q0_centered_moment_v2",
        "q1_method": "ima_current_paper_initial_velocity_q1_centered_moment_v2",
        "q0_order": 2,
        "q1_order": 3,
        "q0_cell": "constant",
        "q1_cell": "affine_taylor",
        "q0_remainder": "coordinatewise_linear_derivative_envelope",
        "q1_remainder": "quadratic_hessian_envelope",
        "q0_formula": "eq:appendix-initial-velocity-first-coefficients",
        "q1_formula": "eq:appendix-initial-velocity-second-coefficients",
    },
}

@dataclass(frozen=True)
class InitialPair:
    policy: Mapping[str, Any]
    q0: Mapping[str, Any]
    q1: Mapping[str, Any]
    q0_path: Path
    q1_path: Path

    @property
    def key(self) -> tuple[str, str, int, int, int]:
        return (
            str(self.policy["family"]),
            str(self.policy["equation"]),
            int(self.policy["dimension"]),
            int(self.policy["layers"]),
            int(self.policy["width"]),
        )

    @property
    def checkpoint_sha256(self) -> str:
        return str(self.policy["checkpoint_sha256"])


@dataclass(frozen=True)
class PdeRow:
    policy: Mapping[str, Any]
    data: Mapping[str, Any]
    path: Path
    affine_l2: float
    cross_l2: float
    selected_l2: float

    @property
    def key(self) -> tuple[str, int, int, int]:
        return (
            str(self.policy["equation"]),
            int(self.policy["dimension"]),
            int(self.policy["layers"]),
            int(self.policy["width"]),
        )


@dataclass(frozen=True)
class EnergyRow:
    equation: str
    dimension: int
    layers: int
    width: int
    lhs: str
    displacement: Decimal
    velocity: Decimal | None
    pde: Decimal
    total: Decimal


def _require(condition: bool, where: Path | str, message: str) -> None:
    if not condition:
        raise VerificationError(f"{where}: {message}")


def _finite_decimal(data: Mapping[str, Any], key: str, where: Path) -> Decimal:
    if key not in data:
        raise VerificationError(f"{where}: missing {key}")
    value = data[key]
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    if not value.is_finite() or value < 0:
        raise VerificationError(f"{where}: invalid {key}={value}")
    return value


def _finite_float(data: Mapping[str, Any], key: str, where: Path) -> float:
    if key not in data:
        raise VerificationError(f"{where}: missing {key}")
    value = float(data[key])
    if not math.isfinite(value) or value < 0:
        raise VerificationError(f"{where}: invalid {key}={value}")
    return value


def _validate_initial_artifact(
    data: Mapping[str, Any], policy: Mapping[str, Any], rule: str, path: Path
) -> None:
    family = str(policy["family"])
    schema = FAMILY_SCHEMA[family]
    expected_rule = rule.upper()
    checks = {
        "certified flag": data.get("certified") is True,
        "equation": data.get("equation") == policy["equation"],
        "dimension": int(data.get("grid_dimension", -1)) == int(policy["dimension"]),
        "quadrature rule": data.get("quadrature_rule") == expected_rule,
        "method": data.get("method_identifier") == schema[f"{rule}_method"],
        "cell model": data.get("cell_model") == schema[f"{rule}_cell"],
        "remainder": data.get("remainder") == schema[f"{rule}_remainder"],
        "selector": data.get(str(schema["selector"]))
        == ("moment_standard" if rule == "q0" else "moment_taylor"),
        "residual type": data.get("residual_type") == schema["residual_type"],
        "derivative order": int(data.get("network_spatial_derivative_order", -1))
        == int(schema[f"{rule}_order"]),
        "fixed-time geometry": data.get("nn_envelope_geometry") == "spatial_trace_t0",
        "zero time radius": Decimal(str(data.get("nn_envelope_time_eps", -1))) == 0,
        "spatial radius": Decimal(str(data.get("eps", -1))) == Decimal("0.001"),
        "cell count": int(data.get("num_boxes", -1)) == int(policy["num_boxes"]),
        "dtype": data.get("dtype") == "float32",
        "device": data.get("device") == "cuda",
        "checkpoint": data.get("weights_sha256") == policy["checkpoint_sha256"],
        "formula": schema[f"{rule}_formula"] in data.get("formula_equations", []),
    }
    if rule == "q0":
        checks["explicit Lipschitz source"] = (
            family != "displacement"
            or data.get("data_derivative_envelope")
            == "separable_sine_global_componentwise"
        )
    if family == "gradient" and rule == "q0":
        checks["component aggregation"] = (
            data.get("component_aggregation")
            == "sum_cell_component_integrals_before_global_sqrt"
        )
    failed = [name for name, valid in checks.items() if not valid]
    _require(not failed, path, "failed checks: " + ", ".join(failed))

    midpoint = _finite_decimal(data, "midpoint_l2", path)
    bound = _finite_decimal(data, "l2_bound", path)
    squared = _finite_decimal(data, "l2_squared_bound", path)
    _require(bound >= midpoint, path, "bound is below midpoint diagnostic")
    _require(
        abs(squared - bound * bound)
        <= max(Decimal("1e-18"), squared * Decimal("5e-7")),
        path,
        "l2_squared_bound is inconsistent",
    )


def load_initial_pairs(root: Path, catalog: Mapping[str, Any]) -> list[InitialPair]:
    rows = catalog.get("initial_rows")
    _require(isinstance(rows, list), "catalog", "initial_rows must be a list")
    _require(
        [row.get("task_id") for row in rows] == list(TASK_ORDER),
        "catalog",
        "initial task matrix is partial or reordered",
    )
    counts = {
        family: sum(row.get("family") == family for row in rows)
        for family in FAMILY_COUNTS
    }
    _require(counts == FAMILY_COUNTS, "catalog", "initial family counts differ from 6/6/3")

    pairs: list[InitialPair] = []
    for policy in rows:
        q0_path, q0 = read_hashed_json(
            root, policy["q0"], parse_float=Decimal
        )
        q1_path, q1 = read_hashed_json(
            root, policy["q1"], parse_float=Decimal
        )
        _validate_initial_artifact(q0, policy, "q0", q0_path)
        _validate_initial_artifact(q1, policy, "q1", q1_path)
        midpoint0 = _finite_decimal(q0, "midpoint_l2", q0_path)
        midpoint1 = _finite_decimal(q1, "midpoint_l2", q1_path)
        tolerance = max(Decimal("1e-10"), abs(midpoint1) * Decimal("2e-5"))
        _require(
            abs(midpoint0 - midpoint1) <= tolerance,
            policy["task_id"],
            "Q0/Q1 midpoint diagnostics differ",
        )
        pairs.append(InitialPair(policy, q0, q1, q0_path, q1_path))
    return pairs


def _configuration(data: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    config = data.get("configuration", data.get("global_configuration"))
    _require(isinstance(config, dict), path, "missing configuration object")
    return config


def _validate_pde_artifact(
    data: Mapping[str, Any], policy: Mapping[str, Any], path: Path
) -> tuple[float, float, float]:
    dimension = int(policy["dimension"])
    grid = tuple(int(value) for value in policy["grid"])
    checkpoint = str(policy["checkpoint_sha256"])
    config = _configuration(data, path)
    architecture = {"layers": int(policy["layers"]), "width": int(policy["width"])}

    checks = {
        "certified flag": data.get("certified") is True,
        "equation": data.get("equation") == policy["equation"],
        "dimension": int(data.get("dimension", -1)) == dimension,
        "kind": data.get("kind") == policy["kind"],
        "method": data.get("method") == policy["method"],
        "grid": tuple(data.get("grid", ())) == grid,
        "architecture": data.get("architecture", config.get("architecture")) == architecture,
        "checkpoint": data.get("weights_sha256", data.get("checkpoint_sha256")) == checkpoint,
        "configuration checkpoint": config.get("weights_sha256", config.get("checkpoint_sha256")) == checkpoint,
        "float32 caveat": data.get("ordinary_float32_not_outward_rounded") is True,
        "float64 accumulation": data.get("accumulation_dtype", config.get("accumulation_dtype")) == "float64",
    }
    failed = [name for name, valid in checks.items() if not valid]
    _require(not failed, path, "failed checks: " + ", ".join(failed))

    midpoint = _finite_float(data, "midpoint_l2", path)
    affine = _finite_float(data, "affine_l2", path)
    selected = _finite_float(data, "l2_bound", path)
    cross = _finite_float(data, "moment_cross_l2", path)
    _require(
        cross >= selected >= affine >= midpoint,
        path,
        "expected cross >= selected >= affine >= midpoint",
    )
    for value_key, squared_key in (
        ("affine_l2", "affine_l2_squared"),
        ("moment_cross_l2", "moment_cross_l2_squared"),
    ):
        value = _finite_float(data, value_key, path)
        squared = _finite_float(data, squared_key, path)
        _require(
            math.isclose(squared, value * value, rel_tol=5e-7, abs_tol=1e-18),
            path,
            f"{squared_key} is inconsistent",
        )

    family = str(policy["family"])
    total_cells = math.prod(grid)
    if family in {"uniform", "wave3_x_refined"}:
        _require(data.get("all_cells_enumerated") is True, path, "cell coverage flag missing")
        _require(int(data.get("num_boxes", -1)) == total_cells, path, "cell count mismatch")
        _require(int(data.get("processed_boxes", -1)) == total_cells, path, "processed count mismatch")
        _require(config.get("dtype") == "float32", path, "execution dtype mismatch")
        _require(config.get("certificate") == policy["certificate"], path, "certificate mismatch")
        _require(config.get("method_identifier") == policy["method_identifier"], path, "method identifier mismatch")
        _require(config.get("rigorous_remainder") == policy["rigorous_remainder"], path, "remainder mismatch")
        radii = tuple(float(value) for value in data.get("cell_radii", ()))
        _require(len(radii) == dimension + 1, path, "cell radii shape mismatch")
        expected_radii = tuple(1.0 / (2.0 * cells) for cells in grid)
        _require(
            all(
                math.isclose(actual, expected, rel_tol=5e-7, abs_tol=1e-12)
                for actual, expected in zip(radii, expected_radii)
            ),
            path,
            "cell radii mismatch",
        )
        if policy["equation"] == "Wave" and dimension == 3:
            acceptance = data.get("acceptance")
            _require(isinstance(acceptance, dict), path, "Wave-3D acceptance missing")
            assert isinstance(acceptance, dict)
            _require(acceptance.get("accepted_as_final") is True, path, "Wave-3D final acceptance missing")
            _require(
                acceptance.get("status")
                == "strict_merge_passed_and_user_accepted_final",
                path,
                "Wave-3D acceptance status mismatch",
            )
            _require(
                config.get("spatial_radius_policy")
                == "coordinatewise_eps_vec_spatial",
                path,
                "Wave-3D spatial radius policy mismatch",
            )
            _require(
                int(data.get("kernel_batch_size", -1)) == 8192,
                path,
                "Wave-3D batch size mismatch",
            )
    elif family == "heat3_adaptive_current_paper":
        _require(data.get("accepted_as_heat3d_moment_pde_residual_bound") is True, path, "acceptance flag missing")
        _require(data.get("all_base_roots_enumerated") is True, path, "base-root coverage missing")
        _require(data.get("all_adaptive_leaves_enumerated") is True, path, "leaf coverage missing")
        _require(int(data.get("num_base_roots", -1)) == total_cells, path, "base-root count mismatch")
        _require(int(data.get("processed_base_roots", -1)) == total_cells, path, "processed root count mismatch")
        leaves = tuple(int(value) for value in data.get("leaf_counts_by_depth", ()))
        splits = tuple(int(value) for value in data.get("split_counts_by_depth", ()))
        children = int(config.get("children_per_split", -1))
        _require(len(leaves) == 3 and len(splits) == 2, path, "adaptive count shape mismatch")
        _require(children == 16, path, "adaptive child count mismatch")
        _require(leaves[0] + splits[0] == total_cells, path, "depth-0 tree identity mismatch")
        _require(leaves[1] + splits[1] == children * splits[0], path, "depth-1 tree identity mismatch")
        _require(leaves[2] == children * splits[1], path, "depth-2 tree identity mismatch")
        _require(config.get("execution_dtype") == "float32", path, "execution dtype mismatch")
        _require(config.get("certificate") == policy["certificate"], path, "certificate mismatch")
        _require(config.get("method_identifier") == policy["method_identifier"], path, "method identifier mismatch")
        _require(config.get("rigorous_remainder") == policy["rigorous_remainder"], path, "remainder mismatch")
        _require(config.get("spatial_radius_policy") == "coordinatewise_eps_vec_spatial", path, "spatial radius policy mismatch")
        _require(int(config.get("root_block_size", -1)) == 2048, path, "root block size mismatch")
        _require(int(config.get("kernel_batch_size", -1)) == 1600, path, "kernel batch size mismatch")
        _require(int(data.get("root_block_size", -1)) == 2048, path, "top-level root block size mismatch")
        _require(int(data.get("kernel_batch_size", -1)) == 1600, path, "top-level kernel batch size mismatch")
        sums = data.get("sums")
        _require(isinstance(sums, dict), path, "adaptive sums missing")
        assert isinstance(sums, dict)
    else:
        raise VerificationError(f"{path}: unknown PDE family {family}")

    if policy["equation"] == "Wave" and dimension == 3:
        _require(float(config.get("lambda_ic_grad", -1)) == 0.0, path, "lambda_ic_grad must be zero")
    return affine, cross, selected


def load_pde_rows(root: Path, catalog: Mapping[str, Any]) -> list[PdeRow]:
    policies = catalog.get("pde_rows")
    _require(isinstance(policies, list) and len(policies) == 6, "catalog", "expected six PDE rows")
    expected = [
        ("Heat", 1, 2, 128), ("Heat", 2, 3, 128), ("Heat", 3, 4, 128),
        ("Wave", 1, 2, 256), ("Wave", 2, 3, 256), ("Wave", 3, 3, 256),
    ]
    actual = [
        (row["equation"], row["dimension"], row["layers"], row["width"])
        for row in policies
    ]
    _require(actual == expected, "catalog", "PDE row matrix is partial or reordered")

    rows: list[PdeRow] = []
    for policy in policies:
        path, data = read_hashed_json(root, policy["artifact"])
        affine, cross, selected = _validate_pde_artifact(data, policy, path)
        rows.append(PdeRow(policy, data, path, affine, cross, selected))
    return rows


def _fmt_initial(value: Decimal) -> str:
    mantissa, exponent = f"{value:.4e}".split("e")
    return f"\\num{{{mantissa}e{int(exponent):+03d}}}"


def render_initial_table(pairs: list[InitialPair], family: str) -> str:
    selected = [pair for pair in pairs if pair.policy["family"] == family]
    _require(len(selected) == FAMILY_COUNTS[family], family, "partial initial table")
    if family == "displacement":
        label = "tab:initial-displacement-residuals-q0-q1"
        heading = (
            "Equation & $(d,L,w)$ & $\\sqrt{Q_{\\mathcal P_\\Omega}^0(e_0)}$ & "
            "$\\mathcal U_{\\mathcal P_\\Omega}^0(e_0)$ & "
            "$\\mathcal U_{\\mathcal P_\\Omega}^1(e_0)$ " + "\\\\"
        )
        begin, tabular_end, wrapper_end = "\\begin{table}[t]", "\\end{tabular}", "}%"
    elif family == "gradient":
        label = "tab:initial-displacement-gradient-residuals-q0-q1"
        heading = (
            "Equation & $(d,L,w)$ & "
            "$\\left(\\sum_j Q_{\\mathcal P_\\Omega}^0(\\partial_j e_0)\\right)^{1/2}$ & "
            "$\\mathcal U_{\\mathcal P_\\Omega}^0(\\nabla e_0)$ & "
            "$\\mathcal U_{\\mathcal P_\\Omega}^1(\\nabla e_0)$ " + "\\\\"
        )
        begin, tabular_end, wrapper_end = "\\begin{table}", "\\end{tabular}%", "}"
    elif family == "velocity":
        label = "tab:wave-initial-velocity-residuals-q0-q1"
        heading = (
            "Equation & $(d,L,w)$ & $\\sqrt{Q_{\\mathcal P_\\Omega}^0(e_{\\mathrm v})}$ & "
            "$\\mathcal U_{\\mathcal P_\\Omega}^0(e_{\\mathrm v})$ & "
            "$\\mathcal U_{\\mathcal P_\\Omega}^1(e_{\\mathrm v})$ " + "\\\\"
        )
        begin, tabular_end, wrapper_end = "\\begin{table}[t]", "\\end{tabular}", "}%"
    else:
        raise VerificationError(f"unknown initial family: {family}")

    lines = [
        "% Generated by scripts/render_ima_initial_moment_master_tables.py; do not edit by hand.",
        begin,
        "\\centering",
        INITIAL_CAPTIONS[family],
        f"\\label{{{label}}}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        heading,
        "\\midrule",
    ]
    for index, pair in enumerate(selected):
        if index == 3 and family != "velocity":
            lines.append("\\midrule")
        policy = pair.policy
        midpoint = _finite_decimal(pair.q0, "midpoint_l2", pair.q0_path)
        q0_bound = _finite_decimal(pair.q0, "l2_bound", pair.q0_path)
        q1_bound = _finite_decimal(pair.q1, "l2_bound", pair.q1_path)
        lines.append(
            f"{policy['equation']} & $({policy['dimension']},{policy['layers']},{policy['width']})$ & "
            f"{_fmt_initial(midpoint)} & {_fmt_initial(q0_bound)} & {_fmt_initial(q1_bound)} \\\\"
        )
    lines.extend(["\\bottomrule", tabular_end, wrapper_end, "\\end{table}", ""])
    return "\n".join(lines)


def render_pde_table(rows: list[PdeRow]) -> str:
    _require(len(rows) == 6, PDE_TABLE_PATH, "partial PDE table")
    lines = [
        "% Generated by scripts/render_ima_pde_q1_master_table.py; do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        PDE_CAPTION,
        "\\label{tab:pde-residuals-q1-master}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcrr}",
        "\\toprule",
        (
            "Equation & $(d,L,w)$ & Grid & "
            "$\\sqrt{Q_{\\mathcal T_{\\mathrm{fin}}}^1(R_{\\mathfrak p}(Bf))}$ & "
            "$\\mathcal U_{\\mathcal T_{\\mathrm{fin}}}^1(R_{\\mathfrak p}(Bf))$ "
            + "\\\\"
        ),
        "\\midrule",
    ]
    for index, row in enumerate(rows):
        if index == 3:
            lines.append("\\midrule")
        policy = row.policy
        lines.append(
            f"{policy['equation']} & $({policy['dimension']},{policy['layers']},{policy['width']})$ & "
            f"{policy['grid_tex']} & \\num{{{row.affine_l2:.4e}}} & "
            f"\\num{{{row.cross_l2:.4e}}} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}%", "\\end{table}", ""])
    return "\n".join(lines)


PI_LOWER = Decimal("3.14159265358979323846264338327950288419716939937510")
SQRT10_LOWER = Decimal("3.1622776601683793319988935444327185337195551393251")
SQRT10_UPPER = Decimal("3.1622776601683793319988935444327185337195551393252")


def _sqrt_lower(value: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 70
        context.rounding = ROUND_FLOOR
        return value.sqrt()


def _sqrt_upper(value: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = 70
        context.rounding = ROUND_CEILING
        return value.sqrt()


def _ceil_sig(value: Decimal, digits: int = 5) -> Decimal:
    if value < 0:
        raise VerificationError(f"cannot upward-round negative contribution {value}")
    if not value:
        return Decimal(0)
    quantum = Decimal(1).scaleb(value.adjusted() - digits + 1)
    return value.quantize(quantum, rounding=ROUND_CEILING)


def _c_omega_upper(dimension: int) -> Decimal:
    return _sqrt_upper(
        Decimal(1)
        + Decimal(1) / (Decimal(dimension) * PI_LOWER**2)
        + Decimal(1) / (Decimal(dimension) ** 2 * PI_LOWER**4)
    )


def _heat_displacement_coefficient() -> Decimal:
    return Decimal(1) + SQRT10_UPPER + Decimal(1) / SQRT10_LOWER


def _heat_gradient_coefficient(dimension: int) -> Decimal:
    return Decimal(1) + Decimal(1) / SQRT10_LOWER + SQRT10_UPPER * _c_omega_upper(dimension)


def _heat_pde_coefficient(dimension: int, lhs: str) -> Decimal:
    if lhs == "L2":
        return (Decimal(12) + SQRT10_UPPER) / (PI_LOWER * _sqrt_lower(Decimal(dimension)))
    if lhs == "H1":
        return Decimal(2) + SQRT10_UPPER + Decimal(10) * _c_omega_upper(dimension)
    raise VerificationError(f"unknown Heat norm {lhs}")


def _wave_pde_coefficient(dimension: int) -> Decimal:
    return Decimal(3) + Decimal(1) / (PI_LOWER * _sqrt_lower(Decimal(dimension)))


def _q1_bound(pair: InitialPair) -> Decimal:
    return _finite_decimal(pair.q1, "l2_bound", pair.q1_path)


def build_energy_rows(pde_rows: list[PdeRow], initial_pairs: list[InitialPair]) -> list[EnergyRow]:
    initial = {pair.key: pair for pair in initial_pairs}
    _require(len(initial) == 15, ENERGY_TABLE_PATH, "expected 15 initial pairs")
    rows: list[EnergyRow] = []
    for pde_row in pde_rows:
        equation, dimension, layers, width = pde_row.key
        pde_value = Decimal(str(pde_row.cross_l2))
        checkpoint = str(pde_row.policy["checkpoint_sha256"])
        if equation == "Heat":
            displacement = initial[("displacement", equation, dimension, layers, width)]
            gradient = initial[("gradient", equation, dimension, layers, width)]
            _require(displacement.checkpoint_sha256 == checkpoint, pde_row.path, "displacement checkpoint mismatch")
            _require(gradient.checkpoint_sha256 == checkpoint, pde_row.path, "gradient checkpoint mismatch")
            displacement_component = _ceil_sig(_heat_displacement_coefficient() * _q1_bound(displacement))
            gradient_component = _ceil_sig(_heat_gradient_coefficient(dimension) * _q1_bound(gradient))
            for lhs, initial_component in (("L2", displacement_component), ("H1", gradient_component)):
                pde_component = _ceil_sig(_heat_pde_coefficient(dimension, lhs) * pde_value)
                rows.append(EnergyRow(equation, dimension, layers, width, lhs, initial_component, None, pde_component, initial_component + pde_component))
        else:
            gradient = initial[("gradient", equation, dimension, layers, width)]
            velocity = initial[("velocity", equation, dimension, layers, width)]
            _require(gradient.checkpoint_sha256 == checkpoint, pde_row.path, "gradient checkpoint mismatch")
            _require(velocity.checkpoint_sha256 == checkpoint, pde_row.path, "velocity checkpoint mismatch")
            displacement_component = _ceil_sig(Decimal(3) * _q1_bound(gradient))
            velocity_component = _ceil_sig(Decimal(3) * _q1_bound(velocity))
            pde_component = _ceil_sig(_wave_pde_coefficient(dimension) * pde_value)
            rows.append(EnergyRow(equation, dimension, layers, width, "Wave", displacement_component, velocity_component, pde_component, displacement_component + velocity_component + pde_component))
    _require(len(rows) == 9, ENERGY_TABLE_PATH, "expected nine energy rows")
    return rows


def _tex_num(value: Decimal, *, fixed_sig: bool) -> str:
    if not value:
        return "\\num{0}"
    exponent = value.adjusted()
    mantissa = value.scaleb(-exponent)
    text = f"{mantissa:.4f}" if fixed_sig else format(mantissa.normalize(), "f")
    return f"\\num{{{text}e{exponent}}}"


def render_energy_table(rows: list[EnergyRow]) -> str:
    _require(len(rows) == 9, ENERGY_TABLE_PATH, "partial energy table")
    lines = [
        "% Generated by scripts/render_ima_energy_estimate_q1_master_table.py; do not edit by hand.",
        "% Initial-data terms are recomputed from the strict validated Q1 moment artifacts.",
        "% PDE terms are recomputed from the strict SHA-pinned global affine-cross-moment bounds.",
        "\\begin{table}[t]",
        "\\centering",
        ENERGY_CAPTION,
        "\\label{tab:energy-estimate-q1-master}",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        (
            "Equation & $(d,L,w)$ & LHS & Initial displacement & Initial velocity & "
            "PDE residual & Verified LHS upper bound " + "\\\\"
        ),
        "\\midrule",
    ]
    for index, row in enumerate(rows):
        if index == 6:
            lines.append("\\midrule")
        equation_label = (
            "Heat ($L^2$ data)" if row.equation == "Heat" and row.lhs == "L2"
            else "Heat ($H_0^1$ data)" if row.equation == "Heat"
            else "Wave"
        )
        lhs_tex = (
            "$\\mathcal N_{\\mathrm H}^{(0)}(e_{\\mathrm H})$" if row.lhs == "L2"
            else "$\\mathcal N_{\\mathrm H}^{(1)}(e_{\\mathrm H})$" if row.lhs == "H1"
            else "$\\mathcal N_{\\mathrm W}(e_{\\mathrm W})$"
        )
        velocity = "--" if row.velocity is None else _tex_num(row.velocity, fixed_sig=True)
        architecture = (
            "" if row.equation == "Heat" and row.lhs == "H1"
            else f"\\multirow{{2}}{{*}}{{$(%d,%d,%d)$}}" % (row.dimension, row.layers, row.width)
            if row.equation == "Heat"
            else f"$({row.dimension},{row.layers},{row.width})$"
        )
        lines.append(
            f"{equation_label} & {architecture} & {lhs_tex} & "
            f"{_tex_num(row.displacement, fixed_sig=True)} & {velocity} & "
            f"{_tex_num(row.pde, fixed_sig=True)} & {_tex_num(_ceil_sig(row.total), fixed_sig=True)} \\\\"
        )
        if row.equation == "Heat" and row.lhs == "H1" and row.dimension < 3:
            lines.append("\\addlinespace")
    lines.extend(["\\bottomrule", "\\end{tabular}", "}%", "\\end{table}", ""])
    return "\n".join(lines)


def render_tables(root: Path, catalog: Mapping[str, Any]) -> dict[str, str]:
    initial = load_initial_pairs(root, catalog)
    pde = load_pde_rows(root, catalog)
    rendered = {
        INITIAL_TABLE_PATHS[family]: render_initial_table(initial, family)
        for family in ("displacement", "gradient", "velocity")
    }
    rendered[PDE_TABLE_PATH] = render_pde_table(pde)
    rendered[ENERGY_TABLE_PATH] = render_energy_table(build_energy_rows(pde, initial))

    authority = {entry["path"]: entry for entry in catalog["authority"]["tables"]}
    _require(set(rendered) == set(authority), "catalog", "table authority set mismatch")
    for relative, text in rendered.items():
        encoded = text.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        expected = authority[relative]
        _require(len(encoded) == int(expected["size"]), relative, "rendered size differs from authority")
        _require(digest == expected["sha256"], relative, "rendered SHA-256 differs from authority")
    return rendered


def check_tables(root: Path, catalog: Mapping[str, Any]) -> dict[str, str]:
    rendered = render_tables(root, catalog)
    for relative, expected in rendered.items():
        path = safe_path(root, relative)
        _require(path.is_file(), relative, "table is missing")
        _require(path.read_bytes() == expected.encode("utf-8"), relative, "table is stale")
    return rendered


def write_tables(root: Path, catalog: Mapping[str, Any]) -> dict[str, str]:
    rendered = render_tables(root, catalog)
    for relative, text in rendered.items():
        path = safe_path(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    return rendered
