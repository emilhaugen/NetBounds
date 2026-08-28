from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from netbounds.cli import main
from netbounds.numerics.cases import CASES, case_names
from netbounds.tables import INITIAL_CAPTIONS, PDE_CAPTION
from netbounds.data import VerificationError, load_catalog

ROOT = Path(__file__).resolve().parents[1]


def test_public_scope_is_all_36_fixed_computations() -> None:
    names = case_names()
    assert len(names) == 36
    assert names == tuple(sorted(CASES))


@pytest.mark.parametrize(
    "case_name",
    [
        "heat1-pde-q1",
        "heat2-pde-q1",
        "heat3-pde-q1",
        "wave1-pde-q1",
        "wave2-pde-q1",
        "wave3-pde-q1",
    ],
)
def test_all_pde_authorities_pass_full_semantic_comparison(case_name: str) -> None:
    from netbounds.numerics.compare import (
        _PDE_ADAPTIVE_NUMERICAL_FIELDS,
        _PDE_NUMERICAL_FIELDS,
        compare_reproduction,
    )

    case = CASES[case_name]
    authority = json.loads((ROOT / case.authority).read_text())
    fields = (
        _PDE_ADAPTIVE_NUMERICAL_FIELDS
        if case_name == "heat3-pde-q1"
        else _PDE_NUMERICAL_FIELDS
    )
    payload = {
        "schema_version": 1,
        "kind": "netbounds_paper_checkpoint_reproduction",
        "case": case.name,
        "authority": case.authority,
        "identity": {
            "equation": case.equation,
            "quantity": case.quantity,
            "quadrature_rule": "Q1",
            "architecture": {"layers": case.L, "width": case.width},
            "checkpoint": case.checkpoint,
            "checkpoint_sha256": case.checkpoint_sha256,
            "dtype": "float32",
            "grid": list(case.grid),
        },
        "execution": {
            "device": "cuda:0",
            "batch_size": case.batch_size,
            "accumulation_dtype": "float64",
        },
        "numerical": {field: authority[field] for field in fields},
    }
    comparison = compare_reproduction(payload, root=ROOT)
    assert comparison["passed"], comparison


@pytest.mark.parametrize(
    ("case_name", "mutation"),
    [
        (
            "heat3-pde-q1",
            lambda data: data.__setitem__("all_adaptive_leaves_enumerated", False),
        ),
        (
            "wave3-pde-q1",
            lambda data: data.__setitem__("cell_radii", [0.005] * 4),
        ),
        (
            "wave3-pde-q1",
            lambda data: data["acceptance"].__setitem__(
                "accepted_as_final", False
            ),
        ),
    ],
)
def test_3d_policy_mutations_fail_closed(case_name, mutation) -> None:
    from netbounds.tables import _validate_pde_artifact

    case = CASES[case_name]
    data = json.loads((ROOT / case.authority).read_text())
    changed = deepcopy(data)
    mutation(changed)
    policy = next(
        row
        for row in load_catalog(ROOT)["pde_rows"]
        if row["artifact"]["path"] == case.authority
    )
    with pytest.raises(VerificationError):
        _validate_pde_artifact(changed, policy, ROOT / case.authority)


@pytest.mark.torch
def test_d2_flat_grid_decoding_keeps_time_fastest() -> None:
    torch = pytest.importorskip("torch")
    from netbounds.numerics.reproduce import _decode_centers

    grid = (3, 3, 5)
    centers = _decode_centers(0, 45, grid=grid, device=torch.device("cpu"))
    torch.testing.assert_close(
        centers[0], torch.tensor([1 / 6, 1 / 6, 1 / 10], dtype=torch.float32)
    )
    torch.testing.assert_close(
        centers[-1], torch.tensor([5 / 6, 5 / 6, 9 / 10], dtype=torch.float32)
    )
    torch.testing.assert_close(
        centers[1] - centers[0], torch.tensor([0.0, 0.0, 0.2])
    )


def test_current_table_captions_are_plain() -> None:
    assert INITIAL_CAPTIONS["displacement"] == (
        r"\caption{Upper bounds for the initial displacement residual norm, as "
        r"constructed in \cref{sec:initial-displacement-quad}.}"
    )
    assert INITIAL_CAPTIONS["gradient"] == (
        r"\caption{Upper bounds for the initial displacement gradient residual norm, "
        r"as constructed in \cref{sec:initial-displacement-gradient-quad}.}"
    )
    assert INITIAL_CAPTIONS["velocity"] == (
        r"\caption{Upper bounds for the initial-velocity residual norm, as constructed "
        r"in \cref{sec:initial-velocity-quad}.}"
    )
    assert PDE_CAPTION == (
        r"\caption{Upper bounds for the PDE residual norms, as constructed in "
        r"\cref{sec:pde-residual-quad}.}"
    )


def test_cli_requires_acknowledgement_for_large_cases(capsys) -> None:
    assert main(["reproduce", "--case", "heat3-displacement-q0"]) == 1
    assert "--allow-large-initial" in capsys.readouterr().err
    assert main(["reproduce", "--case", "heat2-pde-q1"]) == 1
    assert "--allow-full-pde" in capsys.readouterr().err
