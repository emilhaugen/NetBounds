from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from netbounds.cli import main
from netbounds.data import load_catalog, verify_file
from netbounds.numerics.cases import CASES, case_names
from netbounds.numerics.compare import compare_reproduction


ROOT = Path(__file__).resolve().parents[1]


def _torch():
    return pytest.importorskip("torch", reason="declared PyTorch dependency is unavailable")


def test_bundled_checkpoint_entries_are_integrity_checked() -> None:
    catalog = load_catalog(ROOT)
    bundled = [entry for entry in catalog["checkpoints"] if entry["bundled"]]
    assert [entry["path"] for entry in bundled] == [
        "data/checkpoints/heat1-l2-w128.pt",
        "data/checkpoints/heat2-l3-w128.pt",
        "data/checkpoints/heat3-l4-w128.pt",
        "data/checkpoints/wave1-l2-w256.pt",
        "data/checkpoints/wave2-l3-w256.pt",
        "data/checkpoints/wave3-l3-w256.pt",
    ]
    payload_paths = {entry["path"] for entry in catalog["payload_files"]}
    assert {entry["path"] for entry in bundled} <= payload_paths
    for entry in bundled:
        assert verify_file(ROOT, entry).is_file()
    assert catalog["checkpoint_reproduction"] == {
        "covered_artifact_computations": 36,
        "full_table_recomputation": True,
        "supported_initial_dimensions": [1, 2, 3],
        "supported_pde_dimensions": [1, 2, 3],
        "total_artifact_computations": 36,
    }
    assert catalog["capabilities"]["full_checkpoint_reproduction"] is True
    assert catalog["capabilities"]["checkpoint_backed_initial_reproduction"] is True
    assert catalog["capabilities"]["checkpoint_backed_all_pde_reproduction"] is True
    assert catalog["capabilities"]["gpu_numerical_recomputation"] is True
    assert catalog["release_complete"] is True


def test_comparator_rejects_empty_pde_payload_before_authority_read() -> None:
    case = CASES["heat1-pde-q1"]
    payload = {
        "schema_version": 1,
        "kind": "netbounds_paper_checkpoint_reproduction",
        "case": case.name,
        "authority": case.authority,
        "identity": {
            "equation": case.equation,
            "quantity": case.quantity,
            "quadrature_rule": case.rule.upper(),
            "architecture": {"layers": 2, "width": case.width},
            "checkpoint": case.checkpoint,
            "checkpoint_sha256": case.checkpoint_sha256,
            "dtype": "float32",
            "grid": [500, 500],
        },
        "execution": {"device": "cpu", "batch_size": case.batch_size, "accumulation_dtype": "float64"},
        "numerical": {},
    }
    with pytest.raises(ValueError, match="PDE numerical fields differ"):
        compare_reproduction(payload, root=ROOT)

    payload["execution"]["batch_size"] = case.batch_size // 2
    with pytest.raises(ValueError, match=f"recorded value {case.batch_size}"):
        compare_reproduction(payload, root=ROOT)


def test_every_declared_case_is_public() -> None:
    assert case_names() == tuple(sorted(CASES))


def test_pde_comparator_rejects_unsupported_execution_device() -> None:
    from netbounds.numerics.compare import _PDE_NUMERICAL_FIELDS

    case = CASES["heat1-pde-q1"]
    authority = json.loads((ROOT / case.authority).read_text())
    payload = {
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
            "device": "mps",
            "batch_size": case.batch_size,
            "accumulation_dtype": "float64",
        },
        "numerical": {field: authority[field] for field in _PDE_NUMERICAL_FIELDS},
    }
    with pytest.raises(ValueError, match="unsupported PDE device"):
        compare_reproduction(payload, root=ROOT)


@pytest.mark.torch
def test_python_reproducer_rejects_cpu_3d_cases() -> None:
    _torch()
    from netbounds.numerics.reproduce import reproduce

    with pytest.raises(ValueError, match="3D PDE paper cases require a CUDA device"):
        reproduce("wave3-pde-q1", root=ROOT, device="cpu")
    with pytest.raises(ValueError, match="require a CUDA device"):
        reproduce("wave3-displacement-q0", root=ROOT, device="cpu")


@pytest.mark.torch
@pytest.mark.parametrize("case_name", ["heat1-displacement-q0", "wave1-displacement-q0"])
def test_bundled_checkpoint_schema_and_fixed_model_load(case_name: str) -> None:
    torch = _torch()
    from netbounds.numerics.model import load_checkpoint

    case = CASES[case_name]
    model = load_checkpoint(ROOT / case.checkpoint, width=case.width, device="cpu")
    expected_shapes = {
        "hidden_layers.0.weight": (case.width, 2),
        "hidden_layers.0.bias": (case.width,),
        "hidden_layers.1.weight": (case.width, case.width),
        "hidden_layers.1.bias": (case.width,),
        "output_layer.weight": (1, case.width),
        "output_layer.bias": (1,),
    }
    state = torch.load(ROOT / case.checkpoint, map_location="cpu", weights_only=True)
    assert {key: tuple(value.shape) for key, value in state.items()} == expected_shapes
    assert {value.dtype for value in state.values()} == {torch.float64}
    assert model(torch.zeros((3, 2), dtype=torch.float32)).shape == (3, 1)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert model.AB.dtype == torch.float32
    assert not any(parameter.requires_grad for parameter in model.parameters())


@pytest.fixture(scope="module")
def cpu_initial_reproductions() -> dict[str, dict[str, object]]:
    _torch()
    from netbounds.numerics.reproduce import reproduce

    results: dict[str, dict[str, object]] = {}
    for case_name in case_names():
        if CASES[case_name].quantity == "pde" or CASES[case_name].d != 1:
            continue
        result = reproduce(case_name, root=ROOT, device="cpu", batch_size=500)
        comparison = compare_reproduction(result, root=ROOT)
        assert comparison["passed"], comparison
        assert comparison["tolerance_class"] == "initial_cpu_portability_d1"
        results[case_name] = result
    return results


@pytest.mark.torch
def test_all_fixed_initial_cases_reproduce_on_full_cpu_grid(cpu_initial_reproductions) -> None:
    assert set(cpu_initial_reproductions) == {
        name for name, case in CASES.items() if case.quantity != "pde" and case.d == 1
    }


@pytest.mark.torch
def test_comparator_rejects_mutated_or_incomplete_semantic_payload(cpu_initial_reproductions) -> None:
    baseline = cpu_initial_reproductions["heat1-displacement-q0"]
    changed = deepcopy(baseline)
    changed["numerical"]["l2_bound"] = 0.0
    comparison = compare_reproduction(changed, root=ROOT)
    assert not comparison["passed"]
    assert "l2_bound" in comparison["failures"]["numerical"]

    incomplete = deepcopy(baseline)
    del incomplete["numerical"]["eps"]
    with pytest.raises(ValueError, match="initial numerical fields differ"):
        compare_reproduction(incomplete, root=ROOT)


@pytest.mark.torch
@pytest.mark.parametrize(
    ("case_name", "kernel_name"),
    [("heat1-pde-q1", "heat"), ("wave1-pde-q1", "wave")],
)
def test_tiny_pde_kernel_smoke(case_name: str, kernel_name: str) -> None:
    torch = _torch()
    from netbounds.numerics.model import load_checkpoint
    from netbounds.numerics.pde import heat_hyper_taylor_rect_batch, wave_hyper_taylor_rect_batch

    case = CASES[case_name]
    model = load_checkpoint(ROOT / case.checkpoint, width=case.width, device="cpu")
    y = torch.tensor(
        [[0.25, 1.0 / 6.0], [0.25, 0.5], [0.25, 5.0 / 6.0], [0.75, 1.0 / 6.0], [0.75, 0.5], [0.75, 5.0 / 6.0]],
        dtype=torch.float32,
    )
    eps = torch.tensor([0.25, 1.0 / 6.0], dtype=torch.float32)
    if kernel_name == "heat":
        residual, rho, diagnostics = heat_hyper_taylor_rect_batch(
            model=model, y=y, eps_vec=eps, alpha=0.1, n_taylor=None
        )
    else:
        residual, rho, diagnostics = wave_hyper_taylor_rect_batch(
            model=model, y=y, eps_vec=eps, c2=1.0, n_taylor=None
        )
    assert residual.shape == rho.shape == (6,)
    assert torch.isfinite(residual).all()
    assert torch.isfinite(rho).all() and (rho >= 0).all()
    for key in (
        "hyper_taylor_centered_moment_best_bound_l2sq",
        "hyper_taylor_centered_moment_cross_bound_l2sq",
        "hyper_taylor_rho_hyper",
        "hyper_taylor_rho_boundary",
    ):
        assert diagnostics[key].shape == (6,)
        assert torch.isfinite(diagnostics[key]).all()


@pytest.mark.torch
def test_pde_replay_rejects_non_authority_batch_size() -> None:
    _torch()
    from netbounds.numerics.reproduce import reproduce

    with pytest.raises(ValueError, match="requires batch_size=65536"):
        reproduce("heat1-pde-q1", root=ROOT, device="cpu", batch_size=8192)


@pytest.mark.torch
def test_cli_fixed_case_check_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    _torch()
    assert main([
        "reproduce",
        "--case",
        "heat1-displacement-q0",
        "--device",
        "cpu",
        "--batch-size",
        "500",
        "--check",
        "--root",
        str(ROOT),
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["comparison"]["passed"] is True
