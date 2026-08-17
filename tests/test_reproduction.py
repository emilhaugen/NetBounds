from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from netbounds.cli import main
from netbounds.data import VerificationError, load_catalog
from netbounds.tables import check_tables, render_tables, write_tables


ROOT = Path(__file__).resolve().parents[1]


def _working_copy(tmp_path: Path) -> Path:
    for name in ("data", "paper", "provenance"):
        shutil.copytree(ROOT / name, tmp_path / name)
    return tmp_path


def test_catalog_is_portable_and_has_exact_scope() -> None:
    catalog = load_catalog(ROOT)
    assert len(catalog["initial_rows"]) == 15
    assert len(catalog["pde_rows"]) == 6
    assert len(catalog["authority"]["tables"]) == 5
    assert len(catalog["checkpoints"]) == 6
    assert not any(item["bundled"] for item in catalog["checkpoints"])
    assert "/mn/" not in json.dumps(catalog)


def test_all_tables_render_byte_exactly() -> None:
    catalog = load_catalog(ROOT)
    rendered = render_tables(ROOT, catalog)
    checked = check_tables(ROOT, catalog)
    assert rendered == checked


def test_stale_table_fails_closed(tmp_path: Path) -> None:
    root = _working_copy(tmp_path)
    table = root / "paper" / "tables" / "pde_residuals_q1_master.tex"
    table.write_text(table.read_text() + "% stale\n")
    with pytest.raises(VerificationError, match="table is stale"):
        check_tables(root, load_catalog(root))


def test_changed_artifact_fails_before_render(tmp_path: Path) -> None:
    root = _working_copy(tmp_path)
    artifact = root / "data" / "artifacts" / "initial" / "q1" / "wave3_gradient.json"
    artifact.write_text(artifact.read_text().replace('"certified": true', '"certified": null'))
    with pytest.raises(VerificationError, match="SHA-256 mismatch"):
        render_tables(root, load_catalog(root))


def test_write_tables_atomically_restores_output(tmp_path: Path) -> None:
    root = _working_copy(tmp_path)
    table = root / "paper" / "tables" / "energy_estimate_q1_master.tex"
    table.write_text("stale\n")
    write_tables(root, load_catalog(root))
    check_tables(root, load_catalog(root))


def test_cli_verifies_repository(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify", "--root", str(ROOT)]) == 0
    output = capsys.readouterr().out
    assert "30 initial + 6 PDE" in output
    assert "5 tables (byte-exact)" in output
    assert "0/6 checkpoints bundled" in output
