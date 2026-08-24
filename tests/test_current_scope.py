from __future__ import annotations

import torch

from netbounds.cli import main
from netbounds.numerics.cases import CASES, case_names
from netbounds.numerics.reproduce import _decode_centers
from netbounds.tables import INITIAL_CAPTIONS, PDE_CAPTION


def test_public_scope_is_all_initial_plus_d1_d2_pde() -> None:
    names = case_names()
    assert len(names) == 34
    assert {name for name, case in CASES.items() if name not in names} == {
        "heat3-pde-q1",
        "wave3-pde-q1",
    }
    assert all(CASES[name].quantity != "pde" or CASES[name].d <= 2 for name in names)


def test_d2_flat_grid_decoding_keeps_time_fastest() -> None:
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
