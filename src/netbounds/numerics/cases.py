"""Torch-free registry for the fixed checkpoint-backed paper cases.

This is deliberately a closed list of the 36 retained computations behind
the five current IMA tables.  It is not a campaign or training API.  The
checkpoint identities are repeated here only so command-line argument
validation remains available without importing PyTorch; ``reproduce`` checks
that each identity agrees with the hash-pinned catalog before execution.
"""

from __future__ import annotations

from dataclasses import dataclass

_SHA256 = {
    "heat1": "514531c1d38d138b053de85842e4c4a92de0bc82a629d448437ad1c8b5028ba6",
    "heat2": "1f3c05a3b8a93322dba8b666fd772858bdcb8217a91bb7c1c087ada85b2b000f",
    "heat3": "71a01a8f97081ae6e8b1ef9466c3ff04a246fc5426acc8eab94e87cacf2abe64",
    "wave1": "a9f625f853c3c1b7bcefd2ce7ab6c5418056f03092a3e055054e4c793b1564c2",
    "wave2": "382e464004cb1a723df708e09b07c72a1d2fed0d2b97669d0bf18ef37d9fd40f",
    "wave3": "4ad49e0334fd05b99d68106fa3846353014b86a57a8fe6640beeb965feba3f06",
}

_ARCHITECTURE = {
    "heat1": (1, 2, 128),
    "heat2": (2, 3, 128),
    "heat3": (3, 4, 128),
    "wave1": (1, 2, 256),
    "wave2": (2, 3, 256),
    "wave3": (3, 3, 256),
}

_PDE_GRIDS = {
    "heat1": (500, 500),
    "heat2": (300, 300, 750),
    "heat3": (100, 100, 100, 500),
    "wave1": (500, 500),
    "wave2": (300, 300, 750),
    "wave3": (200, 100, 100, 500),
}

_INITIAL_CELLS_PER_DIM = 500


@dataclass(frozen=True)
class Case:
    """Immutable definition of one retained numerical result."""

    name: str
    equation: str
    quantity: str
    rule: str
    d: int
    L: int
    width: int
    cells_per_dim: int
    grid: tuple[int, ...]
    checkpoint: str
    checkpoint_sha256: str
    storage_dtype: str
    nested: bool
    authority: str


def _case(equation: str, dimension: int, quantity: str, rule: str) -> Case:
    prefix = equation.lower()
    key = f"{prefix}{dimension}"
    d, L, width = _ARCHITECTURE[key]
    checkpoint = f"data/checkpoints/{key}-l{L}-w{width}.pt"
    nested = key == "wave3"
    storage_dtype = "float32" if nested else "float64"
    if quantity == "pde":
        authority = f"data/artifacts/pde/q1/{key}.json"
        grid = _PDE_GRIDS[key]
    else:
        authority = f"data/artifacts/initial/{rule}/{key}_{quantity}.json"
        grid = (_INITIAL_CELLS_PER_DIM,) * d
    return Case(
        name=f"{key}-{quantity}-{rule}",
        equation=equation,
        quantity=quantity,
        rule=rule,
        d=d,
        L=L,
        width=width,
        cells_per_dim=grid[0],
        grid=grid,
        checkpoint=checkpoint,
        checkpoint_sha256=_SHA256[key],
        storage_dtype=storage_dtype,
        nested=nested,
        authority=authority,
    )


_CASES: tuple[Case, ...] = ()
for _equation in ("Heat", "Wave"):
    for _dimension in (1, 2, 3):
        _key = f"{_equation.lower()}{_dimension}"
        _cases = [
            _case(_equation, _dimension, "displacement", "q0"),
            _case(_equation, _dimension, "displacement", "q1"),
            _case(_equation, _dimension, "gradient", "q0"),
            _case(_equation, _dimension, "gradient", "q1"),
        ]
        if _equation == "Wave":
            _cases += [
                _case(_equation, _dimension, "velocity", "q0"),
                _case(_equation, _dimension, "velocity", "q1"),
            ]
        _cases += [_case(_equation, _dimension, "pde", "q1")]
        _CASES += tuple(_cases)

CASES: dict[str, Case] = {case.name: case for case in _CASES}


def all_case_names() -> tuple[str, ...]:
    """Return every declared paper case, including pending 2D/3D closures."""

    return tuple(sorted(CASES))


def case_names() -> tuple[str, ...]:
    """Return the currently validated public reproduction cases.

    The six selected checkpoints are bundled, but only the complete 1D
    numerical closure is exposed until the fixed 2D/3D implementations pass
    their authority comparisons and slow/GPU release gates.
    """

    return tuple(sorted(name for name, case in CASES.items() if case.d == 1))
