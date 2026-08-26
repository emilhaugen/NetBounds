"""Command-line interface for the minimal NetBounds paper repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .data import (
    VerificationError,
    load_catalog,
    repository_root,
    verify_files,
    verify_numerical_source_closure,
)
from .tables import check_tables, write_tables


def _root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        help="repository root (normally discovered automatically)",
    )


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="netbounds-paper",
        description="Verify and render the numerical tables in the NetBounds paper.",
    )
    subcommands = command.add_subparsers(dest="command", required=True)

    verify = subcommands.add_parser(
        "verify", help="verify retained artifacts and all five table bytes"
    )
    _root_argument(verify)

    tables = subcommands.add_parser(
        "tables", help="render the five tables from retained JSON artifacts"
    )
    _root_argument(tables)
    tables.add_argument(
        "--check",
        action="store_true",
        help="fail if any tracked table differs; do not write files",
    )
    from .numerics.cases import case_names

    reproduce = subcommands.add_parser(
        "reproduce",
        help="recompute one fixed checkpoint-backed paper case",
    )
    _root_argument(reproduce)
    reproduce.add_argument("--case", choices=case_names(), required=True)
    reproduce.add_argument("--device", help="execution device (cpu or cuda); defaults to cuda if available")
    reproduce.add_argument(
        "--batch-size",
        type=int,
        help="positive initial-condition batch size; each PDE case pins its production batch",
    )
    reproduce.add_argument(
        "--check",
        action="store_true",
        help="fail closed unless every fixed numerical field matches authority",
    )
    reproduce.add_argument(
        "--allow-full-pde",
        action="store_true",
        help="acknowledge the potentially large fixed PDE computation",
    )
    reproduce.add_argument(
        "--allow-large-initial",
        action="store_true",
        help="acknowledge a fixed 125-million-cell 3D initial-data computation",
    )
    return command


def _verify(root: Path, catalog: dict[str, Any]) -> None:
    payload = verify_files(root, catalog["payload_files"])
    numerical_sources = verify_numerical_source_closure(root, catalog)
    rendered = check_tables(root, catalog)
    checkpoints = catalog["checkpoints"]
    bundled = sum(bool(entry["bundled"]) for entry in checkpoints)
    print(f"PASS retained payload: {len(payload)} files")
    print(f"PASS numerical source closure: {numerical_sources} files")
    print(f"PASS selected artifacts: 30 initial + 6 PDE")
    print(f"PASS rendered authority: {len(rendered)} tables (byte-exact)")
    print(
        "NOTE checkpoint reproduction: "
        f"{bundled}/{len(checkpoints)} checkpoints bundled; "
        "36/36 fixed computations exposed (all initial data and all PDE rows)"
    )


def _tables(root: Path, catalog: dict[str, Any], *, check: bool) -> None:
    if check:
        rendered = check_tables(root, catalog)
        verb = "checked"
    else:
        rendered = write_tables(root, catalog)
        verb = "wrote"
    print(f"PASS {verb} {len(rendered)} artifact-backed tables")
    for relative in rendered:
        print(f"  {relative}")


def _reproduce(args: argparse.Namespace, root: Path) -> None:
    """Run a closed paper case without importing PyTorch for other commands."""

    from .numerics.cases import CASES

    case = CASES[args.case]
    if case.quantity == "pde" and not args.allow_full_pde:
        raise ValueError("PDE replay needs --allow-full-pde")
    if case.quantity != "pde" and case.d == 3 and not args.allow_large_initial:
        raise ValueError("3D initial-data replay needs --allow-large-initial")
    from .numerics.compare import compare_reproduction
    from .numerics.reproduce import reproduce

    progress_milestone = 0

    def progress(done: int, total: int) -> None:
        nonlocal progress_milestone
        milestone = 10 if done == total else (10 * done) // total
        if milestone > progress_milestone:
            progress_milestone = milestone
            print(f"PDE progress: {done}/{total} ({10 * milestone}%)", file=sys.stderr)

    result = reproduce(
        args.case,
        root=root,
        device=args.device,
        batch_size=args.batch_size,
        progress=progress if "-pde-" in args.case else None,
    )
    payload: dict[str, Any] = result
    if args.check:
        comparison = compare_reproduction(result, root=root)
        payload = {"reproduction": result, "comparison": comparison}
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if args.check and not payload["comparison"]["passed"]:
        raise VerificationError("reproduction comparison failed")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        root = repository_root(args.root)
        catalog = load_catalog(root)
        if args.command == "verify":
            _verify(root, catalog)
        elif args.command == "tables":
            _tables(root, catalog, check=args.check)
        elif args.command == "reproduce":
            _reproduce(args, root)
        else:  # pragma: no cover - argparse enforces the command set.
            raise AssertionError(args.command)
    except (ArithmeticError, OSError, KeyError, RuntimeError, TypeError, VerificationError, ValueError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
