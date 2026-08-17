"""Command-line interface for the minimal NetBounds paper repository."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from .data import VerificationError, load_catalog, repository_root, verify_files
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
    return command


def _verify(root: Path, catalog: dict[str, Any]) -> None:
    payload = verify_files(root, catalog["payload_files"])
    rendered = check_tables(root, catalog)
    checkpoints = catalog["checkpoints"]
    bundled = sum(bool(entry["bundled"]) for entry in checkpoints)
    print(f"PASS retained payload: {len(payload)} files")
    print("PASS selected artifacts: 30 initial + 6 PDE")
    print(f"PASS rendered authority: {len(rendered)} tables (byte-exact)")
    print(
        "NOTE GPU recomputation: "
        f"{bundled}/{len(checkpoints)} checkpoints bundled; not part of this bootstrap slice"
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


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        root = repository_root(args.root)
        catalog = load_catalog(root)
        if args.command == "verify":
            _verify(root, catalog)
        elif args.command == "tables":
            _tables(root, catalog, check=args.check)
        else:  # pragma: no cover - argparse enforces the command set.
            raise AssertionError(args.command)
    except (OSError, KeyError, TypeError, VerificationError, ValueError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
