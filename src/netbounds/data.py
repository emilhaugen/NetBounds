"""Portable catalog loading and byte-level integrity checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


class VerificationError(ValueError):
    """Raised when retained evidence is missing, malformed, or stale."""


def repository_root(explicit: Path | None = None) -> Path:
    """Locate the checkout containing ``data/catalog.json``.

    An explicit root wins. Otherwise the current directory and its parents are
    searched before falling back to the source checkout containing this module.
    """

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    else:
        candidates.extend((Path.cwd(), *Path.cwd().parents))
        source_checkout = Path(__file__).resolve().parents[2]
        candidates.extend((source_checkout, *source_checkout.parents))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "data" / "catalog.json").is_file():
            return resolved
    requested = explicit if explicit is not None else Path.cwd()
    raise VerificationError(f"cannot locate data/catalog.json from {requested}")


def safe_path(root: Path, relative: str) -> Path:
    """Resolve a catalog path while rejecting absolute paths and traversal."""

    raw = Path(relative)
    if raw.is_absolute():
        raise VerificationError(f"catalog path must be relative: {relative}")
    path = (root / raw).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise VerificationError(f"catalog path escapes repository: {relative}") from error
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_catalog(root: Path) -> dict[str, Any]:
    path = root / "data" / "catalog.json"
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read {path}: {error}") from error
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        raise VerificationError(f"{path}: unsupported catalog schema")
    if catalog.get("scope") not in {
        "artifact_table_replay",
        "current_paper_initial_and_pde_d1_d2_with_frozen_pde_d3",
        "current_paper_all_initial_and_pde_d1_d2_d3",
    }:
        raise VerificationError(f"{path}: unexpected reproduction scope")
    return catalog


def verify_file(root: Path, entry: Mapping[str, Any]) -> Path:
    """Require exact path, size, and SHA-256 agreement for one catalog entry."""

    relative = str(entry.get("path", ""))
    path = safe_path(root, relative)
    if not path.is_file():
        raise VerificationError(f"missing retained file: {relative}")
    expected_size = int(entry.get("size", -1))
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise VerificationError(
            f"{relative}: size mismatch ({actual_size} != {expected_size})"
        )
    expected_hash = str(entry.get("sha256", ""))
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise VerificationError(
            f"{relative}: SHA-256 mismatch ({actual_hash} != {expected_hash})"
        )
    return path


def verify_files(
    root: Path, entries: Iterable[Mapping[str, Any]]
) -> list[Path]:
    return [verify_file(root, entry) for entry in entries]


def read_hashed_json(
    root: Path, entry: Mapping[str, Any], *, parse_float: Any = float
) -> tuple[Path, dict[str, Any]]:
    path = verify_file(root, entry)
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_float=parse_float)
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"{path}: expected a JSON object")
    return path, value
