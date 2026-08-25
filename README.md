# NetBounds

Minimal, artifact-backed reproduction of the five numerical tables in the NetBounds paper.

This repository is being constructed from the dependency closure of the published tables, not by pruning the development repository. The current bootstrap slice is intentionally small: 36 selected JSON artifacts, five table files, exact campaign/formula provenance, and a standard-library-only renderer and verifier.

## Current capabilities

```console
netbounds-paper verify
netbounds-paper tables --check
netbounds-paper tables
```

- `verify` checks every retained payload file by size and SHA-256, validates the artifact schemas and policy matrix, regenerates all five tables in memory, and compares them byte-for-byte with the paper authority.
- `tables --check` is a fail-closed, no-write table check.
- `tables` atomically regenerates the tracked TeX tables from the retained JSON artifacts.

The five outputs are:

1. initial-displacement bounds (Q0 and Q1),
2. initial-displacement-gradient bounds (Q0 and Q1),
3. initial-velocity bounds (Q0 and Q1),
4. PDE-residual Q1 moment bounds, and
5. the combined energy-estimate bounds.

## Install and verify

Python 3.11 or newer is required. Runtime rendering uses only the Python standard library.

```console
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/netbounds-paper verify
.venv/bin/pytest
```

With `uv`:

```console
uv venv
uv pip install -e '.[test]'
.venv/bin/netbounds-paper verify
.venv/bin/pytest
```

No network, GPU, another NetBounds checkout, or `/mn/...` filesystem is needed after installation.

## Reproduction boundary

This bootstrap slice performs **offline artifact verification and exact table rendering**. It does not yet recompute the 36 JSON artifacts from neural-network checkpoints. The six immutable checkpoint identities are declared in `data/catalog.json`, but the checkpoint files and reached producer kernels will be added in a separate reviewed slice.

The retained artifacts report float32 cell-kernel arithmetic and float64 accumulation. They are not outward-rounded interval computations. See `provenance/numerical-trust-boundary.md` for the exact interpretation and comparison policy.

Heat-3D is explicitly a same-policy re-evaluation, not an exact replay of the unavailable historical leaf map. The accepted result differs from the old partition by 210 leaves; the qualification is validated by the renderer and retained in the JSON artifact.

## Layout

```text
src/netbounds/              small public CLI and table renderer
data/catalog.json           portable dependency and hash authority
data/artifacts/             30 initial + 6 PDE selected JSON artifacts
paper/tables/               five current byte-exact TeX tables
provenance/campaigns/       original campaign manifests and validation records
provenance/formulas/        exact formula bytes bound by both campaigns
provenance/                 producer inventory and numerical trust boundary
tests/                      fail-closed integrity and rendering tests
```

The original campaign manifests are retained as provenance and therefore contain historical absolute paths. Runtime code never reads those paths; all operational paths come from the portable catalog.
