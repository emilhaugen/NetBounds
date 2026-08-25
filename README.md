# NetBounds

Minimal reproduction of the numerical tables in the NetBounds paper, with all six selected trained checkpoints bundled.

This repository is built from the dependency closure of the paper tables, not by pruning the development repository. The current scope reproduces all 30 initial-data artifacts and the four current-paper d=1/2 appendix-direct PDE artifacts from hash-pinned checkpoints. The two 3D PDE artifacts remain replay-only until their running NetBounds-dev campaign finishes and the 3D closure is integrated.

## Current capabilities

```console
netbounds-paper verify
netbounds-paper tables --check
netbounds-paper tables
netbounds-paper reproduce --case heat1-displacement-q0 --device cpu --check
```

- `verify` checks every retained payload file by size and SHA-256, validates the artifact schemas and policy matrix, regenerates all five tables in memory, and compares them byte-for-byte with the paper authority.
- `tables --check` is a fail-closed, no-write table check.
- `tables` atomically regenerates the tracked TeX tables from the retained JSON artifacts.
- `reproduce` exposes all 30 fixed initial-data cases and the four fixed d=1/2 PDE cases. With `--check`, it rejects missing, unknown, empty, or numerically mismatched fields against the hash-pinned retained authority. The two d=3 PDE cases remain unavailable.
- PDE cases require `--allow-full-pde`; 3D initial-data cases require `--allow-large-initial` and CUDA. Every PDE case freezes its authority-recorded batch size.

The five outputs are:

1. initial-displacement bounds (Q0 and Q1),
2. initial-displacement-gradient bounds (Q0 and Q1),
3. initial-velocity bounds (Q0 and Q1),
4. PDE-residual Q1 moment bounds, and
5. the combined energy-estimate bounds.

## Install and verify

Python 3.11 or newer is required. PyTorch is a required package dependency: a repository that promises checkpoint-backed recomputation must install the network runtime. `verify` and `tables` remain operationally Torch-free—they do not import PyTorch or execute network kernels—so the artifact-only CI job installs this project with `--no-deps`.

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

All five tracked table snapshots are reproduced exactly from 36 retained, hash-checked JSON artifacts. The three initial-data tables and the d=1/2 PDE rows use the current-paper artifacts; the two d=3 PDE rows remain frozen historical replay authority until their current campaign completes. All six checkpoints are bundled and integrity checked. The public fixed-case registry covers 34 of 36 computations: all initial-data dimensions and PDE dimensions one and two.

Current authority was generated with ordinary CUDA float32 kernels and float64 PDE accumulation. Initial d=1/2 cases also have explicit CPU portability gates; d=3 initial cases require CUDA. See `provenance/numerical-trust-boundary.md` and `provenance/current-paper-reproduction.json` for the comparison policy and producer lineage.

The retained artifacts and portable replays are ordinary floating-point computations, not outward-rounded interval arithmetic.

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

The retained raw artifacts and original campaign manifests are provenance evidence and therefore contain historical absolute paths. Runtime code never reads those paths; all operational paths come from the portable catalog.

## Citation and license

Please cite the accompanying paper, *Trustworthy AI in numerics: On verification algorithms for neural network-based PDE solvers*, by Emil Haugen, Alexei Stepanenko, and Anders C. Hansen. Machine-readable citation metadata is provided in `CITATION.cff`.

The software is released under the MIT License; see `LICENSE`.
