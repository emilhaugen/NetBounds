# NetBounds

Minimal reproduction of the numerical tables in the NetBounds paper, with all six selected trained checkpoints bundled.

This repository is built from the dependency closure of the paper tables, not by pruning the development repository. It retains all 30 initial-data artifacts and all six current-paper appendix-direct PDE artifacts together with the hash-pinned checkpoints and fixed numerical closure needed to recompute them.

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
- `reproduce` exposes all 36 fixed paper computations. With `--check`, it rejects missing, unknown, empty, or numerically mismatched fields against the hash-pinned retained authority.
- PDE cases require `--allow-full-pde`; 3D initial-data cases require `--allow-large-initial` and CUDA. Every PDE case freezes its authority-recorded batch size.

The five outputs are:

1. initial-displacement bounds (Q0 and Q1),
2. initial-displacement-gradient bounds (Q0 and Q1),
3. initial-velocity bounds (Q0 and Q1),
4. PDE-residual Q1 moment bounds, and
5. the combined energy-estimate bounds.

## Install and verify

Python 3.11 or newer is required. PyTorch is a required package dependency: a repository that promises checkpoint-backed recomputation must install the network runtime. `verify` and `tables` remain operationally Torch-free—they do not import PyTorch or execute network kernels—so the artifact-only CI job installs this project with `--no-deps`.

NetBounds is intentionally checkout-rooted rather than a standalone wheel: checkpoints, artifacts, provenance, and tables remain single-copy top-level scientific payloads. Run the commands from a clone or extracted source archive.

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

All five tracked table snapshots are reproduced exactly from 36 retained, hash-checked current-paper JSON artifacts. All six checkpoints are bundled and integrity checked. The closed public registry covers all 36 computations: all initial-data dimensions and all six PDE rows.

Current authority was generated with ordinary CUDA float32 kernels and float64 PDE accumulation. Initial d=1/2 cases also have explicit CPU portability gates; d=3 initial cases require CUDA. See `provenance/numerical-trust-boundary.md` and `provenance/current-paper-reproduction.json` for the comparison policy and producer lineage.

The retained artifacts and portable replays are ordinary floating-point computations, not outward-rounded interval arithmetic.

Heat-3D is explicitly a complete same-policy re-evaluation, not an exact replay of the unavailable historical leaf map. The user-accepted current-paper result differs from the historical aggregate by 10,995 leaves out of approximately 1.64 billion; complete base-root coverage and the internal adaptive tree identities are validated by the renderer and retained in the JSON artifact.

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
