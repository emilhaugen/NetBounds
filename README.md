# NetBounds

Minimal reproduction of the five numerical tables in the NetBounds paper, with all six selected trained checkpoints bundled and a validated checkpoint-backed 1D sanity slice.

This repository is being constructed from the dependency closure of the paper tables, not by pruning the development repository. The current branch contains artifact-exact replay for the frozen August 17 table authority plus checkpoint-backed, tolerance-checked 1D sanity reproduction: 36 selected JSON artifacts, five table files, all six selected checkpoints, and only the reached 1D producer closure. The 2D/3D numerical closure and the refresh to the latest current-paper artifacts remain explicit release gates.

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
- `reproduce` accepts only the twelve fixed 1D paper cases. With `--check`, it rejects missing, unknown, empty, or numerically mismatched fields against the hash-pinned retained authority.
- PDE cases are deliberate full CPU computations: use `--allow-full-pde`, for example `netbounds-paper reproduce --case heat1-pde-q1 --allow-full-pde --check`. They freeze the authority-recorded batch size of 4096; overriding it can change CPU float32 kernels enough to violate the strict comparison gate.

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

All five frozen table snapshots are reproduced exactly from their 36 retained, hash-checked JSON artifacts. All six selected checkpoints are bundled and integrity checked. Checkpoint-backed recomputation currently exposes the twelve 1D artifact computations associated with Heat-1 L2/W128 and Wave-1 L2/W256; the remaining checkpoints are present so the fixed 2D/3D closures can be added without changing model authority. The 2D/3D cases remain unavailable through the public CLI until their current-paper implementations, slow/GPU tests, and authority comparisons pass. `data/catalog.json` records the implemented recomputation scope as 12 of 36 computations and does not claim release completion or full GPU reproduction.

The supported execution boundary is fixed rather than generic: historical 1D initial-condition cases use CUDA float32 kernels (with a CPU float32 portability gate), while 1D PDE cases use CPU float32 kernels with float64 accumulation. The retained artifacts and portable replays are ordinary floating-point computations, not outward-rounded interval arithmetic. See `provenance/numerical-trust-boundary.md` and `provenance/checkpoint-reproduction-1d.json` for the exact comparison policy and producer-trace basis.

The frozen artifact/table authority on this branch predates the current-paper centered-moment initial-data and appendix-direct PDE recomputations now running in `NetBounds-dev`. Those validated outputs and final table bytes must replace the frozen artifacts before a public release; the old byte-exact replay remains a reproducible baseline, not a claim that the repository is publication-current.

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
