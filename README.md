# NetBounds

Code and data for the paper *Trustworthy AI in numerics: On verification algorithms for neural network-based PDE solvers*, by Emil Haugen, Alexei Stepanenko, and Anders C. Hansen.

## Install and verify

Requires Python 3.11 or newer. The install commands below read the project and test dependencies from [`pyproject.toml`](pyproject.toml).


```
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/netbounds-paper verify
.venv/bin/pytest
```


## Usage

```
netbounds-paper verify
netbounds-paper tables --check
netbounds-paper tables
netbounds-paper reproduce --case heat1-displacement-q0 --device cpu --check
```

- `verify` checks that the saved numerical results and tables are consistent.
- `tables --check` checks whether the saved tables are up to date.
- `tables` recreates the tables from the saved results.
- `reproduce` reruns one of the 36 calculations. With `--check`, it compares the result with the saved value.
- PDE calculations require `--allow-full-pde`. 3D initial-data calculations require CUDA and `--allow-large-initial`.

The five outputs are:

1. initial displacement bounds (Q0 and Q1),
2. initial displacement gradient bounds (Q0 and Q1),
3. initial velocity bounds (Q0 and Q1),
4. PDE residual Q1 bounds
5. combined energy estimate bounds.

## Scripts

`scripts/check_derivative_bounds.py` compares sampled derivatives of a tanh
network with the local derivative bounds used in the paper. It loads a trained
paper network when `n`, `L`, and `w` match one; otherwise it uses a seeded random
network. It checks every symmetric multi-index of the requested order and saves
one bar-chart panel per component in `scripts/plots/`.

```
.venv/bin/python scripts/check_derivative_bounds.py
.venv/bin/python scripts/check_derivative_bounds.py --n 4 --L 3 --w 256 --order 4 --eps 0.001
```

The default arguments are `n=2`, `L=2`, `w=128`, `N=100`, `Nsamples=10`,
`eps=1e-2`, and `seed=123`. `--order` selects derivative orders one through
four; `--plot` optionally overrides the default output filename.



## Layout

```text
src/netbounds/              code for numerics and table generation
data/catalog.json           list of files used to build the tables
data/artifacts/             JSON files containing quadrature results
paper/tables/               LaTeX tables
scripts/                    standalone derivative-bound checker and plots
tests/                      tests 
```

## Citation and license

Citation metadata is provided in `CITATION.cff`.
The software is released under the MIT License; see `LICENSE`.
