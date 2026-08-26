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



## Layout

```text
src/netbounds/              code for numerics and table generation
data/catalog.json           list of files used to build the tables
data/artifacts/             JSON files containing quadrature results
paper/tables/               LaTeX tables
tests/                      tests 
```

## Citation and license

Citation metadata is provided in `CITATION.cff`.
The software is released under the MIT License; see `LICENSE`.
