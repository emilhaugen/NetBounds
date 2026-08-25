# Numerical trust boundary and recomputation policy

## What the retained artifacts establish

The six PDE artifacts enumerate their declared grids/leaves, bind to checkpoint and configuration hashes, apply analytic derivative/remainder envelopes, and accumulate additive squared quantities in float64.

They also explicitly record:

```text
ordinary_float32_not_outward_rounded = true
```

Local neural-network and derivative-bound calculations were performed in ordinary float32. This is not outward-rounded interval arithmetic. The minimal repository must preserve this statement verbatim and must not strengthen “certified” into a claim of end-to-end interval-rigorous floating-point computation.

The current trust boundary is therefore:

- mathematical envelope formulas and coverage checks are certificate-oriented;
- local floating-point evaluation is ordinary float32;
- additive accumulation is float64;
- publication display values are generated deterministically from retained JSON fields with directed decimal rounding;
- exact table replay is stronger than cross-hardware raw-artifact byte identity, but it does not repair the absence of outward rounding in the original computation.

## Published PDE field

The PDE table and PDE part of the energy table publish `moment_cross_l2`. They do not publish the slightly smaller `l2_bound`, which uses a per-cell minimum across multiple valid candidates. This choice must stay explicit in the table-lineage manifest.

## Two recomputation modes

### Fixed current-paper scope

The public `netbounds-paper reproduce` command is a closed 34-case interface, not a generic training or campaign API. It exposes all 30 initial-data cases and the four d=1/2 appendix-direct PDE cases; d=3 PDE remains gated. The comparator accepts exactly the declared output schema, validates the selected authority through the same table-policy validators, and compares every declared numerical field. Missing, empty, unknown, non-finite, or structurally changed fields fail closed.

The numerical gates are:

- CUDA initial-data replay: `rtol=5e-5`, `atol=5e-12`;
- CPU initial-data portability: d=1 `rtol=1.5e-4`, d=2 `rtol=3e-4`, both with `atol=5e-12`;
- CUDA PDE replay with float32 kernels and float64 accumulation: `rtol=5e-7`, `atol=5e-14`;
- CPU PDE portability: `rtol=5e-5`, `atol=5e-12`.

Initial-data comparisons include the complete fixed-time grid geometry, cell volume, midpoint and bound diagnostics, squared quantities, ratio, and cell count. PDE comparisons include every retained additive sum, derived norm and ratio, all cell radii, coverage counts, and retained `rho` diagnostics. Identity, checkpoint hash, architecture, rule, execution dtype, accumulation dtype, grid, and fixed case-specific PDE batch size are exact fields.

These gates assess ordinary floating-point computations only. They are not interval-enclosure or outward-rounding guarantees.

### Reference replay

Same locked container, software versions, GPU architecture, checkpoint bytes, grid, shard partition, and command configuration.

Requirements:

- all configuration, identity, coverage, integer, and Boolean fields exact;
- all normalized additive sums and derived norms bit-identical;
- ignored metadata restricted to the declared allowlist below;
- all five generated TeX files byte-identical.

### Portable recomputation

Used only when GPU architecture or low-level kernels differ while the frozen algorithm/configuration is unchanged.

Requirements:

- exact checkpoint/configuration/formula/method identities;
- exact grid, radii, shard intervals, coverage, counts, and policy flags;
- every finite additive squared quantity and derived norm satisfies `abs(new-ref) <= 1e-12 + 1e-6*abs(ref)`;
- all ordering and nonnegativity invariants hold;
- every publication value rounds upward to the exact current displayed table cell;
- no candidate sum or diagnostic may be dropped merely because the displayed cell agrees.

The portable tolerance is a provisional maximum, not a target. It must be calibrated downward from repeated clean runs before release. Widening it requires an explicit reviewed change and supporting rerun evidence.

## Exact fields

The following compare exactly:

- schema and kind;
- equation, dimension, architecture, datum family;
- checkpoint and configuration hashes;
- method identifiers and remainder/certificate identifiers;
- execution and accumulation dtypes;
- grid/radii and index ordering;
- processed/expected cells, roots, leaves, shards, and half-open coverage;
- adaptive depth/split/leaf counts for replay of the accepted policy;
- all `certified`, completeness, same-policy, and exact-replay qualification flags.

## Floating fields

Compare all of:

- midpoint and affine squared sums;
- moment remainder squared sum;
- moment cross integral and squared bound;
- Minkowski squared bound;
- selected per-cell-minimum squared bound;
- all derived square roots and ratios;
- method-specific diagnostics that contribute to acceptance.

Comparisons are made on additive squared quantities before square roots whenever both are present.

## Metadata allowlist

Only these fields may vary without scientific review:

- start/completion/merge timestamps;
- elapsed seconds and throughput;
- hostname, GPU UUID, process ID, and transient output path;
- non-semantic shard scheduling order when shard intervals and merged sums remain valid.

Unknown, missing, or newly added fields fail closed until classified.

## Heat-3D exception

The selected Heat-3D artifact is a same-policy re-evaluation and explicitly not an exact legacy partition replay. Portable reproduction must preserve that qualification. It may compare the accepted policy/tree against the selected artifact; it may not infer equality with the older historical leaf map.
