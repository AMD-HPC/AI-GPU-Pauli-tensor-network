# Chi + v_hop Budget Optimization (2026-11-14)

## Key insight

chi saturates at step 3 even at chi=40 with 20% VRAM (period-6 + v_hop=0.9).
This means the chi budget is not fully used — we can push chi higher at the
same sparsity to get better accuracy.

## Multi-center lightcone union (new feature)

`include/grid.hpp`: new overload `apply_lightcone(const std::vector<int>& centers, ...)`

Algorithm:
1. Deactivate all bonds: `chi_alloc = 1`
2. For each center: BFS, activate bonds in cone (OR rule)
3. Union is monotonic — never deactivate once activated

Wired in `examples/main_pauli_sweep.cpp`: when `centers.size() > 1`,
calls multi-center overload automatically.

## Chi sweep results (period-6, v_hop=0.9, bond 200)

| chi | VRAM | Step 3 time | BP residual | δlnT score | Detection |
|-----|------|-------------|-------------|------------|---------|
| 40  | 20%  | 63s         | 135         | 5.86       | ✓ |
| 60  | 98%  | TBD         | TBD         | TBD        | TBD |
| 70  | OOM  | —           | —           | —          | — |

chi=60 is the maximum at chi=70 OOM. 98% VRAM is tight but runs.

## Operating points for 16x32 period-6 seismic

| chi | v_hop | VRAM | Use case |
|-----|-------|------|---------|
| 40  | none  | 53%  | fast seismic, no savings |
| 40  | 0.9   | 20%  | production: good BP, good detection |
| 60  | 0.9   | 98%  | max chi: best accuracy, tight memory |

## Script

`scripts/run_seismic_period6_chi_16x32.sh <bond> <J_pert> [trial] [chi] [v_hop]`

Parameterized for chi sweeps. Same 2-scan period-6 layout, 2-GPU parallel.

## Next

1. Check if chi=60 improves detection score vs chi=40
2. Run full 63-trial scaling study at chi=60 — does 94% become 97%+?
3. 2-GPU simulation for 1024 qubits (32x32): same period-6, 2 GPUs split grid

## Chi=60 result (2026-11-14)

Score: 5.84 at chi=60 vs 5.86 at chi=40 — essentially identical.

**Conclusion**: δlnT is insensitive to chi once the wavefront is faithfully
represented. The ratio n_base/n_pert cancels the absolute scale — same argument
as why δlnT is scale-invariant across grid sizes.

**Production operating point**: chi=40 + v_hop=0.9 + period-6
- 20% VRAM, 63s/scan, score 5.86, 94% single-bond detection
- Higher chi wastes memory and time with no detection benefit

## Full comparison table (period-6, scan 1, bond 200, step 3)

| chi | v_hop | VRAM | Step 3 | BP iters/residual | gate_err | δlnT score | Detection |
|-----|-------|------|--------|-------------------|----------|------------|---------|
| 40  | none  | 53%  | 81s    | 100 / 3.72e+02    | 6.45e-05 | 5.49       | ✓ |
| 40  | 0.9   | 20%  | 63s    | 100 / 1.35e+02    | 6.46e-05 | 5.86       | ✓ |
| 60  | 0.9   | 98%  | 84s    | 100 / 4.30e+02    | 1.60e-05 | 5.84       | ✓ |

## gate_err as chi quality indicator

gate_err drops from 6.5e-05 (chi=40) to 1.6e-05 (chi=60) — chi=60 represents
the internal dynamics 4× more accurately. This matters for C(t) and deep interior
physics, but NOT for detection.

**Why**: detection reads L(v,t) at the frontier where chi≈1 by definition.
The frontier is the first-arrival wavefront — Pauli weight is just crossing the
threshold, the state is barely entangled. BP messages at the frontier are rank-1.
Higher chi improves the interior (gate_err) but the frontier is chi-free.

This is the fundamental reason δlnT is chi-invariant for detection:
the signal lives at the frontier, the frontier lives at chi=1,
chi is irrelevant to anything measured at chi=1.

**Practical implication**: use chi just large enough to faithfully propagate
the wavefront to the frontier (chi=40 suffices). Save the chi budget for
other uses (larger grid, more steps, more centers).
