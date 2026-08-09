# Future Checkpoint — Next Development Phase

This file is a briefing for the next Claude session. Read it before touching any code.

---

## What this project is

CppSim is a C++/HIP implementation of real-time quantum dynamics for the 2D
transverse-field Ising model using Tensor Network methods in the Heisenberg picture
on AMD GPUs. It computes the spin autocorrelation function C(t) = <Z_c(t) Z_c(0)>
via Belief Propagation on the Bethe lattice approximation.

**Paper**: `Paper/lightconesimulation.tex` (41 pages, complete, committed).
**Primary results**: first χ=100 simulation of 10×10 and 11×11 grids on AMD MI355X
(309 GB HBM3E). Paper is in author review as of 2026-05-22.

---

## Build — CRITICAL

```bash
make clean && make main PHYS_DIM=4
```

**PHYS_DIM=4 is mandatory**. The Makefile default is also 4, but always pass it
explicitly. A build without it compiles silently but produces wrong physics
(static_assert will catch it, but only at runtime).

---

## Three planned development phases (in order)

### Phase 1 — HIP Streams (parallelise gate application)

**Goal**: run 4 concurrent gate streams on a 4×4 grid (one per independent colour
of the Trotter checkerboard). Expected ~1.5–2× Trotter speedup.

**The one blocker**: Step 11 of `apply_binary_gate` in `include/gate.hpp`.

```cpp
// Step 11 — THIS IS THE ONLY REMAINING CPU SYNC POINT
// Singular values S are copied D2H to compute chi_new and inv_snorm on CPU
// Must be replaced with an on-device reduction before streams can be added
rocsolver_sgesvd(...);
hipMemcpy(hS, dS, ...);          // ← D2H sync: blocks all streams
chi_new = compute_truncation(hS); // ← CPU loop over kfull floats
inv_snorm = ...;
```

**Fix**: replace the D2H + CPU loop with:
1. A HIP kernel that computes `chi_new` on-device (count values above threshold)
2. A HIP kernel that computes `inv_snorm` on-device (sqrt of sum of squares)
3. Pass both as device scalars to the subsequent `build_M_from_U` /
   `build_M_from_Vt` kernels

Once Step 11 is on-device, `apply_binary_gate` becomes a pure GPU function with
no CPU sync points, and a `WorkspacePool` with 4 independent workspaces can run
4 gates concurrently on separate streams.

**WorkspacePool design** (not yet implemented):
- 4 copies of `Workspace` (one per stream/colour)
- Outer loop dispatches gates by colour; same-colour gates are sequential,
  different-colour gates are concurrent
- MI355X 309 GB makes 4× workspace allocation straightforward (~1.6 GB × 4 at χ=100)

### Phase 2 — Float64 gate arithmetic (eliminate Adaptive QR fallback)

**Goal**: compute gate Trotter expansion (Rz, Rxx matrices) in float64, then
downcast to float32 for storage. This fixes two compounding precision problems
in the Cholesky-QR path:

1. **GEMM accumulation error**: forming G = V^T V via SGEMM accumulates m ≈ 10^6
   terms at χ=100. Float32 floor (ε ≈ 10^-7) gives absolute error ~10^-1 per entry
   of G — enough to destroy positive-definiteness and trigger the Householder fallback.

2. **κ(G) = κ(V)^2 squaring**: any mild ill-conditioning in V is squared in the
   Gram matrix.

**Fix**: use `rocblas_gemm_ex` with `ROCBLAS_DATATYPE_F64_R` accumulators for the
Gram matrix formation, plus `DPOTRF` for the Cholesky factor. Tensor storage stays
float32. This eliminates the Householder fallback entirely and makes Adaptive QR
unconditionally safe at all χ.

**Why this matters**: on MI355X, Adaptive QR is already 22–35% faster than
Householder (HBM3E exposes the Cholesky compute advantage). Float64 gate arithmetic
removes the last remaining reason to fall back to Householder.

### Phase 3 — Honeycomb lattice

**Goal**: extend from rectangular 2D grid to honeycomb lattice geometry.

**Current scope**: noise-free Hamiltonian evolution only (no noisy gates planned).

**What changes**:
- `include/grid.hpp`: new lattice type with degree-3 coordination (vs degree-4
  for rectangular interior sites)
- Bond structure: honeycomb has 3 bonds per interior site, not 4
- Gate application: same `apply_binary_gate` logic; only the bond topology changes
- BP: same message-passing; degree-3 reduces BP cost per sweep
- Center site: must be chosen carefully for C4-equivalent symmetry where applicable

**Key reference**: the C4 symmetry story (odd grids trap-free on rectangular lattice)
will have a different character on honeycomb — document whether the analogous
symmetry argument holds.

---

## Architecture decisions already made (do not revisit)

| Decision | Choice | Reason |
|----------|--------|--------|
| QR mode default | Householder | Uniform gauge; safe across all GPU generations |
| Adaptive QR | Use on MI355X | 22–35% faster; safe with float64 fix (Phase 2) |
| BP messages | Diagonal (chi-vector) | Correct Bethe approximation for bilinear C(t) form |
| Float precision | float32 storage | Sufficient; Vidal norm prevents overflow |
| PHYS_DIM | 4 (Pauli basis) | Required; do not change |

---

## Key files

```
include/gate.hpp        — apply_binary_gate: the hot path; Step 11 is the streams blocker
include/measures.hpp    — autocorrelation_with_bp, bp_update, BP convergence
include/workspace.hpp   — pre-allocated GPU buffer pool; buffer reuse map in header comment
include/grid.hpp        — lattice topology; start here for honeycomb
include/permute_gpu.hpp — HIP tensor permutation kernel (7.6× speedup; do not touch)
main.cpp                — CLI flags: --qr=, --bp=, positional args
```

---

## Buffer reuse map (workspace.hpp)

At χ=100 the workspace is ~1.6 GB. Key aliasing opportunities already documented
in `workspace.hpp` header comment. The O(χ^4) lower bound argument shows the
current allocation is asymptotically optimal. Do not add new large buffers without
checking the reuse map first.

---

## BP trap summary (do not re-investigate)

- **Type 1** (metastable): sweeps ≥ 30 AND delta < ε_tol → converged to wrong fixed point
- **Type 2** (non-convergence): sweeps = 100, delta >> ε_tol → BP failed to converge
- **C4 symmetry**: odd grids on rectangular lattice are trap-free at χ=50 (confirmed
  on MI100, MI60, MI355X). Even grids trap at early layers; retry with different α.
- **C(t)→0 at late time on large grids**: correct physics (scrambling), not a failure.

---

## MI355X cluster (Rainier)

- Login: `rainier-login` (AMD ntid/password)
- Partition: `mi355x` (nodes: thor-1..4, pollara-1)
- No module system — ROCm in PATH on compute nodes
- `/scratch` = local NVMe per node; use for all log writes, sync to `/home` on exit
- Submit: `cd ~/CppSim && sbatch slurm_mi355x_chi100.sh`
- Job 2208 (chi=100, 4 GPUs, 16h) completed 38/70-100 requested layers (wall time limit)
- Logs synced to `logs/mi355x_20260519/`
