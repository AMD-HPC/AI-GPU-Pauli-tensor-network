# CHECKPOINT 15 — 2026-05-16 (paper read-through + implementation audit)

## What was done this session

### Paper corrections — stale "future optimization" claims fixed

| Location | Was | Now |
|----------|-----|-----|
| Cholesky-QR paragraph | "future optimization: replace with STRSM" | STRSM already implemented; only D2H is 4-byte potrf_info |
| Householder paragraph | "future optimization: extract R on-device" | extract_upper_triangle kernel already implemented |
| BP normalization | "D2H χ scalars, future: snrm2/sscal POINTER_MODE_DEVICE" | Already on-device via rocblas_sdot + normalize_and_delta kernel |
| BP convergence | "D2H χ² scalars" | Corrected to "one float (4 bytes) per sweep via atomicMax + d_delta" |

### Paper additions — optimality and memory framing

- **Design section opening**: "maximum possible size" → "worst-case size (at χ_max)"; added sentence clarifying this is "minimum allocation that sustains maximum parallelism" with O(χ⁴) lower bound argument
- **Alignment claim**: added sentence — leading dimensions fixed at startup → can be padded to HBM cache-line boundaries at allocation time; benefit realized at saturation only
- **Tensor permutation**: added appositive — "the complete site tensor Ψ_v, carrying the full bond-index state of a single lattice site" — anchors the 100MB/9.6GB numbers to the physics object
- **Intro optimality paragraph**: added "asymptotically optimal" claim with O(χ⁴) lower bound argument and "six of nine buffers reused" constant-factor claim

### workspace.hpp — buffer reuse analysis added

Added `BUFFER REUSE ANALYSIS` block to header comment:
- cv1/cv2, A1/A2, Q1/Q2 dual roles documented
- RR4↔U aliasing opportunity identified (saves one medium buffer, not done for clarity)
- O(χ⁴) optimality argument stated

### workspace.hpp — Cholesky diagnostic counters

Added three counters to `Workspace` struct:
- `chol_attempted` — Cholesky path entered
- `chol_fallback` — SPOTRF failed → fell back to Householder
- `chol_skipped` — matrix too small to attempt Cholesky

Printed in `ws.print()` and at end of run in `main.cpp`.

**MI100 result at chi=50:** `attempted=248  fallback=155  skipped=712`
- **62% fallback rate** — SPOTRF fails frequently in float32 at chi=50
- Root cause: Julia uses Float64 (zero fallbacks), CppSim uses float32; Gram matrix A^T A loses precision with m=125000 terms, ε=1e-8 insufficient
- Fallback is numerically clean — householder_qr takes original A (const), fresh start
- Paper footnote needs update: "zero fallbacks" scoped to Julia Float64, not CppSim float32

### gate.hpp — QrMode enum + --qr= flag

Added `QrMode` enum: `Adaptive | Cholesky | Householder`

`GateOps::qr_mode` member (default: Adaptive). `adaptive_qr` now respects the mode.

CLI: `--qr=adaptive|cholesky|householder` (same pattern as `--bp=`).

Printed in header: `QR: adaptive` / `QR: cholesky (forced)` / `QR: householder (forced)`.

**Use cases:**
- `--qr=householder` → clean timing baseline, zero Cholesky overhead
- `--qr=cholesky` → force Cholesky to measure raw fallback rate
- `--qr=adaptive` → production default

### gate.hpp — SVD sync point documented

Added `SYNC POINT` comment block at the S D2H (Step 7):
- Explains why it's unavoidable (data-dependent truncation rank)
- Documents the on-device reduction strategy to eliminate before adding streams

---

## Communication inventory (current state)

| Transfer | Size | Avoidable? |
|----------|------|-----------|
| Gate tensor G H2D (Step 6) | 256 bytes | No |
| potrf_info D2H (Cholesky path) | 4 bytes | No (branch condition) |
| S D2H → new_chi (Step 7) | ≤6.4 KB | Yes — on-device reduction (streams blocker) |
| d_delta D2H (BP convergence) | 4 bytes/sweep | No (loop control) |
| BP normalization | 0 | Already on-device ✓ |

---

## Paper state

Content complete. All figures in. n* fixed. All stale "future optimization" claims corrected.
One footnote still needs update: Julia Float64 vs CppSim float32 Cholesky comparison.

### Remaining before submission
- Update Cholesky fallback footnote (Float64 vs float32 precision)
- MI300X experimental results (run_mi300x.sh ready)

---

## Previous checkpoints
- CHECKPOINT_14.md: figures session, paper content complete, run_mi300x.sh written
- CHECKPOINT_13.md: figures session, Δn analysis
- CHECKPOINT_12.md: tab:findings, §6.1–§6.3 complete
