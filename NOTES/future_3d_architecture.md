# 3D Architecture — Memory and Computation Directions (2026-11-10)

## The core tension

3D cubic lattice (5×5×5, χ=50):
- GPU VRAM: 32GB → too small for interior site tensors (χ^6×D)
- CPU RAM: 503GB → fits with lightcone allocation at step 1
- GPU compute: needed for speed
- CPU compute: feasible but slow

The 3D lattice forces architectural questions that 2D sidestepped.
These are not 3D-specific — they apply to any large 2D lattice too.

---

## Direction 1: State on CPU, dispatch gates to GPU

**Concept**: store the full tensor network state in CPU RAM.
For each gate application:
1. Copy the two active site tensors from CPU → GPU (PCIe)
2. Apply Trotter gate on GPU (rocBLAS GEMM)
3. Copy result back CPU ← GPU (PCIe)

**BP messages**: also stored on CPU. Each BP sweep:
- For each bond (by color class): copy site tensors to GPU, compute message, copy back
- Or: batch multiple bonds per GPU transfer if memory allows

**Bottleneck**: PCIe bandwidth (~32 GB/s bi-directional).
For 5×5×5 at χ=50 with 125 sites and 7 active:
- Active site tensor: χ^6×D = 50^6×4 ≈ 60GB — too large for one GPU transfer!
- **Mode decomposition needed**: don't transfer full tensor, only the dominant components

**When this makes sense**: χ is moderate (10-20), active sites are few (lightcone),
bond-level transfers are small. Essentially the existing 2-GPU pipeline but with
CPU as the third tier for cold storage.

---

## Direction 2: Rank-k truncated BP messages

**Concept**: instead of full χ×χ BP messages, keep only the top-k eigenvalues.

For each bond message M (χ×χ):
1. Eigendecompose: M = A D A^T (already implemented in `diag_gauge_ip.hpp`)
2. Keep top-k eigenvalues, zero the rest
3. BP sweep with rank-k messages

**Memory**: instead of χ² per message, store only k×χ (eigenvectors) + k (eigenvalues).
For k=1: each message is O(χ) instead of O(χ²) — 50× smaller at χ=50.

**Key question**: does rank-k BP give physically correct n(v,t) for detection?

The DG gauge experiments (2026-08) showed:
- δ_eig=1e-3 beats QR at steps 4-5 by 2.4× on inner products
- The dominant eigenvalue direction carries the physically relevant information
- Near-zero eigenvalue directions carry loop-graph noise

For detection: we only need the sign of L(v,t) = -log(n_pert/n_base) to be correct.
If rank-k BP preserves the sign flip at the broken bond endpoints, detection works.

**Testable hypothesis in 2D**: try rank-1 or rank-2 BP on rect_7x7.
Compare PAULI_NL to full BP. If sign flips survive, 3D is feasible with rank-k BP.

**Implementation**: add `--bp-rank=k` flag to `main_pauli_sweep.cpp`.
In `measures.hpp`, modify `bp_converge_norm()` to truncate messages after each sweep.

---

## Direction 3: MPS along one axis (future)

For a 3D lattice L×L×L, treat it as L layers of L×L 2D lattices connected by
inter-layer bonds. Store each layer as a 2D tensor network, connect layers with
Matrix Product State bonds.

This keeps per-layer memory at O(χ^4) (2D) rather than O(χ^6) (3D).
The inter-layer bonds add O(χ^2×χ_MPS) where χ_MPS is the bond dimension
between layers — which can be kept small if layers are weakly entangled.

This is a significant architecture change — not for immediate implementation.

---

## Immediate plan: test rank-k BP in 2D

Before building 3D infrastructure, verify the core hypothesis:

1. Add `--bp-rank=k` flag (k=1,2,5,10)
2. Run rect_7x7 with rank-k BP
3. Compare PAULI_NL to full BP at each step
4. Run detection on mawgm_f64 trials — does 10/10 survive?

If rank-1 BP gives correct detection → 3D feasible with minimal memory.
If rank-5 needed → still 50× smaller than full BP at χ=50.

**Expected**: rank-1 may work at step 1 (clean wavefront, few active modes).
Step 2+ likely needs rank-3 to 5 as the wavefront spreads and more modes activate.

---

## Memory summary

| Configuration | Site tensor | Total active | Fits on GPU? |
|--------------|-------------|--------------|--------------|
| 2D 7×7, χ=50, full | χ^4×D = 25MB | ~1GB | ✓ |
| 3D 3×3×3, χ=20, full | χ^6×D = 400MB | ~4GB | ✓ (tight) |
| 3D 4×4×4, χ=20, full | χ^6×D = 400MB | ~8GB | ✗ OOM |
| 3D 5×5×5, χ=50, lightcone | 7×χ^6×D = 110GB | ~110GB | ✗ GPU, ✓ CPU |
| 3D 5×5×5, rank-1 BP | 7×χ×D + 7×χ = tiny | ~1GB | ✓ GPU |

The rank-k BP direction is the key to fitting 3D on GPU.

---

## Information-theoretic perspective

The rank-k BP experiment is not just an engineering optimization — it probes
the **information-theoretic lower bound** on what the simulation needs to
represent for detection to succeed.

The sign flip L(v,t) > 0 requires knowing n_pert(v) < n_base(v) — a binary
comparison, not the full value. The dominant eigenvalue of the BP message
encodes the strongest correlation in the environment. If the broken bond
perturbation projects onto the dominant mode, rank-1 BP captures it.

**The key question**: which eigenvalue of the BP message carries the broken
bond signal? This becomes a spectral analysis of the detection observable:

- If the signal is always in the top-1 eigenvalue: rank-1 BP is sufficient,
  detection is a low-rank phenomenon, and 3D is feasible at minimal memory
- If the signal is in top-3: rank-3 suffices, still 10-50× compression
- If scattered: full BP needed, detection requires high-rank environment

**The experiment design**:
1. Run rank-k BP for k=1,2,5,10 on rect_7x7
2. For each k: measure detection rate on mawgm_f64 trials
3. For failing bonds at each k: compare which eigenvalue carries their signal
4. Result: a spectral curve — detection rate vs rank-k

This is potentially a small paper of its own:
*"Minimum information for Hamiltonian fault detection: a rank-k BP analysis"*

The question is not "does it work?" but "why does it work, and what is the
minimum information required for it to work?" — connecting the empirical
detection results to the information geometry of the tensor network.
