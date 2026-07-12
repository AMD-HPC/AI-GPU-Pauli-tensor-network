# PaperHeavyHex

Short paper: CppSim extended to heavy-hex lattice.

## TODO list (fill in as data arrives)

- [ ] Chi convergence table (from plot_ct_convergence.py after run_chi_convergence.sh)
- [ ] ct_convergence.png figure
- [ ] lightcone_sites_spatial_site13.png (selected cycles: 1,5,10,15,20,25,30)
- [ ] timing_vs_chi.png
- [ ] chi_growth.png
- [ ] CppSim vs Julia comparison table (chi=430, cycles 8-10)
- [ ] Convergence threshold chi value
- [ ] Lightcone boundary cycle number
- [ ] References (CppSim paper, IBM heavy-hex, MPF)
- [ ] Acknowledgments

## Build

```sh
pdflatex heavyhex.tex
bibtex heavyhex
pdflatex heavyhex.tex
pdflatex heavyhex.tex
```

## Key findings so far (2026-07-10)

### C(t) convergence
- χ=150 gives converged C(t) at cutoff=1e-4 — error < 1.45e-4 vs χ=300 reference
- χ=200+ identical to 5 decimal places
- **cutoff drives physics, cap prevents runaway memory**
- C(t) is NOT a decay — it's a quantum oscillation (XXX model SU(2) symmetry)
  - Fast drop cycles 1-4 (operator spreading)
  - Valley at cycle 4-5 (maximum spread)
  - Recovery cycles 7-9 (wavefront reflection from boundary)
  - Quasi-periodic oscillation cycles 10+ (finite-size recurrence)

### Chi_min table (lightcone boundary)
- chi_min=1 for cycles 1-13: some bonds still product state (outside lightcone)
- chi_min=4 at cycle 14: **lightcone reaches last bond** — entire graph entangled
- chi_min grows identically across ALL chi caps until cap starts to bite
- cap=50,100,150 fully saturate by cycles 24,27,28 respectively
- cap=200,250 will saturate; cap=300 will not in 30 steps (cutoff controls)

### Lightcone visualization
- Site 13 (mid-graph junction): most symmetric, 3-fold hexagonal spreading
- 20 steps reaches outer boundary of 3x3 graph
- GIF and multi-panel PNG generated for site 2, 13, 22

## Cholesky QR experiment (2026-07-11)

Ran `run_chi_convergence.sh 30 0.1 1.0 1e-4 --qr=adaptive` on all 9 chi values.

**Results:**

| χ | Householder t_last (s) | Adaptive t_last (s) | Speedup | C(t) correct? |
|---|----------------------|---------------------|---------|----------------|
| 200 | 129.5 | 129.4 | ~1x | ✓ |
| 250 | 203.3 | 205.2 | ~1x | ✓ |
| 300 | 398.3 | 293.1 | **26%** | ✓ |
| 350 | 504.7 | 410.6 | **19%** | ✓ |
| 400 | 857.0 | crashed | — | ✗ GPU hang |
| 430 | 1153.9 | 320s then wrong | — | ✗ C(t=20)=0.205 (should be ~0.075) |

**Root cause: float32 Gram matrix accumulation error**

Cholesky QR forms G = A'A via SGEMM. At chi=430:
- m = 740,000 rows accumulated per Gram matrix entry
- Float32 rounding: ε × m ≈ 1e-7 × 740,000 = **0.07 absolute error per entry**
- κ(G) = κ(V)² squaring amplifies any mild ill-conditioning in V
- G becomes indefinite in float32 → SPOTRF either fails (fallback) or
  succeeds with wrong result → **wrong physics**, then GPU hang

**Conclusion:**
- Adaptive QR is **safe and beneficial up to χ≈300** (~20-26% speedup)
- At χ≥400, float32 Gram matrix is too ill-conditioned for Cholesky
- **Householder is the correct choice for the paper** (used in all published results)

**The fix (Phase 2 of CppSim roadmap):**
Form G in float64 via `rocblas_gemm_ex` with F64 accumulators, then DPOTRF,
then STRSM back in float32 for Q. This is the mixed-precision Cholesky that
would make adaptive QR safe at all χ. Non-trivial but well-understood.
See FUTURE_CHECKPOINT.md Phase 2.

## CppSim vs Julia parameter matching — OPEN QUESTION (2026-07-12)

### Julia parameters (from compute_overlaps_gpu.jl)
- `epsilon_ref = 0.25` — Hamiltonian coupling scale
- `k_value = 1` → `dt = 1.0/k = 1.0` — time step
- `phi = 1.41371669` — kick gate angle
- Gate decomposition: `multi_xx_plus_yy_plus_zz_gate_new` → 4 sublayers:
  `(Ry+Rx) → Rzz(α) → (Ry+Rx) → Rzz(α) → Rx → Rzz(Δ)` per layer
- Three gate layers (colors 0,1,2) + kick gates reversed
- Observable: site 3 (1-based) = site 2 (0-based), Z operator
- Noise: PTM from JSON, gamma=1.0, applied after each sublayer
- cutoff=1e-4, no chi cap

### CppSim parameters (current)
- `J = 1.0`, `dt = 0.1` → effective angle θ = J*dt = 0.1
- Gate: single full PTM `exp(-iθ(XX+YY+ZZ))` applied once per color class
- No kick gates
- Observable: site 2 (0-based), Z operator
- Noise: PTM from JSON, gamma=1.0, applied after hardware gates per color class
- cutoff=1e-4, chi_max=430

### Key differences causing chi mismatch
1. **Different effective time step**: Julia's effective angle per step is ~0.25
   (epsilon * coupling * dt), CppSim uses J*dt=0.1. Different entanglement growth rate.
2. **Different gate decomposition**: Julia applies 4 sublayers of single-qubit + Rzz
   gates per color class. CppSim applies one full XX+YY+ZZ PTM. Not equivalent.
3. **Kick gates**: Julia adds kick gates (Rz rotations) after each Trotter step.
   CppSim has no kick gates. Different physics.
4. **Noise PTM application**: Julia applies noise after each sublayer (4x per color).
   CppSim applies once after all hardware gates per color. Different noise accumulation.

### What we need to reproduce Julia's noisy chi scaling
To see χ_noisy ≈ 166 at cycle 10 (as in Julia):
1. Match the gate decomposition exactly (implement Ry/Rx/Rzz sublayers in CppSim)
2. Match the effective coupling (epsilon=0.25 scaling)
3. Add kick gates
4. Apply noise after each sublayer (not once per color class)

**OR**: accept that CppSim computes different physics and use it as a standalone
MPF engine with its own parametrization — not required to match Julia exactly.

### Data needed for debugging
Run Julia with `--spatial` output for noisy state to see per-site Pauli weights
and confirm chi saturation. Compare with CppSim noisy at matched parameters.

## Mixed precision roadmap (2026-07-11)

Three levels of mixed precision being considered:

### Level 1: fp16 tensor storage (discussed 2026-07-10)
Store psi in fp16, compute in fp32. Halves state tensor memory.
- chi 430→600 on 32GB GPU
- Implementation: `__half*` psi[], upcast before gate, downcast after
- Safe: singular values below fp16 ε (~1e-3) are truncated anyway at cutoff=1e-4

### Level 2: float64 Gram matrix (Phase 2 roadmap)
Compute A'A in float64, Cholesky in float64, TRSM back to float32.
- Enables Cholesky QR at large χ without physics errors
- 20-26% speedup at χ=300+, more at higher χ
- Implementation: `rocblas_gemm_ex` with F64_R accumulators + DPOTRF

### Level 3: combined (future)
fp16 tensors + float32 gates + float64 Gram = maximum chi range with correct physics.
- Chi could reach 600+ on 32GB GPU with correct Cholesky QR
- The "right" architecture for production use

## Future directions (discussed 2026-07-10)

### fp16 tensor storage (mixed precision)
Store `psi` tensors in fp16, keep gate computation in fp32:
- State tensors: 16.5 GB → 8.25 GB (-50%)
- Messages: 97 MB → 49 MB (-50%)
- Workspace: unchanged (fp32, ~5 GB)
- Measures: unchanged (fp64, ~4 GB)
- **Net: ~17 GB vs ~25 GB → chi 430→600 on MI100**

Pipeline: load fp16 psi → upcast to fp32 → gate compute → downcast to fp16 store.
Precision safe: cutoff=1e-4 truncates singular values < 1e-4; fp16 ε~1e-3 only
affects values that would be truncated anyway. Meaningful singular values are O(1).

Implementation: change `Scalar* psi[]` from `float*` to `__half*`, add
upcast/downcast wrappers in `absorb_env` and step 10 of `apply_binary_gate`.

### Pauli basis vs computational basis
Pauli basis (D=4, real F32) vs computational basis (D=2, complex F32):
- **Memory**: identical — 4×4B = 2×8B = 16 bytes per element group
- **Theory**: complex multiply = 4 muls + 2 adds; Pauli has 4× more elements
  → same multiply count, more additions
- **Practice**: SGEMM (real) fully exploits GPU tensor cores; CGEMM does not
  → 20-40% throughput advantage for Pauli basis confirmed empirically
  → fp16 advantage would be similar: HGEMM >> complex fp16 GEMM

### Julia Pauli basis removal
ITensors.jl planning to remove Pauli basis (`siteinds("Pauli")`).
CppSim is unaffected — Pauli basis is the native representation (D=4 compile-time).
This makes CppSim the long-term reference implementation for PTM-based simulation.
The Julia fork must be maintained frozen at the current commit for the MPF work.

## Data sources

- C(t) and chi growth: `logs/chi_convergence/chi_{N}.log`
- Pauli weights: `logs/chi_convergence/spatial_chi{N}.dat`
- Lightcone figures: `lightcone_sites_spatial_site13.{png,gif}`
- Performance: `logs/chi_convergence/` + Julia comparison in Notes.md
