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
