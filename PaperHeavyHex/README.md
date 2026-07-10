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

## Data sources

- C(t) and chi growth: `logs/chi_convergence/chi_{N}.log`
- Pauli weights: `logs/chi_convergence/spatial_chi{N}.dat`
- Lightcone figures: `lightcone_sites_spatial_site13.{png,gif}`
- Performance: `logs/chi_convergence/` + Julia comparison in Notes.md
