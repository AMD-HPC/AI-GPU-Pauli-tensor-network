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

## Data sources

- C(t) and chi growth: `logs/chi_convergence/chi_{N}.log`
- Pauli weights: `logs/chi_convergence/spatial_chi{N}.dat`
- Lightcone figures: `lightcone_sites_spatial_site13.{png,gif}`
- Performance: `logs/chi_convergence/` + Julia comparison in Notes.md
