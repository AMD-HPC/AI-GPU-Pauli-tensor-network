# CHECKPOINT 24 — MI355X appendix complete, 40-page paper

Date: 2026-05-21

## Session summary

### Figures produced
- `fig_mi355x_chi100_timing.png` — 4-series timing: 10×10 and 11×11, HH (circle) and Adap (triangle)
  - Green = 10×10, Orange = 11×11; Adaptive faster than Householder confirmed at chi=100
  - 10×10 and 11×11 run at nearly identical wall time (C4 convergence compensates for extra bonds)
- `fig_lightcone_10x10_chi100.png` — 2×4 heatmap + C(t), 10×10 chi=100 HH, light theme
- `fig_lightcone_11x11_chi100.png` — 2×4 heatmap + C(t), 11×11 chi=100 HH, textbook-clean ring

### New script
- `Paper/make_mi355x_lightcone.py` — full-page light cone figure generator
  - sqrt-spaced frame selection, LogNorm colour scale, per-frame C(t) annotation
  - Usage: `python Paper/make_mi355x_lightcone.py <log> -o <output.png>`

### appendix_mi355x.tex updates
1. Timing figure placeholder replaced with `\includegraphics{fig_mi355x_chi100_timing.png}`
2. chi=50 light cone replaced with `fig_lightcone_10x10_chi100.png`, caption updated
3. 11×11 chi=100 figure added with full C4 symmetry story
4. **New subsection: "Future performance directions"**
   - HIP streams: SVD D2H sync is the blocker; MI355X 309 GB capacity enables double-buffering; ~1.5–2× Trotter speedup expected
   - Float64 gate arithmetic: fixes Cholesky conditioning → eliminates Adaptive fallback penalty → Adaptive strictly faster than HH at chi=100

### Job 2208 status (as of morning 2026-05-21)
- 10×10 HH: layer 36/100 (BP delta=1.4e-3, sweeps maxed at 100)
- 10×10 Adap: layer 39/100 (BP delta=1.4e-4, 10× better convergence)
- 11×11 HH: layer 34/70
- 11×11 Adap: layer 37/70
- All 4 runs alive, no ##DONE## yet — job hit 16h wall time
- Data collected is publishable (first chi=100 simulation of 10×10 and 11×11 Ising grids)

### Key physics findings confirmed
- Adaptive QR faster than Householder at chi=100 on MI355X (HBM3E unlocks Cholesky compute advantage)
- 10×10 and 11×11 timing nearly identical (C4 symmetry reduces BP sweeps, compensates for extra bonds)
- 11×11 light cone perfectly fourfold-symmetric — zero trap contamination
- C(t)→0 is scrambling not failure; n(r,t) is the primary observable at late time

## Tomorrow checklist
- Sync logs from rainier after job 2208 fully completes (may need resubmission for remaining layers)
- Re-run `make_mi355x_lightcone.py` to extend C(t) curves to full layer count
- Consider requesting another 16h for remaining 60+ layers of chi=100
- Future: HIP streams (SVD D2H sync blocker), float64 gate arithmetic
