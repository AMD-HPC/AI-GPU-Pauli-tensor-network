# CHECKPOINT 22 — Appendix written, SLURM updated, job 2163 running

Date: 2026-05-19

## Summary
Full MI355X analysis session. Timing table extracted, trap analysis done, light cone
figure generated, spatial dispersion tooling written, appendix drafted, SLURM
continuation script updated with all missing --measure=all runs. Job 2163 running
on mi355x-thor-1 (4-hour cap, ~26 min remaining at checkpoint time).

---

## What was done this session

### slurm_mi355x_cont.sh — four additions
1. `sec_B_grid_scaling`: 8×8 chi=50 HH + Adap `--measure=all` (was missing entirely)
2. `sec_F_12x12`: 12×12 chi=50 HH `--measure=all`
3. `sec_D_scrambling`: 10×10 chi=100 n=100 Adap `--measure=all` (HH already present)
4. `sec_E_11x11`: 11×11 chi=100 n=70 Adap `--measure=all` (HH already present)

**NEW VERSION ON LOCAL MACHINE** — must be copied to ~/CppSim on rainier-login
before submitting job 3.

### Paper/compute_dispersion.py (NEW)
Spatial dispersion metrics from --measure=all logs.
Metrics: R_L1, R_L2, S_frac (spatial analogue of Se/logchi), weight.
Usage: `python3 Paper/compute_dispersion.py LOG [LOG ...] [--plot] [-o fig.png]`

### Paper/appendix_mi355x.tex (NEW) + lightconesimulation.tex updated
Appendix C `\label{app:mi355x}` added, \input'd from main file.
Sections:
- Hardware table (MI60 / MI100 / MI355X)
- Grid-scaling timing table (4-column, saturated layers only)
- BP trap table per grid
- Spatial operator spreading: n(r,t) defined by reference to §6.3,
  S_frac equation (spatial analogue of Se/logchi), fig_lightcone_10x10_chi050.png
- Pending paragraph (10×10 chi=100, 11×11 chi=100)
- Summary Finding/Evidence/Implication table

### lightconesimulation.tex — Outlook paragraph updated
Added streaming sentence: SVD sync point removal → 4 concurrent gate streams on
4×4 grid → multiplier best realised on MI300X (192 GB).

---

## Writing improvements made
- Removed defensive "This is correct physics, not a numerical failure" sentence
- Reframed C(t)→0 motivation: "at sufficiently long times on any finite lattice"
  (about time, not grid size) — cleaner and more honest
- Removed redundant equations for R_L1, R_L2 (described in words instead)
- n(r,t) defined by cross-reference to §6.3 (first definition in paper), not
  re-introduced in appendix
- S_frac framed as "spatial analogue of Se/logchi" — reader already knows the pattern
- Bandwidth comparison (6× HBM3E → only 1.2× speedup) left in for now; user flagged
  as potentially sensitive (ROCm version confound + flagship comparison optics).
  To fix before final submission: note ROCm 6.2/7.0/7.2 confound, remove raw
  bandwidth-to-speedup ratio framing.

---

## Key scientific points in appendix

### Two-component speedup decomposition
- HH-vs-HH: ~1.2× (architecture only, HBM3E bandwidth)
- Adap-vs-Adap: ~1.6× = 1.2× (architecture) × 1.3× (Cholesky FLOPS unlocked)
- Adap advantage is hardware-generation-dependent: neutral on MI100 (bandwidth
  saturated), 22-35% on MI355X (HBM3E relieves bandwidth pressure)

### Why 6× bandwidth → only 1.2× speedup at chi=50
At chi=50 the matrix sizes (QR input 200×50 = 40 KB, SVD input 200×200 = 160 KB)
fit in L2 cache. Neither chip drives HBM at peak — both are latency/occupancy
limited for small matrices. The bandwidth advantage is simply not exercised.
At chi=100 (matrices 4× larger, starting to spill L2) the advantage will grow.

### C4 symmetry confirmed on MI355X
Odd grids trap-free across MI60, MI100, MI355X — hardware-independent. Trap layer
position shifts between chips (MFMA reduction order differs) but rate is same.

---

## Job status at checkpoint
- Job 2163: running on mi355x-thor-1, 3h34m elapsed, ~26 min remaining (4h cap)
- Likely complete: GPU 0 (grid scaling), partial GPU 1/2 (chi-conv)
- Likely incomplete: GPU 3 (10×10 chi=100 n=100 — estimated 5-6h alone)
- EXIT trap syncs all completed logs to ~/CppSim/logs/mi355x_20260519/ on exit

## Next steps (after lunch)
1. Check which logs landed in logs/mi355x_20260519/ after job 2163 ends
2. Copy updated slurm_mi355x_cont.sh to rainier-login ~/CppSim/
3. Submit job 3: `sbatch -t 08:00:00 slurm_mi355x_cont.sh` (if Kevin approved 8h)
   or `sbatch slurm_mi355x_cont.sh` (4h default, will need job 4)
4. Run compute_dispersion.py on available _all.log files
5. Update make_grid_figure.py with MI355X data
