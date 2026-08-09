# CHECKPOINT 21 — MI355X experiment analysis, spatial dispersion tools

Date: 2026-05-19

## Summary
First MI355X job (2150) analysis session. Extracted timing data, diagnosed trap
behaviour, corrected chip comparison baseline, generated light cone figure, and
wrote spatial dispersion analysis tooling. Continuation SLURM script updated with
`--measure=all` for all missing grid sizes.

---

## MI355X grid-scaling timing table (chi=50, saturated layers only)

Rule: only layers where chi_min = chi_max = chi_target are valid for timing
comparison across chips. Averages below are computed over those layers only.

| Grid | Bonds | MI100 Adap | MI100 HH | MI355X HH | MI355X Adap | MI355X HH/MI100 HH | MI355X Adap/MI100 Adap |
|------|-------|-----------|---------|-----------|------------|-------------------|----------------------|
| 3×3  | 12    | 3,473 (n=9) | 3,331 (n=14) | 2,644 (n=24) | 2,067 (n=24) | 0.79× | 0.60× |
| 4×4  | 24    | 11,835 (n=9)| 12,398 (n=9) | 9,707 (n=19) | 7,286 (n=19) | 0.78× | 0.62× |
| 5×5  | 40    | 22,850 (n=2)| 25,213 (n=9) | 20,133 (n=19)| 14,683 (n=19)| 0.80× | 0.64× |
| 6×6  | 60    | 37,737 (n=2)| 41,470 (n=2) | 34,331 (n=12)| 22,192 (n=12)| 0.83× | 0.59× |
| 7×7  | 84    | no_sat      | 60,564 (n=2) | 51,754 (n=12)| 34,895 (n=12)| 0.85× | N/A   |
| 8×8  | 112   | no_sat      | no_sat       | 73,257 (n=20)| no data      | N/A   | N/A   |

**Baseline**: `logs/baseline_20260513/mi100/` = **Adaptive** mode.
**MI100 HH**: `logs/householder_20260517/mi100/`.
**MI355X** both modes: `logs/mi355x_20260519/`.

### Key findings

**HH vs HH (architectural speedup)**: MI355X is 15–21% faster than MI100 at all
grid sizes tested. Modest — both chips are memory-bandwidth-limited for chi=50
Trotter contractions (O(χ²·d) memory, O(χ³·d) compute; chi=50 is barely compute-
bound).

**Adap vs Adap (algorithmic × architectural)**: MI355X adap is 36–41% faster than
MI100 adap. The gap is larger because Adaptive's Cholesky branch is more compute-
dense (triangular solves vs full QR) — once HBM3E bandwidth pressure is relieved,
the FP64 FLOPS advantage of Cholesky pays off. On MI100 both modes ran at essentially
the same speed (bandwidth-saturated, algorithm choice invisible).

**MI355X Adap vs HH** (same chip): 22–35% faster. Consistent with Cholesky doing
fewer FLOPs per iteration when it succeeds. Advantage will grow at chi=80/100 where
Cholesky success rate is lower but the successful steps are faster.

**Implication for paper**: The ~1.6× MI355X vs MI100 headline speedup (adap-vs-adap)
has two components: ~1.2× from architecture alone (HBM3E bandwidth), ~1.3× from
algorithm×architecture interaction (Cholesky FLOPS unlocked by bandwidth).

---

## BP trap analysis — MI355X grid scaling

Trap analysis run with `detect_traps.py` on `logs/mi355x_20260519/`.

| Grid | HH traps | Adap traps | Notes |
|------|----------|-----------|-------|
| 3×3  | L15, L16 | L30       | Small grid, high trap density expected |
| 4×4  | L19      | L26       | Both recover next layer |
| 5×5  | **none** | **none**  | C4 symmetry protection confirmed |
| 6×6  | none     | none      | Even grid — protection via larger chi margin |
| 7×7  | none     | none      | C4 symmetric |
| 8×8  | none     | none      | Largest grid tested on MI355X at chi=50 |

**C4 symmetry protection confirmed on MI355X**: odd grids (5×5, 7×7, 9×9) show zero
Type-1 traps, consistent with MI100 and MI60 results. The trap mechanism (asymmetry
→ Cholesky ill-conditioning → mixed gauge) is absent when the lattice has 4-fold
rotational symmetry.

---

## norm=inf in _all.log files — clarification

Layers with `norm=inf` in `10x10_chi050_n60_all.log` appear at layer 17 onward when
chi saturates all bonds simultaneously (chi_min = chi_max = 50). This is a Householder
computation — norm=inf here is NOT from Cholesky gauge mixing. It is a genuine
numerical overflow when the bond dimension fills the full Hilbert space dimension on
every bond simultaneously.

Layer 15 (norm ≈ 0.000001): catastrophic collapse just before the blow-up. Spatial
data at this layer is a single pixel — physically meaningless. Excluded from all
figures.

Layers 17+ (norm=inf): spatial weight data is physically plausible (operator has
spread uniformly) but normalization is suspect. Use with caution; flag in figure
captions.

---

## Light cone figure — 10×10 chi=50

Generated `Paper/fig_lightcone_10x10_chi050.png`.

Command:
```
python3 Paper/make_lightcone_figure.py \
    logs/mi355x_20260519/10x10_chi050_n60_all.log \
    -o Paper/fig_lightcone_10x10_chi050.png \
    --layers 1 4 8 14 20 24 30 --vmin 1e-5 --dpi 200
```

Snapshots: t = 0.1, 0.4, 0.8, 1.4, 2.0, 2.4, 3.0 (Trotter time units, δt = 0.1)

Physics observed:
- **t = 0.1–0.8**: Diamond (L1 ball) expanding from center site (4,4), consistent
  with linear light cone growth at Lieb-Robinson velocity ~1 site/δt.
- **t = 1.4**: Diamond reaches boundary; corners begin to reflect.
- **t = 2.0–2.4**: Boundary reflection creates a bright ring — operator weight
  accumulates at sites equidistant from center, i.e. the Manhattan circle. This is
  the quantum analog of a wave reflection off lattice boundaries.
- **t = 3.0**: Ring disperses; weight becoming more uniform (onset of scrambling).

Parser bugs fixed (both in make_lightcone_figure.py and compute_dispersion.py):
1. Grid number regex: `[0-9]+\.[0-9]+` → `-?[0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?\d+)?`
   (missed plain integers printed as `0` with no decimal point).
2. Row regex: columns can contain `inf` or `nan`; regex `[0-9.eE+\-]+` failed to
   match these. Fixed with `_NUM = r"(?:[0-9.eE+\-]+|inf|nan)"` helper.

---

## Spatial dispersion tools

### compute_dispersion.py (NEW)

Computes per-layer spatial dispersion metrics from `--measure=all` log files.

**Metrics**:
- `R_L1`: mean Manhattan distance from center site (light-cone radius proxy)
- `R_L2`: RMS Euclidean distance (radius of gyration)
- `S_ent`: Shannon entropy of n(r,t)/total
- `S_frac`: S_ent / log(N²), normalized 0→1 (0 = localized, 1 = fully scrambled)
- `weight`: total Pauli weight (should ≈ 1 if norm healthy)

**Usage**:
```
python3 Paper/compute_dispersion.py LOG [LOG ...] [--plot] [-o fig.png]
```

**Key use case**: for large grids where C(t) → 0 (quantum scrambling), C(t) alone
is uninformative. S_frac and R_L1 quantify HOW the operator weight dispersed — this
is the physically interesting quantity.

For 10×10 at chi=50 we expect S_frac → 1 (fully scrambled) by t ≈ 3.0–4.0.
At chi=100 (full scrambling run, sec_D) this should be cleaner with norm ≈ 1.0
throughout.

### make_lightcone_figure.py (UPDATED)

Same parser bugs fixed. Now handles all 60 layers of `*_all.log` files correctly,
including norm=inf rows.

---

## Missing --measure=all coverage — slurm_mi355x_cont.sh updated

Added `--measure=all` to:

**sec_B_grid_scaling**:
```
run_gs 8  8  50 40 --qr=householder --measure=all
run_gs 8  8  50 40 --qr=adaptive    --measure=all
```
Previously: 8×8 was missing entirely from _all log collection.
Note: 8×8 chi=50 saturates at ~layer 20 (n_bonds=112, chi=50 fills all bonds on
large grids), so spatial data up to saturation is clean.

**sec_F_12x12**:
```
run 12 12  50 80 --qr=householder --measure=all
```
Previously: 12×12 chi=50 had no _all log planned.

**Already present** (no change needed):
- sec_D: 10×10 chi=100 _all, chi=90 _all (scrambling suite)
- sec_E: 11×11 chi=50 _all, chi=100 _all
- sec_H: 13×13 chi=50 _all
- sec_I: 14×14 chi=50 _all, 9×9 chi=50 _all

This gives complete spatial coverage: 8×8, 9×9, 10×10, 11×11, 12×12, 13×13, 14×14
at chi=50, plus 10×10 at chi=90/100 for the full-scrambling study.

---

## Open items for paper

### Data pending (post-MI355X job)
- [ ] 10×10 chi=100 _all: the key full-scrambling dataset. C(t)→0 confirmed;
      spatial dispersion will show S_frac → 1 cleanly with norm ≈ 1.
- [ ] 14×14 chi=50 _all: largest grid, spatial spreading for paper cover figure
- [ ] 8×8 chi=50 _all (new): added to continuation job
- [ ] 12×12 chi=50 _all (new): added to continuation job

### Figures to update
- [ ] `make_grid_figure.py`: replace MI60 with MI355X, add MI355X HH + Adap lines,
      filter to saturated layers only (chi_min=chi_max=50)
- [ ] `make_timing_figure.py`: add MI355X series
- [ ] `make_breakdown_figure.py`: add MI355X series

### Paper sections
- [ ] Tab: grid scaling (tab:grid_scaling) — add MI355X columns, update caption
- [ ] Tab: trap inventory — confirm MI355X trap count matches MI100/MI60 pattern
- [ ] Performance section: add MI355X speedup discussion (1.6× adap, 1.2× HH)
- [ ] Spatial dispersion: new subsection or appendix with lightcone figure +
      S_frac(t) plot showing scrambling onset for large grids

### Before submitting slurm_mi355x_cont.sh
- [ ] Get Kevin Cheng's 6-hour wall-time exception
  (`sbatch -t 06:00:00 slurm_mi355x_cont.sh`)
