# CHECKPOINT 20 — Paper figures and performance section complete

Date: 2026-05-17

## Summary
Full paper revision session. All stale references fixed, three new figures written
and inserted, performance section rewritten with real data. Paper now reflects
the complete MI100+MI60, Adaptive+Householder experimental picture.

---

## LaTeX fixes in experiments_section.tex

### Stale reference cleanup
- `\ref{tab:ct}` (×4) → replaced with `fig:ct_4x4`, `fig:ct_5x5`, `tab:trap_inventory`
- `\ref{fig:ct}` (×1) → `\ref{fig:ct_4x4}` with updated surrounding text
- `\ref{tab:symmetry_traps}` → removed (table deleted, data folded into prose)

### tab:trap_inventory — expanded to 4 columns
Added MI60 Adaptive column from `logs/baseline_20260514/mi60/` (previously missed):
- chi=20: L17 recovers
- chi=50: L20 **persistent**
- chi=80: L16, L18, L20 **persistent** (worst case — 3 traps, Householder chi=80 is clean)
Full 4-column layout: MI100 Adaptive | MI100 HH | MI60 Adaptive | MI60 HH

### tab:symmetry_traps — REMOVED
Redundant after trap_inventory. 5×5 final C(t) values folded into prose:
chi=20→0.000716, chi=40→0.000907, chi=60→0.001073, chi=80→0.028089, chi=100→0.042065

---

## Grid scaling section (experiments_section.tex)

### tab:grid_scaling — rewritten with real data
Old numbers were stale adaptive baseline. New 4-column table (ms, average over all layers):

| Grid | Bonds | MI100 Adap | MI100 HH | MI60 Adap | MI60 HH | MI60/MI100 |
|------|-------|-----------|---------|----------|--------|-----------|
| 3×3  | 12    | 2,167     | 2,678   | 10,656   | 11,935 | 4.5×      |
| 4×4  | 24    | 7,748     | 8,274   | 31,453   | 37,323 | 4.5×      |
| 5×5  | 40    | 12,802    | 16,316  | 91,044   | 83,090 | 5.1×      |
| 6×6  | 60    | 21,270    | 23,332  | 135,010  | 118,046| 5.1×      |
| 7×7  | 84    | 25,219    | 31,208  | 138,317  | 156,576| 5.0×      |
| 8×8  | 112   | 32,880    | 36,654  | 207,509  | 198,907| 5.4×      |

Key: MI100 Adaptive 10-20% faster than HH (Cholesky saves work when it succeeds).
MI60/MI100 ratio stable 4.5-5.4× across all grids.

### fig_grid_grid.png — new make_grid_figure.py
4 lines (MI100 HH, MI100 Adap, MI60 HH, MI60 Adap) + linear fits.
Solid=HH, dashed=Adap, blue=MI100, red=MI60. Circle=HH, square=Adap markers.
Table and figure combined into single figure float.

---

## Performance section (lightconesimulation.tex)

### Replaced: tab:julia_comparison + fig_breakdown.png (stale)
Old numbers: trotter 13,000ms (wrong), chi=120 (never run on HH), 1.5× speedup.
New: prose sentence with correct numbers — 8,300ms HH vs 20,000ms Julia → **2.4× speedup**.

### New fig_timing_chi.png — make_timing_figure.py
Two-panel log-log: Trotter (left) + BP (right) vs chi.
Filled=4×4, hollow=5×5. Power-law fits with exponent annotations.
Key findings:
- Trotter: ~chi^3.0-3.5 MI100, ~chi^3.5-4.0 MI60
- BP: steeper, diverges at chi≥80 where 100-sweep budget kicks in
- BP > Trotter at chi≥80 on MI100; chi≥70 on MI60

### New fig_breakdown_chi80.png — make_breakdown_figure.py
Two stacked panels: Trotter (top) + BP (bottom) per layer at chi=80, 4×4.
All 4 series. Ramp-up (blue) / saturation (gold) shading.
Caption explicitly states "same chi, same layers → fair architectural benchmark."
MI60 Adaptive BP spikes at L16, L18, L20 (persistent traps) visible.

---

## BP trap causal chain — saved to memory
Full three-layer story:
1. Asymmetry (necessary): even grid → multiple BP fixed points
2. Cholesky ill-conditioning (trigger): frequent fallback = near-degenerate state
3. Mixed gauge within layer (amplifier): Cholesky+HH mix breaks residual symmetry
Julia's silent Float64 Cholesky = always taken, uniform gauge → fewer traps

---

## Figures produced this session
- fig_ct_4x4.png — already done (CHECKPOINT 19)
- fig_ct_5x5.png — already done (CHECKPOINT 19)
- fig_grid_grid.png — **NEW** (make_grid_figure.py)
- fig_timing_chi.png — **NEW** (make_timing_figure.py)
- fig_breakdown_chi80.png — **NEW** (make_breakdown_figure.py)

## Scripts produced this session
- Paper/make_grid_figure.py
- Paper/make_timing_figure.py
- Paper/make_breakdown_figure.py

---

## Remaining / Next steps
- Update fig_timing_chi.png caption exponents once actual power-law values confirmed
- MI300X — still pending access
- Memory model section (tab:vram, tab:vram_grid) — check if stale
- Roofline figure (fig_eff_roofline.png) — check if stale
- tab:comparison (Julia vs CppSim summary at end) — may reference old numbers
