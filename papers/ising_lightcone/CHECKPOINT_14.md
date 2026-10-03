# CHECKPOINT 14 — 2026-05-16 (paper complete + MI300X prep)

## What was done this session

### make_diff_figure.py — fixes
- Added `--row1-label` / `--row2-label` CLI args (override hardcoded "(trap)"/"(clean)")
- C(t) legend now uses the same labels (was hardcoded "χ=N (trap)")
- Suptitle auto-adds "(BP trap)" only when `--trap-layer` is set, otherwise generic
- Fixed 5×5 figure: chi=80 as LOG1, chi=50 as LOG2, no trap-layer, explicit labels

### Final figures produced
| File | Description |
|------|-------------|
| `fig_lightcone_trap.png` | 4×4 chi=80, 7 snapshots, "trapped" banner at L17, red × on C(t) |
| `fig_lightcone_clean.png` | 4×4 chi=40, clean monotone decay |
| `fig_lightcone_9x9.png` | 9×9 chi=20, correct center (4,4), C4 symmetric, C(t)→0 |
| `fig_diff_4x4.png` | Δn: chi=80 vs chi=40, asymmetric, rearrangement at L17 |
| `fig_diff_5x5.png` | Δn: chi=80 vs chi=50, C4-symmetric control, no anomaly |

### experiments_section.tex — figures inserted
All 5 figures inserted into §6.3 (Operator Spreading and Lightcone Saturation):
- `fig:lightcone_trap`, `fig:lightcone_clean`, `fig:lightcone_9x9` — heatmap strips
- New paragraph "Spatial χ-convergence: the difference field Δn(r,t)" with `fig:diff_4x4`, `fig:diff_5x5`
- Key scientific framing: 5×5 shows pure χ-convergence (C4 symmetric), 4×4 shows χ-convergence + trap asymmetry

### Float placement fix
- Added `\usepackage{placeins}` to lightconesimulation.tex
- Added `\FloatBarrier` before Grid Scaling and Memory Model subsections
- Fixes: Table 11 (tab:grid_scaling "Trotter time per layer") was drifting to end of paper

### n* inconsistency — RESOLVED
- **Root cause**: equation used v_LR=|J|, table used v_LR=2|J|; 3×3 entry was the odd one out
- **Fix**: equation changed to n*=d_max/(2|J|δt)=5·d_max; 3×3 table entry corrected 20→10
- All table entries now consistent with v_LR=2|J| (correct for Heisenberg model)

### Repository URL — RESOLVED
- Replaced `\TODO{repository URL}` with "available from the corresponding author upon request"

### run_mi300x.sh — written
Location: `CppSim/run_mi300x.sh` (211 lines)
Sections:
| Section | Experiments | Peak VRAM |
|---------|-------------|-----------|
| A | 4×4 chi=50,80; 5×5 chi=80 (validation) | 6.5 GB |
| B ★ | 10×10 chi=50,70,90,100,110 (primary target) | ~159 GB |
| C | 10×10 chi=90,100 n=80 --measure=all | 109 GB |
| D | 11×11 chi=50,70,90,100 (C4 symmetric) | ~134 GB |
| E | 12×12 chi=50,70,100 (new frontier) | ~160 GB |
| F | 9×9 chi=100 (C4 control) | 85 GB |
| G | 10×10 chi=100,110 damped (if traps in B) | ~159 GB |
| H | 11×11, 12×12 --measure=all (figures) | 10 GB |

Usage: `nohup ./run_mi300x.sh > run_mi300x.out 2>&1 &`
- Idempotent (skips if log exists), VRAM-checked before large runs, summary table at end

---

## Paper state — CONTENT COMPLETE

### All open items resolved
| Item | Status |
|------|--------|
| tab:findings in conclusion | ✓ done |
| 5 lightcone/diff figures | ✓ inserted into §6.3 |
| Float placement (Table 11) | ✓ \FloatBarrier |
| n* inconsistency | ✓ fixed (v_LR=2|J|) |
| Repository URL TODO | ✓ "contact author" |

### One remaining item
- **MI300X experimental results**: when run_mi300x.sh completes, results go into a new §6.x or extend §6.2/§6.3 with high-chi 10×10 data. The script is ready; just needs the machine.

---

## Previous checkpoints
- CHECKPOINT_13.md: figures session, Δn analysis, fig_diff built
- CHECKPOINT_12.md: tab:findings written, §6.1–§6.3 complete
- CHECKPOINT_11.md: chi=20 analysis, 5×5 initial findings
