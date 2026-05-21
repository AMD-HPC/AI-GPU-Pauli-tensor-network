# CHECKPOINT 13 — 2026-05-16 (figures session)

## What was built this session

### Figure scripts (all in Paper/)

| Script | Purpose |
|--------|---------|
| `make_lightcone_figure.py` | 1–3 row n(r,t) strip + C(t) panel. Args: `--layers --trap-layer --tstar --row-labels` |
| `make_diff_figure.py` | 3-row chi=80 / chi=40 / Δn comparison + C(t). Args: `--layers --trap-layer` |
| `make_gif.py` | Animated GIF, single or multi-grid side-by-side |

### Figures produced

| File | Description | Status |
|------|-------------|--------|
| `fig_lightcone_trap.png` | 4×4 chi=80, 7 snapshots, "trapped"/"recovered" banners, × on C(t) | done |
| `fig_lightcone_clean.png` | 4×4 chi=40, same time points, monotone decay | done |
| `fig_lightcone_9x9.png` | 9×9 chi=20 correct center (4,4), ring structure, C(t)→0 | done |
| `fig_diff.png` | 3-row spatial diff | done, see Δn analysis below |
| `fig_ct.png` | C(t) chi convergence all chi | done (prior session) |
| GIFs | spreading_9x9.gif, spreading_4x4_chi80.gif, spreading_comparison.gif | done |

### Key design decisions

- **Three separate figures** (not combined): each grid gets full width — user's insight that 9×9 looks small when squeezed into 1/3 of a row
- **Correct vs wrong center**: `extended_20260514/9x9` has wrong center (3,3); `odd_fix_20260515/9x9` has correct center (4,4). Wrong-center run is an illustrative example of asymmetry.
- **trap-layer=17**: "trapped"/"recovered" banners on heatmap + red × on C(t) curve
- **Diverging colorscale (RdBu_r)**: Δn row uses TwoSlopeNorm centered at zero

---

## Δn analysis — spatial trap signature

Run output from make_diff_figure.py:
```
Δn range across all layers: [-0.00120, 0.00270]
L 1  max|Δn|=0.00000
L10  max|Δn|=0.00000
L15  max|Δn|=0.00020
L17  max|Δn|=0.00040  ← TRAP
L18  max|Δn|=0.00070
L20  max|Δn|=0.00270
```

**Interpretation**: The Δn grows monotonically with time — this is NOT primarily a trap signature, it is the chi=80 vs chi=40 physical difference accumulating. By L20 the two runs have diverged by 0.0027 simply because higher chi gives a richer environment. The trap at L17 is NOT clearly isolated as a spike — it is one point on a monotone growth curve.

**Revised physics understanding**: n(r,t) is insensitive to the BP trap because:
1. It averages over all 16 sites × all Pauli components → error diluted
2. The chi=80 vs chi=40 physical difference (richer environment) grows with time and dominates over the BP trap effect

**The trap IS visible in C(t) because** C(t) selects the Z_center component specifically — a single matrix element of the BP environment is amplified, not averaged.

**Paper implication**: The Δn figure may not make the trap visible as a spike. It shows chi-convergence of the spatial weights instead. This is still a valid result (spatial weights converge in chi faster than C(t)), but the original intent (isolate trap distortion spatially) requires a different approach — e.g. comparing chi=80 trapped vs chi=80 damped (alpha=0.5) at the same chi.

---

## Open items

1. **fig_diff.png**: view and decide if monotone Δn growth is publishable as chi-convergence of n(r,t), or redesign with same-chi comparison (chi=80 trapped vs chi=80 with damped BP / alpha retry)
2. **Paper insertion**: none of the three lightcone figures inserted into tex yet — waiting for user approval of layouts
3. **n* table inconsistency**: v_LR=|J| vs v_LR=2|J| — text vs table disagree
4. **\TODO{repository URL}**: lightconesimulation.tex
5. **Matched step counts**: user will re-run with same n_steps for all grids
6. **9×9 C(t) panel**: starts at ~0.1 not 1.0 — t=0 point missing from plot (minor bug in ct_series parsing)

---

## Log files used for figures

| Log | Center | Trap | Notes |
|-----|--------|------|-------|
| `extended_20260514/mi100/4x4_chi080.log` | (1,1) off-center | L17 | Main trap case |
| `extended_20260514/mi100/4x4_chi040.log` | (1,1) off-center | none | Clean baseline |
| `odd_fix_20260515/mi100/9x9_chi020_n50.log` | (4,4) correct | none | C4 symmetric showpiece |
| `extended_20260514/mi100/9x9_chi020_n50.log` | (3,3) WRONG | none | Asymmetry illustration |

## Previous checkpoints
- CHECKPOINT_12.md: tab:findings written into conclusion, §6.1–§6.3 complete
- CHECKPOINT_11.md: chi=20 analysis, 5×5 initial findings
