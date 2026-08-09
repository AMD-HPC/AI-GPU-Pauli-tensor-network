# CHECKPOINT 12 — 2026-05-16

## Session summary

Continued from CHECKPOINT_11. All six 5×5 chi logs analyzed, full paper draft reached.

---

## 5×5 symmetry experiment — complete results

| chi | n_steps | C(t) final | norm max | Type 1 traps | Notes |
|-----|---------|-----------|----------|--------------|-------|
| 20  | 40      | 0.000716  | 1.000465 | 0            | Type 2 at L34-36 only |
| 40  | 40      | 0.000907  | 1.001393 | 0            | Type 2 at L21,23,24,34,36,39 |
| 60  | 40      | 0.001073  | 1.004159 | 0            | Clean |
| 80  | 30      | 0.028089  | 1.006484 | 0            | 4×4 chi=80 trapped at L17 |
| 100 | 25      | 0.042065  | —        | 0            | 4×4 chi=100 trapped at L20 |
| 110 | —       | OOM       | —        | —            | 9424.9 MB workspace alone |

**Key finding**: 5×5 has MORE loops (40 bonds vs 24 for 4×4) yet ZERO Type 1 traps.
C4 symmetry of initial operator placement is the dominant factor in BP fixed-point multiplicity.

---

## Paper sections completed this session

### §6.1 — Experimental setup and validation strategy
- Two-tier validation: (1) implementation correctness via Julia reference; (2) physical fidelity via physics predictions
- Four physics predictions listed: operator spreading, early-time universality, C4 symmetry of n(r,t), scrambling to zero
- Closing statement: "the implementation reported here codes the physics directly"
- Rhetorical questions removed; "independent contribution" replaced

### §6.2 — C(t) chi convergence
- `fig_ct.png` inserted with caption
- Log y-scale, isolated × markers for trapped layers (not V-shapes)
- Annotations: chi*, portable operating points, universal agreement region

### §6.3 — BP fixed-point multiplicity (full rewrite)
- **Structure**: failure modes defined FIRST, then root cause, then evidence
  1. Opening: four isolated × markers are trapped layers
  2. Type 1 (metastable trap) and Type 2 (non-convergence) defined with diagnostics
  3. "With this vocabulary in place, we now identify the root cause"
  4. C4 symmetry as root cause (5×5 controlled experiment)
  5. `tab:symmetry_traps`: 4×4 vs 5×5 comparison table
  6. Mechanism: symmetric fixed point unique in symmetric subspace
  7. Further evidence: cross-architecture + alpha experiments
  8. Trap remediation: 4-step procedure
  9. Capacity boundary: Se/logX ≥ 0.65
  10. Portable operating points: chi=40,60 (both architectures); chi=80,100 (symmetric grid)

### Conclusion — `tab:findings` added
Written into `lightconesimulation.tex` immediately after `tab:comparison`.
8-row summary table with Finding / Evidence / Implication columns:

| Row | Finding |
|-----|---------|
| 1 | C4 symmetry eliminates Type 1 traps |
| 2 | Two distinct BP failure modes |
| 3 | Metastable traps are reversible |
| 4 | Portable operating points (chi=40,60) |
| 5 | Diminishing returns above chi*≈60 |
| 6 | Capacity boundary Se/logX≥0.65 |
| 7 | Validated memory model (<2% error) |
| 8 | Lightcone saturation governs run length |

Conclusion prose updated to reference both `tab:comparison` and `tab:findings`.

---

## File inventory

| File | Status |
|------|--------|
| `Paper/experiments_section.tex` | §6.1–§6.3 complete; §6.4–§6.8 unchanged |
| `Paper/lightconesimulation.tex` | Conclusion: tab:comparison + tab:findings added |
| `Paper/fig_ct.png` | Final; isolated × markers, log scale |
| `Paper/make_ct_figure.py` | Clean; SyntaxWarning fixed |
| `5x5_symmetry_20260515/5x5_chi020_n40.log` | 40 layers, analyzed |
| `5x5_symmetry_20260515/5x5_chi040_n40.log` | 40 layers, analyzed |
| `5x5_symmetry_20260515/5x5_chi060_n40.log` | 40 layers, analyzed |
| `5x5_symmetry_20260515/5x5_chi080_n30.log` | 30 layers, analyzed |
| `5x5_symmetry_20260515/5x5_chi100_n25.log` | 25 layers, analyzed |
| `5x5_symmetry_20260515/5x5_chi110_n20.log` | OOM immediately |

---

## Open items

1. **n* table inconsistency**: prose says n*=10·d_max (v_LR=|J|), lightcone table implies n*=5·d_max (v_LR=2|J|). Need to pick one and reconcile.
2. **\TODO{repository URL}**: still unfilled in lightconesimulation.tex §Availability
3. **fig:lightcone**: static strip figure from make_gif.py frames — still placeholder in paper
4. **chi=110 5×5**: OOM on MI100. Would need MI250X (128 GB) or MI350.
5. **Streams and mixed precision**: discussed, not yet implemented (future work).

---

## Key terminology established

- **Trapped layer**: a layer at which BP converged (delta < tol) to a wrong metastable fixed point. C(t) collapses to ~0.007–0.009. Recovers at next layer. Previously called "cliff" — term retired.
- **Type 1 failure**: trapped layer (silent, requires cross-chi comparison to detect)
- **Type 2 failure**: BP non-convergence (sweeps=100, delta>>tol, self-announcing, C(t) may still be valid)
- **chi*≈60**: diminishing-returns threshold on 4×4; gain <0.2% per step above this
- **Capacity boundary**: Se/logX≥0.65 → increase chi, not alpha
- **Portable operating points**: chi=40,60 trap-free on MI100 and MI60
- **"Codes the physics directly"**: philosophy of building from first principles vs porting library code

---

## Previous checkpoints
- CHECKPOINT_11.md: chi=20 analysis, 5×5 initial findings, paper state before symmetry rewrite
- CHECKPOINT_10.md: C(t) figure, §6.2 insertion
- Earlier checkpoints in Paper/
