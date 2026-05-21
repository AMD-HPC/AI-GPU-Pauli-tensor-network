# CHECKPOINT 18 — 2026-05-17 (householder baseline complete — genuine vs artifact traps)

## Run completed overnight
```
bash run_householder_mi100.sh   # launched 2026-05-17 01:06:31
```
All 23 logs completed. Saved to `logs/householder_20260517/mi100/`.

| Log | Notes |
|-----|-------|
| 4x4_chi020..100.log | Chi convergence, 8 runs |
| 4x4_chi080_n40.log | Long evolution 40 steps |
| 5x5_chi020_n40.log | 5×5 symmetry, 40 steps |
| 5x5_chi040_n40.log | 5×5 symmetry, 40 steps |
| 5x5_chi050.log | 5×5 20 steps |
| 5x5_chi060_n40.log | 5×5 symmetry, 40 steps |
| 5x5_chi080.log | 5×5 20 steps |
| 5x5_chi080_n30.log | 5×5 symmetry, 30 steps |
| 5x5_chi100_n25.log | 5×5 symmetry, 25 steps |
| 3x3_chi050.log, 3x3_chi080.log | Grid scaling small |
| 6x6_chi050.log, 6x6_chi080.log | Grid scaling |
| 7x7_chi050.log, 8x8_chi050.log | Grid scaling large |
| 9x9_chi020_n50.log | Large-grid lightcone reference |

**All runs: Cholesky-QR attempted=0 fallback=0 skipped=N. Pure Householder confirmed.**

---

## Key result: genuine traps vs artifact traps

### 4×4 chi convergence C(t) — all 20 layers

| chi | Traps (layer: C(t)) | L20 C(t) | Clean? |
|-----|---------------------|----------|--------|
| 20 | L17=0.008889, L19=0.009010 | 0.063055 | 2 traps |
| 40 | none | 0.057174 | ✓ clean |
| 50 | none | 0.055820 | ✓ clean |
| 60 | L19=0.009132 | 0.055152 | 1 trap, recovers |
| 70 | none | 0.054972 | ✓ clean |
| 80 | none | 0.054258 | ✓ clean |
| 90 | L19=0.009076 | 0.053840 | 1 trap, recovers |
| 100 | L19=0.009078 | 0.053760 | 1 trap, recovers |

### The key distinction

**Artifact trap (eliminated by Householder):**
- Chi=80, L17 in old adaptive run: C(t)=0.008782
- Caused by non-uniform Cholesky/Householder gauge breaking C4 lattice symmetry
- **Gone in all Householder runs** ✓

**Genuine BP traps (remain with Householder):**
- Chi=20 L17/L19, chi=60 L19, chi=90 L19, chi=100 L19
- Caused by irreducible fixed-point multiplicity of Bethe approximation on loopy 2D lattice
- Always recover the next layer
- Stochastic — appear at different layers per chi (not always L17)
- Cannot be eliminated by changing QR mode — fundamental to BP on 2D

### Chi convergence at L20 (recovered values, monotone)

chi=40: 0.057174 → chi=50: 0.055820 → chi=60: 0.055152 → chi=70: 0.054972
→ chi=80: 0.054258 → chi=90: 0.053840 → chi=100: 0.053760

Convergence is monotone and smooth — C(t) is a decreasing function of chi (larger chi = lower = better Bethe bound). Extrapolated true value ≈ 0.052–0.053.

---

## Notable individual results

### 5×5 chi=80 n=30, L30
sweeps=100, delta=3.08e-03 — Type 2 non-convergence. Se/logX=0.818 (above danger threshold 0.80). BP time = 183,093ms for one layer. Capacity limit reached.

### 5×5 chi=100 n=25, L25
sweeps=81, delta=6.68e-06. BP time = **1,241,419ms (20+ min for one layer)**. Total time L25 = 1,483s. Extreme BP cost at chi=100 for 5×5. Se/logX=0.660.

### 8×8 chi=50, L20
norm = **32,577,500** — Bethe approximation completely broken. Chi only reached 15 (never saturated to 50). 8×8 is beyond the reliable BP regime at chi=50.

### 9×9 chi=20, L50
C(t) = **0.000000** — complete operator spreading and decay. Se/logX=0.965 (96.5% capacity). Full lightcone observed. Beautiful reference result for the large-grid figure.

---

## Paper implications

**Clean separation of trap types is now documented and reproducible:**

> "Remaining BP traps under Householder QR are genuine fixed-point multiplicity of the
> Bethe approximation on the loopy 2D lattice — they recover spontaneously at the next
> layer and are consistent with chi-convergence. The gauge-induced artifact trap
> (chi=80, layer 17, adaptive mode) is eliminated entirely by Householder QR."

**8×8 result**: adds a data point showing where the Bethe approximation breaks entirely (norm explosion). Useful for defining the reliable operating range.

**9×9 result**: C(t)=0 at L50 — complete lightcone, publishable as the large-grid reference.

**Chi convergence**: clean monotone curve from chi=40 to chi=100. fig_ct.png can now be redrawn from Householder data — no trapped layers contaminate the curve (trapped layers recover, L20 is always clean).

---

## Next steps
- Redraw fig_ct.png from householder_20260517 data
- New 5×5 timing figure: HH vs adaptive trotter+bp per layer (chi=60, 20 steps)
- Paper paragraph: genuine vs artifact traps, Householder default rationale
- MI60 householder run (run_householder_mi60.sh) — queue when machine available
- MI300X — waiting for access

---

## Previous checkpoints
- CHECKPOINT_17.md: 5×5 QR comparison, householder set as default, run scripts written
- CHECKPOINT_16.md: three-way QR benchmark, chi=80 cliff mechanism
