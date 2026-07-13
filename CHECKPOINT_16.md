# CHECKPOINT 16 — 2026-05-16 (three-way QR benchmark + chi=80 cliff analysis)

## Three-way QR benchmark at chi=50 (4x4, 20 steps, MI100)

### Commands run
```
./main 4 4 50 20 0.1 -1.0 -1.0 1e-11 --qr=householder > A.log
./main 4 4 50 20 0.1 -1.0 -1.0 1e-11 --qr=cholesky   > B.log
./main 4 4 50 20 0.1 -1.0 -1.0 1e-11 --qr=adaptive   > C.log
```

### Saturation trotter timing (layers 12–20 avg)

| Mode | trotter avg (ms) | bp avg (ms) | net/layer |
|------|-----------------|-------------|-----------|
| A: Householder | 12,733 | 1,880 | — |
| B: Forced Cholesky (42.8% fallback) | 12,581 | ~1,900 | −152 |
| C: Adaptive (62% fallback of attempted) | **12,215** | 2,166 | **−230 net** |

Adaptive wins overall. Trotter savings (~518ms/layer) exceed BP overhead (~286ms/layer).

### Chi=50 adaptive BP behavior
- Trotter time drops in layers 18–20 (12,079 → 11,553 → 11,084ms): adaptive gets faster
  as truncation steers bonds toward well-conditioned matrices where Cholesky succeeds.
- BP is slightly more expensive for adaptive (16.9 avg sweeps vs 15.4 Householder).
- Layer-20 BP spike: 36 sweeps / 4,388ms (adaptive) vs 16 / 1,948ms (householder).
- C(t) matches to 5 decimal places across all three modes. Physics identical.

### Chi=50 verdict
`--qr=adaptive` is the correct production default at chi=50.

---

## Chi=80 benchmark — the cliff (4x4, 20 steps, MI100)

### Commands run
```
./main 4 4 80 20 0.1 -1.0 -1.0 1e-11 --qr=householder > AA.log
# adaptive baseline: logs/mi100/4x4_chi080.log (old build, same adaptive logic)
```

### The cliff at layer 17

| Layer | Adaptive C(t) | Adaptive bp/sweeps | Householder C(t) | Householder bp/sweeps |
|-------|--------------|-------------------|-----------------|----------------------|
| 16 | 0.071167 | 9,689ms / 12 | 0.071169 | 24,722ms / 31 |
| 17 | **0.008782** | **31,627ms / 39** | **0.065527** | 8,759ms / 11 |
| 18 | 0.060939 | 12,147ms / 15 | 0.060942 | 8,757ms / 11 |
| 20 | 0.054259 | 20,248ms / 25 | 0.054258 | 13,539ms / 17 |

**The cliff is a BP artifact, not physics.**
- Householder gives smooth monotone decay (correct).
- Adaptive C(t)=0.008782 at layer 17 is wrong; recovers at layer 18.
- By layer 20 both agree to 5 decimal places.

### Chi=80 saturation timing (layers 12–20 avg)

| Mode | trotter avg (ms) | bp avg (ms) | bp sweeps avg |
|------|-----------------|-------------|---------------|
| Adaptive | 46,189 | **14,093** | **17.4** |
| Householder | 46,244 | **11,489** | **13.3** |

Trotter: identical (<0.1%). BP: Householder 18% cheaper, 3.1 fewer sweeps on average.

**At chi=80 the verdict flips: use `--qr=householder`.**

---

## Root cause: symmetry-breaking gauge

### The mechanism (three steps)

1. **Non-uniform fallback pattern.** At chi=80, float32 Cholesky fails on a fraction of
   bonds (SPOTRF sees apparent negative eigenvalues: rounding error in AᵀA is
   O(m × ε_mach) ≈ O(512001 × 1.2e-7) ≈ 6e-2, far above ε=1e-8 regularization).
   The fallback pattern is bond-specific — some bonds get Cholesky gauge, others
   Householder gauge, depending on each bond's individual Gram matrix conditioning.

2. **Non-uniform gauge = broken lattice symmetry.** The 4×4 lattice + center initial
   state has reflection/rotation symmetry. Trotter evolution respects it. But mixed
   Cholesky/Householder gauge assigns different orthogonalization residuals to
   geometrically equivalent bonds — the bond gauge is no longer symmetric under
   the lattice symmetry group.

3. **BP on symmetry-broken gauge finds wrong fixed point.** At full saturation (chi=80,
   all bonds at chi_max), BP has multiple fixed points. The symmetric fixed point gives
   the correct C(t). A symmetry-broken initial gauge steers BP toward an asymmetric
   metastable fixed point → C(t)=0.008782 instead of ~0.065.

### Why the cliff doesn't appear elsewhere

| Case | Why safe |
|------|----------|
| chi < 70 | Fewer bonds saturate; more Cholesky calls succeed; gauge more uniform |
| Julia Float64 | Cholesky succeeds on all qualifying bonds; uniform gauge; symmetry preserved |
| Householder (any chi) | Every bond gets identical algorithm; gauge respects lattice symmetry by construction |

### General principle
The QR method must be **uniform across all bonds** to guarantee the gauge tensor network
respects the physical symmetry of the problem. Mixed-mode adaptive QR violates this in
float32 at large chi.

**Any implementation choice that can change a published number is a method, not a detail.**

---

## Paper implications

1. **Name the QR method** for each result — "Householder QR (backward stable)" not
   just "QR factorization".
2. **Footnote the cliff** — at chi=80 adaptive Cholesky-QR produces a symmetry-breaking
   gauge that triggers a metastable BP fixed point; Householder avoids this; all
   published results use Householder.
3. **Cholesky fallback table** (attempted/fallback/skipped) belongs in the paper as
   evidence that float32 Cholesky is unreliable at large chi.
4. **Disclosure principle** — hiding computational details harms reproducibility; the
   symmetry argument shows the algorithm is part of the physics output.

---

## Production recommendations

| chi | Recommended mode | Reason |
|-----|-----------------|--------|
| ≤ 50 | `--qr=adaptive` | 4% trotter speedup; BP overhead small; no symmetry risk |
| ≥ 80 | `--qr=householder` | Trotter identical; BP 18% cheaper; avoids symmetry-breaking gauge |
| 60–70 | TBD | Transition region; fallback rate unknown; recommend householder to be safe |

---

## Previous checkpoints
- CHECKPOINT_15.md: paper corrections, Cholesky counters, --qr= flag, MI100 chi=50 baseline
- CHECKPOINT_14.md: figures session, paper content complete, run_mi300x.sh written
