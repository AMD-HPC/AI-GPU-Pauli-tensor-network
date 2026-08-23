# Frontier MPF Theory — Session Notes 2026-08-22

## Notation

- n(v,t): Pauli weight for noiseless state at site v, time t
- n_γi(v,t): Pauli weight for noisy state γ=i (i=1,2,3)
- M_p(v,t) = Σi αi n_γi(v,t): MPF reconstruction minimizing MSE
- M_e(v,t) = Σi αi n_γi(v,t): MPF reconstruction minimizing cross-entropy

---

## Frontier Definition

**Spatial frontier at time t:**
F_γi(t) = {v : n_γi(v,t-1) = 0 AND n_γi(v,t) > 0}  (first arrival)

**Ordering (causal):**
F_nl(t) ≥ F_γ1(t) ≥ F_γ2(t) ≥ F_γ3(t)  for all t
(noiseless fastest, heavier noise slower)

**Union frontier:**
F(t) = F_nl(t) ∪ F_γ1(t) ∪ F_γ2(t) ∪ F_γ3(t)

**BFS shell:**
V_d = {v : d(v, center) = d}  (precise spatial partition)

---

## Delay and Cross-Entropy Loss

**Arrival time:** t_γi(v) = min{t : n_γi(v,t) > 0}

**Site delay:** D(v, t_γi) = |t_nl(v) - t_γi(v)|
- D is a DESCRIPTOR/OBSERVABLE, NOT an optimization variable
- Defined in the surrounding of t_γi(v), not t_nl(v)
- Works for both orderings (noisy faster or slower)
- Handles rotation noise cases where noisy wave may precede noiseless locally

**Per-site cross-entropy loss:**
L_γi(v) = average over window around t_γi(v) of -ln(n_γi(v,t) / n_nl(v,t))

**Window choice:** average over fixed window Δ around t_γi(v):
L_γi(v) ≈ (1/Δ) Σ_{|t - t_γi(v)| ≤ Δ} -ln(n_γi(v,t) / n_nl(v,t))

**Heisenberg-like constraint:** Space-time duality
- α(t): t sharp, space fuzzy (F(t) shell width) → ΔT→0, ΔX ~ shell
- α(v): space sharp (BFS shell d exact), time fuzzy → ΔX→0, ΔT ~ window Δ
- ΔX · ΔT ~ constant

**Natural regularization:** Waves that don't reach shell d within window Δ
contribute nothing → their αi → 0 naturally. No artificial regularization needed.

---

## Current MPF Objectives

**M_p (frontier position):**
α = argmin Σ_t (Σi αi F_γi(t) - F_nl(t))²
- Fits once, globally over all t
- Uses frontier POSITION as scalar observable

**M_e (entropy):**
α = argmin Σ_{v,t: n_nl>ε} -ln(Σi αi n_γi(v,t) / n_nl(v,t))
- Fits once, globally over ALL v and ALL t inside noiseless lightcone
- Problem: interior sites (chi large, approximate) DOMINATE the sum
  (interior ~ t³ sites, frontier surface ~ t²)

---

## Time-Dependent MPF α(t)

Fix t, minimize over v ∈ F(t):
α(t) = argmin Σ_{v ∈ F(t)} -ln(Σi αi(t) n_γi(v,t) / n_nl(v,t))

**Meaning:** "How does noise distort the spatial field at this moment?"
- Captures instantaneous noise structure
- Changes over time as wave penetrates deeper
- Natural for asymmetric grids/noise (heavy-hex hardware noise)

---

## Space-Dependent MPF α(d) per BFS shell

Fix shell d, minimize over window around t_γi(d):
α(d) = argmin Σ_{v ∈ V_d} Σ_{|t-t_γi(d)|≤Δ} -ln(Σi αi(d) n_γi(v,t) / n_nl(v,t))

**Meaning:** "How does noise accumulate along the path to reach shell d?"
- Captures path-length dependent damping
- Sites at same shell share same path-length history
- α(d) is SPARSE in γ: at far shells only fast noisy waves contribute

**Frontier resilience:** If chi is large enough to preserve dominant eigenvalues
without complete energy loss, the frontier is resilient — damped but finite.
The wave arrives with reduced amplitude but t_γi(v) is still finite.

---

## The Computational Lightcone Bound

**RULE:** v_hop must cover the SLOWEST noisy frontier:
v_hop_correct ≥ v_γ3 × n_steps × safety_factor

**NOT** v_nl (noiseless velocity) — γ=3 sites extend BEYOND the noiseless frontier.

**Why it matters for the fit:**
Within the correctly-bounded lightcone → all frontier sites allocated (chi_alloc=chi_max)
→ at first arrival chi≈1 → EXACT measurement
→ MPF fit operates entirely on exact data

**The failure mode (--v_hop=0.500 on rect_7×7):**
Two errors compounding:
1. WRONG SPATIAL DOMAIN: sites where n_γi should be >0 have chi_alloc=1
   → n_γi=0 by CONSTRUCTION (not by physics)
2. WRONG TEMPORAL AVERAGE: average over all t includes these artificial zeros
   → indistinguishable from "wave never arrived" vs "computation prevented arrival"

Result: γ=1, γ=2, γ=3 all have n_γi=0 at cut sites → all look identical
→ degenerate α = [0.44, 1.67, -1.11] → MPF entropy = MPF frontier

**The fix:** measure v_γ3 from lightcone sweep, set v_hop ≥ v_γ3 × n_steps × 1.1
→ all zeros in sum are PHYSICAL (too much noise) not COMPUTATIONAL (we cut them)
→ measure is self-consistent with the paper's claim that frontier is exact

---

## Pipeline Consistency Rule

1. Run lightcone sweep → measure v_hop for NL AND all noisy states (γ=1,2,3)
2. Set v_hop = max(v_γi) × n_steps × 1.1  (slowest noisy frontier)
3. Production run with this v_hop → all frontier sites guaranteed allocated
4. MPF fit operates on exact data at every frontier site

Within the correctly-bounded lightcone: NO computational loss corrupts the measurement.
The frontier analysis claim ("exact at chi≈1") is self-consistent.

---

## Observed: 20-cycle rect_7x7 frontier lifecycle (PAPER_r7_50_20c)

- Cycles 1-9: frontier MPF perfect — all waves separated, clean signal
- Cycle 10: γ=1 catches F_nl=6 (hits lattice boundary) → δ1=0 → MPF anchor lost → coefficients blow up [+10.67, -22.38, +12.71]
- Cycles 11-20: boundary saturation — F_nl=F_γ1=6, F_γ2=4, F_γ3=3 frozen
- LR-constrained MPF stays stable throughout (causal constraint prevents blow-up)
- Entropy MPF also stays good — uses full (v,t) not just frontier position

Boundary is not an enemy — it's a stabilizing constraint.
Full lifecycle: growth → saturation → reflection → only entropy/LR MPF reliable.

## Handwritten Notes 2026-08-22 — Complete Theory (3 pages)

### Page 1: TIME-first formulation

Fix time t. n(v,t) is Pauli's weights.

F_n(t) = {v : n(v,t) ≠ 0 AND n(v,t-1) = 0}  (first arrival set)
F(t) = F_n(t) ∪ ∪_i F_γi(t)  (union frontier)

**α(t) = [α₁(t), α₂(t), α₃(t)]** — fit at fixed t over v ∈ F(t):

MSE:          min  1/|F(t)| Σ_{v∈F(t)} (Σi αi n_γi(v,t) - n(v,t))²
Cross-entropy: min  1/|F(t)| Σ_{v∈F(t)} -ln(Σi αi n_γi(v,t) / n(v,t))

"This is an average space frontier that will be spatially separated and damped."
t ≠ fixed → space is fuzzy boundary (the Heisenberg duality).

---

### Page 2: SPACE-first formulation

F(d) = {v : d(center, v) = d}  (BFS shell at depth d)

Arrival times: t_n(v) for noiseless, t_γi(v) ≤ t_n(v) for noisy (in normal ordering).

**α(d) = [α₁(d), α₂(d), α₃(d)]** — fit at fixed shell d:

min_α  Σ_{v∈F(d)} Σ_{t∈T} (Σi αi n_γi(v,t) - n(v,t))²

**Key question: WHICH time window T?**

[1] Evaluate at t_γi (noisy arrival, single snapshot):
    1/|F(d)| Σ_{v∈F(d)} -ln(Σi αi n_γi(v, t_γi) / n(v, t_n))

[2] Sum over t ≤ t_n (all pre-noiseless-arrival):
    1/|F(d)| Σ_{v∈F(d)} Σ_{t≤t_n} -ln(Σi αi n_γi(v,t) / n(v,t))
    ⚠ Problem: t ≤ t_γi → n_γi(v,t) = 0 (wave hasn't arrived yet)
    → artificial zeros contaminate the average

[3] Restrict to active window t_γi ≤ t ≤ t_γ3:
    Σ_{v∈F(d)} Σ_{t_γi ≤ t ≤ t_γ3} -ln(Σi αi n_γi(v,t) / n(v,t))
    Only sum when noisy wave is actually active → no artificial zeros ✓
    If n_γi always active with n(v,t)

**Option [3] is the cleanest:** avoids pre-arrival zeros, natural window.

---

### Page 3: GENERALIZATION — Space-Time bias correction

Label: "Generalization / Space Timeline"

"There is a delay in time because t_n < t_g [t_γi]."

Extended model with bias b(d) per BFS shell:

    Σi αi n_γi(v,t) + b(d(v)) ≈ n(v,t)
    ↳ d-bias to compensate the arrival delay

Valid when: ∀v, n_γi(v,t) ≠ 0 when t ≤ t_γi and t_γi ≤ t_n

**Physical meaning of b(d):**
- α_i: which noisy state to combine (amplitude scaling)
- b(d): how much the delay shifts the amplitude at shell d
- Together: decouple spatial (which wave?) from temporal (how late?) correction

**Connection to window option [3]:**
At t = t_γi: b(d) = n(v,t_n) - Σi αi n_γi(v,t_γi)
             = the amplitude gap between noiseless at ITS arrival and noisy at ITS arrival
This is the "d-bias" — the amplitude deficit caused by the delay D(v) = t_n - t_γi.

**Critical: addressing -ln → -∞ when t_n > t_γi**

The cross-entropy -ln(Σi αi n_γi(v,t) / n(v,t)) diverges when:
- t < t_γi: n_γi(v,t) = 0 → numerator = 0 → -ln(0) = **+∞**
- t_γi ≤ t but amplitude tiny: n_γi ≈ 0 → -ln(~0) = **large positive**
- t > t_n and wave grows: ratio unstable in both directions

**Window [3] prevents the +∞:** by restricting to t_γi ≤ t ≤ t_γ3, n_γi(v,t) > 0
for ALL i by construction — all waves have arrived, no zero denominators.

**Bias b(d) provides additional protection:** when t_n > t_γi (noiseless arrives AFTER
noisy — e.g. rotation noise), the reconstruction Σi αi n_γi + b(d) is shifted upward
by b(d) > 0, bounding the argument of -ln away from zero even at early t.

Together: window [3] + bias b(d) make the cross-entropy numerically well-defined
and physically meaningful throughout the active frontier window.

## NEXT SESSION: fitting by frontiers

Explore α(t) and α(d) — time-dependent and shell-dependent MPF:
- α(t): fit at each cycle t over F(t) = frontier union at time t
- α(d): fit per BFS shell d, window around t_γi(d)
- Both give spatially/temporally resolved coefficients vs single global α
- Key: sparse in γ naturally — far shells only fast waves contribute
- Implementation: modify plot_mpf_entropy.py or new script

## Open Questions

1. Implement α(t) and α(d) — time-dependent and shell-dependent MPF
2. Compare α(t) to global α: does it improve on asymmetric grids (heavy-hex hw noise)?
3. Correct v_hop measurement: run lightcone sweep with all 4 states, use v_γ3
4. Window size Δ for α(d): how sensitive are results to Δ?

## Figure restructure 2026-08-23

New 3-row layout making space-time duality explicit:

```
Row 0: Spatial maps L(v) — one per method, μ annotated with ↓/↑
Row 1: L(t) [left] | L(v) per BFS shell [right]   — loss in TIME and SPACE
Row 2: α(t) [left] | α(d) opt1+opt2 [right]       — coefficients in TIME and SPACE
```

Left column = TIME view. Right column = SPACE view.
Row 1 = LOSS (how bad is the noise?). Row 2 = COEFFICIENTS (how to correct?).

Styling: γ=1,2,3 dashed (inputs), MPF variants solid (outputs).

Results on PAPER_hh_dep_200_10c.log:
- MPF α(t) time-proj: μ=0.481 — BEST (55% vs best noisy 1.058)
- MPF entropy (global): μ=0.596 — 44% improvement
- MPF frontier (global): μ=1.058 — tied with γ=1
- MPF α(d) opt2: μ=1.050 — slightly worse (only d=1,2,3 have valid fits)
- MPF α(d) opt1: μ=1.218 — worse

α(t) insight: α_N1 increases over time (γ=1 more important at later cycles),
α_N3 becomes more negative (larger subtraction needed), α_N2 ≈ 0 throughout.
The time-resolved fit adapts to changing noise structure — global α averages it away.

α(d) insight: only d=1,2,3 valid (γ=3 velocity ≈ 0.152 hops/cycle × 10 = 1.5 hops
max, so γ=3 never reaches d≥4). Need 20-cycle data or rect_7x7 for more shells.
