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

## Open Questions

1. Implement α(t) and α(d) — time-dependent and shell-dependent MPF
2. Compare α(t) to global α: does it improve on asymmetric grids (heavy-hex hw noise)?
3. Correct v_hop measurement: run lightcone sweep with all 4 states, use v_γ3
4. Window size Δ for α(d): how sensitive are results to Δ?
