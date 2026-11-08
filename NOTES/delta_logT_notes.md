# Delta Log-Transfer Vector Field — Notes (2026-11-07)

## The observable

For each directed bond (v→u) and each cycle t, define the log bond transfer:

    ln T_{v→u}(t) = ln( n(u,t+1) / n(v,t) )

This is the log-probability of weight moving from site v at time t
to neighbor u at time t+1 — the log bond transfer coefficient from §3
of the tomography paper, now in its natural log form.

The delta log-transfer:

    δ ln T_{v→u}(t) = ln T_{v→u}^{base}(t) - ln T_{v→u}^{pert}(t)
                    = ln( T_{base} / T_{pert} )

This is a **log-likelihood ratio** (log Bayes factor) for the hypothesis
"this bond is broken" at each directed bond transition. Positive = baseline
transfers more than perturbed = broken bond on this path.

**Connection to existing observables**:
- L(v,t) = -log(n_pert/n_base) is the **spatial** log-likelihood ratio at sites
- δ ln T is the **spatio-temporal** log-likelihood ratio at bonds
- Both measure the same underlying perturbation but from different perspectives
- δ ln T has direction (v→u); L(v,t) is a scalar at each site

**Gating**: only compute where n_base(v,t) > ε_th and n_pert(v,t) > ε_th.
This filters the lightcone boundary naturally — same criterion as detection.

---

## Physical interpretation — the broken bond as a diffracting obstacle

The first visualization of δ ln T as a vector field (cycle t=1→2, 2→3, 3→4)
revealed that a broken bond does not simply damp the wave — it **diffracts** it:

**t=1→2 (first wavefront crossing)**:
- RED 11→12 large: baseline transfers more through the broken bond
- BLUE spread into three directions from site 12: shadow site "searches" for
  weight from alternative neighbors (13, 5, 19). Broken bond forces redistribution
  of flow topology.

**t=2→3 (wall face saturated)**:
- BLUE > RED: weight has accumulated at site 11 (wall face). n_pert(11) > n_base(11)
  → perturbed run now has MORE weight to transfer → blue dominates.
  The perturbation reverses — wall face becomes a secondary source.

**t=3→4 (diffraction pattern emerges)**:
- Blue diagonal 11→12→19→20→27: shadow paths — perturbed wave depleted
- Red diagonal 24→25→31→32→39: rerouted paths — baseline wavefront undisturbed,
  perturbed has been bent around the obstacle

The two diagonals are **diffraction fringes** — they point directly at the broken
bond location and orientation. The pattern is centered on bond 9 (11↔12) and
the 45° structure reflects the bond orientation on the rectangular lattice.

**Key physics**:
The broken bond does not just absorb — it creates:
1. A shadow (depleted sites beyond the bond)
2. Wall face accumulation (enriched near side)
3. Secondary wavefronts along rerouted paths that arrive EARLIER than baseline
4. A characteristic diffraction pattern that is unique to the broken bond location

"The good ones get more earlier" — bonds NOT on the broken path receive
rerouted weight before the baseline wavefront arrives there, because
the wall face excess radiates in new directions.

---

## Why this is free of artifacts

Unlike the scalar L(v,t) and the Gram matrix coherence G:

- **No color ordering dependence**: δ ln T is computed from n(v,t) and n(v,t+1),
  physical observables independent of which Trotter color fired first
- **No BP convergence debate**: both baseline and perturbed use the same BP;
  whatever artifacts exist cancel in the log ratio
- **No dominant site bias**: normalized by n(v,t), high-weight sites don't dominate
- **No metric choice**: the diffraction pattern is physics, not a statistical construction
- **Physical gating**: n(v,t) > ε_th means "only where the wavefront has arrived"

---

## Connection to paper §3 and the tomography paper

- T_base(v→u) = n_base(u,t+1)/n_base(v,t): bond transfer coefficient (§3, eq.3)
- T_pert(v→u) = n_pert(u,t+1)/n_pert(v,t): perturbed transfer coefficient (§3, eq.4)
- L(v,t) ≈ ln(T_base/T_pert) when u is on center-side, v is shadow-side (§3, eq.13)
- δ ln T generalizes this to ALL bond directions simultaneously, as a vector field

The simulation captures the diffraction correctly because it implements faithful
Trotter evolution. The cleanliness of the physics **justifies the simulation** —
the fact that the diffraction pattern is so structured confirms that n(v,t) is
physically meaningful even where BP is non-converged, because the perturbation
is small and the ratio cancels systematic errors.

---

## Quantification — pointing at the broken bond (TODO)

To identify the broken bond from δ ln T:

**Bond score**: aggregate δ ln T along each bond direction across cycles:

    score(u,v) = sum_t |δ ln T_{v→u}(t)| + |δ ln T_{u→v}(t)|

The broken bond should have the maximum score. The asymmetry between forward
and reverse directions encodes the bond orientation (which side is source, which shadow).

**Next steps**:
1. Implement bond score from δ ln T
2. Compare with existing BFS detection: does it agree? Does it give cleaner signal?
3. Test on 2-bond case: do we see two separate diffraction patterns?
4. Build Gram matrix from δ ln T bond vectors — should be more physically grounded
   than raw L(v,t) cosine similarity

---

## Figures

- `NOTES/delta_logT_vector.png`: δ ln T vector field for cycles t=1→2, 2→3, 3→4
- `NOTES/delta_J_magnitude.png`: |δJ| scalar per site for comparison

## Script

`scripts/compute_vector_field.py` (extend to include δ ln T)

---

## Detection results (2026-11-07)

**Script**: `scripts/detect_delta_logT.py`
**Parameters**: threshold=0.01, top_k=n_bonds, all 3 scans, all cycles

### Single bond (mawgm_f64, 10 trials)
**10/10 BOTH** — every bond rank #1. Pure physics, no failures.

### Two bonds (mawgm_v2, 10 trials)
**9/10 BOTH** — same as BFS. Only trial 4 (bonds 9,51) fails:
- Bond 8 (10↔11) sits between bonds 9 and 51 — their overlapping
  diffraction patterns constructively interfere at bond 8
- Scan-by-scan analysis: bond 51 appears top-3 in ALL scans; bond 9
  appears top-3 in scans 2 and 3; bond 8 dominates only scan 2
- Voting across scans would correctly identify both true bonds

### Four bonds (mawgm_4b, 5 trials)
**1/5 BOTH, avg 3/4** — BFS gets 3/5. Methods fail on different trials.

### Eight bonds (mawgm_8b, 2 trials)
**avg 6/8** — δ ln T finds 5/8 and 7/8. BFS gets trial 2 (8/8), fails trial 1.

### Comparison

| n_bonds | δ ln T | BFS |
|---------|--------|-----|
| 1 | 10/10 | 10/10 |
| 2 | 9/10 | 9/10 |
| 4 | 1/5 BOTH, avg 3/4 | 3/5 BOTH |
| 8 | avg 6/8 | mixed |

### Failure mode
Both methods fail when broken bonds are **adjacent** — their signals
overlap and boost a neighbor bond's score. The methods fail on
**different** trials, making them complementary.

**Conclusion**: the iterative calibration loop is the right architecture.
Fix the bonds found in one pass, re-run, and the remaining bonds surface
cleanly without interference. Static centers will always have corner cases
when broken bonds cluster together.
