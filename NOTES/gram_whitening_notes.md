# Gram Matrix Whitening — Notes and Response to Muhammad's Finding 2

## What whitening does

For V shaped (n_obs × n_bonds) = (245 × 84):

Each row is one observation (v,t) — L(v,t) across all 84 bonds.
Whitening divides each row by its std across bonds:

    sd = V.std(axis=1, keepdims=True)   # std across 84 bonds per observation
    W  = V / sd                          # zero-std rows → inf → excluded

Effect:
- Rows where all bonds respond identically → sd≈0 → excluded
- Rows where bonds respond differently → amplified to unit variance
- Rows with all zeros (lightcone) → sd=0 → excluded naturally

## What we learned from measuring it

**Zero rows (coverage)**:
- 5-corner: 12 zero rows (4.9%) — all at t=1, sites ≥2 hops from all centers
- 3-scan: 25 zero rows (3.4%) — same structure, more covered by additional scans
- These are sites outside the t=1 lightcone — a coverage property, not a defect
- Whitening correctly drops them (division by zero → excluded)

**Low-variance non-zero rows (hub sites)**:
- At late cycles (t=4,5), some interior sites have low variance across bonds
- These are sites at the boundary of the noiseless lightcone — small signal,
  similar for all bonds because the wavefront just barely arrived
- Small variance ≠ no information — it means the discriminating signal is weak,
  not absent. Whitening amplifies these rows, potentially amplifying noise too.

**Row sparsity**:
- 5-corner: 193/245 rows have all 84 bonds non-zero (78.8%) — nearly dense
- Multi-center seismic illuminates most sites from multiple angles
- Only 2 rows have exactly 4 non-zero bonds (early-cycle boundary sites)
- Concern about sparse rows amplifying coverage artifacts is minimal in practice

## Results: raw vs whitened G

| Config | Raw max μ | Whitened max μ | Change |
|--------|-----------|----------------|--------|
| 5-corner H-first | 0.804 | 0.788 | -2% |
| 5-corner V-first | 0.804 | 0.789 | -2% |
| 3-scan H-first | 0.662 | 0.535 | -19% |
| 3-scan V-first | 0.662 | 0.535 | -19% |

**Whitened G is color-ordering robust for 5-corner**:
- Both colorings → bonds 18,56 (v=22) as #1
- Color artifact substantially removed

**Whitened G partially removes color artifact for 3-scan**:
- Pairs 14,58 (v=17), 27,67 (v=33), 34,75 (v=41) are identical rank 4,5,7
  in both H-first and V-first → these are the truly geometric confusable pairs
- Pairs involving site 25 still differ between colorings → residual artifact

## Remaining limitations of whitened cosine

**1. Absolute value collapses sign**:
The cosine uses |cos(s_i, s_j)| — treating same-direction (both RED) and
opposite-direction (one RED, one BLUE) as equally "similar." But opposite signs
mean maximally distinguishable — that's the sign flip our detector exploits.
The absolute value discards the most important information.

**2. Time is unweighted**:
The inner product gives equal weight to t=1 (converged BP, causal, clean) and
t=5 (non-converged BP, interference, noisy). Late cycles dominate because they
have larger L values, but they are less physically reliable. A time-weighted
inner product with α^t decay (α<1) would better reflect the causal structure.

**3. Whitening amplifies weak late-cycle rows**:
Rows with small variance (late-cycle boundary sites) get amplified. The signal
is real but weak, and whitening cannot distinguish weak real signal from weak
noise. A threshold on minimum mean signal before whitening would help.

## What survives all metrics

Across raw G, whitened G, f32, f64, Householder, Cholesky, H-first, V-first:
- **Vertex-sharing structure: 10/10 in all cases**

The confusable pairs are the vertex-sharing ones. That is the only claim
robust to all metric and precision choices.

The specific ranked list and μ values depend on the metric convention and
should not be quoted without naming the metric and aggregation method.

## The right metric for our detection method

Our BFS algorithm uses |L(u) - L(v)| — the sign-flip contrast across bond
endpoints. A G built from this would directly measure detection difficulty
rather than response vector similarity. This is equivalent to asking:
"does perturbing bond i create a sign flip that looks like the sign flip
from perturbing bond j?" — which is the right identifiability question.

The aperture G (49 centers × 84 bonds, single-center runs) would also
address the time weighting problem: with enough cycles each bond is seen
at its own t0(v) frontier crossing — the most reliable signal moment.

## Script

```bash
python3 scripts/compare_gram_whitened.py \
    --configs logs/gram_5c_standard:5-corner-H \
             logs/gram_5c_swapped:5-corner-V \
             logs/gram_standard:3-scan-H \
             logs/gram_swapped:3-scan-V \
    --top_k 10
```
