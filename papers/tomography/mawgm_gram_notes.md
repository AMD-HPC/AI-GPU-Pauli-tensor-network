# Gram Matrix Coherence Analysis — Notes on Muhammad's Finding 1

## What he did

Ran 85 simulations: 1 baseline + 84 perturbed (one per bond, J=1.2).
Each bond e produces a response vector:

    s_e = [ L(v,t) ]_{v=1..49, t=1..5}  ∈ R^245

where L(v,t) = -log(n_pert_e(v,t) / n_base(v,t)) using the shared baseline.

Then computed the 84×84 Gram matrix of pairwise cosine similarities:

    G[i,j] = |<s_i_normalized, s_j_normalized>|

This is an **offline characterization** of the scan plan — computed once,
reusable for any diagnostic run on the same lattice with the same centers.

## What it means

G[i,j] measures how similar the response of bond i looks to the response
of bond j under the current scan plan:

- G[i,j] ≈ 0: bonds produce orthogonal response fields → easy to distinguish
- G[i,j] ≈ 1: bonds produce nearly identical response fields → confusable

Results on our 3-scan plan (rect_7x7):
- Median pairwise coherence: 0.103 (most bonds easily separated)
- Worst pair: 0.867 (bonds 39 and 81, sharing vertex (6,4))
- Top-10 worst pairs: 10/10 share a vertex — always H-V pairs at a corner

Detectability (max |L| over all readings) spans 43× across bonds:
- Weakest: bond 81, detectability 0.089
- Strongest: bond 82, detectability 3.80

## Two separate problems

1. **Coverage**: is ||s_e|| > 0? If no center reaches bond e within the
   cycle budget, the bond is invisible (detectability = 0). Must be fixed
   first by ensuring scan plan covers all bonds.

2. **Coherence**: among visible bonds, which pairs have high G[i,j]?
   These are the pairs the algorithm cannot distinguish when one is broken.
   Coverage is necessary but not sufficient.

## Why vertex-sharing bonds are confusable

Two bonds (u,v) and (v,w) sharing vertex v produce similar response fields
because the wavefront signal localizes to the shared vertex v — the algorithm
sees "something is wrong near v" but cannot tell which of the two adjacent
bonds is broken. The response vectors s_(u,v) and s_(v,w) differ only in
subtle downstream effects, which the current centers do not resolve.

## G is a function of the scan plan

The same lattice produces different G matrices for different center
placements. The optimization problem:

    minimize over centers:  max_{i≠j} G[i,j]

This is computable: for each candidate scan plan, run 85 simulations,
build G, read off the worst-case coherence. Try different placements,
pick the one with lowest worst-case coherence.

## Proposed improvement: targeted center placement

Given the top-10 worst pairs from G, each sharing a vertex v*:

1. Place a new center at v* (or one hop to one side of v*)
2. This breaks the symmetry: bond (u,v*) and bond (v*,w) are now illuminated
   from different angles — one becomes wall face (BLUE), the other shadow (RED)
3. Run 84+1 calibration with augmented centers
4. Recompute G — verify worst-case coherence drops

Cost: ~10 extra simulations. Can be run in one GPU session.

The hypothesis: a center at the shared vertex v* makes the two vertex-sharing
bonds distinguishable because it sits between them, creating opposite signs
in their response vectors.

## Lessons from computing G ourselves

**1. f32 vs f64 — we did not reproduce Muhammad's G**

We ran f32, he ran f64. The top-10 pairs are almost entirely different (only bonds 26,27 in common). Part of this is the multi-center vs single-center difference, part is precision. Before any G-based statement goes in the paper, the two must be reproduced on the same configuration (single-center, f32, convergence-gated). The structure (vertex-sharing = confusable) survives; the specific pairs do not.

**2. Long cycles introduce interference trails**

With 5 cycles and multiple centers, the wavefront has traversed loops and the BP fields are non-converged from step 2 onward. The response vectors at steps 3-5 are contaminated by wave interference — reflections, secondary arrivals — not just the bond perturbation. This inflates coherence artificially and shifts which pairs appear most confusable. For a valid G, use at most 2 cycles (interference-free window).

**3. G is dominated by the highest-weight site in each response vector**

The cosine similarity is dominated by whatever site has the largest L(v,t) values. For our seismic configuration, the scan centers sit ON corners — so corner sites have enormous L values and dominate the response vectors of all adjacent bonds. Two bonds sharing a corner center appear maximally confusable in G (μ=0.93) even though they are physically the easiest to distinguish (center on shared vertex = clean shadow detection). The G metric is wrong for our detection method. The right inner product is the sign-flip contrast |L(u)-L(v)| across bond endpoints — not the cosine of full (site,cycle) response vectors.

## Connection to the paper

Our paper identifies the hard case empirically: "two orthogonal bonds sharing
an endpoint — the hardest geometric configuration." Muhammad's contribution:

1. This is now **predictable in advance** from G, before any trial
2. It gives a **formal identifiability bound** (compressed sensing: k-sparse
   recovery requires µ < 1/(2k-1); for k=2, µ < 1/3 ≈ 0.33; our worst pair
   at 0.87 exceeds this → two-bond recovery not guaranteed for that pair)
3. It enables **scan optimization**: choose centers to minimize worst µ

The detectability 43× span is also worth noting in the paper — a single
sensitivity threshold ε_th does not apply uniformly to all bonds. The weakest
bond (0.089) requires significantly more shots or a tighter cutoff than the
strongest (3.80).

## What Finding 1 does NOT change

- Single-bond detection: 20/20 result stands. G diagonal = 1 by definition,
  no ambiguity for single bonds (unless coverage fails).
- The sign-flip detection method is correct — G just quantifies when the
  inverse problem is well-posed.
- The iterative calibration loop still works: fix the easy bonds first,
  then the hard pairs surface more clearly.
