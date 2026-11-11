# Onion Peel Simulation Architecture (2026-11-11)

## The core idea

The standard lightcone allocation keeps all sites within the wavefront
in memory simultaneously (volume). The onion peel advances the wavefront
one shell at a time, keeping only the current boundary layer in memory,
carrying the discarded interior as compact BP messages pointing outward.

**Memory scaling:**
- Standard lightcone: O(χ^degree × r³) in 3D, O(χ^degree × r²) in 2D
- Onion peel:         O(χ^degree × r²) in 3D, O(χ^degree × r)   in 2D

The boundary (surface) is always smaller than the volume by a factor of r.

---

## Algorithm

**Cycle 0**: initialize center site, compute n(center, 0).

**Cycle i → i+1**:
1. Current state: layer i sites + BP messages from layer i-1 pointing outward
2. Apply Trotter gates across the boundary (between layer i and layer i+1)
3. Compute n(v, i+1) for all new boundary sites v in layer i+1
4. Update BP messages: layer i becomes the new "interior" message layer
5. **Discard** layer i site tensors — keep only BP messages pointing outward
6. Layer i+1 is now the active boundary

**What is kept**: only the current shell of sites + a ring of outward-pointing
BP messages from the previous shell. Everything else is discarded.

**What is carried**: the BP messages from the discarded interior encode
the entanglement built up behind the wavefront. On trees, this is exact.
On loopy graphs, it is approximate — but the approximation is most accurate
at the boundary where fewer loops have been traversed.

---

## Why it's most accurate at the boundary

The BP approximation degrades with the number of loops traversed. The
boundary sites at cycle i have been touched by exactly i Trotter gates
— they have the fewest accumulated loop corrections. The deep interior
has been through more gate applications and accumulated more loop error.

For detection: we only need n(v,t) at the frontier t₀(v) — the first
arrival. This is exactly where onion peel is most accurate. The method
computes precisely what we need and nothing else.

---

## Memory savings

**2D rectangular lattice (degree 4)**:

| Step r | Volume (sites) | Surface (sites) | Saving factor |
|--------|---------------|-----------------|---------------|
| 2      | ~12           | ~8              | 1.5×          |
| 5      | ~80           | ~20             | 4×            |
| 10     | ~315          | ~40             | 8×            |

**3D cubic lattice (degree 6)**:

| Step r | Volume (sites) | Surface (sites) | Saving factor |
|--------|---------------|-----------------|---------------|
| 1      | 7             | 6               | 1.2×          |
| 2      | ~33           | ~26             | 1.3×          |
| 3      | ~93           | ~54             | 1.7×          |
| 5      | ~515          | ~150            | 3.4×          |

The saving grows with r — most beneficial for larger lattices and more cycles.

---

## Consequence: larger χ or larger lattice

The freed memory from the discarded interior can be reallocated to either:

**Option A — Larger χ at same lattice**:
- Freed memory factor F → χ can grow by F^(1/degree)
- 2D at step 5: F≈4 → χ grows by 4^(1/4) ≈ 1.4×
- 3D at step 3: F≈1.7 → χ grows by 1.7^(1/6) ≈ 1.1× (modest)
- Modest improvement — χ^degree kills the gain quickly

**Option B — Larger lattice at same χ** (better):
- Use freed memory to simulate a bigger lattice
- 2D 7×7 at χ=50 → 20×20 at χ=50 same memory with onion peel
- 3D 4×4×4 at χ=20 → 5×5×5 at χ=20 same memory

**Option C — Both**: moderately larger χ AND moderately larger lattice.
This is the realistic operating point — not a dramatic χ increase but
a compound improvement.

---

## Scalability implication

Currently: "seismic cost is fixed at 6 simulations, memory scales as χ^4 × area"

With onion peel: "seismic cost is fixed at 6 simulations, memory scales as χ^4 × perimeter"

For large 2D lattices (20×20, 30×30), the perimeter grows as k while
the area grows as k² — the method becomes **truly scalable** to arbitrarily
large lattices at fixed memory, as long as the detection horizon (d_max hops)
is small.

This connects directly to the paper's scalability claim: aperture cost per
center is independent of grid size (each center sees only its local ball).
Onion peel makes the seismic approach equally independent of grid size.

---

## Implementation plan

### Phase 1: Verify in 2D
1. Implement onion peel in `measures.hpp` as an alternative to full lightcone
2. Add `--mode=onion` flag
3. Run mawgm_f64 trials — verify 20/20 detection survives
4. Compare n(v,t) at boundary sites: full vs onion peel
5. Measure memory savings at each step

### Phase 2: Apply to 3D
1. Same code, different grid class
2. Run 5×5×5 at χ=20 or χ=30 — previously OOM, now fits
3. Inject broken bond, run 2 steps, check detection

### Phase 3: Larger 2D lattices
1. 20×20 or 30×30 at χ=50
2. Test whether detection generalizes to larger lattices
3. This validates the scalability claim with real data

---

## Connection to rank-k BP

The onion peel reduces the number of active sites. The rank-k BP reduces
the information content per message. They are **orthogonal compressions**:

- Onion peel: fewer sites in memory
- Rank-k BP: smaller messages per bond

Combined: onion peel × rank-k BP = O(k × χ × surface) instead of O(χ² × volume)
for the memory footprint. This is a dramatic reduction — potentially making
5×5×5 at χ=50 feasible on a single GPU.

The rank-k BP is the prerequisite — without it, the BP messages from the
discarded interior still cost χ² per bond, partially canceling the onion peel gain.

---

## Key insight: detection needs only the boundary

For Hamiltonian tomography, we evaluate L(v, t₀(v)) at the **frontier crossing
time** — the moment the wavefront first arrives at site v. This is exactly the
boundary of the onion peel at that cycle. We never need n(v,t) deep in the
interior for detection purposes.

The onion peel is not a heuristic approximation — it is the **natural
computational structure** for wavefront detection: advance the boundary,
read the signal, discard what's behind.
