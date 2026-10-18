# MAWGM — Results Notes (2026-10-18)

## What we built

**MAWGM** (Micro Aperture Wavefront Geometry Measure) — a lightweight protocol
for detecting broken or miscalibrated bonds in a quantum lattice, using Pauli
wavefront propagation as the probe.

Two circuits only: baseline (clean Hamiltonian) and perturbed (candidate bond weakened).
Observable: n(v,t) = fraction of trajectories reaching site v by cycle t.
Signal: log(B(v)/N(v)) = integrated log-ratio across all cycles.

## Core algorithm

**BFS threshold-crossing** — sites are colored by when the wavefront arrives:
- RED (log > 0): baseline arrives earlier → site in shadow of broken bond
- BLUE (log < 0): perturbed arrives earlier → site on wall face of broken bond
- Sign flip across bond endpoints → broken bond declared

**Integral detection** B(v)/N(v) = Σ_t n_base / Σ_t n_pert:
- More robust than point detection L(v,t0)
- No threshold on individual n values
- Captures both delay and sustained amplitude difference

**Seismic scan**: 3 center placements (corners+center, quadrants, mid-edges)
covering all 84 bonds of the 7×7 grid. 6 simulations total (base+pert × 3 scans).

## Detection results (single iteration, seismic only)

| n_bonds | bonds found | rate | ALL trials |
|---------|-------------|------|------------|
| 1       | 20/20       | 100% | 10/10      |
| 2       | 19/20       | 95%  | 9/10       |
| 4       | 18/20       | 90%  | 3/5        |
| 8       | 15/16       | 94%  | 1/2        |

**Per-bond detection rate stays above 90% even at 8 simultaneous faults.**

Systematic failures:
- **Bond 8 (10↔11)**: geometrically awkward location — wavefront from the 3 seismic
  centers creates a weak/symmetric illumination. When paired with bond 45 (BLUE-BLUE
  pattern), sign flip is invisible from seismic centers. Caught by full antenna grid
  (col2 gives sign flip) in iteration 2.
- **Bond 36**: similar weak illumination from seismic centers.

## Iterative loop — the key insight

MAWGM is not a one-shot detector. It is a **calibration loop**:

1. Run seismic → declare top-k broken bonds (>90% correct per bond)
2. Recalibrate those bonds on hardware
3. Re-run → fixed bonds disappear, remaining ones surface cleanly
4. Stop when n_pert(v,t) ≈ n_base(v,t) everywhere

**Why this works:**
- At 8 bonds, find 7 in iteration 1, fix them → 1 bond remains → clean sign flip
- The hard cases (BLUE-BLUE, weak illumination) become easy once interference removed
- Stopping criterion is physical: wavefront matches baseline = device is calibrated

**Design choices are jointly optimal for the iterative loop:**
- Short cycles (3-5): only need to rank the next bond to fix, not all at once
- Reasonable threshold (1e-3): broad BFS coverage > high per-site SNR
- 3 seismic placements: covers all bonds, no blind spots (except geometric edge cases)

## Connection to L(v,t) and MPF — the sequitur

This work is the **inverse** of the MPF papers:

| Direction | Known | Inferred | Observable |
|-----------|-------|----------|------------|
| MPF (noise) | Hamiltonian H | noise γ | n(v,t) |
| MAWGM (geometry) | noise baseline | coupling deviations δJ | n(v,t) |

Same observable. Same circuits. Same hardware protocol.

L(v,t) = -log(n_pert/n_base) was introduced in the L-MPF paper as the
local cross-correlation signal for noise inference. Here it becomes the
detection signal for Hamiltonian geometry. The integral version
log(B(v)/N(v)) = log(Σn_base / Σn_pert) is more robust for bond detection.

**On hardware:** measuring n(v,t) uses the same circuits already used for MPF —
MAWGM monitoring is zero additional overhead. Run MPF to characterize noise,
run MAWGM to characterize coupling geometry. Two inferences from one measurement.

## Heavy-hex geometry — likely easier

The 7×7 rectangular grid is a worst case in some sense:
- High connectivity (degree 3-4) → more interference paths
- No natural seismic "corridor" for some bond orientations

Heavy-hex has degree 2-3 and a natural linear chain structure. The wavefront
propagates more cleanly along chains — fewer interference paths, stronger
sign flips, less BLUE-BLUE ambiguity. MAWGM should work at least as well
on heavy-hex, likely better.

## Paper position

MAWGM positions as a **targeted geometric health check**:
- Not full Hamiltonian tomography (exponentially expensive)
- Not MPF noise characterization (different inference direction)
- Lightweight continuous monitoring: 6 simulations, 3-5 cycles, ~5 minutes
- Scalable: cost grows with grid size, not with number of broken bonds
- Hardware-ready: same circuits as MPF, Binomial statistics, shot-budget analysis
  carries over directly from MPF framework

The paper narrative:
1. We showed n(v,t) characterizes noise (MPF papers)
2. Now we show the same observable characterizes coupling geometry (MAWGM)
3. Same hardware, same circuits, opposite inference direction
4. Validated on 7×7 grid: >90% per-bond detection at up to 8 simultaneous faults
5. Iterative loop provides stopping criterion and makes hard cases tractable
