# Red-Blue Tomography: Experimental Roadmap

*Created 2026-10-05. Check off experiments as completed.*

---

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done — results in `logs/`

---

## Part 0: Baseline (DONE)

- [x] **Single bond, identity PTM, 7×7, aperture synthesis**
  - Bond 12 (15↔16), χ=50, 5 cycles, 6 centers
  - Result: 82× margin, ✓ CORRECT
  - Logs: `logs/tomo_6c/`

- [x] **Single bond, identity PTM, 7×7, seismic (multi-center)**
  - Bond 12, χ=80, 5 cycles, 6 centers simultaneous
  - Result: ✓ CORRECT, L(15)=+0.53, L(16)=-0.02
  - Logs: `logs/tomo_v2/base_B_bond12.log`, `logs/tomo_v2/pert_B_bond12.log`

---

## Part 1: Complexity and Timing

- [x] **1a. Timing benchmark: Aperture vs Seismic on 7×7**
  - Seismic: 272s (2 GPUs parallel, no v_hop)
  - Aperture no v_hop: 300s (3 GPUs, 1.10× seismic)
  - Aperture v_hop=0.5: **191s (3 GPUs, 0.70× seismic — 30% faster)**
  - Both ✓ CORRECT. Logs: `logs/timing_benchmark/`
  - Run both methods, measure wall-clock and VRAM
  - Aperture: 6 independent runs (parallelizable)
  - Seismic: 1 multi-center run (chi saturates faster)
  - Script: `bash scripts/run_timing_benchmark.sh`
  - Expected result: table of (method, topology, chi, cycles, time, VRAM)

- [ ] **1b. Timing benchmark: Aperture vs Seismic on heavy-hex**
  - Same as 1a but on heavy-hex 3×3
  - Binary: `main_pauli_sweep_hh_dep`
  - Note: need to define p-centers for heavy-hex topology first

---

## Part 2: Overlapping Shadows (Resolution Test)

- [x] **breaker.py + C++ multi-bond support**
  - `--bond 12,13` in breaker.py; `--perturb-bond=12;13,1.2` in C++
  - Commit: 4913a9a

- [x] **2a. Two adjacent bonds: 7×7, bonds 12+13**
  - Break bond 12 (15↔16) AND bond 13 (16↔17) simultaneously
  - Result (3 cycles, chi=50): **red → white → blue** triplet pattern
    - Site 15: red (+0.55) — upstream of bond 12
    - Site 16: white (0.00) — shared endpoint, destructive interference
    - Site 17: blue — downstream of bond 13
  - **Key result**: the white site IS the interference node, not a detection failure
  - Analog: thin-film interference dark fringe — tells you path length difference
  - Searcher WRONG (looking for red-blue pair); needs triplet detection
  - Logs: `logs/tomo_v2/*bond12_13*`
  - GIF: `logs/tomo_v2/ce_B_bond12_13_3c.gif`

  **New signal vocabulary:**
  - Single broken bond: **red — blue** (two sites, sign flip)
  - Two adjacent broken bonds: **red — white — blue** (three sites, interference node)
  - The number of white sites between red and blue = number of broken bonds in the wall

- [ ] **2b. Two non-adjacent bonds: 7×7, bonds 12+35**
  - Break bond 12 (15↔16) AND bond 35 (41↔42) — far apart
  - Shadows should NOT overlap
  - Control experiment: if the method fails here, it's a fundamental limit
  - Expected: both bonds identified independently

- [ ] **2c. Two adjacent bonds: ensemble resolution**
  - Same as 2a but with aperture synthesis (6 independent centers)
  - Question: does ensemble separate what seismic cannot?
  - Hypothesis: aperture has higher "angular resolution" for nearby bonds

---

## Part 3: Failure Taxonomy

- [ ] **3a. Collinear bonds: 7×7, bonds on row 2**
  - Break all 6 bonds in row 2 (bonds 12,13,14,15,16,17)
  - Creates a "wall" blocking the entire row
  - Question: does the method see a wall (whole row red/blue) or individual bonds?

- [ ] **3b. Hexagonal cluster: heavy-hex**
  - Break all bonds touching one site (2-3 bonds in heavy-hex)
  - Simulates a physically faulty qubit
  - Question: RED cluster + BLUE halo? Can we identify the faulty site?
  - Script: `bash scripts/run_hex_cluster.sh <site>`

- [ ] **3c. Graduated failure (dimmer test): 7×7**
  - Single bond, δJ = 0.50, 0.20, 0.10, 0.05, 0.02, 0.01
  - Find the minimum detectable δJ
  - Use: physically correct PTM (not identity) — need to fix PTM computation
  - Question: when does the switch become a dimmer?

---

## Part 4: Physical PTM (Realistic Coupling Change)

- [ ] **4a. Fix PTM perturbation to be physically correct**
  - Current: identity PTM (maximally disruptive, too clean)
  - Need: recompute Heisenberg gate PTM with new J analytically in C++
  - Or: use breaker.py output (correct Python computation) fed into C++
  - Status: off-by-one bug fixed (commit d67d457), trig matching still approximate

- [ ] **4b. Repeat Part 0 with physically correct PTM**
  - Bond 12, δJ=0.20, correct Heisenberg PTM
  - Compare signal strength to identity PTM result
  - Expected: smaller but still detectable sign flip

---

## Part 5: Heavy-Hex Architecture

- [ ] **5a. p-center placement for heavy-hex**
  - Run `python3 scripts/find_centers.py --graph data/graphs/heavy_hex_3x3.json --d 3`
  - Already done (8 centers needed for d=3) — choose best subset

- [ ] **5b. Single bond on heavy-hex, aperture synthesis**
  - Pick a bond near center of heavy-hex lattice
  - Run 8-center aperture, compare with 7×7 result
  - Question: does irregular topology affect detection quality?

---

## Paper Figures Needed

- [ ] **Fig 1**: Cross-entropy GIF → 4-panel static PNG (cycles 1,2,3,5)
  - From `logs/tomo_v2/ce_B_bond12.png` — already exists!
- [ ] **Fig 2**: Individual center directional arrows (6 panels)
  - From `logs/tomo_6c/ce_c*.gif` → static PNGs
- [ ] **Fig 3**: Ensemble bond score bar chart (bond 12 vs others)
- [ ] **Fig 4**: Failure taxonomy: single / adjacent / collinear / hexagonal

---

## Notes

- All identity PTM experiments use `--perturb-bond=N,J` with identity in C++
- Multi-bond: `--perturb-bond=12;13,1.2` — semicolons separate bonds, comma before J
- All perturbed bonds currently receive the SAME identity PTM (symmetric perturbation)
  - Future: vector of independent PTMs for heterogeneous multi-bond perturbations
- Physically correct PTM needs fixing before sensitivity experiments
- The 82× margin is the headline number — protect it with careful experiment design
- Hong-Kung nod: the paper title is "Red-Blue Hamiltonian Tomography via the Operator Lightcone Wavemap"

## Seismic vs Aperture: Different Information (2026-10-06)

*What we learned this morning — radar and sonar analogy.*

### The temporal patterns

**Seismic (simultaneous multi-center):**
- C1: W  W  W  — wave not yet arrived
- C2: W  R  B  — bond 13 crossed first (site 16 red, 17 blue)
- C3: R  W  B  — bond 12 crossed (site 15 red), site 16 WHITES (interference node)

**Aperture (independent centers, sum L fields):**
- C1: W  W  W  — nothing
- C2: W  R  b  — right-side centers (25,32) hit site 16 first
- C3: W  R  B  — shadow deepens at site 17
- C4: R  R  B  — left-side centers (10,30) finally reach site 15

### The key difference

**Seismic**: interference IS cancellation. Site 16 whites because it is simultaneously
upstream of bond 13 (should be red) AND downstream of bond 12 (should be blue).
The two effects cancel → white node = shared endpoint signature.
Information: the WHITE SITE identifies the wall between two bonds.

**Aperture**: interference WITHOUT cancellation. Each center's L computed independently,
then summed. No cancellation because the signs are preserved per center:
- Right centers (25,32): positive L at site 16 (blocked by bond 13)
- Left centers (10,30): positive L at site 15 (blocked by bond 12)
Sum = both positive → red-red-blue, not red-white-blue.
Information: the TEMPORAL SEQUENCE reveals which centers are upstream of which bonds.

### The radar/sonar analogy

**Seismic = active sonar (ping)**: all sources fire simultaneously, interference pattern
in the received signal encodes the geometry. The white node (destructive interference)
is the spatial signature of a wall.

**Aperture = passive radar (ELINT)**: each receiver independently tracks the signal
from its direction. The temporal ordering of arrivals encodes directionality.
Signal appears at site 16 before site 15 because right-side centers are closer to
bond 13 than left-side centers are to bond 12.

### What each method knows

| Property | Seismic | Aperture |
|----------|---------|---------|
| Single bond detection | ✓ (sign flip) | ✓ (sign flip) |
| Wall detection (n bonds) | ✓ (white node count) | ✗ (no cancellation) |
| Directional information | ✗ (simultaneous) | ✓ (temporal sequence) |
| Bond ordering in wall | ✗ | ✓ (arrival time order) |
| Computation cost | 2 simulations | 2p simulations |
| Chi saturation onset | Earlier (multi-center) | Later (single-center) |

### The temporal pattern IS the measurement

Seismic: spatial pattern (red-white-blue) at a given cycle identifies wall structure.
Aperture: temporal pattern (order of signal arrival) identifies bond positions and
which centers are on which side.

For single broken bond: both methods equivalent.
For multiple adjacent bonds: seismic sees the wall structure; aperture sees the
directional tomogram. They are complementary, not competing.

### Paper framing

"Seismic tomography detects wall structure through spatial interference; aperture
synthesis detects bond positions through temporal directionality. Together they
form a complete Hamiltonian tomography toolkit."

## Future Direction: Phased Seismic Array

**Idea (2026-10-06)**: Instead of firing all p-centers simultaneously, introduce
a one-cycle delay between successive centers:

  Center c_1 fires at t=0 → wave reaches bond by t=d/v_hop
  Center c_2 fires at t=1 → wave reaches bond one cycle later
  ...

The time-staggered initialization preserves the directional information that
simultaneous firing loses — the red and blue signals from opposite sides
arrive at different times and do NOT cancel at the shared endpoint.

**Connection to physics**:
- Radio astronomy: phased array antennas introduce delays to steer the beam
- Acoustics: time-reversal focusing uses delayed firing to focus on a target
- Here: delayed centers create a "beam" that focuses detection on specific bonds

**Implementation challenge**: in the Heisenberg picture, the operator is
initialized once at t=0. Injecting Z_k at mid-evolution changes the simulation
structure fundamentally.

**Workaround A — time-shifted aperture**:
Run p separate simulations, center k starts at cycle offset t_k.
Align L fields on a common absolute time axis before summing.
Essentially aperture synthesis with time-shifted comparisons.
Cost: same as aperture (2p simulations), gain: directional phase information.

**Workaround B — weighted initialization**:
Initialize all centers at t=0 but with different weights w_k (e.g. w_k = e^{iφ_k}).
Algebraically clean but n(v,t) = sum |c_P|^2 loses phase information.
May not achieve the desired directional effect.

**Key benefit**: in the interference case (two adjacent bonds, shared endpoint),
the phased approach would separate the red and blue signals in TIME rather than
needing the ensemble to separate them spatially. Site 16 sees red at t=t_first(16)
from center 10, and blue at t=t_first(16)+1 from center 25 — the sign flip is
time-resolved rather than spatially resolved.

**Status**: theoretical proposal, not yet implemented.
Connection: time-reversal focusing in acoustics / phased array radar.
Paper note: "Phased seismic initialization — future direction."

## Open Theoretical Questions

**Seismic blind spot:**
If p-centers are placed symmetrically around the broken bond, waves arrive from
opposite sides with equal amplitude. The seismic L field cancels:
L_seismic(v) = L_red + L_blue ≈ 0. The bond becomes invisible.
Aperture synthesis is immune — each center's L computed independently before summing.
Implication: p-center placement must be asymmetric relative to any bond to guarantee
seismic detection. The geodesic bond cover (d_max=2) does not guarantee this.

**Perturbation symmetry:**
The identity PTM is symmetric — blocks propagation equally in both directions.
The wavemap signal is asymmetric because the measurement is directional (wave
travels from center outward). Different centers see different sides of the bond.
The same symmetric perturbation produces antisymmetric L signals at the two
endpoints — this is the detection mechanism.

**Interference (experiment 2a):**
Two adjacent bonds 12+13 share endpoint site 16. The seismic L at site 16 receives:
- Red signal from upstream (bond 12 blocks wave coming from center 10)
- Blue signal from downstream (bond 13 blocks wave going further right)
Net effect at site 16: partial cancellation. The interference pattern at site 16
is the key diagnostic for the resolution limit of the method.
