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

- [ ] **2a. Two adjacent bonds: 7×7, bonds 12+13**
  - Break bond 12 (15↔16) AND bond 13 (16↔17) simultaneously
  - Their blue shadows both hit site 16 — maximum overlap
  - Question: does ensemble resolve them as two bonds or merge into one?
  - Script: `bash scripts/run_tomography_v2.sh B 12;13 0.20`

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
