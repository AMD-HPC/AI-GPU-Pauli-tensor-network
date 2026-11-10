# Future Direction: 3D Lattice Tomography (2026-11-09)

## Goal

Extend the tomography method to a 4×4×4 cubic lattice — the smallest 3D case
that fits on one MI100 GPU and tests all the new physics of 3D wavefront propagation.

---

## Physics differences: 2D → 3D

**Coordination number**: 4 (2D rectangular) → 6 (3D cubic)

**Wavefront speed**: more paths per site → effective v_hop likely higher.
Need to measure on the new lattice (same method: lightcone sweep, threshold crossing).

**BP convergence**: 3D has shorter loops (minimum loop length = 4 in cubic)
compared to 2D (also 4). But with 6-coordination, more loops pass through each site.
Multi-center BP interference will be at least as severe as 2D — ratio cancellation
argument still applies.

**Color classes**: 3D Heisenberg needs 3 color classes:
- Color 0,1: x-direction bonds (even/odd columns)
- Color 2,3: y-direction bonds (even/odd rows)
- Color 4,5: z-direction bonds (even/odd layers)
Or a reduced 3-coloring may work for the cubic lattice.

---

## Lattice specification

**4×4×4 cubic lattice**:
- Sites: 64
- Bonds: 3 × 3 × 4 × 4 = 144 (each direction has (4-1)×4×4 = 48 bonds × 3 = 144)
- Coordination: 6 (interior), 4 (face), 2 (edge corner... wait, min 3)

---

## Scan design: two tetrahedral scans

The 4×4×4 cube has two natural sets of non-coplanar corners:

**Scan 1 (tetrahedral set A)**: sites at corners (0,0,0), (3,3,0), (0,3,3), (3,0,3)
  → maximum angular separation in 3D
  
**Scan 2 (tetrahedral set B)**: sites at corners (3,0,0), (0,3,0), (0,0,3), (3,3,3)
  → complementary tetrahedral set

Together: 8 centers, 2 scans, covering the cube from all tetrahedral directions.
Every interior bond is within d_max=2 hops of at least one center in at least one scan.

**Key advantage over 2D**: the two tetrahedral scans approach bonds from non-coplanar
directions — maximum angular diversity impossible in 2D.

---

## Code plan

### 1. New grid class: `FlatCubic4x4x4Grid`
- 64 sites, 144 bonds, 6 color classes (or 3)
- Site indexing: s = z*16 + y*4 + x + 1 (1-indexed)
- Bond PTMs: same Heisenberg exp(-iθ(XX+YY+ZZ)) for all directions
- Build: `make PHYS_DIM=4 main_pauli_sweep_cubic4x4x4`

### 2. PTM data
- Build with `scripts/build_ptm.py --graph data/graphs/cubic_4x4x4.json`
- Need: ideal kick PTM + noisy PTMs for 3 gamma levels

### 3. Graph JSON
- Create `data/graphs/cubic_4x4x4.json`
- 64 sites, 144 bonds, uniform J=1 couplings

### 4. Scan plan JSON
- `data/scans/cubic_4x4x4_scan_2gpu.json`
- 2 scans, tetrahedral centers A and B

### 5. Detection scripts
- `bfs_detect.py` and `detect_delta_logT.py` are graph-agnostic — should work
- `compute_gram.py` — graph-agnostic
- Need new visualization (3D → 2D projection for figures)

---

## VRAM estimate

Currently 7×7 (49 sites, 84 bonds, χ=50): ~3.5GB per state.
Workspace scales as O(χ³ × n_bonds × n_colors).

4×4×4 (64 sites, 144 bonds): ~1.7× more bonds.
If χ can be kept at χ=30-40 (wavefront is small at 3-5 cycles):
Estimate ~2-4GB per state → fits on MI100 32GB.

**Key**: at 3-5 cycles, the wavefront covers only ~2 hops → small active region
→ χ saturates slowly → 3D should be tractable at small cycle count.

---

## Experiment plan

1. Build cubic_4x4x4 grid class and binary
2. Run lightcone sweep: measure v_hop in 3D
3. Determine d_max and cycle budget
4. Generate 10 single-bond trials
5. Run bfs_detect and detect_delta_logT
6. Measure detection rate — expect 10/10 or close
7. Visualize: 3-panel (z-slices) or isometric projection

---

## Multi-GPU simulation (future)

See separate note: `future_multigpu_simulation.md`

---

## First experiments (2026-11-10)

### What we built
- `include/cubic_3x3x3.hpp` and `cubic_4x4x4.hpp`: grid classes
- PTM data files for both
- `data/graphs/cubic_3x3x3.json` and `cubic_4x4x4.json`
- Makefile targets with `-DMAX_DEGREE=6`

### Key findings

**χ^6 is brutal**: Interior degree-6 sites have tensors of size χ^6×D.
- χ=20, 3×3×3: workspace 4GB → fits but step 3 gate_err=0.11
- χ=50, 3×3×3: OOM
- χ=50, 4×4×4: OOM

**Boundary reflection at step 2 for 3×3×3**: 
The 3×3×3 lattice is too small — wavefront from center (1,1,1) reaches
all 6 boundary walls at cycle 2 and reflects. The interference is physical,
not a BP artifact. BP=100/0.67 at step 2 is both BP loops AND physical
reflection compounding.

**Only step 1 is clean for 3×3×3**: bp=4, gate_err=0 at step 1.
The wavefront correctly shows asymmetry between z-neighbors due to
color-class ordering (colors 4,5 fire last in cycle 1).

**4×4×4 step 1**: also clean (bp=5, gate_err=0 at χ=20, center=(1,1,1)).
One more hop before boundary → step 2 may still be usable.

**Practical operating point**: χ=20, 1-2 cycles, single interior center.
Step 1 gives converged physics in 3D — sufficient for detection if the
broken bond is 1 hop from the center.

### This is exploratory
- 5×5×5 would be symmetric but doesn't fit in one GPU at any useful χ
- χ^6 fundamentally limits 3D — for serious 3D work need multi-GPU or
  lighter tensor network (e.g., tree TN or MPS along one axis)
- The key question to explore: can a single broken bond be detected
  from step-1 data alone in 3D?

### Next steps
1. Visualize 3D fields (layer-by-layer: z=0,1,2 panels)
2. Inject a single broken bond, run with χ=20, 2 cycles
3. Check if δlnT score identifies the bond at step 1
4. If yes: 3D tomography works in principle, just limited by χ
