# Scan Geometry — Why 3 Scans, Not 1 (2026-11-13)

## The key insight

The 7×7 used **3 separate scans** (not 1 combined seismic) for two reasons that are
easy to forget after context compaction:

### 1. Triangulation requires angular diversity

A broken bond creates a RED-BLUE signature that points FROM the nearest center TOWARD
the bond. If you fire all centers simultaneously, the wavefronts interfere and you lose
the directionality — you can detect the bond but cannot triangulate its location as
precisely.

With 3 separate scans, each scan gives you an independent directional measurement.
The broken bond appears RED in some scans, BLUE in others, depending on which side
the scan's nearest center is on. The intersection of the three measurements triangulates
the bond unambiguously.

### 2. The 7×7 scan geometry has NO symmetry

```
Scan 1 (corners+center):   Scan 2 (quadrant mid):   Scan 3 (edge mid):
  1.....1                    .......                   ...3...
  .......                    .2...2.                   .......
  .......                    .......                   .......
  ...1...                    .......                   3.....3
  .......                    .......                   .......
  .......                    .2...2.                   .......
  1.....1                    .......                   ...3...

Combined:
  1..3..1
  .2...2.
  .......
  3..1..3
  .......
  .2...2.
  1..3..1
```

Every bond in the 7×7 is seen from at least 2 different scan angles.
No two scans have the same center placement.
The pattern is a non-repeating, non-symmetric cover of the grid.

### 3. The periodic tiling curse for 16×32

The 16×32 single seismic uses an 8×8 Voronoi tile repeated 2×4 times:
```
 0 ....C.......C.......C.......C...
 1 .C.......C.......C.......C......
 ...
 7 ................................   ← blind row (tile boundary)
 8 (repeat rows 0-7)
 15 ................................  ← blind row (tile boundary)
```

Problems:
- **Symmetric**: every 8×8 block sees exactly the same center pattern → no angular diversity
- **Blind rows**: rows 7 and 15 (tile boundaries) have no nearby centers → bonds there missed
- **No triangulation**: a bond at (row 3, col 5) sees centers from the same relative positions
  in every tile → all wavefronts arrive from the same direction → cannot triangulate
- **Single combined wavefront**: 56 centers fired simultaneously → maximum interference →
  BP struggles, signal washed out by symmetry

### 4. The correct approach for 16×32

Tile the 7×7's 3-scan structure to 16×32:

**Scan 1** (corners+center of each 8×8 block): tile scan 1 centers × 8 blocks
**Scan 2** (quadrant midpoints of each 8×8 block): tile scan 2 centers × 8 blocks
**Scan 3** (edge midpoints of each 8×8 block): tile scan 3 centers × 8 blocks

Each scan fires ~5 × 8 = 40 centers (or ~4 × 8 = 32) simultaneously.
Run base+pert for each scan on 2 GPUs in parallel.
3 scans × 2 runs = 6 GPU runs total (~18 minutes at 6 min/run).

Then run BFS/δlnT aggregating across all 3 scan logs → triangulation restored.

### 5. Memory check for 3-scan approach

Each scan fires ~32-40 centers on 512 sites at chi=40.
Memory scales with sites × chi², not number of centers.
→ Same 53% VRAM as the current 56-center seismic.
→ No memory penalty for splitting into 3 scans.

## Summary rule (DO NOT FORGET)

**More centers in one scan ≠ better detection.**
**More scans with fewer centers = triangulation = better detection.**

The 7×7 achieved 10/10 single-bond with 3 scans of 4-5 centers each.
The 16×32 achieves 81% single-bond with 1 scan of 56 centers.
The gap is triangulation, not scale.

## TODO next session

1. Map 7×7 scan 1/2/3 center positions to 8×8 coordinates
2. Tile each scan independently across 16×32 (8 blocks)
3. Build `data/scans/rect_16x32_3scan.json` with 3 placements
4. Update `run_seismic_16x32.sh` to run 3 scans sequentially
5. Aggregate BFS/δlnT across all 3 scan logs
6. Expect: single-bond detection >> 81%

---

## Sparse quincunx + v_hop: the right operating point

### The key insight

v_hop savings depend on how much of the grid is dark (outside any lightcone).
- **Dense tiling (56 centers)**: lightcone union covers whole grid → v_hop saves nothing
- **Sparse quincunx (16+12+12=40 centers, spacing ~8 hops)**: large dark regions between scan lines → v_hop saves significantly

### Memory estimate for sparse quincunx + v_hop

Scan 1 (16 centers, spacing ~8 hops, v_hop=0.5, 3 steps):
- Active radius per center: 3 × 0.5 = 1.5 hops → ~7 sites per center
- Active sites: 16 × 7 ≈ 112 / 512 = 22% of grid
- VRAM: 53% × 0.22 ≈ **~12%** (vs 53% without v_hop)

At 12% VRAM → can push to chi=70-100 on single GPU.
Or run all 3 scans simultaneously on 3 GPUs with v_hop → wall time ~6 min total.

### Operational modes comparison

| Strategy | Centers/scan | v_hop | VRAM | chi | wall time | triangulation |
|----------|-------------|-------|------|-----|-----------|--------------|
| Dense single scan | 56 | no | 53% | 40 | ~6 min | none |
| Dense 3-scan tiled | 40 | no | 53% | 40 | ~18 min | weak (boundary doubling) |
| Sparse quincunx 3-scan | 16+12+12 | no | 53% | 40 | ~18 min | strong |
| Sparse quincunx 3-scan | 16+12+12 | 0.5 | ~12% | 70+ | ~6 min | strong |

**Best operating point**: sparse quincunx + v_hop → strong triangulation + higher chi + fast.

### Why sparsity enables v_hop

The lightcone allocation (v_hop) gives memory savings proportional to the fraction
of the grid that is dark. Dense center placement lights up the whole grid → v_hop
useless. Sparse placement leaves large dark regions → v_hop highly effective.

Sparsity and v_hop are complementary: sparsity creates the dark regions,
v_hop exploits them. The quincunx pattern maximizes dark area while maintaining
angular diversity for triangulation.

### TODO

1. Test sparse quincunx (current run) for detection quality on bond 200
2. If detection works → rerun with v_hop=0.5 and measure VRAM
3. If VRAM ~ 12% → push chi to 70, run all 3 scans on 3 GPUs in parallel
4. Run full 63-trial scaling study with sparse quincunx + v_hop + chi=70
5. Compare detection rates vs dense single-scan (81% baseline)

---

## v_hop: aperture only (by design)

v_hop works by BFS from each center, allocating chi=chi_max within radius
v_hop x steps and chi=1 outside. For a single center this gives a small local ball.

For seismic (multiple centers), the union of all balls covers the whole grid as
soon as centers are denser than 2 x v_hop x steps apart. At that point v_hop
saves nothing.

For period-6 (centers 6 hops apart), need v_hop x steps >= 3 to reach midpoint.
At v_hop=0.9, steps=3: radius=2.7 -- just short. Whole grid stays dark between centers.

Rule: v_hop is aperture-only. Seismic always runs without v_hop (full allocation).
To use v_hop with seismic would require a new allocation rule based on
inter-center distance, not per-center BFS. The code was designed for aperture first.

---

## Multi-center lightcone union (2026-11-14)

### Implementation

New overload of `apply_lightcone` in `include/grid.hpp`:

```cpp
void apply_lightcone(const std::vector<int>& centers, int n_steps, float v_hop)
```

Algorithm:
1. Deactivate all bonds: `chi_alloc = 1`
2. For each center: BFS, activate bonds where either endpoint is in cone
3. Union is monotonic — bonds never deactivated once activated

Wired in `examples/main_pauli_sweep.cpp`: when `centers.size() > 1`, calls
multi-center overload instead of single-center.

### Results (period-6 quincunx, 18 centers, v_hop=0.9, steps=3)

| Metric | No v_hop | v_hop=0.9 (single) | v_hop=0.9 (multi-center union) |
|--------|---------|-------------------|-------------------------------|
| VRAM   | 53%     | 3% (too sparse)   | **20%** |
| Step 1 | 8s      | 0.3s (no signal)  | **1.6s** |
| Step 3 | 81s     | <1s (no signal)   | **63s** |
| BP residual step 3 | 400 | — | **135** |
| Detection | ✓ | ✗ | TBD |

### Key insight

chi saturates at step 3 even with 20% VRAM. This means the chi budget is
not fully utilized — we can either:
1. **Increase chi** (from 40 to ~60) at same VRAM → better accuracy
2. **More centers** at same chi → better coverage
3. **More steps** (4-5) → signal has more time to propagate

The multi-center union makes v_hop viable for seismic when centers are sparse
enough that individual lightcones don't cover the whole grid. This is the
missing piece that makes sparse quincunx + v_hop work together.

### v_hop rule (updated)

v_hop now works for BOTH aperture AND seismic, provided:
- **Aperture**: single center, always works
- **Seismic**: works when centers are sparse enough that lightcone union
  covers < 100% of grid. Period-6 spacing (6 hops) with v_hop=0.9, steps=3
  gives 20% VRAM coverage.
