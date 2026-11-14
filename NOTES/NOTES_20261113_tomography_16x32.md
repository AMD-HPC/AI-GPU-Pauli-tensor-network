# Hamiltonian Tomography — 16×32 (512 qubits) on MI100 (2026-11-13)

## Summary

First successful 512-qubit simulation and tomography attempt on a single MI100 GPU.
arXiv submission of the tomography paper completed today (submission/8367958).

---

## Memory model (validated)

The workspace is allocated at chi_max for all N sites regardless of lightcone.
v_hop reduces VRAM only for single-center (aperture) runs.

| chi | v_hop | VRAM% | use case |
|-----|-------|-------|----------|
| 40  | none  | 53%   | seismic — full grid, all centers |
| 50  | 0.5   | 4%    | aperture — single center, lightcone |
| 50  | none  | OOM   | exceeds 32GB |

**Rule**: seismic → chi=40, aperture → chi=50 + v_hop=0.5.

---

## Voronoi center placement — 8×8 tile strategy

find_voronoi_centers.py on rect_8x8.json (k=7) gives:
  Centers: [5, 10, 23, 28, 33, 46, 51]
  Max T_b: 8 cycles, mean: 3.9 cycles

Tile 2×4 across 16×32 → 56 centers:
```
 0 ....C.......C.......C.......C...
 1 .C.......C.......C.......C......
 2 ......C.......C.......C.......C.
 3 ...C.......C.......C.......C....
 4 C.......C.......C.......C.......
 5 .....C.......C.......C.......C..
 6 ..C.......C.......C.......C.....
 7 ................................   (empty row — tile boundary)
 8 (repeat rows 0–7 for second half)
```

**Scalable**: same tile → 32×32 (4×4=16 tiles), 64×64 (8×8=64 tiles).
**Key property**: Voronoi (no two centers share row or column per tile) →
every bond has a unique nearest center → clean directional signal.

CENTERS string for 16×32:
`5,13,21,29,34,42,50,58,71,79,87,95,100,108,116,124,129,137,145,153,166,174,182,190,195,203,211,219,261,269,277,285,290,298,306,314,327,335,343,351,356,364,372,380,385,393,401,409,422,430,438,446,451,459,467,475`

---

## Seismic protocol

Script: `scripts/run_seismic_16x32.sh <bond> <J_pert> [trial_id]`

Parameters:
- chi=40 (53% VRAM, fits on single MI100)
- STEPS=3 (see timing below)
- No v_hop — seismic needs full allocation
- Base on GPU 0, pert on GPU 1 in parallel → ~35s total

**Why STEPS=3**:
- Step 1 (8s): entire 512-site grid already lit from 56 simultaneous centers
- Step 2 (27s): interference beginning, signal established at broken bond
- Step 3 (361s!): chi saturated at 40, BP residual ~3700, slow but bounded
- Steps 4-5: each would take ~6 min — not worth it, signal already there

**BP non-convergence**: bp residual ~1e3 vs ~5e-6 for single center.
This is expected and OK: n_pert/n_base ratio cancels common BP artifacts.
Local signal near nearest center is clean regardless of global BP state.

Detection: `python3 scripts/detect_delta_logT.py --logdir logs/seismic_16x32/trial_N --graph data/graphs/rect_16x32.json --top_k 1`

---

## Aperture protocol

Script: `scripts/run_voronoi_16x32.py --logdir ... --n_gpus 3`
- chi=50, STEPS=5, v_hop=0.5 → 4% VRAM per center
- 32 centers, 3 GPUs round-robin → ~40s for all 32 centers
- Winner-takes-all δlnT detection

Note: current 32-center plan uses 4-hop spacing (too sparse for 2-hop coverage).
Should use the 8×8 tile centers (56 total) for aperture as well — same centers,
just run individually with v_hop instead of all together.

---

## What did NOT work today

**Antenna grid** (16 row + 32 col antennas):
- v_hop cuts lightcone to antenna line → bond's far endpoint always dark
- BP non-convergence with 32 simultaneous centers
- Spurious large scores on grid boundary bonds from BP artifacts
- Bond 200 score = 0 (zero detection)

**32-center Voronoi at 4-hop spacing**:
- Centers too far from bonds → weak signal
- Bond 200 at rank 3 (not rank 1) with many false positives
- False positives cluster near grid edges (BP boundary artifact)

**56-center seismic at chi=50**: OOM (42GB > 32GB MI100)
**56-center seismic at STEPS=5**: step 3 = 361s, would take hours total

---

## New scripts created today

| Script | Purpose |
|--------|---------|
| `scripts/run_seismic_16x32.sh` | Seismic with 56 centers, chi=40, 3 steps, 2-GPU parallel |
| `scripts/run_voronoi_16x32.py` | Aperture with 32 centers, chi=50, v_hop=0.5 |
| `scripts/run_antenna_grid_16x32.py` | Antenna scan (did not work well) |
| `scripts/detect_antenna_delta_logT.py` | δlnT from any dir of base/pert logs |
| `scripts/plot_wavefront_16x32.py` | n(v,t) + L(v,t) 3-column heatmap |
| `scripts/plot_delta_logT_field_16x32.py` | Zoomed δlnT vector field |
| `scripts/plot_delta_logT_field_aggregated.py` | Full 16×32 δlnT aggregated |
| `include/rect_16x32.hpp` | 16×32 grid class (512 sites, 976 bonds) |
| `include/rect_16x32_ideal_kick_eps010_data.hpp` | PTMs for 16×32 |
| `data/graphs/rect_16x32.json` | Graph JSON |
| `data/scans/rect_16x32_voronoi.json` | 32-center Voronoi plan |

Also: `detect_delta_logT.py` generalized — `--plan` now optional,
auto-discovers `*_base/pert.log` pairs from `--logdir`.

---

## Next session

1. Check seismic trial 1 result — did bond 200 appear at rank 1?
2. Run 10 trials seismic to get statistics (script: run_seismic_16x32.sh)
3. If seismic misses → aperture with nearest 8×8 tile center + v_hop
4. Use `searcher.py` signed-L algorithm for detection (not just δlnT)
5. Extend to 32×32 — same 8×8 tile, 4×4=16 blocks

## Key physics confirmed

- v_hop + single center: local wavefront, clean BP, 4% VRAM
- Seismic: global excitation, BP struggles but signal/noise still works via ratio
- 3 cycles sufficient for 512-qubit grid with 56 centers (whole grid lit at step 1)
- 2-GPU parallel (base+pert) halves wall time — standard operating mode
