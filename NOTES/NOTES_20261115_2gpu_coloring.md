# 2-GPU Simulation: Coloring-Driven Sync Protocol (2026-11-15)

## The key insight (missing from Julia 2-GPU work)

The Trotter gate COLOR decomposition already encodes the dependency graph
between sites across the GPU boundary. The P2P sync points are NOT arbitrary —
they are dictated by the color ordering.

This was the missing piece when working on 2-GPU simulation with Julia:
the sync protocol must follow the color schedule, not the cycle boundary.

---

## 3×3 example (split at row 1)

```
Row 0:  [0][1][2]   ← GPU 0 owns
Row 1:  [3][4][5]   ← GPU 0 owns (boundary row)
Row 2:  [6][7][8]   ← GPU 1 owns
```

**Color decomposition** (2D rectangular, 4 colors):
- Color H_even: (0,1), (3,4), (6,7)   — horizontal even bonds
- Color H_odd:  (1,2), (4,5), (7,8)   — horizontal odd bonds
- Color V_even: (0,3), (1,4), (2,5)   — vertical even bonds  ← BOUNDARY
- Color V_odd:  (3,6), (4,7), (5,8)   — vertical odd bonds   ← BOUNDARY

**Implicit dependency**: gate (0,3) updates site 3 → gate (3,6) needs updated site 3.
These are in DIFFERENT colors by construction — the coloring already resolves it.

---

## Sync protocol per Trotter step

```
GPU 0: Color H_even (rows 0-1)         GPU 1: Color H_even (row 2)
       [parallel, no sync needed]              [parallel, no sync needed]

GPU 0: Color H_odd  (rows 0-1)         GPU 1: Color H_odd  (row 2)
       [parallel, no sync needed]              [parallel, no sync needed]

GPU 0: Color V_even (0,3),(1,4),(2,5)  GPU 1: [wait]
       → site 3,4,5 updated on GPU 0
       ↓ SYNC 1: P2P row 1 → GPU 1
GPU 0: [continue]                       GPU 1: Color V_odd (3,6),(4,7),(5,8)
                                               → site 3,4,5 AND 6,7,8 updated on GPU 1
                                        ↓ SYNC 2: P2P row 1 → GPU 0
GPU 0: next color uses updated row 1   GPU 1: [continue]
```

**Two P2P syncs per Trotter step** at the boundary-crossing colors.
Each sync: 32 site tensors × χ^degree × 4 bytes = tiny at χ=40.

---

## Ownership rule

- GPU 0 **owns** row 1 (boundary row) — it is the authoritative writer
- GPU 1 has a **ghost copy** of row 1 — read-only during most colors
- After V_odd completes on GPU 1, row 1 is written back to GPU 0 (sync 2)
- Ownership is logical, not physical — row 1 must be valid on both GPUs at the right moments

---

## BP sync (once per cycle, after all Trotter colors)

After all 4 colors complete:
1. GPU 0 runs local BP on rows 0-1
2. GPU 1 runs local BP on rows 1-2
3. Exchange boundary BP messages (row 1↔2 bonds, χ-vector per bond direction)
4. One additional BP pass on each GPU to propagate boundary messages
5. Done — global BP convergence achieved

**BP sync cost**: 32 bonds × 2 directions × χ × 4 bytes = 32 × 2 × 40 × 4 = 10KB at χ=40.
Completely negligible.

---

## Total sync cost (3 cycles, 2 vertical color pairs per step)

- Trotter syncs: 3 cycles × 2 P2P/step × 32 sites × χ^4 × 4B
  = 6 × 32 × 2,560,000 × 4 ≈ 1.9 GB total transfer at χ=40
  (spread over 6 P2P calls — each ~320MB, fast on NVLink)
- BP syncs: 3 × 10KB = negligible

**NVLink bandwidth ~300GB/s → each Trotter sync ≈ 1ms. Negligible.**

---

## Why this was missing in Julia

The Julia 2-GPU split likely used a cycle-level sync (copy full state at end of
each cycle) rather than a color-level sync (copy boundary row after each
boundary-crossing color). The cycle-level approach is correct but:
1. Requires storing full state on both GPUs (2× memory)
2. Loses the color-ordering guarantee (must wait for full cycle)

The color-level sync:
1. Only syncs the boundary row (32 sites, not full grid)
2. GPU 1 can start its colors as soon as GPU 0 finishes the relevant color
3. Parallelism is preserved within each color — only boundary colors need sync

---

## Implementation plan (for 32x32 on 2 GPUs)

1. Build 32×32 grid class (or use two 16×32 grids with shared row)
2. Allocate separate GPU state on each device (HIP multi-device)
3. Per color: check if any bond in this color crosses the GPU boundary
   - If yes: finish color on owning GPU → hipMemcpyPeerAsync → continue
   - If no: both GPUs run in parallel
4. After all colors (end of Trotter step): exchange boundary BP messages
5. Run detection: each GPU outputs PAULI_NL for its half, merge for detection

**Binary**: `main_pauli_sweep_2gpu_rect32x32`
**First test**: 8×32 split as 2×(8×32) on GPU 0 and GPU 1

---

## Connection to tomography

For detection (3 cycles), the sync cost is minimal. The broken bond either:
- Falls entirely within one GPU's half → detected locally, no cross-GPU signal needed
- Falls on the boundary row → both GPUs see it, strongest signal from owning GPU

The period-6 quincunx centers can be assigned per GPU half — same design,
just each GPU runs its local centers with v_hop. No cross-GPU seismic needed.
