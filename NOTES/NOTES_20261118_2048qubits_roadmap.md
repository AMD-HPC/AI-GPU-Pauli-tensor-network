# Roadmap to 2048 Qubits (and beyond) (2026-11-18)

## Current state: 1024 qubits on 3 GPUs (validated)

- 32×32 lattice, 2 GPUs (Trotter) + 1 GPU (slice BP)
- Validated: max_diff < 6e-05 at chi=10
- Parallel Trotter: GPU0 and GPU1 fully independent (std::thread, per-thread rocBLAS)
- Parallel BP: GPU0 BP || GPU1 BP (std::thread, per-thread Measures)
- Sequential: psi copy → GPU2 slice BP → inject → local BP

## Architecture for 2048 qubits (64×32 lattice)

### Option A: 4 GPUs (2 split + 1 slice BP + 1 coordinator)

Split 64×32 horizontally at row 32:
- GPU 0: rows 0-32  (33×32 = 1056 sites)
- GPU 1: rows 32-63 (33×32 = 1056 sites, row 0 = ghost)
- GPU 2: 7×32 slice BP (rows 29-35, 224 sites) — SAME slice as before!
- GPU 3: global coordinator (future, optional)

**Key**: the 7×32 slice BP is ALWAYS 224 sites regardless of total grid size.
Memory and time for GPU2 is CONSTANT as grid grows.

### Option B: 2×(2 GPU) pipeline

Split 64×32 into 4 quadrants of 16×32 each:
- Machine 1: GPU0 rows 0-15, GPU1 rows 16-31 (using existing 2-GPU 32×32 code!)
- Machine 2: GPU0 rows 32-47, GPU1 rows 48-63

Merge outputs on CPU. Same 7×32 slice at each machine boundary.
Communication between machines: only boundary messages (~10KB) via NVLink or InfiniBand.

## Implementation steps for 2048 qubits

### Step 1: Build 33×32 grid class

```bash
# Generate graph
python3 -c "NX,NY=32,33; ..." > data/graphs/rect_33x32.json
python3 scripts/build_ptm.py IDEAL_KICK --graph=data/graphs/rect_33x32.json \
    --epsilon=0.10 --prefix=RECT_33X32 --out=include/rect_33x32_ideal_kick_eps010_data.hpp
# Generate grid class (copy rect_17x32.hpp, change NY=33)
cat include/rect_17x32.hpp | sed 's/17/33/g; ...' > include/rect_33x32.hpp
```

### Step 2: Build 64×32 single-GPU reference

```bash
# Same as rect_32x32 but NY=64
python3 -c "NX,NY=32,64; ..." > data/graphs/rect_64x32.json
python3 scripts/build_ptm.py ... --prefix=RECT_64X32 ...
cat include/rect_32x32.hpp | sed 's/32/64/g; ...' > include/rect_64x32.hpp
# Add to main_pauli_sweep.cpp as GRAPH_rect_64x32
make PHYS_DIM=4 main_pauli_sweep_rect64x32
```

### Step 3: Build 2-GPU 64×32 binary

Copy `main_pauli_sweep_2gpu_rect32x32.cpp` → `main_pauli_sweep_2gpu_rect64x32.cpp`

Key changes (search-and-replace + manual):
- `RECT_17X32` → `RECT_33X32` (half-grid class)
- `RECT_32X32` → `RECT_64X32` (full reference — not needed if uniform J)
- Split row: 16 → 32 (cut at row 32)
- `cut_site_top = 32 * 32 = 1024` (first site of row 32 in grid0)
- `cut_bond_id` = first vertical bond at row 32↔33 in 33×32 grid
  - 33×32 horizontal bonds: 33*(32-1)=1023
  - Vertical bonds start at 1023, row 31↔32: `1023 + 31*32 = 2015`
  - Wait — cut is at row 32 for GPU1, so cut_bond_id = first vertical bond in grid1 = 1023
- Boundary offsets:
  - `bond_offset0 = 1023 + 31 * 32` (row 31↔32 vertical in grid0)
  - `bond_offset1 = 1023` (row 0↔1 vertical in grid1)
- Slice BP: 7×32 slice covers rows 29-35 → same rect_7x32.hpp!
  - Slice rows 0-3 from GPU0 (rows 29-32), rows 4-6 from GPU1 (rows 1-3)
  - `(row + 29) * 32` for GPU0 copy, same as `(row + 13) * 32` → `(row + 29) * 32`
- Normalization: `33.0f / 64.0f` (instead of 17/32)
- PAULI_NL: 1024 sites (GPU0 rows 0-31) + 1024 (GPU1 rows 1-32) = 2048

### Step 4: Validation at chi=10

```bash
python3 scripts/validate_2gpu_full_64x32.py --chi 10 --steps 3 --center 497
# Expected: PASS, max_diff < 1e-4
```

### Step 5: Chi scaling study

```bash
bash scripts/run_chi_scaling_64x32.sh  # same 3 strategies A/B/C
# Expected:
# A OOM at chi~22 (1024 sites × chi^4)
# B survives to chi~30 (lightcone ~18% of 2048 sites)
# C survives to chi~40 (each half 1056 sites)
```

## Scaling laws

| Grid | Sites | A OOM | B max chi | C max chi | VRAM/GPU (C, chi=40) |
|------|-------|-------|-----------|-----------|----------------------|
| 16×32 | 512 | chi=50 | chi=50 | chi=50 | 53% |
| 32×32 | 1024 | chi=30 | chi=30 | chi=40 | 57% |
| 64×32 | 2048 | chi~22 | chi~26 | chi~35 | ~55% |
| 128×32 | 4096 | chi~16 | chi~20 | chi~28 | ~50% |

**Key insight**: C (2-GPU) VRAM per GPU stays roughly constant because
each half-grid grows proportionally to N/2. The 7×32 slice BP cost is O(1).

## Why the slice BP cost is constant

For ANY grid of size NX×NY split at row NY/2:
- Slice size: 7×NX = 7×32 = 224 sites (independent of NY)
- Slice BP time: O(1) in NY — always the same 224-site problem
- P2P cost: 32 bonds × chi² = O(1) in NY

This is the key architecture insight: the inter-GPU communication
cost is O(lattice width), not O(lattice size). Scales to arbitrary depth.

## Connection to distributed BP architecture (notes from 2026-11-11)

See `NOTES/distributed_bp_architecture.md` for the general N-GPU case.
The current 2-GPU implementation is the simplest case of that architecture:
- 2 compute GPUs (Trotter)
- 1 boundary GPU (slice BP coordinator)
- O(width) communication per cycle

For 4+ GPUs: chain of slice BPs at each partition boundary.
Each additional GPU pair adds one more 7×32 slice coordinator.

## Note on parallel execution

After implementing parallel Trotter (std::thread) and parallel BP (std::thread):
- Trotter: GPU0 || GPU1 — true parallel (2× speedup confirmed)
- BP step 4: GPU0 BP || GPU1 BP — true parallel (expected 2× speedup)
- Remaining sequential: psi copy → GPU2 slice BP → inject

The sequential slice BP is unavoidable (needs psi from both GPUs first).
But it's cheap (~18 sweeps on 224 sites) compared to main BP (100 sweeps × 544 sites).
