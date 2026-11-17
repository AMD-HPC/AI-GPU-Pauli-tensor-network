# 2-GPU 16×32 Split — Implementation (2026-11-16)

## What was built

### New files
- `include/trotter_2gpu.hpp` — TrotterColorCtx, do_trotter_one_color, do_trotter_2gpu
- `include/rect_9x32.hpp` — 9×32 grid class (288 sites, 535 bonds)
- `include/rect_9x32_ideal_kick_eps010_data.hpp` — PTMs for 9×32
- `data/graphs/rect_9x32.json` — graph JSON
- `examples/main_pauli_sweep_2gpu_rect16x32.cpp` — validation binary
- `NOTES/fig_2gpu_schedule.png` — schedule diagram
- `scripts/plot_2gpu_schedule.py` — diagram script

### Key design decisions
- `do_trotter` (trotter.hpp) **unchanged** — new functionality in separate header
- 9×32 = rows 0-8, row 8 = overlap row (owned by GPU 0, ghost on GPU 1)
- P2P sync: **unidirectional** GPU0→GPU1 before cut color (not bidirectional)
- PTMs: use 9×32 PTMs directly (uniform J=1, no remapping needed)
- Cut color = color 2 (vertical even, connects rows 0↔1 in grid1)

## Validation results

Step 1 matches single-GPU exactly:
- chi=4, bp=5/1.55e-06, gate_err=0 ✓
- PAULI_NL values identical to single-GPU ✓

## Known issues to fix next session

### 1. Ping-pong (performance)
2-GPU runs 2× SLOWER than single GPU — GPUs run sequentially, not in parallel.
Root cause: `hipMemcpy` inside `do_trotter_one_color` is synchronous.

Fix: add `hipStream_t` parameter to `do_trotter_one_color`, use `hipMemcpyAsync`.
Color loop becomes:
```cpp
hipSetDevice(0); do_trotter_one_color(ctx0, c, ..., stream0);  // async
hipSetDevice(1); do_trotter_one_color(ctx1, c, ..., stream1);  // async
hipStreamSynchronize(stream0);  // barrier before next color
hipStreamSynchronize(stream1);
```
Expected result: 2-GPU wall time ≈ single-GPU wall time.

### 2. Center mapping
GPU1 initialized with same center as GPU0 → wavefront appears twice in output.
Fix: map requested 16×32 center to GPU1 coordinates:
```cpp
// If center in rows 8-15: GPU1 center = (center_row - 8) * 32 + center_col
// If center in rows 0-7:  GPU1 center = -1 (no center, silent)
```

### 3. apply_binary_gate needs stream
For true async parallelism, `apply_binary_gate` in gate.hpp also needs
hipStream_t. Check how deep the stream threading goes before implementing.

## Schedule diagram

See `NOTES/fig_2gpu_schedule.png`:
- Horizontal colors: parallel (no sync)
- Before cut color: unidirectional P2P GPU0→GPU1
- Cut color: both GPUs compute simultaneously
- End of cycle: BP message exchange (bidirectional, 10KB)

## Next session checklist

1. Fix center mapping (easy, 10 min)
2. Add hipStream_t to do_trotter_one_color (medium)
3. Verify 2-GPU wall time ≈ single-GPU wall time
4. Run full validation: compare PAULI_NL output to single-GPU reference
5. Once validated: extend to 32×32 (two 17×32 grids, same protocol)
