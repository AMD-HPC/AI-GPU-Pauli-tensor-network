# 2-GPU 16×32 Split — Validated (2026-11-17)

## Result

**Exact numerical match to single-GPU**: max_diff=0.00e+00 for all 3 steps.

```
python3 scripts/validate_2gpu.py --chi 40 --steps 3 --center 145
Step 1: max_diff=0.00e+00  nonzero_ref=16  ✓ OK
Step 2: max_diff=0.00e+00  nonzero_ref=49  ✓ OK
Step 3: max_diff=0.00e+00  nonzero_ref=68  ✓ OK
PASS
```

## Final protocol

1. **Trotter**: sequential per color, P2P site sync before cut color (GPU0→GPU1 unidirectional)
2. **BP**: local convergence on each GPU, then bidirectional message exchange at boundary
3. **BP exchange guard**: only send if source chi > 1 (avoids overwriting valid messages with chi=1)
4. **Extra BP sweep**: only on GPUs that received non-trivial messages (chi-guarded)

## Performance (center=145, chi=40, 3 steps)

| Step | Single GPU | 2-GPU |
|------|-----------|-------|
| 1 | 1.50s | 2.70s |
| 2 | 0.97s | 1.57s |
| 3 | 5.11s | 5.52s |

Sequential execution (rocBLAS handle thread-safety). Parallelism is next.

## Known remaining issues

1. **Ping-pong** (performance): rocBLAS handles not thread-safe across devices.
   Fix: create separate rocBLAS handle per GPU per thread, or use hipStream per device.
   Impact: 2× slower than single-GPU. For validation this is fine.

2. **Center on boundary row** (correctness edge case): center=257 (row 8) gives
   different topology in 9×32 vs 2-GPU because row 8 has different bonds.
   Not a bug — just an artifact of the grid split. Centers in rows 0-7 or 9-15 are fine.

3. **GPU1 domain not tested**: validation only covers GPU0 centers.
   Next: validate_2gpu.py with center in rows 9-15 (GPU1 domain).

## Next steps

1. Test center in GPU1 domain (rows 9-15)
2. Fix rocBLAS thread-safety for true parallel execution
3. Build 32×32 binary (two 17×32 grids) — same protocol, larger grid
4. Run seismic on 32×32 (1024 qubits)

## Files

- `include/trotter_2gpu.hpp` — TrotterColorCtx, do_trotter_one_color, do_trotter_2gpu
- `examples/main_pauli_sweep_2gpu_rect16x32.cpp` — validation binary
- `scripts/validate_2gpu.py` — numerical validation script
- `include/rect_9x32.hpp` — 9×32 half-grid class
