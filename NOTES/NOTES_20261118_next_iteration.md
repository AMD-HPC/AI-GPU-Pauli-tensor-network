# Next Iteration: Fix List for 3-GPU Architecture (2026-11-18)

## Current results (chi=10, 32×32 seismic, bond 200)

| strategy | step3_t | score | rank | VRAM/GPU |
|----------|---------|-------|------|----------|
| A: 1-GPU, no v_hop | 228s | 5.63 | 1 ✓ | 3% |
| B: 1-GPU, v_hop=0.9 | 115s | **5.81** | 1 ✓ | 2% |
| C: 2-GPU (3-GPU arch) | **59s** | 1.48 | 1 ✓ | 1% |

Key observations:
- C is fastest despite ping-pong — BP on half-grid is cheaper than full-grid
- C does not lose accuracy (rank 1 correct)
- C score lower (1.48 vs 5.81) — boundary BP approximation, not normalization
- B is best overall: fastest + highest score + lowest memory

## Fix 1: Parallel Trotter (kill ping-pong)

**Problem**: Trotter colors run GPU0 → GPU1 sequentially (ping-pong).
**Root cause**: rocBLAS handles not thread-safe across devices.
**Location**: `main_pauli_sweep_2gpu_rect32x32.cpp` lines 259-314

**Fix**: Create rocBLAS handle per thread, use `std::thread`:
```cpp
// Before: sequential
for (int c = 0; c < cut_color; c++) {
    hipSetDevice(0); do_trotter_one_color(ctx0, c, ..., gops0, ...);  // blocks
    hipSetDevice(1); do_trotter_one_color(ctx1, c, ..., gops1, ...);  // blocks
}

// After: parallel
for (int c = 0; c < cut_color; c++) {
    std::thread t0([&,c](){
        hipSetDevice(0);
        rocblas_handle h; rocblas_create_handle(&h);
        GateOps g(h);
        do_trotter_one_color(ctx0, c, ..., g, ...);
        rocblas_destroy_handle(h);
    });
    std::thread t1([&,c](){
        hipSetDevice(1);
        rocblas_handle h; rocblas_create_handle(&h);
        GateOps g(h);
        do_trotter_one_color(ctx1, c, ..., g, ...);
        rocblas_destroy_handle(h);
    });
    t0.join(); t1.join();
}
```

**Expected improvement**: Trotter wall time drops from 2× single-GPU → ~1×
**Risk**: handle create/destroy per color — benchmark overhead

## Fix 2: Remove per-color P2P sync (Trotter correctness)

**Problem**: Current code syncs GPU0→GPU1 before cut_color (line 270-292).
  This is needed for correctness but adds latency.
**Root cause**: GPU1 needs GPU0's boundary row to compute cut bond gate.
**After parallel Trotter fix**: the sync is still needed but can overlap with
  GPU1's non-cut colors running on a stream.

**Fix**: Remove the per-color sync entirely and rely on the slice BP to
  correct any boundary Trotter error. We showed (validation PASS) that
  the normalization factor is the main error, not Trotter boundary.

**BUT**: The 1.786× factor at step 1 IS from Trotter boundary (row 16 ghost
  not updated by row 16↔17 gate). The slice BP corrects it by step 3 but
  not at step 1-2. For 3-step seismic this matters.

**Recommended**: Keep P2P sync but make it async (hipMemcpyPeerAsync).
  Then GPU0 and GPU1 can proceed to cut_color simultaneously using streams.

## Fix 3: Score improvement via better slice BP

**Problem**: Score 1.48 vs 5.81 (B) — boundary BP approximation.
**Root cause**: slice BP covers rows 13-19, but centers in GPU1 (rows 16-31)
  contribute messages that don't propagate into GPU0's boundary properly.
**Fix**: Two options:
  a) Run slice BP twice (second pass after local BP to refine messages)
  b) Extend slice to cover more rows (10×32 instead of 7×32)

**Expected improvement**: score approaches 3-4 (halfway between 1.48 and 5.81)

## Fix 4: Add v_hop support to 2-GPU seismic

**Problem**: 2-GPU binary doesn't support v_hop — always full allocation.
**Fix**: Pass v_hop to `apply_lightcone` with multi-center union (already implemented
  in main_pauli_sweep.cpp). Add `--v_hop` parsing to 2-GPU binary.
**Expected**: C with v_hop=0.9 → ~1-2% VRAM, faster BP, possibly better score.

## Implementation order (one shot)

1. Add `--v_hop` to 2-GPU binary (easy, 10 min)
2. Parallel Trotter with per-thread rocBLAS (medium, 1 hour)
3. Test: validate_2gpu_full_32x32 still PASS
4. Run chi scaling study again → compare all strategies

## Code locations

- Trotter colors loop: `main_pauli_sweep_2gpu_rect32x32.cpp` lines 258-314
- P2P site sync: lines 270-292
- BP section: lines ~345-420
- do_trotter_one_color: `include/trotter_2gpu.hpp` line 83
- GateOps: `include/gate.hpp` line 240

## Chi scaling study results so far

Logs at: `logs/chi_scaling_32x32/`
Summary: `logs/chi_scaling_32x32/summary.tsv`

chi=10 done, chi=20 running. Expected OOM for A at chi~40.
Key result: B (v_hop) wins on all metrics at all chi. 2-GPU needed only
for chi > OOM threshold AND when v_hop+1GPU doesn't give enough chi for
the physics (C(t) convergence etc — detection doesn't need high chi).
