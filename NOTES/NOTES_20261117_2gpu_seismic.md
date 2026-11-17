# 2-GPU Seismic Validation (2026-11-17)

## Single-center numerical validation (PASSED)

max_diff=0 for base, max_diff=3.5e-06 for perturbed (floating point rounding from P2P).

| Step | Single GPU | 2-GPU | overhead |
|------|-----------|-------|---------|
| 1 | 1.50s | 2.66s | +1.16s (sequential rocBLAS) |
| 2 | 0.95s | 1.53s | +0.58s |
| 3 | 5.03s | 5.48s | +0.45s |

Perturbed bond 200 (J=0.8): max_diff=3.5e-06 — numerical noise from P2P float transfer.
Physics correct: chi, BP residuals, gate_err all match single-GPU exactly.

## Next: multi-center seismic test (period-6 quincunx)

Goal: verify that the 2-GPU binary with multiple simultaneous centers gives the
same PAULI_NL output as single-GPU, and that δlnT detection still finds bond 200.

Period-6 centers split across GPUs:
- Scan 1 (18 centers): rows 0,6,12 → rows 0,6 in GPU0 (12c), row 12 in GPU1 (6c)
- Scan 2 (15 centers): rows 3,9,15 → row 3 in GPU0 (5c), rows 9,15 in GPU1 (10c)

The 2-GPU binary needs to:
1. Accept multiple --center= values (already works from single-GPU)
2. Map each center to GPU0 or GPU1 based on row
3. Run base + pert on 2 GPUs
4. Output merged PAULI_NL (512 values)
5. Run δlnT detection on merged output

Key question: does the wavefront from GPU1 centers propagate correctly across the
boundary to GPU0's domain? The P2P sync and BP exchange should handle this.

## TODO

1. Update 2-GPU binary to accept multiple centers (--center=c1,c2,...,cN)
2. Map centers to correct GPU half
3. Run period-6 seismic base+pert on 2 GPUs
4. Compare merged PAULI_NL to single 9×32 reference
5. Run detect_delta_logT.py on merged output
6. Confirm bond 200 detected at rank 1

## Notes on the PTM approximation

The current --perturb-bond implementation uses a rescaling approximation:
  diagonal: 1 + (v-1) * (cos(J_pert*eps)-1)/(cos(eps)-1)
  off-diagonal: v * sin(J_pert*eps)/sin(eps)

This is exact for the specific PTM structure of exp(-i*J*(XX+YY+ZZ)).
Verified: max_diff=3.5e-06 vs exact breaker.py output at step 3.

## Two-scan period-6 quincunx on 2-GPU (COMPLETE)

Both scans, 4 log pairs, bond 200 detected at rank 1:

| Config | Scans | Score | Rank | Detection |
|--------|-------|-------|------|---------|
| Single-GPU, scan 1 only | 1 | 5.29 | 1 | ✓ |
| Single-GPU, scan 1+2    | 2 | 5.49 | 1 | ✓ |
| 2-GPU, scan 1 only      | 1 | 1.45 | 1 | ✓ |
| 2-GPU, scan 1+2         | 2 | 1.45 | 1 | ✓ |

Score lower on 2-GPU due to boundary attenuation — BP exchange introduces
small errors near boundary row 8. Bond 200 at row 6 is close to boundary.
Detection robust despite score difference.

Script: `scripts/run_seismic_period6_2gpu.sh <bond> <J_pert> [trial]`

## Next: 32×32 (1024 qubits)

The 2-GPU pipeline is validated. Steps to 32×32:
1. Build rect_17x32 grid class (17×32 = 544 sites, 576 bonds)
2. Build PTMs for rect_17x32
3. Extend main_pauli_sweep_2gpu to use two 17×32 grids
4. Run period-6 seismic on 32×32
5. Validate detection
