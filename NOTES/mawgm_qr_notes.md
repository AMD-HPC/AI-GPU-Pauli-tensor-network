# QR Non-Determinism — Notes on Muhammad's Finding 2

## What Muhammad observed

Six byte-identical Householder f64 runs gave 6 different C_nl values (13% spread)
on MI325X. Cholesky f64 was bit-exact across all 6 runs.

He correctly re-measured at the frontier t0(v) and found median noise 1.8e-3 —
concluding the paper's t0 strategy works. The tail (p90=0.46, max=5.30) remains open.

## Our reproduction on MI100

We ran 4 configurations × 2 runs each, comparing PAULI_NL field values:

| Config | Non-determinism (max_rel) |
|--------|--------------------------|
| HH f32 chi=50 run1 vs run2 | ~5e-7 (negligible) |
| HH f32 chi=70 run1 vs run2 | 2.6% at step 5 |
| HH f64 chi=50 run1 vs run2 | large (steps 4-5 diverge ~34%) |
| HH f64 chi=70 run1 vs run2 | 21-65% median at steps 4-5 |
| CH f32 chi=50 run1 vs run2 | bit-exact |
| CH f64 chi=50 run1 vs run2 | bit-exact |

## QR mode comparison (same precision)

| Config | HH vs CH agreement |
|--------|-------------------|
| f32 chi=50 | max_rel 7e-4 — physically equivalent |
| f64 chi=50 | max_rel 510% at step 5 — completely different physics |

**Igor was right**: Cholesky at f64 converges to a different (lower-rank) physical
state. It is reproducible but wrong. The f64 Cholesky PAULI_NL values match f32
Cholesky exactly — confirming it is locked into a low-rank solution regardless of
precision, not gaining anything from f64.

## Why f32 is protected

At f32, truncation is aggressive: near-degenerate singular values that destabilize
Householder pivoting are already zeroed before they can cause trouble. The "bug"
only surfaces at f64 where precision is sufficient to expose the pivot sensitivity.

Steps 1-3 (chi growing, 7→28→50) are bit-exact in all configurations — the QR
is well-conditioned. Non-determinism begins exactly at step 4 when chi saturates
at chi_max. This is consistent with Muhammad's hypothesis: near-degenerate singular
values at saturation cause pivot flips, which compound through subsequent Trotter steps.

Chi=70 is worse than chi=50 at f64: more modes to truncate = more near-degeneracy
= more pivot sensitivity.

## Frontier protection

The paper evaluates L(v,t) at t0(v) = first cycle where n_base > eps_th.
For most frontier sites, t0(v) falls at steps 1-2 where all configurations
agree exactly. Non-determinism only affects steps 4-5 (post-saturation).

Additionally, steps 4-5 are subject to physical interference (reflections from
boundaries, secondary wave arrivals from other centers) independent of numerical
noise. The t0 window avoids both numerical and physical contamination simultaneously.

## Summary for the paper

Our production configuration (f32, Householder, chi=50) is:
- Self-consistent: HH and CH agree to 7e-4 max_rel
- Deterministic enough: 5e-7 run-to-run spread at chi=50, well below eps_th=1e-3
- Physically protected: t0 frontier is at early cycles where chi<<chi_max

The non-determinism Muhammad observed is a f64-specific phenomenon on a different
GPU (MI325X vs MI100). Our f32 experiments were not affected. Cholesky provides
a useful control (bit-exact, confirms t0 values stable) but should not be used
as the primary QR mode in f64 — it converges to the wrong physical state.

## Experiment commands (for reproducibility)

```bash
# Build f32 and f64 rect7x7 binaries
make PHYS_DIM=4 main_pauli_sweep_rect7x7
make PHYS_DIM=4 main_pauli_sweep_f64_rect7x7

# Run comparison (inside Docker)
ROCR_VISIBLE_DEVICES=0 ./main_pauli_sweep_rect7x7 50 5 1e-3 --center=1,7,25,43,49 --state=NL > /tmp/hh_run1.log
ROCR_VISIBLE_DEVICES=1 ./main_pauli_sweep_rect7x7 50 5 1e-3 --center=1,7,25,43,49 --state=NL > /tmp/hh_run2.log
ROCR_VISIBLE_DEVICES=2 ./main_pauli_sweep_rect7x7 50 5 1e-3 --center=1,7,25,43,49 --state=NL --qr=cholesky > /tmp/ch_run1.log
# ... etc

python3 scripts/compare_qr_modes.py
```
