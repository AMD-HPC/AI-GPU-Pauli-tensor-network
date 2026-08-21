# Collapse Registry
# Track every simulation collapse for SVD_DEBUG characterization

## Format per entry:
# - Run ID, binary, chi, δ, topology, precision, collapse step
# - C_nl before/at collapse, smax if known
# - Type: A=gauge drift at saturation, B=BP trap (spurious fixed point)
# - SVD_DEBUG run: pending/done/log

---

## Confirmed Collapses

### C001 — A4 f64, heavy-hex, chi=200 δ=1e-3
- **Binary**: main_pauli_sweep_f64
- **Step**: 10 (C_nl +0.413 → healthy was +0.792 in f32)
- **Chi at collapse**: 200 (saturated exactly at step 10)
- **C_n1/2/3**: HEALTHY (+0.813/+0.743/+0.724) — noisy states protected
- **Mechanism**: chi growth faster in f64 → saturation at step 10 → gauge drift cascade
- **Type**: A (gauge drift at chi saturation)
- **Special**: noiseless collapses, noisy survive — noise is protective
- **SVD_DEBUG**: PENDING → run with same params + SVD_DEBUG=1

### C002 — A5, heavy-hex, chi=380 δ=1e-4 (f32)
- **Binary**: main_pauli_sweep
- **Step**: 9 (C_nl -0.095 → healthy was -0.804)
- **Chi at collapse**: 380 (saturated at step 8)
- **Mechanism**: flat spectrum at saturation, 1/√λ amplification cascade
- **Type**: A (gauge drift at chi saturation)
- **smax at collapse**: 1.07-1.76 (bonds 32/33/34) — physically impossible
- **SVD_DEBUG**: DONE (logs/e3_svd_debug.log, e7_fix_svd.log)

### C003 — B1, rect_7x7, chi=70 δ=1e-3 (f32)
- **Binary**: main_pauli_sweep_rect7x7
- **Step**: 6 (C_nl +0.417, gate_err=1.72e-3)
- **Chi at collapse**: 70 (saturated at step 2!)
- **Mechanism**: degree-4 sites → chi saturates in 2 steps → immediate gauge drift
- **Type**: A (gauge drift, accelerated by degree-4 topology)
- **Special**: C4 symmetric but still collapses — C4 protects BP traps, not gauge drift
- **SVD_DEBUG**: PENDING

### C004 — B_r7_chi30, rect_7x7, chi=30 δ=1e-3 (f32)
- **Binary**: main_pauli_sweep_rect7x7
- **Step**: ~3-4 (C_nl degrading, gate_err=1.59e-4 at step 3)
- **Chi at collapse**: 30 (saturates step 2)
- **Type**: A
- **SVD_DEBUG**: PENDING

### C005 — B_r7_chi40, rect_7x7, chi=40 δ=1e-3 (f32)
- **Step**: ~3-4, gate_err=6.45e-5 at step 3
- **Type**: A
- **SVD_DEBUG**: PENDING

### C006 — B_r7_chi50, rect_7x7, chi=50 δ=1e-3 (f32)
- **Step**: ~3-4, gate_err=3.26e-5 at step 3
- **Type**: A
- **SVD_DEBUG**: PENDING

---

## SVD_DEBUG Run Queue

Priority order (most informative first):

1. **C001** — f64 hh chi=200: the "noisy survives, noiseless collapses" case
   ```bash
   SVD_DEBUG=1 ROCR_VISIBLE_DEVICES=? ./main_pauli_sweep_f64 200 10 1e-3 \
       --center=13 --v_hop=0.500 > logs/C001_svd_debug.log 2>&1
   ```

2. **C003** — r7 chi=70: degree-4 collapse, compare smax to hh
   ```bash
   SVD_DEBUG=1 ROCR_VISIBLE_DEVICES=? ./main_pauli_sweep_rect7x7 70 10 1e-3 \
       --center=24 --v_hop=0.500 > logs/C003_svd_debug.log 2>&1
   ```

3. **C004-C006** — r7 chi probe: how does smax scale with chi_max on rect_7x7?
   (batch after C003)

---

## Collapse Type Classification

**Type A: Gauge drift at chi saturation**
- Signature: smax > 1 after SVD (physically impossible)
- ||ψ||_F grows monotonically, explodes at saturation
- N(v,t): loop-shell spikes, center drain, ΣN +46%
- Causal violation: N_outside spikes
- Cause: frozen-message deabsorb 1/√λ amplification
- Present in: ALL collapses seen so far

**Type B: BP spurious fixed point (trap)**
- Signature: BP converges but to wrong fixed point
- C(t) wrong sign/magnitude from early steps
- N(v,t): no causal violation, wrong distribution shape
- Cause: loopy graph BP non-uniqueness
- Present in: NOT YET OBSERVED in CppSim
  (C4 symmetry protects rect_7x7; heavy-hex traps may exist at larger chi)

---

## Key Findings

1. **All observed collapses are Type A** (gauge drift), not Type B (BP trap)
2. **Noise is protective**: noisy states have smaller chi → never saturate → survive
3. **Degree-4 (rect_7x7) collapses faster**: chi saturates in 2 steps vs 7-8 steps (hh)
4. **C4 symmetry ≠ collapse-free**: protects BP uniqueness, not gauge drift
5. **δ controls threshold**: δ=1e-3 safe, δ=1e-4 collapses for both topologies
6. **f64 collapses EARLIER than f32**: faster chi growth → earlier saturation
