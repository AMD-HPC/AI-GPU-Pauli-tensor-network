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
- **Type**: B (BP/measurement failure at chi saturation) — NOT Type A
- **Special**: noiseless collapses, noisy survive — noise is protective
- **SVD_DEBUG**: DONE → logs/C001_hh_f64_svd.log
  smax=0.77-0.94 throughout — SVD physically valid, collapse is in BP/measurement

### C002 — A5, heavy-hex, chi=380 δ=1e-4 (f32)
- **Binary**: main_pauli_sweep
- **Step**: 9 (C_nl -0.095 → healthy was -0.804)
- **Chi at collapse**: 380 (saturated at step 8)
- **Mechanism**: flat spectrum at saturation, 1/√λ amplification cascade
- **Type**: A (gauge drift at chi saturation)
- **smax at collapse**: 1.07-1.76 (bonds 32/33/34) — physically impossible
- **SVD_DEBUG**: DONE (logs/e3_svd_debug.log, e7_fix_svd.log)

### C003 — B_r7_chi30/40/50, rect_7x7, chi=30/40/50 δ=1e-3 (f32)
- **Binary**: main_pauli_sweep_rect7x7
- **Pattern**: GRADUAL DECAY, not sudden collapse
  - Step 2: chi saturates at 28 for ALL chi_max values (30/40/50)
  - C_nl follows nearly identical trajectory regardless of chi_max
  - Step 10: C_nl ≈ 0.22-0.23 (decays from -0.913 at step 1)
  - gate_err grows monotonically (1.7e-2 at step 10)
- **Type**: UNKNOWN — may be Type A (gauge drift) or physical C(t) decay
  - NOT like heavy-hex Type A: no sudden flip, no sign inversion
  - chi_max irrelevant → physical entanglement demand is ~28 channels
  - Step 1 C_nl=-0.913 (not -0.924 like hh) → different initial behavior
- **Key finding**: chi_max=28 is the physical entanglement demand for rect_7x7 δ=1e-3
  After saturation at step 2, truncation error accumulates steadily
- **SVD_DEBUG**: PENDING — needed to determine if smax>1 (Type A) or physical decay
  ```bash
  SVD_DEBUG=1 ROCR_VISIBLE_DEVICES=? ./main_pauli_sweep_rect7x7 50 10 1e-3 \
      --center=24 --v_hop=0.500 > logs/C003_r7_svd_debug.log 2>&1
  ```

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

## SVD_DEBUG Results (C001, C003)

### C001 — hh f64 chi=200 δ=1e-3 step 10: smax < 1 throughout
- bonds 32/33/34: smax=0.77-0.94, snorm≈0.984-1.002
- **No Type A gauge collapse** despite C_nl=+0.413
- Chi=200 fully saturated at step 10 but SVD is physically valid
- Collapse comes from BP/measurement failure at chi saturation, not gauge
- **Classification: Type B (BP/measurement at saturation)**

### C003 — rect_7x7 chi=50 δ=1e-3: smax < 1 throughout, snorm≈1
- All bonds: smax=0.87-0.99, snorm=0.997-0.999
- **No collapse of any type** — physically honest truncation
- C(t) decays because chi=50 insufficient for true entanglement demand (~28+ channels)
- The simulation fails gracefully, not catastrophically
- **Classification: Physical truncation error (not a collapse)**

---

## Collapse Type Classification (UPDATED)

**Type A: Gauge drift at chi saturation**
- Signature: **smax > 1** after SVD (physically impossible for unitary gate)
- ||ψ||_F grows monotonically, explodes at saturation
- N(v,t): loop-shell spikes, center drain, ΣN +46%, causal violation
- Cause: frozen-message deabsorb 1/√λ amplification cascades
- Present in: hh f32 chi=380 δ=1e-4 (C002)
- Detectable: hard invariant violation, unambiguous

**Type B: BP/measurement failure at chi saturation**
- Signature: **smax < 1** (SVD valid) but C_nl collapses
- Noisy states (smaller chi) survive; noiseless (larger chi) collapses
- N(v,t): TBD — needs investigation
- Cause: chi saturation makes BP fixed point unstable for the measurement
- Present in: hh f64 chi=200 δ=1e-3 NL only (C001)
- Detectable: C_nl collapses but C_n1/2/3 healthy; smax < 1

**Type P: Physical truncation (not a collapse)**
- Signature: **smax < 1**, snorm ≈ 1, C(t) decays monotonically
- Chi_max insufficient for true entanglement; simulation degrades gracefully
- No sudden inversion, no causal violation
- Present in: rect_7x7 chi=30/40/50 δ=1e-3 (C003)
- Detectable: gate_err grows smoothly, C_nl decays without inverting

---

## Key Findings (UPDATED)

1. **Three distinct failure modes**, all separable by smax:
   - Type A (smax>1): catastrophic gauge cascade — hard stop, trust nothing
   - Type B (smax<1, C_nl collapses): BP/measurement failure — noiseless collapses, noisy ok
   - Type P (smax<1, C_nl decays): physical truncation — graceful, honest
2. **Noise is protective against Type A and B**: smaller chi → never saturate → survive
3. **f64 reaches Type B earlier**: faster chi growth → saturation at step 10 where f32 is at 162
4. **Rect_7x7 shows Type P only**: C4 symmetry + girth-4 loops → no gauge traps, honest decay
5. **smax is the diagnostic**: >1 = stop immediately; <1 = characterize further
6. **δ=1e-3, chi≤200 f32**: safe regime for heavy-hex production runs
