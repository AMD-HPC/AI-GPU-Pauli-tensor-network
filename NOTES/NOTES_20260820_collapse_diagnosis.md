# Session Notes — 2026-08-20

## The Collapse Diagnosis — Complete Chain

### What we proved empirically

1. **||cv|| > 1 from step 2** (chi=23, no saturation, near-tree)
   - `||cv||_F` = Frobenius norm of absorbed tensor before QR
   - Must be 1 for exact Vidal gauge
   - Grows monotonically: 1.000→1.136→1.291→...→1.844→6.833 (collapse)
   - SAME trajectory for chi=200 and chi=380 through steps 1-6

2. **The source of ||cv|| > 1 is the MESSAGES, not ψ**
   - `sqrt(Σλ²_msg) = ||cv||` exactly at every step
   - `Σλ²` of the messages going into pseudo_sqrt is already > 1
   - At step 9 collapse: bond 32 Σλ²=5.79, bond 33 Σλ²=7.21, bond 34 Σλ²=7.54

3. **Saturation is the trigger, not the cause**
   - Chi=200 and chi=380 follow identical ||cv|| trajectory until saturation
   - After saturation, chi=200 drift stabilizes (~1.77 at step 10), chi=380 blows up
   - Chi saturation removes the damping that kept exponential growth subcritical

4. **Inflation enters via the chi×chi path, not messages_v**
   - `pseudo_sqrt` reads `env_messages` = `frozen_msg` = snapshot of `state.messages` (chi×chi)
   - `state.messages` (chi×chi) may carry inflated values
   - `state.messages_v` (chi-vector) is correctly Vidal-normalized by the gate SVD
   - **Phase 0b is INCOMPLETE**: gate still reads chi×chi as the source of sqrt_env_v

---

## The Inconsistency (root cause of the fix failing)

Phase 0b migrated absorb_env to use sqrt_env_v (chi-vector) for the
CONTRACTION, but left pseudo_sqrt reading from chi×chi env_messages as the
SOURCE. The full chain is:

```
do_trotter:
  frozen_msg ← state.messages (chi×chi snapshot)    ← WRONG SOURCE
    ↓
apply_binary_gate:
  pseudo_sqrt(frozen_msg_chi×chi → sqrt_env_v)       ← reads chi×chi diagonal
  absorb_env uses sqrt_env_v (chi-vector)            ← correct after Phase 0
```

The fix: pseudo_sqrt should read from a frozen snapshot of state.messages_v
(chi-vector), not from frozen_msg (chi×chi). Then:
- sqrt_env_v is derived from the clean Vidal-normalized messages_v
- chi×chi state.messages truly becomes inner-product-only
- The gate path is fully chi-vector end-to-end

---

## What we tried that DIDN'T work (and why)

1. **snorm = Σ_all kfull hS²** (full spectrum snorm):
   Reduces smax 1.509→1.169 but doesn't prevent collapse. Float Householder
   only sees ~1500 of the true ~6000 singular values (Gram matrix rank loss).

2. **CV renormalization** (scale cv to unit norm before QR):
   Changes baseline (physics altered), still collapses. The inflation is in the
   messages, not addressable by rescaling cv after the fact.

3. **Cholesky regularization** (CHOL_DIAG_REG 1e-8→1e-5):
   1e-5 prevents collapse but drifts C_nl (adds artificial noise to spectrum).
   Not a real fix — patches the symptom by lifting near-zero eigenvalues.

4. **f64 + Cholesky**:
   WORKS (no collapse, 600s/10 cycles). Sees true rank=6000, snorm≈1.000.
   Correct but too slow for production.

---

## The Friday Plan — Complete Phase 0b properly

### The fix (one conceptual change, multiple code sites)

**Replace:** `pseudo_sqrt(env_messages[chi×chi], ...)` with a frozen snapshot of `messages_v` (chi-vector).

**Why this fixes ||cv|| > 1:**
- `messages_v` is written by the gate's `out_messages_v`: `lam[k] = hS[k]/snorm`
  with `snorm = sqrt(Σ_kept hS²)` → Σlam²_kept ≤ 1 (Vidal normalization)
- Chi×chi `messages` diagonal is the same values — but may pick up stale/inflated
  values through the chi×chi path
- By reading `messages_v` directly, sqrt_env_v = sqrt(lam) with lam properly
  normalized → Σ(sqrt_env_v)² = Σlam ≤ sqrt(chi) (Cauchy-Schwarz), and more
  importantly the gauge is consistent

### Code changes needed

**1. trotter.hpp: freeze messages_v alongside messages**
```cpp
// Add frozen_msg_v (chi-vector snapshots) alongside frozen_msg (chi×chi)
std::vector<Scalar*> frozen_msg_v(nm, nullptr);  // chi-vector snapshot
// At color-class start: copy state.messages_v → frozen_msg_v
// Pass frozen_msg_v to apply_binary_gate as env_messages_v
```

**2. gate.hpp: pseudo_sqrt reads from frozen messages_v directly**
```cpp
// Instead of:
pseudo_sqrt(env_messages[eb.msg_incoming(sv1.id)], chi, ws, sqV, isqV, cutoff)
//          ↑ reads chi×chi, extracts diagonal

// Use:
// frozen_msg_v[eb.msg_incoming(sv1.id)] is already a chi-vector
// Just call diag_sqrt_vectors directly on it:
diag_sqrt_vectors(frozen_msg_v[eb.msg_incoming(sv1.id)], sqV, isqV, chi, cut)
```
No extract_diagonal needed — the chi-vector IS the diagonal.

**3. gate.hpp: apply_binary_gate signature**
Add `const std::vector<Scalar*>* env_messages_v = nullptr` parameter.
When provided, use it for pseudo_sqrt. Keeps backward compatibility.

**4. trotter.hpp: alloc + free frozen_msg_v**
Same lifetime as frozen_msg — allocated once per color class or pre-allocated.
Chi-vector so much smaller: chi_max floats vs chi_max² floats.

**5. heavyhex_physics.hpp: same changes as trotter.hpp**
(3 blocks at ~223, ~321, ~419 — same pattern)

### Validation
- Step 1: baseline chi=70 δ=1e-3 unchanged (||cv||=1 throughout since no saturation)
- Step 2: chi=380 δ=1e-4 — does ||cv|| stay closer to 1? does collapse move/disappear?
- Step 3: run SVD_DEBUG=1 to confirm smax < 1 throughout

### Expected outcome
||cv|| at step 2 should drop from 1.136 toward 1.0 — since messages_v carries
the correctly-normalized lam[k] from the gate SVD, not the potentially-inflated
chi×chi diagonal. Whether this fully prevents the collapse depends on whether the
Σlam² drift is the sole source or if the within-color-class staleness (frozen
messages vs updated ψ within one color class) also contributes.

If ||cv|| still > 1 after this fix → the remaining source is within-color-class
staleness (ψ updated by gate N, but env_messages frozen before gate N ran). That
is a different and harder problem (requires per-gate message updates = more sweeps).

---

## What we know about the within-color-class staleness

At step 2 with v_hop=0.5: lightcone = 1 hop = star graph (tree). Yet ||cv||=1.136.
On a tree, BP gives exact messages → messages_v should have Σlam²=1 exactly.
So if messages_v IS properly normalized and ||cv|| is still > 1 at step 2 on a tree
→ the residual is within-color-class staleness (ψ₁_output used by gate 2 with
env from before gate 1 ran).

This is separable from the chi×chi/messages_v inconsistency — fix the inconsistency
first and measure the residual.

---

## Experiments run today

Logs in /scratch/MPF/CppSim/logs/:
- e0_f32_chi380_d1e4.log — baseline collapse (confirmed)
- e0_f32_chi380_d1e3.log — no collapse (δ=1e-3)
- e1_f64_chi200_d1e4.log — no collapse (chi=200)
- e2_f32_chi100/200/380_d1e5.log — chi=380 collapses step 7
- e3_svd_debug.log — SVD spectrum showing smax>1
- e5_svd_*.log — QR method comparison (all collapse same step)
- e6_f64_cholesky_svd.log — f64+Cholesky: NO COLLAPSE, snorm=0.999999 ✓
- e7_fix_f32_chi380.log — snorm fix: still collapses
- e8_chol_reg*.log — regularization sweep: 1e-5 prevents collapse but drifts
- e9_cv_norm.log — ||cv|| diagnostic: inflation source identified
- e10_cv_norm.log — chi=200 control: same ||cv|| trajectory, stabilizes after saturation
- e11_msg_norm.log — MSG diagnostic: Σλ²_msg > 1 is the source of ||cv|| > 1
- e12_renorm_cv.log — cv renormalization: still collapses, baseline shifted

## Current git state

HEAD: 18a7820 (collapse diagnosis commit, pushed)
Clean working tree (bad patches reverted).
gate.hpp has SVD_DEBUG + CV diagnostic + --qr flag + --chol-reg flag (committed).
