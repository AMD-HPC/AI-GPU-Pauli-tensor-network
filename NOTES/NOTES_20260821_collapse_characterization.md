# Collapse Characterization — 2026-08-21

## Summary

The chi-saturation collapse at δ=1e-4 is a **fundamental limitation of
frozen-message Trotter on loopy graphs**, not a numerical bug. It is fully
characterized and the valid operating regime is identified.

---

## The mechanism (proven empirically)

```
Gate rotation: ψ_new = G · (ψ absorbed with sqrt(λ_frozen))
Deabsorb:      ψ_out = ψ_new / sqrt(λ_frozen)  ← channel-dependent amplification
```

The gate is a **rotation** — it changes which channels dominate. But messages
(λ_frozen) are frozen at color-class start. After the gate, `1/sqrt(λ_k_old)`
amplifies dimensions that the rotation has moved weight into. This is NOT a
global scale — it's a channel-dependent spectral distortion.

**Observable:** `||ψ||_F` (Frobenius norm, = sqrt(Σσ_k²)) grows monotonically:
- Step 1: 1.00 → 1.09 → 1.56 within the first Trotter step
- Step 7: ||ψ|| ≈ 13–18 at center bonds
- Step 8: ||ψ|| ≈ 26–42 (saturation hit, spectrum flattens)
- Step 9: SVD produces smax > 1 → physically impossible → cascade collapse

**The invariant:** `smax > 1` after SVD is UNAMBIGUOUSLY WRONG (gate is
unitary, input cannot produce smax > 1 from a valid normalized state). This is
the alarm bell — when it fires, trust NOTHING downstream.

---

## What precision changes

**||cv|| trajectory: identical between float32 and float64** (to 6 decimal
places). The gauge drift is purely algorithmic — frozen-message Trotter on a
loopy graph — independent of arithmetic precision.

**What precision DOES change:** chi growth rate. Float64 keeps more singular
values (true rank ~6000 vs float's ~1500 from Gram matrix precision loss) →
chi grows faster → saturation hits earlier → collapse earlier. This is why
double collapses at step 8 while float collapses at step 9.

**Bond 34 (13↔50) is the trigger:** it is the THIRD gate touching site 13 in
the color sequence. It accumulates three deabsorb amplifications without any
message correction — whichever precision gets there with larger ||ψ|| collapses
first. Float32 and float64 diverge at bond 34 step 8 while bonds 33 and 32
remain nearly identical.

---

## What we tried and why we stopped

### inv_cap (REVERTED — philosophical)

Added `inv_cap=cutoff` to `diag_sqrt_vectors` to cap `1/sqrt(λ)` at
`1/sqrt(δ)`. Reduced smax from 1.76→1.13 at collapse. BUT:

**Reverted because:** smax > 1 is the alarm bell that physics is broken. Capping
it makes a sick computation appear healthy. The invariant violation is a GIFT —
it is unambiguous, measurable, and tells you to stop trusting results. Removing
it hides the problem without fixing the cause. Better to know the valid regime
and enforce it.

### Per-color BP (abandoned)

Running BP after each color class (like Julia) reconverges messages so that
absorb/deabsorb cancel. BUT: any BP sweep on color-c bonds reconverges them
to the current chi state → SVD sees no new entanglement → chi stays at 4.
Chi growth and per-color BP are in fundamental conflict in our frozen-message
scheme. Julia's simple_update does this differently (simultaneous update).

### snorm over full kfull spectrum

Changed `snorm = sqrt(Σ_kept hS²)` → `sqrt(Σ_all hS²)`. Reduces smax from
1.509→1.169 at collapse (float Householder). Doesn't prevent collapse because
float Gram matrix precision kills ~4500 of the true ~6000 singular values —
kfull is artificially small, so "all" is still missing most of the spectrum.

---

## Valid operating regime

| Config | Status | Reason |
|---|---|---|
| δ=1e-3, any chi | ✓ Safe | Chi never saturates, no amplification cascade |
| δ=1e-4, chi ≤ 200 | ✓ Safe | Chi saturates but spectrum not flat enough |
| δ=1e-4, chi ≥ 380 | ✗ Collapses | Spectrum flattens at saturation, ||ψ|| → ∞ |
| δ=1e-5, any chi | ✗ Collapses | Amplification 1/sqrt(δ)≈316 per channel |

**δ = 1e-3 is the production config.** It avoids saturation entirely.

---

## Diagnostic tools (in codebase)

- `SVD_DEBUG=1` env var: prints smax/snorm/spectrum for center bonds 32/33/34.
  **If smax > 1: results are invalid.** Stop and reduce δ or chi_max.
- `[PSI] bond=... ||psi1||=`: Frobenius norm before absorb_env.
  Growing ||ψ|| = expected (algorithmic). Explosion = approaching collapse.
- `[CV] bond=... ||cv1||=`: Frobenius norm after absorb_env, before QR.
  Should track ||ψ|| × (correction from sqrt(λ)). > 5 = danger zone.
- `--qr=cholesky`: reveals true tensor rank (kfull much larger in float64+Cholesky).
  Diagnostic only — too slow for production, prevents chi growth in f32.

---

## Future work

- **||cv|| normalization before QR**: trivial cost (one D2H + scalar divide).
  Corrects global scale but NOT channel-dependent spectral distortion. May
  delay collapse without fixing it. Not yet tried — may be worth characterizing.
- **Loop corrections (Evenbly PRR 2026)**: the real fix. Makes BP fixed point
  exact on loops → messages are true Schmidt values → absorb/deabsorb cancel
  exactly → no gauge drift. Large implementation effort.
- **MI355X**: chi=400 double with lightcone — need to verify collapse behavior
  at the true production scale.

---

## Git state

- Committed working state: 26180ba
- Step 1 (P2P chi-vector): b2698c4
- Step 2 (env_messages_v path): 26180ba
- inv_cap committed then REVERTED: 59e99c8 → reverted to 26180ba
- All experimental code (per-color BP, PSI diagnostic, inv_cap) reverted.
  Clean working tree.
