# CppSim Heavy-Hex MPF — Two-GPU Implementation Summary

---

## SESSION 2026-07-15 — Restart: Julia as Ground Truth

### What we discovered

**1. The models were never apple-to-apple.**
CppSim was simulating a different physical model than Julia:
- CppSim: pure Heisenberg `exp(-iθ(XX+YY+ZZ))`, uniform coupling J=1.0, dt=0.1, NO kick gate
- Julia: hardware-native 3-Rzz decomposition, heterogeneous couplings from device JSON, epsilon=0.25, WITH kick gate `Rx(2φ)` on every site after every Trotter step (φ=1.41371669)

**2. The kick gate is essential physics.**
The kick gate makes the model quantum chaotic. Without it, the pure Heisenberg model
is integrable and chi grows slowly. With it, chi doubles every cycle for the noiseless
state. This is intentional — chaotic dynamics is what makes MPF meaningful.

**3. Noiseless chi is intractable beyond cycle 8.**
With the kick gate and epsilon=0.25 on heavy_hex_3x3:
- Chi_nl doubles every cycle: 4→14→28→58→110→202→364→637→1086 (cycles 1-9)
- Cycle 10 OOMs at 32GB
- Practical chi_max for noiseless: ~400 (saturates at cycle 7)

**4. Noisy chi is manageable.**
Noise suppresses entanglement — chi_n grows linearly ~25/cycle:
- Cycles 1-9: 4→14→24→39→58→82→108→134→154
- At chi_max=200, noisy states run comfortably to 10+ cycles

**5. C(t) oscillates in sign due to kick gate.**
The kick `Rx(2φ)` with 2φ≈9π/10 rotates Z → -0.951·Z + ... each cycle.
C(t) alternates sign: negative at odd cycles, positive at even cycles.
CppSim's always-positive C(t) is a different observable.

**6. NP=25 approximation was wrong.**
Pre-composing the noise PTM (`noise^25`) is not equivalent to applying it
interleaved with gates in the TN state. Julia applies noise PTM as a gate
to the full TN with fresh BP messages — a many-body effect that per-bond
PTM composition cannot capture.

**7. BP frequency matters.**
Julia runs BP between every gate group (~6× per Trotter step).
CppSim runs BP once per full Trotter cycle.
This causes 5× chi difference and 10× difference in gamma spread.

### Golden Reference (Julia, Float32, epsilon=0.25, gamma=1.0, maxdim=uncapped, cutoff=1e-4)

Script: `workloads/noisy_ptm_simulation/compute_overlaps_gpu.jl heavy_hex_3x3 10 200`
Log: `logs/golden_noiseless_gamma10_eps025_maxdim200.log`

| Cycle | C_nl | C_n(γ=1.0) | overlap | χ_nl | χ_n | t(s) |
|-------|------|------------|---------|------|-----|------|
| 1 | -0.6786 | -0.6665 | 0.9809 | 4 | 4 | 157 |
| 2 | +0.8127 | +0.7396 | 0.9312 | 14 | 14 | 78 |
| 3 | -0.7205 | -0.6146 | 0.8320 | 28 | 24 | 81 |
| 4 | +0.6808 | +0.5475 | 0.7217 | 58 | 39 | 78 |
| 5 | -0.6916 | -0.5058 | 0.6208 | 110 | 58 | 82 |
| 6 | +0.6207 | +0.4435 | 0.5330 | 202 | 82 | 90 |
| 7 | -0.6346 | -0.4044 | 0.4528 | 364 | 108 | 119 |
| 8 | +0.5921 | +0.3555 | 0.3832 | 637 | 134 | 214 |
| 9 | -0.5998 | -0.3292 | 0.3243 | 1086 | 154 | 626 |
| 10 | OOM | — | — | ~2000+ | ~175 | — |

Notes:
- maxdim was NOT applied in this run (bug: missing from apply_kwargs) — chi_nl is uncapped
- C(t) oscillates in sign every cycle (kick gate effect)
- |C_nl| stays ~0.60-0.81 — kicked model does not decay to zero
- |C_n| decays steadily: 0.667→0.329 over 9 cycles
- overlap decays monotonically: 0.981→0.324

### C(t) comparison — current state

| Cycle | C(t) Julia | C(t) CppSim | ratio |
|-------|-----------|-------------|-------|
| 1 | -0.6786 | -0.8033 | 0.845 |
| 2 | +0.8127 | +0.6005 | 0.739 |
| 3 | -0.7205 | -0.4816 | 0.668 |
| 4 | +0.6808 | +0.5043 | 0.741 |

Signs match ✓. Magnitudes differ by ~15-33% and growing with cycle.

**Known sources of difference (investigated):**

1. **Initial state ρ** — Julia measures `Tr(Z(t)·ρ)` where ρ has per-qubit +1/−1 from device JSON.
   CppSim measures `Tr(Z(t)·I⊗...⊗I)` — assumes all qubits in |0⟩.
   
   Attempted fix: per-site rho operator in `bp_partitionfunction`. Failed because in CppSim's
   canonical gauge, Z content at non-center sites lives in bond tensors (messages), not in
   the site tensor's physical index d=3. The per-site operator change has nothing to act on.
   
   **Correct fix**: initialize a second TN representing ρ and contract `inner(ψ(t), ρ)` via BP,
   exactly as Julia does. This requires a new measurement function.

2. **Gate PTM sign convention** — Julia uses negative Rzz angles (Δ = −coupling).
   CppSim tried both signs — C(t) identical. The gate is symmetric under sign flip for Z observable.

3. **Truncation frequency** — Julia SVDs after every individual gate; CppSim SVDs once per
   composed PTM. This explains chi difference but NOT C(t) difference (verified: C(t) stable
   across cutoff 1e-2 to 1e-4 in CppSim).

4. **Site indexing** — Julia site (3,) = CppSim site 2 ✓. Confirmed correct.

**Root cause**: CppSim measurement assumes all-zero initial state. Center site 2's three
neighbors (sites 32, 33, 34) all have spin=−1 (|1⟩) in the device initial state. The correct
C(t) requires contracting the evolved operator with the actual device ρ, not the all-zero state.

**TODO before proceeding**:
Implement `measure_with_rho(ψ_bpc, ρ_bpc)` — a BP inner product between the evolved operator
TN and the initial state TN. This matches Julia's `inner(network(bpc), rho; alg="bp")` exactly.
The ρ TN is initialized once from `HEAVY_HEX_3x3_IDEAL_KICK_INITIAL_STATE` and never evolved.

### inner_op_state — implementation status (2026-07-16)

**Goal**: Implement `C(t) = Tr(O(t) · ρ) = inner(ψ(t), ρ)` matching Julia's
`inner(network(bpc), rho; alg="bp")`.

**Correct algorithm (3 steps):**
1. Contract physical index of each ψ_v with ρ_v → φ_v (no physical index)
   `φ_v[χ_1,...] = Σ_d ψ_v[χ_1,...,d] × ρ_v[d]`
2. Run fresh BP on φ_v tensors (physical dim=1)
3. Compute Bethe partition function on φ_v

**Key facts learned:**
- φ_v (step 1) correctly contracts ρ into ψ — verified at t=0: C(0)=1 ✓
- Messages from ψ BP are WRONG for φ BP — need fresh BP on φ_v
- `bp_converge` uses hardcoded D — cannot be reused for physical dim=1
- Need dedicated BP sweep for φ_v tensors

**Next step**: implement `bp_update_phi` — same message update formula
as `bp_update` but with physical dim=1 site tensors φ_v. Then:
```
BpResult bp_converge_phi(grid, phi, bp_msg, n_sweeps)
```
followed by Bethe partition function on φ_v.

**What works:**
- Step 1 (physical contraction) ✓
- t=0 test: C(0)=1 ✓
- flip32 changes C(t) ✓ (rho correction is active)

**What doesn't work yet:**
- Messages not correct for φ_v bilinear form
- `all+1 rho ≠ autocorrelation` — different observables, expected
- `device rho = -0.403` vs Julia `-0.679` — wrong messages

**Files:**
- `test_inner_op_state.cpp` — standalone test (4/5 pass currently)
- `include/measures.hpp` — `inner_op_state()` function

**Build note**: `msg_from()` not found — use `msg_from_a()`/`msg_from_b()`
depending on direction. Fix before next build.

### Physicist conversation — key insights (2026-07-16)

**1. Gate decomposition intent confirmed**
The Trotter decomposition into gates1→gates2→gates3→gates4 exists for one reason:
noise must be inserted at specific physical points. The chi-damping side effect is
accidental. CppSim's precomputed PTM is a valid optimization — physics preserved,
fewer SVDs.

**2. sqrt(Complex) is a mathematical artifact**
Physicists agree intuitively. The vertex scalar `a_v` can go negative due to BP
approximation errors compounding through many intermediate states. With precomputed
large PTMs (one SVD per color, not per gate), BP sees fewer intermediate states →
less drift → fewer negative vertex scalars. CppSim's log-space sign tracking avoids
sqrt entirely — numerically cleaner.

**3. MPF consistency > accuracy**
The MPF sum `Σ cᵢ × C(γᵢ)` works by cancellation — systematic errors that are
identical across noise levels cancel out. HOWEVER: the A matrix `A[i,j] = ⟨ψᵢ|ψⱼ⟩`
and B vector `B[i] = ⟨ψᵢ|ψ_nl⟩` require inner products that must be computed with
HIGH CONSISTENCY across noise levels. Any asymmetric systematic error gets amplified
by the solver — especially when A becomes ill-conditioned (late cycles).

**4. MPF coefficient divergence observed**
From Julia epsilon-based MPF output (noise={0.1, 0.15, 0.2}):
- noise=0.1 coefficients ~0.3-0.4 — stable
- noise=0.15 grows 1.4→2.7 — unstable  
- noise=0.2 grows -0.7→-2.2 — unstable, wrong sign
This is A matrix ill-conditioning at late cycles when noise damps all signals
toward the same value — eigenvalues shrink, condition number explodes.
Confirms: consistency in inner product computation is critical.

**5. Precomputed PTM argument is alive**
Physicists had not considered precomputing PTM_eff = gate·noise interleaved.
The argument: for fixed hardware calibration, precompute once offline for each
gamma → same physics, 12× fewer runtime SVDs. Valid optimization for paper.

---

### Plan going forward (Choice A — match Julia exactly)

**Step 1**: Add kick gate to CppSim `trotter_layer()` — `Rx(2φ)` on every site after each color class
**Step 2**: Implement 3-Rzz hardware gate decomposition with heterogeneous couplings
**Step 3**: Match epsilon=0.25 (J=1.0, dt=0.25 equivalent via coupling scaling)
**Step 4**: Validate noiseless C(t) matches Julia cycle-by-cycle
**Step 5**: Add noise interleaved between gate groups (not pre-composed)
**Step 6**: Run with chi_max=400 noiseless, chi_max=200 noisy — get golden MPF dataset

### Script fixes applied (2026-07-15)
- `src/utils.jl`: `pseudo_sqrt_inv_sqrt` — adapt D to CPU before `map_diag`, adapt back with `datatype(D)`
- `compute_overlaps_gpu.jl`: epsilon_ref=0.25 restored, Float32 states, added C_nl/C_n via rho inner product
- **Bug to fix**: `maxdim` missing from `apply_kwargs` — chi was uncapped in golden run

---

**Hardware**: 2× AMD MI100 (32 GB HBM2, peer-to-peer enabled)  
**Graph**: heavy_hex_3x3 — 68 sites, 76 bonds, 3-colorable, D=4 (Pauli basis)  
**Hamiltonian**: Isotropic Heisenberg H = J·Σ(XX+YY+ZZ), J=1, dt=0.1, cutoff=1e-4

## Architecture

| GPU | Role | States | χ_max | Memory |
|-----|------|--------|-------|--------|
| GPU 0 | Noiseless evolution | 1 × ideal | **400** | ~21 GB |
| GPU 1 | Noisy evolution | 3 × noisy (γ=1.0, 1.4, 1.8) | **150** | ~3.6 GB |

**χ_max parameters are set to fit within 32 GB per GPU** — not physical limits.
Larger hardware (e.g. MI355X, 309 GB) would allow χ>>400 for the noiseless state
and remove truncation errors entirely.

- **Parallel trotter**: `std::thread` — GPU 0 ∥ GPU 1 per cycle
- **Overlap**: `⟨ψ_nl|ψ_n(γ)⟩` via HIP peer-to-peer (GPU1→GPU0, zero CPU involvement)
- **Output**: C_nl + 3 overlaps per cycle → direct input to MPF linear regression

## Gamma values

The three gamma values (1.0, 1.4, 1.8) are taken directly from hardware noise
characterization data provided in the Julia MPF simulator JSON
(`heavy_hex_3x3_gamma1.00-1.40-1.80_hardware.json`). Each gamma scales the
noise PTM per bond:
- **γ=1.0**: hardware noise as measured on the real device
- **γ=1.4, 1.8**: amplified noise for MPF zero-noise extrapolation

## Results (chi_nl=400, chi_n=150, 10 cycles, Householder QR)

| Cycle | C_nl | OV(γ=1.0) | OV(γ=1.4) | OV(γ=1.8) | χ_nl | χ_n | t_nl(s) | t_n(s) | t_tot(s) |
|-------|------|-----------|-----------|-----------|------|-----|---------|--------|----------|
| 1  | 0.886203 | 0.886398 | 0.886475 | 0.886553 | 4   | 4   | 0.7  | 1.0  | 1.5  |
| 3  | 0.322922 | 0.326683 | 0.327884 | 0.328960 | 33  | 31  | 0.5  | 1.5  | 1.9  |
| 5  | 0.105018 | 0.112656 | 0.114581 | 0.116096 | 99  | 92  | 0.9  | 4.1  | 4.7  |
| 7  | 0.189173 | 0.235314 | 0.246240 | 0.254351 | 278 | 150 | 3.9  | 11.9 | 14.6 |
| 8  | 0.189680 | 0.257312 | 0.272946 | 0.284375 | 400 | 150 | 9.1  | 18.7 | 26.3 |
| 10 | 0.107682 | 0.172697 | 0.186936 | 0.197447 | 400 | 150 | 30.7 | 39.3 | 61.8 |

C_nl = noiseless autocorrelation C(t) = ⟨Z_c(t)Z_c(0)⟩.  
OV(γ) = ⟨ψ_nl(t)|ψ_n(γ,t)⟩ overlap for MPF fit.  
The 4 curves per cycle (γ=0, 1.0, 1.4, 1.8) feed directly into the MPF
linear extrapolation to zero noise.

## Physics notes (pending validation)

**1. Noisy χ evolution**  
The noisy state χ grows to 150 (cap) by cycle 7. In the Julia reference
implementation, noisy χ stays ~150 at cycle 10. The gate decomposition differs:
CppSim uses a single full XX+YY+ZZ PTM per color class; Julia uses a multi-sublayer
decomposition (Ry/Rx/Rzz) with intermediate noise applications after each sublayer.
**Matching Julia's χ scaling is an open item** — requires matching the gate
decomposition, coupling strength (epsilon=0.25), kick gates, and noise timing.

**2. Noiseless C(t)**  
The quantum oscillation (dip at cycle 4-5, recovery at cycles 7-8) is confirmed
independent of χ (converged at χ=200) and is a genuine finite-lattice recurrence
of the XXX model — not a numerical artifact.

**3. Overlap approximation**  
`⟨ψ_nl|ψ_n(γ)⟩` currently uses BP on the noisy state alone as an approximation.
The exact bilinear form requires BP on the double-layer network — a known next step.

## Performance context

These timings represent **bare-metal C++/HIP performance** — the upper bound
achievable on this hardware without framework overhead. The 9-10× speedup over
Julia/AMDGPU.jl (measured at matched χ=430, cutoff=1e-4) comes from:
- Zero GPU memory allocation during simulation (pre-allocated workspaces)
- Direct rocBLAS/rocSOLVER calls without framework dispatch overhead
- On-device `pseudo_sqrt` kernel eliminating CPU–GPU transfers

These numbers are estimates of the best achievable on MI100 for this computation.
Larger hardware (MI355X) would extend the achievable χ range proportionally.

## MPF Extrapolation Results (2026-07-13)

Script: `mpf_extrapolate.py --data mpf_out.dat`

Coefficients for γ={1.0, 1.4, 1.8}: **c=(+7.875, -11.250, +4.375)**
(large alternating signs — typical for MPF, amplifies statistical noise on hardware)

| Cycle | C_nl | OV_extrap | \|error\| | Quality | Notes |
|-------|------|-----------|----------|---------|-------|
| 1 | 0.886 | 0.886 | 5.7e-5 | excellent | within lightcone, chi small |
| 2 | 0.611 | 0.613 | 2.0e-3 | good | |
| 3 | 0.323 | 0.324 | 1.4e-3 | good | |
| 4 | 0.145 | 0.142 | 2.6e-3 | good | |
| 5 | 0.105 | 0.108 | 2.6e-3 | good | |
| 6 | 0.146 | 0.179 | 3.3e-2 | fair | chi_nl approaching saturation |
| 7 | 0.189 | 0.279 | 9.0e-2 | poor | chi_nl=278, truncation growing |
| 8-10 | — | — | 0.15-0.19 | poor | **chi_nl saturated at 400** |

**Key finding**: MPF extrapolation is excellent/good for cycles 1-5 and degrades
sharply from cycle 6 onward — precisely when χ_nl begins to saturate (163→278→400).

**Two interpretations (both likely true):**
1. **Noiseless truncation**: once χ_nl is capped, C_nl itself is approximate —
   the "reference" we're comparing against is no longer exact
2. **Noise model breakdown**: at late cycles OV(γ) >> C_nl with large gamma
   dependence; the linear-in-gamma MPF assumption breaks down in this regime

**The lightcone connection**: MPF works within the lightcone boundary (cycles 1-5,
where chi_min=1 meaning some bonds are still product state). Beyond the boundary
(cycle 14 where chi_min first grows), the full graph is entangled and truncation
errors are unavoidable with the current chi_max=400.

**Next explorations:**
- Repeat with NOISE_POWER=1 and NOISE_POWER=25 — does MPF accuracy change?
- Repeat with larger chi_nl (need larger GPU) — does extrapolation improve at late cycles?
- Spatial MPF: apply site-by-site extrapolation to Pauli weights — does it work
  within the lightcone and fail outside?
- Different observable sites (site 13 vs site 2) — center of graph vs corner
- Different gamma spacing — {1.0, 1.3, 1.6} vs {1.0, 1.4, 1.8}
- Vary dt to change effective Trotter step and chi growth rate

## Files

| File | Description |
|------|-------------|
| `main_heavyhex_mpf.cpp` | Two-GPU MPF driver |
| `main_heavyhex.cpp` | Noiseless standalone |
| `main_heavyhex_noisy.cpp` | Noisy standalone (--gamma=X) |
| `include/heavyhex.hpp` | FlatHeavyHexGrid, NoisyHeavyHexGrid |
| `include/heavyhex_physics.hpp` | HeavyHexSimulator, NoisyHeavyHexSimulator |
| `include/heavyhex_noise_data.hpp` | Hardcoded PTMs (76 bonds × 3 gammas) |
| `include/overlap_gpu.hpp` | P2P overlap computation |
| `extract_noise_ptm.py` | JSON → C++ PTM array generator |
| `PaperHeavyHex/heavyhex.tex` | Paper draft |
| `PaperHeavyHex/README.md` | Detailed implementation notes |

## Noise interleaving approximation (2026-07-13)

The number of noise applications per Trotter step is controlled by `NOISE_POWER`
in `extract_noise_ptm.py`. This approximates different physical models:

| NOISE_POWER | Model | Noisy χ at cycle 10 | Notes |
|-------------|-------|---------------------|-------|
| 1 | Once per color class (CppSim default) | ~410 | Chi grows too fast |
| 3 | Once per sublayer group | ~316 | Intermediate |
| 25 | Once per individual gate (~Julia) | ~146 | Matches Julia chi |

With NOISE_POWER=25:
- Noisy chi matches Julia (~146 vs ~166 at cycle 10) ✓
- t_n drops from 39s → 6s per cycle (small chi = fast) ✓
- C(t) no longer shows quantum recurrence — noise washes it out
- OV(γ) > C_nl throughout — noise inflates the overlap in Heisenberg picture

**Key observation**: OV ordering is non-monotonic at late cycles (OV(1.4) > OV(1.0)
at cycle 10). A simple linear MPF regression may not be stable for all cycles.

**Next step**: Apply proper MPF linear regression (solve for coefficients c₀,c₁,c₂,c₃
such that c₀·C_nl + c₁·OV(1.0) + c₂·OV(1.4) + c₃·OV(1.8) extrapolates to zero noise)
and compare extrapolated values against C_nl as a validation.

The `NOISE_POWER` parameter should be treated as a **physics approximation choice**,
not just a tuning knob — the correct value depends on the gate decomposition used
on the hardware. For the heavy_hex_3x3 hardware gates, NOISE_POWER≈25 matches
Julia's chi scaling, suggesting this is the physically appropriate choice.

## MPF Failure Mode Matrix (2026-07-14)

### Why MPF could fail — taxonomy

| # | Failure mode | Assumption broken | Experiment | Wave |
|---|-------------|-------------------|------------|------|
| 1 | **Cutoff kills gamma info** | Small eigenvalues carry γ-dependent signal; relative truncation breaks polynomial assumption | cutoff=1e-3/1e-4/1e-5 | Wave 1 |
| 2 | **Chi cap destroys info** | Same but absolute truncation — late cycles capped | chi_nl=200/300/400 | Wave 4 |
| 3 | **Noise model wrong** | MPF assumes noise interleaved with gates; offline noise breaks O(γ) polynomial | NP=1/3/10/25 | Wave 3 |
| 4 | **Bethe approx breaks** | BP messages don't converge; overlap is approximate not exact | monitor sweeps+delta | All |
| 5 | **Linear regime violated** | γ=1.0–1.8 too far from 0; polynomial expansion not valid | PTM^(1/2)~γ=0.5, PTM^(1/5)~γ=0.2 | Wave 6 (synthetic) |
| 6 | **Finite size / boundary** | Quantum recurrence violates "spreading to infinity" MPF assumption | center=2/13/22 | Wave 2 |
| 7 | **Observable non-polynomial in γ** | OV(γ) has non-analytic dependence; MPF formula not applicable | NP + cutoff combined | Wave 1+3 |
| 8 | **Lightcone boundary** | MPF valid only within lightcone; breaks at late cycles | steps=10/20/30 | Wave 5 |

### Gap: Missing experiment for failure mode 5

Need γ values closer to 0 to test linear regime validity:
- Options: γ={0.5, 0.75, 1.0} or γ={0.2, 0.4, 0.6}
- **Problem**: JSON only has γ={1.0, 1.4, 1.8} — hardware recalibration needed
- **Workaround**: use PTM^α for fractional α (synthetic scaling) — approximate but testable
- Add to future wave: `NOISE_POWER` as a float, not integer

### Scoring rubric

After each experiment, score MPF quality:
- **A**: |error| < 1e-3 for cycles 1-10 — MPF works
- **B**: |error| < 1e-2 cycles 1-5, degrades after — MPF works within lightcone
- **C**: |error| < 5e-2 throughout — marginal
- **D**: |error| > 5e-2 from early cycles — MPF fails

**Current anchor** (co4_ct2_np25_nl400_nn150_s10): **B**

### Key insight (2026-07-14)

MPF breaks at cycle 6 regardless of cutoff (1e-3 vs 1e-4 gives same C(t) but same MPF quality).
This suggests the breakdown is NOT primarily due to truncation of C(t).
The more likely cause: **the gamma-dependent information in the small eigenvalues
is lost by truncation, even when C(t) appears converged** — cutoff and chi_max
are two faces of the same coin. The observable C(t) is insensitive to small
singular values; the MPF extrapolation may not be.

## Exact noise interleaving PTM (2026-07-14) — TODO after experiments

### The problem with NOISE_POWER

`extract_noise_ptm.py` with NOISE_POWER=n raises the noise PTM to the nth power:
```
PTM_eff = M_ε^n
```
This is an approximation — it assumes noise commutes with gates.

### The exact solution

The physically correct model ε(C·ε(B·ε(A·ρ·A†)·B†)·C†) in PTM form is:
```
PTM_eff = M_ε · M_C · M_ε · M_B · M_ε · M_A
```
where M_A, M_B, M_C are the gate PTMs and M_ε is the noise PTM.
All 16×16 real matrices — exact composition via matrix multiplication.

**Key insight**: The result is still a single PTM per bond. CppSim doesn't change
at all — just feed it a different (exact) matrix. Only the Python extractor changes.

### What we need

1. Gate PTMs in Pauli basis for Julia's decomposition:
   - Ry(π/2), Rx(π), Rx(-π), Ry(-π/2) — single-qubit, 4×4
   - Rzz(α·dt) — two-qubit, 16×16
   See Julia's `multi_xx_plus_yy_plus_zz_gate_new` for exact sequence

2. Compose: M_ε interleaved at correct positions in the gate sequence

3. Store result in `heavyhex_noise_data.hpp` — same format, exact physics

### Script: `build_exact_noise_ptm.py` (separate from extract_noise_ptm.py)

Inputs: JSON noise file + graph JSON (for coupling constants)
Output: heavyhex_noise_data.hpp with exact interleaved PTMs

**Status**: TODO — implement after current experiment wave completes.
The NOISE_POWER=25 approximation is good enough for the current experiments.

## Lightcone Frontier MPF Result (2026-07-14) — KEY FINDING

### Hypothesis
MPF extrapolation is accurate at **frontier sites** (where the operator just arrived,
chi not yet saturated) and less accurate in the bulk (where chi is large or saturated).

### Test
Script: `analyze_lightcone_mpf.py`
Experiment: `mpf_co4_ct2_np25_nl400_nn150_s10` (anchor run)
Threshold: w_nl=0.001 to define frontier vs bulk

Sites partitioned per cycle:
- **BULK**: w_nl > threshold (operator established, chi may be saturated)
- **FRONTIER**: 0 < w_nl < threshold (operator just arrived, chi growing)
- **DARK**: w_nl ≈ 0 (operator not arrived, chi=1, exact)

### Result: frontier_better = ✓ YES for ALL 10 cycles

| Cycle | n_bulk | n_front | err_bulk | err_front | Ratio |
|-------|--------|---------|----------|-----------|-------|
| 1  | 1  | 5  | 3.6e-6 | 2.8e-6 | 1.3x |
| 5  | 7  | 19 | 5.5e-4 | 8.2e-5 | 6.7x |
| 8  | 7  | 30 | 5.2e-4 | 2.7e-4 | 1.9x |
| 10 | 10 | 36 | 7.3e-4 | 2.9e-4 | 2.5x |

Even at cycles 8-10 where scalar C(t) MPF was "poor", the **spatial MPF
at the frontier is accurate**. The scalar analysis was too coarse — it averaged
over bulk and frontier, hiding the spatially structured validity.

### Implication
The scalar MPF failure at late cycles is NOT a failure of MPF itself — it's a
consequence of averaging over bulk sites where the reference C_nl is truncated.
At frontier sites, where chi is still small and the reference is exact, MPF works.

**MPF validity is spatially structured, tracking the lightcone frontier.**

### Prediction test
MPF extrapolation at frontier sites does NOT reliably predict the next cycle —
the current noiseless value is a better baseline predictor. This suggests MPF
is a better corrector (noise → noiseless) than a predictor (t → t+1).

### Next steps
- Repeat for other experiments (different NP, center, cutoff)
- Does frontier_better hold for NP=1 (less noise)?
- Plot: frontier error vs chi_n at frontier sites — is it correlated?
- This result may justify the paper claim: MPF is valid within the lightcone

## Hardware architecture observation (2026-07-14)

With NP=25, noisy χ saturates at ~80 regardless of chi_n cap.
GPU 1 (noisy × 3 gammas) uses only ~30% of a 16GB GPU.
This means heterogeneous hardware (large + small GPU) maps naturally to
the noiseless/noisy split — observed on MI60 (32GB) + Radeon VII (16GB).

**Physical reason**: noise damps operator entanglement → χ_n << χ_nl is a
physical property, not a hardware accident. Generalizes to any hardware pair
where GPU_large ≥ ~20GB and GPU_small ≥ ~4GB.

**Not yet confirmed by experiments** — pending full wave results.

## BP consistency for cross-correlation inner products (2026-07-14)

### The mathematical guarantee
In the Pauli basis all operator coefficients are real. Inner products between any two
states are guaranteed real — the Pauli basis is orthonormal. No exceptions.

### Why BP produces complex intermediates
BP approximates the inner product via vertex/edge scalars. For single-state
observables (C(t) = ⟨ψ|ψ₀⟩), vertex scalars are norms — always positive → real.

For cross-correlation ⟨ψᵢ|ψⱼ⟩ with i≠j, BP forms a double-layer network.
Vertex scalars represent cross-correlations between two different states — can be
negative at intermediate BP iterations even though the true fixed point is real.

The `sqrt(vn) < 0` is BP leaving the real domain during convergence — a violation
of the mathematical guarantee. The imaginary parts cancel at convergence (~1e-14)
but this is the algorithm finding the right answer via a wrong path.

### The correct fix (not yet implemented)
A BP formulation for the double-layer network that:
- Maintains real messages throughout convergence
- Keeps vertex scalars non-negative at every iteration
- Different message initialization/update rules — not a sqrt workaround

The `adapt(Complex, ψ)` fix in Igor's code is a numerical band-aid that works but
obscures the real problem. `sqrt(abs(vn))` loses sign information — wrong.

### Implication for CppSim full MPF
When we implement the A matrix: `A[i,j] = ⟨ψ_noisy(εᵢ)|ψ_noisy(εⱼ)⟩`
we will face the same issue. Our current P2P approach uses noiseless messages as
environment — works for ⟨ψ_nl|ψ_noisy⟩ but may not be consistent for cross-noisy.

**Open question**: can the A matrix be computed using one noisy state's messages as
environment for the other, staying real throughout? Or does the double-layer BP
fundamentally require complex arithmetic for cross-correlation inner products?

This is the key research question before implementing full MPF in CppSim.

## Exact interleaved noise PTM — TODO (2026-07-14)

The current `extract_noise_ptm.py` with NOISE_POWER=n is an approximation.
The exact model composes gate and noise PTMs in the correct order:

```
PTM_eff = M_ε · M_C · M_ε · M_B · M_ε · M_A
```

Script skeleton: `build_exact_noise_ptm.py` — needs `julia_gate_sequence_ptm()` implemented.
Required: coupling constants from graph JSON + gate sequence from Julia's decomposition.
Output: `heavyhex_exact_noise_data.hpp` (different name to avoid confusion).
**Status**: TODO — implement when experiments are analyzed and wave runs complete.

## Igor's bp_mpf workload — Float32 fix (2026-07-14)

**Repository**: `AMD-HPC/amd-TensorNetworkQuantumSimulator.jl` branch `feature/gpu-bp-mpf`
**PR #23**: `fix/float32-loopcorrections-clean2` → `feature/gpu-bp-mpf`

**Problem**: `loopcorrections` inner product computes vertex scalars of `BilinearForm`
which can be **strongly negative** (vn=-0.94) for Float32 Pauli-basis states.
`sqrt(vn)` fails with DomainError in Float32.

**Fix**: in `compute_inner()` (`tnqs_noisy_epsilon_evolution.jl`), promote states
to ComplexF32 before calling `inner(alg="loopcorrections")`. States stay Float32
throughout Trotter evolution — only the inner product step is temporarily complex.
Imaginary part of result is ~1e-14 (machine precision).

**Igor's workload structure**:
- Epsilon parametrization: ε={0.1, 0.15, 0.2} (Hamiltonian coupling, not noise scaling)
- Gamma={1.0, 2.0, 3.0} for noise models
- Per cycle: A matrix (3×3 inner products between noisy states) + B vector (noisy vs noiseless)
- MPF coefficients: convex optimization via JuMP+ECOS (more robust than Vandermonde)
- 13 inner product calls per cycle (all currently on CPU — GPU acceleration is next)
- `_binned_pool`: not available in our AMDGPU version, set to `nothing` locally

**Key observation**: z_nl alternates sign (quantum oscillation); noisy values always
positive (noise suppresses oscillation). Negative inner products are genuine physics —
Z operator anti-correlates with initial state during quantum recurrence.

---
*Date: 2026-07-13/14 — Paolo D'Alberto & Claude (Anthropic)*
