# CppSim Heavy-Hex MPF — Two-GPU Implementation Summary

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

## Hardware architecture observation (2026-07-14)

With NP=25, noisy χ saturates at ~80 regardless of chi_n cap.
GPU 1 (noisy × 3 gammas) uses only ~30% of a 16GB GPU.
This means heterogeneous hardware (large + small GPU) maps naturally to
the noiseless/noisy split — observed on MI60 (32GB) + Radeon VII (16GB).

**Physical reason**: noise damps operator entanglement → χ_n << χ_nl is a
physical property, not a hardware accident. Generalizes to any hardware pair
where GPU_large ≥ ~20GB and GPU_small ≥ ~4GB.

**Not yet confirmed by experiments** — pending full wave results.

---
*Date: 2026-07-13/14 — Paolo D'Alberto & Claude (Anthropic)*
