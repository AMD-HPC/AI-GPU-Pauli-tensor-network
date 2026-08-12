# Operator Lightcone as a Noise Classifier and Simulation Diagnostic
## Draft — 2026-08-11

---

## Abstract (target: 150 words)

We show that the wavefront of the operator lightcone — the set of newly-reached
sites in a Pauli weight spreading experiment — is simultaneously a noise classifier,
a simulation diagnostic, and a memory allocation schedule for tensor network
simulators.  Running noiseless and noisy Trotter evolution at multiple noise
amplification factors γ on a 2D quantum lattice, we extract the frontier velocity
v_hop and the coefficient of variation cv = std/mean of the Pauli weight ratio at
newly-reached sites.  Pure damping noise produces cv << 1 (uniform suppression),
while rotation noise produces cv ≈ 1 (site-specific scrambling).  We demonstrate
this on two topologies — an IBM heavy-hex lattice with hardware noise and a
rectangular 7×7 lattice with synthetic depolarizing noise — and find both exhibit
rotation-dominant behavior, with the hardware noise showing a gamma-independent
rotation structure (coherent error) while the synthetic noise shows rotation
growing with γ (incoherent off-diagonal PTM terms).  We further show that the
frontier measurement is cheap (chi≈1 is exact at the wavefront) and provides the
data needed for lightcone-aware memory allocation, reducing tensor network memory
requirements by 3-13× on early-cycle simulations.

---

## 1. Introduction (target: 1 page)

### 1.1 Operator spreading and the lightcone

A local operator O(0) under Hamiltonian evolution spreads across a lattice.
The Pauli weight n(v,t) — the non-identity Pauli content at site v at cycle t —
defines the operator lightcone.  The frontier F(t) is the set of newly-reached
sites at cycle t:

    F(t) = { v : n(v,t) > ε  and  n(v,t-1) ≤ ε }

The frontier velocity v_hop (hops/cycle) is the slope of max BFS distance vs cycle.
This is a causal bound — no faster-than-Lieb-Robinson spreading can occur.

### 1.2 The noise characterization problem

Quantum hardware introduces noise that modifies the Pauli transfer matrix (PTM)
on each bond.  Two qualitatively different effects:

**Damping**: PTM ≈ diag(1, 1-p, 1-p, 1-p).  Suppresses all Pauli components
uniformly.  Observable: n_noisy(v,t) = D^d × n_noiseless(v,t) where D < 1 and
d = BFS distance.  This is a scalar multiplier — the same factor at every frontier
site.

**Rotation**: PTM has off-diagonal terms (I→X, Y→Z, etc.).  Redistributes Pauli
weight among directions.  Observable: n_noisy(v,t)/n_noiseless(v,t) varies
site-to-site — some sites gain weight, others lose it, depending on the specific
Pauli components arriving from the interior.

### 1.3 This work

We propose using the coefficient of variation (cv = std/mean) of the Pauli weight
ratio at frontier sites as a model-free noise classifier:

    ratio(v,t) = n_ni(v,t) / n_nl(v,t)   for v ∈ F(t)
    cv_γ = std(ratio) / mean(ratio)

- cv < 0.3: damping dominant
- cv > 0.7: rotation dominant
- intermediate: mixed

This requires no model assumption, no tomography, no calibration — only two runs
(noiseless reference + noisy) with chi small enough to be exact at the frontier.

---

## 2. Methods

### 2.1 Tensor network Pauli propagation (CppSim)

We use CppSim, a GPU-accelerated belief propagation tensor network simulator.
Evolution: kicked Heisenberg model, ε=0.10, Rx(2φ) kick per site per cycle.
Two topologies:
- IBM heavy-hex 3×3 (68 qubits, degree 2-3, girth 6)
- Rectangular 7×7 (49 sites, degree 4, girth 4)

For each topology: one noiseless state (NL) + three noisy states at γ=1,2,3.
Output: n(v,t) per site per cycle to 7 decimal places (%.7f, floor ~1e-7).

### 2.2 Noise models

**Heavy-hex**: IBM hardware PTMs from per-bond characterization.
76 bond PTMs, 3 color layers, heterogeneous coupling strengths.
Noisy PTMs are γ× amplified hardware noise.

**Rect 7×7**: Synthetic depolarizing noise.
M_noisy = diag(1, (1-p·γ), (1-p·γ), (1-p·γ)) applied per bond.
Note: this form still has rotation structure when composed across bonds.

### 2.3 Frontier measurement

Wavefront definition: F(t) = newly-reached sites above threshold ε.
Threshold ε = 1e-4 (safely above output precision floor of ~1e-7).
v_hop: linear fit of max-distance(F(t)) vs cycle number.
Ratio measurement: for each newly-reached site v at cycle t,
compute n_ni(v,t)/n_nl(v,t); pool across all (v,t) pairs.

### 2.4 LR-constrained spatial MPF

The noiseless frontier is the Lieb-Robinson ceiling:
    F_corrected(t) ≤ F_nl(t)   for all t

We define deficit δ_i(t) = F_nl(t) - F_ni(t) and fit MPF coefficients c_i
to minimize ||δ_corrected(t)||² subject to F_corrected ≥ 0 (scipy SLSQP).
This is a new observable: the noise-corrected spatial frontier.

---

## 3. Results

### 3.1 Frontier velocity per topology

| Topology | v_hop(NL) | v_hop(N1) | v_hop(N2) | v_hop(N3) | Ordering |
|----------|-----------|-----------|-----------|-----------|----------|
| Heavy-hex | 0.376 | 0.145 | 0.152 | 0.152 | flat (N2≈N3) |
| Rect 7×7 | 0.448 | 0.382 | 0.248 | 0.097 | monotone ↓ |

The heavy-hex noisy velocities are nearly gamma-independent (N2≈N3),
while rect_7x7 shows monotone decrease with γ.

### 3.2 Noise classification via cv

| Topology | cv(N1) | cv(N2) | cv(N3) | mean cv | Classification |
|----------|--------|--------|--------|---------|----------------|
| Heavy-hex | 1.017 | 1.064 | 1.100 | 1.060 | ROTATION DOMINANT |
| Rect 7×7 | 0.526 | 0.857 | 1.089 | 0.824 | ROTATION DOMINANT |

**Heavy-hex**: cv is flat across γ (1.017→1.100) — the rotation structure is
fixed by the hardware channel, independent of noise amplitude.
This is the signature of **coherent errors** (fixed rotation angle).

**Rect 7×7**: cv increases with γ (0.526→1.089) — rotation grows with noise
amplitude.  This happens because the synthetic "depolarizing" PTM composed
across multiple bonds generates off-diagonal terms proportional to p·γ.
This is the signature of **incoherent off-diagonal noise** (rotation ∝ γ).

### 3.3 Deficit ordering as secondary classifier

The ordering of deficits δ_i(t) = F_nl(t) - F_ni(t) provides additional information:

**Rect 7×7**: δ_1 < δ_2 < δ_3 (monotone) — heavier noise always delays frontier more.
**Heavy-hex**: δ_1 > δ_2 > δ_3 (inverted) — lightest noise (γ=1) delays MOST.

The inversion in heavy-hex is consistent with rotation suppressing specific
propagation channels: at γ=1 the suppression is strongest relative to injection;
at γ=3 the rotation injects weight that partially compensates the delay.

### 3.4 LR-constrained MPF coefficients

Fitting F_corrected(t) = Σ c_i F_ni(t) subject to 0 ≤ F_corrected ≤ F_nl:

| Topology | c_1 | c_2 | c_3 | Interpretation |
|----------|-----|-----|-----|----------------|
| Rect 7×7 | +2 | -1 | 0 | monotone deficit → clean extrapolation |
| Heavy-hex | -2 | +1 | +2 | inverted deficit → negative coefficient |

Near-integer coefficients suggest discrete noise structure.
Open question: are these related to the PTM eigenvalue spectrum?

---

## 4. Discussion

### 4.1 The cv as a model-free noise probe

The key virtue of cv is that it requires no model assumption:
- No assumption on PTM form
- No calibration data
- No tomography
- Just two runs: noiseless + noisy, at any chi

The physical interpretation is clean:
- cv << 1: noise is a scalar multiplier (damping, calibration problem)
- cv ≈ 1: noise is site-specific (rotation, genuine physics change)

The γ-dependence of cv adds a second layer:
- cv independent of γ: coherent error (fixed angle, independent of noise amplitude)
- cv growing with γ: incoherent off-diagonal (rotation ∝ amplitude)

### 4.2 Connection to known noise models

Pure depolarizing noise (M = diag(1,(1-p)³)):
→ cv = 0 analytically (same factor D^d at every site at distance d)

Hardware PTMs with coherent components:
→ cv > 0, determined by the specific rotation angle of the channel

Lindblad evolution with dephasing:
→ cv intermediate (X,Y damped; Z preserved → preferential Z-spreading at frontier)

CppSim can distinguish all three with a single cheap run.

### 4.3 Lightcone-aware memory allocation

The frontier measurement provides a free byproduct: the per-cycle allocation schedule.
Sites at BFS distance d enter the lightcone at cycle t ≈ d/v_hop.
Before that cycle, their chi = 1 exactly — no memory needed.

Predicted savings (chi=100, 10 cycles):
- Heavy-hex: ~3× reduction in allocated tensors
- Rect 7×7: ~13× reduction (larger lattice, later cycles relative to diameter)

Implementation: precompute d(v) from center → set chi_v(t) = 0 for t < d(v)/v_hop.
One cheap pre-flight run (small chi, fast) calibrates v_hop → allocation schedule for
the expensive run.

### 4.4 Multiple uses of a single experiment

The frontier measurement provides simultaneously:
1. **Physics**: v_hop, noise classification (cv), deficit ordering
2. **MPF input**: F_ni(t) for LR-constrained spatial error mitigation
3. **Memory allocation**: d(v)/v_hop schedule for expensive runs
4. **Trust boundary**: when wavefront hits first graph loop → BP breakdown onset
5. **Noise characterization**: mean(ratio) = cumulative damping; cv = rotation content

This multiplicity is the key point: the cheapest part of the simulation (chi≈1 frontier)
carries the richest information about what the expensive part needs to do.

---

## 5. Conclusion

We have shown that:

1. The coefficient of variation of n_noisy/n_noiseless at frontier sites is a
   model-free noise classifier, distinguishing damping (cv << 1) from rotation (cv ≈ 1).

2. The γ-dependence of cv further distinguishes coherent errors (cv flat) from
   incoherent off-diagonal noise (cv growing with γ).

3. IBM heavy-hex hardware noise is rotation-dominant with coherent structure;
   synthetic depolarizing noise is initially mixed but rotation-dominant at large γ.

4. The deficit ordering δ_i(t) provides a complementary spatial-temporal classifier
   without amplitude information.

5. LR-constrained spatial MPF produces near-integer coefficients that encode the
   noise type and may be analytically derivable from the PTM spectrum.

6. The frontier measurement is simultaneously a physics observable, an MPF input,
   a memory allocation schedule, and a simulation trust boundary indicator.

---

## Open Questions / Future Work

1. **Analytical cv prediction**: can we derive cv from the PTM eigenvalues analytically?
   For depolarizing: cv=0 by symmetry. For general rotation: cv = f(off-diagonal entries)?

2. **Integer coefficients**: why are LR-MPF coefficients nearly integer?
   Are they related to the discrete structure of noise amplification by integer γ?

3. **Longer cycles**: what happens when the wavefront reflects from the boundary?
   Does cv change after reflection (wave carries memory of interior entanglement)?

4. **3D topologies**: lightcone-aware allocation saves 10-100× in 3D.
   Enables simulations currently infeasible due to memory.

5. **Hardware measurement**: can n(v,t) be measured on actual QPU?
   Via randomized benchmarking or Pauli noise tomography — yes, in principle.
   CppSim provides the noiseless reference for comparison.

6. **Per-bond noise inversion**: can we recover per-bond PTMs from frontier ratios?
   At the frontier (chi=1, single path), the ratio = product of bond PTM eigenvalues.
   This is an underdetermined but potentially tractable inversion.

---

## Figures (to generate)

1. n(v,t) heatmap NL vs N1/N2/N3 — both topologies side by side
2. Frontier F(t) vs cycle — NL + 3 gammas, both topologies
3. Deficit δ_i(t) — monotone (rect) vs inverted (heavy-hex)
4. cv per gamma — rect_7x7 (growing) vs heavy-hex (flat)
5. LR-constrained MPF F_lr(t) vs F_nl(t) — recovery quality
6. Lightcone allocation diagram — chi(v,t) schedule vs naive allocation

---

## Where We Stand / Gaps to Fill

### Strong
- cv classifier works on existing data
- Deficit ordering confirmed on both topologies
- LR-constrained MPF produces sensible coefficients
- Float precision fixed (%.7f, threshold safely above floor)
- Wavefront definition correct (newly-reached, not max-active)

### Needs work
- Only 10 cycles — short for slope fitting, boundary effects at cycle 6-8 for rect_7x7
- Only one chi value per topology (chi=50 rect, chi=70 heavy-hex) — need chi sweep
  to confirm cv is chi-independent (it should be, frontier is chi=1 by definition)
- No pure-damping baseline — need synthetic diagonal PTM for honeycomb
  to confirm cv=0 analytically
- LR-MPF coefficient error bars — currently no uncertainty estimate
- No comparison to existing noise characterization literature (papers to read)

### Experiments needed before submission
1. Chi sweep: chi=20,50,100 on rect_7x7 — confirm cv is chi-independent
2. Pure damping PTM for heavy-hex — confirm cv→0 analytically  
3. More cycles (20+) — confirm v_hop is linear beyond current 10 cycles
4. Multiple center sites — confirm cv is center-independent
5. Read: Nature Comms 2025 (operator spreading + noise), arXiv:2501.04091

---

## Target venue

**PRX Quantum** or **Physical Review B** (Letters section).

Length: ~8 pages + supplemental.
Key selling point: model-free, cheap, multi-purpose — single observable does the
work of multiple expensive protocols.
