# From Simulated to Measured n(v,t): Implications for Tomography (2026-11-18)

*Context: after the 1024-qubit 3-GPU milestone, the simulator is no longer the
bottleneck. To improve, tomography now needs a practical setup in which n(v,t)
is **estimated from hardware shots**, not computed. This note records what changes.
Written for the next session (human or agent) picking this up.*

---

## 1. n(v,t) on hardware is an OTOC, not a local expectation value

Average over the single-site Paulis at v. For a non-identity P_v, σ P σ cancels in
the average over σ ∈ {I,X,Y,Z}; for P_v = I it is kept. This gives an exact identity
(O(t) = U† Z_c U, normalized so that Σ c_P² = 1):

```
n(v,t) = (3 - F_X - F_Y - F_Z) / 4
F_σ    = 2^{-N} Tr[ O(t) σ_v O(t) σ_v ]      (infinite-temperature OTOC, butterfly σ at v)
```

Ways to measure it:
- **Echo protocol**: random initial state, then U, then σ_v, then U†, then measure.
  Average over initial states to get the infinite-temperature trace. This is the same
  family as Google's operator-scrambling and "quantum echoes" OTOC experiments.
- **Randomized measurements**: can estimate OTOCs for *all* v from one dataset.
  This saves roughly a factor of n_sites in circuits.

**Action:** AGENTS.md and CLAUDE.md say "n(v,t) measurable via local Pauli
measurements". That is imprecise and should be reworded before reviewers see it.
Per-site measurement *after the echo* is local; the observable itself is an OTOC.

---

## 2. Consequences of estimating instead of computing

### 2.1 Shot noise works against the frontier
When F ≈ 1, Var(F̂) ≈ (1 − F²)/N ≈ 2(1 − F)/N, which gives

```
std(n̂) ≈ sqrt(n / N)      relative error ≈ 1 / sqrt(n · N)
```

The frontier (χ ≈ 1, where the simulation is exact) is where n is **smallest**, so
the hardware SNR is **worst** there. This is a tension between simulation and
measurement. The best fitting and detection region is probably **a shell just
behind the frontier**, not F(t) itself.

### 2.2 The threshold is set by shots, not by χ truncation
- N = 10³ at n = 10⁻³ gives relative error of about 100%, so the signal is barely visible.
- Resolving L = −log(n_pert/n_base) to about 10% at n = 10⁻³ needs about 10⁵ shots
  **for each site, cycle, σ and scan**, for both base and perturbed runs.
- AGENTS.md treats `ε_th = 1e-3 ↔ N ≥ 10³` as equivalent. That is optimistic: it
  covers detection, not quantification.

### 2.3 Echo doubles the noise
Running U then U† roughly doubles the circuit's noise exposure, and the measured F
decays from decoherence alone. Normalize by a reference echo with no butterfly
(σ_v = I), or by an independently measured fidelity. Speculative: this doubling acts
as a built-in γ ≈ 2 amplification, which might connect tomography back to the MPF
γ-extrapolation machinery.

### 2.4 SPAM and readout bias matter most where n is small
Readout and state-preparation bias are a fixed offset, so they dominate small-n
sites. Readout error mitigation is required, not optional.

### 2.5 Detection becomes a statistical test
- The sign flip of L across a bond becomes a significance test (z-score or likelihood ratio).
- 32×32 has about 2000 bonds, so multiple testing needs **false-discovery control**
  (for example Benjamini–Hochberg).

### 2.6 The simulator changes role
- **Forward model**: fit δJ (per bond) to the measured n̂ by maximum likelihood,
  using simulated templates.
- **Experiment designer**: choose antennas, scan geometry, cycles, and how to allocate
  a fixed shot budget, using the 1024-qubit capability to search designs.

### 2.7 Circuit budget
Number of circuits ≈ sites × cycles × 3 (σ) × scans × 2 (base/pert) × random initial
states. Randomized measurements, or a reduced set of butterflies (for example only Z,
if the symmetry of the XXX model allows it; this needs checking), could cut this a lot.

---

## 3. Proposed first step (no hardware, no GPU)

**A shot-noise injection study on the existing ground-truth logs.**

Inputs: `logs/mawgm_f64/` (10 single-bond), `logs/mawgm_v2/` (2-bond),
`logs/mawgm_4b/`, `logs/mawgm_8b/`, and the 32×32 seismic logs.

1. For each simulated n(v,t), generate F_σ = 1 − (4/3)·n (isotropic split; check this
   assumption against per-σ data if available). Sample F̂_σ from N ±1 outcomes,
   then rebuild n̂.
2. Optional layers: echo decay F → f·F (with f measured from the no-butterfly
   reference), and readout bias.
3. Rerun `scripts/bfs_detect.py` (and `integral_detect.py`) on the noisy n̂ for
   N ∈ {10³, 10⁴, 10⁵, 10⁶} with many repetitions.
4. Plot detection rate and rank against N, |δJ| and the number of broken bonds.
   Compare a frontier-only fit with a fit on the shell behind the frontier.

Deliverable: a **shots-needed vs coupling-error** curve. This is the first number
an experimental group will ask for, and the test of whether the method holds up
on hardware-like data.

Then: replace the log-ratio sign-flip with a likelihood-ratio test with FDR control,
and fit δJ directly by maximum likelihood with simulated templates.

---

## 4. Open questions

- Does the XXX symmetry make F_X = F_Y = F_Z for the evolved Z_c? If so, one butterfly suffices.
- Which is better under shot noise: 3 cycles with high N, or many cycles with low N?
- Can randomized measurements get all-v OTOCs at the needed precision for a reasonable budget?
- Does the doubled echo noise really map onto MPF γ = 2? This could unify the two research lines.

---

*Paolo D'Alberto (AMD) · notes with Claude (Anthropic)*
