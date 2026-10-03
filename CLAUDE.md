# CLAUDE.md — AI-GPU-Pauli-tensor-network

## What this project is

A C++/HIP GPU-accelerated belief propagation tensor network simulator for
real-time quantum dynamics in the Heisenberg picture. Computes the non-identity
Pauli weight n(v,t) — the probability that the evolved operator has support at
site v at cycle t — as a space-time observable for quantum noise characterization
and error mitigation.

**Formerly known as CppSim.**

## Philosophy: Physics to Compute

The central design principle is that physical observables should drive the
computational structure, not the other way around. Key consequences:

- n(v,t) is a probability field, not a heuristic — Jaynes (1957) justifies
  cross-entropy as the correct minimization objective
- The lightcone frontier (where χ≈1) is where the simulation is MOST faithful —
  this is where we fit MPF coefficients α(t), not in the interior
- Noise amplification γ=1,2,3 maps directly to ZNE gate-folding — the physics
  connects to hardware experiments without model assumptions
- BP once per full Trotter cycle (after all color classes) makes the computation
  color-order independent — a deliberate design choice

## Build (CRITICAL)

```bash
# Always inside Docker:
bash docker_run.sh
cd /scratch/MPF/CppSim

# Build any binary with PHYS_DIM=4 mandatory:
make PHYS_DIM=4 <binary_name>

# Examples:
make PHYS_DIM=4 main_pauli_sweep        # heavy-hex hardware PTMs
make PHYS_DIM=4 main_pauli_sweep_hh_dep # heavy-hex uniform dep noise
make PHYS_DIM=4 main_pauli_sweep_rect7x7 # rect 7x7 synthetic dep
```

Python environment: `source /scratch/.venv/bin/activate`

## Key binaries and what they do

| Binary | Topology | Noise | Purpose |
|--------|---------|-------|---------|
| `main_pauli_sweep_hh_dep` | Heavy-hex 3×3 | Uniform dep | Paper figures (CLEAN) |
| `main_pauli_sweep_rect7x7` | Rect 7×7 | Synthetic dep | Paper figures (CLEAN) |
| `main_pauli_sweep` | Heavy-hex 3×3 | Hardware PTMs | Research only |
| `main_heavyhex_gamma_mpf_2gpu` | Heavy-hex | Hardware PTMs | 2-GPU inner product MPF |

**IMPORTANT**: For paper figures, NEVER use `--v_hop` flag — it kills distant-site
signal and makes MPF entropy = MPF frontier (degenerate). No `--v_hop` for papers.

## Output format (PAULI logs)

```
PAULI_NL  <step>  <w_0> <w_1> ... <w_{n_sites-1}>
PAULI_N1  <step>  <w_0> ...
PAULI_N2  <step>  <w_0> ...
PAULI_N3  <step>  <w_0> ...
STEP <t>  C_nl <val>  C_n1 <val>  C_n2 <val>  C_n3 <val>  chi <val> ...
```

## Key files (hot path — touch carefully)

- `include/gate.hpp` — `apply_binary_gate`: hot path, GPU kernel
- `include/measures.hpp` — BP, autocorrelation, Pauli weight profile
- `include/heavyhex_physics.hpp` — HeavyHexState, noise PTM interleaving
- `include/trotter.hpp` — topology-agnostic Trotter step
- `include/workspace.hpp` — pre-allocated GPU buffer pool (zero malloc during sim)
- `examples/main_pauli_sweep.cpp` — primary research binary (compile-time topology)

## Physics context (read before touching anything)

**n(v,t)**: Non-identity Pauli weight at site v, cycle t. It is the probability
(from normalization Σ c_P² = 1) that the operator acts non-trivially at v.
Exact at the frontier where χ≈1. This is not an approximation.

**Frontier F(t)**: Sites first crossing threshold ε_th = 1e-4 at cycle t. 
At frontier sites: χ≈1 by construction → simulation exact → ideal fitting point.
On real quantum hardware: no χ at all, n(v,t) measurable via local Pauli measurements.

**Wavemap**: l_γ(v) = arrival delay, L_γ(v) = cross-entropy loss per site.
Both bounded by Lieb-Robinson. Both exact at frontier. Together they characterize
how noise deforms the lightcone, site by site and cycle by cycle.

**α(t) MPF**: Fit one coefficient vector per cycle over the union frontier
F(t) = F_nl(t) ∪ F_γ1(t) ∪ F_γ2(t) ∪ F_γ3(t). Thin spatial shell, χ≈1 everywhere.
Extrapolate inward to all sites. Achieves 55-65% information loss recovery.

**χ and the frontier**: χ saturates in the interior (e.g., χ=50 for rect 7×7).
Truncation error affects the interior. The frontier is protected — χ≈1 there by
construction. The frontier observables are therefore least sensitive to χ choices.
Full χ study in arXiv:2606.30985.

**Near-degeneracy v_γ2 ≈ v_γ3 on heavy-hex**: This is a topological property
of the degree-2/3 graph structure, consistent with a propagation limit set by the
minimum-degree bonds. Not proven purely topological — finer γ sampling would
distinguish. Do not overclaim.

**Information-starved regime**: When noise is heterogeneous per bond (different
rotation angles per bond), the MPF optimizer finds no room to improve because the
LR constraint is already saturated. This is a prediction — demonstrated numerically
but requiring hardware validation for confirmation.

## PTM noise models

The code compiles different PTM sets at build time:
- `HEAVY_HEX_3x3_IDEAL_KICK_EPS010`: uniform J=1 gate (used in ALL binaries)
- `HEAVY_HEX_3x3_DEP_IDEAL_KICK_EPS010`: uniform synthetic depolarizing noise
- `HEAVY_HEX_3x3_NOISY_EPS010`: hardware PTMs (research only, not in paper)
- `RECT_7x7_IDEAL_KICK_EPS010`: rect 7×7 uniform J=1 gate + synthetic dep

Couplings are uniform J=1 in all paper results. The methodology is coupling-agnostic
— device-characterized J_ij can be substituted without modifying the code.

## Analysis scripts

```bash
# MPF entropy analysis (main script):
python3 scripts/plot_mpf_entropy.py \
    --log logs/PAPER_hh_dep_200_10c.log \
    --graph data/graphs/heavy_hex_3x3.json \
    --center 13 --threshold 1e-4 --layout graph \
    --out output.png

# QIML-style 4-row figure:
python3 scripts/plot_mpf_entropy.py ... --qiml --out fig_qiml.png

# Wavefront GIF:
python3 scripts/make_gif.py logs/mi355x_20260519/11x11_chi100_n70_all.log \
    -o wavefront.gif --fps 3
```

## Papers in this repo

| Directory | Title | arXiv |
|-----------|-------|-------|
| `Paper/` | GPU-First Heisenberg-Picture TN Dynamics | 2606.30985 |
| `PaperHeavyHex/` | TN Simulation on Heavy-Hex Lattices | 2607.22827 |
| `PaperFrontier/` | The Pauli Lightcone (long version) | 2608.25254 |
| `PaperFrontier/aaai/` | QIML 2026 short paper | accepted |

## Known implementation gaps (fix before production)

1. **n_MPF ≥ 0 constraint missing**: `fit_alpha_t()` and `fit_alpha_d()` in
   `scripts/plot_mpf_entropy.py` enforce n_MPF ≤ n_nl but NOT n_MPF ≥ 0.
   Current workaround: floor values at zero in post-processing.
   Fix: add `{'type': 'ineq', 'fun': lambda a, A=A: A @ a}` to SLSQP constraints.

2. **b(d) information latency disqualified**: option 3 in `fit_alpha_d()` was
   removed — it violates Lieb-Robinson (n_MPF > n_nl at some sites).
   A per-site LR constraint reformulation is needed to revive it.

## What NOT to do

- Do NOT add `--v_hop` to paper figure runs — kills distant-site signal
- Do NOT use `main_pauli_sweep` (hardware PTMs) for paper figures
- Do NOT suggest Python rewrites of the hot path — it's HIP C++ by design
- Do NOT cite the unpublished Lindblad MPF paper (robertson2025lmpf / lmpf-ref)
  — it was removed from all papers; use arXiv:2609.05024 instead
- Do NOT add `ibm_basquecountry` or device-specific identifiers to papers

## Future directions

1. **Boundary physics**: systematic study of wavefront reflection at lattice
   boundary, phantom waves, frontier collapse at cycle 13 on heavy-hex
2. **Per-site LR constraint**: enforce n_MPF ≥ 0 AND n_MPF ≤ n_nl pointwise
   in the optimizer — enables b(d) latency correction revival
3. **ML noise classifier**: GNN on wavemap features (l_γ(v), L_γ(v)) trained
   on CppSim ground truth, deployed on hardware for noise diagnosis

## Collaboration

Paolo D'Alberto (AMD) · developed in active collaboration with Claude (Anthropic)
