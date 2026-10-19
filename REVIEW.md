# Review Guide — Red-Blue Hamiltonian Tomography

This document is for reviewers of the paper `papers/tomography/tomography.pdf`.
It explains what the paper claims, what evidence supports each claim, where to
find the code and logs to verify results, and what questions are still open.

---

## What the paper claims

The paper introduces a method to detect miscalibrated bonds (coupling deviations)
in a quantum lattice using Pauli wavefront propagation. The core claims are:

1. **A broken bond creates a causal signature**: the signed cross-entropy
   $\mathcal{L}(v,t) = -\ln(n_{\rm pert}/n_{\rm base})$ shows a red-blue sign
   flip at the two bond endpoints, visible from the first wavefront crossing.

2. **The detection is blind**: the algorithm receives only the two wavemaps
   $n_{\rm base}(v,t)$ and $n_{\rm pert}(v,t)$ and identifies the broken bond
   without knowing the Hamiltonian, perturbation type, or magnitude.

3. **Seismic scanning is sufficient**: 3 center placements (6 simulations total)
   cover all 84 bonds of a 7×7 lattice and achieve:
   - 20/20 single-bond detection (100%)
   - 9/10 two-bond detection (90%), 10/10 combined with antenna grid
   - >90% per-bond rate at up to 8 simultaneous faults

4. **Iterative calibration converges**: fixing detected bonds and re-running
   reduces the multi-bond problem to a sequence of single-bond problems.

---

## What to read first

| Item | Where |
|------|-------|
| Paper | `papers/tomography/tomography.pdf` |
| Results notes | `papers/tomography/mawgm_results_notes.md` |
| Algorithm | `scripts/bfs_detect.py` |
| Scan runner | `scripts/run_multibond_sweep.py` |
| Antenna grid | `scripts/run_full_antenna_grid.py` |

The paper is self-contained. Sections 1--4 introduce the signal vocabulary and
observable. Section 5 shows the single-bond result (82× margin). Section 6
describes the seismic 3-scan approach. The BFS algorithm is in Section 8.
Sections 9--10 cover multi-bond results and measurability thresholds.

---

## Key design decisions — and why

| Decision | Reason |
|----------|--------|
| $\varepsilon_{\rm th} = 10^{-3}$ | Matches both TN truncation floor and $N=10^6$ shot noise floor (§10) |
| 5 simulation cycles | $d_{\rm max}=2$ hops at $v_{\rm hop}\approx0.45$ requires 5 cycles |
| 3 seismic scans | Covers all 84 bonds with complementary center placements |
| Integral $\Lambda(v)$ over pointwise $\mathcal{L}(v,t_0)$ | More robust when signal is weak; captures delay + amplitude |
| Top-k declaration | No assumption on number of faults; declare as many as expected |

---

## How to reproduce the results

All logs are precomputed. No GPU required to reproduce the detection results.

**Single-bond (20/20):**
```bash
# Dataset 1
for t in $(seq 1 10); do
    python3 scripts/bfs_detect.py \
        --logdir logs/mawgm_f64/trial_$t \
        --broken logs/mawgm_f64/trial_$t/broken.json \
        --top_k 1
done

# Dataset 2
for t in $(seq 1 10); do
    python3 scripts/bfs_detect.py \
        --logdir logs/mawgm/trial_$t \
        --broken logs/mawgm/trial_$t/broken.json \
        --top_k 1
done
```

**Two-bond (9/10 seismic):**
```bash
for t in $(seq 1 10); do
    python3 scripts/bfs_detect.py \
        --logdir logs/mawgm_v2/trial_$t \
        --broken logs/mawgm_v2/trial_$t/broken.json \
        --top_k 2
done
```

**Four-bond and eight-bond:**
```bash
python3 scripts/run_multibond_sweep.py --eval_only --n_bonds 4 --n_trials 5
python3 scripts/run_multibond_sweep.py --eval_only --n_bonds 8 --n_trials 2
```

---

## What is the baseline / ground truth

Each trial directory contains `broken.json` which specifies the true broken
bonds, their coupling values, and the full PTM. The detection algorithm never
reads this file — it is used only for scoring after the fact.

The baseline $n_{\rm base}(v,t)$ comes from simulation of the nominal
Hamiltonian. The perturbed $n_{\rm pert}(v,t)$ comes from simulation with
the broken bonds. On real hardware, the baseline would come from simulation
and the perturbed from hardware shots ($N \geq 10^6$ shots required).

---

## Known limitations and open questions

1. **Bond 8 (10↔11) and bond 36** are systematic seismic failures — geometrically
   weak illumination from the 3-scan center placements. Caught by the antenna
   grid in iteration 2.

2. **Antenna grid** (full row/col coverage) works for bond 8 but not yet
   reliably across all trials — aperture approach is a work in progress
   (see Appendix A).

3. **Identity PTM** (perfect switch) was used in early experiments; current
   experiments use $\delta J = -0.20$ (realistic coupling degradation). The
   single-bond 82× margin result (§5, Table 1) uses identity PTM.

4. **Shot budget**: $N = 10^6$ shots required on hardware to match the
   simulation floor $\varepsilon_{\rm th} = 10^{-3}$. At $N = 10^7$,
   minimum detectable $|\delta J| \approx 0.10$.

5. **Heavy-hex geometry**: expected to perform better than 7×7 rectangular
   due to lower connectivity and cleaner wavefront propagation. Not yet tested.

---

## Connection to prior work

This paper is the third in a series using the observable $n(v,t)$:

| Paper | Direction | Observable use |
|-------|-----------|----------------|
| Frontier MPF (arXiv:2608.25254) | Known H → infer noise $\gamma$ | $n(v,t)$ as noise probe |
| Lindblad MPF (in preparation) | Known H → infer Lindblad $\gamma$ | $n(v,t)$ as noise probe |
| **This work** | Known noise baseline → infer $\delta J$ | $n(v,t)$ as geometry probe |

Same observable, same circuits, opposite inference direction.

---

## Questions for the reviewer

- Is the signal vocabulary (WHITE/BLUE/RED, §3) sufficiently defined to be
  unambiguous? Can you identify a bond from Figure 1 without reading the caption?
- Is the BFS algorithm (§8, unnumbered subsection) stated precisely enough to
  be re-implemented independently?
- Is the iterative calibration argument (§9) convincing as a practical protocol,
  or does it require more experimental support?
- Are the measurability thresholds (§10) consistent with current hardware shot
  budgets on IBM or Google processors?
