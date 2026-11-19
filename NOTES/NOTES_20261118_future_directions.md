# agp-tns: Where It Stands and Where It Could Go (2026-11-18)

*A review of AI-GPU-Pauli-tensor-network (formerly CppSim) at commit `56e7312`,
written to support a discussion about future directions.*

---

## 1. Summary

agp-tns is a GPU-native tensor-network simulator (C++/HIP) for real-time quantum
dynamics in the Heisenberg picture. It works in the Pauli transfer matrix (PTM) basis
and uses belief propagation (BP) for contraction. Its central observable is the
non-identity Pauli weight **n(v,t)**, a local, per-site, per-cycle probability field.
Local BP contraction keeps n(v,t) tractable where global observables are not: about
O(χ⁴) per site in 2D and O(χ⁶) in 3D, against O(χ⁸) and O(χ¹²) for full contraction.

The code supports the current research well: three arXiv papers, an accepted
QIML 2026 paper, and a tomography paper in preparation. A validated 1024-qubit run
now works on 3 GPUs. **Broadening the scope (new lattices, models, hardware data or
users) is mainly limited by how the code is structured, not by the physics or by
performance.**

---

## 2. Current state

| Area | Status |
|------|--------|
| Core engine | About 25k lines of C++/HIP; buffers allocated once, no malloc in the hot loop; float32 with rocBLAS/rocSOLVER |
| Hardware | AMD MI60, MI100, MI355X |
| Model | Kicked Heisenberg (XXX), Trotter ε = 0.10, uniform J = 1 |
| Topologies | Heavy-hex 3×3; rectangular 7×7 to 32×32; early cubic 3×3×3 and 4×4×4 |
| Noise | Ideal, synthetic depolarizing, per-bond hardware PTMs, γ-amplification for ZNE-style MPF |
| Scale | 32×32 (1024 qubits) on 3 GPUs: 2 for Trotter in parallel, 1 for BP on the boundary slice. Validated against a single GPU (max diff < 6e-05); about 10× faster at χ = 40 |
| Analysis | 126 Python scripts: MPF fitting, wavemaps, bond detection, figures |
| Research lines | (1) MPF error mitigation and noise characterization (published); (2) Hamiltonian tomography, which finds miscalibrated bonds from wavefront sign flips (current) |

**Strengths worth keeping in any new direction**

- The "physics to compute" principle: the observable decides how the computation is structured.
- Frontier sites have χ ≈ 1 and are exact there. This protects the fitting and detection
  signals from truncation error.
- Gate PTMs are pre-composed, and BP runs once per full Trotter cycle, so results do not
  depend on color order.
- Multi-GPU communication is planned ahead of time and kept small. The boundary-slice BP
  costs the same no matter how large the grid gets (always 7×32 = 224 sites).

---

## 3. What limits a change of direction

### 3.1 Each topology is compiled in
Every lattice needs a hand-written grid class (`include/rect_*.hpp`) plus a generated
PTM data header. For example, `rect_32x32_ideal_kick_eps010_data.hpp` is 2000 lines.
The 2048-qubit roadmap scales up by copying and running `sed` on these files.

**Suggestion:** load the graph JSON and PTMs at runtime (the JSON graphs and
`build_ptm.py` already exist). One binary would then run any topology, model or
device calibration. This is the most useful single refactor.

### 3.2 The multi-GPU split is hand-coded per size
Cut rows, boundary bond offsets and normalization factors (17/32, 33/64, ...) are
worked out by hand for each lattice size.

**Suggestion:** a general partition layer: split the graph into k pieces, find the
boundary strip, and compute offsets and normalization automatically. Going to 2048
qubits and up then needs a config change, not a new binary.

### 3.3 Several BP implementations
`bilinear_form_bp`, `_bp_gpu`, `_bp_gpu3`, `_blf_gpu2` and `_bp_cpu2` overlap.

**Suggestion:** merge them into one interface with separate backends. New message
types (rank-k truncation, CPU-resident state for 3D) then plug in without new forks.

### 3.4 One model
Everything uses the kicked XXX model at a fixed ε and uniform J. The method does not
depend on the coupling values, but the code is tied to this model.

**Suggestion:** make the gate and PTM layer take any 2-qubit PTM per bond and any
single-qubit layer. Optionally import circuits (for example from OpenQASM or Qiskit)
and convert them to PTMs.

### 3.5 No regression tests
Validation is done with one-off scripts (`validate_2gpu_full*.py`). Before any large
refactor, a small automated suite should pin known n(v,t) outputs: a few lattices,
χ = 10, 3 steps, CPU or GPU.

---

## 4. Possible directions

| # | Direction | What it means | Builds on | Effort |
|---|-----------|---------------|-----------|--------|
| A | **Hardware diagnostic tool** | Input a device coupling graph and calibration data; output a wavemap and a ranked list of suspect bonds | Tomography workflow, runtime topology (3.1), general model (3.4) | Medium |
| B | **ML noise classifier** | Graph neural network trained on simulated wavemap features (l_γ(v), L_γ(v)), then applied to hardware data | Existing logs and wavemap features; the simulator generates ground truth | Medium |
| C | **3D and very large scale** | CPU-resident state with GPU dispatch; rank-k truncated BP messages; 2048 qubits and beyond across nodes | `NOTES/future_3d_architecture.md`, 2048-qubit roadmap, partition layer (3.2) | High |
| D | **General open-source library** | Clean C++ API with Python bindings, runtime topology, pip/conda install, documentation, tests | All of section 3 | High |
| E | **Better MPF and LR constraints** | Enforce 0 ≤ n_MPF ≤ n_nl at every site; bring back the b(d) latency correction | `scripts/plot_mpf_entropy.py` (known gaps listed in `CLAUDE.md`) | Low |
| F | **Boundary physics** | Systematic study of wavefront reflection at the lattice edge, "phantom" waves, and the frontier collapse at cycle 13 on heavy-hex | Existing logs; `COLLAPSE_REGISTRY.md`; collapse-diagnosis notes | Low–Medium |

**Suggested order:** E (quick win) → 3.5 tests → 3.1 runtime topology → then A or B,
whichever is closer to available hardware partners. C and D become much cheaper once
3.1–3.3 are done.

---

## 5. Repository housekeeping (low effort)

- The 14 untracked `NOTES_2026*.md` files in the top folder are identical copies of
  files in `NOTES/`. Delete them.
- `Paper/`, `PaperAnonymous/`, `PaperFrontier/` and `PaperHeavyHex/` look superseded by `papers/`.
- Compiled binaries (`main_heavyhex_*`, `gpu_check`, `gpu_used_check`) sit in the top
  folder. Move them to a `build/` folder covered by `.gitignore`.
- Out-of-date documentation:
  - `PHILOSOPHY.md` says the GPUs exchange only scalars; the 3-GPU design now exchanges
    boundary messages and tensors.
  - `CLAUDE.md` describes MPF as the main line; `AGENTS.md` names tomography as primary.
    Align the two.
- Move the root-level `PLAN_*.md` and `EXPERIMENT_PLAN_*.md` files into `NOTES/` or `docs/`.

---

*Paolo D'Alberto (AMD) · prepared with Claude (Anthropic)*
