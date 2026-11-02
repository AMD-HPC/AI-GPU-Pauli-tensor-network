# AGENTS.md — AI Agent Guide for AI-GPU-Pauli-tensor-network

This file briefs an AI agent on how to navigate, build, and work in this
repository. Read it before touching any code or running any commands.

---

## What this repository is

A GPU-accelerated belief propagation tensor network simulator for real-time
quantum dynamics on AMD GPUs, plus a library of analysis scripts for two
research directions:

1. **MPF (Multi-Product Formula)**: noise characterization and error mitigation
   via the Pauli lightcone wavemap observable `n(v,t)`.
2. **Hamiltonian Tomography**: detection of miscalibrated bonds (coupling
   deviations) in a quantum lattice using wavefront sign-flip patterns.

The simulation runs in C++/HIP. Analysis and plotting are in Python.
All GPU work runs inside a Docker container — do not run GPU binaries directly
on the host.

---

## Environment

```bash
# Always activate the Python venv first
source /scratch/.venv/bin/activate

# Run GPU commands inside Docker
bash /scratch/docker_run.sh   # launches container with GPU access
# or use the existing container if already running
```

Working directory inside container: `/scratch/MPF/CppSim`

GPU availability: AMD MI100 (3× GPUs, ROCm). Use `rocm-smi` to check.
Set `ROCR_VISIBLE_DEVICES=0` (or 1, 2) to target a specific GPU.

---

## Build

```bash
# Always pass PHYS_DIM=4 — mandatory, never omit
make PHYS_DIM=4 main_pauli_sweep_rect7x7   # PRIMARY tomography binary
make PHYS_DIM=4 main_heavyhex_gamma_mpf_2gpu  # MPF binary (2-GPU)
make PHYS_DIM=4 main_lightcone              # lightcone measurement
```

Never run `make` without `PHYS_DIM=4`. The build will silently produce
wrong results without it.

---

## Key directories

| Path | Contents |
|------|----------|
| `include/` | C++ headers — topology-agnostic kernels, PTMs, GPU pools |
| `examples/` | Simulation binaries (source) |
| `scripts/` | Python analysis, detection, plotting |
| `data/graphs/` | Graph JSON files (`rect_7x7.json`, `heavy_hex_3x3.json`) |
| `data/scans/` | Scan plans (`rect_7x7_scan_3gpu.json`) |
| `data/ptm_*.json` | Pre-built PTM matrices — **do not modify** |
| `logs/` | Simulation output logs — large, many are pre-computed |
| `papers/tomography/` | Tomography paper (LaTeX) |
| `REVIEW.md` | Reviewer guide for the tomography paper |

---

## Two main workflows

### 1. Hamiltonian Tomography (bond fault detection)

The primary current research direction. All logs are pre-computed — no GPU
needed to reproduce detection results.

```bash
# Detect broken bonds from existing logs (pure analysis, no GPU)
python3 scripts/bfs_detect.py \
    --logdir logs/mawgm_f64/trial_1 \
    --broken logs/mawgm_f64/trial_1/broken.json \
    --top_k 1

# Run new multi-bond trials (requires GPU inside Docker)
python3 scripts/run_multibond_sweep.py --n_bonds 4 --n_trials 5

# Generate the 3-scan seismic figure
python3 scripts/plot_seismic_3scan.py \
    --logdir logs/mawgm_f64/trial_4 \
    --plan data/scans/rect_7x7_scan_3gpu.json \
    --graph data/graphs/rect_7x7.json \
    --out /tmp/fig.png
```

Key scripts for this workflow:

| Script | Purpose |
|--------|---------|
| `scripts/bfs_detect.py` | BFS + integral log-ratio bond detection |
| `scripts/integral_detect.py` | B(v)/N(v) integrated weight ratio |
| `scripts/run_multibond_sweep.py` | Generate + evaluate multi-bond trials |
| `scripts/run_full_antenna_grid.py` | Full row/col antenna grid scan |
| `scripts/plot_seismic_3scan.py` | 3-scan × 3-cycle seismic figure |
| `scripts/breaker.py` | Inject bond perturbations, write broken.json |
| `scripts/searcher.py` | Parse simulation logs, graph utilities |

Key log directories:

| Directory | Contents |
|-----------|----------|
| `logs/mawgm_f64/` | 10 single-bond trials, float64, pre-computed |
| `logs/mawgm/` | 10 single-bond trials, earlier format |
| `logs/mawgm_v2/` | 10 two-bond trials, pre-computed |
| `logs/mawgm_4b/` | 5 four-bond trials |
| `logs/mawgm_8b/` | 2 eight-bond trials |

Each trial directory contains:
- `scan{1,2,3}_{base,pert}.log` — simulation output
- `broken.json` — ground truth (bonds, coupling values, PTMs)

### 2. MPF noise characterization

Older but published workflow. Requires GPU for new runs.

```bash
# Run MPF simulation (inside Docker, 2-GPU)
./main_heavyhex_gamma_mpf_2gpu 250 10 1e-3 --center=13 2>/dev/null | \
    tee logs/mpf_run.log

# Analyze
python3 scripts/plot_ip_timing.py --ip logs/mpf_run.log
```

---

## Simulation log format

Logs are plain text. Parse with `scripts/searcher.py:parse_pauli_log()`:

```python
from searcher import parse_pauli_log
times, sites, data = parse_pauli_log('logs/mawgm_f64/trial_1/scan1_base.log', 'NL')
# data.shape = (n_cycles, n_sites)
# data[t, v] = n(v, t) — Pauli weight at site v, cycle t
```

State types: `'NL'` (noiseless), `'N1'`/`'N2'`/`'N3'` (noisy γ=1,2,3).

---

## Graph conventions

Sites are 1-indexed. For `rect_7x7`: site `s` has column `(s-1)%7` and
row `6-(s-1)//7` (row 0 = bottom). Load graphs with:

```python
from searcher import load_graph, build_adj, get_bond_sites
g = load_graph('data/graphs/rect_7x7.json')
adj, bond_map, all_sites = build_adj(g)
a, b = get_bond_sites(g['edges'][bond_idx])
```

---

## What NOT to do

- **Never run GPU binaries outside Docker** — they will fail or corrupt state.
- **Never modify `data/ptm_*.json`** — these are pre-built PTM matrices used
  by all simulations. Regenerate with `scripts/build_ptm.py` if needed.
- **Never commit large log files** — logs are local only. The `.gitignore`
  covers most, but check before `git add .`.
- **Never run `make` without `PHYS_DIM=4`**.
- **Do not edit `papers/tomography/tomography.bbl`** — regenerate with
  `bibtex tomography` inside the paper directory.

---

## Paper build

```bash
cd papers/tomography
pdflatex tomography.tex && bibtex tomography && pdflatex tomography.tex && pdflatex tomography.tex
```

The PDF is committed to the repo. Rebuild after any `.tex` change.

---

## Physics conventions

- `n(v,t)` is a **probability** ∈ [0,1]: fraction of Pauli operators with
  non-identity support at site `v` after `t` Trotter cycles.
- Detection threshold `ε_th = 1e-3` — sites with `n_base < ε_th` are WHITE
  (no signal). This matches both the TN truncation floor and `N ≥ 10³` shots.
- Coupling deviation: `δJ = -0.20`, `J = 0.80` in all tomography experiments.
- Trotter step: `ε = 0.10`. Lightcone velocity: `v_hop ≈ 0.45` hops/cycle
  on rect_7x7, `0.253` on heavy_hex_3x3.
- Detection signal: `L(v,t) = -log(n_pert / n_base)`. Positive = RED (shadow),
  negative = BLUE (wall face). Sign flip across a bond = broken bond.

---

## Authors

Paolo D'Alberto (AMD) · developed in collaboration with Claude (Anthropic)
