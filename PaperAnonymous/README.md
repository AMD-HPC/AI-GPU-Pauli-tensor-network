# Paper — Reproducibility Guide

This directory contains the LaTeX source and all figure-generation scripts for:

> *Light-Cone Simulation of 2D Ising Dynamics via Tensor Networks on AMD GPUs*

---

## Compiling the paper

```bash
cd Paper/
pdflatex lightconesimulation.tex
bibtex lightconesimulation
pdflatex lightconesimulation.tex
pdflatex lightconesimulation.tex
```

The main entry point is `lightconesimulation.tex`, which `\input`s:
- `experiments_section.tex` — §6 Numerical Experiments
- `appendix_mi355x.tex` — Appendix: MI355X results

All figures are pre-built PNGs already present in this directory. A clean
recompile from the `.tex` files alone (no Python re-runs needed) produces
the full paper.

---

## Figure inventory

| Figure file | Script | Source logs |
|-------------|--------|-------------|
| `fig_ct.png` | `make_ct_figure.py` | `logs/baseline_20260513/mi100/4x4_chi*.log` |
| `fig_timing_chi.png` | `make_timing_figure.py` | `logs/householder_20260517/mi100/` |
| `fig_breakdown_chi80.png` | `make_breakdown_figure.py` | `logs/householder_20260517/mi100/4x4_chi080.log` |
| `fig_grid_grid.png` | `make_grid_figure.py` | `logs/householder_20260517/mi100/` |
| `fig_lightcone_trap.png` | `make_lightcone_figure.py` | `logs/extended_20260514/mi100/4x4_chi080.log` |
| `fig_lightcone_clean.png` | `make_lightcone_figure.py` | `logs/extended_20260514/mi100/4x4_chi040.log` |
| `fig_lightcone_9x9.png` | `make_lightcone_figure.py` | `logs/odd_fix_20260515/mi100/9x9_chi020_n50.log` |
| `fig_diff_4x4.png` | `make_diff_figure.py` | `logs/extended_20260514/mi100/4x4_chi08*.log` |
| `fig_diff_5x5.png` | `make_diff_figure.py` | `logs/householder_20260517/mi100/5x5_*.log` |
| `fig_lightcone_10x10_chi050.png` | `make_lightcone_figure.py` | `logs/mi355x_20260519/10x10_chi050.log` |
| `fig_mi355x_10x10_chi100.png` | `make_mi355x_figure.py` | `logs/mi355x_20260519/10x10_chi100_n100_all.log` |
| `fig_lightcone_10x10_chi100.png` | `make_mi355x_lightcone.py` | `logs/mi355x_20260519/10x10_chi100_n100_all.log` |
| `fig_lightcone_11x11_chi100.png` | `make_mi355x_lightcone.py` | `logs/mi355x_20260519/11x11_chi100_n70_all.log` |

---

## Regenerating figures

All scripts are in `Paper/` and run from the `CppSim/` root:

```bash
cd CppSim/

# C(t) chi-convergence figure
python3 Paper/make_ct_figure.py

# Per-layer timing vs chi (MI100 HH)
python3 Paper/make_timing_figure.py

# Trotter/BP/Cfn breakdown at chi=80
python3 Paper/make_breakdown_figure.py

# Grid scaling figure (timing vs grid size)
python3 Paper/make_grid_figure.py

# Light cone strip: trap vs clean (4×4)
python3 Paper/make_lightcone_figure.py \
    logs/extended_20260514/mi100/4x4_chi080.log \
    -o Paper/fig_lightcone_trap.png --layers 1 5 10 15 17 18 20 --trap-layer 17

python3 Paper/make_lightcone_figure.py \
    logs/extended_20260514/mi100/4x4_chi040.log \
    -o Paper/fig_lightcone_clean.png --layers 1 5 10 15 17 18 20

# MI355X chi=100 timing figure (4 series: 10×10/11×11 × HH/Adap)
python3 Paper/make_mi355x_figure.py .

# MI355X light cone (full-page, 2×4 heatmap + C(t))
python3 Paper/make_mi355x_lightcone.py \
    logs/mi355x_20260519/10x10_chi100_n100_all.log \
    -o Paper/fig_lightcone_10x10_chi100.png

python3 Paper/make_mi355x_lightcone.py \
    logs/mi355x_20260519/11x11_chi100_n70_all.log \
    -o Paper/fig_lightcone_11x11_chi100.png
```

---

## Log file locations

All raw simulation logs are in `CppSim/logs/`:

| Directory | Hardware | Contents |
|-----------|----------|----------|
| `baseline_20260513/mi100/` | MI100 | Grid scaling, chi convergence (baseline) |
| `baseline_20260514/mi60/` | MI60 | MI60 baseline runs |
| `householder_20260517/mi100/` | MI100 | Full HH re-run (primary paper data) |
| `householder_20260517/mi60/` | MI60 | MI60 HH re-run |
| `extended_20260514/mi100/` | MI100 | 4×4 extended chi range, trap characterization |
| `odd_fix_20260515/mi100/` | MI100 | 5×5, 9×9 with corrected center site |
| `mi100_damped/` | MI100 | BP damping study (α sweep) |
| `mi355x_20260519/` | MI355X | All MI355X results (grid scaling + chi=100) |

---

## Analysis scripts

| Script | Purpose |
|--------|---------|
| `detect_traps.py` | Identify Type-1 BP trap layers from a log file |
| `check_cppsim.py` | Cross-validate C(t) across logs; flag anomalies |
| `parse_logs.py` | Extract timing columns from log files |
| `estimate_memory.py` | Predict GPU VRAM requirements before running |
