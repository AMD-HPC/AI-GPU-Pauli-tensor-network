# AI-GPU-Pauli-tensor-network (agp-tns)

AI-assisted, GPU-direct tensor network simulator for real-time quantum dynamics
on AMD GPUs, using the Pauli transfer matrix (PTM) basis throughout.
Implements operator evolution in the Heisenberg picture with belief propagation (BP)
on the Bethe lattice, designed on the philosophy of Physics to Compute.

*Formerly known as CppSim.*

## Papers

- **2D Ising lightcone** (`Paper/`): first χ=100 TN simulation of 10×10 and 11×11
  grids on AMD MI355X. arXiv:2606.30985
- **Heavy-hex MPF** (`PaperHeavyHex/`): Heisenberg model on heavy-hex lattice,
  gamma-MPF extrapolation to zero noise. arXiv:2607.22827

## Repository Structure

```
CppSim/
├── include/          # Core library headers (topology-agnostic kernels)
│   ├── types.hpp             # Scalar type, D constant
│   ├── grid.hpp              # Base Grid class (Site, Bond, Gate)
│   ├── heavyhex.hpp          # FlatHeavyHexGrid — 68 sites, 76 bonds
│   ├── heavyhex_physics.hpp  # HeavyHexState, SimNoiseless, SimNoisy
│   ├── gate.hpp              # apply_binary_gate (hot path)
│   ├── measures.hpp          # BP, autocorrelation, Pauli weight profile
│   ├── workspace.hpp         # Pre-allocated GPU buffer pool
│   ├── diag_gauge_ip.hpp     # DG gauge inner product (GPU)
│   └── heavyhex_*_data.hpp   # Pre-built PTMs (ideal, noisy, ergodic)
│
├── examples/         # Simulation binaries (build with Makefile)
│   ├── main_heavyhex_gamma_mpf_2gpu.cpp   # PRIMARY: gamma-MPF, 2-GPU
│   ├── main_heavyhex_gamma_mpf_DG_2gpu.cpp  # DG gauge (research)
│   ├── main_lightcone.cpp                 # Lightcone measurement
│   ├── main_gauge_compare.cpp             # Gauge comparison
│   └── ...                               # Legacy variants
│
├── scripts/          # Analysis and plotting
│   ├── analyze_ip_mpf.py      # MPF extrapolation analysis
│   ├── analyze_lightcone.py   # Lightcone frontier velocity
│   ├── make_fig4_qr_sweep.py  # Figure 4 generation
│   ├── plot_ip_timing.py      # Timing + MPF figure
│   ├── build_ptm.py           # Build PTMs from graph JSON
│   ├── run_lightcone_sweep.sh # Parallel lightcone sweep
│   └── run_dg_sweep.sh        # DG parameter sweep
│
├── test/             # Unit and validation tests
├── data/graphs/      # Graph topologies (heavy_hex_3x3, rect_7x7, rect_8x8)
├── logs/             # Simulation output logs
├── archive/          # Legacy code and checkpoints
├── Paper/            # 2D Ising lightcone paper (arXiv:2606.30985)
├── PaperAnonymous/   # Anonymous submission version
└── PaperHeavyHex/   # Heavy-hex MPF paper (arXiv:2607.22827)
```

## Build

```bash
# Primary MPF binary (gamma-MPF, 2-GPU pipeline)
make PHYS_DIM=4 main_heavyhex_gamma_mpf_2gpu

# Lightcone measurement
make PHYS_DIM=4 main_lightcone

# Gauge comparison tool
make PHYS_DIM=4 main_gauge_compare
```

Requires: ROCm ≥ 6.0, rocBLAS, rocSOLVER, HIP C++17.
Always pass `PHYS_DIM=4` (Pauli basis, D=4).

## Quick Start

### Gamma-MPF: zero-noise extrapolation

```bash
# 2-GPU: GPU 0 Trotter, GPU 1 inner products (overlapped)
./main_heavyhex_gamma_mpf_2gpu 250 10 1e-3 --center=13 2>/dev/null | \
    tee logs/mpf_run.log

# Analyze
python3 scripts/plot_ip_timing.py --ip logs/mpf_run.log
```

### Lightcone calibration

```bash
# Measure operator spreading
./main_lightcone 70 20 1e-3 --center=13 > logs/lightcone.dat

# Analyze: v_hop, heatmap
python3 scripts/analyze_lightcone.py \
    --dat logs/lightcone.dat \
    --graph data/graphs/heavy_hex_3x3.json \
    --center 13 --out lightcone_analysis.png
```

### Delta sweep (Figure 4)

```bash
bash scripts/run_dg_sweep.sh 250 10
python3 scripts/make_fig4_qr_sweep.py --dir logs/dg_sweep_XXXXX
```

## Key Results (heavy-hex 3×3, 68 sites)

| Metric | Value |
|--------|-------|
| Chi convergence | χ=200: \|err\| < 10⁻⁴ (ref χ=430) |
| Lightcone velocity | v_hop = 0.253 ± 0.002 hops/cycle (ε=0.10) |
| Faithful cycles | 12 from site 13 (3 hops to boundary) |
| Speedup vs Julia | ~10× at χ=430 |
| IP overhead at δ=10⁻³ | Zero — hidden under Trotter |
| MPF error (QR, δ=10⁻³) | ~3.8e-2 mean \|ΔC\| over 10 cycles |

### Timing at χ=250, 10 cycles

| δ | Trotter (s) | QR IP (s) | Hidden? |
|---|------------|-----------|---------|
| 10⁻³ | ~10 | ~8 | yes |
| 10⁻⁴ | ~25 | ~60 | no |

Operating point: **δ=10⁻³** balances accuracy and zero IP overhead.

### MPF physics (gamma=1,2 noisy states)

- γ=1: slower decay than C_nl (noise slows quantum spreading)
- γ=2: kills quantum recurrence at cycle 5, drops below C_nl
- From cycle 5: C_nl bracketed between γ=1 (above) and γ=2 (below)
  → MPF operates as **interpolation**, not extrapolation → better conditioning

## Design

**Topology-agnostic**: `apply_binary_gate`, `bp_update`, `autocorrelation_with_bp`
work on any `Grid` subclass. New topology = new grid class + PTMs, no kernel changes.

**Heisenberg picture + PTMs**: operators evolve, not states. PTMs are 16×16 real
matrices acting on the Pauli basis {I,X,Y,Z}⊗².

**2-GPU pipeline**: GPU 0: Trotter (4 states); GPU 1: 9 inner product pairs
(parallel streams). Cycle time = max(t_Trotter, t_IP).

**Lightcone**: n(v,t) Pauli weight per site measures operator spreading.
v_hop = 0.253 hops/cycle is topology+ε dependent only — not χ or δ.
Center site 13 gives 12 faithful cycles before boundary reflection.

## Authors

Paolo D'Alberto (AMD) · developed in collaboration with Claude (Anthropic)
