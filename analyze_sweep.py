#!/usr/bin/env python3
"""
analyze_sweep.py — Parse CppSim sweep logs and plot C_nl + C_mpf with bands

Usage:
    python3 analyze_sweep.py [log_dir]
    python3 analyze_sweep.py logs/sweep_20260723_173437

Plots:
    plot_cnl_bands.png  — C_nl(t) with chi/cutoff band
    plot_cmpf_bands.png — C_mpf(t) with chi/cutoff band
    plot_chi_growth.png — chi growth per cycle
"""

import os, re, sys, glob
import numpy as np

log_dir = sys.argv[1] if len(sys.argv) > 1 else \
    sorted(glob.glob('logs/sweep_*'))[-1]
print(f"Reading: {log_dir}")

# ── Parse all log files ────────────────────────────────────────────────────────
def parse_log(path):
    """Returns dict: step → (C_nl, C_mpf, delta_mpf, chi_nl, chi_n, t_s)"""
    data = {}
    with open(path) as f:
        for line in f:
            # Match step lines
            m = re.match(
                r'\s*(\d+)\s+([+-][\d.e+-]+)\s+([+-][\d.e+-]+)\s+([\d.e+-]+)'  # step C_nl Julia_nl Δnl
                r'(?:\s+[+-][\d.e+-]+\s+[\d.e+-]+){3}'  # 3x C_n |Δ|
                r'\s+([+-][\d.e+-]+)\s+([\d.e+-]+)'      # C_mpf |Δmpf|
                r'\s+(\d+)\s+(\d+)'                       # chi_nl chi_n
                r'(?:\s+\d+\s+[\d.]+)?'                   # optional #sw bp_ms
                r'\s+([\d.]+)', line)
            if m:
                step     = int(m.group(1))
                C_nl     = float(m.group(2))
                C_mpf    = float(m.group(5))
                d_mpf    = float(m.group(6))
                chi_nl   = int(m.group(7))
                chi_n    = int(m.group(8))
                t_s      = float(m.group(9))
                data[step] = (C_nl, C_mpf, d_mpf, chi_nl, chi_n, t_s)
    return data

# Group logs by (chi, cutoff)
logs = {}
mpf_type = 'gamma'
for logfile in sorted(glob.glob(f"{log_dir}/*.log")):
    m = re.search(r'(main_heavyhex_(\w+)_mpf)_chi(\d+)_cut([\de.-]+)', logfile)
    if not m: continue
    mtype = m.group(2)  # gamma or eps
    chi = int(m.group(3))
    cutoff = m.group(4).rstrip('.')  # remove trailing dot if any
    data = parse_log(logfile)
    if data:
        logs[(mtype, chi, cutoff)] = data
        mpf_type = mtype

print(f"Found {len(logs)} experiments, type={mpf_type}")
if not logs:
    print("No data found"); sys.exit(1)

# Get all (chi, cutoff) combos and cycles
all_chi     = sorted(set(chi for (_, chi, _) in logs))
all_cutoffs = sorted(set(cut for (_, _, cut) in logs), key=lambda x: float(x))
all_steps   = sorted(set(s for d in logs.values() for s in d))
N = max(all_steps)
cycles = list(range(1, N+1))

print(f"Chi values: {all_chi}")
print(f"Cutoffs: {all_cutoffs}")
print(f"Cycles: {N}")

# ── Plotting ───────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available")

COLORS = {'1e-2': 'tab:blue', '1e-3': 'tab:orange', '1e-4': 'tab:green',
          '1e-5': 'tab:red'}
CUTOFF_LABELS = {'1e-2': 'cutoff=1e-2', '1e-3': 'cutoff=1e-3',
                 '1e-4': 'cutoff=1e-4', '1e-5': 'cutoff=1e-5'}

def get_series(mtype, chi, cutoff, key_idx):
    """key_idx: 0=C_nl, 1=C_mpf, 2=d_mpf, 3=chi_nl, 4=chi_n"""
    d = logs.get((mtype, chi, cutoff), {})
    return [d[s][key_idx] if s in d else None for s in cycles]

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot |C_nl| and |C_mpf| to see envelope clearly (absolute value removes sign oscillation)
    for ax_idx, (key_idx, ylabel, title_suffix, show_nl_ref) in enumerate([
        (0, '|C_nl(t)|', 'Noiseless |C_nl(t)| — chi/cutoff convergence', False),
        (1, '|C_mpf(t)|', 'MPF |C_mpf(t)| vs |C_nl| reference', True)
    ]):
        ax = axes[ax_idx]

        # Band across ALL chi AND cutoff values
        band_vals = []
        for c_idx in range(len(cycles)):
            vals = []
            for cutoff in all_cutoffs:
                for chi in all_chi:
                    v = get_series(mpf_type, chi, cutoff, key_idx)[c_idx]
                    if v is not None: vals.append(abs(v))
            band_vals.append(vals)
        bx  = [cycles[i] for i,v in enumerate(band_vals) if v]
        blo = [min(v) for v in band_vals if v]
        bhi = [max(v) for v in band_vals if v]
        if blo:
            ax.fill_between(bx, blo, bhi, alpha=0.15, color='gray',
                            label='all chi/cutoff band')

        # Lines per cutoff, largest chi
        for cutoff in all_cutoffs:
            color = COLORS.get(cutoff, 'gray')
            chi_max_local = max(all_chi)
            y = get_series(mpf_type, chi_max_local, cutoff, key_idx)
            yv = [abs(v) for v in y if v is not None]
            xv = [cycles[i] for i,v in enumerate(y) if v is not None]
            if yv:
                ax.plot(xv, yv, '-', color=color, lw=2,
                        label=f'{CUTOFF_LABELS[cutoff]} chi={chi_max_local}')

        # Add C_nl reference on C_mpf plot
        if show_nl_ref:
            y_nl = get_series(mpf_type, max(all_chi), '1e-4', 0)
            yv_nl = [abs(v) for v in y_nl if v is not None]
            xv_nl = [cycles[i] for i,v in enumerate(y_nl) if v is not None]
            if yv_nl:
                ax.plot(xv_nl, yv_nl, 'k--', lw=2, label='|C_nl| (chi=120, cut=1e-4)')

        ax.set_xlabel('Cycle', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(f'{mpf_type.capitalize()}-MPF: {title_suffix}', fontsize=13)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(cycles[::2])

    fig.tight_layout()
    outpath = f"{log_dir}/plot_cnl_cmpf_bands.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Written: {outpath}")

    # Chi growth plot
    fig2, ax2 = plt.subplots(figsize=(10,6))
    for cutoff in all_cutoffs:
        color = COLORS.get(cutoff, 'gray')
        for chi in all_chi:
            y = get_series(mpf_type, chi, cutoff, 3)  # chi_nl
            yv = [v for v in y if v is not None]
            xv = [cycles[i] for i,v in enumerate(y) if v is not None]
            if yv:
                ax2.plot(xv, yv, '-', color=color, alpha=0.6, lw=1.5,
                         label=f'chi_max={chi} cut={cutoff}')
    ax2.set_xlabel('Cycle', fontsize=13)
    ax2.set_ylabel('chi_nl', fontsize=13)
    ax2.set_title(f'{mpf_type.capitalize()}-MPF: Chi growth', fontsize=13)
    ax2.legend(fontsize=7, ncol=3)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{log_dir}/plot_chi_growth.png", dpi=150)
    plt.close(fig2)
    print(f"Written: {log_dir}/plot_chi_growth.png")

# ── Text summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"{'chi':>6}  {'cutoff':>8}  {'cycles':>6}  {'C_nl_final':>12}  {'C_mpf_final':>12}  {'|Δmpf|':>10}  {'chi_final':>10}")
print('='*70)
for cutoff in all_cutoffs:
    for chi in all_chi:
        d = logs.get((mpf_type, chi, cutoff), {})
        if not d: continue
        last_step = max(d.keys())
        C_nl, C_mpf, d_mpf, chi_nl, chi_n, t_s = d[last_step]
        print(f"{chi:>6}  {cutoff:>8}  {last_step:>6}  {C_nl:+12.6f}  {C_mpf:+12.6f}  {d_mpf:10.2e}  {chi_nl:>10}")
