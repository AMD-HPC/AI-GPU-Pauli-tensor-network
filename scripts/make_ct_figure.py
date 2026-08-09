#!/usr/bin/env python3
"""
make_ct_figure.py  —  C(t) chi-convergence figures: 4×4 and 5×5

Produces:
    fig_ct_4x4.png  — χ=20,40,50,60,70,80,100  HH (solid) vs Adaptive (dashed)
                      with zoomed inset showing late-time chi fan-out
    fig_ct_5x5.png  — χ=50,60,80  HH (solid) vs Adaptive (dashed), all 30 layers

Usage:
    python make_ct_figure.py             # auto-locate logs
    python make_ct_figure.py /path/CppSim
"""

import os, sys, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from pathlib import Path

DT             = 0.1
TRAP_THRESHOLD = 0.02   # C(t) below this is a BP trap

# ── Log parser ────────────────────────────────────────────────────────────────
def parse_log(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            m = re.match(r'^\s*(\d+)\s+\S+\s+\S+\s+([0-9eE.+-]+)', line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2))))
    return rows

def arrays(data, max_layer=None):
    if not data:
        return np.array([]), np.array([])
    if max_layer is not None:
        data = [(l, c) for l, c in data if l <= max_layer]
    t = np.array([r[0] for r in data], dtype=float) * DT
    c = np.array([r[1] for r in data], dtype=float)
    return t, c

# ── CppSim root ───────────────────────────────────────────────────────────────
def find_root():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    for candidate in [Path(__file__).parent.parent, Path(__file__).parent, Path.cwd()]:
        if (candidate / 'logs').exists():
            return candidate
    sys.exit('Cannot find CppSim logs directory. Pass path as argument.')

# ── Color palette ─────────────────────────────────────────────────────────────
COLOR = {
    20:  '#aaaaaa',
    40:  '#2166ac',
    50:  '#4dac26',
    60:  '#f46d43',
    70:  '#9970ab',
    80:  '#d73027',
    100: '#1a1a1a',
}

# ── Figure A: 4×4 ────────────────────────────────────────────────────────────
def make_4x4(root):
    HH   = root / 'logs/householder_20260517/mi100'
    ADAP = root / 'logs/baseline_20260513/mi100'

    chi_list  = [20, 40, 50, 60, 70, 80, 100]
    hh_files   = {chi: HH   / f'4x4_chi{chi:03d}.log' for chi in chi_list}
    adap_files = {chi: ADAP / f'4x4_chi{chi:03d}.log' for chi in chi_list}

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.set_xlim(0, 2.08)

    # Region shading
    for x0, x1, col, lbl, xpos in [
        (0.0, 1.4, 'steelblue', 'all χ agree',  0.70),
        (1.4, 1.6, 'goldenrod', 'fan-out',       1.50),
        (1.6, 2.1, 'salmon',    'trap zone',     1.85),
    ]:
        ax.axvspan(x0, x1, alpha=0.055, color=col, zorder=0)
        ax.axvline(x1, color=col, ls=':', lw=0.8, alpha=0.45)
        ax.text(xpos, 0.975, lbl, ha='center', va='top',
                transform=ax.get_xaxis_transform(), fontsize=8, color=col, alpha=0.85)

    # Trap storage: (t, c, chi_idx) for jitter
    recoverable_traps = []   # HH recoverable traps
    persistent_traps  = []   # Adaptive persistent traps, with chi index for jitter

    for idx, chi in enumerate(chi_list):
        col = COLOR[chi]

        # Householder (solid)
        t, c = arrays(parse_log(hh_files[chi]))
        if len(t):
            is_trap = c < TRAP_THRESHOLD
            ax.plot(t, np.where(is_trap, np.nan, c), '-',
                    color=col, lw=1.8, label=f'χ={chi}')
            for ti, ci in zip(t[is_trap], c[is_trap]):
                recoverable_traps.append((ti, ci))

        # Adaptive (dashed)
        t, c = arrays(parse_log(adap_files[chi]))
        if len(t):
            is_trap = c < TRAP_THRESHOLD
            ax.plot(t, np.where(is_trap, np.nan, c), '--',
                    color=col, lw=1.2, alpha=0.72)
            if is_trap.any():
                if is_trap[-1]:   # persistent: last layer still trapped
                    for ti, ci in zip(t[is_trap], c[is_trap]):
                        persistent_traps.append((ti, ci, idx))
                else:
                    for ti, ci in zip(t[is_trap], c[is_trap]):
                        recoverable_traps.append((ti, ci))

    # Recoverable traps (open circles)
    if recoverable_traps:
        rx = [p[0] for p in recoverable_traps]
        ry = [p[1] for p in recoverable_traps]
        ax.scatter(rx, ry, marker='o', s=55, zorder=7,
                   facecolors='none', edgecolors='#666666', linewidths=1.4,
                   label='BP trap — recovers')

    # Persistent traps (red ✕) — jitter x by chi index so overlapping points separate
    if persistent_traps:
        n_chi = len(chi_list)
        jitter_scale = 0.012
        px = [p[0] + (p[2] - n_chi / 2) * jitter_scale for p in persistent_traps]
        py = [p[1] for p in persistent_traps]
        ax.scatter(px, py, marker='x', s=90, zorder=7,
                   color='#cc0000', linewidths=2.2,
                   label='BP trap — persistent (adaptive)')

    # ── Inset: linear-scale zoom on chi fan-out ───────────────────────────────
    axins = ax.inset_axes([0.44, 0.42, 0.54, 0.50])

    for chi in chi_list:
        col = COLOR[chi]
        t, c = arrays(parse_log(hh_files[chi]))
        if len(t):
            mask = t >= 1.35
            is_trap = c < TRAP_THRESHOLD
            axins.plot(t[mask], np.where(is_trap, np.nan, c)[mask],
                       '-', color=col, lw=1.5)
        t, c = arrays(parse_log(adap_files[chi]))
        if len(t):
            mask = t >= 1.35
            is_trap = c < TRAP_THRESHOLD
            axins.plot(t[mask], np.where(is_trap, np.nan, c)[mask],
                       '--', color=col, lw=1.0, alpha=0.72)

    # χ labels at rightmost clean point
    labeled = set()
    for chi in sorted(chi_list, reverse=True):
        t, c = arrays(parse_log(hh_files[chi]))
        if not len(t):
            continue
        is_trap = c < TRAP_THRESHOLD
        t_ok = t[~is_trap]; c_ok = c[~is_trap]
        if not len(t_ok) or t_ok[-1] < 1.6:
            continue
        val = c_ok[-1]
        # avoid label collision
        too_close = any(abs(val - v) < 0.0008 for v in labeled)
        y_lbl = val + (0.0008 if too_close else 0)
        axins.annotate(f'χ={chi}', xy=(t_ok[-1], val),
                       xytext=(2.01, y_lbl), fontsize=6.5,
                       color=COLOR[chi], va='center',
                       arrowprops=dict(arrowstyle='-', color=COLOR[chi],
                                       lw=0.5, alpha=0.5))
        labeled.add(val)

    axins.set_xlim(1.35, 2.07)
    axins.set_ylim(0.050, 0.068)
    axins.tick_params(labelsize=7)
    axins.grid(True, alpha=0.25, lw=0.5)
    axins.set_title('χ fan-out  (linear scale)', fontsize=7.5, pad=3)
    ax.indicate_inset_zoom(axins, edgecolor='#888888', lw=0.8, alpha=0.6)

    # ── Legend ────────────────────────────────────────────────────────────────
    proxies = [
        Line2D([0],[0], color='gray', ls='-',  lw=1.8,         label='Householder QR'),
        Line2D([0],[0], color='gray', ls='--', lw=1.2, alpha=0.72, label='Adaptive QR'),
    ]
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=proxies + h, labels=[p.get_label() for p in proxies] + l,
              loc='lower left', fontsize=8, framealpha=0.95,
              ncol=2, columnspacing=0.8, handlelength=2.2)

    ax.set_yscale('log')
    ax.set_ylim(4e-3, 2.0)
    ax.set_xlabel(r'Time  $t = n\,\delta t$  ($\delta t=0.1$)', fontsize=12)
    ax.set_ylabel(r'Spin autocorrelation  $C(t)$', fontsize=12)
    ax.set_title(r'$C(t)$ $\chi$-convergence — $4{\times}4$ grid, MI100', fontsize=12)
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.grid(True, which='both', alpha=0.18, lw=0.5)
    ax.grid(True, which='major', alpha=0.32, lw=0.7)

    plt.tight_layout()
    out = Path(__file__).parent / 'fig_ct_4x4.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close()

# ── Figure B: 5×5 ────────────────────────────────────────────────────────────
def make_5x5(root):
    HH   = root / 'logs/householder_20260517/mi100'
    ADAP = root / 'logs/odd_fix_20260515/mi100'

    hh_files = {
        50: HH   / '5x5_chi050.log',
        60: HH   / '5x5_chi060_n40.log',
        80: HH   / '5x5_chi080_n30.log',
    }
    adap_files = {
        50: ADAP / '5x5_chi050_n30.log',
        60: ADAP / '5x5_chi060_n30.log',
        80: ADAP / '5x5_chi080_n30.log',
    }
    COLOR_5x5 = {50: COLOR[50], 60: COLOR[60], 80: COLOR[80]}
    MAX_LAYER = 30   # cap all curves at 30 layers for uniform x-range

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for chi in [50, 60, 80]:
        col = COLOR_5x5[chi]

        t, c = arrays(parse_log(hh_files[chi]), max_layer=MAX_LAYER)
        if len(t):
            ax.plot(t, c, '-', color=col, lw=2.0, label=f'χ={chi}')

        t, c = arrays(parse_log(adap_files[chi]), max_layer=MAX_LAYER)
        if len(t):
            ax.plot(t, c, '--', color=col, lw=1.3, alpha=0.72)

    # Agreement annotation — centred in axes coordinates
    ax.text(0.50, 0.94,
            r'Householder $\equiv$ Adaptive for all $\chi$'
            '\n'
            r'(C4 symmetry protects BP — solid and dashed overlap)',
            ha='center', va='top', transform=ax.transAxes,
            fontsize=9, color='steelblue', alpha=0.9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='steelblue', alpha=0.6, lw=0.8))

    # Full agreement shading
    ax.axvspan(0, MAX_LAYER * DT * 1.03, alpha=0.05, color='steelblue', zorder=0)

    # Legend
    proxies = [
        Line2D([0],[0], color='gray', ls='-',  lw=2.0,         label='Householder QR'),
        Line2D([0],[0], color='gray', ls='--', lw=1.3, alpha=0.72, label='Adaptive QR'),
    ]
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=proxies + h, labels=[p.get_label() for p in proxies] + l,
              loc='upper right', fontsize=9, framealpha=0.95,
              ncol=1, handlelength=2.2)

    ax.set_yscale('log')
    ax.set_xlim(0, MAX_LAYER * DT * 1.03)
    ax.set_ylim(1e-2, 2.0)
    ax.set_xlabel(r'Time  $t = n\,\delta t$  ($\delta t=0.1$)', fontsize=12)
    ax.set_ylabel(r'Spin autocorrelation  $C(t)$', fontsize=12)
    ax.set_title(r'$C(t)$ $\chi$-convergence — $5{\times}5$ grid, MI100', fontsize=12)
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ax.grid(True, which='both', alpha=0.18, lw=0.5)
    ax.grid(True, which='major', alpha=0.32, lw=0.7)

    plt.tight_layout()
    out = Path(__file__).parent / 'fig_ct_5x5.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close()

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    root = find_root()
    print(f'CppSim root: {root}')
    make_4x4(root)
    make_5x5(root)
