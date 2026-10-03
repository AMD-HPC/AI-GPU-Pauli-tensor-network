#!/usr/bin/env python3
"""
make_grid_figure.py — Grid scaling (timing) figure

Produces fig_grid_grid.png:
  Trotter time per layer (s) vs number of bonds
  Four lines: MI100 Adaptive, MI100 Householder, MI60 Adaptive, MI60 Householder
  Solid = Householder, dashed = Adaptive  (consistent with C(t) figures)
  Blue = MI100, red = MI60
  Linear fits through the origin overlaid as thin dotted lines

Usage:
    python make_grid_figure.py             # auto-locate logs
    python make_grid_figure.py /path/CppSim
"""

import os, sys, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
GRIDS = ['3x3', '4x4', '5x5', '6x6', '7x7', '8x8']
BONDS = np.array([12, 24, 40, 60, 84, 112], dtype=float)
SITES = [9, 16, 25, 36, 49, 64]

def find_root():
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    for candidate in [Path(__file__).parent.parent, Path(__file__).parent, Path.cwd()]:
        if (candidate / 'logs').exists():
            return candidate
    sys.exit('Cannot find CppSim logs directory. Pass path as argument.')

def avg_trotter_ms(log_path):
    """Return mean trotter time (ms) over all layers > 0, or None if file missing."""
    p = Path(log_path)
    if not p.exists():
        return None
    vals = []
    with open(p) as f:
        for line in f:
            m = re.match(r'^\s*(\d+)\s+\d+\s+\d+\s+\S+\s+\S+\s+(\S+)', line)
            if m and int(m.group(1)) > 0:
                try:
                    vals.append(float(m.group(2)))
                except ValueError:
                    pass
    return np.mean(vals) if vals else None

def collect(root):
    series = {
        'MI100 Adaptive':   ('baseline_20260513/mi100', '#2166ac', '--'),
        'MI100 Householder':('householder_20260517/mi100', '#2166ac', '-'),
        'MI60 Adaptive':    ('baseline_20260514/mi60',  '#d73027', '--'),
        'MI60 Householder': ('householder_20260517/mi60',  '#d73027', '-'),
    }
    result = {}
    for label, (subdir, color, ls) in series.items():
        times = []
        for g in GRIDS:
            path = root / 'logs' / subdir / f'{g}_chi050.log'
            t = avg_trotter_ms(path)
            times.append(t)  # None if missing
        result[label] = {'times': times, 'color': color, 'ls': ls}
    return result

def linear_fit(bonds, times_s):
    """Least-squares slope through origin, ignoring None entries."""
    x, y = [], []
    for b, t in zip(bonds, times_s):
        if t is not None:
            x.append(b)
            y.append(t)
    x, y = np.array(x), np.array(y)
    slope = np.dot(x, y) / np.dot(x, x)
    return slope

# ── Plot ──────────────────────────────────────────────────────────────────────
def make_figure(root):
    data = collect(root)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    fit_x = np.linspace(0, 120, 200)

    for label, d in data.items():
        times_ms = d['times']
        times_s  = [t / 1000.0 if t is not None else None for t in times_ms]

        # Plot line (skip None gaps)
        x_plot, y_plot = [], []
        for b, t in zip(BONDS, times_s):
            if t is not None:
                x_plot.append(b)
                y_plot.append(t)

        marker = 's' if 'Adaptive' in label else 'o'
        ax.plot(x_plot, y_plot,
                color=d['color'], ls=d['ls'], lw=2.0,
                marker=marker, ms=5, zorder=3, label=label)

        # Linear fit through origin
        slope = linear_fit(BONDS, times_s)
        ax.plot(fit_x, slope * fit_x,
                color=d['color'], ls=':', lw=0.9, alpha=0.55, zorder=2)

        # Slope annotation at right edge
        y_end = slope * 112
        ax.annotate(f'{slope:.2f} s/bond',
                    xy=(112, y_end),
                    xytext=(114, y_end),
                    fontsize=7, color=d['color'], va='center')

    # Grid labels on x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(BONDS)
    ax2.set_xticklabels(GRIDS, fontsize=8)
    ax2.set_xlabel('Grid', fontsize=10)

    # Legend — one entry per series, explicit color+style, plus fit line
    proxies = [
        Line2D([0],[0], color='#2166ac', ls='-',  lw=2.0, marker='o', ms=5, label='MI100  Householder'),
        Line2D([0],[0], color='#2166ac', ls='--', lw=2.0, marker='s', ms=5, label='MI100  Adaptive'),
        Line2D([0],[0], color='#d73027', ls='-',  lw=2.0, marker='o', ms=5, label='MI60   Householder'),
        Line2D([0],[0], color='#d73027', ls='--', lw=2.0, marker='s', ms=5, label='MI60   Adaptive'),
        Line2D([0],[0], color='gray',    ls=':',  lw=1.0, alpha=0.7,  label='Linear fit (origin)'),
    ]
    ax.legend(handles=proxies, loc='upper left', fontsize=9,
              framealpha=0.95, handlelength=2.8, labelspacing=0.4)

    ax.set_xlabel('Number of bonds  $n_{\\mathrm{bonds}} = 2N(N-1)$', fontsize=12)
    ax.set_ylabel('Trotter time per layer  (s)', fontsize=12)
    ax.set_title(r'Grid scaling at $\chi=50$: Trotter time vs.\ bonds', fontsize=12)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.grid(True, which='major', alpha=0.4, lw=0.7)

    plt.tight_layout()
    out = Path(__file__).parent / 'fig_grid_grid.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close()

if __name__ == '__main__':
    root = find_root()
    print(f'CppSim root: {root}')
    make_figure(root)
