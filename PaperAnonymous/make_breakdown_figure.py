#!/usr/bin/env python3
"""
make_breakdown_figure.py — Per-layer timing at chi=80, 4x4 grid

Produces fig_breakdown_chi80.png:
  Two panels sharing the x-axis (layer):
    Top:    Trotter time per layer (ms)
    Bottom: BP time per layer (ms)
  Four series: GPU-B HH, GPU-B Adaptive, GPU-A HH, GPU-A Adaptive
  Log-scale y-axis. BP spikes at trapped layers are visible.

Usage:
    python make_breakdown_figure.py             # auto-locate logs
    python make_breakdown_figure.py /path/CppSim
"""

import sys, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

CHI = 80

SERIES = {
    'GPU-B Householder': {
        'log':   'logs/householder_20260517/mi100/4x4_chi080.log',
        'color': '#2166ac', 'ls': '-',  'lw': 2.0,
    },
    'GPU-B Adaptive': {
        'log':   'logs/baseline_20260513/mi100/4x4_chi080.log',
        'color': '#2166ac', 'ls': '--', 'lw': 1.8,
    },
    'GPU-A Householder': {
        'log':   'logs/householder_20260517/mi60/4x4_chi080.log',
        'color': '#d73027', 'ls': '-',  'lw': 2.0,
    },
    'GPU-A Adaptive': {
        'log':   'logs/baseline_20260514/mi60/4x4_chi080.log',
        'color': '#d73027', 'ls': '--', 'lw': 1.8,
    },
}

def parse_log(path):
    """Return list of (layer, chi_max, trotter_ms, bp_ms)."""
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p) as f:
        for line in f:
            m = re.match(r'^\s*(\d+)\s+\d+\s+(\d+)\s+\S+\s+\S+\s+(\S+)\s+(\S+)', line)
            if m and int(m.group(1)) > 0:
                try:
                    rows.append((int(m.group(1)), int(m.group(2)),
                                 float(m.group(3)), float(m.group(4))))
                except ValueError:
                    pass
    return rows

def find_root():
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    for candidate in [Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent, Path.cwd().resolve()]:
        if (candidate / 'logs').exists():
            return candidate
    sys.exit('Cannot find CppSim logs directory.')

def make_figure(root):
    fig, (ax_t, ax_b) = plt.subplots(2, 1, figsize=(9, 7),
                                      sharex=True,
                                      gridspec_kw={'hspace': 0.08})

    for label, cfg in SERIES.items():
        rows = parse_log(root / cfg['log'])
        if not rows:
            print(f'  Missing: {cfg["log"]}')
            continue

        layers   = np.array([r[0] for r in rows])
        chi_max  = np.array([r[1] for r in rows])
        trotter  = np.array([r[2] for r in rows])
        bp       = np.array([r[3] for r in rows])

        kw = dict(color=cfg['color'], ls=cfg['ls'], lw=cfg['lw'],
                  marker='o', ms=3.5, markerfacecolor=cfg['color'],
                  zorder=3, label=label)

        ax_t.plot(layers, trotter, **kw)
        ax_b.plot(layers, bp,      **kw)

        # Mark saturation start (chi_max first reaches CHI)
        sat = np.where(chi_max >= CHI)[0]
        if len(sat):
            l_sat = layers[sat[0]]
            for ax in (ax_t, ax_b):
                ax.axvline(l_sat, color=cfg['color'], ls=':', lw=0.8, alpha=0.4)

    # Shared saturation annotation (use GPU-B HH as reference)
    rows_ref = parse_log(root / SERIES['GPU-B Householder']['log'])
    if rows_ref:
        chi_arr = np.array([r[1] for r in rows_ref])
        layers_arr = np.array([r[0] for r in rows_ref])
        sat = np.where(chi_arr >= CHI)[0]
        if len(sat):
            l_sat = layers_arr[sat[0]]
            for ax in (ax_t, ax_b):
                ax.axvspan(0, l_sat - 0.5, alpha=0.04, color='steelblue', zorder=0)
                ax.axvspan(l_sat - 0.5, layers_arr[-1] + 0.5,
                           alpha=0.04, color='goldenrod', zorder=0)
            ax_t.text(l_sat / 2, 0.88, 'ramp-up',
                      ha='center', fontsize=8, color='steelblue', alpha=0.8,
                      transform=ax_t.get_xaxis_transform(), va='top')
            ax_t.text((l_sat + layers_arr[-1]) / 2, 0.88, 'saturation',
                      ha='center', fontsize=8, color='goldenrod', alpha=0.8,
                      transform=ax_t.get_xaxis_transform(), va='top')

    for ax, ylabel in [(ax_t, 'Trotter  (ms)'), (ax_b, 'BP  (ms)')]:
        ax.set_yscale('log')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, which='both', alpha=0.18, lw=0.5)
        ax.grid(True, which='major', alpha=0.35, lw=0.7)

    ax_b.set_xlabel('Trotter layer', fontsize=11)
    ax_t.set_title(rf'Per-layer timing at $\chi={CHI}$, $4\times4$ grid', fontsize=12)

    # Legend
    proxies = [
        Line2D([0],[0], color='#2166ac', ls='-',  lw=2, marker='o', ms=4,
               label='GPU-B  Householder'),
        Line2D([0],[0], color='#2166ac', ls='--', lw=2, marker='o', ms=4,
               label='GPU-B  Adaptive'),
        Line2D([0],[0], color='#d73027', ls='-',  lw=2, marker='o', ms=4,
               label='GPU-A   Householder'),
        Line2D([0],[0], color='#d73027', ls='--', lw=2, marker='o', ms=4,
               label='GPU-A   Adaptive'),
    ]
    ax_t.legend(handles=proxies, fontsize=9, framealpha=0.95,
                handlelength=2.2, loc='upper left')

    fig.subplots_adjust(left=0.10, right=0.88, top=0.93, bottom=0.09, hspace=0.08)
    out = Path(__file__).parent / 'fig_breakdown_chi80.png'
    plt.savefig(out, dpi=200)
    print(f'Saved {out}')
    plt.close()

if __name__ == '__main__':
    root = find_root()
    print(f'CppSim root: {root}')
    make_figure(root)
