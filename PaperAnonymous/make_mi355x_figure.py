#!/usr/bin/env python3
"""
make_mi_c_figure.py — MI_C per-layer timing at 10x10 chi=100

Produces fig_mi_c_10x10_chi100.png:
  Two panels sharing the x-axis (layer):
    Top:    Trotter time per layer (ms)
    Bottom: BP time per layer (ms)
  Two series: MI_C Householder, MI_C Adaptive
  Saturation shading and annotation.
  MI_C only — this run (109 GB) is not feasible on MI_B or MI_A.

Usage:
    python Paper/make_mi_c_figure.py             # auto-locate logs
    python Paper/make_mi_c_figure.py /path/CppSim
"""

import sys, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

CHI     = 100
COLOR_A = '#1a9850'   # green  — 10×10
COLOR_B = '#d95f02'   # orange — 11×11

SERIES = {
    '10×10 Householder': {
        'log':    'logs/mi355x_20260519/10x10_chi100_n100_all.log',
        'color':  COLOR_A, 'ls': '-',  'lw': 2.0, 'marker': 'o', 'ms': 3.5,
    },
    '10×10 Adaptive': {
        'log':    'logs/mi355x_20260519/10x10_chi100_n100_adap_all.log',
        'color':  COLOR_A, 'ls': '--', 'lw': 1.8, 'marker': '^', 'ms': 4.0,
    },
    '11×11 Householder': {
        'log':    'logs/mi355x_20260519/11x11_chi100_n70_all.log',
        'color':  COLOR_B, 'ls': '-',  'lw': 2.0, 'marker': 'o', 'ms': 3.5,
    },
    '11×11 Adaptive': {
        'log':    'logs/mi355x_20260519/11x11_chi100_n70_adap_all.log',
        'color':  COLOR_B, 'ls': '--', 'lw': 1.8, 'marker': '^', 'ms': 4.0,
    },
}

def parse_log(path):
    """Return list of (layer, chi_min, chi_max, trotter_ms, bp_ms)."""
    p = Path(path)
    if not p.exists():
        print(f'  Missing: {path}')
        return []
    rows = []
    with open(p) as f:
        for line in f:
            m = re.match(r'^\s*(\d+)\s+(\d+)\s+(\d+)\s+\S+\s+\S+\s+(\S+)\s+(\S+)', line)
            if m and int(m.group(1)) > 0:
                try:
                    rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                 float(m.group(4)), float(m.group(5))))
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

    sat_layer_ref = None

    for label, cfg in SERIES.items():
        rows = parse_log(root / cfg['log'])
        if not rows:
            continue

        layers  = np.array([r[0] for r in rows])
        chi_min = np.array([r[1] for r in rows])
        chi_max = np.array([r[2] for r in rows])
        trotter = np.array([r[3] for r in rows])
        bp      = np.array([r[4] for r in rows])

        c = cfg['color']
        kw = dict(color=c, ls=cfg['ls'], lw=cfg['lw'],
                  marker=cfg['marker'], ms=cfg['ms'], markerfacecolor=c,
                  zorder=3, label=label)
        ax_t.plot(layers, trotter, **kw)
        ax_b.plot(layers, bp,      **kw)

        # Saturation: first layer where chi_max >= CHI
        sat = np.where(chi_max >= CHI)[0]
        if len(sat):
            l_sat = layers[sat[0]]
            for ax in (ax_t, ax_b):
                ax.axvline(l_sat, color=c, ls=':', lw=0.8, alpha=0.4)
            if sat_layer_ref is None:
                sat_layer_ref = (l_sat, layers[-1])

    # Saturation shading from Householder reference
    if sat_layer_ref:
        l_sat, l_end = sat_layer_ref
        for ax in (ax_t, ax_b):
            ax.axvspan(0, l_sat - 0.5, alpha=0.05, color='steelblue', zorder=0)
            ax.axvspan(l_sat - 0.5, l_end + 0.5, alpha=0.05, color='goldenrod', zorder=0)
        ax_t.text(l_sat / 2, 0.88, 'ramp-up',
                  ha='center', fontsize=8, color='steelblue', alpha=0.8,
                  transform=ax_t.get_xaxis_transform(), va='top')
        ax_t.text((l_sat + l_end) / 2, 0.88, r'saturation  ($\chi=100$)',
                  ha='center', fontsize=8, color='goldenrod', alpha=0.8,
                  transform=ax_t.get_xaxis_transform(), va='top')

    for ax, ylabel in [(ax_t, 'Trotter  (ms)'), (ax_b, 'BP  (ms)')]:
        ax.set_yscale('log')
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, which='both', alpha=0.18, lw=0.5)
        ax.grid(True, which='major', alpha=0.35, lw=0.7)

    ax_b.set_xlabel('Trotter layer', fontsize=11)
    ax_t.set_title(rf'MI_C: per-layer timing on $10\times10$ lattice at $\chi_{{max}}={CHI}$'
                   '\n(109 GB HBM3e — not feasible on MI_B/MI_A)',
                   fontsize=11)

    proxies = [
        Line2D([0],[0], color=COLOR_A, ls='-',  lw=2, marker='o', ms=5,
               label=r'$10\times10$  Householder'),
        Line2D([0],[0], color=COLOR_A, ls='--', lw=2, marker='^', ms=5,
               label=r'$10\times10$  Adaptive'),
        Line2D([0],[0], color=COLOR_B, ls='-',  lw=2, marker='o', ms=5,
               label=r'$11\times11$  Householder'),
        Line2D([0],[0], color=COLOR_B, ls='--', lw=2, marker='^', ms=5,
               label=r'$11\times11$  Adaptive'),
    ]
    ax_t.legend(handles=proxies, fontsize=9, framealpha=0.95,
                handlelength=2.2, loc='upper left')

    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.09, hspace=0.08)
    out = Path(__file__).parent / 'fig_mi_c_10x10_chi100.png'
    plt.savefig(out, dpi=200)
    print(f'Saved {out}')
    plt.close()

if __name__ == '__main__':
    root = find_root()
    print(f'CppSim root: {root}')
    make_figure(root)
