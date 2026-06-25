#!/usr/bin/env python3
"""
make_timing_figure.py — Chi scaling of Trotter and BP timing

Produces fig_timing_chi.png:
  Two-panel figure (log-log):
    Left:  Trotter time per layer vs chi
    Right: BP time per layer vs chi
  Four series: MI_B HH, MI_B Adaptive, MI_A HH, MI_A Adaptive
  Two grids: 4x4 (solid markers) and 5x5 (hollow markers)
  Power-law fits chi^alpha overlaid as dotted lines

Usage:
    python make_timing_figure.py             # auto-locate logs
    python make_timing_figure.py /path/CppSim
"""

import sys, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ── Log parser ─────────────────────────────────────────────────────────────────
def avg_times(log_path):
    """Return (avg_trotter_ms, avg_bp_ms) over layers>0, or (None,None)."""
    p = Path(log_path)
    if not p.exists():
        return None, None
    trotter, bp = [], []
    with open(p) as f:
        for line in f:
            m = re.match(r'^\s*(\d+)\s+\d+\s+\d+\s+\S+\s+\S+\s+(\S+)\s+(\S+)', line)
            if m and int(m.group(1)) > 0:
                try:
                    trotter.append(float(m.group(2)))
                    bp.append(float(m.group(3)))
                except ValueError:
                    pass
    if not trotter:
        return None, None
    return np.mean(trotter), np.mean(bp)

def find_root():
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    for candidate in [Path(__file__).resolve().parent.parent, Path(__file__).resolve().parent, Path.cwd().resolve()]:
        if (candidate / 'logs').exists():
            return candidate
    sys.exit('Cannot find CppSim logs directory. Pass path as argument.')

# ── Data collection ────────────────────────────────────────────────────────────
# Only canonical files: 4x4_chi{NNN}.log  (no _n40, _damped, etc.)
SERIES = {
    'MI_B Householder': {
        'color': '#2166ac', 'ls': '-',
        '4x4': ('householder_20260517/mi100', [20,40,50,60,70,80,90,100]),
        '5x5': ('householder_20260517/mi100', [20,40,50,60,80,100]),
    },
    'MI_B Adaptive': {
        'color': '#2166ac', 'ls': '--',
        '4x4': ('baseline_20260513/mi100', [20,40,50,60,70,80,100]),
        '5x5': ('odd_fix_20260515/mi100',  [50,60,80]),
    },
    'MI_A Householder': {
        'color': '#d73027', 'ls': '-',
        '4x4': ('householder_20260517/mi60', [20,40,50,60,70,80]),
        '5x5': ('householder_20260517/mi60', [50,80]),
    },
    'MI_A Adaptive': {
        'color': '#d73027', 'ls': '--',
        '4x4': ('baseline_20260514/mi60', [20,50,80]),
        '5x5': (None, []),
    },
}

GRID_MARKER = {'4x4': 'o', '5x5': 's'}   # filled=4x4, square=5x5
GRID_FILL   = {'4x4': True, '5x5': False}

def collect(root):
    out = {}
    for label, cfg in SERIES.items():
        out[label] = {}
        for grid in ['4x4', '5x5']:
            subdir, chis = cfg[grid]
            if subdir is None:
                out[label][grid] = {'chi': [], 'trotter': [], 'bp': []}
                continue
            chi_list, trot_list, bp_list = [], [], []
            for chi in chis:
                fname = f'{grid}_chi{chi:03d}.log'
                t, b = avg_times(root / 'logs' / subdir / fname)
                if t is not None:
                    chi_list.append(chi)
                    trot_list.append(t)
                    bp_list.append(b)
            out[label][grid] = {
                'chi': np.array(chi_list, float),
                'trotter': np.array(trot_list),
                'bp': np.array(bp_list),
            }
    return out

# ── Power-law fit ──────────────────────────────────────────────────────────────
def powerlaw_fit(chi, vals):
    """Fit log(val) = alpha*log(chi) + const. Returns (alpha, scale)."""
    mask = (chi > 0) & (vals > 0)
    if mask.sum() < 2:
        return None, None
    lx = np.log(chi[mask])
    ly = np.log(vals[mask])
    alpha = np.polyfit(lx, ly, 1)[0]
    scale = np.exp(np.mean(ly - alpha * lx))
    return alpha, scale

# ── Plot ───────────────────────────────────────────────────────────────────────
def make_figure(root):
    data = collect(root)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    ax_t, ax_b = axes

    chi_fit = np.linspace(18, 105, 200)

    for label, cfg in SERIES.items():
        color = cfg['color']
        ls    = cfg['ls']

        for grid in ['4x4', '5x5']:
            d = data[label][grid]
            if len(d['chi']) == 0:
                continue
            marker = GRID_MARKER[grid]
            fc     = color if GRID_FILL[grid] else 'white'
            ms     = 6 if GRID_FILL[grid] else 8

            # Trotter panel
            ax_t.plot(d['chi'], d['trotter'] / 1000,
                      color=color, ls=ls, lw=1.8,
                      marker=marker, ms=ms,
                      markerfacecolor=fc, markeredgecolor=color, markeredgewidth=1.4,
                      zorder=3)

            # BP panel
            ax_b.plot(d['chi'], d['bp'] / 1000,
                      color=color, ls=ls, lw=1.8,
                      marker=marker, ms=ms,
                      markerfacecolor=fc, markeredgecolor=color, markeredgewidth=1.4,
                      zorder=3)

        # Power-law fit over 4x4 (most complete chi range)
        d4 = data[label]['4x4']
        if len(d4['chi']) >= 3:
            for ax, key in [(ax_t, 'trotter'), (ax_b, 'bp')]:
                alpha, scale = powerlaw_fit(d4['chi'], d4[key])
                if alpha is not None:
                    ax.plot(chi_fit, scale * chi_fit**alpha / 1000,
                            color=color, ls=':', lw=0.9, alpha=0.55, zorder=2)
                    # Annotate exponent at right edge
                    y_end = scale * 100**alpha / 1000
                    ax.annotate(f'$\\chi^{{{alpha:.1f}}}$',
                                xy=(100, y_end), xytext=(108, y_end),
                                fontsize=7, color=color, va='center',
                                arrowprops=dict(arrowstyle='-', color=color,
                                                lw=0.4, alpha=0.4))

    for ax, title in [(ax_t, 'Trotter time per layer'),
                      (ax_b,  'BP time per layer')]:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Bond dimension $\chi$', fontsize=11)
        ax.set_ylabel('Time  (s)', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(17, 120)
        ax.grid(True, which='both', alpha=0.18, lw=0.5)
        ax.grid(True, which='major', alpha=0.35, lw=0.7)
        ax.set_xticks([20, 40, 60, 80, 100])
        ax.set_xticklabels(['20', '40', '60', '80', '100'])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    # ── Legend ────────────────────────────────────────────────────────────────
    proxies = [
        Line2D([0],[0], color='#2166ac', ls='-',  lw=2, marker='o', ms=6, label='MI_B  Householder'),
        Line2D([0],[0], color='#2166ac', ls='--', lw=2, marker='o', ms=6, label='MI_B  Adaptive'),
        Line2D([0],[0], color='#d73027', ls='-',  lw=2, marker='o', ms=6, label='MI_A   Householder'),
        Line2D([0],[0], color='#d73027', ls='--', lw=2, marker='o', ms=6, label='MI_A   Adaptive'),
        Line2D([0],[0], color='gray', ls='-',  lw=2, marker='o', ms=6,
               markerfacecolor='gray',  label='4×4 grid'),
        Line2D([0],[0], color='gray', ls='-',  lw=2, marker='s', ms=6,
               markerfacecolor='white', markeredgecolor='gray', markeredgewidth=1.4,
               label='5×5 grid'),
        Line2D([0],[0], color='gray', ls=':', lw=1, alpha=0.7, label=r'Power-law fit $\chi^\alpha$'),
    ]
    ax_t.legend(handles=proxies, fontsize=8, framealpha=0.95,
                handlelength=2.2, labelspacing=0.35, loc='upper left')

    fig.suptitle(r'Chi scaling of Trotter and BP timing — 4$\times$4 and 5$\times$5 grids',
                 fontsize=12)
    plt.tight_layout()
    out = Path(__file__).parent / 'fig_timing_chi.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close()

if __name__ == '__main__':
    import matplotlib.ticker
    root = find_root()
    print(f'CppSim root: {root}')
    make_figure(root)
