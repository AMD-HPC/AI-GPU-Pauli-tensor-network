#!/usr/bin/env python3
"""
plot_nbase_cycles.py — visualize the baseline n(v,t) wave evolution.

Usage:
  python3 scripts/plot_nbase_cycles.py \
      --baseline logs/mawgm/trial_1/scan3_base.log \
      --graph data/graphs/rect_7x7.json \
      --centers 4,22,28,46 \
      --max-cycles 5 \
      --out logs/mawgm/trial_1/scan3_nbase.png
"""

import argparse
import json
import numpy as np
import sys
sys.path.insert(0, 'scripts')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline',   required=True)
    ap.add_argument('--perturbed',  default=None)
    ap.add_argument('--graph',      required=True)
    ap.add_argument('--centers',    default=None)
    ap.add_argument('--broken',     default=None)
    ap.add_argument('--out',        default='/tmp/nbase_cycles.png')
    ap.add_argument('--max-cycles', type=int, default=5)
    ap.add_argument('--vmax',       type=float, default=None)
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    pos = {s: ((s-1)%7, 6-(s-1)//7) for s in all_sites}
    xs = [pos[s][0] for s in sorted(all_sites)]
    ys = [pos[s][1] for s in sorted(all_sites)]
    site_idx = {s: i for i, s in enumerate(all_sites)}

    _, _, db = parse_pauli_log(args.baseline, 'NL')
    dp = None
    if args.perturbed:
        _, _, dp = parse_pauli_log(args.perturbed, 'NL')
    nc = min(db.shape[0], args.max_cycles)

    centers = [int(c) for c in args.centers.split(',')] if args.centers else []

    broken_sites = []
    if args.broken:
        with open(args.broken) as f: br = json.load(f)
        for bi in br.get('bonds', [br.get('bond')]):
            a, b = get_bond_sites(g['edges'][bi])
            broken_sites += [a, b]

    n_rows = 2 if dp is not None else 1
    vmax = args.vmax or db[:nc].max()

    fig, axes = plt.subplots(n_rows, nc, figsize=(4*nc, 5*n_rows))
    if n_rows == 1: axes = [axes] if nc == 1 else [axes]
    else:
        if nc == 1: axes = [[axes[0]], [axes[1]]]

    titles = ['Baseline n(v,t)', 'Perturbed n(v,t)']
    datasets = [db] + ([dp] if dp is not None else [])

    for row, (data, title) in enumerate(zip(datasets, titles)):
        row_axes = axes[row] if n_rows > 1 else axes
        fig.text(0.01, 1 - (row+0.5)/n_rows, title, va='center',
                 rotation='vertical', fontsize=11, color='black')
        for t in range(nc):
            ax = row_axes[t] if hasattr(row_axes, '__iter__') else row_axes
            ax.set_facecolor('#1a1a2e')
            vals = [data[t, site_idx[s]] for s in sorted(all_sites)]
            sc = ax.scatter(xs, ys, c=vals, cmap='hot', vmin=0, vmax=vmax,
                           s=400, zorder=3)
            for s in broken_sites:
                ax.scatter([pos[s][0]], [pos[s][1]], s=600,
                          facecolors='none', edgecolors='cyan', linewidths=3, zorder=5)
            for c in centers:
                if c in pos:
                    ax.scatter([pos[c][0]], [pos[c][1]], marker='*',
                              s=400, color='white', zorder=6)
                    ax.annotate(str(c), (pos[c][0], pos[c][1]),
                                fontsize=6, color='white', ha='center', va='bottom',
                                xytext=(0, 8), textcoords='offset points', zorder=7)
            plt.colorbar(sc, ax=ax, shrink=0.7)
            if row == 0:
                ax.set_title(f'Cycle {t+1}', fontsize=10)
            ax.set_aspect('equal'); ax.axis('off')

    fig.suptitle(f'n(v,t) wave — centers={centers}', fontsize=12)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f"Saved: {args.out}")

if __name__ == '__main__':
    main()
