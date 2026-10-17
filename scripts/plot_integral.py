#!/usr/bin/env python3
"""
plot_integral.py — Visualize log(B(v)/N(v)) for each scan as spatial map.

Usage:
  python3 scripts/plot_integral.py --logdir logs/mawgm_v2/trial_5 \
      --plan data/scans/rect_7x7_scan_3gpu.json \
      --graph data/graphs/rect_7x7.json \
      --out /tmp/trial5_integral.png
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
from integral_detect import compute_integral

FLOOR = 1e-7

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',  required=True)
    ap.add_argument('--plan',    default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--graph',   default='data/graphs/rect_7x7.json')
    ap.add_argument('--broken',  default=None)
    ap.add_argument('--out',     default='/tmp/integral.png')
    ap.add_argument('--vmax',    type=float, default=1.5)
    ap.add_argument('--T_max',   type=int,   default=5)
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    plan = json.load(open(args.plan))

    pos = {s: ((s-1)%7, 6-(s-1)//7) for s in all_sites}
    xs = [pos[s][0] for s in sorted(all_sites)]
    ys = [pos[s][1] for s in sorted(all_sites)]

    broken_sites = []
    title_bonds = ''
    broken_file = args.broken or f'{args.logdir}/broken.json'
    try:
        br = json.load(open(broken_file))
        true_bonds = br.get('bonds', [br.get('bond', -1)])
        title_bonds = str(sorted(true_bonds))
        for bi in true_bonds:
            a, b = get_bond_sites(g['edges'][bi])
            broken_sites += [a, b]
    except:
        true_bonds = []

    n_scans = len(plan['placements'])
    fig, axes = plt.subplots(1, n_scans + 1, figsize=(5*(n_scans+1), 5))
    fig.suptitle(f'Integral log(B/N) — Broken bonds: {title_bonds}', fontsize=13)

    norm = mcolors.TwoSlopeNorm(vmin=-args.vmax, vcenter=0, vmax=args.vmax)
    combined = np.zeros(len(all_sites))

    for col, sc in enumerate(plan['placements']):
        sid = sc['scan']
        centers = sc['centers']
        try:
            _, _, db = parse_pauli_log(f'{args.logdir}/scan{sid}_base.log', 'NL')
            _, _, dp = parse_pauli_log(f'{args.logdir}/scan{sid}_pert.log', 'NL')
        except:
            axes[col].axis('off')
            continue

        B, N, log_ratio = compute_integral(db, dp, args.T_max)
        combined += log_ratio  # accumulate across scans

        vals = [log_ratio[site_idx[s]] for s in sorted(all_sites)]

        ax = axes[col]
        ax.set_facecolor('#1a1a2e')
        sc_plot = ax.scatter(xs, ys, c=vals, cmap='RdBu_r', norm=norm, s=400, zorder=3)
        for s in broken_sites:
            ax.scatter([pos[s][0]], [pos[s][1]], s=600,
                       facecolors='none', edgecolors='gold', linewidths=3, zorder=5)
        for c in centers:
            if c in pos:
                ax.scatter([pos[c][0]], [pos[c][1]], marker='*', s=300,
                           color='white', zorder=6)
                ax.annotate(str(c), pos[c], fontsize=7, color='white',
                            ha='center', va='bottom', xytext=(0,8),
                            textcoords='offset points', zorder=7)
        plt.colorbar(sc_plot, ax=ax, shrink=0.7)
        ax.set_title(f'Scan {sid}: log(B/N)', fontsize=11)
        ax.set_aspect('equal'); ax.axis('off')

    # Combined panel
    ax = axes[n_scans]
    ax.set_facecolor('#1a1a2e')
    vals_combined = [combined[site_idx[s]] for s in sorted(all_sites)]
    norm2 = mcolors.TwoSlopeNorm(vmin=-args.vmax*n_scans, vcenter=0, vmax=args.vmax*n_scans)
    sc_plot = ax.scatter(xs, ys, c=vals_combined, cmap='RdBu_r', norm=norm2, s=400, zorder=3)
    for s in broken_sites:
        ax.scatter([pos[s][0]], [pos[s][1]], s=600,
                   facecolors='none', edgecolors='gold', linewidths=3, zorder=5)
    plt.colorbar(sc_plot, ax=ax, shrink=0.7)
    ax.set_title('Combined Σ log(B/N)', fontsize=11)
    ax.set_aspect('equal'); ax.axis('off')

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f"Saved: {args.out}")

if __name__ == '__main__':
    main()
