#!/usr/bin/env python3
"""
plot_seismic_3scan.py — 3 seismic scans × 3 cycles grid figure.

Rows = scans (scan1, scan2, scan3), Columns = cycles (1, 2, 3).
Each panel shows L(v,t) with its own scan centers marked as stars.

Usage:
  python3 scripts/plot_seismic_3scan.py \
      --logdir logs/mawgm_f64/trial_4 \
      --plan data/scans/rect_7x7_scan_3gpu.json \
      --graph data/graphs/rect_7x7.json \
      --out papers/tomography/fig_ce_bond12.png
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

FLOOR = 1e-7
N_CYCLES = 3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',  required=True)
    ap.add_argument('--plan',    default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--graph',   default='data/graphs/rect_7x7.json')
    ap.add_argument('--out',     default='/tmp/seismic_3scan.png')
    ap.add_argument('--vmax',    type=float, default=0.6)
    ap.add_argument('--thr',     type=float, default=1e-3)
    ap.add_argument('--lmin',    type=float, default=0.0)
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    pos = {s: ((s-1)%7, 6-(s-1)//7) for s in all_sites}
    xs = [pos[s][0] for s in sorted(all_sites)]
    ys = [pos[s][1] for s in sorted(all_sites)]

    plan = json.load(open(args.plan))
    br   = json.load(open(f'{args.logdir}/broken.json'))
    true_bonds = br.get('bonds', [br.get('bond')])
    broken_sites = []
    for bi in true_bonds:
        a, b = get_bond_sites(g['edges'][bi])
        broken_sites += [a, b]

    norm = mcolors.TwoSlopeNorm(vmin=-args.vmax, vcenter=0, vmax=args.vmax)
    n_scans = len(plan['placements'])

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(4*N_CYCLES + 1, 4*n_scans))
    # 3 rows x (3 cols + 1 colorbar col), with gaps between rows
    gs = GridSpec(n_scans, N_CYCLES + 1,
                  width_ratios=[1, 1, 1, 0.06],
                  hspace=0.35, wspace=0.08,
                  top=0.88, bottom=0.04, left=0.12, right=0.96)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(N_CYCLES)] for r in range(n_scans)]
    caxes = [fig.add_subplot(gs[r, N_CYCLES]) for r in range(n_scans)]

    scan_labels = ['Scan 1\n(corners + center)',
                   'Scan 2\n(quadrants)',
                   'Scan 3\n(mid-edges)']

    for row, sc in enumerate(plan['placements']):
        sid     = sc['scan']
        centers = sc['centers']

        try:
            _, _, db = parse_pauli_log(f'{args.logdir}/scan{sid}_base.log', 'NL')
            _, _, dp = parse_pauli_log(f'{args.logdir}/scan{sid}_pert.log', 'NL')
        except Exception as e:
            print(f'Scan {sid}: {e}')
            continue

        nc = min(db.shape[0], dp.shape[0], N_CYCLES)
        ns = db.shape[1]

        L = np.zeros((nc, ns))
        for t in range(nc):
            for v in range(ns):
                if db[t, v] > args.thr:
                    val = -np.log(max(dp[t, v], FLOOR) / db[t, v])
                    L[t, v] = val if abs(val) >= args.lmin else 0.0

        for col in range(N_CYCLES):
            ax = axes[row][col]
            ax.set_facecolor('white')

            sc_plot = ax.scatter(xs, ys, c=L[col], cmap='RdBu_r',
                                 norm=norm, s=280, zorder=3)

            for s in broken_sites:
                ax.scatter([pos[s][0]], [pos[s][1]], s=450,
                           facecolors='none', edgecolors='gold',
                           linewidths=2.5, zorder=5)

            for c in centers:
                if c in pos:
                    ax.scatter([pos[c][0]], [pos[c][1]],
                               marker='*', s=150, color='black', zorder=7)

            if col == N_CYCLES - 1:
                fig.colorbar(sc_plot, cax=caxes[row])

            if row == 0:
                ax.set_title(f'Cycle {col+1}', fontsize=11)
            ax.set_aspect('equal')
            ax.axis('off')
            if col == 0:
                ax.text(-0.12, 0.5, scan_labels[row], fontsize=9,
                        ha='right', va='center', transform=ax.transAxes,
                        multialignment='center')

    # Horizontal separator lines between scan rows
    for r in range(1, n_scans):
        y = axes[r][0].get_position().y1 + 0.01
        line = plt.Line2D([0.10, 0.96], [y, y], transform=fig.transFigure,
                          color='gray', linewidth=0.8, linestyle='--')
        fig.add_artist(line)

    bond_str = ', '.join(f'{get_bond_sites(g["edges"][bi])[0]}↔{get_bond_sites(g["edges"][bi])[1]}' for bi in true_bonds)
    fig.suptitle(f'Seismic $\\mathcal{{L}}(v,t)$ — broken bond {true_bonds} (sites {bond_str})\n'
                 f'Stars: scan centers   Gold circles: bond endpoints',
                 fontsize=11)

    plt.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f'Saved: {args.out}')

if __name__ == '__main__':
    main()
