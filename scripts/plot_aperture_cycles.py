#!/usr/bin/env python3
"""
plot_aperture_cycles.py — L(v,t) cycle-by-cycle for aperture ensemble (sum of runs).

Usage:
  python3 scripts/plot_aperture_cycles.py \
      --baseline "logs/tomo_6c/base_c*.log" \
      --perturbed "logs/tomo_6c/pert_c*.log" \
      --graph data/graphs/rect_7x7.json \
      --broken logs/tomo_6c/broken.json \
      --out /tmp/aperture_cycles.png
"""

import argparse
import json
import glob
import numpy as np
import sys
sys.path.insert(0, 'scripts')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites

FLOOR = 1e-7

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline',   required=True, help='Glob pattern for baseline logs')
    ap.add_argument('--perturbed',  required=True, help='Glob pattern for perturbed logs')
    ap.add_argument('--graph',      required=True)
    ap.add_argument('--broken',     default=None)
    ap.add_argument('--centers',    default='25,10,34,30,6,32')
    ap.add_argument('--out',        default='/tmp/aperture_cycles.png')
    ap.add_argument('--vmax',       type=float, default=0.6)
    ap.add_argument('--thr',        type=float, default=1e-4)
    ap.add_argument('--lmin',       type=float, default=0.0, help='Zero L below this magnitude')
    ap.add_argument('--max-cycles', type=int,   default=5)
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    pos = {s: ((s-1)%7, 6-(s-1)//7) for s in all_sites}
    xs = [pos[s][0] for s in sorted(all_sites)]
    ys = [pos[s][1] for s in sorted(all_sites)]

    base_logs = sorted(glob.glob(args.baseline))
    pert_logs = sorted(glob.glob(args.perturbed))

    nc = args.max_cycles
    ns = None
    L_sum = None

    for b_log, p_log in zip(base_logs, pert_logs):
        try:
            _, ns_i, db = parse_pauli_log(b_log, 'NL')
            _, ns_i, dp = parse_pauli_log(p_log, 'NL')
        except ValueError:
            continue
        nc_i = min(db.shape[0], dp.shape[0], nc)
        if L_sum is None:
            ns = ns_i
            L_sum = np.zeros((nc, ns))
        for t in range(nc_i):
            for v in range(ns):
                if db[t, v] > args.thr:
                    val = -np.log(max(dp[t, v], FLOOR) / max(db[t, v], FLOOR))
                    L_sum[t, v] += val if abs(val) >= args.lmin else 0.0

    if L_sum is None:
        print("ERROR: no valid log pairs found")
        return

    broken_sites = []
    if args.broken:
        with open(args.broken) as f: br = json.load(f)
        for bi in br.get('bonds', [br.get('bond')]):
            a, b = get_bond_sites(g['edges'][bi])
            broken_sites += [a, b]

    centers = [int(c) for c in args.centers.split(',')]
    norm = mcolors.TwoSlopeNorm(vmin=-args.vmax, vcenter=0, vmax=args.vmax)

    fig, axes = plt.subplots(1, nc, figsize=(4*nc, 5))
    if nc == 1: axes = [axes]
    fig.suptitle(f'Aperture ensemble L(v,t) — cycle by cycle\n'
                 f'Broken bonds: {sorted(set(br.get("bonds",[br.get("bond")]))) if args.broken else "?"}'
                 f'  ({len(base_logs)} centers)',
                 fontsize=12)

    for t in range(nc):
        ax = axes[t]
        ax.set_facecolor('#1a1a2e')
        sc = ax.scatter(xs, ys, c=L_sum[t], cmap='RdBu_r', norm=norm, s=400, zorder=3)
        for s in broken_sites:
            ax.scatter([pos[s][0]], [pos[s][1]], s=600,
                      facecolors='none', edgecolors='gold', linewidths=3, zorder=5)
        for c in centers:
            if c in pos:
                ax.scatter([pos[c][0]], [pos[c][1]], marker='*', s=300, color='white', zorder=6)
                ax.annotate(str(c), (pos[c][0], pos[c][1]),
                            fontsize=6, color='white', ha='center', va='bottom',
                            xytext=(0, 8), textcoords='offset points', zorder=7)
        plt.colorbar(sc, ax=ax, shrink=0.7)
        ax.set_title(f'Cycle {t+1}', fontsize=10)
        ax.set_aspect('equal'); ax.axis('off')

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f"Saved: {args.out}")

if __name__ == '__main__':
    main()
