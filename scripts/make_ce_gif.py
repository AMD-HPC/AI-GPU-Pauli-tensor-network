#!/usr/bin/env python3
"""
make_ce_gif.py — Animate cross-entropy perturbation L(v,t) = -ln(n_pert/n_base)

Shows how the perturbation signal propagates from the broken bond outward:
  Red   = n_pert < n_base  (wave blocked/reflected)
  Blue  = n_pert > n_base  (shadow downstream)
  White = no change
  Gold circles = broken bond endpoints
  Stars = seismic centers

Usage:
  python3 scripts/make_ce_gif.py \
      --baseline  logs/tomo_v2/base_B_bond12.log \
      --perturbed logs/tomo_v2/pert_B_bond12.log \
      --graph data/graphs/rect_7x7.json \
      --broken logs/tomo_v2/broken_bond12.json \
      --centers 25,10,34,30,6,32 \
      --out logs/tomo_v2/ce_perturbation.gif \
      --fps 2
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
import matplotlib.animation as animation
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline',  required=True)
    ap.add_argument('--perturbed', required=True)
    ap.add_argument('--graph',     required=True)
    ap.add_argument('--broken',    default=None,  help='breaker.py JSON (gold circles)')
    ap.add_argument('--centers',   default=None,  help='Comma-separated center sites (stars)')
    ap.add_argument('--out',       default='ce_perturbation.gif')
    ap.add_argument('--fps',       type=int, default=2)
    ap.add_argument('--vmax',      type=float, default=0.5)
    ap.add_argument('--size',      type=int, default=7, help='Figure size in inches')
    args = ap.parse_args()

    # Load data
    nc_b, ns, data_base = parse_pauli_log(args.baseline,  'NL')
    nc_p, ns, data_pert = parse_pauli_log(args.perturbed, 'NL')
    nc = min(nc_b, nc_p)

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    pos = {s: ((s-1)%7, 6-(s-1)//7) for s in all_sites}
    xs = [pos[s][0] for s in sorted(all_sites)]
    ys = [pos[s][1] for s in sorted(all_sites)]

    # Optional: broken bond endpoints
    broken_sites = []
    if args.broken:
        with open(args.broken) as f: broken = json.load(f)
        broken_sites = [broken['site_a'], broken['site_b']]
        bond_label = f"bond {broken['bond']} ({broken['site_a']}↔{broken['site_b']})"
    else:
        bond_label = ""

    # Optional: centers
    centers = []
    if args.centers:
        centers = [int(c) for c in args.centers.split(',')]

    FLOOR = 1e-7
    THR   = 1e-4   # n_base must exceed this to compute L; else white
    norm = mcolors.TwoSlopeNorm(vmin=-args.vmax, vcenter=0, vmax=args.vmax)

    fig, ax = plt.subplots(figsize=(args.size, args.size))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    ax.set_aspect('equal')
    ax.axis('off')

    # Initial scatter
    sc = ax.scatter(xs, ys, c=np.zeros(ns), cmap='RdBu_r',
                    norm=norm, s=500, zorder=3)

    # Gold circles for broken bond endpoints
    for s in broken_sites:
        ax.scatter([pos[s][0]], [pos[s][1]], s=700,
            facecolors='none', edgecolors='gold', linewidths=3, zorder=5)

    # Stars for centers
    for c in centers:
        if c in pos:
            ax.scatter([pos[c][0]], [pos[c][1]],
                       marker='*', s=300, color='white', zorder=6)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label('$\\mathcal{L}(v,t) = -\\ln(n_{\\rm pert}/n_{\\rm base})$',
                   color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    title = ax.set_title('', color='white', fontsize=11, pad=8)

    def update(t):
        ce = np.array([
            -np.log(max(data_pert[t, vi], FLOOR) / max(data_base[t, vi], FLOOR))
            if data_base[t, vi] > THR else 0.0
            for vi in range(ns)
        ])
        sc.set_array(ce)
        n_red  = (ce >  0.05).sum()
        n_blue = (ce < -0.05).sum()
        title.set_text(
            f'Cross-entropy perturbation — cycle {t+1}/{nc}\n'
            f'{bond_label}  |  Red={n_red} Blue={n_blue} sites'
        )
        return sc, title

    ani = animation.FuncAnimation(
        fig, update, frames=nc, interval=1000//args.fps, blit=False)
    ani.save(args.out, writer=animation.PillowWriter(fps=args.fps))
    plt.close()
    print(f"Saved: {args.out}  ({nc} frames, {args.fps} fps)")

if __name__ == '__main__':
    main()
