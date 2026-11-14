#!/usr/bin/env python3
"""
plot_delta_logT_field_16x32.py — δlnT vector field on 16x32 grid.

For each bond (u,v) and step t, computes:
    δlnT_{v→u}(t) = ln(n_base(u,t+1)/n_base(v,t)) - ln(n_pert(u,t+1)/n_pert(v,t))

Draws directed arrows on the grid: color = sign (red=positive, blue=negative),
length = magnitude. Broken bond endpoints marked with green squares.

Usage:
  python3 scripts/plot_delta_logT_field_16x32.py \
      --base logs/mawgm_16x32/trial_1/voronoi/c207_base.log \
      --pert logs/mawgm_16x32/trial_1/voronoi/c207_pert.log \
      --broken logs/mawgm_16x32/trial_1/broken.json \
      --graph data/graphs/rect_16x32.json \
      --out /tmp/fig_delta_logT_16x32.png
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'scripts')
from searcher import load_graph, build_adj, get_bond_sites, parse_pauli_log

NX, NY = 32, 16
FLOOR  = 1e-7


def site_pos(s):
    """Return (col, row) for site s (1-indexed), row 0 at top."""
    return (s - 1) % NX, (s - 1) // NX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',   required=True)
    ap.add_argument('--pert',   required=True)
    ap.add_argument('--broken', default=None)
    ap.add_argument('--graph',  default='data/graphs/rect_16x32.json')
    ap.add_argument('--out',    default='/tmp/fig_delta_logT_16x32.png')
    ap.add_argument('--steps',  type=int, default=4)
    ap.add_argument('--thr',    type=float, default=0.001)
    ap.add_argument('--center', type=int, default=207)
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}

    nc_b, ns, db = parse_pauli_log(args.base, 'NL')
    nc_p, ns, dp = parse_pauli_log(args.pert, 'NL')
    n_steps = min(nc_b, nc_p, args.steps)

    broken_bond = site_a = site_b = None
    if args.broken:
        br = json.load(open(args.broken))
        broken_bond = br.get('bonds', [br.get('bond')])[0]
        site_a = br.get('site_a')
        site_b = br.get('site_b')

    # Zoom to active region around center
    cx, cy = site_pos(args.center)
    pad = 4
    xlim = (max(-0.5, cx - pad - 0.5), min(NX - 0.5, cx + pad + 0.5))
    ylim = (min(NY - 0.5, cy + pad + 0.5), max(-0.5, cy - pad - 0.5))

    fig, axes = plt.subplots(1, n_steps, figsize=(n_steps * 4, 4))
    if n_steps == 1:
        axes = [axes]
    fig.suptitle(f'δlnT vector field — 16×32, center={args.center}, bond {broken_bond} broken',
                 fontsize=12)

    for t in range(n_steps):
        ax = axes[t]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)  # row 0 at top
        ax.set_aspect('equal')
        ax.set_title(f't={t+1}→{t+2}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor('#111111')

        nb_t  = db[t];   np_t  = dp[t]
        nb_t1 = db[t+1]; np_t1 = dp[t+1]

        # Draw all bonds as thin grey lines first
        for edge in g['edges']:
            u, v = get_bond_sites(edge)
            xu, yu = site_pos(u)
            xv, yv = site_pos(v)
            ax.plot([xu, xv], [yu, yv], color='#333333', lw=0.5, zorder=1)

        # Compute and draw δlnT arrows
        max_dT = 0.0
        arrows = []
        for edge in g['edges']:
            u, v = get_bond_sites(edge)
            su = site_idx[u]; sv = site_idx[v]
            xu, yu = site_pos(u)
            xv, yv = site_pos(v)

            for src, dst, x0, y0, dx, dy in [
                (su, sv, xu, yu, xv-xu, yv-yu),
                (sv, su, xv, yv, xu-xv, yu-yv),
            ]:
                if (nb_t[src]  > FLOOR and np_t[src]  > FLOOR and
                        nb_t1[dst] > FLOOR and np_t1[dst] > FLOOR):
                    dT = (np.log(nb_t1[dst] / nb_t[src]) -
                          np.log(np_t1[dst] / np_t[src]))
                    if abs(dT) > args.thr:
                        arrows.append((x0, y0, dx*0.4, dy*0.4, dT))
                        max_dT = max(max_dT, abs(dT))

        for x0, y0, dx, dy, dT in arrows:
            color = '#ff4444' if dT > 0 else '#4488ff'
            alpha = min(1.0, abs(dT) / max(max_dT, 1e-6))
            ax.annotate('', xy=(x0+dx, y0+dy), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='->', color=color,
                                        lw=1.0 + 2*alpha, alpha=0.5 + 0.5*alpha),
                        zorder=3)

        # Mark center
        cx, cy = site_pos(args.center)
        ax.plot(cx, cy, 'c+', markersize=10, markeredgewidth=2, zorder=5)

        # Mark broken bond endpoints
        if site_a and site_b:
            ax.plot(*site_pos(site_a), 'gs', markersize=8,
                    markerfacecolor='none', markeredgewidth=2, zorder=5)
            ax.plot(*site_pos(site_b), 'gs', markersize=8,
                    markerfacecolor='none', markeredgewidth=2, zorder=5)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='#111111')
    print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()
