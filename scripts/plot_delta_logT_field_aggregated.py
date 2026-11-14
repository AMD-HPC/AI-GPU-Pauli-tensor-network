#!/usr/bin/env python3
"""
plot_delta_logT_field_aggregated.py — Aggregated δlnT vector field.

Reads ALL base/pert log pairs from a directory, aggregates δlnT per directed
bond across all scans, and plots arrows on the full grid.

Consistent with fig_delta_logT.png: arrow length ~ magnitude, red=base>pert, blue=pert>base.

Usage:
  python3 scripts/plot_delta_logT_field_aggregated.py \
      --logdir logs/mawgm_16x32/trial_1/voronoi \
      --broken logs/mawgm_16x32/trial_1/broken.json \
      --graph data/graphs/rect_16x32.json \
      --out /tmp/fig_delta_logT_16x32_agg.png
"""

import argparse
import glob
import json
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'scripts')
from searcher import load_graph, build_adj, get_bond_sites, parse_pauli_log

NX, NY = 32, 16
FLOOR  = 1e-7


def site_pos(s):
    return (s - 1) % NX, (s - 1) // NX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',  required=True)
    ap.add_argument('--broken',  default=None)
    ap.add_argument('--graph',   default='data/graphs/rect_16x32.json')
    ap.add_argument('--out',     default='/tmp/fig_delta_logT_16x32_agg.png')
    ap.add_argument('--steps',   type=int, default=3)
    ap.add_argument('--thr',     type=float, default=0.001)
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    n_bonds = len(g['edges'])

    broken_bond = site_a = site_b = None
    if args.broken:
        br = json.load(open(args.broken))
        broken_bond = br.get('bonds', [br.get('bond')])[0]
        site_a = br.get('site_a')
        site_b = br.get('site_b')

    # Aggregate δlnT per directed bond per step across all log pairs
    # scores[bi][direction][t] where direction 0=u->v, 1=v->u
    bond_scores = {}  # (bi, src, dst) -> list of dT per step
    for bi in range(n_bonds):
        u, v = get_bond_sites(g['edges'][bi])
        su, sv = site_idx[u], site_idx[v]
        bond_scores[(bi, su, sv)] = np.zeros(args.steps)
        bond_scores[(bi, sv, su)] = np.zeros(args.steps)

    base_logs = sorted(glob.glob(f'{args.logdir}/*_base.log'))
    n_pairs = 0
    for base_log in base_logs:
        pert_log = base_log.replace('_base.log', '_pert.log')
        if not os.path.exists(pert_log):
            continue
        try:
            nc_b, ns, db = parse_pauli_log(base_log, 'NL')
            nc_p, ns, dp = parse_pauli_log(pert_log, 'NL')
        except Exception:
            continue
        n_pairs += 1
        n_steps = min(nc_b, nc_p, args.steps)

        for t in range(n_steps):
            nb_t  = db[t];   np_t  = dp[t]
            nb_t1 = db[t+1]; np_t1 = dp[t+1]

            for bi in range(n_bonds):
                u, v = get_bond_sites(g['edges'][bi])
                su, sv = site_idx[u], site_idx[v]

                for src, dst in [(su, sv), (sv, su)]:
                    if (nb_t[src]  > FLOOR and np_t[src]  > FLOOR and
                            nb_t1[dst] > FLOOR and np_t1[dst] > FLOOR):
                        dT = (np.log(nb_t1[dst] / nb_t[src]) -
                              np.log(np_t1[dst] / np_t[src]))
                        if abs(dT) > args.thr:
                            bond_scores[(bi, src, dst)][t] += dT

    print(f"Aggregated {n_pairs} log pairs")

    # Plot one panel per step
    fig, axes = plt.subplots(1, args.steps, figsize=(args.steps * 10, 5))
    if args.steps == 1:
        axes = [axes]
    fig.suptitle(f'δlnT vector field — 16×32 aggregated ({n_pairs} centers), bond {broken_bond} broken',
                 fontsize=12)

    for t in range(args.steps):
        ax = axes[t]
        ax.set_xlim(-0.5, NX - 0.5)
        ax.set_ylim(NY - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_title(f't={t+1}→{t+2}', fontsize=10)
        ax.set_facecolor('#111111')

        # Grid lines
        for x in range(NX):
            ax.axvline(x, color='#222222', lw=0.3, zorder=0)
        for y in range(NY):
            ax.axhline(y, color='#222222', lw=0.3, zorder=0)

        # Collect max for normalization
        all_dT = []
        for bi in range(n_bonds):
            u, v = get_bond_sites(g['edges'][bi])
            su, sv = site_idx[u], site_idx[v]
            all_dT.append(abs(bond_scores[(bi, su, sv)][t]))
            all_dT.append(abs(bond_scores[(bi, sv, su)][t]))
        max_dT = max(all_dT) if all_dT else 1.0
        if max_dT < 1e-9:
            max_dT = 1.0

        for bi in range(n_bonds):
            u, v = get_bond_sites(g['edges'][bi])
            su, sv = site_idx[u], site_idx[v]
            xu, yu = site_pos(u)
            xv, yv = site_pos(v)

            for src, dst, x0, y0, x1, y1 in [
                (su, sv, xu, yu, xv, yv),
                (sv, su, xv, yv, xu, yu),
            ]:
                dT = bond_scores[(bi, src, dst)][t]
                if abs(dT) < args.thr:
                    continue
                color = '#ff3333' if dT > 0 else '#3399ff'
                scale = abs(dT) / max_dT
                dx = (x1 - x0) * 0.45 * scale
                dy = (y1 - y0) * 0.45 * scale
                ax.annotate('', xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle='->', color=color,
                                            lw=0.5 + 2.5*scale,
                                            alpha=0.4 + 0.6*scale),
                            zorder=3)

        # Mark broken bond
        if site_a and site_b:
            xa, ya = site_pos(site_a)
            xb, yb = site_pos(site_b)
            ax.plot(xa, ya, 'o', color='yellow', markersize=8,
                    markerfacecolor='yellow', alpha=0.8, zorder=5)
            ax.plot(xb, yb, 'o', color='yellow', markersize=8,
                    markerfacecolor='yellow', alpha=0.8, zorder=5)
            ax.plot([xa, xb], [ya, yb], '-', color='yellow', lw=2, alpha=0.6, zorder=4)

        ax.set_xticks(range(0, NX, 4))
        ax.set_yticks(range(0, NY, 2))
        ax.tick_params(colors='grey', labelsize=7)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='#111111')
    print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()
