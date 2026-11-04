#!/usr/bin/env python3
"""
plot_gram_color_compare.py — Side-by-side plot of top-10 confusable pairs
for standard (H-first) and swapped (V-first) color orderings.

Usage:
  python3 scripts/plot_gram_color_compare.py \
      --standard logs/gram_standard/gram.json \
      --swapped  logs/gram_swapped/gram.json \
      --out /tmp/gram_color_compare.png
"""

import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.insert(0, 'scripts')
from searcher import load_graph, build_adj, get_bond_sites

def site_pos(s, ncols=7, nrows=7):
    return (s-1) % ncols, 6 - (s-1) // ncols

def draw_panel(ax, title, pairs, g, plan, pair_colors):
    ax.set_facecolor('white')
    ax.set_aspect('equal')

    # All sites
    for s in range(1, 50):
        x, y = site_pos(s)
        ax.plot(x, y, 'o', color='#dddddd', ms=10, zorder=1)
        ax.text(x, y, str(s), ha='center', va='center',
                fontsize=5, color='#999999', zorder=2)

    # All bonds as thin gray lines
    for e in g['edges']:
        a, b = e['edge'][0][0], e['edge'][1][0]
        xa, ya = site_pos(a); xb, yb = site_pos(b)
        ax.plot([xa, xb], [ya, yb], '-', color='#eeeeee', lw=1, zorder=1)

    # Scan centers
    scan_markers = ['*', 's', '^']
    scan_colors  = ['gold', 'cyan', 'lime']
    for i, sc in enumerate(plan['placements']):
        for c in sc['centers']:
            x, y = site_pos(c)
            ax.plot(x, y, scan_markers[i], color=scan_colors[i],
                    ms=12, zorder=5, markeredgecolor='black', markeredgewidth=0.6)

    # Hard pairs
    for rank, p in enumerate(pairs):
        col = pair_colors[rank]
        bi, bj = p['bond_i'], p['bond_j']
        for bond_idx in [bi, bj]:
            a, b = get_bond_sites(g['edges'][bond_idx])
            xa, ya = site_pos(a); xb, yb = site_pos(b)
            ax.plot([xa, xb], [ya, yb], '-', color=col, lw=5, alpha=0.8, zorder=3)

        # Shared vertex
        sv = p.get('shared_vertex')
        if sv:
            x, y = site_pos(sv)
            ax.plot(x, y, 'o', color='red', ms=14, alpha=0.5, zorder=4)

    ax.set_xlim(-0.5, 6.5); ax.set_ylim(-0.5, 6.5)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.grid(True, alpha=0.15)
    ax.set_title(title, fontsize=11, fontweight='bold')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--standard', default='logs/gram_standard/gram.json')
    ap.add_argument('--swapped',  default='logs/gram_swapped/gram.json')
    ap.add_argument('--graph',    default='data/graphs/rect_7x7.json')
    ap.add_argument('--plan',     default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--out',      default='/tmp/gram_color_compare.png')
    ap.add_argument('--top_k',    type=int, default=10)
    args = ap.parse_args()

    g    = load_graph(args.graph)
    plan = json.load(open(args.plan))
    std  = json.load(open(args.standard))
    swp  = json.load(open(args.swapped))

    pair_colors = plt.cm.tab10(np.linspace(0, 1, args.top_k))

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Gram matrix top-10 confusable pairs\n'
                 'Left: H fires first (standard)   Right: V fires first (swapped)\n'
                 'Colored segments: confusable pairs   Red dots: shared vertex   Stars/squares/triangles: scan centers',
                 fontsize=10)

    draw_panel(axes[0],
               f'Standard (H-first)\nmedian μ={std["median_coh"]:.3f}  max μ={std["max_coh"]:.3f}',
               std['pairs_detail'][:args.top_k], g, plan, pair_colors)

    draw_panel(axes[1],
               f'Swapped (V-first)\nmedian μ={swp["median_coh"]:.3f}  max μ={swp["max_coh"]:.3f}',
               swp['pairs_detail'][:args.top_k], g, plan, pair_colors)

    # Shared legend
    handles = []
    for rank in range(args.top_k):
        sp = std['pairs_detail'][rank]
        sw = swp['pairs_detail'][rank]
        handles.append(mpatches.Patch(color=pair_colors[rank],
            label=f'#{rank+1} std: b{sp["bond_i"]},{sp["bond_j"]} μ={sp["coherence"]:.3f} | '
                  f'swp: b{sw["bond_i"]},{sw["bond_j"]} μ={sw["coherence"]:.3f}'))

    fig.legend(handles=handles, loc='lower center', fontsize=7,
               ncol=2, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f'Saved: {args.out}')

if __name__ == '__main__':
    main()
