#!/usr/bin/env python3
"""
plot_gram_hardpairs.py — Visualize top-10 confusable bond pairs on the 7x7 grid.

Shows:
- Bond pairs as colored line segments (one color per pair)
- Shared vertices as large red dots
- Three seismic scan center sets as different markers
- Grid sites as small gray dots

Usage:
  python3 scripts/plot_gram_hardpairs.py \
      --out /tmp/gram_hardpairs.png
"""

import argparse
import json
import numpy as np
import sys
sys.path.insert(0, 'scripts')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def site_pos(s, ncols=7, nrows=7):
    col = (s-1) % ncols
    row = 6 - (s-1) // ncols
    return col, row

def bond_sites(bi, g):
    e = g['edges'][bi]
    return e['edge'][0][0], e['edge'][1][0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', default='data/graphs/rect_7x7.json')
    ap.add_argument('--plan',  default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--out',   default='/tmp/gram_hardpairs.png')
    args = ap.parse_args()

    g    = json.load(open(args.graph))
    plan = json.load(open(args.plan))

    # Top-10 hard pairs from Muhammad's coherence data
    hard_pairs = [
        (39, 81, 0.8668, 47),
        ( 3, 46, 0.8446,  5),
        ( 2, 45, 0.8368,  4),
        (32, 33, 0.8355, 39),
        (14, 15, 0.8302, 18),
        (29, 69, 0.8234, 35),
        (35, 83, 0.8133, 42),
        (18, 56, 0.8071, 22),
        (30, 77, 0.8053, 36),
        (26, 27, 0.7968, 32),
    ]

    scan_markers = ['*', 's', '^']
    scan_colors  = ['gold', 'cyan', 'lime']
    scan_labels  = ['Scan 1 (corners+center)', 'Scan 2 (quadrants)', 'Scan 3 (mid-edges)']

    pair_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor('#f8f8f8')
    ax.set_aspect('equal')

    # Draw all sites as small gray dots
    for s in range(1, 50):
        x, y = site_pos(s)
        ax.plot(x, y, 'o', color='#cccccc', ms=8, zorder=1)
        ax.text(x, y, str(s), ha='center', va='center', fontsize=5,
                color='#666666', zorder=2)

    # Draw all bonds as thin gray lines
    for bi, e in enumerate(g['edges']):
        a, b = e['edge'][0][0], e['edge'][1][0]
        xa, ya = site_pos(a); xb, yb = site_pos(b)
        ax.plot([xa, xb], [ya, yb], '-', color='#dddddd', lw=1, zorder=1)

    # Draw hard pairs as colored thick lines
    for rank, (bi, bj, coh, sv) in enumerate(hard_pairs):
        col = pair_colors[rank]
        for bond_idx in [bi, bj]:
            a, b = bond_sites(bond_idx, g)
            xa, ya = site_pos(a); xb, yb = site_pos(b)
            ax.plot([xa, xb], [ya, yb], '-', color=col, lw=4,
                    alpha=0.8, zorder=3)
        # Label the pair
        a1, b1 = bond_sites(bi, g)
        xa, ya = site_pos(a1)
        ax.text(xa - 0.15, ya + 0.15,
                f'#{rank+1}\nb{bi},b{bj}\nμ={coh:.2f}',
                fontsize=5.5, color=col, zorder=6,
                bbox=dict(facecolor='white', alpha=0.7, pad=1, edgecolor='none'))

    # Draw shared vertices as red circles
    shared_verts = set(sv for _, _, _, sv in hard_pairs)
    for sv in shared_verts:
        x, y = site_pos(sv)
        ax.plot(x, y, 'o', color='red', ms=16, zorder=4, alpha=0.6)
        ax.plot(x, y, 'o', color='darkred', ms=16, zorder=4,
                fillstyle='none', lw=2)

    # Draw scan centers
    for i, sc in enumerate(plan['placements']):
        for c in sc['centers']:
            x, y = site_pos(c)
            ax.plot(x, y, scan_markers[i], color=scan_colors[i],
                    ms=14, zorder=5, markeredgecolor='black',
                    markeredgewidth=0.8)

    # Legend
    handles = []
    for i in range(3):
        handles.append(plt.Line2D([0],[0], marker=scan_markers[i],
                                  color=scan_colors[i], ms=10,
                                  markeredgecolor='black', lw=0,
                                  label=scan_labels[i]))
    handles.append(plt.Line2D([0],[0], marker='o', color='red',
                              ms=10, lw=0, alpha=0.6,
                              label='Shared vertex (confusable)'))
    for rank, (bi, bj, coh, sv) in enumerate(hard_pairs):
        handles.append(mpatches.Patch(color=pair_colors[rank],
                       label=f'#{rank+1}: bonds {bi},{bj} (μ={coh:.2f}, v={sv})'))

    ax.legend(handles=handles, loc='upper left', fontsize=7,
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

    ax.set_xlim(-0.5, 6.5); ax.set_ylim(-0.5, 6.5)
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.grid(True, alpha=0.2)
    ax.set_title('Top-10 confusable bond pairs (Gram matrix coherence)\n'
                 'Red circles: shared vertices — candidate new centers',
                 fontsize=11)

    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f'Saved: {args.out}')

if __name__ == '__main__':
    main()
