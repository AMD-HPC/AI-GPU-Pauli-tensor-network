#!/usr/bin/env python3
"""
plot_lightcone_diff.py — Plot |n_full(v,t) - n_lc(v,t)| to validate
lightcone allocation.

Shows three panels per noise level:
  Left:  n_full(v,t) heatmap (log scale)
  Mid:   n_lc(v,t)   heatmap (log scale)
  Right: |difference| heatmap (log scale)

Sites are sorted by BFS distance from center so the lightcone boundary
is visible as a horizontal line.

Usage:
  python3 scripts/plot_lightcone_diff.py \
      --full /tmp/out_full_150.log \
      --lc   /tmp/out_lc_150.log \
      --graph data/graphs/heavy_hex_3x3.json \
      --center 13 \
      --out   logs/lightcone_diff.png
"""

import argparse, re, json
import numpy as np
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_log(path):
    data = {g: [] for g in ['NL','N1','N2','N3']}
    with open(path) as f:
        for line in f:
            for g in data:
                m = re.match(rf'PAULI_{g} (\d+)(.*)', line)
                if m:
                    data[g].append([float(x) for x in m.group(2).split()])
    return {g: np.array(v) for g,v in data.items() if v}

def bfs(graph_path, center):
    with open(graph_path) as f: d = json.load(f)
    ns = d.get('n_sites', len(d.get('vertices',[])))
    adj = [[] for _ in range(ns)]
    for e in d['edges']:
        u=e['edge'][0][0]-1; v=e['edge'][1][0]-1
        adj[u].append(v); adj[v].append(u)
    dist=[-1]*ns; dist[center]=0; q=deque([center])
    while q:
        v=q.popleft()
        for u in adj[v]:
            if dist[u]<0: dist[u]=dist[v]+1; q.append(u)
    return ns, np.array(dist)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full',   required=True)
    ap.add_argument('--lc',     required=True)
    ap.add_argument('--graph',  required=True)
    ap.add_argument('--center', type=int, default=13)
    ap.add_argument('--out',    default='logs/lightcone_diff.png')
    ap.add_argument('--gamma',  default='NL', choices=['NL','N1','N2','N3'])
    args = ap.parse_args()

    full = parse_log(args.full)
    lc   = parse_log(args.lc)
    ns, dist = bfs(args.graph, args.center)

    # Sort sites by BFS distance for clean visualization
    order = np.argsort(dist)
    dist_sorted = dist[order]

    gammas = [g for g in ['NL','N1','N2','N3'] if g in full and g in lc]
    n_gamma = len(gammas)
    nc = min(full[gammas[0]].shape[0], lc[gammas[0]].shape[0])

    fig, axes = plt.subplots(n_gamma, 3,
                             figsize=(14, 3.5 * n_gamma),
                             squeeze=False)

    FLOOR = 1e-8
    labels = {'NL':'Noiseless','N1':'γ=1','N2':'γ=2','N3':'γ=3'}

    for row, g in enumerate(gammas):
        F = full[g][:nc, order]   # (nc, ns) sorted by dist
        L = lc[g][:nc, order]
        D = np.abs(F - L)

        vmin_phy = np.log10(FLOOR)
        vmax_phy = np.log10(max(F.max(), L.max(), FLOOR))
        vmin_dif = np.log10(FLOOR)
        vmax_dif = np.log10(max(D.max(), FLOOR))

        def imshow(ax, mat, title, vmin, vmax, cmap):
            im = ax.imshow(np.log10(np.clip(mat, FLOOR, None)).T,
                           aspect='auto', origin='lower',
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=[0.5, nc+0.5, 0, ns])
            plt.colorbar(im, ax=ax, label='log₁₀', fraction=0.046, pad=0.04)
            ax.set_title(f'{labels[g]}: {title}', fontsize=9)
            ax.set_xlabel('Cycle', fontsize=8)
            ax.set_ylabel('Site (sorted by BFS dist)', fontsize=8)
            ax.tick_params(labelsize=7)
            # Lightcone boundary: mark where dist changes
            for d_val in np.unique(dist_sorted):
                idx = np.searchsorted(dist_sorted, d_val, side='last')
                ax.axhline(idx, color='cyan', lw=0.4, alpha=0.5)

        imshow(axes[row,0], F, 'Full allocation',   vmin_phy, vmax_phy, 'hot_r')
        imshow(axes[row,1], L, 'Lightcone allocation', vmin_phy, vmax_phy, 'hot_r')
        imshow(axes[row,2], D, '|Full − Lightcone|', vmin_dif, vmax_dif, 'Blues')

        # Print summary
        inside  = dist_sorted <= int(0.376 * nc)
        outside = ~inside
        if outside.any():
            print(f"{g}: outside-lightcone max diff = {D[:,outside].max():.2e}  "
                  f"(full run leakage)")
        if inside.any():
            print(f"{g}: inside-lightcone  max diff = {D[:,inside].max():.2e}  "
                  f"(numerical precision)")

    fig.suptitle('Lightcone allocation validation: n(v,t) comparison\n'
                 'Cyan lines = BFS shell boundaries; '
                 'right panel shows outside-lightcone leakage in full run',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {args.out}")

if __name__ == '__main__':
    main()
