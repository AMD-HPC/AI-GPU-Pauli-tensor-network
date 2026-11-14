#!/usr/bin/env python3
"""
plot_wavefront_16x32.py — n(v,t) wavefront heatmap on 16x32 grid.

Shows log10(n(v,t)) at each step for base and pert, side by side.
Broken bond visible as wavefront asymmetry between base and pert.

Usage:
  python3 scripts/plot_wavefront_16x32.py \
      --base logs/mawgm_16x32/trial_1/voronoi/c207_base.log \
      --pert logs/mawgm_16x32/trial_1/voronoi/c207_pert.log \
      --broken logs/mawgm_16x32/trial_1/broken.json \
      --out fig_wavefront_16x32.png
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
sys.path.insert(0, 'scripts')
from searcher import parse_pauli_log

NX, NY = 32, 16
FLOOR  = 1e-7


def to_grid(vals):
    """Reshape 512-vector to (NY, NX) grid."""
    return vals.reshape(NY, NX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',   required=True)
    ap.add_argument('--pert',   required=True)
    ap.add_argument('--broken', default=None)
    ap.add_argument('--out',    default='fig_wavefront_16x32.png')
    ap.add_argument('--steps',  type=int, default=5)
    ap.add_argument('--center', type=int, default=207)
    args = ap.parse_args()

    nc_b, ns, db = parse_pauli_log(args.base, 'NL')
    nc_p, ns, dp = parse_pauli_log(args.pert, 'NL')
    n_steps = min(nc_b, nc_p, args.steps)

    broken_bond = None
    site_a = site_b = None
    if args.broken:
        br = json.load(open(args.broken))
        broken_bond = br.get('bonds', [br.get('bond')])[0]
        site_a = br.get('site_a')
        site_b = br.get('site_b')

    # Compute log10(n), floor at -6
    vmin, vmax = -6, -1

    # Compute L(v,t) = ln(n_base / n_pert), clipped to [-2, 2]
    L_all = []
    for t in range(n_steps):
        nb = db[t]; np_ = dp[t]
        L = np.where((nb > FLOOR) & (np_ > FLOOR),
                     np.log(nb / np_), 0.0)
        L_all.append(to_grid(L))

    fig, axes = plt.subplots(n_steps, 3, figsize=(18, n_steps * 2.8))
    fig.suptitle(f'n(v,t) and L(v,t) — 16×32, center={args.center}, bond {broken_bond} broken',
                 fontsize=13, y=1.01)

    crow = (args.center - 1) // NX
    ccol = (args.center - 1) % NX
    if site_a and site_b:
        ra = (site_a - 1) // NX; ca = (site_a - 1) % NX
        rb = (site_b - 1) // NX; cb = (site_b - 1) % NX

    for t in range(n_steps):
        nb = to_grid(np.log10(np.maximum(db[t], FLOOR)))
        np_ = to_grid(np.log10(np.maximum(dp[t], FLOOR)))
        L  = L_all[t]
        lmax = max(abs(L.max()), abs(L.min()), 0.01)

        for col, (data, label, cmap, vlo, vhi, cbar_label) in enumerate([
            (nb,  f't={t+1} base', 'hot',     vmin, vmax, 'log₁₀ n'),
            (np_, f't={t+1} pert', 'hot',     vmin, vmax, 'log₁₀ n'),
            (L,   f't={t+1} L(v,t)', 'RdBu', -lmax, lmax, 'L=ln(base/pert)'),
        ]):
            ax = axes[t, col]
            im = ax.imshow(data, vmin=vlo, vmax=vhi, cmap=cmap, origin='upper',
                           aspect='auto', interpolation='nearest')
            ax.set_title(label, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.plot(ccol, crow, 'c+', markersize=8, markeredgewidth=1.5)
            if site_a and site_b:
                ax.plot(ca, ra, 'gs', markersize=5, markerfacecolor='none', markeredgewidth=1.5)
                ax.plot(cb, rb, 'gs', markersize=5, markerfacecolor='none', markeredgewidth=1.5)
            plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=cbar_label)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()
