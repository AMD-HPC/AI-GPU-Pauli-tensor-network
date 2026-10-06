#!/usr/bin/env python3
"""
compare_seismic_aperture.py — Cycle-by-cycle comparison of L(v,t):
  Row 1: Seismic   (one simultaneous multi-center run)
  Row 2: Aperture  (sum of per-center L fields)

Usage:
  python3 scripts/compare_seismic_aperture.py \
      --seismic-base  logs/tomo_v2/base_B_bond12_13.log \
      --seismic-pert  logs/tomo_v2/pert_B_bond12_13.log \
      --aperture-base "logs/tomo_6c/base_c*.log" \
      --aperture-pert "logs/tomo_6c/pert_c*.log" \
      --graph data/graphs/rect_7x7.json \
      --broken logs/tomo_6c/broken.json \
      --out /tmp/compare_seismic_aperture.png
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
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites, arrival_time

FLOOR = 1e-7

def compute_L_field(base_logs, pert_logs, thr=1e-4):
    """Compute L(v,t) = -ln(n_pert/n_base) for each cycle.
    For multiple log pairs: sum the L fields (ensemble)."""
    L_sum = None
    nc_min = None
    ns = None

    for b_log, p_log in zip(base_logs, pert_logs):
        try:
            nc_b, ns_i, data_base = parse_pauli_log(b_log.strip(), 'NL')
            nc_p, ns_i, data_pert = parse_pauli_log(p_log.strip(), 'NL')
        except ValueError:
            continue
        nc_i = min(nc_b, nc_p)
        if L_sum is None:
            nc_min = nc_i
            ns = ns_i
            L_sum = np.zeros((nc_i, ns_i))
        else:
            nc_min = min(nc_min, nc_i)

        for t in range(nc_i):
            for v in range(ns_i):
                nb = max(data_base[t, v], FLOOR)
                np_ = max(data_pert[t, v], FLOOR)
                if data_base[t, v] > thr:  # only where baseline has signal
                    L_sum[t, v] += -np.log(np_ / nb)

    return L_sum[:nc_min], nc_min, ns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seismic-base', required=True)
    ap.add_argument('--seismic-pert', required=True)
    ap.add_argument('--aperture-base', required=True, help='Glob pattern')
    ap.add_argument('--aperture-pert', required=True, help='Glob pattern')
    ap.add_argument('--graph',   required=True)
    ap.add_argument('--broken',  default=None)
    ap.add_argument('--out',     default='/tmp/compare_seismic_aperture.png')
    ap.add_argument('--vmax',    type=float, default=0.6)
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    pos = {s: ((s-1)%7, 6-(s-1)//7) for s in all_sites}
    xs = [pos[s][0] for s in sorted(all_sites)]
    ys = [pos[s][1] for s in sorted(all_sites)]

    broken_sites = []
    if args.broken:
        with open(args.broken) as f: br = json.load(f)
        for bi in br.get('bonds', [br.get('bond')]):
            a, b = get_bond_sites(g['edges'][bi])
            broken_sites += [a, b]

    # Compute L fields
    L_seismic, nc_s, ns = compute_L_field(
        [args.seismic_base], [args.seismic_pert])

    aper_base = sorted(glob.glob(args.aperture_base))
    aper_pert = sorted(glob.glob(args.aperture_pert))
    L_aperture, nc_a, _ = compute_L_field(aper_base, aper_pert)

    nc = min(nc_s, nc_a, 5)  # show up to 5 cycles

    norm = mcolors.TwoSlopeNorm(vmin=-args.vmax, vcenter=0, vmax=args.vmax)

    fig, axes = plt.subplots(2, nc, figsize=(4*nc, 9))
    fig.suptitle(f'Seismic vs Aperture: L(v,t) cycle by cycle\n'
                 f'Bonds {sorted(set(br.get("bonds", [br.get("bond")])) if args.broken else [])} perturbed',
                 fontsize=12)

    row_labels = ['Seismic', 'Aperture\n(ensemble sum)']
    L_fields = [L_seismic, L_aperture]

    for row, (label, L) in enumerate(zip(row_labels, L_fields)):
        for t in range(nc):
            ax = axes[row, t]
            ax.set_facecolor('#1a1a2e')
            ce = L[t]
            sc = ax.scatter(xs, ys, c=ce, cmap='RdBu_r', norm=norm,
                           s=400, zorder=3)
            # Mark broken bond sites
            for s in broken_sites:
                ax.scatter([pos[s][0]], [pos[s][1]], s=600,
                          facecolors='none', edgecolors='gold',
                          linewidths=3, zorder=5)
            ax.set_aspect('equal'); ax.axis('off')
            if row == 0:
                ax.set_title(f'Cycle {t+1}', fontsize=11)
            if t == 0:
                ax.set_ylabel(label, fontsize=10)
                ax.text(-0.5, 0.5, label, transform=ax.transAxes,
                       fontsize=10, ha='right', va='center', rotation=0)

        plt.colorbar(sc, ax=axes[row, -1], shrink=0.7,
                    label='L(v,t)' if row==0 else '')

    plt.tight_layout()
    plt.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f"Saved: {args.out}")

if __name__ == '__main__':
    main()
