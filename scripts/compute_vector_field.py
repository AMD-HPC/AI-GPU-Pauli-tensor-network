#!/usr/bin/env python3
"""
compute_vector_field.py — Compute probability current, divergence, and
plaquette curl from n(v,t) logs. Visualize the delta field between
baseline and perturbed runs.

Usage:
  python3 scripts/compute_vector_field.py \
      --base  logs/mawgm_f64/trial_4/scan1_base.log \
      --pert  logs/mawgm_f64/trial_4/scan1_pert.log \
      --broken logs/mawgm_f64/trial_4/broken.json \
      --graph data/graphs/rect_7x7.json \
      --out /tmp/vector_field.png
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
from searcher import load_graph, build_adj, get_bond_sites, parse_pauli_log

FLOOR  = 1e-7
NX, NY = 7, 7

def site_pos(s):
    return (s-1) % NX, 6 - (s-1) // NX

def normalize_field(n_vec):
    """Normalize n(v,t) to a spatial PDF: p(v,t) = n(v,t)/sum_v n(v,t)."""
    total = n_vec.sum()
    if total < FLOOR:
        return np.zeros_like(n_vec)
    return n_vec / total

def bond_currents(p_vec, g, all_sites, site_idx):
    """
    Compute directed bond currents J_{uv} = p(v) - p(u) for each bond.
    Returns dict: bond_idx -> (J, u, v)
    """
    J = {}
    for bi, e in enumerate(g['edges']):
        u, v = get_bond_sites(e)
        J[bi] = (p_vec[site_idx[v]] - p_vec[site_idx[u]], u, v)
    return J

def plaquette_curl(J_dict, g, all_sites):
    """
    Compute curl at each elementary 2x2 plaquette.
    A plaquette is defined by its bottom-left corner (col,row).
    Returns: dict (col,row) -> curl value
    """
    # Build adjacency for quick bond lookup
    bond_between = {}
    for bi, e in enumerate(g['edges']):
        u, v = get_bond_sites(e)
        bond_between[(u,v)] = bi
        bond_between[(v,u)] = bi

    curls = {}
    for row in range(NY - 1):
        for col in range(NX - 1):
            # 4 corners of plaquette (site numbering: s = row*NX + col + 1)
            s00 = (6-row)   * NX + col     + 1  # bottom-left
            s10 = (6-row)   * NX + col + 1 + 1  # bottom-right
            s01 = (6-row-1) * NX + col     + 1  # top-left
            s11 = (6-row-1) * NX + col + 1 + 1  # top-right

            # Counterclockwise: s00 → s10 → s11 → s01 → s00
            # J_{ab} = p(b) - p(a), positive = flow from a to b
            curl = 0.0
            for a, b in [(s00,s10), (s10,s11), (s11,s01), (s01,s00)]:
                if (a,b) in bond_between:
                    bi = bond_between[(a,b)]
                    j, u, v = J_dict[bi]
                    # Direction: if bond goes u->v and we traverse a->b
                    curl += j if (u==a and v==b) else -j
                # If bond doesn't exist (not connected), contribution = 0

            curls[(col, row)] = curl
    return curls

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base',   required=True)
    ap.add_argument('--pert',   required=True)
    ap.add_argument('--graph',  default='data/graphs/rect_7x7.json')
    ap.add_argument('--broken', default=None)
    ap.add_argument('--out',    default='/tmp/vector_field.png')
    ap.add_argument('--cycle',  type=int, default=2,
                    help='Which cycle to visualize (1-indexed)')
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}

    _, _, db = parse_pauli_log(args.base, 'NL')
    _, _, dp = parse_pauli_log(args.pert, 'NL')

    t = args.cycle - 1  # 0-indexed
    n_base = db[t]
    n_pert = dp[t]

    # Use raw n(v,t) for currents (normalization kills signal in delta_curl)
    # Normalized p(v,t) kept for reference but currents use raw values
    p_base = normalize_field(n_base)
    p_pert = normalize_field(n_pert)

    J_base = bond_currents(n_base, g, all_sites, site_idx)
    J_pert = bond_currents(n_pert, g, all_sites, site_idx)

    curl_base = plaquette_curl(J_base, g, all_sites)
    curl_pert = plaquette_curl(J_pert, g, all_sites)

    # Delta fields
    delta_J    = {bi: (J_base[bi][0] - J_pert[bi][0], J_base[bi][1], J_base[bi][2])
                  for bi in J_base}
    delta_curl = {k: curl_base[k] - curl_pert[k] for k in curl_base}

    # Broken bond info
    broken_sites = []
    true_bonds = []
    if args.broken:
        br = json.load(open(args.broken))
        true_bonds = br.get('bonds', [br.get('bond',-1)])
        for bi in true_bonds:
            a, b = get_bond_sites(g['edges'][bi])
            broken_sites += [a, b]

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Vector field analysis — cycle {args.cycle}\n'
                 f'Broken bonds: {true_bonds}', fontsize=12)

    def draw_grid(ax, title):
        ax.set_facecolor('#f8f8f8')
        ax.set_aspect('equal')
        for s in all_sites:
            x, y = site_pos(s)
            ax.plot(x, y, 'o', color='#dddddd', ms=8, zorder=1)
            ax.text(x, y, str(s), ha='center', va='center',
                    fontsize=5, color='#888888', zorder=2)
        for s in broken_sites:
            x, y = site_pos(s)
            ax.plot(x, y, 'o', color='gold', ms=14,
                    markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        ax.set_xlim(-0.5, 6.5); ax.set_ylim(-0.5, 6.5)
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.grid(True, alpha=0.15)
        ax.set_title(title, fontsize=10)

    # Panel 1: delta_J as arrows on bonds
    ax = axes[0]
    draw_grid(ax, f'Delta bond current ΔJ = J_base - J_pert\n(cycle {args.cycle})')
    max_dj = max(abs(dj) for dj, _, _ in delta_J.values()) or 1
    for bi, (dj, u, v) in delta_J.items():
        xu, yu = site_pos(u); xv, yv = site_pos(v)
        mx, my = (xu+xv)/2, (yu+yv)/2
        dx, dy = (xv-xu)*dj/max_dj*0.4, (yv-yu)*dj/max_dj*0.4
        color = 'red' if dj > 0 else 'blue'
        if abs(dj) > max_dj * 0.05:
            ax.annotate('', xy=(mx+dx, my+dy), xytext=(mx-dx, my-dy),
                        arrowprops=dict(arrowstyle='->', color=color,
                                       lw=1.5+2*abs(dj)/max_dj))

    # Panel 2: delta_curl as heatmap on plaquettes
    ax = axes[1]
    draw_grid(ax, f'Delta curl Δcurl = curl_base - curl_pert\n(cycle {args.cycle})')
    max_dc = max(abs(v) for v in delta_curl.values()) or 1
    norm = mcolors.TwoSlopeNorm(vmin=-max_dc, vcenter=0, vmax=max_dc)
    for (col, row), dc in delta_curl.items():
        rect = plt.Rectangle((col-0.5, row-0.5+1), 1, 1,
                              color=plt.cm.RdBu_r(norm(dc)), alpha=0.6, zorder=0)
        ax.add_patch(rect)
        if abs(dc) > max_dc * 0.1:
            ax.text(col+0.0, row+0.5, f'{dc:.2e}',
                    ha='center', va='center', fontsize=5)
    # Highlight plaquettes adjacent to broken bonds
    for bi in true_bonds:
        u, b = get_bond_sites(g['edges'][bi])
        cu, ru = site_pos(u); cb, rb = site_pos(b)
        for col in range(NX-1):
            for row in range(NY-1):
                # Check if this plaquette shares the broken bond
                s00=(6-row)*NX+col+1; s10=(6-row)*NX+col+2
                s01=(6-row-1)*NX+col+1; s11=(6-row-1)*NX+col+2
                if set([u,b]) <= set([s00,s10,s01,s11]):
                    rect2 = plt.Rectangle((col-0.5, row-0.5+1), 1, 1,
                                          fill=False, edgecolor='gold',
                                          linewidth=2.5, zorder=6)
                    ax.add_patch(rect2)

    # Panel 3: L(v,t) for reference
    ax = axes[2]
    draw_grid(ax, f'L(v,t) = -log(n_pert/n_base) reference\n(cycle {args.cycle})')
    L = np.zeros(len(all_sites))
    for v_idx, v in enumerate(all_sites):
        si = site_idx[v]
        if n_base[si] > FLOOR:
            L[v_idx] = -np.log(max(n_pert[si], FLOOR) / n_base[si])
    vmax = max(abs(L).max(), 0.01)
    norm2 = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    for v_idx, v in enumerate(all_sites):
        x, y = site_pos(v)
        ax.plot(x, y, 'o', color=plt.cm.RdBu_r(norm2(L[v_idx])),
                ms=20, zorder=3, markeredgecolor='none')

    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f'Saved: {args.out}')

    # Print top delta_curl plaquettes
    print(f'\nTop delta_curl plaquettes at cycle {args.cycle}:')
    ranked = sorted(delta_curl.items(), key=lambda x: -abs(x[1]))
    for (col,row), dc in ranked[:10]:
        print(f'  plaquette ({col},{row}): delta_curl={dc:.4e}')

if __name__ == '__main__':
    main()
