#!/usr/bin/env python3
"""
show_transfer.py — Show bond transfer ratios n_b(u,t+1)/n_b(v,t) and n_p(u,t+1)/n_p(v,t)
for each bond on the BFS frontier path.

Usage:
  python3 scripts/show_transfer.py --logdir logs/mawgm_f64_2b/trial_5 \
      --scan 3 --plan data/scans/rect_7x7_scan_3gpu.json \
      --graph data/graphs/rect_7x7.json
"""

import argparse
import json
import numpy as np
import sys
sys.path.insert(0, 'scripts')
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites
from bfs_detect import threshold_crossing, bfs_with_color

FLOOR = 1e-7

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',  required=True)
    ap.add_argument('--scan',    type=int, default=3)
    ap.add_argument('--plan',    default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--graph',   default='data/graphs/rect_7x7.json')
    ap.add_argument('--thr',     type=float, default=1e-3)
    ap.add_argument('--T_max',   type=int,   default=5)
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    plan = json.load(open(args.plan))

    sc = plan['placements'][args.scan - 1]
    sid = sc['scan']
    centers = sc['centers']

    _, _, db = parse_pauli_log(f'{args.logdir}/scan{sid}_base.log', 'NL')
    _, _, dp = parse_pauli_log(f'{args.logdir}/scan{sid}_pert.log', 'NL')

    t_base = threshold_crossing(db, args.thr, args.T_max)
    hop_bfs, color_bfs, parent_bond = bfs_with_color(centers, adj, all_sites, g)

    pos = {s: ((s-1)%7, 6-(s-1)//7) for s in range(1, 50)}

    try:
        br = json.load(open(f'{args.logdir}/broken.json'))
        true_bonds = set(br.get('bonds', [br.get('bond', -1)]))
        bond_endpoints = set()
        for bi in true_bonds:
            a, b = get_bond_sites(g['edges'][bi])
            bond_endpoints.add(a); bond_endpoints.add(b)
    except:
        true_bonds = set(); bond_endpoints = set()

    print(f"Bond transfer ratios — Scan {sid} centers={centers}")
    print(f"True bonds: {sorted(true_bonds)}")
    print(f"n_b(u,t+1)/n_b(v,t)  vs  n_p(u,t+1)/n_p(v,t)")
    print()

    # For each frontier site u at cycle t+1, find its BFS parent v at cycle t
    # and compute the transfer ratio
    print(f"{'bond':>5}  {'v→u':>8}  {'t':>4}  "
          f"{'n_b(v,t)':>10}  {'n_b(u,t+1)':>12}  {'T_base':>8}  "
          f"{'n_p(v,t)':>10}  {'n_p(u,t+1)':>12}  {'T_pert':>8}  "
          f"{'ratio':>8}  {'note'}")
    print("-"*110)

    for t in range(1, args.T_max):
        # Sites crossing threshold at cycle t+1 (frontier)
        frontier = [s for s in all_sites if t_base[site_idx[s]] == t+1]

        for u in sorted(frontier):
            ui = site_idx[u]
            pb = parent_bond.get(u, -1)
            if pb < 0: continue  # center itself

            # Parent site v
            v = None
            for neighbor, bi in adj[u]:
                if bi == pb:
                    v = neighbor
                    break
            if v is None: continue

            vi = site_idx[v]
            tb_v = t_base[vi]
            if tb_v < 0 or tb_v != t: continue  # parent should be at cycle t

            # Base transfer: v at cycle t → u at cycle t+1
            nb_v = db[t-1, vi]   # n_base(v, t)
            nb_u = db[t,   ui]   # n_base(u, t+1)
            np_v = dp[t-1, vi]   # n_pert(v, t)
            np_u = dp[t,   ui]   # n_pert(u, t+1)

            if nb_v < 1e-10: continue

            T_base = nb_u / nb_v if nb_v > 1e-10 else 0
            T_pert = np_u / max(np_v, FLOOR)
            ratio  = T_base / T_pert if T_pert > 1e-10 else 0

            col_u, row_u = pos[u]
            col_v, row_v = pos[v]
            tag_u = '*' if u in bond_endpoints else ' '
            tag_v = '*' if v in bond_endpoints else ' '

            note = ''
            if ratio > 1.1:  note = '← BOND REDUCED (broken?)'
            elif ratio < 0.9: note = '← BOND INCREASED'

            a, b = get_bond_sites(g['edges'][pb])
            true_tag = ' ←TRUE' if pb in true_bonds else ''

            print(f"  {pb:>3}  {v:>2}{tag_v}→{u:>2}{tag_u}  {t:>4}  "
                  f"{nb_v:>10.6f}  {nb_u:>12.6f}  {T_base:>8.4f}  "
                  f"{np_v:>10.6f}  {np_u:>12.6f}  {T_pert:>8.4f}  "
                  f"{ratio:>8.4f}  {note}{true_tag}")

if __name__ == '__main__':
    main()
