#!/usr/bin/env python3
"""
show_bfs_frontier.py — Show BFS frontier per cycle with L values and causal order.

Usage:
  python3 scripts/show_bfs_frontier.py --logdir logs/mawgm_f64_2b/trial_5 \
      --scan 2 --plan data/scans/rect_7x7_scan_3gpu.json \
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
    ap.add_argument('--scan',    type=int, default=2)
    ap.add_argument('--plan',    default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--graph',   default='data/graphs/rect_7x7.json')
    ap.add_argument('--thr',     type=float, default=1e-3)
    ap.add_argument('--thr_L',   type=float, default=0.05)
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
    t_pert  = threshold_crossing(dp, args.thr, args.T_max)
    hop_bfs, color_bfs, _ = bfs_with_color(centers, adj, all_sites, g)

    pos = {s: ((s-1)%7, 6-(s-1)//7) for s in range(1, 50)}

    # Load broken bonds if available
    try:
        br = json.load(open(f'{args.logdir}/broken.json'))
        true_bonds = set(br.get('bonds', [br.get('bond', -1)]))
        # Get all endpoints of true bonds
        bond_endpoints = set()
        for bi in true_bonds:
            a, b = get_bond_sites(g['edges'][bi])
            bond_endpoints.add(a); bond_endpoints.add(b)
    except:
        true_bonds = set(); bond_endpoints = set()

    print(f"BFS frontier — Scan {sid} centers={centers}")
    print(f"True bonds: {sorted(true_bonds)}")
    print()

    for t in range(1, args.T_max + 1):
        frontier = [s for s in all_sites if t_base[site_idx[s]] == t]
        if not frontier: continue
        print(f"=== Cycle {t} frontier ({len(frontier)} sites) ===")
        found_signal = False
        for s in sorted(frontier):
            si = site_idx[s]
            nb = db[t-1, si]; np_ = dp[t-1, si]
            L = -np.log(max(np_, FLOOR) / nb) if nb > args.thr else 0.0
            tp = t_pert[si]
            delayed = (tp < 0) or (tp > t)
            tag = '*' if s in bond_endpoints else ' '
            col, row_plot = pos[s]
            h = hop_bfs.get(s, 999)
            c = color_bfs.get(s, -1)

            flag = ''
            if L < -args.thr_L:   flag = '← BLUE (wall face)'
            elif L > args.thr_L:  flag = '← RED  (shadow)'

            delay_str = f'  DELAYED(t*_pert={tp})' if delayed else ''

            if flag or tag == '*':
                found_signal = True
                print(f"  site{s:>3}{tag} ({row_plot},{col})  "
                      f"hop={h} color={c}  L={L:+.3f}  {flag}{delay_str}")
        if not found_signal:
            print(f"  (no signal above thr_L={args.thr_L})")
        print()

if __name__ == '__main__':
    main()
