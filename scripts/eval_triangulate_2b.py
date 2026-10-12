#!/usr/bin/env python3
"""
test_triangulate_2b.py — Test iterative triangulation on 2-bond experiment logs.

Usage:
  python3 scripts/test_triangulate_2b.py --logbase logs/mawgm_2b \
      --plan data/scans/rect_7x7_scan_3gpu.json \
      --graph data/graphs/rect_7x7.json
"""

import argparse
import json
import sys
sys.path.insert(0, 'scripts')
from searcher import load_graph, build_adj, get_bond_sites
from triangulate import triangulate_iterative

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logbase', default='logs/mawgm_2b')
    ap.add_argument('--plan',    default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--graph',   default='data/graphs/rect_7x7.json')
    ap.add_argument('--trials',  type=int, default=10)
    ap.add_argument('--thr',     type=float, default=1e-3)
    ap.add_argument('--thr_L',   type=float, default=0.05)
    ap.add_argument('--T_max',   type=int,   default=3)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    plan = json.load(open(args.plan))

    both_correct = 0
    any_correct  = 0
    none_correct = 0

    print(f"{'Trial':>6}  {'True bonds':>14}  {'Found':>24}  {'Result'}")
    print("-"*68)

    for trial in range(1, args.trials + 1):
        logdir = f'{args.logbase}/trial_{trial}'
        try:
            br = json.load(open(f'{logdir}/broken.json'))
        except FileNotFoundError:
            print(f"  {trial:>4}  (no logs)")
            continue

        true_bonds = set(br.get('bonds', [br.get('bond', -1)]))

        found = triangulate_iterative(
            logdir, plan, g, adj, all_sites, site_idx,
            v_hop=0.448, thr=args.thr, thr_L=args.thr_L,
            T_max=args.T_max, verbose=args.verbose)

        found_set = set(found)
        both    = true_bonds.issubset(found_set)
        partial = bool(true_bonds & found_set) and not both

        if both:
            tag = '✓ BOTH'; both_correct += 1
        elif partial:
            tag = '~ PARTIAL'; any_correct += 1
        else:
            tag = '✗ NONE'; none_correct += 1

        found_str = str([f'b{b}' for b in found])
        print(f"  {trial:>4}  {str(sorted(true_bonds)):>14}  {found_str:>24}  {tag}")

    n = args.trials
    print()
    print(f"Both found:    {both_correct}/{n}")
    print(f"Partial:       {any_correct}/{n}")
    print(f"None found:    {none_correct}/{n}")

if __name__ == '__main__':
    main()
