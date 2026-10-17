#!/usr/bin/env python3
"""
check_antennas.py — Check integral B/N for custom antenna logs.

Usage:
  python3 scripts/check_antennas.py \
      --antennas "Row3:/tmp/row3_base.log:/tmp/row3_pert.log" \
                 "Col5:/tmp/col5_base.log:/tmp/col5_pert.log" \
      --bonds "8:10:11" "45:4:11"
"""

import argparse
import sys
sys.path.insert(0, 'scripts')
from searcher import parse_pauli_log, load_graph, build_adj
from integral_detect import compute_integral

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--antennas', nargs='+', required=True,
                    help='"label:base_log:pert_log"')
    ap.add_argument('--bonds', nargs='+', default=['8:10:11','45:4:11'],
                    help='"bond_idx:site_a:site_b"')
    ap.add_argument('--graph', default='data/graphs/rect_7x7.json')
    ap.add_argument('--T_max', type=int, default=5)
    ap.add_argument('--thr_L', type=float, default=0.05)
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}

    # Parse bond specs
    bonds = []
    for b in args.bonds:
        parts = b.split(':')
        bonds.append((int(parts[0]), int(parts[1]), int(parts[2])))

    print("=" * 60)
    for ant in args.antennas:
        parts = ant.split(':')
        label, base_log, pert_log = parts[0], parts[1], parts[2]

        try:
            _, _, db = parse_pauli_log(base_log, 'NL')
            _, _, dp = parse_pauli_log(pert_log, 'NL')
        except Exception as e:
            print(f"\n{label}: ERROR {e}")
            continue

        B, N, log_ratio = compute_integral(db, dp, args.T_max)

        print(f"\n{label}:")
        print(f"  {'site':>5}  {'B(v)':>8}  {'N(v)':>8}  {'log(B/N)':>10}")
        print(f"  " + "-"*40)

        # Show all bond endpoints
        all_ep = set()
        for bi, a, b in bonds:
            all_ep.add(a); all_ep.add(b)
        for s in sorted(all_ep):
            si = site_idx[s]
            print(f"  {s:>5}  {B[si]:>8.5f}  {N[si]:>8.5f}  {log_ratio[si]:>+10.4f}")

        print()
        for bi, a, b in bonds:
            la = log_ratio[site_idx[a]]
            lb = log_ratio[site_idx[b]]
            flip = (la > args.thr_L and lb < -args.thr_L) or \
                   (la < -args.thr_L and lb > args.thr_L)
            contrast = abs(la) + abs(lb)
            tag = '✓ FLIP' if flip else '✗ no flip'
            print(f"  Bond {bi:3d} ({a}↔{b}): log({a})={la:+.4f}  log({b})={lb:+.4f}  "
                  f"contrast={contrast:.4f}  {tag}")

    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
