#!/usr/bin/env python3
"""
integral_detect.py — Bond detection via integrated weight ratio B(v)/N(v).

B(v) = sum_t n_base(v,t) over all cycles  (no threshold needed)
N(v) = sum_t n_pert(v,t) over all cycles

Signal = log(B(v)/N(v)):
  > +theta: RED (shadow) — baseline accumulates more than perturbed
  < -theta: BLUE (wall face) — perturbed accumulates more than baseline

Bond detection: sign flip of log(B/N) across bond endpoints.

More robust than L(v,t0) because:
- No threshold on individual n values
- Captures both delay and sustained amplitude difference
- Works even when single cycles are below measurement threshold

Usage:
  python3 scripts/integral_detect.py \
      --logbase logs/mawgm_v2 \
      --wallbase logs/mawgm_v2 \
      --graph data/graphs/rect_7x7.json
"""

import argparse
import json
import glob
import os
import numpy as np
import sys
sys.path.insert(0, 'scripts')
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites

FLOOR = 1e-7


def compute_integral(db, dp, T_max=5):
    """
    Compute B(v) and N(v) for each site.
    B(v) = sum_t n_base(v,t)
    N(v) = sum_t n_pert(v,t)
    Returns log_ratio[v] = log(B(v)/N(v))
    """
    nc = min(db.shape[0], dp.shape[0], T_max)
    ns = db.shape[1]

    B = np.zeros(ns)
    N = np.zeros(ns)

    for t in range(nc):
        B += db[t]
        N += dp[t]

    # log ratio: positive = RED (baseline > pert), negative = BLUE (pert > baseline)
    log_ratio = np.zeros(ns)
    for v in range(ns):
        if B[v] > FLOOR and N[v] > FLOOR:
            log_ratio[v] = np.log(B[v] / max(N[v], FLOOR))

    return B, N, log_ratio


def check_integral_flip(base_log, pert_log, bond_idx, g, adj, site_idx,
                         theta=0.1, T_max=5):
    """
    Check sign flip in integrated log-ratio at bond endpoints.
    Returns (confirmed, log_a, log_b, B_a, B_b)
    """
    try:
        _, _, db = parse_pauli_log(base_log, 'NL')
        _, _, dp = parse_pauli_log(pert_log, 'NL')
    except:
        return False, 0.0, 0.0, 0.0, 0.0

    B, N, log_ratio = compute_integral(db, dp, T_max)

    a, b = get_bond_sites(g['edges'][bond_idx])
    log_a = log_ratio[site_idx[a]]
    log_b = log_ratio[site_idx[b]]
    B_a = B[site_idx[a]]; B_b = B[site_idx[b]]

    # Sign flip with minimum magnitude
    flip = (log_a > theta and log_b < -theta) or (log_a < -theta and log_b > theta)
    contrast = abs(log_a) + abs(log_b)

    return flip and contrast > theta, log_a, log_b, B_a, B_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logbase',  default='logs/mawgm_v2')
    ap.add_argument('--graph',    default='data/graphs/rect_7x7.json')
    ap.add_argument('--theta',    type=float, default=0.1,
                    help='Threshold on |log(B/N)| for sign flip')
    ap.add_argument('--T_max',    type=int,   default=5)
    ap.add_argument('--trials',   default='1-10')
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}

    trials = list(range(1,11)) if args.trials == '1-10' else \
             [int(x) for x in args.trials.split(',')]

    print(f"=== Integral Detection B(v)/N(v) ===")
    print(f"theta={args.theta}  T_max={args.T_max}")
    print()

    both_count = 0
    for trial in trials:
        logdir = f'{args.logbase}/trial_{trial}'
        broken_file = f'{logdir}/broken.json'
        if not os.path.exists(broken_file):
            continue

        br = json.load(open(broken_file))
        true_bonds = set(br.get('bonds', [br.get('bond', -1)]))

        print(f"Trial {trial}: true bonds={sorted(true_bonds)}")

        # Use seismic scan logs (all 3 scans, aggregate)
        plan = json.load(open('data/scans/rect_7x7_scan_3gpu.json'))
        confirmed = set()
        bond_scores = {}  # bond_idx -> max |log_a - log_b|

        for sc in plan['placements']:
            sid = sc['scan']
            base_log = f'{logdir}/scan{sid}_base.log'
            pert_log  = f'{logdir}/scan{sid}_pert.log'
            if not os.path.exists(base_log): continue

            try:
                _, _, db = parse_pauli_log(base_log, 'NL')
                _, _, dp = parse_pauli_log(pert_log, 'NL')
            except: continue

            B, N, log_ratio = compute_integral(db, dp, args.T_max)

            # Score all bonds by sign flip in integrated log-ratio
            for bi in range(len(g['edges'])):
                a, b = get_bond_sites(g['edges'][bi])
                log_a = log_ratio[site_idx[a]]
                log_b = log_ratio[site_idx[b]]
                flip = (log_a > args.theta and log_b < -args.theta) or \
                       (log_a < -args.theta and log_b > args.theta)
                if flip:
                    contrast = abs(log_a) + abs(log_b)
                    if bi not in bond_scores or contrast > bond_scores[bi][0]:
                        bond_scores[bi] = (contrast, log_a, log_b, sid)

        # Rank and declare top-k bonds
        n_true = len(true_bonds)
        ranked = sorted(bond_scores.items(), key=lambda x: -x[1][0])
        print(f"  Top bonds by integral contrast:")
        for bi, (contrast, log_a, log_b, sid) in ranked[:7]:
            a, b = get_bond_sites(g['edges'][bi])
            true_tag = ' ← TRUE' if bi in true_bonds else ''
            print(f"  Bond {bi:3d} ({a}↔{b}): contrast={contrast:.3f} "
                  f"log_a={log_a:+.3f} log_b={log_b:+.3f} scan={sid}{true_tag}")

        confirmed = set(bi for bi, _ in ranked[:n_true])

        both = true_bonds.issubset(confirmed)
        partial = bool(true_bonds & confirmed) and not both
        result = 'BOTH' if both else ('PARTIAL' if partial else 'NONE')
        found = true_bonds & confirmed
        print(f"  → {result}  confirmed={sorted(confirmed)}  true={sorted(true_bonds)}")
        if both: both_count += 1
        print()

    print(f"=== Integral FINAL: {both_count}/{len(trials)} BOTH ===")


if __name__ == '__main__':
    import os
    main()
