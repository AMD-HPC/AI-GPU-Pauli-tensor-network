#!/usr/bin/env python3
"""
detect_delta_logT.py — Bond fault detection via delta log-transfer score.

For each directed bond (v→u) and each cycle t, computes:

    δ ln T_{v→u}(t) = ln(n_base(u,t+1)/n_base(v,t)) - ln(n_pert(u,t+1)/n_pert(v,t))

The bond score aggregates |δ ln T| above a threshold across all directions,
all cycles, and all scans. The broken bond is the one with the highest score.

Usage:
  python3 scripts/detect_delta_logT.py \
      --logdir logs/mawgm_f64/trial_4 \
      --broken logs/mawgm_f64/trial_4/broken.json \
      --top_k 1

  # Run all 10 trials:
  for t in $(seq 1 10); do
      python3 scripts/detect_delta_logT.py \
          --logdir logs/mawgm_f64/trial_$t \
          --broken logs/mawgm_f64/trial_$t/broken.json
  done
"""

import argparse
import json
import numpy as np
import sys
sys.path.insert(0, 'scripts')
from searcher import load_graph, build_adj, get_bond_sites, parse_pauli_log

FLOOR = 1e-7


def compute_scores(logdir, plan, g, site_idx, thr_signal):
    """Compute δlnT bond scores from all scans."""
    n_bonds = len(g['edges'])
    scores = np.zeros(n_bonds)

    for sc in plan['placements']:
        sid = sc['scan']
        try:
            _, _, db = parse_pauli_log(f'{logdir}/scan{sid}_base.log', 'NL')
            _, _, dp = parse_pauli_log(f'{logdir}/scan{sid}_pert.log', 'NL')
        except Exception as e:
            continue

        n_steps = min(db.shape[0], dp.shape[0]) - 1
        for t in range(n_steps):
            nb_t = db[t]; np_t = dp[t]
            nb_t1 = db[t+1]; np_t1 = dp[t+1]

            for bi in range(n_bonds):
                u, v = get_bond_sites(g['edges'][bi])
                su = site_idx[u]; sv = site_idx[v]

                # Both directed versions: u→v and v→u
                for src, dst in [(su, sv), (sv, su)]:
                    if (nb_t[src] > FLOOR and np_t[src] > FLOOR and
                            nb_t1[dst] > FLOOR and np_t1[dst] > FLOOR):
                        dT = (np.log(nb_t1[dst] / nb_t[src]) -
                              np.log(np_t1[dst] / np_t[src]))
                        if abs(dT) > thr_signal:
                            scores[bi] += abs(dT)

    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',  required=True)
    ap.add_argument('--plan',    default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--graph',   default='data/graphs/rect_7x7.json')
    ap.add_argument('--broken',  default=None)
    ap.add_argument('--top_k',   type=int,   default=1)
    ap.add_argument('--thr',     type=float, default=0.01,
                    help='Minimum |δlnT| to count toward score')
    args = ap.parse_args()

    g    = load_graph(args.graph)
    plan = json.load(open(args.plan))
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}

    true_bonds = set()
    if args.broken:
        br = json.load(open(args.broken))
        true_bonds = set(br.get('bonds', [br.get('bond', -1)]))

    scores  = compute_scores(args.logdir, plan, g, site_idx, args.thr)
    ranked  = np.argsort(-scores)
    declared = set(int(ranked[i]) for i in range(args.top_k))

    print(f"δlnT detection  thr={args.thr}  top_k={args.top_k}")
    if true_bonds:
        print(f"True bonds: {sorted(true_bonds)}")
    print(f"\nTop-{args.top_k} declared bonds:")
    print(f"{'Rank':>5} {'Bond':>5} {'Sites':>12} {'Score':>10} {'TRUE?':>7}")
    print("-"*45)
    for rank in range(min(args.top_k + 5, len(ranked))):
        bi = int(ranked[rank])
        u, v = get_bond_sites(g['edges'][bi])
        true_tag = '✓' if bi in true_bonds else ''
        marker = '<--' if rank < args.top_k else ''
        print(f"  {rank+1:>3}  {bi:>5}  ({u:>2},{v:>2})  {scores[bi]:>10.4f}  {true_tag:>5} {marker}")

    if true_bonds:
        found = declared & true_bonds
        both  = true_bonds.issubset(declared)
        tag   = '✓ BOTH' if both else (f'~ PARTIAL ({len(found)}/{len(true_bonds)})' if found else '✗ NONE')
        print(f"\nResult: {tag}  declared={sorted(declared)}  true={sorted(true_bonds)}")


if __name__ == '__main__':
    main()
