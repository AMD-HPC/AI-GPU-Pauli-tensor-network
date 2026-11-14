#!/usr/bin/env python3
"""
detect_antenna_delta_logT.py — δlnT bond detection on antenna grid logs.

Reads all antenna base/pert log pairs from an antenna_grid/ directory and
computes the δlnT vector field score per bond, aggregated across all antennas.

    δ ln T_{v→u}(t) = ln(n_base(u,t+1)/n_base(v,t)) - ln(n_pert(u,t+1)/n_pert(v,t))

Usage:
  python3 scripts/detect_antenna_delta_logT.py \
      --logdir logs/mawgm_16x32/trial_1 \
      --broken logs/mawgm_16x32/trial_1/broken.json \
      --graph data/graphs/rect_16x32.json \
      --top_k 1
"""

import argparse
import glob
import json
import numpy as np
import os
import sys
sys.path.insert(0, 'scripts')
from searcher import load_graph, build_adj, get_bond_sites, parse_pauli_log

FLOOR = 1e-7


def compute_scores(antenna_dir, g, site_idx, thr_signal):
    n_bonds = len(g['edges'])
    scores  = np.zeros(n_bonds)
    n_antennas = 0

    base_logs = sorted(glob.glob(f'{antenna_dir}/*_base.log'))
    for base_log in base_logs:
        pert_log = base_log.replace('_base.log', '_pert.log')
        if not os.path.exists(pert_log):
            continue
        try:
            _, _, db = parse_pauli_log(base_log, 'NL')
            _, _, dp = parse_pauli_log(pert_log, 'NL')
        except Exception as e:
            print(f"  skip {os.path.basename(base_log)}: {e}")
            continue

        n_antennas += 1
        n_steps = min(db.shape[0], dp.shape[0]) - 1

        for t in range(n_steps):
            nb_t  = db[t];   np_t  = dp[t]
            nb_t1 = db[t+1]; np_t1 = dp[t+1]

            for bi in range(n_bonds):
                u, v = get_bond_sites(g['edges'][bi])
                su = site_idx[u]; sv = site_idx[v]

                for src, dst in [(su, sv), (sv, su)]:
                    if (nb_t[src]  > FLOOR and np_t[src]  > FLOOR and
                            nb_t1[dst] > FLOOR and np_t1[dst] > FLOOR):
                        dT = (np.log(nb_t1[dst] / nb_t[src]) -
                              np.log(np_t1[dst] / np_t[src]))
                        if abs(dT) > thr_signal:
                            scores[bi] += abs(dT)

    print(f"  Processed {n_antennas} antenna pairs")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',  required=True)
    ap.add_argument('--graph',   default='data/graphs/rect_16x32.json')
    ap.add_argument('--broken',  default=None)
    ap.add_argument('--top_k',   type=int,   default=1)
    ap.add_argument('--thr',     type=float, default=0.01)
    args = ap.parse_args()

    antenna_dir = f'{args.logdir}/antenna_grid'
    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}

    true_bonds = set()
    if args.broken:
        br = json.load(open(args.broken))
        true_bonds = set(br.get('bonds', [br.get('bond', -1)]))
        print(f"True bonds: {sorted(true_bonds)}")

    scores  = compute_scores(antenna_dir, g, site_idx, args.thr)
    ranked  = np.argsort(-scores)
    declared = set(int(ranked[i]) for i in range(args.top_k))

    NCOLS = 32
    print(f"\nδlnT detection  thr={args.thr}  top_k={args.top_k}")
    print(f"\n{'Rank':>4} {'Bond':>5} {'sites (row,col)':>22} {'Score':>10} {'TRUE?':>6}")
    print("-" * 55)
    for rank in range(min(args.top_k + 9, len(ranked))):
        bi = int(ranked[rank])
        if scores[bi] == 0:
            break
        u, v = get_bond_sites(g['edges'][bi])
        ru, cu = (u-1)//NCOLS, (u-1)%NCOLS
        rv, cv = (v-1)//NCOLS, (v-1)%NCOLS
        true_tag = '<- TRUE' if bi in true_bonds else ''
        marker   = '<--' if rank < args.top_k else ''
        print(f"  {rank+1:>2}  {bi:>5}  {u:>3}({ru},{cu:2d})↔{v:>3}({rv},{cv:2d})"
              f"  {scores[bi]:>10.4f}  {true_tag} {marker}")

    if true_bonds:
        found = declared & true_bonds
        tag = 'FOUND' if true_bonds.issubset(declared) else ('PARTIAL' if found else 'MISSED')
        print(f"\nResult: {tag}  declared={sorted(declared)}  true={sorted(true_bonds)}")
        # Also report rank of true bond
        for tb in sorted(true_bonds):
            rank_tb = int(np.where(ranked == tb)[0][0]) + 1
            print(f"  bond {tb} rank: {rank_tb}  score: {scores[tb]:.6f}")


if __name__ == '__main__':
    main()
