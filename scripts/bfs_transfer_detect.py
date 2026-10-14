#!/usr/bin/env python3
"""
bfs_transfer_detect.py — Bond detection via sequential BFS transfer ratios.

For each BFS step v→u (one bond), compute:
  ratio = T_base / T_pert = [n_b(u,t+1)/n_b(v,t)] / [n_p(u,t+1)/n_p(v,t)]

ratio >> 1: broken bond (reduced transfer)
ratio << 1: BLUE accumulation (wall face — can't drain)
ratio ≈ 1:  healthy bond

This is fundamentally different from the frontier L approach:
- L compares n_pert vs n_base at the SAME cycle at a single site
- Transfer ratio compares how much weight flows THROUGH each bond per BFS step
- Directly measures the coupling change normalized by source signal
- Sequential: BFS step by step, each bond scored independently

Usage:
  python3 scripts/bfs_transfer_detect.py --logdir logs/mawgm_f64_2b/trial_5 \
      --plan data/scans/rect_7x7_scan_3gpu.json \
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


def compute_transfer_ratios(db, dp, centers, g, adj, all_sites, site_idx,
                             thr=1e-3, T_max=5):
    """
    For each BFS step v→u, compute transfer ratio T_base/T_pert.

    Returns dict: bond_idx → list of (ratio, t, nb_v, nb_u, np_v, np_u)
    """
    t_base = threshold_crossing(db, thr, T_max)
    hop_bfs, color_bfs, parent_bond = bfs_with_color(centers, adj, all_sites, g)

    bond_ratios = {}

    for t in range(1, T_max):
        # Sites crossing threshold at cycle t+1 = BFS frontier
        frontier_next = [s for s in all_sites if t_base[site_idx[s]] == t+1]

        for u in frontier_next:
            ui = site_idx[u]
            pb = parent_bond.get(u, -1)
            if pb < 0: continue  # center itself

            # Find parent site v
            v = None
            for neighbor, bi in adj[u]:
                if bi == pb:
                    v = neighbor; break
            if v is None: continue

            vi = site_idx[v]
            tb_v = t_base[vi]
            if tb_v != t: continue  # parent should be at cycle t

            nb_v = db[t-1, vi]; nb_u = db[t, ui]
            np_v = dp[t-1, vi]; np_u = dp[t, ui]

            if nb_v < thr / 10: continue  # source too weak

            T_base = nb_u / nb_v if nb_v > FLOOR else 0
            T_pert = np_u / max(np_v, FLOOR)
            if T_pert < FLOOR: continue

            ratio = T_base / T_pert

            if pb not in bond_ratios:
                bond_ratios[pb] = []
            bond_ratios[pb].append((ratio, t, nb_v, nb_u, np_v, np_u, u, v))

    return bond_ratios


def bfs_transfer_detect(logdir, plan, g, adj, all_sites, site_idx,
                         thr=1e-3, ratio_thr=1.05, T_max=5, verbose=True):
    """
    Multi-scan bond detection via sequential BFS transfer ratios.

    For each scan, compute transfer ratio for every BFS step bond.
    Score = max(ratio - 1, 0) weighted by 1/t (earlier = more reliable).
    Aggregate across scans.
    """
    bond_score  = {}
    bond_detail = {}

    for sc in plan['placements']:
        sid     = sc['scan']
        centers = sc['centers']

        try:
            _, _, db = parse_pauli_log(f'{logdir}/scan{sid}_base.log', 'NL')
            _, _, dp = parse_pauli_log(f'{logdir}/scan{sid}_pert.log', 'NL')
        except Exception as e:
            if verbose: print(f"  Scan {sid}: ERROR {e}")
            continue

        bond_ratios = compute_transfer_ratios(
            db, dp, centers, g, adj, all_sites, site_idx, thr, T_max)

        if verbose:
            print(f"\n  Scan {sid} centers={centers}:")
            print(f"  {'bond':>5}  {'sites':>8}  {'t':>4}  {'ratio':>8}  {'T_base':>8}  "
                  f"{'T_pert':>8}  {'note'}")
            print(f"  " + "-"*65)

        for bi, entries in bond_ratios.items():
            # Use max ratio across all BFS steps for this bond
            best = max(entries, key=lambda x: x[0])
            ratio, t, nb_v, nb_u, np_v, np_u, u, v = best

            if ratio < ratio_thr: continue  # not broken

            # Score: deviation from 1, weighted by BFS step earliness
            score = (ratio - 1.0) / t

            if bi not in bond_score:
                bond_score[bi]  = 0.0
                bond_detail[bi] = []
            bond_score[bi] += score
            bond_detail[bi].append({
                'scan': sid, 't': t, 'ratio': ratio,
                'T_base': nb_u/nb_v if nb_v>FLOOR else 0,
                'T_pert': np_u/max(np_v,FLOOR),
                'u': u, 'v': v
            })

            if verbose:
                a, b = get_bond_sites(g['edges'][bi])
                note = '← BROKEN' if ratio > 1.5 else ('← weak signal' if ratio < 1.2 else '')
                print(f"  {bi:>5}  {v:>3}→{u:>3}  {t:>4}  {ratio:>8.4f}  "
                      f"{nb_u/nb_v if nb_v>FLOOR else 0:>8.4f}  "
                      f"{np_u/max(np_v,FLOOR):>8.4f}  {note}")

    ranked = sorted(bond_score.items(), key=lambda x: -x[1])

    if verbose and ranked:
        print(f"\n  Bond ranking (transfer ratio method):")
        for bi, score in ranked[:8]:
            a, b = get_bond_sites(g['edges'][bi])
            evs = bond_detail[bi]
            best_ratio = max(e['ratio'] for e in evs)
            n_sc = len(set(e['scan'] for e in evs))
            print(f"    Bond {bi:3d} ({a}↔{b})  score={score:.4f}  "
                  f"best_ratio={best_ratio:.3f}  n_scans={n_sc}")

    return ranked, bond_detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',     required=True)
    ap.add_argument('--plan',       default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--graph',      default='data/graphs/rect_7x7.json')
    ap.add_argument('--thr',        type=float, default=1e-3)
    ap.add_argument('--ratio_thr',  type=float, default=1.05)
    ap.add_argument('--T_max',      type=int,   default=5)
    ap.add_argument('--top_k',      type=int,   default=2)
    ap.add_argument('--broken',     default=None)
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    plan = json.load(open(args.plan))

    print(f"BFS transfer ratio detection: {args.logdir}")
    ranked, detail = bfs_transfer_detect(
        args.logdir, plan, g, adj, all_sites, site_idx,
        thr=args.thr, ratio_thr=args.ratio_thr,
        T_max=args.T_max, verbose=True)

    declared = [bi for bi, _ in ranked[:args.top_k]]
    print(f"\nTop-{args.top_k} declared: {declared}")
    for bi in declared:
        a, b = get_bond_sites(g['edges'][bi])
        print(f"  Bond {bi}: {a}↔{b}")

    if args.broken:
        br = json.load(open(args.broken))
        true_bonds = set(br.get('bonds', [br.get('bond', -1)]))
        found = set(declared)
        both    = true_bonds.issubset(found)
        partial = bool(true_bonds & found) and not both
        tag = '✓ BOTH' if both else ('~ PARTIAL' if partial else '✗ NONE')
        print(f"\nGround truth: {sorted(true_bonds)}")
        print(f"Result: {tag}")

if __name__ == '__main__':
    main()
