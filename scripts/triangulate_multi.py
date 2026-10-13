#!/usr/bin/env python3
"""
triangulate_multi.py — Multi-bond detection via timing-consistent triangulation.

Algorithm:
  For each scan s, for each cycle t:
    1. Find ALL sign-flip bonds (RED endpoint ↔ BLUE neighbor)
    2. For each candidate bond: check timing consistency
       expected_t = d(nearest_center, bond) / v_hop
       consistent if |t - expected_t| <= tolerance
    3. A bond is confirmed when multiple scans see it with consistent timing

No masking, no iteration — simultaneous multi-bond detection.

Usage:
  python3 scripts/triangulate_multi.py --logdir logs/mawgm_f64_2b/trial_4 \
      --plan data/scans/rect_7x7_scan_3gpu.json \
      --graph data/graphs/rect_7x7.json
"""

import argparse
import json
import numpy as np
import sys
sys.path.insert(0, 'scripts')
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites
from find_centers import bond_distances, bfs_distances

FLOOR = 1e-7


def compute_L_cycles(db, dp, thr, T_max):
    """L(v,t) for each cycle up to T_max."""
    nc = min(db.shape[0], dp.shape[0], T_max)
    ns = db.shape[1]
    L = np.zeros((nc, ns))
    for t in range(nc):
        for v in range(ns):
            if db[t, v] > thr:
                L[t, v] = -np.log(max(dp[t, v], FLOOR) / db[t, v])
    return L


def find_all_signflip_bonds(L_t, adj, site_idx, all_sites, bonds_covered,
                             g, thr_L=0.05, thr_blue=-0.05):
    """Find all sign-flip bonds at a given cycle."""
    from searcher import get_bond_sites
    candidates = []
    for bi in bonds_covered:
        a, b = get_bond_sites(g['edges'][bi])
        La = L_t[site_idx[a]]
        Lb = L_t[site_idx[b]]
        # RED at a, BLUE at b
        if La > thr_L and Lb < thr_blue:
            contrast = La - Lb
            candidates.append((bi, b, a, contrast))  # blue=b, red=a
        # RED at b, BLUE at a
        elif Lb > thr_L and La < thr_blue:
            contrast = Lb - La
            candidates.append((bi, a, b, contrast))  # blue=a, red=b
    return candidates


def triangulate_multi(logdir, plan, g, adj, all_sites, site_idx, v_hop,
                      thr=1e-3, thr_L=0.05, thr_blue=-0.05,
                      T_max=3, t_tol=1.5, verbose=True):
    """
    Multi-bond simultaneous detection with timing consistency.

    For each scan × cycle: find all sign-flip bonds.
    Score each bond by: sum over scans of 1/t* when timing is consistent.
    Consistent: |t* - expected_t| <= t_tol cycles.

    Returns: ranked list of (bond_idx, score, evidence)
    """
    # Precompute bond crossing distances from each center
    all_centers = [c for sc in plan['placements'] for c in sc['centers']]
    center_bond_dist = {c: bond_distances(g, c) for c in set(all_centers)}

    bond_evidence = {}  # bond_idx -> list of (scan, t, contrast, expected_t, consistent)

    for sc in plan['placements']:
        sid     = sc['scan']
        centers = sc['centers']
        bonds_covered = set(sc['bonds_covered'])

        try:
            _, _, db = parse_pauli_log(f'{logdir}/scan{sid}_base.log', 'NL')
            _, _, dp = parse_pauli_log(f'{logdir}/scan{sid}_pert.log', 'NL')
        except Exception as e:
            if verbose: print(f"  Scan {sid}: ERROR {e}")
            continue

        L = compute_L_cycles(db, dp, thr, T_max)
        nc = L.shape[0]

        # For each cycle, find all sign-flip bonds
        for t in range(nc):
            candidates = find_all_signflip_bonds(
                L[t], adj, site_idx, all_sites, bonds_covered, g,
                thr_L=thr_L, thr_blue=thr_blue)

            for bi, blue_site, red_site, contrast in candidates:
                # Expected arrival time = when wave reaches the BLUE (wall face) endpoint
                # Use min site distance from nearest center (not crossing distance)
                from find_centers import bfs_distances
                d_to_blue = min(bfs_distances(g, c).get(blue_site, 9999) for c in centers)
                expected_t = d_to_blue / v_hop
                consistent = abs((t+1) - expected_t) <= t_tol

                if bi not in bond_evidence:
                    bond_evidence[bi] = []
                bond_evidence[bi].append({
                    'scan': sid, 't': t+1, 'contrast': contrast,
                    'expected_t': expected_t, 'consistent': consistent,
                    'blue_site': blue_site, 'red_site': red_site
                })

    # Score each bond
    bond_scores = {}
    for bi, evs in bond_evidence.items():
        consistent_evs = [e for e in evs if e['consistent']]
        if not consistent_evs:
            continue
        # Score = sum of contrast/t weighted by number of consistent scans
        n_scans = len(set(e['scan'] for e in consistent_evs))
        score = sum(e['contrast'] / e['t'] for e in consistent_evs) * n_scans
        bond_scores[bi] = (score, consistent_evs)

    # Rank bonds
    ranked = sorted(bond_scores.items(), key=lambda x: -x[1][0])

    if verbose:
        print(f"\nAll sign-flip bonds with timing consistency:")
        print(f"{'Bond':>5}  {'sites':>8}  {'score':>8}  {'scans':>6}  evidence")
        print("-"*70)
        for bi, (score, evs) in ranked[:8]:
            a, b = get_bond_sites(g['edges'][bi])
            n_scans = len(set(e['scan'] for e in evs))
            ev_str = '  '.join([f"s{e['scan']}@t{e['t']}(c={e['contrast']:.2f},exp={e['expected_t']:.1f})"
                                 for e in evs])
            print(f"  {bi:>3}  {a:>3}↔{b:>3}  {score:>8.3f}  {n_scans:>6}  {ev_str}")

    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',  required=True)
    ap.add_argument('--plan',    default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--graph',   default='data/graphs/rect_7x7.json')
    ap.add_argument('--v_hop',   type=float, default=0.448)
    ap.add_argument('--thr',     type=float, default=1e-3)
    ap.add_argument('--thr_L',   type=float, default=0.05)
    ap.add_argument('--thr_blue',type=float, default=-0.05)
    ap.add_argument('--T_max',   type=int,   default=3)
    ap.add_argument('--t_tol',   type=float, default=1.5)
    ap.add_argument('--top_k',   type=int,   default=2)
    ap.add_argument('--broken',  default=None)
    ap.add_argument('--verbose', action='store_true', default=True)
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    plan = json.load(open(args.plan))

    from searcher import get_bond_sites

    print(f"Multi-bond triangulation: {args.logdir}")
    ranked = triangulate_multi(
        args.logdir, plan, g, adj, all_sites, site_idx,
        v_hop=args.v_hop, thr=args.thr, thr_L=args.thr_L,
        thr_blue=args.thr_blue, T_max=args.T_max, t_tol=args.t_tol,
        verbose=args.verbose)

    declared = [bi for bi, _ in ranked[:args.top_k]]
    print(f"\nTop-{args.top_k} declared bonds: {declared}")
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
