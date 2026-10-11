#!/usr/bin/env python3
"""
triangulate.py — Wavefront triangulation across scans.

For each scan:
  1. Find first cycle t* where signal appears (blue site = wall face)
  2. Find red site = shadow (neighbor of blue with opposite sign)
  3. Candidate bond = blue↔red, with direction constraint:
     center is on blue side, d(center, blue) ≈ t* × v_hop

Cross-scan consistency:
  - Left center: BLUE at u, RED at v → bond u↔v
  - Right center: RED at u, BLUE at v → bond u↔v (same bond, opposite approach)
  Both constraints consistent ↔ same bond, correct distance-time for each center.

Usage:
  python3 scripts/triangulate.py --logdir logs/mawgm_v2/trial_4 \
      --plan data/scans/rect_7x7_scan_3gpu.json \
      --graph data/graphs/rect_7x7.json \
      --v_hop 0.448
"""

import argparse
import json
import numpy as np
import sys
sys.path.insert(0, 'scripts')
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites
from find_centers import bfs_distances

FLOOR = 1e-7

def compute_L_percycle(db, dp, thr, nc):
    """L(v,t) for each cycle up to nc."""
    ns = db.shape[1]
    L = np.zeros((nc, ns))
    for t in range(nc):
        for v in range(ns):
            if db[t, v] > thr:
                L[t, v] = -np.log(max(dp[t, v], FLOOR) / db[t, v])
    return L

def first_signal_cycle(L, thr_L=0.05):
    """First cycle where any site has |L| > thr_L. Returns t (0-indexed) or -1."""
    nc, ns = L.shape
    for t in range(nc):
        if np.any(np.abs(L[t]) > thr_L):
            return t
    return -1

def scan_triangulate(db, dp, centers, bonds_covered, g, adj, all_sites, site_idx,
                     v_hop, thr=1e-3, thr_L=0.05, T_CLEAN=3):
    """
    For one scan: find (blue_site, red_site, candidate_bond, t_star, nearest_center).
    Returns None if no clear signal.
    """
    ns = db.shape[1]
    nc = min(db.shape[0], dp.shape[0], T_CLEAN)
    L = compute_L_percycle(db, dp, thr, nc)

    # Precompute distances from each center to all sites
    center_dist = {c: bfs_distances(g, c) for c in centers}

    t_star = first_signal_cycle(L, thr_L)
    if t_star < 0:
        return None

    # At t_star: find blue site (most negative L) among bonds_covered endpoints
    blue_site = None; blue_L = 0.0
    red_site  = None; red_L  = 0.0

    # Only consider endpoints of bonds in this scan's coverage
    candidate_sites = set()
    for bi in bonds_covered:
        a, b = get_bond_sites(g['edges'][bi])
        candidate_sites.add(a); candidate_sites.add(b)

    # Step 1: find BLUE site (most negative L) among covered bond endpoints
    for s in candidate_sites:
        v = site_idx[s]
        lv = L[t_star, v]
        if lv < blue_L: blue_L = lv; blue_site = s

    if blue_site is None or abs(blue_L) < thr_L:
        return None

    # Step 2: find RED site among NEIGHBORS of blue_site.
    # RED appears one cycle AFTER BLUE — look at t_star AND t_star+1.
    # The broken bond is between blue (wall face) and red (shadow).
    for t_red in [t_star, min(t_star + 1, nc - 1)]:
        for neighbor, bond_idx in adj[blue_site]:
            if bond_idx not in bonds_covered:
                continue
            v = site_idx[neighbor]
            lv = L[t_red, v]
            if lv > red_L:
                red_L = lv; red_site = neighbor

    if red_site is None:
        return None

    # Find nearest center to blue_site (wall face — center is on this side)
    nearest_c = min(centers, key=lambda c: center_dist[c].get(blue_site, 9999))
    d_blue = center_dist[nearest_c].get(blue_site, 9999)
    expected_t = d_blue / v_hop  # expected cycle for signal to appear

    # Candidate bond: bond between blue_site and red_site (must be neighbors)
    bond_candidate = None
    for neighbor, bond_idx in adj[blue_site]:
        if neighbor == red_site:
            bond_candidate = bond_idx
            break

    # If not direct neighbors, find bond incident to blue_site pointing toward red_site
    if bond_candidate is None:
        # Use BFS direction: pick bond from blue_site toward red_site
        dist_from_blue = bfs_distances(g, blue_site)
        for neighbor, bond_idx in adj[blue_site]:
            if bond_idx in bonds_covered:
                d_neighbor_to_red = bfs_distances(g, neighbor).get(red_site, 9999)
                if d_neighbor_to_red < dist_from_blue.get(red_site, 9999):
                    bond_candidate = bond_idx
                    break

    return {
        'blue_site':     blue_site,
        'red_site':      red_site,
        'blue_L':        blue_L,
        'red_L':         red_L,
        't_star':        t_star + 1,  # 1-indexed
        'nearest_center': nearest_c,
        'd_center_blue': d_blue,
        'expected_t':    expected_t,
        'bond_candidate': bond_candidate,
        'contrast':       abs(blue_L) + abs(red_L),
    }

def triangulate(logdir, plan, g, adj, all_sites, site_idx, v_hop,
                thr=1e-3, thr_L=0.05, T_CLEAN=3, verbose=True):
    """
    Run triangulation across all scans. Returns declared bond index.
    """
    results = []
    for sc in plan['placements']:
        sid     = sc['scan']
        centers = sc['centers']
        bonds_covered = sc['bonds_covered']
        try:
            _, _, db = parse_pauli_log(f'{logdir}/scan{sid}_base.log', 'NL')
            _, _, dp = parse_pauli_log(f'{logdir}/scan{sid}_pert.log', 'NL')
        except Exception as e:
            if verbose: print(f"  Scan {sid}: ERROR {e}")
            continue

        r = scan_triangulate(db, dp, centers, bonds_covered, g, adj, all_sites,
                             site_idx, v_hop, thr, thr_L, T_CLEAN)
        if r is None:
            if verbose: print(f"  Scan {sid}: no signal")
            continue

        if verbose:
            bc = r['bond_candidate']
            ea, eb = get_bond_sites(g['edges'][bc]) if bc is not None else (0,0)
            print(f"  Scan {sid}: t*={r['t_star']}  "
                  f"BLUE=site{r['blue_site']}(L={r['blue_L']:+.3f})  "
                  f"RED=site{r['red_site']}(L={r['red_L']:+.3f})  "
                  f"nearest_c={r['nearest_center']}  d={r['d_center_blue']:.0f}hops  "
                  f"expect_t={r['expected_t']:.1f}  "
                  f"bond={bc}({ea}↔{eb})")
        results.append(r)

    if not results:
        return -1

    # Cross-scan consistency:
    # The true bond has: BLUE=u in one scan, BLUE=v in another scan (opposite sides).
    # Find the bond (u,v) where u appears as BLUE in one scan AND v appears as BLUE
    # in another scan — this is the unique consistent bond across scans.
    blue_sites = [r['blue_site'] for r in results]

    # Check all pairs of results for opposite-side consistency
    consistent_bonds = {}
    for i, ri in enumerate(results):
        for j, rj in enumerate(results):
            if i >= j: continue
            # Opposite sides: ri.blue and rj.blue should be endpoints of same bond
            for neighbor, bond_idx in adj[ri['blue_site']]:
                if neighbor == rj['blue_site']:
                    # Found: ri center on ri.blue side, rj center on rj.blue side
                    # Both approach the same bond from opposite sides ✓
                    weight = 1.0/ri['t_star'] + 1.0/rj['t_star']
                    consistent_bonds[bond_idx] = consistent_bonds.get(bond_idx, 0.0) + weight
                    if verbose:
                        ea,eb=get_bond_sites(g['edges'][bond_idx])
                        print(f"  Consistency: scan{ri['scan'] if 'scan' in ri else '?'} "
                              f"BLUE={ri['blue_site']} + scan{rj['scan'] if 'scan' in rj else '?'} "
                              f"BLUE={rj['blue_site']} → bond {bond_idx}({ea}↔{eb}) weight={weight:.2f}")

    if consistent_bonds:
        declared = max(consistent_bonds, key=consistent_bonds.get)
        if verbose:
            print(f"\n  Consistent bonds: {dict(sorted(consistent_bonds.items(), key=lambda x: -x[1]))}")
    else:
        # Fallback: weighted vote on candidate bonds
        bond_score = {}
        for r in results:
            bc = r['bond_candidate']
            if bc is None: continue
            weight = 1.0 / r['t_star']
            bond_score[bc] = bond_score.get(bc, 0.0) + weight
        declared = max(bond_score, key=bond_score.get) if bond_score else -1
        if verbose:
            print(f"\n  No consistent pair — fallback bond scores: {bond_score}")

    if declared >= 0 and verbose:
        ea, eb = get_bond_sites(g['edges'][declared])
        print(f"  Triangulation declares: Bond {declared} ({ea}↔{eb})")

    # Store scan index in results for verbose output
    for k, r in enumerate(results):
        r['scan'] = plan['placements'][k]['scan'] if k < len(plan['placements']) else k

    return declared

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',  required=True)
    ap.add_argument('--plan',    default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--graph',   default='data/graphs/rect_7x7.json')
    ap.add_argument('--v_hop',   type=float, default=0.448)
    ap.add_argument('--thr',     type=float, default=1e-3)
    ap.add_argument('--thr_L',   type=float, default=0.05)
    ap.add_argument('--T_CLEAN', type=int,   default=3)
    ap.add_argument('--broken',  default=None)
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    plan = json.load(open(args.plan))

    print(f"Triangulating: {args.logdir}")
    declared = triangulate(args.logdir, plan, g, adj, all_sites, site_idx,
                           args.v_hop, args.thr, args.thr_L, args.T_CLEAN)

    if args.broken:
        br = json.load(open(args.broken))
        true_bonds = set(br.get('bonds', [br.get('bond', -1)]))
        correct = declared in true_bonds
        ea, eb = get_bond_sites(g['edges'][declared]) if declared >= 0 else (0,0)
        print(f"\n  Ground truth: {sorted(true_bonds)}")
        print(f"  Declared: Bond {declared} ({ea}↔{eb})")
        print(f"  Result: {'✓ CORRECT' if correct else '✗ WRONG'}")

def triangulate_iterative(logdir, plan, g, adj, all_sites, site_idx, v_hop,
                          thr=1e-3, thr_L=0.05, T_max=3, verbose=True):
    """
    Iterative bond peeling:
    1. Triangulate → declare bond b*
    2. Mask b* endpoints in L field
    3. Repeat until no clean signal (t* > T_max or no consistent pair)

    Termination:
    - No consistent opposite-side BLUE pair found
    - t* > T_max (interference, not genuine first arrival)
    - Signal below thr_L

    Returns list of found bonds.
    """
    # Load all L fields once
    scan_data = []
    for sc in plan['placements']:
        sid = sc['scan']
        try:
            _, _, db = parse_pauli_log(f'{logdir}/scan{sid}_base.log', 'NL')
            _, _, dp = parse_pauli_log(f'{logdir}/scan{sid}_pert.log', 'NL')
        except:
            scan_data.append(None)
            continue
        nc = min(db.shape[0], dp.shape[0], T_max)
        ns = db.shape[1]
        L = compute_L_percycle(db, dp, thr, nc)
        scan_data.append({'L': L.copy(), 'L_orig': L.copy(),
                          'db': db, 'dp': dp,
                          'centers': sc['centers'],
                          'bonds_covered': set(sc['bonds_covered']),
                          'sid': sid})

    found_bonds = []
    masked_sites = set()

    iteration = 0
    while True:
        iteration += 1
        if verbose: print(f"\n--- Iteration {iteration} ---")

        # Apply masks: zero out L at masked sites
        for sd in scan_data:
            if sd is None: continue
            for s in masked_sites:
                sd['L'][:, site_idx[s]] = 0.0

        # Run triangulation on current L
        results = []
        for sc, sd in zip(plan['placements'], scan_data):
            if sd is None: continue
            L = sd['L']
            nc, ns = L.shape

            t_star = first_signal_cycle(L, thr_L)
            if t_star < 0 or t_star >= T_max:
                if verbose: print(f"  Scan {sd['sid']}: no signal within T_max={T_max}")
                continue

            # Find BLUE site
            blue_site = None; blue_L = 0.0
            for s in [s for bi in sd['bonds_covered']
                      for s in get_bond_sites(g['edges'][bi])]:
                if s in masked_sites: continue
                lv = L[t_star, site_idx[s]]
                if lv < blue_L: blue_L = lv; blue_site = s

            if blue_site is None or abs(blue_L) < thr_L: continue

            # Find RED among neighbors at t_star and t_star+1
            red_site = None; red_L = 0.0
            for t_red in [t_star, min(t_star+1, nc-1)]:
                for neighbor, bond_idx in adj[blue_site]:
                    if bond_idx not in sd['bonds_covered']: continue
                    if neighbor in masked_sites: continue
                    lv = L[t_red, site_idx[neighbor]]
                    if lv > red_L: red_L = lv; red_site = neighbor

            nearest_c = min(sd['centers'],
                            key=lambda c: bfs_distances(g, c).get(blue_site, 9999))

            if verbose:
                print(f"  Scan {sd['sid']}: t*={t_star+1}  "
                      f"BLUE=site{blue_site}(L={blue_L:+.3f})  "
                      f"RED=site{red_site}(L={red_L:+.3f})  c={nearest_c}")

            results.append({'blue_site': blue_site, 'red_site': red_site,
                            'blue_L': blue_L, 'red_L': red_L,
                            't_star': t_star+1, 'nearest_center': nearest_c})

        if not results:
            if verbose: print("  No signal — stopping.")
            break

        # Cross-scan consistency: find bond with opposite-side BLUEs
        consistent_bonds = {}
        for i, ri in enumerate(results):
            for j, rj in enumerate(results):
                if i >= j: continue
                for neighbor, bond_idx in adj[ri['blue_site']]:
                    if neighbor == rj['blue_site']:
                        weight = 1.0/ri['t_star'] + 1.0/rj['t_star']
                        consistent_bonds[bond_idx] = \
                            consistent_bonds.get(bond_idx, 0.0) + weight

        if not consistent_bonds:
            # Fallback: single-scan detection only if RED is strong (>0.5)
            # — avoids chasing noise bonds
            MIN_RED = 0.5
            for r in results:
                if r['blue_site'] and r['red_site'] and r.get('red_L', 0) > MIN_RED:
                    for neighbor, bond_idx in adj[r['blue_site']]:
                        if neighbor == r['red_site']:
                            if verbose:
                                ea,eb=get_bond_sites(g['edges'][bond_idx])
                                print(f"  Single-scan fallback (RED={r['red_L']:.3f}): "
                                      f"Bond {bond_idx} ({ea}↔{eb})")
                            consistent_bonds[bond_idx] = 1.0/r['t_star']
                            break
                if consistent_bonds: break
            if not consistent_bonds:
                if verbose: print("  No consistent pair — stopping.")
                break

        declared = max(consistent_bonds, key=consistent_bonds.get)
        ea, eb = get_bond_sites(g['edges'][declared])
        if verbose:
            print(f"  → Declared: Bond {declared} ({ea}↔{eb})")

        found_bonds.append(declared)

        # Mask endpoints of declared bond
        masked_sites.add(ea)
        masked_sites.add(eb)

    return found_bonds


if __name__ == '__main__':
    main()
