#!/usr/bin/env python3
"""
targeted_aperture.py — Find optimal center for each candidate bond and validate.

Phase 1: Get candidate bonds from transfer ratio method
Phase 2: For each candidate, find optimal center (clean approach, not sandwiched)
Phase 3: Run targeted aperture simulation
Phase 4: Validate with transfer ratio

Usage:
  # Step 1: generate commands
  python3 scripts/targeted_aperture.py --logdir logs/mawgm_f64_2b/trial_5 \
      --plan data/scans/rect_7x7_scan_3gpu.json \
      --graph data/graphs/rect_7x7.json --mode plan

  # Step 2: after running simulations
  python3 scripts/targeted_aperture.py --logdir logs/mawgm_f64_2b/trial_5 \
      --plan data/scans/rect_7x7_scan_3gpu.json \
      --graph data/graphs/rect_7x7.json --mode validate
"""

import argparse
import json
import numpy as np
import sys
sys.path.insert(0, 'scripts')
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites
from find_centers import bfs_distances
from bfs_detect import threshold_crossing, bfs_with_color, bond_color_class
from bfs_transfer_detect import compute_transfer_ratios, bfs_transfer_detect

FLOOR = 1e-7


def find_optimal_center(bond_idx, g, adj, all_sites, site_idx,
                         n_hops_min=2, n_hops_max=3, exclude_sites=None):
    """
    Find best center to approach bond_idx from one clean direction.

    Criteria:
    1. Center is n_hops_min..n_hops_max hops from nearer endpoint
    2. Center is NOT ON either bond endpoint (avoid baseline blocking)
    3. Center is NOT adjacent to the bond (min distance >= n_hops_min)
    4. Maximize |d(center, a) - d(center, b)| — asymmetry = clean direction, no sandwiching

    Key insight: placing center ON bond endpoint makes the broken bond's
    shadow site unreachable in baseline → no transfer to measure. Must
    approach from n_hops_min away on one side only.

    Returns: (best_center_site, approach_endpoint, score)
    """
    if exclude_sites is None:
        exclude_sites = set()

    a, b = get_bond_sites(g['edges'][bond_idx])
    bond_endpoints = {a, b}

    best_center = -1; best_score = -1
    best_endpoint = a

    for s in all_sites:
        if s in exclude_sites: continue
        if s in bond_endpoints: continue  # NEVER place center on bond endpoint

        dist = bfs_distances(g, s)
        da = dist.get(a, 999)
        db_dist = dist.get(b, 999)

        d_min = min(da, db_dist)
        d_max = max(da, db_dist)

        # Must be n_hops_min..n_hops_max from nearer endpoint
        if d_min < n_hops_min or d_min > n_hops_max: continue

        # Asymmetry: center sees one endpoint much closer than other
        # Higher asymmetry = cleaner direction = less sandwiching
        asymmetry = d_max - d_min

        # Proximity score: prefer closer within the valid range
        proximity_score = max(0, n_hops_max + 1 - d_min)

        score = asymmetry * 3 + proximity_score
        if score > best_score:
            best_score = score
            best_center = s
            best_endpoint = a if da < db_dist else b

    return best_center, best_endpoint, best_score


def validate_bond(logdir, bond_idx, center, g, adj, all_sites, site_idx,
                  thr=1e-3, tag=''):
    """
    Validate bond using targeted aperture log.
    Returns transfer ratio for the specific bond.
    """
    apt_base = f'{logdir}/apt_bond{bond_idx}_c{center}_base.log'
    apt_pert = f'{logdir}/apt_bond{bond_idx}_c{center}_pert.log'

    try:
        _, _, db = parse_pauli_log(apt_base, 'NL')
        _, _, dp = parse_pauli_log(apt_pert, 'NL')
    except Exception as e:
        return None, f"ERROR: {e}"

    bond_ratios = compute_transfer_ratios(
        db, dp, [center], g, adj, all_sites, site_idx, thr, T_max=5)

    if bond_idx in bond_ratios:
        best = max(bond_ratios[bond_idx], key=lambda x: x[0])
        ratio = best[0]
        t = best[1]
        return ratio, f"ratio={ratio:.3f} at BFS step {t}"
    else:
        # Check all ratios near bond endpoints
        a, b = get_bond_sites(g['edges'][bond_idx])
        nearby = []
        for bi, entries in bond_ratios.items():
            ea, eb = get_bond_sites(g['edges'][bi])
            if ea in [a,b] or eb in [a,b]:
                nearby.append((max(e[0] for e in entries), bi, ea, eb))
        if nearby:
            nearby.sort(reverse=True)
            r, bi, ea, eb = nearby[0]
            return r, f"nearby bond {bi}({ea}↔{eb}) ratio={r:.3f}"
        return 1.0, "no signal at bond"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',    required=True)
    ap.add_argument('--plan',      default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--graph',     default='data/graphs/rect_7x7.json')
    ap.add_argument('--thr',       type=float, default=1e-3)
    ap.add_argument('--ratio_thr', type=float, default=1.05)
    ap.add_argument('--confirm_thr', type=float, default=1.05,
                    help='Ratio threshold to confirm bond broken')
    ap.add_argument('--top_k',     type=int,   default=5,
                    help='Number of candidates to test')
    ap.add_argument('--T_max',     type=int,   default=5)
    ap.add_argument('--mode',      choices=['plan','validate','both'], default='plan')
    ap.add_argument('--broken',    default=None)
    ap.add_argument('--binary',    default='./main_pauli_sweep_rect7x7')
    ap.add_argument('--chi',       type=int, default=50)
    ap.add_argument('--steps',     type=int, default=5)
    ap.add_argument('--cutoff',    default='1e-3')
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    plan = json.load(open(args.plan))

    # Get broken bond info
    true_bonds = set()
    bond_tag = ''
    if args.broken:
        br = json.load(open(args.broken))
        true_bonds = set(br.get('bonds', [br.get('bond',-1)]))
        bond_tag = ';'.join(map(str, sorted(true_bonds)))

    # Phase 1: get candidates from existing scan logs
    print("=== Phase 1: Candidate bonds from transfer ratio method ===")
    ranked, _ = bfs_transfer_detect(
        args.logdir, plan, g, adj, all_sites, site_idx,
        thr=args.thr, ratio_thr=args.ratio_thr, T_max=args.T_max, verbose=False)

    candidates = [(bi, score) for bi, score in ranked[:args.top_k]]
    print(f"Top {args.top_k} candidates:")
    for bi, score in candidates:
        a, b = get_bond_sites(g['edges'][bi])
        true_tag = ' ← TRUE' if bi in true_bonds else ''
        print(f"  Bond {bi:3d} ({a}↔{b})  score={score:.4f}{true_tag}")

    # Phase 2: find optimal center for each candidate
    print("\n=== Phase 2: Optimal center selection ===")
    center_assignments = []
    used_centers = set()
    for bi, score in candidates:
        a, b = get_bond_sites(g['edges'][bi])
        # Use centers from all scans + any site as potential center
        center, endpoint, cscore = find_optimal_center(
            bi, g, adj, all_sites, site_idx, n_hops_target=2)
        pos = {s: ((s-1)%7, 6-(s-1)//7) for s in range(1,50)}
        true_tag = ' ← TRUE' if bi in true_bonds else ''
        print(f"  Bond {bi:3d} ({a}↔{b}): optimal center={center} "
              f"pos={pos.get(center,'?')} score={cscore:.1f}{true_tag}")
        center_assignments.append((bi, center, score))

    # Phase 3: generate simulation commands or validate
    if args.mode in ['plan', 'both']:
        print("\n=== Phase 3: Simulation commands ===")
        print("# Run inside docker:")
        for bi, center, score in center_assignments:
            a, b = get_bond_sites(g['edges'][bi])
            base_log = f"{args.logdir}/apt_bond{bi}_c{center}_base.log"
            pert_log = f"{args.logdir}/apt_bond{bi}_c{center}_pert.log"
            print(f"# Bond {bi} ({a}↔{b}) via center {center}:")
            print(f"ROCR_VISIBLE_DEVICES=0 {args.binary} {args.chi} {args.steps} "
                  f"{args.cutoff} --center={center} --state=NL > {base_log} 2>&1")
            j_new = br.get('perturbation',{}).get('J_perturbed', 0.8) if args.broken else 0.8
            print(f"ROCR_VISIBLE_DEVICES=1 {args.binary} {args.chi} {args.steps} "
                  f'{args.cutoff} --center={center} --state=NL '
                  f'"--perturb-bond={bond_tag},{j_new:.3f}" '
                  f"> {pert_log} 2>&1")
            print()

    if args.mode in ['validate', 'both']:
        print("\n=== Phase 4: Validation ===")
        confirmed = []
        for bi, center, score in center_assignments:
            a, b = get_bond_sites(g['edges'][bi])
            ratio, detail = validate_bond(
                args.logdir, bi, center, g, adj, all_sites, site_idx, args.thr)
            if ratio is None:
                print(f"  Bond {bi:3d} ({a}↔{b}): {detail}")
                continue
            confirmed_tag = '✓ CONFIRMED' if ratio > args.confirm_thr else '✗ REJECTED'
            true_tag = ' ← TRUE' if bi in true_bonds else ''
            print(f"  Bond {bi:3d} ({a}↔{b}): {confirmed_tag}  {detail}{true_tag}")
            if ratio > args.confirm_thr:
                confirmed.append(bi)

        print(f"\nConfirmed broken bonds: {confirmed}")
        if true_bonds:
            found = set(confirmed) & true_bonds
            print(f"True bonds found: {sorted(found)} / {sorted(true_bonds)}")
            tag = '✓ BOTH' if true_bonds.issubset(set(confirmed)) else \
                  ('~ PARTIAL' if found else '✗ NONE')
            print(f"Result: {tag}")

if __name__ == '__main__':
    main()
