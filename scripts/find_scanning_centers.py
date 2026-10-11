#!/usr/bin/env python3
"""
find_scanning_centers.py — Find minimum set of Voronoi center placements
such that every bond is covered within 2 hops (crossing distance) by at
least one placement.

Each placement = k centers with no shared row or column (Voronoi property).
Displacements shift the permutation systematically to cover the full grid.

Usage:
  python3 scripts/find_scanning_centers.py --graph data/graphs/rect_7x7.json --hops 2
"""

import argparse
import json
import numpy as np
from itertools import permutations
import sys
sys.path.insert(0, 'scripts')
from find_centers import load_graph, get_bond_sites, bond_distances, all_sites

def site_to_rowcol(s, ncols):
    return (s-1) // ncols, (s-1) % ncols

def rowcol_to_site(row, col, ncols):
    return row * ncols + col + 1

def detect_grid_shape(g):
    import math
    sites = all_sites(g)
    max_s = max(sites)
    ncols = int(round(math.sqrt(max_s)))
    nrows = (max_s + ncols - 1) // ncols
    return nrows, ncols

def coverage_mask(centers, g, d_max, site_bond_dist):
    """Boolean mask: which bonds are within d_max crossing hops of any center."""
    n_bonds = len(g['edges'])
    covered = np.zeros(n_bonds, dtype=bool)
    for c in centers:
        covered |= (site_bond_dist[c] <= d_max)
    return covered

def find_best_placement(g, nrows, ncols, site_bond_dist, d_max, uncovered_mask):
    """Find k=ncols center placement (one per row+col) covering most uncovered bonds."""
    n_bonds = len(g['edges'])
    sites = all_sites(g)

    best_centers = None
    best_count = -1
    best_ties = 9999

    # Try all permutations of columns (rows 0..nrows-1, cols as permutation)
    # For efficiency on 7x7: 7! = 5040 permutations
    for col_perm in permutations(range(ncols)):
        centers = [rowcol_to_site(row, col_perm[row], ncols)
                   for row in range(min(nrows, ncols))]
        # Skip if any center not in graph
        if not all(c in site_bond_dist for c in centers):
            continue

        # Count newly covered bonds
        covered = coverage_mask(centers, g, d_max, site_bond_dist)
        new_covered = (covered & uncovered_mask).sum()

        # Count ties (bonds equidistant from two centers)
        n_ties = 0
        for bi in range(n_bonds):
            dists = sorted(site_bond_dist[c][bi] for c in centers)
            if len(dists) > 1 and dists[0] == dists[1] and dists[0] <= d_max:
                n_ties += 1

        score = (new_covered, -n_ties)
        if best_centers is None or score > (best_count, -best_ties):
            best_count = new_covered
            best_ties = n_ties
            best_centers = list(centers)

    return best_centers, best_count, best_ties

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph',  required=True)
    ap.add_argument('--hops',   type=int, default=2, help='Max crossing hops per scan')
    ap.add_argument('--v_hop',  type=float, default=0.448)
    ap.add_argument('--out',    default=None)
    args = ap.parse_args()

    g = load_graph(args.graph)
    nrows, ncols = detect_grid_shape(g)
    n_bonds = len(g['edges'])
    sites = all_sites(g)
    d_max = args.hops

    print(f"Graph: {args.graph}")
    print(f"  Sites: {len(sites)}, Bonds: {n_bonds}, Grid: {nrows}×{ncols}")
    print(f"  Coverage target: every bond within {d_max} crossing hops")
    print(f"  Cycles needed per scan: ceil({d_max}/{args.v_hop}) = {int(np.ceil(d_max/args.v_hop))}")

    # Precompute bond crossing distances from all sites
    print(f"\nPrecomputing bond distances...")
    site_bond_dist = {}
    for s in sites:
        site_bond_dist[s] = bond_distances(g, s)  # max(d_a,d_b) crossing distance

    # Greedy: find minimum number of placements for full coverage
    uncovered = np.ones(n_bonds, dtype=bool)
    placements = []
    scan = 0

    print(f"\nGreedy scan placement:")
    while uncovered.any():
        scan += 1
        centers, n_new, n_ties = find_best_placement(
            g, nrows, ncols, site_bond_dist, d_max, uncovered)

        if n_new == 0:
            print(f"  Scan {scan}: no new bonds covered — stopping")
            break

        covered = coverage_mask(centers, g, d_max, site_bond_dist)
        newly = covered & uncovered
        uncovered &= ~covered

        T_b = np.array([
            int(np.ceil(min(site_bond_dist[c][bi] for c in centers) / args.v_hop))
            if min(site_bond_dist[c][bi] for c in centers) <= d_max else -1
            for bi in range(n_bonds)
        ])
        cycles = int(np.ceil(d_max / args.v_hop))

        placements.append({
            'scan': scan,
            'centers': centers,
            'n_new_bonds': int(n_new),
            'n_ties': int(n_ties),
            'cycles': cycles,
            'bonds_covered': [i for i in range(n_bonds) if newly[i]]
        })

        print(f"  Scan {scan}: centers={centers}  new_bonds={n_new}  "
              f"ties={n_ties}  cycles={cycles}")
        print(f"    Run: --center={','.join(map(str,centers))}")

        if not uncovered.any():
            break

    print(f"\n{'='*60}")
    print(f"Total scans needed: {len(placements)}")
    print(f"Total bonds covered: {n_bonds - uncovered.sum()}/{n_bonds}")
    print(f"Bonds still uncovered: {uncovered.sum()}")
    print(f"Cycles per scan: {int(np.ceil(d_max/args.v_hop))}")
    print(f"Total cycles if sequential: {len(placements) * int(np.ceil(d_max/args.v_hop))}")

    if args.out:
        result = {
            'graph': args.graph,
            'd_max': d_max,
            'v_hop': args.v_hop,
            'cycles_per_scan': int(np.ceil(d_max/args.v_hop)),
            'n_scans': len(placements),
            'n_bonds': n_bonds,
            'placements': placements
        }
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {args.out}")

if __name__ == '__main__':
    main()
