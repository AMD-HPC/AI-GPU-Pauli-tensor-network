#!/usr/bin/env python3
"""
find_voronoi_centers.py — Find center placement with unique Voronoi ownership.

Key property: no two centers share the same row OR column on the grid.
This guarantees every bond has a strictly unique nearest center (no ties),
so seismic + aperture work in a single shot:
  - Seismic: all centers fire simultaneously → each bond's RED/BLUE is dominated
    by its unique nearest center → clean directional signal everywhere
  - Aperture: per-center L fields are non-overlapping → sum = seismic result
    but with per-center directionality preserved

On a rectangular N×N grid with k centers:
  - Choose k distinct rows and k distinct columns
  - Assign one center per (row, col) pair (permutation)
  - Optimize: maximize minimum bond-to-nearest-center distance

Usage:
  python3 scripts/find_voronoi_centers.py --graph data/graphs/rect_7x7.json --k 4
  python3 scripts/find_voronoi_centers.py --graph data/graphs/rect_7x7.json --k 4 --show
"""

import argparse
import json
import numpy as np
from collections import deque
from itertools import permutations, combinations
import sys
sys.path.insert(0, 'scripts')
from find_centers import load_graph, get_bond_sites, all_sites
from find_centers import bond_distances as bond_dist_fn  # returns np.array per bond

# ── Grid utilities ─────────────────────────────────────────────────────────────

def site_to_rowcol(s, ncols):
    """Convert 1-indexed site to (row, col), 0-indexed."""
    return (s-1) // ncols, (s-1) % ncols

def rowcol_to_site(row, col, ncols):
    """Convert (row, col) to 1-indexed site."""
    return row * ncols + col + 1

def detect_grid_shape(g):
    """Infer (nrows, ncols) from graph edges."""
    sites = all_sites(g)
    max_s = max(sites)
    # For rect grid: ncols = max consecutive horizontal bond + 1
    # Try to infer from site numbering
    edges = g['edges']
    # horizontal bonds: consecutive sites with same row
    # Find ncols by looking at smallest bond step > 0 for horizontal bonds
    adj = {}
    for e in edges:
        a, b = get_bond_sites(e)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    # ncols = max number of sites in a consecutive horizontal run
    # Heuristic: ncols = sqrt(max_s) rounded
    import math
    ncols = int(round(math.sqrt(max_s)))
    nrows = (max_s + ncols - 1) // ncols
    return nrows, ncols

def no_shared_rowcol(sites_list, ncols):
    """Check that no two sites share row or column."""
    rcs = [site_to_rowcol(s, ncols) for s in sites_list]
    rows = [r for r,c in rcs]
    cols = [c for r,c in rcs]
    return len(set(rows)) == len(rows) and len(set(cols)) == len(cols)

def voronoi_quality(centers, g, v_hop=0.448):
    """
    Compute quality of center placement:
    - For each bond: find nearest center (crossing distance = max of two endpoints)
    - Check uniqueness: no bond equidistant from two centers
    - Return (max_T_b, n_ties, coverage_complete)
    """
    n_bonds = len(g['edges'])
    # BFS from each center
    center_dist = {}
    for c in centers:
        center_dist[c] = bfs_distances(g, c)  # crossing distance (max of endpoints)

    T_b = np.zeros(n_bonds)
    nearest = np.full(n_bonds, -1)
    second_dist = np.full(n_bonds, 9999)
    n_ties = 0

    for bi, e in enumerate(g['edges']):
        a, b = get_bond_sites(e)
        dists = sorted([(center_dist[c][bi], c) for c in centers])
        d_min, nc = dists[0]
        d_second = dists[1][0] if len(dists) > 1 else 9999
        T_b[bi] = int(np.ceil(d_min / v_hop)) if d_min < 9999 else -1
        nearest[bi] = nc
        second_dist[bi] = d_second
        if d_min == d_second:
            n_ties += 1

    coverage = (T_b > 0).all()
    return T_b, nearest, n_ties, coverage

# ── Main placement algorithm ───────────────────────────────────────────────────

def find_voronoi_centers(g, k, ncols, nrows, v_hop=0.448, verbose=False):
    """
    Find k centers with no shared row/col that minimize max T_b (ties = 0).

    Strategy:
    1. Choose k rows from nrows (C(nrows,k) options)
    2. For each row subset, try all permutations of k cols from ncols
    3. Score by: (n_ties=0, min max_T_b, max min_second_dist)
    4. Return best placement
    """
    sites = all_sites(g)
    n_bonds = len(g['edges'])

    # Precompute bond distances from all sites
    if verbose:
        print(f"  Precomputing bond distances for {len(sites)} sites × {n_bonds} bonds...")
    site_bond_dist = {}
    for s in sites:
        site_bond_dist[s] = bond_dist_fn(g, s)  # np.array[n_bonds], crossing distance

    best = None  # (n_ties, max_T_b, centers)
    n_evaluated = 0

    row_choices = list(combinations(range(nrows), k))
    col_choices = list(combinations(range(ncols), k))

    if verbose:
        total = len(row_choices) * sum(1 for _ in permutations(range(k)))
        print(f"  Evaluating {len(row_choices)} row combos × {len(col_choices)} col combos × {k}! permutations...")

    for rows in row_choices:
        for cols in col_choices:
            for col_perm in permutations(cols):
                # Build center list: one per (row, col_perm[i])
                centers = [rowcol_to_site(rows[i], col_perm[i], ncols)
                           for i in range(k)]
                # Skip if any site not in graph
                if not all(c in site_bond_dist for c in centers):
                    continue

                # Fast quality check using precomputed distances
                T_b = np.zeros(n_bonds)
                n_ties = 0
                for bi, e in enumerate(g['edges']):
                    a, b = get_bond_sites(e)
                    dists = sorted(site_bond_dist[c][bi] for c in centers)
                    d_min = dists[0]; d_second = dists[1] if len(dists) > 1 else 9999
                    T_b[bi] = int(np.ceil(d_min / v_hop)) if d_min < 9999 else 99
                    if d_min == d_second:
                        n_ties += 1

                max_T_b = int(T_b.max())
                n_evaluated += 1

                score = (n_ties, max_T_b)
                if best is None or score < (best[0], best[1]):
                    best = (n_ties, max_T_b, list(centers))
                    if verbose and n_ties == 0:
                        print(f"  New best: ties={n_ties} max_T_b={max_T_b} centers={centers}")

    return best[2], best[0], best[1]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph',  required=True)
    ap.add_argument('--k',      type=int, default=4, help='Number of centers')
    ap.add_argument('--v_hop',  type=float, default=0.448)
    ap.add_argument('--show',   action='store_true')
    ap.add_argument('--out',    default=None)
    args = ap.parse_args()

    g = load_graph(args.graph)
    nrows, ncols = detect_grid_shape(g)
    n_bonds = len(g['edges'])
    sites = all_sites(g)

    print(f"Graph: {args.graph}")
    print(f"  Sites: {len(sites)}, Bonds: {n_bonds}, Grid: {nrows}×{ncols}")
    print(f"  k={args.k} centers, v_hop={args.v_hop}")
    print(f"\nSearching for Voronoi-safe placement (no shared row/col)...")

    centers, n_ties, max_T_b = find_voronoi_centers(
        g, args.k, ncols, nrows, args.v_hop, verbose=True)

    print(f"\nResult:")
    print(f"  Centers: {centers}")
    print(f"  Ties: {n_ties}  Max T_b: {max_T_b} cycles")
    print(f"  Run with: --center={','.join(map(str,centers))}")

    # Per-center Voronoi stats
    site_bond_dist = {c: bond_dist_fn(g, c) for c in centers}
    T_b_arr = np.zeros(n_bonds)
    nearest_arr = np.zeros(n_bonds, dtype=int)
    for bi, e in enumerate(g['edges']):
        a, b = get_bond_sites(e)
        dists = [(site_bond_dist[c][bi], c) for c in centers]
        d_min, nc = min(dists)
        T_b_arr[bi] = int(np.ceil(d_min / args.v_hop)) if d_min < 9999 else -1
        nearest_arr[bi] = nc

    print(f"\nVoronoi region sizes:")
    for c in centers:
        n = (nearest_arr == c).sum()
        rc = site_to_rowcol(c, ncols)
        print(f"  Center {c} (row={rc[0]},col={rc[1]}): {n} bonds")

    print(f"\nT_b statistics:")
    print(f"  Max T_b: {int(T_b_arr.max())} cycles")
    print(f"  Mean T_b: {T_b_arr.mean():.1f} cycles")
    print(f"  Bonds reachable in 3 cycles: {(T_b_arr<=3).sum()}/{n_bonds}")
    print(f"  Bonds reachable in 5 cycles: {(T_b_arr<=5).sum()}/{n_bonds}")
    print(f"  Bonds reachable in 7 cycles: {(T_b_arr<=7).sum()}/{n_bonds}")

    if args.show:
        # Print grid showing Voronoi regions
        print(f"\nVoronoi grid (nearest center per site):")
        print("  ", end='')
        for col in range(ncols): print(f"  c{col}", end='')
        print()
        for row in range(nrows):
            print(f"  r{row}:", end='')
            for col in range(ncols):
                s = rowcol_to_site(row, col, ncols)
                if s in centers:
                    print(f"  [{centers.index(s)}]", end='')
                else:
                    # Find nearest center to this site
                    dists = {c: bfs_distances(g, s) for c in centers}
                    # use site-to-site distance
                    from find_centers import bfs_distances as bd
                    site_dists = {}
                    for c in centers:
                        d = bd(g, c)  # this gives bond distances, need site distances
                    # Simple: use BFS site distance
                    from collections import deque
                    def site_dist(src, tgt, g):
                        adj = {}
                        for e in g['edges']:
                            a, b = get_bond_sites(e)
                            adj.setdefault(a,[]).append(b)
                            adj.setdefault(b,[]).append(a)
                        dist = {v:-1 for v in adj}
                        dist[src]=0; q=deque([src])
                        while q:
                            v=q.popleft()
                            for u in adj[v]:
                                if dist[u]<0: dist[u]=dist[v]+1; q.append(u)
                        return dist.get(tgt,9999)
                    nc = min(centers, key=lambda c: site_dist(c, s, g))
                    print(f"   {centers.index(nc)} ", end='')
            print()

    if args.out:
        result = {
            'graph': args.graph,
            'k': args.k,
            'centers': centers,
            'n_ties': n_ties,
            'max_T_b': max_T_b,
            'v_hop': args.v_hop,
        }
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {args.out}")

if __name__ == '__main__':
    main()
