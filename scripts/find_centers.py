#!/usr/bin/env python3
"""
find_centers.py — Find optimal center placement for Hamiltonian tomography.

Given a graph, finds the minimum set of centers such that every bond
is within distance d_max of at least one center (geodesic bond cover).

This is the p-center problem: minimize p such that max distance from
any bond to its nearest center <= d_max.

Usage:
  # Find centers covering all bonds within 3 hops:
  python3 scripts/find_centers.py --graph data/graphs/rect_7x7.json --d 3

  # Find centers for heavy-hex:
  python3 scripts/find_centers.py --graph data/graphs/heavy_hex_3x3.json --d 4

  # Show coverage map:
  python3 scripts/find_centers.py --graph data/graphs/rect_7x7.json --d 3 --show
"""

import argparse
import json
import numpy as np
from collections import deque

def load_graph(path):
    with open(path) as f:
        g = json.load(f)
    return g

def get_bond_sites(e):
    if isinstance(e, dict):
        sites = e['edge']
        a = sites[0][0] if isinstance(sites[0], list) else sites[0]
        b = sites[1][0] if isinstance(sites[1], list) else sites[1]
    else:
        a, b = e[0], e[1]
    return a, b

def bfs_distances(g, source):
    """BFS from source, return distance dict."""
    # collect all site ids (may be 0-indexed or 1-indexed)
    all_sites = set()
    for e in g['edges']:
        a, b = get_bond_sites(e)
        all_sites.add(a); all_sites.add(b)
    adj = {s: [] for s in all_sites}
    for e in g['edges']:
        a, b = get_bond_sites(e)
        adj[a].append(b)
        adj[b].append(a)
    dist = {s: -1 for s in all_sites}
    dist[source] = 0
    q = deque([source])
    while q:
        v = q.popleft()
        for u in adj[v]:
            if dist[u] < 0:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist
    dist = np.full(ns, -1)
    dist[source] = 0
    q = deque([source])
    while q:
        v = q.popleft()
        for u in adj[v]:
            if dist[u] < 0:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist

def bond_distances(g, source):
    """Distance from source to cross each bond (max distance to either endpoint).
    The wavefront must reach BOTH endpoints to cross the bond and detect it."""
    dist = bfs_distances(g, source)
    bond_dist = []
    for e in g['edges']:
        a, b = get_bond_sites(e)
        d_a = dist.get(a, 9999)
        d_b = dist.get(b, 9999)
        bond_dist.append(max(d_a, d_b))
    return np.array(bond_dist)

def all_sites(g):
    """Return sorted list of all site ids."""
    sites = set()
    for e in g['edges']:
        a, b = get_bond_sites(e)
        sites.add(a); sites.add(b)
    return sorted(sites)

def greedy_p_center(g, d_max):
    """
    Greedy set cover: iteratively pick the site that covers the most
    uncovered bonds within d_max hops.

    Returns: list of center sites, coverage map
    """
    n_bonds = len(g['edges'])
    sites = all_sites(g)

    # Precompute bond distances from every site
    print(f"  Precomputing bond distances for {len(sites)} sites × {n_bonds} bonds...")
    site_bond_dist = {}
    for s in all_sites(g):
        site_bond_dist[s] = bond_distances(g, s)

    covered = np.zeros(n_bonds, dtype=bool)
    centers = []

    while not covered.all():
        # Find site that covers most uncovered bonds within d_max
        best_site = -1
        best_count = -1
        for s in all_sites(g):
            covers = (site_bond_dist[s] <= d_max) & ~covered
            count = covers.sum()
            if count > best_count:
                best_count = count
                best_site = s

        if best_count == 0:
            print(f"  WARNING: {(~covered).sum()} bonds unreachable within d_max={d_max}")
            break

        centers.append(best_site)
        covered |= (site_bond_dist[best_site] <= d_max)
        print(f"  Center {len(centers)}: site {best_site} covers {best_count} new bonds "
              f"({covered.sum()}/{n_bonds} total)")

    return centers, site_bond_dist

def t_loop(g, v_hop, girth=None):
    """Estimate t_loop = girth / (2 * v_hop)."""
    if girth is None:
        # Estimate girth from graph name
        name = str(g.get('comment', ''))
        if 'hex' in name.lower() or 'heavy' in name.lower():
            girth = 6
        else:
            girth = 4  # rect grid
    return girth / (2 * v_hop)

def random_centers(g, n_centers, seed=42):
    """
    Choose n_centers randomly from all sites with a fixed seed for reproducibility.
    Returns: list of center sites, site_bond_dist dict
    """
    import random
    random.seed(seed)
    sites = sorted(all_sites(g))
    chosen = random.sample(sites, min(n_centers, len(sites)))
    print(f"  Random centers (seed={seed}): {chosen}")

    # Precompute bond distances from chosen centers
    site_bond_dist = {}
    n_bonds = len(g['edges'])
    for s in chosen:
        site_bond_dist[s] = bond_distances(g, s)

    return chosen, site_bond_dist

def compute_bond_cycles(centers, site_bond_dist, g, v_hop):
    """
    For each bond, compute T_b = ceil(d_min / v_hop):
    the minimum cycles needed for the closest center to reach either endpoint.
    """
    n_bonds = len(g['edges'])
    T_b = np.zeros(n_bonds)
    nearest_center = np.full(n_bonds, -1)

    for bond_idx in range(n_bonds):
        d_min = 9999
        nc = -1
        for c in centers:
            bd = site_bond_dist[c][bond_idx]
            if bd < d_min:
                d_min = bd
                nc = c
        T_b[bond_idx] = int(np.ceil(d_min / v_hop)) if d_min < 9999 else -1
        nearest_center[bond_idx] = nc

    return T_b, nearest_center

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph',          required=True)
    ap.add_argument('--d',              type=int, default=3,
                    help='Max bond distance for greedy cover (default: 3)')
    ap.add_argument('--random',         type=int, default=None, metavar='N',
                    help='Use N random centers instead of greedy (requires --seed)')
    ap.add_argument('--seed',           type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
    ap.add_argument('--v_hop',          type=float, default=None,
                    help='Frontier velocity (hops/cycle) for time window estimate')
    ap.add_argument('--show',           action='store_true',
                    help='Show per-bond coverage')
    ap.add_argument('--show-cycles',    action='store_true',
                    help='Show T_b (required cycles) per bond')
    ap.add_argument('--out',            default=None,
                    help='Save center list to JSON')
    args = ap.parse_args()

    g = load_graph(args.graph)
    ns = g.get('n_sites', g.get('num_sites', len(all_sites(g))))
    n_bonds = len(g['edges'])

    print(f"\nGraph: {args.graph}")
    print(f"  Sites: {ns}, Bonds: {n_bonds}, d_max: {args.d}")

    # Default v_hop by topology
    if args.v_hop is None:
        if 'hex' in args.graph.lower() or 'heavy' in args.graph.lower():
            args.v_hop = 0.376
            girth = 6
        else:
            args.v_hop = 0.448
            girth = 4
    else:
        girth = None

    t_clean = args.d / args.v_hop
    t_loop_est = t_loop(g, args.v_hop, girth)
    print(f"  v_hop: {args.v_hop}  d_max={args.d} → T_clean={t_clean:.1f} cycles")
    print(f"  t_loop estimate: {t_loop_est:.1f} cycles")
    if t_clean < t_loop_est:
        print(f"  ✓ T_clean < t_loop: waves independent, BP exact")
    else:
        print(f"  ⚠ T_clean > t_loop: interference expected, BP approximate")

    # Center selection: greedy or random
    if args.random:
        print(f"\nRandom center selection (n={args.random}, seed={args.seed}):")
        centers, site_bond_dist = random_centers(g, args.random, seed=args.seed)
    else:
        print(f"\nGreedy p-center selection (d_max={args.d}):")
        centers, site_bond_dist = greedy_p_center(g, args.d)

    print(f"\nResult: {len(centers)} centers needed")
    print(f"  Centers: {centers}")
    print(f"  Run with: --center={','.join(map(str,centers))}")

    # Coverage statistics
    all_bond_min_dist = np.array([
        min(site_bond_dist[c][b] for c in centers)
        for b in range(n_bonds)
    ])
    print(f"\nCoverage statistics:")
    print(f"  Max bond distance to nearest center: {all_bond_min_dist.max()}")
    print(f"  Mean bond distance: {all_bond_min_dist.mean():.2f}")
    print(f"  Bonds within d_max: {(all_bond_min_dist <= args.d).sum()}/{n_bonds}")

    if args.show:
        print(f"\nPer-bond coverage (bond_idx: site_a ↔ site_b, min_dist):")
        for i, e in enumerate(g['edges']):
            a, b = get_bond_sites(e)
            nearest = min(range(len(centers)),
                         key=lambda ci: site_bond_dist[centers[ci]][i])
            print(f"  Bond {i:3d}: {a:3d} ↔ {b:3d}  "
                  f"dist={all_bond_min_dist[i]}  "
                  f"nearest_center={centers[nearest]}")

    # Per-bond T_b: required cycles for nearest center to reach bond
    T_b, nearest_center = compute_bond_cycles(centers, site_bond_dist, g, args.v_hop)
    T_max = int(T_b[T_b > 0].max()) if (T_b > 0).any() else 0
    T_mean = T_b[T_b > 0].mean() if (T_b > 0).any() else 0

    print(f"\nPer-bond required cycles T_b = ceil(d_min / v_hop):")
    print(f"  Max T_b: {T_max} cycles  (run at least this many)")
    print(f"  Mean T_b: {T_mean:.1f} cycles")
    print(f"  Bonds reachable in 3 cycles: {(T_b <= 3).sum()}/{n_bonds}")
    print(f"  Bonds reachable in 5 cycles: {(T_b <= 5).sum()}/{n_bonds}")

    if args.show_cycles:
        print(f"\nPer-bond T_b:")
        for i, e in enumerate(g['edges']):
            a, b = get_bond_sites(e)
            print(f"  Bond {i:3d}: {a:3d} ↔ {b:3d}  "
                  f"T_b={T_b[i]:.0f}  nearest_center={nearest_center[i]}")

    if args.out:
        result = {
            'graph':            args.graph,
            'method':           'random' if args.random else 'greedy',
            'seed':             args.seed if args.random else None,
            'd_max':            args.d,
            'v_hop':            args.v_hop,
            't_clean_cycles':   t_clean,
            't_loop_cycles':    t_loop_est,
            'centers':          centers,
            'n_centers':        len(centers),
            'max_bond_dist':    int(all_bond_min_dist.max()),
            'max_T_b':          T_max,
            'mean_T_b':         float(T_mean),
            'coverage_complete': bool((all_bond_min_dist <= args.d).sum() == n_bonds)
        }
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {args.out}")

if __name__ == '__main__':
    main()
