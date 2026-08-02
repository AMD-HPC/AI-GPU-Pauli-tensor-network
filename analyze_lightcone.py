#!/usr/bin/env python3
"""
analyze_lightcone.py — Extract frontier velocity from PAULI output.

Reads PAULI lines from main_lightcone output.
Computes graph distance from center for each site.
Finds frontier position per cycle (furthest site with n(v,t) > threshold).
Fits v_hop = frontier_distance / cycle.

Usage:
  python3 analyze_lightcone.py --dat /tmp/lightcone_heavyhex.dat
                               --graph data/graphs/heavy_hex_3x3.json
                               [--center 2] [--threshold 1e-4]
"""

import argparse
import json
import re
import numpy as np
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_graph(path):
    with open(path) as f:
        d = json.load(f)
    # Support both JSON formats
    if 'n_sites' in d:
        n_sites = d['n_sites']
    elif 'vertices' in d:
        n_sites = len(d['vertices'])
    else:
        # Count from edges
        all_v = set()
        for e in d['edges']:
            all_v.add(e['edge'][0][0] - 1)
            all_v.add(e['edge'][1][0] - 1)
        n_sites = max(all_v) + 1
    adj = [[] for _ in range(n_sites)]
    for e in d['edges']:
        u = e['edge'][0][0] - 1
        v = e['edge'][1][0] - 1
        adj[u].append(v)
        adj[v].append(u)
    return n_sites, adj

def bfs_distances(adj, center, n_sites):
    dist = [-1] * n_sites
    dist[center] = 0
    q = deque([center])
    while q:
        v = q.popleft()
        for u in adj[v]:
            if dist[u] < 0:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist

def parse_dat(path):
    steps = {}
    with open(path) as f:
        for line in f:
            if line.startswith('PAULI '):
                parts = line.split()
                step = int(parts[1])
                weights = [float(x) for x in parts[2:]]
                steps[step] = weights
            elif line.startswith('STEP '):
                m = re.search(r'STEP (\d+)\s+C_nl ([+-][\d.e+-]+)\s+chi (\d+)', line)
                if m:
                    s = int(m.group(1))
                    if s not in steps:
                        steps[s] = None
    return steps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dat',       required=True)
    ap.add_argument('--graph',     default='data/graphs/heavy_hex_3x3.json')
    ap.add_argument('--center',    type=int, default=2)
    ap.add_argument('--threshold', type=float, default=1e-3)
    ap.add_argument('--out', default=None, help='Output PNG path (default: dat with _analysis.png)')
    args = ap.parse_args()

    n_sites, adj = load_graph(args.graph)
    dist = bfs_distances(adj, args.center, n_sites)
    max_dist = max(dist)
    print(f"Graph: {n_sites} sites, max distance from center={args.center}: {max_dist}")

    data = parse_dat(args.dat)
    step_list = sorted(s for s in data if data[s] is not None)

    print(f"\n{'Cycle':>6}  {'Frontier':>8}  {'Sites>thresh':>12}  {'Max n(v)':>10}")
    print("-" * 45)

    frontiers = []
    for step in step_list:
        w = data[step]
        if len(w) != n_sites:
            print(f"  step {step}: wrong number of sites ({len(w)} vs {n_sites})")
            continue
        # Frontier = max distance where n(v,t) > threshold
        active = [dist[v] for v in range(n_sites) if w[v] > args.threshold]
        frontier = max(active) if active else 0
        n_active = len(active)
        max_w = max(w)
        frontiers.append((step, frontier))
        print(f"{step:>6}  {frontier:>8}  {n_active:>12}  {max_w:>10.4f}")

    if len(frontiers) < 2:
        print("Not enough data to fit velocity.")
        return

    # Fit v_hop = frontier / cycle (linear through origin)
    cycles   = np.array([f[0] for f in frontiers])
    frontier = np.array([f[1] for f in frontiers])
    # Linear fit with intercept
    A = np.vstack([cycles, np.ones(len(cycles))]).T
    v_hop, b = np.linalg.lstsq(A, frontier, rcond=None)[0]
    print(f"\nFrontier velocity: v_hop = {v_hop:.3f} hops/cycle")
    print(f"  (frontier ≈ {v_hop:.3f} × cycle + {b:.2f})")
    print(f"  Physical velocity: v = {v_hop}/ε = {v_hop/0.10:.2f} hops/unit_time  (ε=0.10)")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: n(v,t) heatmap (distance vs cycle)
    dist_arr = np.array(dist)
    n_data = []
    for step in step_list:
        w = data[step]
        row = np.zeros(max_dist+1)
        for d_val in range(max_dist+1):
            sites_at_d = [v for v in range(n_sites) if dist[v]==d_val]
            row[d_val] = max(w[v] for v in sites_at_d) if sites_at_d else 0
        n_data.append(row)
    n_data = np.array(n_data)
    im = ax1.imshow(n_data.T, aspect='auto', origin='lower',
                    extent=[step_list[0]-0.5, step_list[-1]+0.5, -0.5, max_dist+0.5],
                    cmap='hot')
    ax1.set_xlabel('Cycle'); ax1.set_ylabel('Graph distance from center')
    ax1.set_title('Max n(v,t) per distance shell')
    plt.colorbar(im, ax=ax1, label='max n(v,t)')

    # Right: frontier vs cycle + fit
    ax2.scatter(cycles, frontier, color='blue', label='Frontier (max active distance)')
    fit_line = v_hop * cycles + b
    ax2.plot(cycles, fit_line, 'r--', label=f'Fit: v={v_hop:.3f} hops/cycle')
    ax2.set_xlabel('Cycle'); ax2.set_ylabel('Frontier distance (hops)')
    ax2.set_title('Lightcone frontier velocity')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    outfile = args.out if args.out else args.dat.replace('.dat', '_analysis.png')
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    print(f"\nPlot saved: {outfile}")

if __name__ == '__main__':
    main()
