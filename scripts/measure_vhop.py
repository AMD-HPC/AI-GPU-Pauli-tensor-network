#!/usr/bin/env python3
"""
measure_vhop.py — Measure accurate frontier velocity per state from PAULI output.

Unlike the linear-fit v_hop (slope of frontier vs cycle), this script measures
the ACTUAL maximum BFS shell where n(v,t) > threshold at the LAST cycle.
This is the correct value for lightcone allocation: it covers exactly the sites
that have physical weight above threshold, no more, no less.

Two outputs per state:
  max_shell   : actual BFS shell reached at last cycle (integer, exact)
  v_hop_eff   : max_shell / n_cycles (effective velocity, for --v_hop flag)
  v_hop_slope : linear fit slope (for reference, usually underestimates)

For inner products: use NL v_hop_eff (noiseless is fastest, most conservative).
For wavemap-only:  use per-state v_hop_eff via --v_hops=a,b,c,d.

Usage:
  # After a pre-flight run:
  ./main_pauli_sweep 10 10 1e-3 --center=13 > /tmp/preflight.log

  python3 scripts/measure_vhop.py \
      --log /tmp/preflight.log \
      --graph data/graphs/heavy_hex_3x3.json \
      --center 13 --threshold 1e-4

  python3 scripts/measure_vhop.py \
      --log /tmp/preflight_r7.log \
      --graph data/graphs/rect_7x7.json \
      --center 24 --threshold 1e-4
"""

import argparse, re, json
import numpy as np
from collections import deque

def parse_log(path):
    data = {g: [] for g in ['NL','N1','N2','N3']}
    with open(path) as f:
        for line in f:
            for g in data:
                m = re.match(rf'PAULI_{g} (\d+)(.*)', line)
                if m:
                    data[g].append([float(x) for x in m.group(2).split()])
    return {g: np.array(v) for g,v in data.items() if v}

def bfs(graph_path, center):
    with open(graph_path) as f: d = json.load(f)
    ns = d.get('n_sites', len(d.get('vertices',[])))
    adj = [[] for _ in range(ns)]
    for e in d['edges']:
        u = e['edge'][0][0]-1; v = e['edge'][1][0]-1
        adj[u].append(v); adj[v].append(u)
    dist = [-1]*ns; dist[center]=0; q=deque([center])
    while q:
        v=q.popleft()
        for u in adj[v]:
            if dist[u]<0: dist[u]=dist[v]+1; q.append(u)
    return ns, np.array(dist)

def frontier_per_cycle(N_vt, dist, thr):
    """Max BFS shell of newly-reached sites per cycle."""
    nc, ns = N_vt.shape
    F = []
    for t in range(nc):
        cur  = N_vt[t]
        prev = N_vt[t-1] if t > 0 else np.zeros(ns)
        new  = [dist[v] for v in range(ns) if cur[v]>thr and prev[v]<=thr]
        act  = [dist[v] for v in range(ns) if cur[v]>thr]
        F.append(max(new) if new else (max(act) if act else 0))
    return np.array(F)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log',       required=True)
    ap.add_argument('--graph',     required=True)
    ap.add_argument('--center',    type=int,   default=13)
    ap.add_argument('--threshold', type=float, default=1e-4)
    args = ap.parse_args()

    data = parse_log(args.log)
    ns, dist = bfs(args.graph, args.center)
    thr = args.threshold
    nc  = next(iter(data.values())).shape[0]

    print(f"\nFrontier velocity measurement")
    print(f"Log: {args.log}")
    print(f"Graph: {args.graph}, center={args.center}, threshold={thr:.0e}, cycles={nc}")
    print()

    results = {}
    shared_shell = 0   # max across all states — for inner products

    for g in ['NL','N1','N2','N3']:
        if g not in data:
            continue
        N_vt = data[g]

        # Actual max shell at LAST cycle (self-consistent with threshold)
        last = N_vt[-1]
        active_shells = [dist[v] for v in range(ns) if last[v] > thr and dist[v] >= 0]
        max_shell = max(active_shells) if active_shells else 0
        v_eff     = max_shell / nc

        # Linear fit slope (for reference)
        F = frontier_per_cycle(N_vt, dist, thr)
        t = np.arange(1, nc+1)
        v_slope, _ = np.linalg.lstsq(
            np.vstack([t, np.ones(nc)]).T, F, rcond=None)[0]

        results[g] = {'max_shell': max_shell, 'v_eff': v_eff, 'v_slope': v_slope}
        shared_shell = max(shared_shell, max_shell)

        print(f"  {g}: max_shell={max_shell}  "
              f"v_hop_eff={v_eff:.4f}  v_hop_slope={v_slope:.4f}")

    shared_v = shared_shell / nc
    print(f"\nFor --v_hop (shared, inner products): {shared_v:.4f}  "
          f"(shell={shared_shell}, covers all states)")

    # Per-state string for --v_hops
    order = ['NL','N1','N2','N3']
    v_hops_str = ','.join(f"{results[g]['v_eff']:.4f}"
                          for g in order if g in results)
    print(f"For --v_hops (per-state, wavemap only): {v_hops_str}")
    print()
    print("Memory estimates (lightcone vs full):")

    # Try to load grid for memory estimate
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        # Just print the shells and let the user pass to the allocator
        for g in order:
            if g not in results: continue
            r = results[g]
            print(f"  {g}: allocate shells 0..{r['max_shell']}  "
                  f"(v_hop={r['v_eff']:.4f})")
    except Exception:
        pass

if __name__ == '__main__':
    main()
