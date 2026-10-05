#!/usr/bin/env python3
"""
searcher.py — C_searcher: find the broken bond from wavemap differences.

The Hamiltonian Tomography Game:
  Given baseline (uniform J=1) and perturbed (one bond changed) simulations,
  compute Δl(v) = l_perturbed(v) - l_baseline(v) and identify the trigger site
  — the first site where the wavefront arrives detectably earlier or later.

The trigger site is immediately downstream of the broken bond. The broken bond
is on the shortest path from any center to the trigger site.

Usage:
  python3 scripts/searcher.py \\
      --baseline logs/seismic_baseline_NL_r7.log \\
      --perturbed logs/seismic_broken_bond12_NL_r7.log \\
      --graph data/graphs/rect_7x7.json \\
      --centers 25,10,34,30,6,32 \\
      --out logs/search_result.json

  # With known answer (for scoring):
  python3 scripts/searcher.py ... --broken logs/broken_bond12.json
"""

import argparse
import json
import numpy as np
from collections import deque
from pathlib import Path

# ── Log parsing ───────────────────────────────────────────────────────────────

def parse_pauli_log(path, state='NL'):
    """Parse PAULI_<state> lines from a log file. Returns (nc, ns, data[t, v])."""
    key = f'PAULI_{state}'
    steps = {}
    with open(path) as f:
        for line in f:
            if not line.startswith(key): continue
            parts = line.split()
            t = int(parts[1])
            vals = np.array([float(x) for x in parts[2:]])
            steps[t] = vals
    if not steps:
        raise ValueError(f"No {key} lines found in {path}")
    nc = max(steps.keys())
    ns = len(next(iter(steps.values())))
    data = np.zeros((nc, ns))
    for t, v in steps.items():
        data[t-1] = v
    return nc, ns, data

# ── Graph utilities ───────────────────────────────────────────────────────────

def load_graph(path):
    with open(path) as f:
        return json.load(f)

def get_bond_sites(e):
    if isinstance(e, dict):
        sites = e['edge']
        a = sites[0][0] if isinstance(sites[0], list) else sites[0]
        b = sites[1][0] if isinstance(sites[1], list) else sites[1]
    else:
        a, b = e[0], e[1]
    return a, b

def build_adj(g):
    all_sites = set()
    for e in g['edges']:
        a, b = get_bond_sites(e)
        all_sites.add(a); all_sites.add(b)
    adj = {s: [] for s in all_sites}
    bond_map = {}  # (a,b) -> bond_idx
    for i, e in enumerate(g['edges']):
        a, b = get_bond_sites(e)
        adj[a].append((b, i))
        adj[b].append((a, i))
        bond_map[(min(a,b), max(a,b))] = i
    return adj, bond_map, sorted(all_sites)

def bfs_from(adj, source):
    dist = {s: -1 for s in adj}
    parent = {s: -1 for s in adj}
    parent_bond = {s: -1 for s in adj}
    dist[source] = 0
    q = deque([source])
    while q:
        v = q.popleft()
        for u, bond_idx in adj[v]:
            if dist[u] < 0:
                dist[u] = dist[v] + 1
                parent[u] = v
                parent_bond[u] = bond_idx
                q.append(u)
    return dist, parent, parent_bond

# ── Arrival time ──────────────────────────────────────────────────────────────

def arrival_time(data, thr=1e-4):
    """First cycle t where n(v,t) > thr. Returns -1 if never."""
    nc, ns = data.shape
    t_arr = np.full(ns, -1, dtype=int)
    for v in range(ns):
        for t in range(nc):
            if data[t, v] > thr:
                t_arr[v] = t + 1  # 1-indexed cycles
                break
    return t_arr

# ── Main search ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline',  required=True, help='Baseline log (uniform J=1)')
    ap.add_argument('--perturbed', required=True, help='Perturbed log (broken bond)')
    ap.add_argument('--graph',     required=True, help='Graph JSON')
    ap.add_argument('--centers',   required=True, help='Comma-separated center sites')
    ap.add_argument('--broken',    default=None,  help='breaker.py JSON (for scoring)')
    ap.add_argument('--thr',       type=float, default=1e-4, help='Arrival threshold')
    ap.add_argument('--out',       default=None,  help='Output JSON')
    ap.add_argument('--verbose',   action='store_true')
    args = ap.parse_args()

    centers = [int(x) for x in args.centers.split(',')]
    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)

    print(f"\nSearcher: Hamiltonian Tomography Game")
    print(f"  Baseline:  {args.baseline}")
    print(f"  Perturbed: {args.perturbed}")
    print(f"  Centers:   {centers}")
    print(f"  Threshold: {args.thr}")

    # Parse logs
    nc_b, ns, data_base = parse_pauli_log(args.baseline,  'NL')
    nc_p, ns, data_pert = parse_pauli_log(args.perturbed, 'NL')
    nc = min(nc_b, nc_p)
    print(f"  Sites: {ns}  Cycles: {nc}")

    # ── Method 1: integer arrival time difference ─────────────────────────────
    t_base = arrival_time(data_base[:nc], args.thr)
    t_pert = arrival_time(data_pert[:nc], args.thr)
    delta_l = np.zeros(ns)
    for v in range(ns):
        if t_base[v] > 0 and t_pert[v] > 0:
            delta_l[v] = t_pert[v] - t_base[v]
        elif t_base[v] > 0 and t_pert[v] < 0:
            delta_l[v] = nc
        elif t_base[v] < 0 and t_pert[v] > 0:
            delta_l[v] = -nc

    # ── Method 2: amplitude difference (sub-cycle sensitivity) ────────────────
    # Δn(v) = max over cycles of |n_perturbed(v,t) - n_baseline(v,t)|
    # Normalized by baseline amplitude to get relative change
    delta_n = np.zeros(ns)
    for v in range(ns):
        diffs = np.abs(data_pert[:nc, v] - data_base[:nc, v])
        base_max = data_base[:nc, v].max()
        if base_max > args.thr:
            delta_n[v] = diffs.max() / base_max  # relative amplitude change

    abs_delta_l = np.abs(delta_l)
    max_delta_l = abs_delta_l.max()

    print(f"\nΔl(v) [integer cycle arrival difference]:")
    print(f"  max |Δl| = {max_delta_l:.1f} cycles")
    print(f"  Sites with |Δl| > 0: {(abs_delta_l > 0).sum()}")

    print(f"\nΔn(v) [relative amplitude difference, sub-cycle]:")
    print(f"  max |Δn| = {delta_n.max():.4f}")
    print(f"  Sites with |Δn| > 0.001: {(delta_n > 0.001).sum()}")
    print(f"  Sites with |Δn| > 0.01:  {(delta_n > 0.01).sum()}")

    # Use amplitude difference if arrival times are identical
    if max_delta_l == 0:
        print("\n  Arrival times identical — using amplitude difference (sub-cycle signal)")
        abs_delta = delta_n
        delta_l   = delta_n  # use for direction/ranking
    else:
        abs_delta = abs_delta_l

    max_delta = abs_delta.max()
    if max_delta == 0:
        print("\n  No detectable perturbation.")
        return

    # Trigger sites: those with maximum signal
    trigger_mask = abs_delta >= max_delta * 0.8
    trigger_sites = np.where(trigger_mask)[0].tolist()
    direction = "accelerated" if delta_l[trigger_sites[0]] < 0 else "perturbed"
    print(f"\nTrigger sites ({direction}): {trigger_sites}")
    print(f"  |Δ| values: {[float(f'{abs_delta[v]:.4f}') for v in trigger_sites]}")

    # For each trigger site, find which bond on the path from nearest center
    # is the likely perturbed bond
    candidate_bonds = {}
    for center in centers:
        dist, parent, parent_bond = bfs_from(adj, center)
        for v in trigger_sites:
            if dist.get(v, -1) < 0: continue
            # Walk path from v back to center, collect bonds
            curr = v
            while parent[curr] >= 0:
                b_idx = parent_bond[curr]
                if b_idx >= 0:
                    candidate_bonds[b_idx] = candidate_bonds.get(b_idx, 0) + abs_delta[v]
                curr = parent[curr]

    if not candidate_bonds:
        print("\n  No candidate bonds found.")
        return

    # Rank bonds by accumulated signal
    ranked = sorted(candidate_bonds.items(), key=lambda x: -x[1])
    print(f"\nCandidate bonds (ranked by signal):")
    for bond_idx, score in ranked[:5]:
        a, b = get_bond_sites(g['edges'][bond_idx])
        print(f"  Bond {bond_idx:3d}: {a:3d} ↔ {b:3d}  score={score:.3f}")

    predicted_bond = ranked[0][0]
    a_pred, b_pred = get_bond_sites(g['edges'][predicted_bond])
    print(f"\nC_searcher declares: Bond {predicted_bond} ({a_pred} ↔ {b_pred})")

    # Score if ground truth known
    result = {
        'predicted_bond': predicted_bond,
        'predicted_sites': [a_pred, b_pred],
        'trigger_sites': trigger_sites,
        'max_delta_l': float(max_delta),
        'direction': direction,
        'candidate_bonds': ranked[:10],
        'delta_l': delta_l.tolist()
    }

    if args.broken:
        with open(args.broken) as f:
            broken = json.load(f)
        true_bond = broken['bond']
        true_a, true_b = broken['site_a'], broken['site_b']
        correct = (predicted_bond == true_bond)
        print(f"\nGround truth: Bond {true_bond} ({true_a} ↔ {true_b})")
        print(f"Result: {'✓ CORRECT' if correct else '✗ WRONG'}")
        if not correct:
            # Check if true bond is in top candidates
            rank = next((i for i, (b,_) in enumerate(ranked) if b==true_bond), -1)
            print(f"  True bond ranked #{rank+1} in candidates" if rank>=0 else
                  "  True bond not in candidates")
        result['true_bond'] = true_bond
        result['correct'] = correct

    if args.verbose:
        print(f"\nFull Δl(v) per site:")
        for v in range(ns):
            if abs_delta[v] > 0:
                print(f"  Site {v:3d}: Δl={delta_l[v]:+.1f}  "
                      f"t_base={t_base[v]}  t_pert={t_pert[v]}")

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {args.out}")

if __name__ == '__main__':
    main()
