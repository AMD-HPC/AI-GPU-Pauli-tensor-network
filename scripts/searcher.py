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

    # ── Signed cross-entropy at first baseline arrival ────────────────────────
    # L(v, t) = -ln(n_pert(v,t) / n_base(v,t))
    #   L > 0 : perturbed weaker (red)
    #   L < 0 : perturbed stronger (blue)
    #
    # Evaluate at t_first(v) = first cycle where n_base(v,t) > thr.
    # At that moment the NL wave has just arrived but the perturbed wave
    # may be blocked — peak signal, minimum chi-saturation noise.
    #
    # Bond score: S(b) = L(v_a, t_first(v_a)) * L(v_b, t_first(v_b))
    #   S < 0 : sign flip across bond → broken bond candidate
    #   S > 0 : same sign, not the boundary

    FLOOR = 1e-7
    t_first = arrival_time(data_base[:nc], args.thr)  # first baseline arrival

    # Signed cross-entropy at first arrival
    L_first = np.zeros(ns)
    for v in range(ns):
        t = t_first[v] - 1  # 0-indexed
        if t < 0 or t >= nc:
            continue
        nb = max(data_base[t, v], FLOOR)
        np_ = max(data_pert[t, v], FLOOR)
        L_first[v] = -np.log(np_ / nb)

    # Also compute summed L over early cycles (backup)
    L_sum = np.zeros(ns)
    for v in range(ns):
        for t in range(min(3, nc)):
            nb = data_base[t, v]
            if nb > args.thr:
                nb_c = max(nb, FLOOR)
                np_c = max(data_pert[t, v], FLOOR)
                L_sum[v] += -np.log(np_c / nb_c)

    print(f"\nSigned cross-entropy L(v, t_first(v)):")
    nonzero = np.abs(L_first) > 1e-6
    print(f"  Sites with signal: {nonzero.sum()}")
    if nonzero.sum() > 0:
        print(f"  max L (red):  {L_first.max():.4f} at site {L_first.argmax()+1}")
        print(f"  min L (blue): {L_first.min():.4f} at site {L_first.argmin()+1}")

    if nonzero.sum() == 0:
        # Fallback: use summed L
        L_first = L_sum
        nonzero = np.abs(L_first) > 1e-6
        print(f"  No t_first signal — using summed early cycles")
        if nonzero.sum() == 0:
            print("\n  No detectable perturbation.")
            return

    # Trigger sites: largest |L_first|
    abs_L = np.abs(L_first)
    max_L = abs_L.max()
    trigger_mask = abs_L >= max_L * 0.8
    trigger_sites = np.where(trigger_mask)[0].tolist()
    direction = "red (weaker)" if L_first[trigger_sites[0]] > 0 else "blue (stronger)"
    print(f"\nTrigger sites ({direction}): {[v+1 for v in trigger_sites]}")
    print(f"  L values: {[float(f'{L_first[v]:.4f}') for v in trigger_sites]}")

    # Compatibility: set abs_delta for downstream code
    abs_delta = abs_L
    delta_l   = L_first

    # Bond scoring: S(b) = L(v_a, t_first) * L(v_b, t_first)
    # Negative product = sign flip = broken bond
    # Rank by most negative score.
    site_idx = {s: i for i, s in enumerate(all_sites)}
    candidate_bonds = {}

    for bond_idx, e in enumerate(g['edges']):
        a, b = get_bond_sites(e)
        a_idx = site_idx.get(a, -1)
        b_idx_s = site_idx.get(b, -1)
        if a_idx < 0 or b_idx_s < 0: continue
        La = L_first[a_idx]
        Lb = L_first[b_idx_s]
        # Only score bonds where at least one endpoint has a signal
        if abs(La) < 1e-6 and abs(Lb) < 1e-6: continue
        # Sign product: negative = sign flip across bond
        score = La * Lb
        candidate_bonds[bond_idx] = score

    if not candidate_bonds:
        print("\n  No candidate bonds found.")
        return

    # Rank by most negative score (sign flip = broken bond)
    ranked = sorted(candidate_bonds.items(), key=lambda x: x[1])  # ascending: most negative first
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
        'max_delta_l': float(abs_delta.max()),
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
