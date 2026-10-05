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
  # Single run:
  python3 scripts/searcher.py \\
      --baseline logs/seismic_baseline_NL_r7.log \\
      --perturbed logs/seismic_broken_bond12_NL_r7.log \\
      --graph data/graphs/rect_7x7.json \\
      --centers 25,10,34,30,6,32 \\
      --out logs/search_result.json

  # Ensemble (sum L across multiple center runs):
  python3 scripts/searcher.py \\
      --baseline  logs/tomo_6c/base_c10.log,logs/tomo_6c/base_c25.log,... \\
      --perturbed logs/tomo_6c/pert_c10.log,logs/tomo_6c/pert_c25.log,... \\
      --graph data/graphs/rect_7x7.json \\
      --centers 10,25,... \\
      --broken logs/tomo_6c/broken.json

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

    # Support ensemble mode: comma-separated log file lists
    baseline_logs  = args.baseline.split(',')
    perturbed_logs = args.perturbed.split(',')
    ensemble = len(baseline_logs) > 1 or len(perturbed_logs) > 1
    if len(baseline_logs) != len(perturbed_logs):
        print("ERROR: number of baseline and perturbed logs must match")
        return

    print(f"\nSearcher: Hamiltonian Tomography Game")
    if ensemble:
        print(f"  Mode: ENSEMBLE ({len(baseline_logs)} runs, sum L fields)")
        for i, (b, p) in enumerate(zip(baseline_logs, perturbed_logs)):
            print(f"  Run {i+1}: {Path(b).name} | {Path(p).name}")
    else:
        print(f"  Baseline:  {args.baseline}")
        print(f"  Perturbed: {args.perturbed}")
    print(f"  Centers:   {centers}")
    print(f"  Threshold: {args.thr}")

    # Parse all log pairs and compute ensemble L field
    FLOOR = 1e-7
    nc = None
    ns = None
    L_ensemble = None  # accumulated sum of L fields across all runs

    for b_log, p_log in zip(baseline_logs, perturbed_logs):
        try:
            nc_b, ns_i, data_base = parse_pauli_log(b_log.strip(), 'NL')
            nc_p, ns_i, data_pert = parse_pauli_log(p_log.strip(), 'NL')
        except ValueError as e:
            print(f"  WARNING: skipping {b_log} — {e}")
            continue
        nc_i = min(nc_b, nc_p)
        if nc is None:
            nc = nc_i; ns = ns_i
            L_ensemble = np.zeros(ns)
        else:
            nc = min(nc, nc_i)

        # Arrival time for this run's baseline
        t_arr = arrival_time(data_base[:nc_i], args.thr)

        # L at first arrival for this run
        L_run = np.zeros(ns_i)
        for v in range(ns_i):
            t = t_arr[v] - 1
            if t < 0 or t >= nc_i: continue
            nb = max(data_base[t, v], FLOOR)
            np_ = max(data_pert[t, v], FLOOR)
            L_run[v] = -np.log(np_ / nb)

        L_ensemble += L_run  # sum in log space = multiply likelihoods

    if L_ensemble is None or ns is None:
        print("ERROR: no valid log pairs found")
        return

    # Use single-run variables for compatibility
    data_base = None  # not needed anymore
    data_pert = None
    print(f"  Sites: {ns}  Cycles: {nc}")

    # ── Use ensemble L field (already computed above) ─────────────────────────
    # L_ensemble[v] = sum over runs of -ln(n_pert/n_base) at t_first(v)
    # In ensemble mode: reinforces signal from centers on same side,
    # partially cancels noise from centers on opposite sides.

    L_first = L_ensemble

    print(f"\n{'Ensemble' if ensemble else 'Single-run'} signed cross-entropy L(v):")
    nonzero = np.abs(L_first) > 1e-6
    print(f"  Sites with signal: {nonzero.sum()}")
    if nonzero.sum() > 0:
        print(f"  max L (red):  {L_first.max():.4f} at site {L_first.argmax()+1}")
        print(f"  min L (blue): {L_first.min():.4f} at site {L_first.argmin()+1}")

    if nonzero.sum() == 0:
        print("\n  No detectable perturbation.")
        return

    # Trigger sites: largest |L|
    abs_L = np.abs(L_first)
    max_L = abs_L.max()
    trigger_mask = abs_L >= max_L * 0.8
    trigger_sites = np.where(trigger_mask)[0].tolist()
    direction = "red (weaker)" if L_first[trigger_sites[0]] > 0 else "blue (stronger)"
    print(f"\nTrigger sites ({direction}): {[v+1 for v in trigger_sites]}")
    print(f"  L values: {[float(f'{L_first[v]:.4f}') for v in trigger_sites]}")

    # Compatibility
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
