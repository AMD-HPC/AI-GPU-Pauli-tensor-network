#!/usr/bin/env python3
"""
compare_gram_whitened.py — Compare raw vs whitened Gram matrix for any config.

Computes G twice — raw cosine and whitened (per-observation std normalization) —
and prints top-k confusable pairs for each. Accepts pre-computed logs from
compute_gram.py output directories.

Usage:
  # Compare two configs side by side
  python3 scripts/compare_gram_whitened.py \
      --configs logs/gram_5c_standard:5-corner-H logs/gram_5c_swapped:5-corner-V \
      --top_k 10

  # Single config, raw + whitened
  python3 scripts/compare_gram_whitened.py \
      --configs logs/gram_standard:3-scan-H

  # Also save whitened G to JSON
  python3 scripts/compare_gram_whitened.py \
      --configs logs/gram_5c_standard:5-corner-H \
      --save_whitened
"""

import argparse
import json
import numpy as np
import sys
sys.path.insert(0, 'scripts')
from searcher import load_graph, build_adj, get_bond_sites, parse_pauli_log

FLOOR   = 1e-7
N_SITES = 49
N_STEPS = 5
N_BONDS = 84


def build_V(logdir, plan):
    """Build V matrix (n_rows x N_BONDS) from pre-computed logs."""
    import os
    # Detect number of scans from available log files
    scans = []
    for sc_idx, sc in enumerate(plan['placements']):
        sid = sc['scan']
        base_log = f'{logdir}/scan{sid}_base.log'
        if os.path.exists(base_log):
            scans.append((sc_idx, sid))

    if not scans:
        # Try single-scan (--centers override)
        if os.path.exists(f'{logdir}/scan1_base.log'):
            scans = [(0, 1)]

    n_scans = len(scans)
    n_rows  = n_scans * N_STEPS * N_SITES
    V = np.zeros((n_rows, N_BONDS))

    for sc_idx, sid in scans:
        try:
            _, _, db = parse_pauli_log(f'{logdir}/scan{sid}_base.log', 'NL')
        except Exception as e:
            print(f"  [warn] scan {sid} base: {e}")
            continue
        offset = sc_idx * N_STEPS * N_SITES
        for bi in range(N_BONDS):
            try:
                _, _, dp = parse_pauli_log(f'{logdir}/scan{sid}_pert_b{bi}.log', 'NL')
                nc = min(db.shape[0], dp.shape[0])
                for t in range(nc):
                    for v in range(N_SITES):
                        if db[t, v] > FLOOR:
                            V[offset + t*N_SITES + v, bi] = \
                                -np.log(max(dp[t, v], FLOOR) / db[t, v])
            except:
                pass

    return V


def compute_G(V, whiten=False):
    """Compute 84x84 Gram matrix, optionally whitened."""
    W = V.copy()
    if whiten:
        sd = W.std(axis=1, keepdims=True)
        W  = W / np.where(sd > 1e-12, sd, np.inf)
    norms = np.linalg.norm(W, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    Wn = W / norms
    G  = np.abs(Wn.T @ Wn)
    np.fill_diagonal(G, 0)
    return G


def top_pairs(G, g, k=10):
    """Return top-k confusable pairs with bond and site info."""
    order = np.dstack(np.unravel_index(np.argsort(G.ravel())[::-1], G.shape))[0]
    pairs = []; seen = set()
    for a, b in order:
        if a >= b or (a, b) in seen: continue
        seen.add((a, b))
        a_sites = get_bond_sites(g['edges'][a])
        b_sites = get_bond_sites(g['edges'][b])
        shared  = set(a_sites) & set(b_sites)
        pairs.append({
            'bond_i': int(a), 'bond_j': int(b),
            'coherence': float(G[a, b]),
            'sites_i': list(a_sites), 'sites_j': list(b_sites),
            'shared_vertex': int(shared.pop()) if shared else None,
            'shares_vertex': len(shared) > 0 or bool(set(a_sites) & set(b_sites)),
        })
        if len(pairs) == k: break
    return pairs


def print_pairs(label, pairs, median_coh, max_coh):
    iu_vals = []  # just for display
    print(f"\n  {label}  (median={median_coh:.4f}  max={max_coh:.4f})")
    print(f"  {'Rank':<5} {'Bond i':>6} {'Bond j':>6} {'coh':>8} "
          f"{'sites_i':>10} {'sites_j':>10} {'shared_v':>9}")
    print(f"  {'-'*58}")
    for rank, p in enumerate(pairs, 1):
        sv = str(p['shared_vertex']) if p['shared_vertex'] else '-'
        print(f"  {rank:<5} {p['bond_i']:>6} {p['bond_j']:>6} "
              f"{p['coherence']:>8.4f} {str(p['sites_i']):>10} "
              f"{str(p['sites_j']):>10} {sv:>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', required=True,
                    help='logdir:label pairs, e.g. logs/gram_5c_standard:5-corner-H')
    ap.add_argument('--plan',    default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--graph',   default='data/graphs/rect_7x7.json')
    ap.add_argument('--top_k',   type=int, default=10)
    ap.add_argument('--save_whitened', action='store_true',
                    help='Save whitened G as gram_whitened.json in each logdir')
    args = ap.parse_args()

    g    = load_graph(args.graph)
    plan = json.load(open(args.plan))

    for config_spec in args.configs:
        parts = config_spec.split(':')
        logdir = parts[0]
        label  = parts[1] if len(parts) > 1 else logdir

        print(f"\n{'='*62}")
        print(f"{label}  ({logdir})")

        V = build_V(logdir, plan)
        print(f"  V shape: {V.shape}  "
              f"zero rows: {np.sum(np.all(V==0, axis=1))} "
              f"({100*np.sum(np.all(V==0,axis=1))/V.shape[0]:.1f}%)")

        for whiten, wlabel in [(False, 'Raw G'), (True, 'Whitened G')]:
            G  = compute_G(V, whiten=whiten)
            iu = np.triu_indices(N_BONDS, 1)
            pairs = top_pairs(G, g, args.top_k)
            print_pairs(wlabel, pairs,
                        float(np.median(G[iu])), float(G[iu].max()))

            if whiten and args.save_whitened:
                out = f'{logdir}/gram_whitened.json'
                result = {
                    'median_coh': float(np.median(G[iu])),
                    'max_coh':    float(G[iu].max()),
                    'pairs_detail': pairs,
                    'G': G.tolist(),
                }
                with open(out, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"  Saved whitened G: {out}")


if __name__ == '__main__':
    main()
