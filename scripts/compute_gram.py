#!/usr/bin/env python3
"""
compute_gram.py — Compute Gram matrix of bond response vectors.

Runs 85 simulations (1 baseline + 84 perturbed, one per bond) and
computes the pairwise cosine similarity matrix G[i,j] = |cos(s_i, s_j)|.

Usage:
  python3 scripts/compute_gram.py \
      --logdir logs/gram_f32 \
      --graph data/graphs/rect_7x7.json \
      --plan data/scans/rect_7x7_scan_3gpu.json \
      --binary ./main_pauli_sweep_rect7x7 \
      --j_pert 1.2 \
      --out /tmp/gram_f32.json

  # Swapped color version:
  python3 scripts/compute_gram.py \
      --logdir logs/gram_f32_swapped \
      --binary ./main_pauli_sweep_rect7x7_swapped \
      --out /tmp/gram_f32_swapped.json
"""

import argparse
import json
import os
import subprocess
import numpy as np
from multiprocessing.pool import ThreadPool
import sys
sys.path.insert(0, 'scripts')
from searcher import load_graph, build_adj, get_bond_sites, parse_pauli_log

N_GPUS  = 3
CHI     = 50
STEPS   = 5
CUTOFF  = '1e-3'


def run_sim(args):
    cmd, logfile, gpu = args
    if os.path.exists(logfile):
        return
    env = os.environ.copy()
    env['ROCR_VISIBLE_DEVICES'] = str(gpu)
    with open(logfile, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=f, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',  default='logs/gram_f32')
    ap.add_argument('--graph',   default='data/graphs/rect_7x7.json')
    ap.add_argument('--plan',    default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--binary',  default='./main_pauli_sweep_rect7x7')
    ap.add_argument('--j_pert',  type=float, default=1.2)
    ap.add_argument('--out',     default='/tmp/gram.json')
    ap.add_argument('--top_k',   type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    plan = json.load(open(args.plan))
    n_bonds = len(g['edges'])
    n_sites = len(all_sites)
    n_steps = STEPS

    # Build center string from all scan centers combined (seismic)
    all_centers = []
    for sc in plan['placements']:
        all_centers.extend(sc['centers'])
    all_centers = sorted(set(all_centers))
    center_str = ','.join(map(str, all_centers))

    print(f"Graph: {n_bonds} bonds, {n_sites} sites")
    print(f"Centers: {center_str}")
    print(f"Binary: {args.binary}")
    print(f"Perturbation: J={args.j_pert}")
    print(f"Logdir: {args.logdir}")

    # Build simulation task list
    tasks = []

    # Baseline
    base_log = f'{args.logdir}/base.log'
    cmd_base = [args.binary, str(CHI), str(STEPS), CUTOFF,
                f'--center={center_str}', '--state=NL']
    tasks.append((cmd_base, base_log, 0))

    # 84 perturbed — one per bond
    for bi in range(n_bonds):
        pert_log = f'{args.logdir}/pert_b{bi}.log'
        cmd_pert = cmd_base + [f'--perturb-bond={bi},{args.j_pert:.3f}']
        tasks.append((cmd_pert, pert_log, bi % N_GPUS))

    # Run missing simulations
    missing = [(c, l, g_) for c, l, g_ in tasks if not os.path.exists(l)]
    if missing:
        print(f"\nRunning {len(missing)} simulations ({N_GPUS} GPUs parallel)...")
        with ThreadPool(N_GPUS) as pool:
            pool.map(run_sim, missing)
        print("Done.")
    else:
        print("\nAll logs exist, skipping simulations.")

    # Load baseline
    print("\nLoading response vectors...")
    try:
        _, _, db = parse_pauli_log(base_log, 'NL')
    except Exception as e:
        print(f"ERROR loading baseline: {e}"); return

    # Build response matrix S[bond, t*site] = L(v,t) = -log(n_pert/n_base)
    FLOOR = 1e-7
    n_steps_actual = db.shape[0]
    n_dim = n_steps_actual * n_sites
    S = np.zeros((n_bonds, n_dim))

    for bi in range(n_bonds):
        pert_log = f'{args.logdir}/pert_b{bi}.log'
        try:
            _, _, dp = parse_pauli_log(pert_log, 'NL')
            nc = min(db.shape[0], dp.shape[0])
            for t in range(nc):
                for v in range(n_sites):
                    if db[t, v] > FLOOR:
                        S[bi, t*n_sites + v] = -np.log(
                            max(dp[t, v], FLOOR) / db[t, v])
        except Exception as e:
            print(f"  Bond {bi}: ERROR {e}")

    print(f"Response matrix shape: {S.shape}")

    # Compute Gram matrix
    norms = np.linalg.norm(S, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Sn = S / norms
    G = np.abs(Sn @ Sn.T)
    np.fill_diagonal(G, 0)

    # Statistics
    iu = np.triu_indices(n_bonds, 1)
    vals = G[iu]
    median_coh = float(np.median(vals))
    max_coh    = float(vals.max())

    print(f"\nMedian coherence: {median_coh:.4f}")
    print(f"Max coherence:    {max_coh:.4f}")

    # Detectability
    detectability = np.max(S, axis=1)
    print(f"Detectability min: {detectability.min():.4f} (bond {detectability.argmin()})")
    print(f"Detectability max: {detectability.max():.4f} (bond {detectability.argmax()})")
    print(f"Detectability span: {detectability.max()/max(detectability.min(),1e-6):.1f}x")

    # Top-k worst pairs
    order = np.dstack(np.unravel_index(np.argsort(G.ravel())[::-1], G.shape))[0]
    pairs = []
    seen = set()
    for a, b in order:
        if a >= b or (a, b) in seen: continue
        seen.add((a, b))
        a_sites = get_bond_sites(g['edges'][a])
        b_sites = get_bond_sites(g['edges'][b])
        shared = set(a_sites) & set(b_sites)
        pairs.append({
            'bond_i': int(a), 'bond_j': int(b),
            'coherence': float(G[a, b]),
            'shares_vertex': len(shared) > 0,
            'shared_vertex': int(shared.pop()) if shared else None,
            'sites_i': list(a_sites), 'sites_j': list(b_sites)
        })
        if len(pairs) == args.top_k: break

    print(f"\nTop-{args.top_k} confusable pairs:")
    print(f"{'Rank':<5} {'Bond i':>6} {'Bond j':>6} {'coh':>8} {'sites_i':>10} {'sites_j':>10} {'shared_v':>9}")
    print("-"*60)
    for rank, p in enumerate(pairs, 1):
        sv = p['shared_vertex'] or '-'
        print(f"{rank:<5} {p['bond_i']:>6} {p['bond_j']:>6} {p['coherence']:>8.4f} "
              f"{str(p['sites_i']):>10} {str(p['sites_j']):>10} {str(sv):>9}")

    # Save results
    result = {
        'median_coh': median_coh,
        'max_coh': max_coh,
        'pairs': [[p['bond_i'], p['bond_j'], p['coherence'], p['shares_vertex']]
                  for p in pairs],
        'pairs_detail': pairs,
        'detectability': detectability.tolist(),
        'G': G.tolist(),
    }
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
