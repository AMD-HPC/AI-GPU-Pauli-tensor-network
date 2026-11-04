#!/usr/bin/env python3
"""
compute_gram.py — Compute Gram matrix using the 3-scan seismic structure.

For each bond e:
  - 3 baseline simulations (one per scan, shared across all bonds)
  - 3 perturbed simulations (one per scan)
  - Response vector s_e = concat of L(v,t) from all 3 scans
  - G[i,j] = |cos(s_i, s_j)|

Each scan uses its own center subset -> smaller center set -> BP converges.
Total: 3 baselines + 84*3 = 255 simulations.

Usage:
  python3 scripts/compute_gram.py \
      --logdir logs/gram_standard \
      --binary ./main_pauli_sweep_rect7x7 \
      --out logs/gram_standard/gram.json

  python3 scripts/compute_gram.py \
      --logdir logs/gram_swapped \
      --binary ./main_pauli_sweep_rect7x7_swapped \
      --out logs/gram_swapped/gram.json
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
FLOOR   = 1e-7


import threading
_gpu_local = threading.local()
_gpu_counter = 0
_gpu_lock = threading.Lock()

def _get_worker_gpu():
    if not hasattr(_gpu_local, 'gpu'):
        global _gpu_counter
        with _gpu_lock:
            _gpu_local.gpu = _gpu_counter % N_GPUS
            _gpu_counter += 1
    return _gpu_local.gpu

def run_sim(args):
    cmd, logfile = args
    if os.path.exists(logfile):
        return
    gpu = _get_worker_gpu()
    env = os.environ.copy()
    env['ROCR_VISIBLE_DEVICES'] = str(gpu)
    with open(logfile, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=f, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',  default='logs/gram_standard')
    ap.add_argument('--graph',   default='data/graphs/rect_7x7.json')
    ap.add_argument('--plan',    default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--binary',  default='./main_pauli_sweep_rect7x7')
    ap.add_argument('--j_pert',  type=float, default=0.8)
    ap.add_argument('--out',     default='logs/gram_standard/gram.json')
    ap.add_argument('--top_k',   type=int, default=10)
    ap.add_argument('--centers', default=None,
                    help='Override centers: comma-separated sites for a single scan. '
                         'Bypasses the scan plan entirely.')
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    plan = json.load(open(args.plan))
    n_bonds = len(g['edges'])
    n_sites = len(all_sites)

    # Build scan list — either from --centers override or from scan plan
    if args.centers:
        cstr = args.centers
        scans_to_run = [{'sid': 1, 'cstr': cstr}]
        print(f"Graph: {n_bonds} bonds, {n_sites} sites")
        print(f"Binary: {args.binary}")
        print(f"Perturbation: J={args.j_pert}")
        print(f"Logdir: {args.logdir}")
        print(f"Centers override: {cstr} (single scan)")
    else:
        scans_to_run = [{'sid': sc['scan'],
                         'cstr': ','.join(map(str, sc['centers']))}
                        for sc in plan['placements']]
        print(f"Graph: {n_bonds} bonds, {n_sites} sites")
        print(f"Binary: {args.binary}")
        print(f"Perturbation: J={args.j_pert}")
        print(f"Logdir: {args.logdir}")
        print(f"Scans: {len(scans_to_run)}")

    tasks = []
    scan_info = []

    for sc in scans_to_run:
        sid = sc['sid']; cstr = sc['cstr']
        base_log = f'{args.logdir}/scan{sid}_base.log'
        cmd_base = [args.binary, str(CHI), str(STEPS), CUTOFF,
                    f'--center={cstr}', '--state=NL']
        tasks.append((cmd_base, base_log))
        scan_info.append((sid, cstr, base_log, cmd_base))

        for bi in range(n_bonds):
            pert_log = f'{args.logdir}/scan{sid}_pert_b{bi}.log'
            cmd_pert = cmd_base + [f'--perturb-bond={bi},{args.j_pert:.3f}']
            tasks.append((cmd_pert, pert_log))

    # Run missing simulations
    missing = [(c, l) for c, l in tasks if not os.path.exists(l)]
    total = len(tasks)
    print(f"\nTotal simulations: {total}  Missing: {len(missing)}")
    if missing:
        print(f"Running {len(missing)} simulations ({N_GPUS} GPUs parallel)...")
        with ThreadPool(N_GPUS) as pool:
            pool.map(run_sim, missing)
        print("Done.")
    else:
        print("All logs exist, skipping simulations.")

    # Build response matrix S[bond, features]
    # features = concat of L(v,t) from each scan
    print("\nLoading response vectors...")

    # First pass: get shape from one log
    sample_log = f'{args.logdir}/scan{plan["placements"][0]["scan"]}_base.log'
    _, _, db_sample = parse_pauli_log(sample_log, 'NL')
    n_steps = db_sample.shape[0]
    n_scans = len(plan['placements'])
    n_dim = n_scans * n_steps * n_sites

    S = np.zeros((n_bonds, n_dim))

    for scan_idx, (sid, cstr, base_log, _) in enumerate(scan_info):
        try:
            _, _, db = parse_pauli_log(base_log, 'NL')
        except Exception as e:
            print(f"  Scan {sid} baseline: ERROR {e}")
            continue

        offset = scan_idx * n_steps * n_sites

        for bi in range(n_bonds):
            pert_log = f'{args.logdir}/scan{sid}_pert_b{bi}.log'
            try:
                _, _, dp = parse_pauli_log(pert_log, 'NL')
                nc = min(db.shape[0], dp.shape[0])
                for t in range(nc):
                    for v in range(n_sites):
                        if db[t, v] > FLOOR:
                            S[bi, offset + t*n_sites + v] = -np.log(
                                max(dp[t, v], FLOOR) / db[t, v])
            except Exception as e:
                print(f"  Bond {bi} scan {sid}: ERROR {e}")

    print(f"Response matrix shape: {S.shape}")

    # Gram matrix
    norms = np.linalg.norm(S, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Sn = S / norms
    G = np.abs(Sn @ Sn.T)
    np.fill_diagonal(G, 0)

    iu = np.triu_indices(n_bonds, 1)
    vals = G[iu]
    median_coh = float(np.median(vals))
    max_coh    = float(vals.max())
    detectability = np.max(np.abs(S), axis=1)

    print(f"\nMedian coherence: {median_coh:.4f}")
    print(f"Max coherence:    {max_coh:.4f}")
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

    result = {
        'median_coh': median_coh,
        'max_coh': max_coh,
        'pairs': [[p['bond_i'], p['bond_j'], p['coherence'], p['shares_vertex']]
                  for p in pairs],
        'pairs_detail': pairs,
        'detectability': detectability.tolist(),
        'G': G.tolist(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
