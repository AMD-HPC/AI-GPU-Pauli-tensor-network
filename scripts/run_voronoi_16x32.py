#!/usr/bin/env python3
"""
run_voronoi_16x32.py — Voronoi center scan for 16x32 (512-qubit) tomography.

32 centers on a 4x8 grid (spacing=4 hops), each run independently with v_hop.
Distributes across N GPUs via round-robin.

1-GPU mode:  32 centers x 2 (base+pert) = 64 runs, ~2 min
2-GPU mode:  same, 2x parallel, ~1 min
3-GPU mode:  same, 3x parallel, ~40 sec

Usage:
  python3 scripts/run_voronoi_16x32.py \
      --logdir logs/mawgm_16x32/trial_1 \
      --broken logs/mawgm_16x32/trial_1/broken.json \
      --n_gpus 3 --top_k 1

  # Eval only (logs already exist):
  python3 scripts/run_voronoi_16x32.py \
      --logdir logs/mawgm_16x32/trial_1 \
      --broken logs/mawgm_16x32/trial_1/broken.json \
      --eval_only --top_k 1
"""

import argparse
import json
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool

sys.path.insert(0, 'scripts')
from searcher import load_graph, build_adj, get_bond_sites, parse_pauli_log
from detect_delta_logT import compute_scores, collect_log_pairs
from integral_detect import compute_integral

BINARY  = './main_pauli_sweep_rect16x32'
CHI     = 50
STEPS   = 5
CUTOFF  = '1e-3'
V_HOP   = '--v_hop=0.5'
N_GPUS  = 3
NCOLS   = 32

PLAN_FILE = 'data/scans/rect_16x32_voronoi.json'


def run_center(args):
    center, base_log, pert_log, perturb_arg, gpu = args
    if os.path.exists(base_log) and os.path.exists(pert_log):
        return
    env = os.environ.copy()
    env['ROCR_VISIBLE_DEVICES'] = str(gpu)
    cmd_base = [BINARY, str(CHI), str(STEPS), CUTOFF,
                f'--center={center}', '--state=NL', V_HOP]
    cmd_pert = cmd_base + [f'--perturb-bond={perturb_arg}']
    with open(base_log, 'w') as f:
        subprocess.run(cmd_base, stdout=f, stderr=f, env=env)
    with open(pert_log, 'w') as f:
        subprocess.run(cmd_pert, stdout=f, stderr=f, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',    required=True)
    ap.add_argument('--broken',    default=None)
    ap.add_argument('--graph',     default='data/graphs/rect_16x32.json')
    ap.add_argument('--plan',      default=PLAN_FILE)
    ap.add_argument('--n_gpus',    type=int, default=N_GPUS)
    ap.add_argument('--top_k',     type=int, default=1)
    ap.add_argument('--thr',       type=float, default=0.01)
    ap.add_argument('--eval_only', action='store_true')
    args = ap.parse_args()

    plan = json.load(open(args.plan))
    g    = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}

    true_bonds = set()
    perturb_arg = ''
    if args.broken:
        br = json.load(open(args.broken))
        true_bonds = set(br.get('bonds', [br.get('bond', -1)]))
        j_new = br.get('perturbation', {}).get('J_perturbed', 0.8)
        bond_tag = ';'.join(map(str, sorted(true_bonds)))
        perturb_arg = f'{bond_tag},{j_new:.3f}'
        print(f"True bonds: {sorted(true_bonds)}  perturb: {perturb_arg}")

    scan_dir = f'{args.logdir}/voronoi'
    os.makedirs(scan_dir, exist_ok=True)

    if not args.eval_only:
        tasks = []
        for i, center in enumerate(plan['centers']):
            base_log = f'{scan_dir}/c{center}_base.log'
            pert_log = f'{scan_dir}/c{center}_pert.log'
            gpu = i % args.n_gpus
            tasks.append((center, base_log, pert_log, perturb_arg, gpu))

        print(f"\nRunning {len(tasks)} centers on {args.n_gpus} GPU(s)...")
        with ThreadPool(args.n_gpus) as pool:
            pool.map(run_center, tasks)
        print("All centers done.")

    # Detection using δlnT
    pairs = collect_log_pairs(scan_dir)
    print(f"\nLog pairs: {len(pairs)}")
    scores  = compute_scores(pairs, g, site_idx, args.thr)

    import numpy as np
    ranked   = np.argsort(-scores)
    declared = set(int(ranked[i]) for i in range(args.top_k))

    print(f"\nδlnT detection  thr={args.thr}  top_k={args.top_k}")
    print(f"\n{'Rank':>4} {'Bond':>5} {'sites (row,col)':>24} {'Score':>10} {'TRUE?':>8}")
    print("-" * 60)
    for rank in range(min(args.top_k + 9, len(ranked))):
        bi = int(ranked[rank])
        if scores[bi] == 0:
            break
        u, v = get_bond_sites(g['edges'][bi])
        ru, cu = (u-1)//NCOLS, (u-1)%NCOLS
        rv, cv = (v-1)//NCOLS, (v-1)%NCOLS
        true_tag = '<- TRUE' if bi in true_bonds else ''
        marker   = '<--' if rank < args.top_k else ''
        print(f"  {rank+1:>2}  {bi:>5}  {u:>3}({ru},{cu:2d})↔{v:>3}({rv},{cv:2d})"
              f"  {scores[bi]:>10.4f}  {true_tag} {marker}")

    if true_bonds:
        found = declared & true_bonds
        tag = 'FOUND' if true_bonds.issubset(declared) else ('PARTIAL' if found else 'MISSED')
        print(f"\nResult: {tag}  declared={sorted(declared)}  true={sorted(true_bonds)}")
        for tb in sorted(true_bonds):
            rank_tb = int(np.where(ranked == tb)[0][0]) + 1
            print(f"  bond {tb} rank: {rank_tb}  score: {scores[tb]:.6f}")


if __name__ == '__main__':
    main()
