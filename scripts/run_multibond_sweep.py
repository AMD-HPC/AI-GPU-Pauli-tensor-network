#!/usr/bin/env python3
"""
run_multibond_sweep.py — Generate and evaluate seismic detection for 4-bond and 8-bond trials.

Reuses existing mawgm_v2 logs (2-bond). Generates new logs for 4-bond and 8-bond.
Measures: how many true bonds appear in top-k seismic ranking (single iteration).

Usage:
  python3 scripts/run_multibond_sweep.py --n_bonds 4 --n_trials 5
  python3 scripts/run_multibond_sweep.py --n_bonds 8 --n_trials 2
  python3 scripts/run_multibond_sweep.py --eval_only --logbase logs/mawgm_4b
"""

import argparse
import json
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool
sys.path.insert(0, 'scripts')
from searcher import load_graph, build_adj, get_bond_sites, parse_pauli_log
from bfs_detect import bfs_detect

BINARY   = './main_pauli_sweep_rect7x7'
CHI      = 50; STEPS = 5; CUTOFF = '1e-3'
N_GPUS   = 3
DJ       = -0.20
SEED     = 42


def run_scan(args):
    cstr, base_log, pert_log, perturb_arg, gpu = args
    if os.path.exists(base_log) and os.path.exists(pert_log):
        return
    env = os.environ.copy()
    env['ROCR_VISIBLE_DEVICES'] = str(gpu)
    cmd_base = [BINARY, str(CHI), str(STEPS), CUTOFF,
                f'--center={cstr}', '--state=NL']
    cmd_pert = cmd_base + [f'--perturb-bond={perturb_arg}']
    with open(base_log, 'w') as f:
        subprocess.run(cmd_base, stdout=f, stderr=f, env=env)
    with open(pert_log, 'w') as f:
        subprocess.run(cmd_pert, stdout=f, stderr=f, env=env)


def generate_trial(trial, n_bonds, logbase, graph, plan, dj=DJ, seed=SEED):
    import random
    logdir = f'{logbase}/trial_{trial}'
    os.makedirs(logdir, exist_ok=True)
    broken_file = f'{logdir}/broken.json'

    if not os.path.exists(broken_file):
        random.seed(seed + trial)
        g = load_graph(graph)
        bonds = random.sample(range(len(g['edges'])), n_bonds)
        j_new = round(1.0 + dj, 3)
        # Run breaker
        bond_str = ','.join(map(str, bonds))
        subprocess.run([
            'python3', 'scripts/breaker.py',
            '--graph', graph, '--bond', bond_str,
            '--coupling', str(dj), '--out', broken_file
        ], capture_output=True)
        print(f"  Trial {trial}: bonds={bonds}")
    else:
        br = json.load(open(broken_file))
        bonds = br.get('bonds', [br.get('bond', -1)])
        j_new = br.get('perturbation', {}).get('J_perturbed', 0.8)
        print(f"  Trial {trial}: bonds={bonds} (existing)")

    return logdir


def run_seismic(trial, logbase, plan_file, graph, n_bonds):
    logdir = f'{logbase}/trial_{trial}'
    broken_file = f'{logdir}/broken.json'
    if not os.path.exists(broken_file):
        return None

    br = json.load(open(broken_file))
    bonds = br.get('bonds', [br.get('bond', -1)])
    j_new = br.get('perturbation', {}).get('J_perturbed', 0.8)
    bond_tag = ';'.join(map(str, sorted(bonds)))
    perturb_arg = f'{bond_tag},{j_new:.3f}'

    plan = json.load(open(plan_file))

    # Build scan tasks
    tasks = []
    for i, sc in enumerate(plan['placements']):
        sid = sc['scan']
        cstr = ','.join(map(str, sc['centers']))
        base_log = f'{logdir}/scan{sid}_base.log'
        pert_log = f'{logdir}/scan{sid}_pert.log'
        tasks.append((cstr, base_log, pert_log, perturb_arg, i % N_GPUS))

    # Run missing scans
    missing = [(c, b, p, pa, g) for c, b, p, pa, g in tasks
               if not os.path.exists(b) or not os.path.exists(p)]
    if missing:
        print(f"  Trial {trial}: running {len(missing)} scans...")
        with ThreadPool(N_GPUS) as pool:
            pool.map(run_scan, missing)

    return logdir


def evaluate(logbase, plan_file, graph_file, n_trials, top_k):
    g = load_graph(graph_file)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    plan = json.load(open(plan_file))

    print(f"\n{'Trial':>6}  {'True bonds':>20}  {'Top-{} declared'.format(top_k):>25}  {'Found':>6}  {'Result'}")
    print("-"*80)

    total_found = 0
    total_true = 0
    both_count = 0

    for trial in range(1, n_trials + 1):
        logdir = f'{logbase}/trial_{trial}'
        broken_file = f'{logdir}/broken.json'
        if not os.path.exists(broken_file):
            continue

        br = json.load(open(broken_file))
        true_bonds = set(br.get('bonds', [br.get('bond', -1)]))

        try:
            ranked, _, _ = bfs_detect(logdir, plan, g, adj, all_sites, site_idx,
                                      v_hop=0.448, thr=1e-3, thr_L=0.05,
                                      T_max=5, verbose=False)
        except Exception as e:
            print(f"  Trial {trial}: ERROR {e}")
            continue

        declared = set(bi for bi, _ in ranked[:top_k])
        found = declared & true_bonds
        n_found = len(found)
        n_true = len(true_bonds)
        both = true_bonds.issubset(declared)
        tag = '✓ ALL' if both else f'{n_found}/{n_true}'

        print(f"  {trial:>4}  {str(sorted(true_bonds)):>20}  "
              f"{str(sorted(declared)):>25}  {n_found:>3}/{n_true:<3}  {tag}")

        total_found += n_found
        total_true += n_true
        if both: both_count += 1

    print("-"*80)
    print(f"  Total found: {total_found}/{total_true}  "
          f"ALL found: {both_count}/{n_trials}")
    return both_count, n_trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_bonds',   type=int, default=4)
    ap.add_argument('--n_trials',  type=int, default=5)
    ap.add_argument('--logbase',   default=None)
    ap.add_argument('--plan',      default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--graph',     default='data/graphs/rect_7x7.json')
    ap.add_argument('--eval_only', action='store_true')
    ap.add_argument('--top_k',     type=int, default=None)
    args = ap.parse_args()

    if args.logbase is None:
        args.logbase = f'logs/mawgm_{args.n_bonds}b'
    if args.top_k is None:
        args.top_k = args.n_bonds

    os.makedirs(args.logbase, exist_ok=True)
    print(f"\n=== Multi-bond seismic sweep: n_bonds={args.n_bonds}, "
          f"n_trials={args.n_trials}, top_k={args.top_k} ===")
    print(f"Logbase: {args.logbase}")

    if not args.eval_only:
        print(f"\nGenerating {args.n_trials} trials with {args.n_bonds} bonds...")
        for trial in range(1, args.n_trials + 1):
            generate_trial(trial, args.n_bonds, args.logbase,
                           args.graph, args.plan)

        print(f"\nRunning seismic scans...")
        for trial in range(1, args.n_trials + 1):
            run_seismic(trial, args.logbase, args.plan, args.graph, args.n_bonds)

    print(f"\nEvaluating top-{args.top_k} seismic detection...")
    evaluate(args.logbase, args.plan, args.graph, args.n_trials, args.top_k)


if __name__ == '__main__':
    main()
