#!/usr/bin/env python3
"""
run_seismic_antenna.py — Seismic + wall antenna pipeline using Python GPU pool.

Phase 1: BFS candidates from 3-scan seismic logs (already done)
Phase 2: High confidence (score > CONF_THR) → confirmed directly
         Low confidence → wall antenna, GPU pool (base+pert on same GPU)
Phase 3: Sign flip validation

Usage:
  python3 scripts/run_seismic_antenna.py --trials 1-10 --dj -0.20
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
from bfs_detect import bfs_detect
from wall_aperture import find_wall_centers

FLOOR = 1e-7
BINARY = './main_pauli_sweep_rect7x7'
CHI = 50; STEPS = 5; CUTOFF = '1e-3'
CONF_THR = 0.5   # score > this → confirmed directly from seismic
                 # (lower because 2 broken bonds reduce individual scores)
THR = 1e-3; THR_L = 0.05
N_GPUS = 3


def run_simulation(args):
    """Run one simulation on a specific GPU. Returns (returncode, logfile)."""
    cmd, logfile, gpu = args
    env = os.environ.copy()
    env['ROCR_VISIBLE_DEVICES'] = str(gpu)
    with open(logfile, 'w') as f:
        r = subprocess.run(cmd, stdout=f, stderr=f, env=env)
    return r.returncode, logfile


def run_antenna_pool(jobs, bond_tag, j_new):
    """
    Run wall antenna simulations using GPU pool.
    jobs = list of (bond_idx, centers_str, logdir)
    Returns dict: bond_idx -> (base_log, pert_log)
    """
    tasks = []  # (cmd, logfile, gpu)
    logs = {}   # bond_idx -> (base_log, pert_log)

    gpu = 0
    for bond_idx, centers_str, logdir in jobs:
        base_log = f'{logdir}/wall_b{bond_idx}_base.log'
        pert_log = f'{logdir}/wall_b{bond_idx}_pert.log'
        logs[bond_idx] = (base_log, pert_log)

        if not os.path.exists(base_log):
            cmd_base = [BINARY, str(CHI), str(STEPS), CUTOFF,
                        f'--center={centers_str}', '--state=NL']
            cmd_pert = cmd_base + [f'--perturb-bond={bond_tag},{j_new}']
            # base and pert on SAME GPU — sequential within GPU, parallel across GPUs
            tasks.append((cmd_base, base_log, gpu))
            tasks.append((cmd_pert, pert_log, gpu))

        gpu = (gpu + 1) % N_GPUS

    # Run with thread pool (subprocess releases GIL)
    if tasks:
        with ThreadPool(N_GPUS) as pool:
            pool.map(run_simulation, tasks)

    return logs


def check_sign_flip(base_log, pert_log, bond_idx, g, adj, all_sites, site_idx):
    """Check sign flip at bond endpoints. Returns (confirmed, cycle, contrast)."""
    try:
        _, _, db = parse_pauli_log(base_log, 'NL')
        _, _, dp = parse_pauli_log(pert_log, 'NL')
    except:
        return False, -1, 0.0

    a, b = get_bond_sites(g['edges'][bond_idx])
    nc = min(db.shape[0], dp.shape[0])
    best = 0.0; best_cyc = -1

    for t in range(nc):
        nb_a = db[t, site_idx[a]]; np_a = dp[t, site_idx[a]]
        nb_b = db[t, site_idx[b]]; np_b = dp[t, site_idx[b]]
        La = -np.log(max(np_a, FLOOR) / nb_a) if nb_a > THR else 0.0
        Lb = -np.log(max(np_b, FLOOR) / nb_b) if nb_b > THR else 0.0
        if (La < -THR_L and Lb > THR_L) or (La > THR_L and Lb < -THR_L):
            contrast = abs(La) + abs(Lb)
            if contrast > best:
                best = contrast; best_cyc = t + 1

    return best_cyc > 0, best_cyc, best


def run_trial(trial, logbase, plan, g, adj, all_sites, site_idx):
    logdir = f'{logbase}/trial_{trial}'
    broken_file = f'{logdir}/broken.json'
    if not os.path.exists(broken_file):
        return trial, None, None

    br = json.load(open(broken_file))
    true_bonds = set(br.get('bonds', [br.get('bond', -1)]))
    bond_tag = ';'.join(map(str, sorted(true_bonds)))
    j_new = f"{br.get('perturbation', {}).get('J_perturbed', 0.8):.3f}"

    print(f"\n--- Trial {trial}: bonds={bond_tag} J_new={j_new} ---")

    # Phase 1: BFS candidates
    ranked, _ = bfs_detect(logdir, plan, g, adj, all_sites, site_idx,
                            v_hop=0.448, thr=THR, thr_L=THR_L, T_max=5, verbose=False)

    confirmed = set()
    antenna_jobs = []  # (bond_idx, centers_str, logdir)

    for bi, score in ranked[:5]:
        a, b_site = get_bond_sites(g['edges'][bi])
        if score > CONF_THR:
            print(f"  Bond {bi:3d} ({a}↔{b_site}): CONFIRMED [seismic score={score:.2f}]")
            confirmed.add(bi)
        else:
            wall, orient, direction, near, a, b_site = find_wall_centers(
                bi, g, set(all_sites), offset=-1)
            if not wall:
                # Try opposite side
                wall, orient, direction, near, a, b_site = find_wall_centers(
                    bi, g, set(all_sites), offset=+1)
            if not wall:
                print(f"  Bond {bi:3d}: no wall found, skipping")
                continue
            cstr = ','.join(map(str, wall))
            print(f"  Bond {bi:3d} ({a}↔{b_site}): antenna [{direction} wall, score={score:.2f}]")
            antenna_jobs.append((bi, cstr, logdir))

    # Phase 2: run antenna jobs in GPU pool
    if antenna_jobs:
        logs = run_antenna_pool(antenna_jobs, bond_tag, j_new)

        # Phase 3: validate
        for bi, cstr, _ in antenna_jobs:
            base_log, pert_log = logs[bi]
            ok, cyc, contrast = check_sign_flip(
                base_log, pert_log, bi, g, adj, all_sites, site_idx)
            a, b_site = get_bond_sites(g['edges'][bi])
            tag = '✓ CONFIRMED' if ok else '✗ REJECTED'
            print(f"  Bond {bi:3d} ({a}↔{b_site}): {tag} [antenna cycle={cyc} contrast={contrast:.3f}]")
            if ok:
                confirmed.add(bi)

    both = true_bonds.issubset(confirmed)
    partial = bool(true_bonds & confirmed) and not both
    result = 'BOTH' if both else ('PARTIAL' if partial else 'NONE')
    print(f"  → {result}  confirmed={sorted(confirmed)}  true={sorted(true_bonds)}")
    return trial, result, confirmed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logbase', default='logs/mawgm_f64_2b')
    ap.add_argument('--plan',    default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--graph',   default='data/graphs/rect_7x7.json')
    ap.add_argument('--trials',  default='1-10')
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    plan = json.load(open(args.plan))

    # Parse trial range
    if '-' in args.trials:
        a, b = args.trials.split('-')
        trials = list(range(int(a), int(b)+1))
    else:
        trials = [int(x) for x in args.trials.split(',')]

    print(f"=== Seismic + Antenna Pipeline ===")
    print(f"Trials: {trials}  CONF_THR={CONF_THR}  N_GPUS={N_GPUS}")

    results = {}
    both_count = 0
    for trial in trials:
        t, result, confirmed = run_trial(
            trial, args.logbase, plan, g, adj, all_sites, site_idx)
        if result:
            results[t] = result
            if result == 'BOTH': both_count += 1

    print(f"\n=== FINAL: {both_count}/{len(trials)} BOTH ===")
    for t, r in sorted(results.items()):
        print(f"  Trial {t:2d}: {r}")


if __name__ == '__main__':
    main()
