#!/usr/bin/env python3
"""
run_full_antenna_grid.py — Full antenna grid: all rows + all columns.

For 7x7 grid:
  7 row antennas (rows 0-6): each row fires simultaneously, wave propagates up+down
  7 col antennas (cols 0-6): each col fires simultaneously, wave propagates left+right

3 cycles only — bonds are 1 hop from nearest antenna row/col.
v_hop=0.5 → lightcone=1.5 hops covers immediate neighbors.

Cross-antenna conflict at site v:
  RED in row r (shadow) + BLUE in col c (wall face) → broken bond in direction r→v
  Or vice versa

28 total simulations (14 rows + 14 cols) × ~10s each ≈ 5 min on 3 GPUs.

Usage:
  python3 scripts/run_full_antenna_grid.py \
      --logdir logs/mawgm_v2/trial_5 \
      --broken logs/mawgm_v2/trial_5/broken.json \
      --graph data/graphs/rect_7x7.json
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
from integral_detect import compute_integral

BINARY  = './main_pauli_sweep_rect7x7'
CHI     = 50; STEPS = 3; CUTOFF = '1e-3'
V_HOP   = '--v_hop=0.5'
N_GPUS  = 3
THETA   = 0.1   # minimum |log(B/N)| to consider anomalous
NROWS   = 7; NCOLS = 7


def get_row_centers(row):
    """All sites in a given row (0-based)."""
    return [row*NCOLS + col + 1 for col in range(NCOLS)]


def get_col_centers(col):
    """All sites in a given column (0-based)."""
    return [row*NCOLS + col + 1 for row in range(NROWS)]


def run_antenna(args):
    """Run one antenna (base+pert sequential on same GPU)."""
    cstr, base_log, pert_log, perturb_arg, gpu = args
    if os.path.exists(base_log) and os.path.exists(pert_log):
        return
    env = os.environ.copy()
    env['ROCR_VISIBLE_DEVICES'] = str(gpu)
    cmd_base = [BINARY, str(CHI), str(STEPS), CUTOFF,
                f'--center={cstr}', '--state=NL', V_HOP]
    cmd_pert = cmd_base + [f'--perturb-bond={perturb_arg}']
    with open(base_log, 'w') as f:
        subprocess.run(cmd_base, stdout=f, stderr=f, env=env)
    with open(pert_log, 'w') as f:
        subprocess.run(cmd_pert, stdout=f, stderr=f, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',   required=True)
    ap.add_argument('--graph',    default='data/graphs/rect_7x7.json')
    ap.add_argument('--broken',   default=None)
    ap.add_argument('--theta',    type=float, default=THETA)
    ap.add_argument('--T_max',    type=int,   default=STEPS)
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    pos = {s: ((s-1)%NCOLS, 6-(s-1)//NCOLS) for s in all_sites}

    # Ground truth
    true_bonds = set()
    perturb_arg = ''
    if args.broken:
        br = json.load(open(args.broken))
        true_bonds = set(br.get('bonds', [br.get('bond', -1)]))
        j_new = br.get('perturbation', {}).get('J_perturbed', 0.8)
        bond_tag = ';'.join(map(str, sorted(true_bonds)))
        perturb_arg = f'{bond_tag},{j_new:.3f}'
        print(f"True bonds: {sorted(true_bonds)}  perturb: {perturb_arg}")

    antenna_dir = f'{args.logdir}/antenna_grid'
    os.makedirs(antenna_dir, exist_ok=True)

    # Build task list: all rows + all columns
    tasks = []
    antenna_meta = {}  # key → (type, idx, centers)

    for row in range(NROWS):
        centers = get_row_centers(row)
        cstr = ','.join(map(str, centers))
        key = f'row{row}'
        base_log = f'{antenna_dir}/{key}_base.log'
        pert_log  = f'{antenna_dir}/{key}_pert.log'
        gpu = (row) % N_GPUS
        tasks.append((cstr, base_log, pert_log, perturb_arg, gpu))
        antenna_meta[key] = ('row', row, centers, base_log, pert_log)

    for col in range(NCOLS):
        centers = get_col_centers(col)
        cstr = ','.join(map(str, centers))
        key = f'col{col}'
        base_log = f'{antenna_dir}/{key}_base.log'
        pert_log  = f'{antenna_dir}/{key}_pert.log'
        gpu = (col) % N_GPUS
        tasks.append((cstr, base_log, pert_log, perturb_arg, gpu))
        antenna_meta[key] = ('col', col, centers, base_log, pert_log)

    print(f"\nRunning {len(tasks)} antenna simulations ({NROWS} rows + {NCOLS} cols)...")
    with ThreadPool(N_GPUS) as pool:
        pool.map(run_antenna, tasks)
    print("All antennas done.")

    # Load all log(B/N) results
    print("\nComputing integral log(B/N) for each antenna...")
    antenna_logratios = {}
    for key, (atype, idx, centers, base_log, pert_log) in antenna_meta.items():
        try:
            _, _, db = parse_pauli_log(base_log, 'NL')
            _, _, dp = parse_pauli_log(pert_log, 'NL')
            B, N, lr = compute_integral(db, dp, args.T_max)
            antenna_logratios[key] = lr
        except Exception as e:
            print(f"  {key}: ERROR {e}")

    # Cross-antenna conflict detection
    print(f"\nCross-antenna conflict detection (theta={args.theta}):")
    print(f"{'Bond':>5}  {'sites':>8}  {'conflict':>10}  {'note'}")
    print("-"*60)

    declared = set()
    bond_scores = {}

    for bi in range(len(g['edges'])):
        a, b = get_bond_sites(g['edges'][bi])
        row_a = (a-1)//NCOLS; col_a = (a-1)%NCOLS
        row_b = (b-1)//NCOLS; col_b = (b-1)%NCOLS

        is_horizontal = (row_a == row_b)

        # For each site endpoint, check if it's conflicted across antenna pairs
        max_conflict = 0.0
        for s, row_s, col_s in [(a, row_a, col_a), (b, row_b, col_b)]:
            si = site_idx[s]
            # Compare row antennas above vs below this row
            row_above = f'row{max(0, row_s-1)}'
            row_below = f'row{min(NROWS-1, row_s+1)}'
            col_left  = f'col{max(0, col_s-1)}'
            col_right = f'col{min(NCOLS-1, col_s+1)}'

            for k1, k2 in [(row_above, row_below), (col_left, col_right),
                           (f'row{row_s}', f'col{col_s}')]:
                if k1 not in antenna_logratios or k2 not in antenna_logratios:
                    continue
                v1 = antenna_logratios[k1][si]
                v2 = antenna_logratios[k2][si]
                # Conflict: opposite signs
                if (v1 > args.theta and v2 < -args.theta) or \
                   (v1 < -args.theta and v2 > args.theta):
                    conflict = abs(v1) + abs(v2)
                    max_conflict = max(max_conflict, conflict)

        if max_conflict > args.theta:
            bond_scores[bi] = max_conflict

    # Rank and declare
    ranked = sorted(bond_scores.items(), key=lambda x: -x[1])
    n_true = len(true_bonds) if true_bonds else 5
    for bi, score in ranked[:10]:
        a, b = get_bond_sites(g['edges'][bi])
        true_tag = ' ← TRUE' if bi in true_bonds else ''
        print(f"  {bi:>3}  {a:>3}↔{b:>3}  {score:>10.4f}{true_tag}")

    declared = set(bi for bi, _ in ranked[:n_true])
    print(f"\nDeclared: {sorted(declared)}")
    if true_bonds:
        found = declared & true_bonds
        both = true_bonds.issubset(declared)
        tag = '✓ BOTH' if both else ('~ PARTIAL' if found else '✗ NONE')
        print(f"True bonds found: {sorted(found)} / {sorted(true_bonds)}  → {tag}")


if __name__ == '__main__':
    main()
