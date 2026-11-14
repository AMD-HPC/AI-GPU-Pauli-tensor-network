#!/usr/bin/env python3
"""
run_antenna_grid_16x32.py — Full antenna grid for 16x32 (512-qubit) lattice.

16 row antennas + 32 col antennas = 48 simulations (base+pert each = 96 runs).
Each antenna fires all sites in a row/col simultaneously as --center=s1,s2,...
v_hop=0.5 keeps memory at ~4% VRAM per run — all fits on 1 GPU.

Site numbering: site = row*32 + col + 1  (row in [0,15], col in [0,31])

Detection: broken bond (a,b) produces sign flip in log(B/N):
  RED (log > +theta) on one endpoint, BLUE (log < -theta) on the other.
  Row antenna covers horizontal signal, col antenna covers vertical signal.

Usage:
  python3 scripts/run_antenna_grid_16x32.py \
      --logdir logs/mawgm_16x32/trial_1 \
      --broken logs/mawgm_16x32/trial_1/broken.json \
      --graph data/graphs/rect_16x32.json
"""

import argparse
import json
import os
import subprocess
import sys
from multiprocessing.pool import ThreadPool

sys.path.insert(0, 'scripts')
from searcher import load_graph, build_adj, get_bond_sites, parse_pauli_log
from integral_detect import compute_integral

BINARY  = './main_pauli_sweep_rect16x32'
CHI     = 50
STEPS   = 5
CUTOFF  = '1e-3'
V_HOP   = '--v_hop=0.5'
N_GPUS  = 3
THETA   = 0.1
NROWS   = 16
NCOLS   = 32


def site(row, col):
    return row * NCOLS + col + 1


def get_row_centers(row):
    return [site(row, col) for col in range(NCOLS)]


def get_col_centers(col):
    return [site(row, col) for row in range(NROWS)]


def run_antenna(args):
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
    ap.add_argument('--logdir',  required=True)
    ap.add_argument('--graph',   default='data/graphs/rect_16x32.json')
    ap.add_argument('--broken',  default=None)
    ap.add_argument('--theta',   type=float, default=THETA)
    ap.add_argument('--T_max',   type=int,   default=STEPS)
    ap.add_argument('--top_k',   type=int,   default=2)
    ap.add_argument('--n_gpus',  type=int,   default=N_GPUS)
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}

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

    tasks = []
    antenna_meta = {}

    for row in range(NROWS):
        centers = get_row_centers(row)
        cstr = ','.join(map(str, centers))
        key = f'row{row:02d}'
        base_log = f'{antenna_dir}/{key}_base.log'
        pert_log  = f'{antenna_dir}/{key}_pert.log'
        gpu = row % args.n_gpus
        tasks.append((cstr, base_log, pert_log, perturb_arg, gpu))
        antenna_meta[key] = ('row', row, centers, base_log, pert_log)

    for col in range(NCOLS):
        centers = get_col_centers(col)
        cstr = ','.join(map(str, centers))
        key = f'col{col:02d}'
        base_log = f'{antenna_dir}/{key}_base.log'
        pert_log  = f'{antenna_dir}/{key}_pert.log'
        gpu = col % args.n_gpus
        tasks.append((cstr, base_log, pert_log, perturb_arg, gpu))
        antenna_meta[key] = ('col', col, centers, base_log, pert_log)

    print(f"\nRunning {len(tasks)} antenna simulations "
          f"({NROWS} rows + {NCOLS} cols) on {args.n_gpus} GPU(s)...")
    with ThreadPool(args.n_gpus) as pool:
        pool.map(run_antenna, tasks)
    print("All antennas done.")

    # Compute integral log(B/N) per antenna
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

    # Score each bond by cross-antenna sign flip
    print(f"\nBond scores (cross-antenna sign flip, theta={args.theta}):")
    bond_scores = {}

    for bi, edge in enumerate(g['edges']):
        a, b = get_bond_sites(edge)
        ia, ib = site_idx[a], site_idx[b]

        best_contrast = 0.0
        for key, lr in antenna_logratios.items():
            la = lr[ia]
            lb = lr[ib]
            flip = (la > args.theta and lb < -args.theta) or \
                   (la < -args.theta and lb > args.theta)
            if flip:
                contrast = abs(la) + abs(lb)
                best_contrast = max(best_contrast, contrast)

        if best_contrast > args.theta:
            bond_scores[bi] = best_contrast

    # Print top-10
    print(f"\n{'Bond':>5}  {'sites':>8}  {'score':>10}  {'note'}")
    print("-" * 55)
    ranked = sorted(bond_scores.items(), key=lambda x: -x[1])
    for bi, score in ranked[:10]:
        a, b = get_bond_sites(g['edges'][bi])
        row_a, col_a = (a-1)//NCOLS, (a-1)%NCOLS
        row_b, col_b = (b-1)//NCOLS, (b-1)%NCOLS
        true_tag = ' <- TRUE' if bi in true_bonds else ''
        print(f"  {bi:>3}  {a:>3}({row_a},{col_a:2d})↔{b:>3}({row_b},{col_b:2d})"
              f"  {score:>10.4f}{true_tag}")

    declared = set(bi for bi, _ in ranked[:args.top_k])
    print(f"\nDeclared top-{args.top_k}: {sorted(declared)}")
    if true_bonds:
        found = declared & true_bonds
        success = true_bonds.issubset(declared)
        tag = 'FOUND' if success else ('PARTIAL' if found else 'MISSED')
        print(f"True bonds: {sorted(true_bonds)}  found: {sorted(found)}  -> {tag}")

    # Summary stats
    n_scored = len(bond_scores)
    print(f"\nBonds with sign flip: {n_scored} / {len(g['edges'])}")


if __name__ == '__main__':
    main()
