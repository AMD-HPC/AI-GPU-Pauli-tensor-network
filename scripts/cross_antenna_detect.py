#!/usr/bin/env python3
"""
cross_antenna_detect.py — Cross-antenna conflict detection.

A site v is CONFLICTED if it is RED in some antennas (shadow) and BLUE in
others (wall face). This means broken bonds approach v from multiple directions.
The broken bond in each direction is identified by which antenna is BLUE.

For a BLUE antenna (wave from direction D accumulated at v):
  → broken bond is between v and its neighbor in direction D

Usage:
  python3 scripts/cross_antenna_detect.py \
      --logdir logs/mawgm_v2/trial_5 \
      --antennas "row3:logs/mawgm_v2/trial_5/antennas/row3_base.log:logs/mawgm_v2/trial_5/antennas/row3_pert.log:down" \
                 "col5:logs/mawgm_v2/trial_5/antennas/col5_base.log:logs/mawgm_v2/trial_5/antennas/col5_pert.log:left" \
      --graph data/graphs/rect_7x7.json
"""

import argparse
import json
import sys
sys.path.insert(0, 'scripts')
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites
from integral_detect import compute_integral
import numpy as np

FLOOR = 1e-7


def direction_bond(v, direction, adj, all_sites):
    """
    Find the bond from site v in the given approach direction.
    direction: 'down' → wave came from above, goes down → bond below v
               'up'   → wave came from below, goes up → bond above v
               'left' → wave came from right, goes left → bond to right of v
               'right'→ wave came from left, goes right → bond to left of v

    For a rectangular 7x7 grid:
    row increases downward, col increases rightward.
    """
    row_v = (v-1)//7; col_v = (v-1)%7

    # Target neighbor based on direction wave is BLOCKED
    # BLUE = wall face = wave accumulated here = broken bond is in the direction
    # the wave was trying to go FROM the wall face
    if direction == 'down':    # wave from above, blocked going down → bond below
        nbr_row, nbr_col = row_v+1, col_v
    elif direction == 'up':    # wave from below, blocked going up → bond above
        nbr_row, nbr_col = row_v-1, col_v
    elif direction == 'left':  # wave from right, blocked going left → bond to right
        nbr_row, nbr_col = row_v, col_v+1
    elif direction == 'right': # wave from left, blocked going right → bond to left
        nbr_row, nbr_col = row_v, col_v-1
    else:
        return -1, -1

    if not (0 <= nbr_row < 7 and 0 <= nbr_col < 7):
        return -1, -1

    nbr = nbr_row*7 + nbr_col + 1
    # Find bond between v and nbr
    from bfs_detect import bond_color_class
    for neighbor, bond_idx in adj[v]:
        if neighbor == nbr:
            return bond_idx, nbr
    return -1, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--antennas', nargs='+', required=True,
                    help='"label:base_log:pert_log:direction"')
    ap.add_argument('--graph',   default='data/graphs/rect_7x7.json')
    ap.add_argument('--broken',  default=None)
    ap.add_argument('--theta',   type=float, default=0.1)
    ap.add_argument('--T_max',   type=int,   default=5)
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}

    # Load ground truth
    true_bonds = set()
    if args.broken:
        br = json.load(open(args.broken))
        true_bonds = set(br.get('bonds', [br.get('bond',-1)]))
        print(f"True bonds: {sorted(true_bonds)}")

    # Load all antenna results
    antenna_data = {}  # label → (log_ratio, direction)
    for ant in args.antennas:
        parts = ant.split(':')
        label, base_log, pert_log, direction = parts[0], parts[1], parts[2], parts[3]
        try:
            _, _, db = parse_pauli_log(base_log, 'NL')
            _, _, dp = parse_pauli_log(pert_log, 'NL')
            B, N, log_ratio = compute_integral(db, dp, args.T_max)
            antenna_data[label] = (log_ratio, direction)
            print(f"Loaded antenna: {label} ({direction})")
        except Exception as e:
            print(f"ERROR loading {label}: {e}")

    if len(antenna_data) < 2:
        print("Need at least 2 antennas"); return

    # Find conflicted sites
    print(f"\nConflicted sites (theta={args.theta}):")
    print(f"{'Site':>5}  {'row,col':>7}  {'antennas':>30}  {'bonds'}  {'note'}")
    print("-"*80)

    declared_bonds = set()
    for s in all_sites:
        si = site_idx[s]
        vals = {label: lr[si] for label, (lr, _) in antenna_data.items()}

        max_val = max(vals.values())
        min_val = min(vals.values())

        # Conflicted: RED in some, BLUE in others
        if max_val > args.theta and min_val < -args.theta:
            row = (s-1)//7; col = (s-1)%7

            # For each BLUE antenna: find the broken bond in that direction
            bonds_found = []
            for label, val in vals.items():
                if val < -args.theta:  # BLUE in this antenna
                    direction = antenna_data[label][1]
                    bond_idx, nbr = direction_bond(s, direction, adj, all_sites)
                    if bond_idx >= 0:
                        a, b = get_bond_sites(g['edges'][bond_idx])
                        bonds_found.append((bond_idx, a, b, label))
                        declared_bonds.add(bond_idx)

            ant_str = '  '.join([f"{k}={v:+.2f}" for k,v in sorted(vals.items())])
            bond_str = ' '.join([f"b{bi}({a}↔{b})[{lbl}]"
                                  for bi,a,b,lbl in bonds_found])
            true_tag = ' ← TRUE' if any(bi in true_bonds for bi,_,_,_ in bonds_found) else ''
            print(f"  {s:>3}  ({row},{col})  {ant_str}  {bond_str}{true_tag}")

    print(f"\nDeclared bonds: {sorted(declared_bonds)}")
    if true_bonds:
        found = declared_bonds & true_bonds
        both = true_bonds.issubset(declared_bonds)
        tag = '✓ BOTH' if both else ('~ PARTIAL' if found else '✗ NONE')
        print(f"True bonds found: {sorted(found)} / {sorted(true_bonds)}  → {tag}")


if __name__ == '__main__':
    main()
