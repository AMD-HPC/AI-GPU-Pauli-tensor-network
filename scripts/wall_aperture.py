#!/usr/bin/env python3
"""
wall_aperture.py — Wall of centers aperture for weak-signal bonds.

For a horizontal bond (u↔v, same row, cols c↔c+1):
  Place all centers in column c-1 (one hop left of near endpoint u).
  All centers approach from same side → constructive interference, no sandwiching.
  Run seismically (all wall centers simultaneously) → 2 simulations total.

For a vertical bond (u↔v, same col, rows r↔r+1):
  Place all centers in row r-1 (one row above near endpoint u).

Usage:
  python3 scripts/wall_aperture.py --bond 34 --graph data/graphs/rect_7x7.json
  # → prints center list and simulation commands
"""

import argparse
import json
import sys
sys.path.insert(0, 'scripts')
from searcher import load_graph, build_adj, get_bond_sites

def find_wall_centers(bond_idx, g, all_sites, offset=-1):
    """
    Find wall of centers for a bond.
    For horizontal bond: centers in column (col_near + offset)
    For vertical bond:   centers in row (row_near + offset)

    offset=-1: approach from left/above (default)
    offset=+1: approach from right/below
    """
    a, b = get_bond_sites(g['edges'][bond_idx])
    row_a = (a-1)//7; col_a = (a-1)%7
    row_b = (b-1)//7; col_b = (b-1)%7

    is_horizontal = (row_a == row_b)
    near = a if col_a <= col_b else b  # left endpoint for horizontal
    if not is_horizontal:
        near = a if row_a <= row_b else b  # top endpoint for vertical

    near_row = (near-1)//7; near_col = (near-1)%7

    wall = []
    if is_horizontal:
        wall_col = near_col + offset
        if 0 <= wall_col <= 6:
            for row in range(7):
                s = row*7 + wall_col + 1
                if s in all_sites:
                    wall.append(s)
    else:
        wall_row = near_row + offset
        if 0 <= wall_row <= 6:
            for col in range(7):
                s = wall_row*7 + col + 1
                if s in all_sites:
                    wall.append(s)

    pos = {s: ((s-1)%7, 6-(s-1)//7) for s in range(1,50)}
    direction = 'left' if (is_horizontal and offset==-1) else \
                'right' if (is_horizontal and offset==1) else \
                'above' if offset==-1 else 'below'
    orientation = 'horizontal' if is_horizontal else 'vertical'

    return wall, orientation, direction, near, a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bond',    type=int, required=True)
    ap.add_argument('--graph',   default='data/graphs/rect_7x7.json')
    ap.add_argument('--offset',  type=int, default=-1,
                    help='-1=left/above (default), +1=right/below')
    ap.add_argument('--logdir',  default=None)
    ap.add_argument('--perturb', default=None, help='e.g. "34;53,0.800"')
    ap.add_argument('--chi',     type=int, default=50)
    ap.add_argument('--steps',   type=int, default=5)
    ap.add_argument('--cutoff',  default='1e-3')
    ap.add_argument('--binary',  default='./main_pauli_sweep_rect7x7')
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)

    wall, orientation, direction, near, a, b = find_wall_centers(
        args.bond, g, set(all_sites), args.offset)

    pos = {s: ((s-1)%7, 6-(s-1)//7) for s in range(1,50)}
    print(f"Bond {args.bond} ({a}↔{b}) — {orientation}")
    print(f"Near endpoint: site {near} at {pos[near]}")
    print(f"Wall centers ({direction}): {wall}")
    print(f"Centers as string: {','.join(map(str,wall))}")
    print()

    if args.logdir and args.perturb:
        cstr = ','.join(map(str, wall))
        base_log = f"{args.logdir}/wall_b{args.bond}_base.log"
        pert_log = f"{args.logdir}/wall_b{args.bond}_pert.log"
        print("Simulation commands:")
        print(f"ROCR_VISIBLE_DEVICES=0 {args.binary} {args.chi} {args.steps} {args.cutoff} "
              f"--center={cstr} --state=NL > {base_log} 2>&1")
        print(f"ROCR_VISIBLE_DEVICES=1 {args.binary} {args.chi} {args.steps} {args.cutoff} "
              f"--center={cstr} --state=NL \"--perturb-bond={args.perturb}\" > {pert_log} 2>&1")
        print()
        print("Then validate:")
        print(f"python3 scripts/validate_aperture.py \\")
        print(f"    --logdir {args.logdir} \\")
        print(f"    --candidates \"{args.bond}:{near}\" \\")
        print(f"    --T_max {args.steps} --thr_L 0.05")


if __name__ == '__main__':
    main()
