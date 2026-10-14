#!/usr/bin/env python3
"""
validate_aperture.py — Validate candidate bonds via targeted aperture.

Detection rule: sign flip across bond endpoints within T_max cycles.
- CENTER on near endpoint, RED appears at far endpoint → CONFIRMED
- No sign flip → REJECTED

Threshold thr=1e-3 maintained for hardware measurability.

Usage:
  python3 scripts/validate_aperture.py \
      --logdir logs/mawgm_f64_2b/trial_1 \
      --candidates "4:5 36:29 77:22 46:5 59:4" \
      --broken logs/mawgm_f64_2b/trial_1/broken.json
"""

import argparse
import json
import numpy as np
import sys
sys.path.insert(0, 'scripts')
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites

FLOOR = 1e-7


def check_sign_flip(db, dp, bond_idx, g, adj, site_idx, thr=1e-3, thr_L=0.1, T_max=3):
    """
    Check if bond shows sign flip (BLUE at one endpoint, RED at other)
    within T_max cycles.

    Returns (confirmed, best_cycle, L_a, L_b) where:
    - confirmed: True if sign flip detected
    - best_cycle: cycle with strongest sign flip
    - L_a, L_b: L values at endpoints at best cycle
    """
    a, b = get_bond_sites(g['edges'][bond_idx])
    ai = site_idx[a]; bi = site_idx[b]

    best_contrast = 0.0
    best_cycle = -1
    best_La = 0.0; best_Lb = 0.0

    nc = min(db.shape[0], dp.shape[0], T_max)
    for t in range(nc):
        nb_a = db[t, ai]; np_a = dp[t, ai]
        nb_b = db[t, bi]; np_b = dp[t, bi]

        La = -np.log(max(np_a, FLOOR) / nb_a) if nb_a > thr else 0.0
        Lb = -np.log(max(np_b, FLOOR) / nb_b) if nb_b > thr else 0.0

        # Sign flip: one RED (+), one BLUE (-)
        if (La > thr_L and Lb < -thr_L) or (La < -thr_L and Lb > thr_L):
            contrast = abs(La) + abs(Lb)
            if contrast > best_contrast:
                best_contrast = contrast
                best_cycle = t + 1
                best_La = La; best_Lb = Lb

    confirmed = best_cycle > 0
    return confirmed, best_cycle, best_La, best_Lb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',     required=True)
    ap.add_argument('--candidates', required=True,
                    help='Space-separated "bond:center" pairs')
    ap.add_argument('--broken',     default=None)
    ap.add_argument('--thr',        type=float, default=1e-3)
    ap.add_argument('--thr_L',      type=float, default=0.1)
    ap.add_argument('--T_max',      type=int,   default=3)
    args = ap.parse_args()

    g = load_graph('data/graphs/rect_7x7.json')
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}

    true_bonds = set()
    if args.broken:
        br = json.load(open(args.broken))
        true_bonds = set(br.get('bonds', [br.get('bond', -1)]))

    pairs = [bc.split(':') for bc in args.candidates.split()]
    confirmed = []

    print(f"Aperture validation: {args.logdir}")
    print(f"True bonds: {sorted(true_bonds)}")
    print()
    print(f"{'Bond':>5}  {'sites':>8}  {'center':>7}  {'cycle':>6}  {'L(a)':>8}  {'L(b)':>8}  {'Result'}")
    print("-"*70)

    for bond_str, center_str in pairs:
        bi = int(bond_str); center = int(center_str)
        a, b = get_bond_sites(g['edges'][bi])
        true_tag = ' ← TRUE' if bi in true_bonds else ''

        base_log = f'{args.logdir}/apt_b{bi}_c{center}_base.log'
        pert_log = f'{args.logdir}/apt_b{bi}_c{center}_pert.log'

        try:
            _, _, db = parse_pauli_log(base_log, 'NL')
            _, _, dp = parse_pauli_log(pert_log, 'NL')
        except Exception as e:
            print(f"  {bi:>3}  {a}↔{b}  c{center:>5}   ERROR: {e}")
            continue

        ok, cyc, La, Lb = check_sign_flip(
            db, dp, bi, g, adj, site_idx, args.thr, args.thr_L, args.T_max)

        tag = '✓ CONFIRMED' if ok else '✗ REJECTED'
        if ok: confirmed.append(bi)

        cyc_str = str(cyc) if cyc > 0 else '-'
        print(f"  {bi:>3}  {a:>3}↔{b:>3}  c{center:>5}  {cyc_str:>6}  "
              f"{La:>+8.3f}  {Lb:>+8.3f}  {tag}{true_tag}")

    print(f"\nConfirmed broken: {confirmed}")
    if true_bonds:
        found = set(confirmed) & true_bonds
        both = true_bonds.issubset(set(confirmed))
        tag = '✓ BOTH' if both else ('~ PARTIAL' if found else '✗ NONE')
        print(f"True bonds found: {sorted(found)} / {sorted(true_bonds)}")
        print(f"Result: {tag}")


if __name__ == '__main__':
    main()
