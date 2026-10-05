#!/usr/bin/env python3
"""
test_perturbation.py — Verify that --perturb-bond actually changes n(v,t).

Runs baseline and perturbed simulations from a single center,
then shows which sites differ and whether the broken bond endpoints
are correctly identified.

Usage:
  python3 scripts/test_perturbation.py \
      --bond 12 --J 2.0 --center 25 \
      --graph data/graphs/rect_7x7.json \
      --binary ./main_pauli_sweep_rect7x7

Output:
  - Which sites show Δn(v,t) > 0 at each cycle
  - Whether sites 15,16 (bond 12 endpoints) show the signal
  - PTM_elements_changed from the log
"""

import argparse
import subprocess
import tempfile
import os
import re
import numpy as np
import sys
sys.path.insert(0, 'scripts')

def run_sim(binary, chi, steps, cutoff, center, perturb_bond=None, J=None,
            logfile=None, gpu=0):
    cmd = [f'ROCR_VISIBLE_DEVICES={gpu}', binary,
           str(chi), str(steps), str(cutoff),
           f'--center={center}', '--state=NL']
    if perturb_bond is not None:
        cmd.append(f'--perturb-bond={perturb_bond},{J:.4f}')
    full_cmd = ' '.join(cmd) + f' > {logfile} 2>&1'
    print(f"  Running: {full_cmd}")
    ret = os.system(full_cmd)
    if ret != 0:
        print(f"  ERROR: simulation failed (exit {ret})")
        return False
    return True

def parse_pauli(logfile, state='NL'):
    key = f'PAULI_{state}'
    steps = {}
    with open(logfile) as f:
        for line in f:
            if not line.startswith(key): continue
            parts = line.split()
            t = int(parts[1])
            steps[t] = np.array([float(x) for x in parts[2:]])
    if not steps: return None
    nc = max(steps.keys())
    ns = len(next(iter(steps.values())))
    data = np.zeros((nc, ns))
    for t, v in steps.items():
        data[t-1] = v
    return data

def get_bond_endpoints(graph_path, bond_idx):
    import json
    with open(graph_path) as f:
        g = json.load(f)
    e = g['edges'][bond_idx]
    if isinstance(e, dict):
        sites = e['edge']
        a = sites[0][0] if isinstance(sites[0], list) else sites[0]
        b = sites[1][0] if isinstance(sites[1], list) else sites[1]
    else:
        a, b = e[0], e[1]
    return a, b  # 1-indexed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bond',   type=int,   default=12,    help='Bond index to perturb')
    ap.add_argument('--J',      type=float, default=2.0,   help='New coupling J (default: 2.0 = double)')
    ap.add_argument('--center', type=int,   default=25,    help='Center site')
    ap.add_argument('--chi',    type=int,   default=50)
    ap.add_argument('--steps',  type=int,   default=5)
    ap.add_argument('--cutoff', type=float, default=1e-3)
    ap.add_argument('--gpu',    type=int,   default=0)
    ap.add_argument('--graph',  default='data/graphs/rect_7x7.json')
    ap.add_argument('--binary', default='./main_pauli_sweep_rect7x7')
    ap.add_argument('--thr',    type=float, default=1e-5,  help='Diff threshold to report')
    args = ap.parse_args()

    # Bond endpoints from graph
    site_a, site_b = get_bond_endpoints(args.graph, args.bond)
    print(f"\nTest: perturb bond {args.bond} ({site_a}↔{site_b}) with J={args.J}")
    print(f"  Center: {args.center}  Chi: {args.chi}  Steps: {args.steps}")
    print(f"  Binary: {args.binary}")

    # Run baseline
    base_log = '/tmp/test_baseline.log'
    pert_log = '/tmp/test_perturbed.log'

    print(f"\n[1/2] Baseline simulation...")
    if not run_sim(args.binary, args.chi, args.steps, args.cutoff,
                   args.center, logfile=base_log, gpu=args.gpu):
        return

    print(f"\n[2/2] Perturbed simulation (bond={args.bond}, J={args.J})...")
    if not run_sim(args.binary, args.chi, args.steps, args.cutoff,
                   args.center, perturb_bond=args.bond, J=args.J,
                   logfile=pert_log, gpu=args.gpu):
        return

    # Check PTM elements changed
    with open(pert_log) as f:
        for line in f:
            if 'PTM_elements_changed' in line:
                print(f"\n  {line.strip()}")

    # Parse results
    base = parse_pauli(base_log)
    pert = parse_pauli(pert_log)
    if base is None or pert is None:
        print("ERROR: could not parse PAULI_NL lines")
        return

    nc = min(base.shape[0], pert.shape[0])
    ns = base.shape[1]

    print(f"\n{'='*60}")
    print(f"Sites 1-indexed {site_a} and {site_b} (bond {args.bond} endpoints):")
    for t in range(nc):
        va = base[t, site_a-1]; vap = pert[t, site_a-1]
        vb = base[t, site_b-1]; vbp = pert[t, site_b-1]
        print(f"  Cycle {t+1}: site {site_a}: base={va:.7f} pert={vap:.7f} "
              f"Δ={vap-va:+.2e}  |  "
              f"site {site_b}: base={vb:.7f} pert={vbp:.7f} Δ={vbp-vb:+.2e}")

    print(f"\n{'='*60}")
    print(f"All sites with |Δn| > {args.thr} at any cycle:")
    found = False
    for v in range(ns):
        max_diff = max(abs(pert[t,v] - base[t,v]) for t in range(nc))
        if max_diff > args.thr:
            found = True
            first_cycle = next(t+1 for t in range(nc) if abs(pert[t,v]-base[t,v]) > args.thr)
            print(f"  Site {v+1:3d} (0-idx {v:2d}): max|Δ|={max_diff:.2e}  "
                  f"first at cycle {first_cycle}"
                  f"{'  ← BOND ENDPOINT' if v+1 in (site_a, site_b) else ''}")

    if not found:
        print(f"  None — perturbation has no detectable effect!")
        print(f"\n  Check: PTM_elements_changed should be > 0")
        print(f"  If 0: the trig matching failed — try larger J or check base PTM values")

if __name__ == '__main__':
    main()
