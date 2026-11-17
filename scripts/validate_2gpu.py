#!/usr/bin/env python3
"""
validate_2gpu.py — Compare 2-GPU split output to single-GPU reference.

Runs both binaries with the same center and compares PAULI_NL values.

Usage:
  python3 scripts/validate_2gpu.py --chi 40 --steps 3 --center 257
"""

import argparse
import subprocess
import numpy as np
import sys

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

def parse_pauli(output, step):
    for line in output.splitlines():
        if line.startswith(f'PAULI_NL {step} '):
            vals = [float(x) for x in line.split()[2:]]
            return np.array(vals)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chi',    type=int,   default=40)
    ap.add_argument('--steps',  type=int,   default=3)
    ap.add_argument('--cutoff', default='1e-3')
    ap.add_argument('--center', type=int,   default=257)
    ap.add_argument('--perturb-bond', type=int, default=-1)
    ap.add_argument('--J-pert',       type=float, default=0.8)
    args = ap.parse_args()

    # Center must be in GPU0's domain (rows 0-7, sites 1-256) for valid comparison.
    # Center on boundary row 8 (sites 257-288) has different topology in 9x32 vs 2-GPU.
    r = (args.center - 1) // 32
    if r >= 8:
        print(f"WARNING: center={args.center} is in GPU1 domain (row {r} >= 8).")
        print("         Use a center in rows 0-7 for valid single-GPU comparison.")
        print()

    print(f"Validating 2-GPU vs single-GPU: chi={args.chi} steps={args.steps} center={args.center}")
    print()

    perturb_flag = f"--perturb-bond={args.perturb_bond},{args.J_pert:.3f}" if args.perturb_bond >= 0 else ""

    # Single-GPU reference (9x32 — same grid as GPU0)
    cmd_ref  = f"./main_pauli_sweep_rect9x32 {args.chi} {args.steps} {args.cutoff} --center={args.center} --state=NL {perturb_flag}"
    # 2-GPU split
    cmd_2gpu = f"./main_pauli_sweep_2gpu_rect16x32 {args.chi} {args.steps} {args.cutoff} --center={args.center} {perturb_flag}"

    print(f"Running: {cmd_ref}")
    out_ref  = run(cmd_ref)
    print(f"Running: {cmd_2gpu}")
    out_2gpu = run(cmd_2gpu)

    all_ok = True
    for step in range(1, args.steps + 1):
        ref  = parse_pauli(out_ref,  step)
        gpu2 = parse_pauli(out_2gpu, step)

        if ref is None:
            print(f"Step {step}: ERROR — no PAULI_NL in single-GPU output")
            all_ok = False
            continue
        if gpu2 is None:
            print(f"Step {step}: ERROR — no PAULI_NL in 2-GPU output")
            all_ok = False
            continue

        # Compare GPU0 portion only (first 288 sites)
        n = min(len(ref), 288, len(gpu2))
        diff = np.abs(ref[:n] - gpu2[:n])
        max_diff  = diff.max()
        mean_diff = diff.mean()
        nonzero   = np.count_nonzero(ref[:n])

        status = "✓ OK" if max_diff < 1e-5 else "✗ MISMATCH"
        print(f"Step {step}: max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}  "
              f"nonzero_ref={nonzero}  {status}")
        if max_diff >= 1e-5:
            all_ok = False

    print()
    print("PASS" if all_ok else "FAIL")

if __name__ == '__main__':
    main()
