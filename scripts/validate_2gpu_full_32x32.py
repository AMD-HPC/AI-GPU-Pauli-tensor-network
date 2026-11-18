#!/usr/bin/env python3
"""
validate_2gpu_full_32x32.py — Compare 2-GPU 32x32 output to single-GPU reference.

Usage:
  python3 scripts/validate_2gpu_full_32x32.py --chi 10 --steps 3 --center 497
"""

import argparse
import subprocess
import numpy as np

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

def parse_pauli(output, step):
    for line in output.splitlines():
        if line.startswith(f'PAULI_NL {step} '):
            return np.array([float(x) for x in line.split()[2:]])
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chi',    type=int,   default=10)
    ap.add_argument('--steps',  type=int,   default=3)
    ap.add_argument('--cutoff', default='1e-3')
    ap.add_argument('--center', type=int,   default=497)
    args = ap.parse_args()

    r = (args.center - 1) // 32
    if r >= 16:
        print(f"WARNING: center={args.center} is in GPU1 domain (row {r} >= 16).")

    cmd_ref  = f"./main_pauli_sweep_rect32x32 {args.chi} {args.steps} {args.cutoff} --center={args.center} --state=NL"
    cmd_2gpu = f"./main_pauli_sweep_2gpu_rect32x32 {args.chi} {args.steps} {args.cutoff} --center={args.center}"

    print(f"Reference (32×32 single-GPU): {cmd_ref}")
    print(f"2-GPU split:                  {cmd_2gpu}")
    print()

    out_ref  = run(cmd_ref)
    out_2gpu = run(cmd_2gpu)

    all_ok = True
    for step in range(1, args.steps + 1):
        ref  = parse_pauli(out_ref,  step)
        gpu2 = parse_pauli(out_2gpu, step)

        if ref is None:
            print(f"Step {step}: ERROR — no PAULI_NL in single-GPU output"); all_ok=False; continue
        if gpu2 is None:
            print(f"Step {step}: ERROR — no PAULI_NL in 2-GPU output"); all_ok=False; continue

        n = min(len(ref), len(gpu2))
        diff = np.abs(ref[:n] - gpu2[:n])
        nonzero = np.count_nonzero(ref[:n])
        status = "✓ OK" if diff.max() < 1e-4 else "✗ MISMATCH"
        print(f"Step {step}: sites ref={len(ref)} gpu2={len(gpu2)}  "
              f"max_diff={diff.max():.2e}  mean_diff={diff.mean():.2e}  "
              f"nonzero={nonzero}  {status}")
        if diff.max() >= 1e-4:
            all_ok = False

    print()
    print("PASS" if all_ok else "FAIL")

if __name__ == '__main__':
    main()
