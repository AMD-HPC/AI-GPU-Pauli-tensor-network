#!/usr/bin/env python3
"""
compare_qr_modes.py — Compare PAULI_NL fields across QR modes and precisions.

Usage:
  python3 scripts/compare_qr_modes.py \
      --hh_f32 /tmp/hh_run1.log \
      --ch_f32 /tmp/ch_run1.log \
      --hh_f64 /tmp/hh_f64_run1.log \
      --ch_f64 /tmp/ch_f64_run1.log
"""

import argparse
import re
import numpy as np

def extract_pauli(fname):
    vals = {}
    try:
        with open(fname) as f:
            for line in f:
                m = re.match(r'PAULI_NL (\d+) (.+)', line)
                if m:
                    vals[int(m.group(1))] = np.array(list(map(float, m.group(2).split())))
    except FileNotFoundError:
        print(f"  [missing: {fname}]")
    return vals

def compare(label, va_dict, vb_dict):
    steps = sorted(set(va_dict) & set(vb_dict))
    if not steps:
        print(f"  {label}: no common steps")
        return
    for step in steps:
        va = va_dict[step]; vb = vb_dict[step]
        diff = np.abs(va - vb)
        rel  = diff / np.maximum(np.abs(va), 1e-10)
        print(f"  {label}  step={step}  "
              f"max_abs={diff.max():.2e}  max_rel={rel.max():.2e}  "
              f"median_rel={np.median(rel):.2e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hh_f32', default='/tmp/hh_run1.log')
    ap.add_argument('--ch_f32', default='/tmp/ch_run1.log')
    ap.add_argument('--hh_f64', default='/tmp/hh_f64_run1.log')
    ap.add_argument('--ch_f64', default='/tmp/ch_f64_run1.log')
    ap.add_argument('--hh_f32_chi70', default='/tmp/hh_f32_chi70_run1.log')
    ap.add_argument('--ch_f32_chi70', default='/tmp/ch_f32_chi70_run1.log')
    ap.add_argument('--hh_f64_chi70', default='/tmp/hh_f64_chi70_run1.log')
    args = ap.parse_args()

    hh32 = extract_pauli(args.hh_f32)
    ch32 = extract_pauli(args.ch_f32)
    hh64 = extract_pauli(args.hh_f64)
    ch64 = extract_pauli(args.ch_f64)
    hh32_70 = extract_pauli(args.hh_f32_chi70)
    ch32_70 = extract_pauli(args.ch_f32_chi70)
    hh64_70 = extract_pauli(args.hh_f64_chi70)

    print("\n=== QR mode comparison (same precision) ===")
    compare("HH-f32 vs CH-f32 chi=50", hh32, ch32)
    compare("HH-f32 vs CH-f32 chi=70", hh32_70, ch32_70)
    compare("HH-f64 vs CH-f64 chi=50", hh64, ch64)

    print("\n=== Precision comparison (same QR mode) ===")
    compare("HH-f32 vs HH-f64 chi=50", hh32, hh64)
    compare("CH-f32 vs CH-f64 chi=50", ch32, ch64)

    print("\n=== chi=70 determinism ===")
    if hh32_70 and args.hh_f32_chi70:
        hh32_70b = extract_pauli(args.hh_f32_chi70.replace('run1','run2'))
        compare("HH-f32 chi=70 run1 vs run2", hh32_70, hh32_70b)
    if hh64_70 and args.hh_f64_chi70:
        hh64_70b = extract_pauli(args.hh_f64_chi70.replace('run1','run2'))
        compare("HH-f64 chi=70 run1 vs run2", hh64_70, hh64_70b)

    print("\n=== chi=50 vs chi=70 (HH-f32) ===")
    compare("HH-f32 chi=50 vs chi=70", hh32, hh32_70)

if __name__ == '__main__':
    main()
