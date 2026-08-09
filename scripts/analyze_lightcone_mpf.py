"""
analyze_lightcone_mpf.py — Test MPF accuracy at the lightcone frontier

Hypothesis: MPF extrapolation is accurate at frontier sites (where the operator
just arrived, chi not yet saturated) and less accurate in the bulk.

For each cycle we partition sites into:
  - BULK:     w_nl > threshold (operator well established, chi may be saturated)
  - FRONTIER: 0 < w_nl < threshold (operator just arrived, chi still growing)
  - DARK:     w_nl ≈ 0 (operator not arrived yet, chi=1 exact)

Then compute MPF error separately for each region.

Usage:
  python3 analyze_lightcone_mpf.py \\
      --nl  logs/mpf_experiments/mpf_co4_ct2_np25_nl400_nn150_s10_spatial_nl.dat \\
      --n10 logs/mpf_experiments/mpf_co4_ct2_np25_nl400_nn150_s10_spatial_n10.dat \\
      --n14 logs/mpf_experiments/mpf_co4_ct2_np25_nl400_nn150_s10_spatial_n14.dat \\
      --n18 logs/mpf_experiments/mpf_co4_ct2_np25_nl400_nn150_s10_spatial_n18.dat \\
      --threshold 0.001
"""

import sys
import os
import re
import argparse
import numpy as np

MPF_COEFFS = [7.875, -11.250, 4.375]

# ── Parse spatial data ────────────────────────────────────────────────────────

def parse_spatial(path):
    """Returns dict: {cycle: array[n_sites]}"""
    cycles = {}
    current_cycle = None
    current = {}
    n_sites = None

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('# heavy_hex') or line.startswith('# noisy'):
                m = re.search(r'n_sites=(\d+)', line)
                if m: n_sites = int(m.group(1))
            elif line.startswith('# cycle'):
                if current_cycle is not None and current:
                    ns = n_sites or max(current.keys())+1
                    cycles[current_cycle] = np.array([current.get(i,0.) for i in range(ns)])
                parts = line.split()
                current_cycle = int(parts[2])
                current = {}
            elif line.startswith('#'):
                continue
            elif current_cycle is not None:
                parts = line.split()
                if len(parts) >= 2:
                    current[int(parts[0])] = float(parts[1])

    if current_cycle is not None and current:
        ns = n_sites or max(current.keys())+1
        cycles[current_cycle] = np.array([current.get(i,0.) for i in range(ns)])

    return cycles

# ── Frontier analysis ─────────────────────────────────────────────────────────

def analyze_frontier(spatial_nl, spatial_n, threshold=0.001):
    """
    For each cycle, compute MPF error separately for:
      - BULK:     w_nl > threshold
      - FRONTIER: 0 < w_nl < threshold
      - DARK:     w_nl ≈ 0

    Also compute: predicted next-cycle frontier from current MPF extrapolation.
    """
    gammas = [1.0, 1.4, 1.8]
    common = sorted(set(spatial_nl.keys()) &
                    set.intersection(*[set(d.keys()) for d in spatial_n]))

    print(f"\n{'═'*80}")
    print(f"  LIGHTCONE FRONTIER MPF ANALYSIS  (threshold={threshold:.0e})")
    print(f"{'═'*80}")
    print(f"\n{'Cy':>3}  {'n_bulk':>7} {'n_front':>8} {'n_dark':>7}  "
          f"{'err_bulk':>9} {'err_front':>10} {'err_dark':>9}  "
          f"{'front_better?':>14}")
    print('-'*80)

    for cycle in common:
        w_nl = spatial_nl[cycle]
        w_n  = [d[cycle] for d in spatial_n]
        w_extrap = sum(c*w for c,w in zip(MPF_COEFFS, w_n))

        err = np.abs(w_nl - w_extrap)

        # Partition sites
        bulk_mask     = w_nl > threshold
        frontier_mask = (w_nl > 1e-6) & (w_nl <= threshold)
        dark_mask     = w_nl <= 1e-6

        n_bulk  = bulk_mask.sum()
        n_front = frontier_mask.sum()
        n_dark  = dark_mask.sum()

        err_bulk  = err[bulk_mask].mean()  if n_bulk  > 0 else float('nan')
        err_front = err[frontier_mask].mean() if n_front > 0 else float('nan')
        err_dark  = err[dark_mask].mean()  if n_dark  > 0 else float('nan')

        front_better = '✓ yes' if (n_front > 0 and n_bulk > 0 and
                                    err_front < err_bulk) else '✗ no'

        print(f"{cycle:>3}  {n_bulk:>7} {n_front:>8} {n_dark:>7}  "
              f"{err_bulk:>9.2e} {err_front:>10.2e} {err_dark:>9.2e}  "
              f"{front_better:>14}")

    print(f"{'═'*80}")
    print(f"\nKey: BULK=established sites, FRONTIER=just arrived, DARK=not yet reached")
    print(f"front_better: is frontier MPF error < bulk MPF error?")

    # ── Prediction test ───────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"  PREDICTION TEST: can MPF predict next-cycle frontier?")
    print(f"  Compare MPF_extrap(cycle t) vs w_nl(cycle t+1) at frontier sites")
    print(f"{'═'*80}")
    print(f"\n{'Cy':>3}  {'n_pred_sites':>13}  {'mean_pred_err':>14}  {'mean_baseline_err':>18}")
    print('-'*55)

    cycles_list = sorted(common)
    for i, cycle in enumerate(cycles_list[:-1]):
        next_cycle = cycles_list[i+1]
        if next_cycle != cycle+1: continue

        w_nl_now  = spatial_nl[cycle]
        w_nl_next = spatial_nl[next_cycle]
        w_n_now   = [d[cycle] for d in spatial_n]
        w_extrap  = sum(c*w for c,w in zip(MPF_COEFFS, w_n_now))

        # Prediction sites: currently frontier (just arrived now)
        pred_mask = (w_nl_now > 1e-6) & (w_nl_now <= threshold)
        n_pred = pred_mask.sum()
        if n_pred == 0: continue

        # MPF extrapolation at these sites vs next cycle noiseless
        pred_err     = np.abs(w_nl_next[pred_mask] - w_extrap[pred_mask]).mean()
        # Baseline: just use current noiseless value as predictor
        baseline_err = np.abs(w_nl_next[pred_mask] - w_nl_now[pred_mask]).mean()

        print(f"{cycle:>3}→{next_cycle:<3}  {n_pred:>13}  {pred_err:>14.2e}  {baseline_err:>18.2e}")

    print(f"{'═'*80}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Lightcone frontier MPF analysis")
    parser.add_argument('--nl',  required=True, help='Noiseless spatial data')
    parser.add_argument('--n10', required=True, help='Noisy gamma=1.0 spatial data')
    parser.add_argument('--n14', required=True, help='Noisy gamma=1.4 spatial data')
    parser.add_argument('--n18', required=True, help='Noisy gamma=1.8 spatial data')
    parser.add_argument('--threshold', type=float, default=0.001,
                        help='Frontier threshold on w_nl (default: 0.001)')
    args = parser.parse_args()

    print(f"Loading spatial data...")
    spatial_nl = parse_spatial(args.nl)
    spatial_n  = [parse_spatial(args.n10),
                  parse_spatial(args.n14),
                  parse_spatial(args.n18)]
    print(f"  Noiseless: {len(spatial_nl)} cycles, {len(next(iter(spatial_nl.values())))} sites")

    analyze_frontier(spatial_nl, spatial_n, threshold=args.threshold)

if __name__ == '__main__':
    main()
