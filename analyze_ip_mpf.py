#!/usr/bin/env python3
"""
analyze_ip_mpf.py — Compare inner product MPF vs polynomial MPF vs poor man's MPF

Reads:
  --ip   : output of main_heavyhex_gamma_mpf_ip  (C_nl, C_n, A, B per step)
  --poly : output of main_heavyhex_gamma_mpf      (C_nl, C_n, C_mpf_poly per step)

Computes:
  - C_mpf_ip   : Lagrange solve on A/B from inner products
  - C_mpf_poly : polynomial extrapolation (from --poly log)
  - C_mpf_pm   : poor man's MPF (A/B built from C(t) time correlations, cumulative)

Prints per-step comparison table and errors vs C_nl.

Usage:
  python3 analyze_ip_mpf.py --ip ip.log --poly poly.log
  python3 analyze_ip_mpf.py --ip ip.log   # poly from ip log itself
"""

import argparse
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GAMMAS = np.array([1.0, 2.0, 3.0])
N = 3

# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_ip_log(path):
    """Parse main_heavyhex_gamma_mpf_ip output."""
    steps = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("STEP"): continue
            m = re.match(
                r"STEP (\d+)\s+"
                r"C_nl ([+-]?\d+\.\d+)\s+"
                r"C_n ([+-]?\d+\.\d+) ([+-]?\d+\.\d+) ([+-]?\d+\.\d+)\s+"
                r"IP_A ([\d.+-]+) ([\d.+-]+) ([\d.+-]+) ([\d.+-]+) ([\d.+-]+) ([\d.+-]+)\s+"
                r"IP_B ([\d.+-]+) ([\d.+-]+) ([\d.+-]+)",
                line)
            if not m: continue
            s = int(m.group(1))
            C_nl = float(m.group(2))
            C_n  = np.array([float(m.group(3)), float(m.group(4)), float(m.group(5))])
            a00,a01,a02,a11,a12,a22 = [float(m.group(i)) for i in range(6,12)]
            A = np.array([[a00,a01,a02],[a01,a11,a12],[a02,a12,a22]])
            B = np.array([float(m.group(12)), float(m.group(13)), float(m.group(14))])
            steps[s] = dict(C_nl=C_nl, C_n=C_n, A=A, B=B)
    return steps

def parse_poly_log(path):
    """Parse main_heavyhex_gamma_mpf output — extract C_mpf_poly per step."""
    poly = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or not parts[0].isdigit(): continue
            if len(parts) < 12: continue
            try:
                s      = int(parts[0])
                C_mpf  = float(parts[10])   # C_mpf column
                poly[s] = C_mpf
            except (ValueError, IndexError):
                continue
    return poly

# ── MPF solvers ───────────────────────────────────────────────────────────────

def solve_lagrange(A, B, C_n):
    """Solve [A,1;1^T,0][x;λ]=[B;1] → C_mpf = Σ x_i C_n[i]."""
    M = np.zeros((N+1, N+1))
    M[:N,:N] = A
    M[:N, N] = 1.0
    M[N, :N] = 1.0
    rhs = np.zeros(N+1)
    rhs[:N] = B
    rhs[N]  = 1.0
    try:
        sol = np.linalg.solve(M, rhs)
        x   = sol[:N]
        return float(x @ C_n), x
    except np.linalg.LinAlgError:
        return float('nan'), np.full(N, float('nan'))

def poor_mans_mpf(steps_data):
    """Build A/B from cumulative C(t) correlations, solve per step."""
    results = {}
    A_cum = np.zeros((N, N))
    B_cum = np.zeros(N)
    for s in sorted(steps_data.keys()):
        d    = steps_data[s]
        C_nl = d['C_nl']
        C_n  = d['C_n']
        # Accumulate
        for i in range(N):
            B_cum[i]   += C_n[i] * C_nl
            for j in range(N):
                A_cum[i,j] += C_n[i] * C_n[j]
        C_mpf_pm, x = solve_lagrange(A_cum.copy(), B_cum.copy(), C_n)
        results[s] = C_mpf_pm
    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ip',   required=True, help='IP log from main_heavyhex_gamma_mpf_ip')
    ap.add_argument('--poly', default=None,  help='Poly log from main_heavyhex_gamma_mpf')
    args = ap.parse_args()

    ip_data  = parse_ip_log(args.ip)
    poly_data = parse_poly_log(args.poly) if args.poly else {}
    pm_data  = poor_mans_mpf(ip_data)

    print(f"\n{'Step':>5}  {'C_nl':>10}  {'C_mpf_poly':>12}  {'C_mpf_pm':>10}  "
          f"{'C_mpf_ip':>10}  {'|Δpoly|':>9}  {'|Δpm|':>9}  {'|Δip|':>9}")
    print("-" * 90)

    for s in sorted(ip_data.keys()):
        d    = ip_data[s]
        C_nl = d['C_nl']
        C_n  = d['C_n']
        A    = d['A']
        B    = d['B']

        C_mpf_ip, x_ip = solve_lagrange(A, B, C_n)
        C_mpf_pm  = pm_data.get(s, float('nan'))
        C_mpf_poly = poly_data.get(s, float('nan'))

        d_poly = abs(C_mpf_poly - C_nl) if not np.isnan(C_mpf_poly) else float('nan')
        d_pm   = abs(C_mpf_pm   - C_nl)
        d_ip   = abs(C_mpf_ip   - C_nl) if not np.isnan(C_mpf_ip)   else float('nan')

        print(f"{s:>5}  {C_nl:+10.6f}  {C_mpf_poly:+12.6f}  {C_mpf_pm:+10.6f}  "
              f"{C_mpf_ip:+10.6f}  {d_poly:9.2e}  {d_pm:9.2e}  {d_ip:9.2e}")
        print(f"       x_ip=[{x_ip[0]:+.3f} {x_ip[1]:+.3f} {x_ip[2]:+.3f}] sum={x_ip.sum():.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    steps  = sorted(ip_data.keys())
    C_nl_v = [ip_data[s]['C_nl'] for s in steps]
    C_pm_v = [pm_data.get(s, float('nan')) for s in steps]

    C_ip_v, d_pm_v, d_ip_v = [], [], []
    for s in steps:
        d = ip_data[s]; C_n = d['C_n']; A = d['A']; B = d['B']
        C_mpf_ip, _ = solve_lagrange(A, B, C_n)
        C_mpf_pm    = pm_data.get(s, float('nan'))
        C_ip_v.append(C_mpf_ip)
        d_pm_v.append(abs(C_mpf_pm - d['C_nl']))
        d_ip_v.append(abs(C_mpf_ip - d['C_nl']) if not np.isnan(C_mpf_ip) else float('nan'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: C(t) curves
    ax1.plot(steps, C_nl_v, 'k-o', label='$C_{nl}$', linewidth=2)
    ax1.plot(steps, C_pm_v, 'r-s', label="Poor man's MPF", linewidth=1.5)
    ax1.plot(steps, C_ip_v, 'b-^', label='Inner product MPF', linewidth=1.5)
    ax1.set_xlabel('Cycle'); ax1.set_ylabel('C(t)')
    ax1.set_title('MPF extrapolation'); ax1.legend(); ax1.grid(True, alpha=0.3)

    # Right: error (log scale)
    ax2.semilogy(steps, d_pm_v, 'r-s', label="|Δ| poor man's", linewidth=1.5)
    ax2.semilogy(steps, d_ip_v, 'b-^', label='|Δ| inner product', linewidth=1.5)
    ax2.set_xlabel('Cycle'); ax2.set_ylabel('|C_mpf - C_nl|')
    ax2.set_title('MPF error (log scale)'); ax2.legend(); ax2.grid(True, alpha=0.3)

    outfile = args.ip.replace('.log', '_mpf_comparison.png').replace('/tmp/', './')
    plt.tight_layout(); plt.savefig(outfile, dpi=150)
    print(f"\nPlot saved: {outfile}")

if __name__ == '__main__':
    main()
