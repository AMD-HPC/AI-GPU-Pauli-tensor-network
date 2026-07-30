#!/usr/bin/env python3
"""
plot_ip_sweep.py — Multi-chi, multi-cutoff MPF comparison figure.

For each chi value: shows |Δ| = |C_mpf - C_nl| per cycle for:
  - Poor man's MPF  (solid lines, one per cutoff)
  - Inner product MPF (dashed lines, one per cutoff)

Usage:
  python3 plot_ip_sweep.py --logdir logs/ip_sweep_YYYYMMDD_HHMMSS [--out fig.png]
"""

import argparse
import glob
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

N = 3  # gamma values

def parse_log(path):
    steps = {}
    with open(path) as f:
        for line in f:
            m = re.match(
                r"STEP (\d+)\s+"
                r"C_nl ([+-][\d.e+-]+)\s+"
                r"C_n ([+-][\d.e+-]+) ([+-][\d.e+-]+) ([+-][\d.e+-]+)\s+"
                r"IP_A ([\S]+) ([\S]+) ([\S]+) ([\S]+) ([\S]+) ([\S]+)\s+"
                r"IP_B ([\S]+) ([\S]+) ([\S]+)", line)
            if not m: continue
            s = int(m.group(1))
            C_nl = float(m.group(2))
            C_n  = np.array([float(m.group(3)), float(m.group(4)), float(m.group(5))])
            def safe(x):
                try: v=float(x); return v if abs(v)<1e6 else float('nan')
                except: return float('nan')
            A = np.array([[safe(m.group(6)),  safe(m.group(7)),  safe(m.group(8))],
                          [safe(m.group(7)),  safe(m.group(9)),  safe(m.group(10))],
                          [safe(m.group(8)),  safe(m.group(10)), safe(m.group(11))]])
            B = np.array([safe(m.group(12)), safe(m.group(13)), safe(m.group(14))])
            steps[s] = dict(C_nl=C_nl, C_n=C_n, A=A, B=B)
    return steps

def solve_lagrange(A, B, C_n):
    M = np.zeros((N+1,N+1)); M[:N,:N]=A; M[:N,N]=1; M[N,:N]=1
    rhs = np.zeros(N+1); rhs[:N]=B; rhs[N]=1
    try:
        sol = np.linalg.solve(M, rhs)
        v = float(sol[:N]@C_n)
        return v if abs(v)<10 else float('nan')
    except: return float('nan')

def poor_mans(steps):
    A_c=np.zeros((N,N)); B_c=np.zeros(N); res={}
    for s in sorted(steps):
        d=steps[s]; C_nl=d['C_nl']; C_n=d['C_n']
        for i in range(N):
            B_c[i]+=C_n[i]*C_nl
            for j in range(N): A_c[i,j]+=C_n[i]*C_n[j]
        res[s]=solve_lagrange(A_c.copy(),B_c.copy(),C_n)
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    # Discover chi and cutoff values from filenames
    logs = glob.glob(os.path.join(args.logdir, 'chi*_cut*.log'))
    chi_set = sorted(set(int(re.search(r'chi(\d+)',f).group(1)) for f in logs))
    cut_set = sorted(set(re.search(r'cut([\d.e+-]+)',f).group(1) for f in logs),
                     key=lambda x: float(x))

    n_chi = len(chi_set)
    fig, axes = plt.subplots(1, n_chi, figsize=(4*n_chi, 5), sharey=True)
    if n_chi==1: axes=[axes]

    # Color per cutoff
    colors = cm.viridis(np.linspace(0.1, 0.9, len(cut_set)))

    for ax, chi in zip(axes, chi_set):
        ax.set_title(f'$\\chi={chi}$')
        ax.set_xlabel('Cycle')
        if ax == axes[0]: ax.set_ylabel(r'$|C_{\rm mpf} - C_{\rm nl}|$')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(1e-5, 1.0)

        for col, cut in zip(colors, cut_set):
            logfile = os.path.join(args.logdir, f'chi{chi}_cut{cut}.log')
            if not os.path.exists(logfile): continue
            steps = parse_log(logfile)
            if not steps: continue

            step_list = sorted(steps.keys())
            pm_data = poor_mans(steps)

            d_pm_v, d_ip_v = [], []
            for s in step_list:
                d = steps[s]
                C_pm = pm_data.get(s, float('nan'))
                C_ip = solve_lagrange(d['A'], d['B'], d['C_n'])
                d_pm_v.append(abs(C_pm - d['C_nl']) if not np.isnan(C_pm) else float('nan'))
                d_ip_v.append(abs(C_ip - d['C_nl']) if not np.isnan(C_ip) else float('nan'))

            label = f'cut={cut}'
            ax.plot(step_list, d_pm_v, '-',  color=col, lw=1.5,
                    label=f"PM {label}" if chi==chi_set[0] else None)
            ax.plot(step_list, d_ip_v, '--', color=col, lw=1.5,
                    label=f"IP {label}" if chi==chi_set[0] else None)

    # Legend on first panel
    axes[0].legend(fontsize=7, ncol=2, loc='upper right',
                   title='solid=poor man, dashed=inner prod')

    plt.suptitle('MPF error vs cycle: poor man (solid) vs inner product (dashed)',
                 fontsize=11)
    plt.tight_layout()
    outfile = args.out or os.path.join(args.logdir, 'ip_sweep_comparison.png')
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")

if __name__ == '__main__':
    main()
