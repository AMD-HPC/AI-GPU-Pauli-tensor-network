#!/usr/bin/env python3
"""
plot_ip_timing.py — Two-panel plot from main_heavyhex_gamma_mpf_2gpu log.

Panel 1 (left): Time evolution |C(t)|
  - |C_nl(t)|   black solid
  - |C_mpf_pm|  red dashed  (poor man's MPF)
  - |C_mpf_ip|  blue dashed (inner product MPF, skips invalid entries)

Panel 2 (right): Per-step timing
  - t_trotter   orange bars (GPU 0, all 4 states)
  - t_ip        blue line   (upload time = wall-clock overhead)
  - serial_total green line (sum of 9 inner product calls, serial estimate)

Usage:
  python3 plot_ip_timing.py --ip /tmp/ip_chi200_10cycles.log [--out fig.png]
"""

import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = 3  # number of gamma values

# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_log(path):
    steps = {}
    serial_times = {}  # from [IP] lines
    with open(path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # STEP line
        m = re.match(
            r"STEP (\d+)\s+"
            r"C_nl ([+-][\d.e+-]+)\s+"
            r"C_n ([+-][\d.e+-]+) ([+-][\d.e+-]+) ([+-][\d.e+-]+)\s+"
            r"IP_A ([\S]+) ([\S]+) ([\S]+) ([\S]+) ([\S]+) ([\S]+)\s+"
            r"IP_B ([\S]+) ([\S]+) ([\S]+)\s+"
            r"chi (\d+) (\d+)\s+"
            r"t_cycle ([\d.]+)\s+"
            r"t_trotter_pure ([\d.]+)\s+"
            r"t_ip_wall ([\d.nan]+)\s+"
            r"t_upload ([\d.nan]+)", line)
        if m:
            s = int(m.group(1))
            C_nl = float(m.group(2))
            C_n  = np.array([float(m.group(3)), float(m.group(4)), float(m.group(5))])
            def safe(x):
                try: v=float(x); return v if abs(v)<1e10 else float('nan')
                except: return float('nan')
            A = np.array([[safe(m.group(6)),  safe(m.group(7)),  safe(m.group(8))],
                          [safe(m.group(7)),  safe(m.group(9)),  safe(m.group(10))],
                          [safe(m.group(8)),  safe(m.group(10)), safe(m.group(11))]])
            B = np.array([safe(m.group(12)), safe(m.group(13)), safe(m.group(14))])
            chi_nl  = int(m.group(15))
            chi_n   = int(m.group(16))
            t_cycle = float(m.group(17))
            t_trot_pure = float(m.group(18))
            def to_f(x):
                try: return float(x)
                except: return float('nan')
            t_ip_wall = to_f(m.group(19))
            t_upload  = to_f(m.group(20))
            steps[s] = dict(C_nl=C_nl, C_n=C_n, A=A, B=B,
                            chi_nl=chi_nl, chi_n=chi_n,
                            t_cycle=t_cycle, t_trotter=t_trot_pure,
                            t_ip_wall=t_ip_wall, t_upload=t_upload)

        # [IP] line: serial sum for diagnostics only (not used in plot)
        # t_ip_wall is already in the STEP line directly

    return steps, serial_times

def solve_lagrange(A, B, C_n, normalize=False):
    """Solve MPF. Clip result if |C_mpf| > 3*|C_nl| (ill-conditioned Lagrange)."""
    try:
        M = np.zeros((N+1, N+1))
        M[:N,:N] = A; M[:N,N] = 1.0; M[N,:N] = 1.0
        rhs = np.zeros(N+1); rhs[:N] = B; rhs[N] = 1.0
        sol = np.linalg.solve(M, rhs)
        result = float(sol[:N] @ C_n)
        # Clip wildly ill-conditioned results
        if abs(result) > 3 * abs(C_n).max() + 1.0:
            return float('nan')
        return result
    except: return float('nan')

def poor_mans_mpf(steps):
    A_cum = np.zeros((N,N)); B_cum = np.zeros(N)
    results = {}
    for s in sorted(steps):
        d = steps[s]; C_nl=d['C_nl']; C_n=d['C_n']
        for i in range(N):
            B_cum[i] += C_n[i]*C_nl
            for j in range(N): A_cum[i,j] += C_n[i]*C_n[j]
        results[s] = solve_lagrange(A_cum.copy(), B_cum.copy(), C_n)
    return results

# ── Main ──────────────────────────────────────────────────────────────────────

def parse_dg_log(path):
    """Parse main_heavyhex_gamma_mpf_DG_2gpu log."""
    steps = {}
    with open(path) as f:
        for line in f:
            m = re.match(
                r"STEP (\d+)\s+"
                r"C_nl ([+-][\d.e+-]+)\s+"
                r"C_n ([+-][\d.e+-]+) ([+-][\d.e+-]+) ([+-][\d.e+-]+)\s+"
                r"IP_A ([\S]+) ([\S]+) ([\S]+) ([\S]+) ([\S]+) ([\S]+)\s+"
                r"IP_B ([\S]+) ([\S]+) ([\S]+)\s+"
                r"chi (\d+) (\d+)\s+"
                r"t_cycle ([\d.]+)\s+t_trotter ([\d.]+)\s+"
                r"t_dg_precompute ([\d.nan]+)\s+t_dg_pairs ([\d.nan]+)", line)
            if not m: continue
            s = int(m.group(1))
            def safe(x):
                try: v=float(x); return v if abs(v)<1e10 else float('nan')
                except: return float('nan')
            def to_f(x):
                try: return float(x)
                except: return float('nan')
            C_nl = float(m.group(2))
            C_n  = np.array([float(m.group(3)), float(m.group(4)), float(m.group(5))])
            A = np.array([[safe(m.group(6)),  safe(m.group(7)),  safe(m.group(8))],
                          [safe(m.group(7)),  safe(m.group(9)),  safe(m.group(10))],
                          [safe(m.group(8)),  safe(m.group(10)), safe(m.group(11))]])
            B = np.array([safe(m.group(12)), safe(m.group(13)), safe(m.group(14))])
            t_pre   = to_f(m.group(19))
            t_pairs = to_f(m.group(20))
            steps[s] = dict(C_nl=C_nl, C_n=C_n, A=A, B=B,
                            t_cycle=to_f(m.group(17)), t_trotter=to_f(m.group(18)),
                            t_dg_ip=t_pre+t_pairs if not (np.isnan(t_pre) or np.isnan(t_pairs)) else float('nan'))
    return steps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ip',  required=True, help='QR log from main_heavyhex_gamma_mpf_2gpu')
    ap.add_argument('--dg',  default=None,  help='DG log from main_heavyhex_gamma_mpf_DG_2gpu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    steps, serial_times = parse_log(args.ip)
    pm_data = poor_mans_mpf(steps)

    dg_steps = parse_dg_log(args.dg) if args.dg else {}

    step_list = sorted(steps.keys())
    C_nl_v    = [abs(steps[s]['C_nl']) for s in step_list]
    C_pm_v    = [abs(pm_data.get(s, float('nan'))) for s in step_list]
    C_ip_v    = []
    for s in step_list:
        d = steps[s]
        v = solve_lagrange(d['A'], d['B'], d['C_n'])
        C_ip_v.append(abs(v) if not np.isnan(v) and abs(v)<10 else float('nan'))

    t_cycle_v  = [steps[s]['t_cycle'] for s in step_list]
    t_trot_v   = [steps[s]['t_trotter'] for s in step_list]
    t_ip_wall_v= [steps[s]['t_ip_wall'] for s in step_list]
    t_upload_v = [steps[s]['t_upload'] for s in step_list]
    chi_v      = [steps[s]['chi_nl'] for s in step_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel 1: |C(t)| ──────────────────────────────────────────────────────
    # DG accuracy overlay
    dg_ip_v = []
    for s in step_list:
        if s in dg_steps:
            d = dg_steps[s]
            v = solve_lagrange(d['A'], d['B'], d['C_n'])
            dg_ip_v.append(abs(v) if not np.isnan(v) and abs(v)<10 else float('nan'))
        else:
            dg_ip_v.append(float('nan'))

    ax1.plot(step_list, C_nl_v, 'k-o', lw=2, label=r'$|C_{\rm nl}(t)|$')
    ax1.plot(step_list, C_pm_v, 'r--s', lw=1.5, label="Poor man's MPF")
    ax1.plot(step_list, C_ip_v, 'b--^', lw=1.5, label='IP MPF (QR gauge)')
    if dg_steps:
        ax1.plot(step_list, dg_ip_v, 'm--o', lw=1.5, label='IP MPF (DG gauge)')
    ax1.set_xlabel('Cycle'); ax1.set_ylabel('|C(t)|')
    ax1.set_title('MPF extrapolation')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # twin x for chi
    ax1b = ax1.twiny()
    ax1b.set_xlim(ax1.get_xlim())
    ax1b.set_xticks(step_list[::2])
    ax1b.set_xticklabels([str(chi_v[i]) for i in range(0,len(step_list),2)], fontsize=8)
    ax1b.set_xlabel('$\\chi$', fontsize=9)

    # ── Panel 2: timing ──────────────────────────────────────────────────────
    # t_cycle     = max(t_trotter_pure, t_ip_wall) — total wall-clock
    # t_trotter   = GPU 0 pure Trotter (4 states sequential)
    # t_ip_wall   = GPU 1 IP wall-clock (9 pairs, parallel streams)
    # t_upload    = P2P data transfer overhead

    # DG timing overlay
    dg_ip_t = [dg_steps[s]['t_dg_ip'] if s in dg_steps else float('nan') for s in step_list]

    ax2.plot(step_list, t_cycle_v,   'k-o',  lw=2.5, label='Total cycle = max(Trotter, IP)')
    ax2.plot(step_list, t_trot_v,    'r-s',  lw=1.5, label='Trotter GPU 0 (4 states)')
    ax2.plot(step_list, t_ip_wall_v, 'b-^',  lw=1.5, label='IP wall-clock QR (9 parallel)')
    if dg_steps:
        ax2.plot(step_list, dg_ip_t, 'm-o',  lw=1.5, label='IP wall-clock DG (sequential)')
    ax2.plot(step_list, t_upload_v,  'g--',  lw=1.0, label='Upload overhead')
    ax2.set_xlabel('Cycle'); ax2.set_ylabel('Time (s)')
    ax2.set_title('Per-cycle timing\nt_cycle = max(Trotter, IP)')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis='y')

    # chi on top
    ax2b = ax2.twiny()
    ax2b.set_xlim(ax2.get_xlim())
    ax2b.set_xticks(step_list[::2])
    ax2b.set_xticklabels([str(chi_v[i]) for i in range(0,len(step_list),2)], fontsize=8)
    ax2b.set_xlabel('$\\chi$', fontsize=9)

    plt.tight_layout()
    outfile = args.out or args.ip.replace('.log','_timing.png')
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")

if __name__ == '__main__':
    main()
