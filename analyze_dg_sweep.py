#!/usr/bin/env python3
"""
analyze_dg_sweep.py — Compare DG vs QR inner product MPF across parameter sweep.

Reads all logs in --dir, computes MPF error per step, plots summary.

Usage:
  python3 analyze_dg_sweep.py --dir logs/dg_sweep_XXXXXX [--out fig.png]
"""

import argparse, re, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = 3

def safe(x):
    try: v=float(x); return v if abs(v)<1e10 else float('nan')
    except: return float('nan')

def parse_qr(path):
    steps = {}
    with open(path) as f:
        for line in f:
            m = re.match(
                r"STEP (\d+)\s+C_nl ([+-][\d.e+-]+)\s+C_n ([+-][\d.e+-]+) ([+-][\d.e+-]+) ([+-][\d.e+-]+)"
                r"\s+IP_A (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)"
                r"\s+IP_B (\S+) (\S+) (\S+)"
                r"\s+chi (\d+) (\d+)\s+t_cycle ([\d.]+)\s+t_trotter_pure ([\d.]+)"
                r"\s+t_ip_wall ([\d.nan]+)", line)
            if not m: continue
            s = int(m.group(1))
            C_nl = float(m.group(2))
            C_n  = np.array([float(m.group(i)) for i in [3,4,5]])
            A = np.array([[safe(m.group(i)) for i in [6,7,8]],
                          [safe(m.group(7)), safe(m.group(9)), safe(m.group(10))],
                          [safe(m.group(8)), safe(m.group(10)), safe(m.group(11))]])
            B = np.array([safe(m.group(i)) for i in [12,13,14]])
            chi = int(m.group(15))
            tip = safe(m.group(19))
            steps[s] = dict(C_nl=C_nl, C_n=C_n, A=A, B=B, chi=chi, t_ip=tip)
    return steps

def parse_dg(path):
    steps = {}
    with open(path) as f:
        for line in f:
            m = re.match(
                r"STEP (\d+)\s+C_nl ([+-][\d.e+-]+)\s+C_n ([+-][\d.e+-]+) ([+-][\d.e+-]+) ([+-][\d.e+-]+)"
                r"\s+IP_A (\S+) (\S+) (\S+) (\S+) (\S+) (\S+)"
                r"\s+IP_B (\S+) (\S+) (\S+)"
                r"\s+chi (\d+) (\d+)\s+t_cycle ([\d.]+)\s+t_trotter ([\d.]+)"
                r"\s+t_dg_precompute ([\d.nan]+)\s+t_dg_pairs ([\d.nan]+)", line)
            if not m: continue
            s = int(m.group(1))
            C_nl = float(m.group(2))
            C_n  = np.array([float(m.group(i)) for i in [3,4,5]])
            A = np.array([[safe(m.group(i)) for i in [6,7,8]],
                          [safe(m.group(7)), safe(m.group(9)), safe(m.group(10))],
                          [safe(m.group(8)), safe(m.group(10)), safe(m.group(11))]])
            B = np.array([safe(m.group(i)) for i in [12,13,14]])
            chi = int(m.group(15))
            tp = safe(m.group(19)); tq = safe(m.group(20))
            tip = tp+tq if not (np.isnan(tp) or np.isnan(tq)) else float('nan')
            steps[s] = dict(C_nl=C_nl, C_n=C_n, A=A, B=B, chi=chi, t_ip=tip)
    return steps

def solve_lagrange(A, B, C_n):
    try:
        M = np.zeros((N+1,N+1)); M[:N,:N]=A; M[:N,N]=1; M[N,:N]=1
        rhs = np.zeros(N+1); rhs[:N]=B; rhs[N]=1
        sol = np.linalg.solve(M, rhs)
        result = float(sol[:N] @ C_n)
        if abs(result) > 3*abs(C_n).max()+1.0: return float('nan')
        return result
    except: return float('nan')

def poor_mans(steps):
    Ac=np.zeros((N,N)); Bc=np.zeros(N); pm={}
    for s in sorted(steps):
        d=steps[s]; cn=d['C_n']; cnl=d['C_nl']
        for i in range(N):
            Bc[i]+=cn[i]*cnl
            for j in range(N): Ac[i,j]+=cn[i]*cn[j]
        pm[s]=solve_lagrange(Ac.copy(),Bc.copy(),cn)
    return pm

def mpf_errors(steps):
    pm=poor_mans(steps)
    errs={}
    for s,d in steps.items():
        cnl=d['C_nl']; cn=d['C_n']
        cip=solve_lagrange(d['A'],d['B'],cn)
        errs[s]=dict(pm=abs(pm.get(s,float('nan'))-cnl),
                     ip=abs(cip-cnl) if not np.isnan(cip) else float('nan'),
                     chi=d['chi'], t_ip=d['t_ip'])
    return errs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', default=None)
    args=ap.parse_args()

    # Load all logs
    qr_logs = sorted(glob.glob(os.path.join(args.dir,'QR_*.log')))
    dg_logs = sorted(glob.glob(os.path.join(args.dir,'DG_*.log')))

    print(f"Found {len(qr_logs)} QR logs, {len(dg_logs)} DG logs")

    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    ax1, ax2 = axes
    ax3 = ax2  # panel 3 merged into panel 2

    colors_qr = ['navy','royalblue','cornflowerblue']
    colors_dg = ['darkred','red','salmon','orange','gold']

    # ── Panel 1: MPF error |C_mpf - C_nl| for ALL curves ────────────────────
    # Poor man's from reference log (delta=1e-3 QR)
    ref_log = qr_logs[0]
    ref_steps = parse_qr(ref_log)
    pm = poor_mans(ref_steps)
    sl_ref = sorted(ref_steps.keys())
    pm_err = [abs(pm.get(s,float('nan')) - ref_steps[s]['C_nl']) for s in sl_ref]
    ax1.semilogy(sl_ref, pm_err, 'g--s', lw=2, label="Poor man's (δ=1e-3)", zorder=5)

    # QR MPF curves (one per delta)
    trotter_linestyles = ['-', '--', ':']
    for k,logf in enumerate(qr_logs):
        tag = os.path.basename(logf).replace('QR_','').replace('.log','')
        steps = parse_qr(logf)
        errs  = mpf_errors(steps)
        sl = sorted(s for s in errs)
        ip_v  = [errs[s]['ip'] for s in sl]   # |error| for panel 2
        tip_v = [errs[s]['t_ip'] for s in sl]
        # Actual C_mpf for panel 1
        err_v = [abs(solve_lagrange(steps[s]['A'],steps[s]['B'],steps[s]['C_n']) - steps[s]['C_nl'])
                 for s in sl]
        col=colors_qr[k%len(colors_qr)]
        ax1.semilogy(sl, err_v, '-^', color=col, lw=1.5, label=f'QR {tag}', zorder=3)
        ax2.plot(sl, tip_v, '-^', color=col, lw=1.5, label=f'QR IP {tag}')
        # Trotter in panel 3
        tsl2, ttr_v = [], []
        with open(logf) as f:
            for line in f:
                m = re.match(r"STEP (\d+).*t_trotter_pure ([\d.]+)", line)
                if m: tsl2.append(int(m.group(1))); ttr_v.append(float(m.group(2)))

    # DG MPF curves
    for k,logf in enumerate(dg_logs):
        tag = os.path.basename(logf).replace('DG_','').replace('.log','')
        steps = parse_dg(logf)
        errs  = mpf_errors(steps)
        sl = sorted(s for s in errs)
        tip_v = [errs[s]['t_ip'] for s in sl]
        err_v = [abs(solve_lagrange(steps[s]['A'],steps[s]['B'],steps[s]['C_n']) - steps[s]['C_nl'])
                 for s in sl]
        col=colors_dg[k%len(colors_dg)]
        ax1.semilogy(sl, err_v, '--o', color=col, lw=1.5, label=f'DG {tag}', zorder=3)
        ax2.plot(sl, tip_v, '--o', color=col, lw=1.5, label=f'DG IP {tag}')

    ax1.set_xlabel('Cycle'); ax1.set_ylabel('|C_mpf - C_nl|')
    ax1.set_title('MPF error: all QR and DG configurations')
    ax1.legend(fontsize=6, ncol=2); ax1.grid(True, alpha=0.3)

    ax2.set_yscale('symlog', linthresh=1.0)
    ax2.set_xlabel('Cycle'); ax2.set_ylabel('Time (s)  [symlog]')
    ax2.set_title('Timing: Trotter vs IP (DG always hidden under Trotter)')
    ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

    # Trotter lines on ax2 (same panel as timing)
    for k,logf in enumerate(qr_logs):
        tag = os.path.basename(logf).replace('QR_','').replace('.log','')
        tsl2, ttr_v = [], []
        with open(logf) as f:
            for line in f:
                m = re.match(r"STEP (\d+).*t_trotter_pure ([\d.]+)", line)
                if m:
                    tsl2.append(int(m.group(1)))
                    ttr_v.append(float(m.group(2)))
        if tsl2:
            ls = trotter_linestyles[k%len(trotter_linestyles)]
            ax3.plot(tsl2, ttr_v, 'k'+ls, lw=2, label=f'Trotter {tag}')
    # Overlay DG IP on panel 3 for direct comparison with Trotter
    for k,logf in enumerate(dg_logs):
        tag = os.path.basename(logf).replace('DG_','').replace('.log','')
        steps = parse_dg(logf)
        errs  = mpf_errors(steps)
        sl = sorted(errs.keys())
        tip_v = [errs[s]['t_ip'] for s in sl if not np.isnan(errs[s]['t_ip'])]
        sl2   = [s for s in sl if not np.isnan(errs[s]['t_ip'])]
        if sl2:
            col=colors_dg[k%len(colors_dg)]
            ax3.plot(sl2, tip_v, '--o', color=col, lw=1.5, label=f'DG IP {tag}')

    # Summary table
    print(f"\n{'Log':<35} {'Steps':>5} {'Mean|Δip|':>12} {'Mean t_ip':>10}")
    print("-"*65)
    for logf in qr_logs+dg_logs:
        tag=os.path.basename(logf)
        is_dg='DG' in tag
        steps=(parse_dg if is_dg else parse_qr)(logf)
        errs=mpf_errors(steps)
        ip_v=[errs[s]['ip'] for s in errs if not np.isnan(errs[s]['ip'])]
        ti_v=[errs[s]['t_ip'] for s in errs if not np.isnan(errs[s]['t_ip'])]
        mean_ip=np.nanmean(ip_v) if ip_v else float('nan')
        mean_ti=np.nanmean(ti_v) if ti_v else float('nan')
        print(f"{tag:<35} {len(ip_v):>5} {mean_ip:>12.3e} {mean_ti:>10.2f}s")

    plt.tight_layout()
    outfile=args.out or os.path.join(args.dir,'dg_sweep_summary.png')
    plt.savefig(outfile, dpi=150)
    print(f"\nFigure: {outfile}")

if __name__=='__main__':
    main()
