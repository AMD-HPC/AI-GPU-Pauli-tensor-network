#!/usr/bin/env python3
"""
make_fig4_qr_sweep.py — Figure 4 replacement: QR MPF sweep across Trotter delta.

Two panels:
  Left:  |C(t)| — ideal C_nl, noisy band (gamma=1,2), poor man's MPF, QR MPF per delta
  Right: Timing — Trotter and QR IP wall-clock per delta (symlog)

Usage:
  python3 make_fig4_qr_sweep.py --dir logs/dg_sweep_XXXXX --out fig4_new.png
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
                r"\s+t_ip_wall ([\d.nan]+)\s+t_upload ([\d.nan]+)", line)
            if not m: continue
            s = int(m.group(1))
            C_nl = float(m.group(2))
            C_n  = np.array([float(m.group(i)) for i in [3,4,5]])
            A = np.array([[safe(m.group(i)) for i in [6,7,8]],
                          [safe(m.group(7)), safe(m.group(9)), safe(m.group(10))],
                          [safe(m.group(8)), safe(m.group(10)), safe(m.group(11))]])
            B = np.array([safe(m.group(i)) for i in [12,13,14]])
            chi = int(m.group(15))
            def to_f(x):
                try: return float(x)
                except: return float('nan')
            t_trot = to_f(m.group(18))
            t_ip   = to_f(m.group(19))
            steps[s] = dict(C_nl=C_nl, C_n=C_n, A=A, B=B,
                            chi=chi, t_trotter=t_trot, t_ip=t_ip)
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

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', default=None)
    args=ap.parse_args()

    qr_logs = sorted(glob.glob(os.path.join(args.dir,'QR_delta*.log')))
    if not qr_logs:
        print("No QR_delta*.log files found"); return

    print(f"Found {len(qr_logs)} QR logs")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,5))

    colors = ['navy','royalblue','cornflowerblue']
    ls_trot = ['-','--',':']
    deltas  = ['δ=1e-3','δ=1e-4','δ=1e-5']

    # ── Panel 1: |C(t)| ───────────────────────────────────────────────────────
    # Noisy band from first log (delta=1e-3)
    ref = parse_qr(qr_logs[0])
    sl_ref = sorted(ref.keys())
    Cnl  = [abs(ref[s]['C_nl'])   for s in sl_ref]
    Cn1  = [abs(ref[s]['C_n'][0]) for s in sl_ref]
    Cn2  = [abs(ref[s]['C_n'][1]) for s in sl_ref]

    ax1.plot(sl_ref, Cnl, 'k-o', lw=2.5, zorder=5,
             label=r'$|C_{\rm nl}|$ (ideal)')

    # Noisy band — filled first (behind), then lines on top
    ax1.fill_between(sl_ref,
                     [min(Cn1[i],Cn2[i]) for i in range(len(sl_ref))],
                     [max(Cn1[i],Cn2[i]) for i in range(len(sl_ref))],
                     alpha=0.20, color='gray', zorder=1,
                     label='Noisy band (γ=1,2)')
    # γ=1: slower decay, stays above C_nl
    ax1.plot(sl_ref, Cn1, color='steelblue', lw=1.8, alpha=0.85,
             marker='s', ms=5, zorder=3, label=r'$|C_{n,\gamma=1}|$ (slow decay)')
    # γ=2: kills recurrence at cycle 5, drops below C_nl
    ax1.plot(sl_ref, Cn2, color='salmon', lw=1.8, alpha=0.85,
             marker='D', ms=5, zorder=3, label=r'$|C_{n,\gamma=2}|$ (kills recurrence)')

    # Poor man's from reference
    pm = poor_mans(ref)
    pm_v = [abs(pm.get(s,float('nan'))) for s in sl_ref]
    ax1.plot(sl_ref, pm_v, 'g--s', lw=1.8, zorder=4,
             label="Poor man's MPF")

    # QR MPF per delta
    for k,logf in enumerate(qr_logs):
        steps = parse_qr(logf)
        sl = sorted(steps.keys())
        cmpf_v = [abs(solve_lagrange(steps[s]['A'],steps[s]['B'],steps[s]['C_n']))
                  for s in sl]
        col = colors[k%len(colors)]
        lbl = deltas[k] if k<len(deltas) else f'δ=?'
        ax1.plot(sl, cmpf_v, '-^', color=col, lw=1.8,
                 label=f'QR MPF {lbl}', zorder=3, alpha=0.85)

    ax1.set_xlabel('Cycle', fontsize=12)
    ax1.set_ylabel('|C(t)|', fontsize=12)
    ax1.set_title(r'$|C(t)|$: ideal, noisy band, QR MPF corrections'
                  '\n(χ=250, 10 cycles, γ-MPF, center=13)', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Twin x-axis for chi
    chi_v = [ref[s]['chi'] for s in sl_ref]
    ax1b = ax1.twiny()
    ax1b.set_xlim(ax1.get_xlim())
    ax1b.set_xticks(sl_ref[::2])
    ax1b.set_xticklabels([str(chi_v[i]) for i in range(0,len(sl_ref),2)], fontsize=8)
    ax1b.set_xlabel('χ', fontsize=9)

    # ── Panel 2: Timing (symlog) ───────────────────────────────────────────────
    for k,logf in enumerate(qr_logs):
        steps = parse_qr(logf)
        sl = sorted(steps.keys())
        tip_v  = [steps[s]['t_ip']      for s in sl]
        trot_v = [steps[s]['t_trotter'] for s in sl]
        col = colors[k%len(colors)]
        lbl = deltas[k] if k<len(deltas) else f'δ=?'
        ax2.plot(sl, tip_v,  '-^', color=col, lw=1.8,
                 label=f'QR IP {lbl}', alpha=0.85)
        ax2.plot(sl, trot_v, 'k'+ls_trot[k%3], lw=1.5, alpha=0.7,
                 label=f'Trotter {lbl}')

    ax2.set_yscale('symlog', linthresh=1.0)
    ax2.set_xlabel('Cycle', fontsize=12)
    ax2.set_ylabel('Wall time (s)  [symlog]', fontsize=12)
    ax2.set_title('Timing: Trotter vs QR inner products\n'
                  '(IP hidden under Trotter at δ=1e-3; dominates at δ=1e-5)', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Twin x for chi
    ax2b = ax2.twiny()
    ax2b.set_xlim(ax2.get_xlim())
    ax2b.set_xticks(sl_ref[::2])
    ax2b.set_xticklabels([str(chi_v[i]) for i in range(0,len(sl_ref),2)], fontsize=8)
    ax2b.set_xlabel('χ', fontsize=9)

    plt.tight_layout()
    outfile = args.out or os.path.join(args.dir,'fig4_qr_sweep.png')
    plt.savefig(outfile, dpi=150)
    print(f"Saved: {outfile}")

if __name__ == '__main__':
    main()
