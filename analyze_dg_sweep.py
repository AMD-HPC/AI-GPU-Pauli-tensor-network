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

    # ── Panel 1: |C(t)| — ideal, noisy, MPF corrections ─────────────────────
    # Collect C_nl and C_n across all QR delta values to show bands
    all_Cnl  = {}  # step → list of |C_nl| across runs
    all_Cn1  = {}  # step → list of |C_n[0]| (gamma=1)
    all_Cn2  = {}  # step → list of |C_n[1]| (gamma=2)
    for logf in qr_logs:
        qs = parse_qr(logf)
        for s,d in qs.items():
            all_Cnl.setdefault(s,[]).append(abs(d['C_nl']))
            all_Cn1.setdefault(s,[]).append(abs(d['C_n'][0]))
            all_Cn2.setdefault(s,[]).append(abs(d['C_n'][1]))

    sl_ref = sorted(all_Cnl.keys())
    Cnl_mid = [np.mean(all_Cnl[s]) for s in sl_ref]
    Cn1_mid = [np.mean(all_Cn1[s]) for s in sl_ref]
    Cn2_mid = [np.mean(all_Cn2[s]) for s in sl_ref]
    Cn1_lo  = [np.min(all_Cn1[s])  for s in sl_ref]
    Cn1_hi  = [np.max(all_Cn1[s])  for s in sl_ref]
    Cn2_lo  = [np.min(all_Cn2[s])  for s in sl_ref]
    Cn2_hi  = [np.max(all_Cn2[s])  for s in sl_ref]

    ax1.plot(sl_ref, Cnl_mid, 'k-o', lw=2.5, label=r'$|C_{\rm nl}|$ (ideal)', zorder=5)
    ax1.plot(sl_ref, Cn1_mid, 'steelblue', lw=1.5, alpha=0.8, label=r'$|C_{n,\gamma=1}|$')
    ax1.fill_between(sl_ref, Cn1_lo, Cn1_hi, alpha=0.15, color='steelblue')
    ax1.plot(sl_ref, Cn2_mid, 'salmon', lw=1.5, alpha=0.8, label=r'$|C_{n,\gamma=2}|$')
    ax1.fill_between(sl_ref, Cn2_lo, Cn2_hi, alpha=0.15, color='salmon')

    # Poor man's — accurate but potentially biased (same sign correction each cycle)
    ref_log = qr_logs[0]
    ref_steps = parse_qr(ref_log)
    pm = poor_mans(ref_steps)
    pm_v = [abs(pm.get(s,float('nan'))) for s in sl_ref]
    ax1.plot(sl_ref, pm_v, 'g--s', lw=1.8, label="Poor man's MPF", zorder=4)

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
        cmpf_v = [abs(solve_lagrange(steps[s]['A'],steps[s]['B'],steps[s]['C_n']))
                  for s in sl]
        col=colors_qr[k%len(colors_qr)]
        ax1.plot(sl, cmpf_v, '-^', color=col, lw=1.5, alpha=0.8,
                 label=f'QR MPF {tag}', zorder=3)
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
        cmpf_v = [abs(solve_lagrange(steps[s]['A'],steps[s]['B'],steps[s]['C_n']))
                  for s in sl]
        col=colors_dg[k%len(colors_dg)]
        ax1.plot(sl, cmpf_v, '--o', color=col, lw=1.5, alpha=0.8,
                 label=f'DG MPF {tag}', zorder=3)
        ax2.plot(sl, tip_v, '--o', color=col, lw=1.5, label=f'DG IP {tag}')

    ax1.set_xlabel('Cycle'); ax1.set_ylabel('|C(t)|')
    ax1.set_title('|C(t)|: ideal, noisy band, MPF corrections\n'
                  '(QR biased, DG unbiased but larger scatter)')
    ax1.legend(fontsize=6, ncol=2)
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

    # ── C(t) table: rows=cycles, cols=C_nl, C_n1, C_n2, QR_*, DG_* ──────────
    # Collect all configs
    all_configs = {}  # label → {step: |C_mpf|}
    for logf in qr_logs:
        tag = os.path.basename(logf).replace('QR_','QR ').replace('.log','')
        steps = parse_qr(logf)
        all_configs[tag] = {s: abs(solve_lagrange(d['A'],d['B'],d['C_n']))
                            for s,d in steps.items()}
    for logf in dg_logs:
        tag = os.path.basename(logf).replace('DG_','DG ').replace('.log','')
        steps = parse_dg(logf)
        all_configs[tag] = {s: abs(solve_lagrange(d['A'],d['B'],d['C_n']))
                            for s,d in steps.items()}

    # Reference C_nl, C_n1, C_n2 from first QR log
    ref = parse_qr(qr_logs[0])
    pm_ref = poor_mans(ref)
    all_steps = sorted(ref.keys())

    # Header
    cols = ['C_nl','C_n1','C_n2','PM'] + list(all_configs.keys())
    col_w = 10
    print(f"\n{'Cycle':>5}  " + "  ".join(f"{c[:col_w]:>{col_w}}" for c in cols))
    print("─" * (7 + (col_w+2)*len(cols)))

    for s in all_steps:
        d = ref[s]
        cnl = abs(d['C_nl'])
        cn1 = abs(d['C_n'][0])
        cn2 = abs(d['C_n'][1])
        pm_v = abs(pm_ref.get(s, float('nan')))
        row = [cnl, cn1, cn2, pm_v]
        for cfg in all_configs:
            v = all_configs[cfg].get(s, float('nan'))
            row.append(v)
        print(f"{s:>5}  " + "  ".join(f"{v:>{col_w}.4f}" if not np.isnan(v)
                                       else f"{'nan':>{col_w}}" for v in row))
    print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Config':<30} {'Steps':>5} {'Mean|Δ|':>10} {'Max|Δ|':>10} "
          f"{'Mean t_ip':>10} {'Bias?':>8}")
    print("─"*78)

    # Poor man's reference
    ref_steps = parse_qr(qr_logs[0])
    pm = poor_mans(ref_steps)
    pm_errs = [abs(pm.get(s,float('nan'))-ref_steps[s]['C_nl'])
               for s in sorted(ref_steps) if not np.isnan(pm.get(s,float('nan')))]
    # Bias: check if errors are consistently one-sided
    pm_signed = [pm.get(s,float('nan'))-ref_steps[s]['C_nl']
                 for s in sorted(ref_steps)]
    pm_bias = "yes(+)" if all(v>0 for v in pm_signed if not np.isnan(v)) else \
              "yes(-)" if all(v<0 for v in pm_signed if not np.isnan(v)) else "mixed"
    print(f"{'Poor mans (δ=1e-3)':<30} {len(pm_errs):>5} "
          f"{np.mean(pm_errs):>10.2e} {np.max(pm_errs):>10.2e} {'—':>10} {pm_bias:>8}")

    print()
    for logf in qr_logs+dg_logs:
        tag = os.path.basename(logf).replace('.log','')
        is_dg = 'DG' in tag
        steps = (parse_dg if is_dg else parse_qr)(logf)
        errs_d = mpf_errors(steps)
        sl = sorted(s for s in errs_d if not np.isnan(errs_d[s]['ip']))
        if not sl: continue
        ip_v  = [errs_d[s]['ip'] for s in sl]
        ti_v  = [errs_d[s]['t_ip'] for s in sl if not np.isnan(errs_d[s]['t_ip'])]
        # Signed errors to detect bias
        signed = [solve_lagrange(steps[s]['A'],steps[s]['B'],steps[s]['C_n'])
                  - steps[s]['C_nl'] for s in sl]
        signed = [v for v in signed if not np.isnan(v) and abs(v)<2]
        bias = "yes(+)" if len(signed)>2 and sum(v>0 for v in signed)/len(signed)>0.75 else \
               "yes(-)" if len(signed)>2 and sum(v<0 for v in signed)/len(signed)>0.75 else "mixed"
        mean_ip = np.mean(ip_v) if ip_v else float('nan')
        max_ip  = np.max(ip_v)  if ip_v else float('nan')
        mean_ti = np.mean(ti_v) if ti_v else float('nan')
        print(f"{tag:<30} {len(sl):>5} {mean_ip:>10.2e} {max_ip:>10.2e} "
              f"{mean_ti:>10.2f}s {bias:>8}")

if __name__=='__main__':
    main()
