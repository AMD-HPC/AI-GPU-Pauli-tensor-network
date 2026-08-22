#!/usr/bin/env python3
"""
analyze_pauli_mpf.py — Poor-man's MPF from main_pauli_sweep output.

Reads STEP lines with C_nl, C_n1, C_n2, C_n3 and computes:
  - Poor-man's MPF: A_ij = Σ_t C_ni(t)*C_nj(t), B_i = Σ_t C_ni(t)*C_nl(t)
  - MPF extrapolated C_mpf = Σ c_i * C_ni
  - |Δpm| = |C_mpf - C_nl|

Usage:
  python3 scripts/analyze_pauli_mpf.py logs/A2_hh_200_10c_f32.log
  python3 scripts/analyze_pauli_mpf.py logs/A2*.log logs/A4*.log  # compare
"""

import sys, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GAMMAS = [1.0, 2.0, 3.0]
N = 3

def parse_log(path):
    steps = []
    cur = {}
    with open(path) as f:
        for line in f:
            m = re.match(
                r'STEP (\d+)\s+C_nl ([+-][\d.e+-]+)\s+'
                r'C_n1 ([+-][\d.e+-]+)\s+C_n2 ([+-][\d.e+-]+)\s+C_n3 ([+-][\d.e+-]+)',
                line)
            if m:
                if cur: steps.append(cur)
                cur = {
                    'step': int(m.group(1)),
                    'C_nl': float(m.group(2)),
                    'C_n':  [float(m.group(3)), float(m.group(4)), float(m.group(5))],
                    'pauli_nl': None, 'pauli_n': [None, None, None]
                }
            # Parse spatial N(v,t) rows
            m2 = re.match(r'PAULI_NL (\d+)\s+(.*)', line)
            if m2 and cur:
                cur['pauli_nl'] = np.array(list(map(float, m2.group(2).split())))
            for k, tag in enumerate(['N1','N2','N3']):
                m3 = re.match(rf'PAULI_{tag} (\d+)\s+(.*)', line)
                if m3 and cur:
                    cur['pauli_n'][k] = np.array(list(map(float, m3.group(2).split())))
    if cur: steps.append(cur)
    return steps

def cross_entropy(p, q, eps=1e-10):
    """Cross-entropy H(p,q) = -Σ p log q. p=reference, q=approximation."""
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    # Normalize to proper distributions
    ps, qs = p.sum(), q.sum()
    if ps < eps or qs < eps: return np.nan
    p = p / ps; q = q / qs
    # Clip for numerical safety
    q = np.clip(q, eps, None)
    return float(-np.sum(p * np.log(q)))

def spatial_mpf(pauli_nl, pauli_n_list, x):
    """Apply MPF coefficients spatially: N_mpf[v] = Σ c_i N_ni[v]."""
    if pauli_nl is None or any(p is None for p in pauli_n_list):
        return None
    result = np.zeros_like(pauli_nl)
    for k, p in enumerate(pauli_n_list):
        result += x[k] * p
    return result

def poor_mans_mpf(steps):
    results = []
    A_cum = np.zeros((N, N))
    B_cum = np.zeros(N)
    for s in steps:
        C_nl = s['C_nl']
        C_n  = np.array(s['C_n'])
        for i in range(N):
            B_cum[i]   += C_n[i] * C_nl
            for j in range(N):
                A_cum[i,j] += C_n[i] * C_n[j]
        # Lagrange solve
        M = np.zeros((N+1, N+1))
        M[:N,:N] = A_cum; M[:N,N] = 1; M[N,:N] = 1
        rhs = np.zeros(N+1); rhs[:N] = B_cum; rhs[N] = 1
        try:
            sol = np.linalg.solve(M, rhs)
            x = sol[:N]
            C_mpf = float(x @ C_n)
        except:
            x = np.full(N, np.nan); C_mpf = np.nan
        results.append({
            'step': s['step'], 'C_nl': C_nl, 'C_n': C_n,
            'C_mpf': C_mpf, 'x': x,
            'err': abs(C_mpf - C_nl) if not np.isnan(C_mpf) else np.nan
        })
    return results

def print_table(results, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"{'Step':>4}  {'C_nl':>9}  {'C_n1':>9}  {'C_n2':>9}  {'C_n3':>9}  "
          f"{'C_mpf':>9}  {'|Δpm|':>8}  {'x1':>6} {'x2':>6} {'x3':>6}")
    print('-'*85)
    for r in results:
        C_n = r['C_n']
        x = r['x']
        err_s = f"{r['err']:.2e}" if not np.isnan(r['err']) else '  nan   '
        mpf_s = f"{r['C_mpf']:+8.4f}" if not np.isnan(r['C_mpf']) else '   nan  '
        print(f"{r['step']:>4}  {r['C_nl']:+9.6f}  {C_n[0]:+9.6f}  {C_n[1]:+9.6f}  "
              f"{C_n[2]:+9.6f}  {mpf_s}  {err_s}  "
              f"{x[0]:+5.2f} {x[1]:+5.2f} {x[2]:+5.2f}")

def plot_results(all_results, all_labels, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Poor-man\'s MPF comparison', fontsize=12)
    colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3']
    markers = ['o','s','^','D']

    for ri, (results, label) in enumerate(zip(all_results, all_labels)):
        steps = [r['step'] for r in results]
        C_nl  = [r['C_nl'] for r in results]
        C_mpf = [r['C_mpf'] for r in results]
        errs  = [r['err'] for r in results]
        color = colors[ri % len(colors)]
        marker = markers[ri % len(markers)]

        axes[0].plot(steps, C_nl, '-', color=color, lw=1.5, label=f'{label} C_nl')
        axes[0].plot(steps, C_mpf, '--', color=color, lw=1.5, marker=marker, ms=4,
                    label=f'{label} C_mpf')
        axes[1].semilogy(steps, [max(e,1e-10) for e in errs], '-o',
                        color=color, lw=1.5, ms=4, label=label)
        # coefficients
        for k in range(N):
            axes[2].plot(steps, [r['x'][k] for r in results],
                        color=color, lw=1, alpha=0.7+0.1*k,
                        linestyle=['-','--',':'][k],
                        label=f'{label} c{k+1}' if ri==0 else '')

    axes[0].set_title('C(t): noiseless vs MPF'); axes[0].legend(fontsize=7)
    axes[0].set_xlabel('step'); axes[0].grid(True, alpha=0.3)
    axes[1].set_title('|C_mpf - C_nl|'); axes[1].legend(fontsize=7)
    axes[1].set_xlabel('step'); axes[1].grid(True, alpha=0.3)
    axes[2].set_title('MPF coefficients c_i'); axes[2].legend(fontsize=7)
    axes[2].set_xlabel('step'); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    print(f"\nPlot saved: {outpath}")

def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_pauli_mpf.py log1.log [log2.log ...]")
        sys.exit(1)

    paths = sys.argv[1:]
    all_results = []
    all_labels  = []

    for path in paths:
        label = path.split('/')[-1].replace('.log','')
        steps = parse_log(path)
        if not steps:
            print(f"WARNING: no STEP lines found in {path}"); continue
        results = poor_mans_mpf(steps)
        print_table(results, label)
        all_results.append(results)
        all_labels.append(label)

    if all_results:
        outpath = paths[0].replace('.log', '_pauli_mpf.png')
        plot_results(all_results, all_labels, outpath)

if __name__ == '__main__':
    main()
