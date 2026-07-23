#!/usr/bin/env python3
"""
analyze_sweep.py — Parse CppSim sweep logs and plot C_nl + C_mpf with bands

Usage:
    python3 analyze_sweep.py [log_dir]
    python3 analyze_sweep.py logs/sweep_20260723_173437

Plots:
    plot_cnl_bands.png  — C_nl(t) with chi/cutoff band
    plot_cmpf_bands.png — C_mpf(t) with chi/cutoff band
    plot_chi_growth.png — chi growth per cycle
"""

import os, re, sys, glob
import numpy as np

log_dir = sys.argv[1] if len(sys.argv) > 1 else \
    sorted(glob.glob('logs/sweep_*'))[-1]
print(f"Reading: {log_dir}")

# ── Parse all log files ────────────────────────────────────────────────────────
def parse_log(path):
    """Returns dict: step → (C_nl, C_mpf, delta_mpf, chi_nl, chi_n, t_s)"""
    data = {}
    with open(path) as f:
        for line in f:
            # Match step lines
            m = re.match(
                r'\s*(\d+)\s+([+-][\d.e+-]+)\s+([+-][\d.e+-]+)\s+([\d.e+-]+)'  # step C_nl Julia_nl Δnl
                r'(?:\s+[+-][\d.e+-]+\s+[\d.e+-]+){3}'  # 3x C_n |Δ|
                r'\s+([+-][\d.e+-]+)\s+([\d.e+-]+)'      # C_mpf |Δmpf|
                r'\s+(\d+)\s+(\d+)'                       # chi_nl chi_n
                r'(?:\s+\d+\s+[\d.]+)?'                   # optional #sw bp_ms
                r'\s+([\d.]+)', line)
            if m:
                step     = int(m.group(1))
                C_nl     = float(m.group(2))
                C_mpf    = float(m.group(5))
                d_mpf    = float(m.group(6))
                chi_nl   = int(m.group(7))
                chi_n    = int(m.group(8))
                t_s      = float(m.group(9))
                data[step] = (C_nl, C_mpf, d_mpf, chi_nl, chi_n, t_s)
    return data

# Group logs by (chi, cutoff)
logs = {}
mpf_type = 'gamma'
for logfile in sorted(glob.glob(f"{log_dir}/*.log")):
    m = re.search(r'(main_heavyhex_(\w+)_mpf)_chi(\d+)_cut([\de.-]+)', logfile)
    if not m: continue
    mtype = m.group(2)  # gamma or eps
    chi = int(m.group(3))
    cutoff = m.group(4).rstrip('.')  # remove trailing dot if any
    data = parse_log(logfile)
    if data:
        logs[(mtype, chi, cutoff)] = data
        mpf_type = mtype

print(f"Found {len(logs)} experiments, type={mpf_type}")
if not logs:
    print("No data found"); sys.exit(1)

# Get all (chi, cutoff) combos and cycles
all_chi     = sorted(set(chi for (_, chi, _) in logs))
all_cutoffs = sorted(set(cut for (_, _, cut) in logs), key=lambda x: float(x))
all_steps   = sorted(set(s for d in logs.values() for s in d))
N = max(all_steps)
cycles = list(range(1, N+1))

print(f"Chi values: {all_chi}")
print(f"Cutoffs: {all_cutoffs}")
print(f"Cycles: {N}")

# ── Plotting ───────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available")

COLORS = {'1e-2': 'tab:blue', '1e-3': 'tab:orange', '1e-4': 'tab:green',
          '1e-5': 'tab:red'}
CUTOFF_LABELS = {'1e-2': 'cutoff=1e-2', '1e-3': 'cutoff=1e-3',
                 '1e-4': 'cutoff=1e-4', '1e-5': 'cutoff=1e-5'}

def get_series(mtype, chi, cutoff, key_idx):
    """key_idx: 0=C_nl, 1=C_mpf, 2=d_mpf, 3=chi_nl, 4=chi_n"""
    d = logs.get((mtype, chi, cutoff), {})
    return [d[s][key_idx] if s in d else None for s in cycles]

if HAS_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot |C_nl| and |C_mpf| to see envelope clearly (absolute value removes sign oscillation)
    for ax_idx, (key_idx, ylabel, title_suffix, show_nl_ref) in enumerate([
        (0, '|C_nl(t)|', 'Noiseless |C_nl(t)| — chi/cutoff convergence', False),
        (1, '|C_mpf(t)|', 'MPF |C_mpf(t)| vs |C_nl| reference', True)
    ]):
        ax = axes[ax_idx]

        # Band across ALL chi AND cutoff values
        band_vals = []
        for c_idx in range(len(cycles)):
            vals = []
            for cutoff in all_cutoffs:
                for chi in all_chi:
                    v = get_series(mpf_type, chi, cutoff, key_idx)[c_idx]
                    if v is not None: vals.append(abs(v))
            band_vals.append(vals)
        bx  = [cycles[i] for i,v in enumerate(band_vals) if v]
        blo = [min(v) for v in band_vals if v]
        bhi = [max(v) for v in band_vals if v]
        if blo:
            ax.fill_between(bx, blo, bhi, alpha=0.15, color='gray',
                            label='all chi/cutoff band')

        # Lines per cutoff, largest chi
        for cutoff in all_cutoffs:
            color = COLORS.get(cutoff, 'gray')
            chi_max_local = max(all_chi)
            y = get_series(mpf_type, chi_max_local, cutoff, key_idx)
            yv = [abs(v) for v in y if v is not None]
            xv = [cycles[i] for i,v in enumerate(y) if v is not None]
            if yv:
                ax.plot(xv, yv, '-', color=color, lw=2,
                        label=f'{CUTOFF_LABELS[cutoff]} chi={chi_max_local}')

        # Add C_nl reference on C_mpf plot
        if show_nl_ref:
            y_nl = get_series(mpf_type, max(all_chi), '1e-4', 0)
            yv_nl = [abs(v) for v in y_nl if v is not None]
            xv_nl = [cycles[i] for i,v in enumerate(y_nl) if v is not None]
            if yv_nl:
                ax.plot(xv_nl, yv_nl, 'k--', lw=2, label='|C_nl| (chi=120, cut=1e-4)')

        ax.set_xlabel('Cycle', fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(f'{mpf_type.capitalize()}-MPF: {title_suffix}', fontsize=13)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(cycles[::2])

    fig.tight_layout()
    outpath = f"{log_dir}/plot_cnl_cmpf_bands.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Written: {outpath}")

    # Chi growth plot
    fig2, ax2 = plt.subplots(figsize=(10,6))
    for cutoff in all_cutoffs:
        color = COLORS.get(cutoff, 'gray')
        for chi in all_chi:
            y = get_series(mpf_type, chi, cutoff, 3)  # chi_nl
            yv = [v for v in y if v is not None]
            xv = [cycles[i] for i,v in enumerate(y) if v is not None]
            if yv:
                ax2.plot(xv, yv, '-', color=color, alpha=0.6, lw=1.5,
                         label=f'chi_max={chi} cut={cutoff}')
    ax2.set_xlabel('Cycle', fontsize=13)
    ax2.set_ylabel('chi_nl', fontsize=13)
    ax2.set_title(f'{mpf_type.capitalize()}-MPF: Chi growth', fontsize=13)
    ax2.legend(fontsize=7, ncol=3)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(f"{log_dir}/plot_chi_growth.png", dpi=150)
    plt.close(fig2)
    print(f"Written: {log_dir}/plot_chi_growth.png")

# ── Text summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"{'chi':>6}  {'cutoff':>8}  {'cycles':>6}  {'C_nl_final':>12}  {'C_mpf_final':>12}  {'|Δmpf|':>10}  {'chi_final':>10}")
print('='*70)
for cutoff in all_cutoffs:
    for chi in all_chi:
        d = logs.get((mpf_type, chi, cutoff), {})
        if not d: continue
        last_step = max(d.keys())
        C_nl, C_mpf, d_mpf, chi_nl, chi_n, t_s = d[last_step]
        print(f"{chi:>6}  {cutoff:>8}  {last_step:>6}  {C_nl:+12.6f}  {C_mpf:+12.6f}  {d_mpf:10.2e}  {chi_nl:>10}")

# ── Poor man's MPF: build A,B from Z(t) correlations ─────────────────────────
# A[i,j] = Σ_t C_n(γᵢ,t) × C_n(γⱼ,t)   (correlation matrix of noisy curves)
# B[i]   = Σ_t C_n(γᵢ,t) × C_nl(t)      (correlation with noiseless)
# Solve: min ||Ax - B||² s.t. Σxᵢ=1, Σ|xᵢ|≤condition_number
# C_mpf_pm(t) = Σᵢ xᵢ × C_n(γᵢ, t)
print(f"\n{'='*70}")
print("Poor man's MPF: A,B from Z(t) correlations (gauge-invariant)")
print(f"Scaffold for full A[i,j]=⟨ψᵢ|ψⱼ⟩ when inner products are available")
print('='*70)

# Use best (largest chi, finest cutoff) for the poor man's MPF
best_cutoff = '1e-4' if '1e-4' in all_cutoffs else all_cutoffs[0]
best_chi    = max(all_chi)
gamma_keys  = sorted([c for c in all_cutoffs], key=float)  # use cutoff as proxy for states

# For gamma-MPF: we have C_n at gamma=1,2,3 (all at same chi/cutoff)
# Collect C_n per gamma and C_nl from the best chi/cutoff run
if mpf_type == 'gamma':
    # Use fixed best cutoff, vary chi as proxy for "different noise levels" → wrong
    # Actually for gamma: each log is same chi/cutoff but different gamma
    # The gamma info is in the Z(t) columns of each log
    # Parse C_n(γ=1), C_n(γ=2), C_n(γ=3) from the log directly

    d = logs.get((mpf_type, best_chi, best_cutoff), {})
    if d:
        N_steps = sorted(d.keys())
        C_nl_arr = np.array([d[s][0] for s in N_steps])

        # Re-read log to get C_n per gamma
        logfile = f"{log_dir}/main_heavyhex_{mpf_type}_mpf_chi{best_chi}_cut{best_cutoff}.log"
        C_n_gammas = []  # list of arrays, one per gamma
        n_gammas = 3
        gamma_data = [[] for _ in range(n_gammas)]
        with open(logfile) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 12 or not parts[0].isdigit(): continue
                try:
                    # C_n columns: parts[4], 6, 8
                    for g in range(n_gammas):
                        gamma_data[g].append(float(parts[4 + g*2]))
                except: pass

        if all(len(gamma_data[g]) == len(N_steps) for g in range(n_gammas)):
            C_n_arr = [np.array(gamma_data[g]) for g in range(n_gammas)]

            # Build A and B
            ng = n_gammas
            A = np.zeros((ng, ng))
            B = np.zeros(ng)
            for i in range(ng):
                B[i] = np.dot(C_n_arr[i], C_nl_arr)
                for j in range(ng):
                    A[i,j] = np.dot(C_n_arr[i], C_n_arr[j])

            print(f"\nUsing chi={best_chi}, cutoff={best_cutoff}, {len(N_steps)} cycles")
            print(f"A matrix (correlation):\n{np.round(A,4)}")
            print(f"B vector: {np.round(B,4)}")

            # Solve with constraint sum(x)=1
            # Use least-squares with equality constraint via Lagrange multiplier
            # Augmented system: [A, 1; 1^T, 0] [x; λ] = [B; 1]
            n = ng
            M = np.zeros((n+1, n+1))
            M[:n,:n] = A
            M[:n,n] = 1.0
            M[n,:n] = 1.0
            rhs = np.zeros(n+1)
            rhs[:n] = B
            rhs[n] = 1.0
            try:
                sol = np.linalg.solve(M, rhs)
                c = sol[:n]
                print(f"Coefficients (poor man's): {np.round(c,4)}  sum={c.sum():.4f}")

                # Apply to get C_mpf_pm per cycle
                C_mpf_pm = sum(c[g] * C_n_arr[g] for g in range(ng))
                delta_pm = np.abs(C_mpf_pm - C_nl_arr)
                print(f"\n{'Cycle':>6}  {'C_nl':>10}  {'C_mpf(poly)':>12}  {'C_mpf(pm)':>12}  {'|Δ(poly)|':>10}  {'|Δ(pm)|':>10}")
                for idx, s in enumerate(N_steps):
                    C_poly = d[s][1]
                    print(f"{s:>6}  {C_nl_arr[idx]:+10.6f}  {C_poly:+12.6f}  {C_mpf_pm[idx]:+12.6f}  "
                          f"{abs(C_poly-C_nl_arr[idx]):10.2e}  {delta_pm[idx]:10.2e}")

                # Plot comparison
                if HAS_MPL:
                    fig3, ax3 = plt.subplots(figsize=(10,6))
                    ax3.plot(N_steps, np.abs(C_nl_arr), 'k-', lw=2.5, label='|C_nl| (noiseless)')
                    ax3.plot(N_steps, np.abs([d[s][1] for s in N_steps]), 'b--', lw=2,
                             label='|C_mpf| polynomial fit')
                    ax3.plot(N_steps, np.abs(C_mpf_pm), 'r-', lw=2,
                             label="|C_mpf| poor man's (Z(t) corr)")
                    ax3.set_xlabel('Cycle', fontsize=13)
                    ax3.set_ylabel('|C(t)|', fontsize=13)
                    ax3.set_title(f"Poor man's MPF vs polynomial (chi={best_chi}, cut={best_cutoff})", fontsize=13)
                    ax3.legend(fontsize=10)
                    ax3.grid(True, alpha=0.3)
                    fig3.tight_layout()
                    fig3.savefig(f"{log_dir}/plot_poorman_mpf.png", dpi=150)
                    plt.close(fig3)
                    print(f"Written: {log_dir}/plot_poorman_mpf.png")
            except np.linalg.LinAlgError as e:
                print(f"Linear system singular: {e}")
