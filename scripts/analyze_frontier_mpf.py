#!/usr/bin/env python3
"""
analyze_frontier_mpf.py — Frontier-weight MPF for spatial Pauli weight.

Instead of using C(t) at the center for MPF coefficients, uses the full
spatial field n(v,t) at sites where n_nl(v,t) > threshold.

Physical motivation:
  - No chi error at frontier (chi≈1, simulation exact)
  - Vector of observables → better conditioned than scalar C(t)
  - Threshold filters out noise floor
  - c_i correct the full spatial pattern, not just autocorrelation

Formulation:
  A_ij = sum_{v,t: n_nl>eps} n_ni(v,t) * n_nj(v,t)
  B_i  = sum_{v,t: n_nl>eps} n_ni(v,t) * n_nl(v,t)
  min ||Ac - B||^2  s.t. sum c_i = 1

Usage:
  python3 scripts/analyze_frontier_mpf.py \
      --log logs/spatial_mpf_chi70_1e-3.log \
      --graph data/graphs/heavy_hex_3x3.json \
      --center 13 --threshold 1e-4 \
      --out logs/frontier_mpf_chi70.png
"""

import argparse, re, json, numpy as np
from collections import deque
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_GAMMA = 3

def parse_log(path):
    steps = {}
    with open(path) as f:
        for line in f:
            # Format 1: main_heavyhex_gamma_mpf_2gpu: C_n x y z
            m = re.match(r"STEP (\d+)\s+C_nl ([+-][\d.e+-]+)\s+C_n ([+-][\d.e+-]+) ([+-][\d.e+-]+) ([+-][\d.e+-]+)", line)
            if m:
                s = int(m.group(1))
                if s not in steps: steps[s] = {}
                steps[s]['C_nl'] = float(m.group(2))
                steps[s]['C_n']  = np.array([float(m.group(i)) for i in [3,4,5]])
            # Format 2: main_pauli_sweep: C_n1 x C_n2 y C_n3 z
            m = re.match(r"STEP (\d+)\s+C_nl ([+-][\d.e+-]+)\s+C_n1 ([+-][\d.e+-]+)\s+C_n2 ([+-][\d.e+-]+)\s+C_n3 ([+-][\d.e+-]+)", line)
            if m:
                s = int(m.group(1))
                if s not in steps: steps[s] = {}
                steps[s]['C_nl'] = float(m.group(2))
                steps[s]['C_n']  = np.array([float(m.group(i)) for i in [3,4,5]])
            m = re.match(r"PAULI_NL (\d+)(.*)", line)
            if m:
                s = int(m.group(1))
                if s not in steps: steps[s] = {}
                steps[s]['N_nl'] = np.array([float(x) for x in m.group(2).split()])
            for g in range(1, N_GAMMA+1):
                m = re.match(rf"PAULI_N{g} (\d+)(.*)", line)
                if m:
                    s = int(m.group(1))
                    if s not in steps: steps[s] = {}
                    if 'N_n' not in steps[s]: steps[s]['N_n'] = [None]*N_GAMMA
                    steps[s]['N_n'][g-1] = np.array([float(x) for x in m.group(2).split()])
    return steps

def load_graph(path):
    with open(path) as f: d = json.load(f)
    ns = d.get('n_sites', len(d.get('vertices',[])))
    adj = [[] for _ in range(ns)]
    for e in d['edges']:
        u=e['edge'][0][0]-1; v=e['edge'][1][0]-1
        adj[u].append(v); adj[v].append(u)
    return ns, adj

def bfs(adj, center, ns):
    dist=[-1]*ns; dist[center]=0; q=deque([center])
    while q:
        v=q.popleft()
        for u in adj[v]:
            if dist[u]<0: dist[u]=dist[v]+1; q.append(u)
    return dist

def frontier(N_vt, dist, threshold):
    """Frontier distance per cycle."""
    f = []
    for t in range(N_vt.shape[0]):
        active = [dist[v] for v in range(N_vt.shape[1]) if N_vt[t,v] > threshold]
        f.append(max(active) if active else 0)
    return np.array(f)

def cycle_frontier_coeffs(N_nl_t, N_n_t, dist, f_nl_t, threshold):
    """
    Per-cycle coefficients using ONLY frontier shell sites at cycle t.
    Frontier shell: sites where n_nl > threshold AND d(v) == F_nl(t).
    """
    mask = np.array([dist[v] == f_nl_t and N_nl_t[v] > threshold
                     for v in range(len(dist))])
    n_frontier = mask.sum()
    if n_frontier == 0:
        return np.array([1./N_GAMMA]*N_GAMMA)

    A = np.zeros((N_GAMMA, N_GAMMA))
    B = np.zeros(N_GAMMA)
    for i in range(N_GAMMA):
        B[i] = np.sum(N_n_t[i][mask] * N_nl_t[mask])
        for j in range(N_GAMMA):
            A[i,j] = np.sum(N_n_t[i][mask] * N_n_t[j][mask])

    M = np.zeros((N_GAMMA+1, N_GAMMA+1))
    M[:N_GAMMA,:N_GAMMA] = A; M[:N_GAMMA,N_GAMMA] = 1; M[N_GAMMA,:N_GAMMA] = 1
    rhs = np.zeros(N_GAMMA+1); rhs[:N_GAMMA] = B; rhs[N_GAMMA] = 1
    try:
        sol = np.linalg.solve(M, rhs)
        c = sol[:N_GAMMA]
        if np.any(np.abs(c) > 100): c = np.array([1./N_GAMMA]*N_GAMMA)
    except:
        c = np.array([1./N_GAMMA]*N_GAMMA)
    return c

def frontier_mpf_coeffs(N_nl, N_n, threshold):
    """
    Compute MPF coefficients from frontier weights.
    A_ij = sum_{v,t: N_nl>eps} N_ni(v,t) * N_nj(v,t)
    B_i  = sum_{v,t: N_nl>eps} N_ni(v,t) * N_nl(v,t)
    """
    n_steps, n_sites = N_nl.shape
    mask = N_nl > threshold  # [n_steps, n_sites]

    A = np.zeros((N_GAMMA, N_GAMMA))
    B = np.zeros(N_GAMMA)

    for i in range(N_GAMMA):
        B[i] = np.sum(N_n[i][mask] * N_nl[mask])
        for j in range(N_GAMMA):
            A[i,j] = np.sum(N_n[i][mask] * N_n[j][mask])

    # Lagrange: min ||Ac-B||^2 s.t. sum c = 1
    M = np.zeros((N_GAMMA+1, N_GAMMA+1))
    M[:N_GAMMA,:N_GAMMA] = A; M[:N_GAMMA,N_GAMMA] = 1; M[N_GAMMA,:N_GAMMA] = 1
    rhs = np.zeros(N_GAMMA+1); rhs[:N_GAMMA] = B; rhs[N_GAMMA] = 1
    try:
        sol = np.linalg.solve(M, rhs)
        c = sol[:N_GAMMA]
        # Clip if wildly ill-conditioned
        if np.any(np.abs(c) > 100): c = np.array([1./N_GAMMA]*N_GAMMA)
    except:
        c = np.array([1./N_GAMMA]*N_GAMMA)
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log',    required=True)
    ap.add_argument('--graph',  default='data/graphs/heavy_hex_3x3.json')
    ap.add_argument('--center', type=int, default=13)
    ap.add_argument('--threshold', type=float, default=1e-4)
    ap.add_argument('--out',    default=None)
    args = ap.parse_args()

    steps = parse_log(args.log)
    ns, adj = load_graph(args.graph)
    dist = bfs(adj, args.center, ns)
    max_dist = max(dist)

    step_list = sorted(s for s in steps
                       if 'N_nl' in steps[s] and 'N_n' in steps[s]
                       and all(x is not None for x in steps[s]['N_n']))
    if not step_list:
        print("No spatial data found"); return
    print(f"Loaded {len(step_list)} steps with spatial data")

    N_nl = np.array([steps[s]['N_nl'] for s in step_list])  # [T, V]
    N_n  = [np.array([steps[s]['N_n'][g] for s in step_list]) for g in range(N_GAMMA)]

    # ── Frontier-weight MPF coefficients ──────────────────────────────────────
    c = frontier_mpf_coeffs(N_nl, N_n, args.threshold)
    print(f"\nFrontier MPF coefficients: c = [{c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f}]  sum={c.sum():.4f}")

    # ── Apply coefficients spatially ──────────────────────────────────────────
    N_mpf = np.zeros_like(N_nl)
    for g in range(N_GAMMA):
        N_mpf += c[g] * N_n[g]
    N_mpf = np.clip(N_mpf, 0, None)

    # ── C(t) poor man's MPF for comparison ───────────────────────────────────
    Ac = np.zeros((N_GAMMA, N_GAMMA)); Bc = np.zeros(N_GAMMA)
    c_pm_list = []
    for i, s in enumerate(step_list):
        if 'C_n' not in steps[s] or 'C_nl' not in steps[s]:
            c_pm_list.append(np.array([1./N_GAMMA]*N_GAMMA)); continue
        cn = steps[s]['C_n']; cnl = steps[s]['C_nl']
        for ii in range(N_GAMMA):
            Bc[ii] += cn[ii]*cnl
            for jj in range(N_GAMMA): Ac[ii,jj] += cn[ii]*cn[jj]
        try:
            M = np.zeros((N_GAMMA+1,N_GAMMA+1))
            M[:N_GAMMA,:N_GAMMA]=Ac; M[:N_GAMMA,N_GAMMA]=1; M[N_GAMMA,:N_GAMMA]=1
            rhs=np.zeros(N_GAMMA+1); rhs[:N_GAMMA]=Bc; rhs[N_GAMMA]=1
            c_pm = np.linalg.solve(M,rhs)[:N_GAMMA]
        except: c_pm = np.array([1./N_GAMMA]*N_GAMMA)
        c_pm_list.append(c_pm)

    N_pm = np.zeros_like(N_nl)
    for i in range(len(step_list)):
        for g in range(N_GAMMA):
            N_pm[i] += c_pm_list[i][g] * N_n[g][i]
    N_pm = np.clip(N_pm, 0, None)

    # ── Per-cycle frontier coefficients ───────────────────────────────────────
    f_nl  = frontier(N_nl,  dist, args.threshold)
    f_n   = [frontier(N_n[g], dist, args.threshold) for g in range(N_GAMMA)]
    f_mpf = frontier(N_mpf, dist, args.threshold)
    f_pm  = frontier(N_pm,  dist, args.threshold)

    # Per-cycle: c_i(t) from frontier shell only, apply to get corrected frontier
    N_cycle_fmpf = np.zeros_like(N_nl)
    c_per_cycle = []
    for i, s in enumerate(step_list):
        N_n_t = [N_n[g][i] for g in range(N_GAMMA)]
        c_t = cycle_frontier_coeffs(N_nl[i], N_n_t, dist, int(f_nl[i]), args.threshold)
        c_per_cycle.append(c_t)
        for g in range(N_GAMMA):
            N_cycle_fmpf[i] += c_t[g] * N_n[g][i]
    N_cycle_fmpf = np.clip(N_cycle_fmpf, 0, None)
    f_cycle_fmpf = frontier(N_cycle_fmpf, dist, args.threshold)

    # ── Deficit formulation: δ_i(t) = F_nl(t) - F_ni(t) ─────────────────────
    # MPF on deficits: find c_i s.t. sum c_i * δ_i(t) ≈ 0 (recover noiseless)
    # Equivalent: F_corrected = F_nl - sum c_i * δ_i  (deficit cancellation)
    delta_n = [f_nl - f_n[g] for g in range(N_GAMMA)]  # [N_GAMMA][T]

    # Fit: c_i minimize ||sum c_i * delta_i||^2 s.t. sum c_i = 1
    # Build system from deficits
    T = len(step_list)
    A_def = np.zeros((N_GAMMA, N_GAMMA))
    B_def = np.zeros(N_GAMMA)
    for i in range(N_GAMMA):
        B_def[i] = np.sum(delta_n[i] * f_nl)  # want sum c_i delta_i = 0 → B = 0
        for j in range(N_GAMMA):
            A_def[i,j] = np.sum(delta_n[i] * delta_n[j])

    # Min ||A_def c||^2 s.t. sum c = 1
    # → A_def^T A_def c + lambda*1 = 0, sum c = 1
    try:
        M = np.zeros((N_GAMMA+1, N_GAMMA+1))
        M[:N_GAMMA,:N_GAMMA] = A_def.T @ A_def
        M[:N_GAMMA,N_GAMMA] = 1; M[N_GAMMA,:N_GAMMA] = 1
        rhs = np.zeros(N_GAMMA+1); rhs[N_GAMMA] = 1
        c_def = np.linalg.solve(M, rhs)[:N_GAMMA]
        if np.any(np.abs(c_def) > 50): c_def = np.array([1./N_GAMMA]*N_GAMMA)
    except:
        c_def = np.array([1./N_GAMMA]*N_GAMMA)

    # Corrected frontier: F_nl - sum c_i * delta_i (should ≈ F_nl if deficit cancelled)
    f_deficit = np.array([f_nl[i] - sum(c_def[g]*delta_n[g][i] for g in range(N_GAMMA))
                          for i in range(T)])
    f_deficit = np.clip(np.round(f_deficit).astype(int), 0, max_dist)

    print(f"\nDeficit MPF coefficients: c = [{c_def[0]:+.4f}, {c_def[1]:+.4f}, {c_def[2]:+.4f}]  sum={c_def.sum():.4f}")
    print(f"Mean deficit: δ1={delta_n[0].mean():.2f}  δ2={delta_n[1].mean():.2f}  δ3={delta_n[2].mean():.2f}")

    # ── LR-constrained deficit MPF ─────────────────────────────────────────────
    # Correct LR bound: F_corrected(t) <= F_nl(t)  (noiseless IS the causal ceiling)
    # i.e. F_nl(t) - sum c_i * delta_i(t) <= F_nl(t)
    # i.e. sum c_i * delta_i(t) >= 0  for all t  (correction cannot exceed noiseless)
    # Also: F_corrected(t) >= 0  (non-negative frontier)
    # i.e. sum c_i * delta_i(t) <= F_nl(t)  for all t
    v_hop = 0.253  # stored for display only

    def objective(c):
        return sum((sum(c[g]*delta_n[g][i] for g in range(N_GAMMA)))**2
                   for i in range(T))

    constraints = [
        {'type': 'eq', 'fun': lambda c: c.sum() - 1},  # sum c_i = 1
    ]
    for i in range(T):
        idx = i
        # delta >= 0: correction cannot make F_corrected > F_nl
        constraints.append({
            'type': 'ineq',
            'fun': lambda c, ii=idx: sum(c[g]*delta_n[g][ii] for g in range(N_GAMMA))
        })
        # delta <= F_nl: corrected frontier >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda c, ii=idx: f_nl[ii] - sum(c[g]*delta_n[g][ii] for g in range(N_GAMMA))
        })

    try:
        res = minimize(objective, c_def, method='SLSQP', constraints=constraints,
                       options={'ftol':1e-10, 'maxiter':1000})
        c_lr = res.x if res.success else c_def
        if np.any(np.abs(c_lr) > 50): c_lr = c_def
    except:
        c_lr = c_def

    f_lr = np.array([f_nl[i] - sum(c_lr[g]*delta_n[g][i] for g in range(N_GAMMA))
                     for i in range(T)])
    f_lr = np.clip(np.round(f_lr).astype(int), 0, max_dist)

    print(f"LR-constrained deficit: c = [{c_lr[0]:+.4f}, {c_lr[1]:+.4f}, {c_lr[2]:+.4f}]  sum={c_lr.sum():.4f}")
    print(f"Ceiling F_nl(t):        {list(f_nl.astype(int))}")

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'Cycle':>5}  {'F_nl':>5}  {'F_n1':>5}  {'F_n2':>5}  {'F_n3':>5}  "
          f"{'δ1':>4}  {'δ2':>4}  {'δ3':>4}  "
          f"{'F_pm':>6}  {'F_fmpf':>7}  {'F_cycle':>8}  {'F_deficit':>10}  {'F_lr':>6}")
    print("-"*95)
    for i, s in enumerate(step_list):
        print(f"{s:>5}  {f_nl[i]:>5}  {f_n[0][i]:>5}  {f_n[1][i]:>5}  {f_n[2][i]:>5}  "
              f"{delta_n[0][i]:>4}  {delta_n[1][i]:>4}  {delta_n[2][i]:>4}  "
              f"{f_pm[i]:>6}  {f_mpf[i]:>7}  {f_cycle_fmpf[i]:>8}  {f_deficit[i]:>10}  {f_lr[i]:>6}")

    # ── Figure ────────────────────────────────────────────────────────────────
    def heatmap(N, dist, max_dist, step_list):
        data = []
        for t in range(len(step_list)):
            row = np.zeros(max_dist+1)
            for d in range(max_dist+1):
                sites=[v for v in range(ns) if dist[v]==d]
                if sites: row[d]=max(N[t,v] for v in sites)
            data.append(row)
        return np.array(data)

    fig, axes = plt.subplots(2, 3, figsize=(15,8))

    titles = ['Noiseless', 'Noisy γ=1', 'Noisy γ=2',
              'Noisy γ=3', 'PM MPF (C(t))', 'Frontier MPF (n(v,t))']
    maps   = [N_nl, N_n[0], N_n[1], N_n[2], N_pm, N_mpf]

    for k in range(6):
        ax = axes[k//3][k%3]
        hm = heatmap(maps[k], dist, max_dist, step_list)
        hm_log = np.log10(np.clip(hm, 1e-6, None))
        im = ax.imshow(hm_log.T, aspect='auto', origin='lower', cmap='hot',
                       extent=[step_list[0]-0.5, step_list[-1]+0.5, -0.5, max_dist+0.5],
                       vmin=-6, vmax=0)
        ax.set_xlabel('Cycle'); ax.set_ylabel('BFS distance')
        ax.set_title(titles[k])
        cbar = plt.colorbar(im, ax=ax); cbar.set_label('log₁₀ n(v,t)')

    # Replace last panel with frontier comparison
    ax = axes[1][2]
    ax.cla()
    ax.plot(step_list, f_nl,  'k-o', lw=2.5, label='Noiseless')
    ax.plot(step_list, f_n[0],'b-s', lw=1.5, alpha=0.7, label='Noisy γ=1')
    ax.fill_between(step_list,
                    [min(f_n[g][i] for g in range(N_GAMMA)) for i in range(len(step_list))],
                    [max(f_n[g][i] for g in range(N_GAMMA)) for i in range(len(step_list))],
                    alpha=0.15, color='gray', label='Noisy band')
    ax.plot(step_list, f_pm,         'g--^', lw=1.8, label='PM MPF (C(t))')
    ax.plot(step_list, f_mpf,        'm-o',  lw=2,   label=f'Frontier MPF (global)\nc=[{c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}]')
    ax.plot(step_list, f_cycle_fmpf, 'r-D',  lw=2,   label='Per-cycle frontier MPF', zorder=5)
    ax.plot(step_list, f_deficit,    'c-*',  lw=2,   label=f'Deficit MPF\nc=[{c_def[0]:+.2f},{c_def[1]:+.2f},{c_def[2]:+.2f}]', zorder=5)
    ax.plot(step_list, f_lr,         'y-P',  lw=2.5, label=f'LR-constrained deficit\nc=[{c_lr[0]:+.2f},{c_lr[1]:+.2f},{c_lr[2]:+.2f}]', zorder=6)
    ax.plot(step_list, f_nl, 'k:', lw=1.5, alpha=0.4, label='F_nl ceiling (LR bound)')
    ax.set_xlabel('Cycle'); ax.set_ylabel('Frontier distance (hops)')
    ax.set_title('Frontier comparison: PM vs Frontier-weight MPF')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.suptitle(f'Frontier-weight MPF — center={args.center}, threshold={args.threshold}', fontsize=11)
    plt.tight_layout()
    outfile = args.out or args.log.replace('.log','_frontier_mpf.png')
    plt.savefig(outfile, dpi=150)
    print(f"\nFigure: {outfile}")

if __name__ == '__main__':
    main()
