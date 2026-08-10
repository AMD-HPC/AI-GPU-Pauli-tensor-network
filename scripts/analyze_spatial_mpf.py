#!/usr/bin/env python3
"""
analyze_spatial_mpf.py — Spatial MPF correction of Pauli weight n(v,t).

Reads output of main_heavyhex_gamma_mpf_2gpu --measure-all, computes:
  1. Poor man's MPF coefficients c_i(t) from C(t) at center
  2. N_mpf(v,t) = sum_i c_i(t) * N_ni(v,t)  [spatially corrected lightcone]
  3. Frontier velocity for noiseless, noisy, and MPF-corrected
  4. Comparison figure: heatmaps + frontier positions

Usage:
  python3 scripts/analyze_spatial_mpf.py \
      --log logs/spatial_mpf_chi70.log \
      --graph data/graphs/heavy_hex_3x3.json \
      --center 13 \
      --out logs/spatial_mpf_fig.png

Output:
  - Figure: 3×2 grid of heatmaps (noiseless / noisy / MPF) + frontier comparison
  - Table: frontier position per cycle for each state
"""

import argparse
import re
import json
import numpy as np
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_GAMMA = 3

# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_log(path):
    """Parse main_heavyhex_gamma_mpf_2gpu --measure-all output."""
    steps = {}
    with open(path) as f:
        for line in f:
            # STEP line: C_nl, C_n, IP_A, IP_B
            m = re.match(
                r"STEP (\d+)\s+C_nl ([+-][\d.e+-]+)\s+"
                r"C_n ([+-][\d.e+-]+) ([+-][\d.e+-]+) ([+-][\d.e+-]+)",
                line)
            if m:
                s = int(m.group(1))
                if s not in steps: steps[s] = {}
                steps[s]['C_nl'] = float(m.group(2))
                steps[s]['C_n']  = np.array([float(m.group(i)) for i in [3,4,5]])
                continue

            # PAULI_NL line
            m = re.match(r"PAULI_NL (\d+)(.*)", line)
            if m:
                s = int(m.group(1))
                if s not in steps: steps[s] = {}
                steps[s]['N_nl'] = np.array([float(x) for x in m.group(2).split()])
                continue

            # PAULI_N1/N2/N3 lines
            for g in range(1, N_GAMMA+1):
                m = re.match(rf"PAULI_N{g} (\d+)(.*)", line)
                if m:
                    s = int(m.group(1))
                    if s not in steps: steps[s] = {}
                    if 'N_n' not in steps[s]: steps[s]['N_n'] = [None]*N_GAMMA
                    steps[s]['N_n'][g-1] = np.array([float(x) for x in m.group(2).split()])
    return steps

def load_graph(path):
    with open(path) as f:
        d = json.load(f)
    if 'n_sites' in d:
        n_sites = d['n_sites']
    elif 'vertices' in d:
        n_sites = len(d['vertices'])
    else:
        all_v = set()
        for e in d['edges']:
            all_v.add(e['edge'][0][0]-1); all_v.add(e['edge'][1][0]-1)
        n_sites = max(all_v)+1
    adj = [[] for _ in range(n_sites)]
    for e in d['edges']:
        u=e['edge'][0][0]-1; v=e['edge'][1][0]-1
        adj[u].append(v); adj[v].append(u)
    return n_sites, adj

def bfs_distances(adj, center, n_sites):
    dist = [-1]*n_sites; dist[center]=0
    q = deque([center])
    while q:
        v=q.popleft()
        for u in adj[v]:
            if dist[u]<0: dist[u]=dist[v]+1; q.append(u)
    return dist

# ── Poor man's MPF ────────────────────────────────────────────────────────────

def poor_mans_coefficients(steps):
    """Accumulate poor man's MPF coefficients cumulatively."""
    A_cum = np.zeros((N_GAMMA, N_GAMMA))
    B_cum = np.zeros(N_GAMMA)
    coeffs = {}
    for s in sorted(steps):
        d = steps[s]
        if 'C_nl' not in d or 'C_n' not in d: continue
        C_nl = d['C_nl']; C_n = d['C_n']
        for i in range(N_GAMMA):
            B_cum[i] += C_n[i]*C_nl
            for j in range(N_GAMMA): A_cum[i,j] += C_n[i]*C_n[j]
        try:
            M = np.zeros((N_GAMMA+1, N_GAMMA+1))
            M[:N_GAMMA,:N_GAMMA]=A_cum; M[:N_GAMMA,N_GAMMA]=1; M[N_GAMMA,:N_GAMMA]=1
            rhs = np.zeros(N_GAMMA+1); rhs[:N_GAMMA]=B_cum; rhs[N_GAMMA]=1
            c = np.linalg.solve(M, rhs)[:N_GAMMA]
            coeffs[s] = c
        except:
            coeffs[s] = np.array([1./N_GAMMA]*N_GAMMA)
    return coeffs

# ── Frontier extraction ────────────────────────────────────────────────────────

def frontier(N_vt, dist, max_dist, threshold=1e-3):
    """Per-cycle frontier: max distance with n(v,t) > threshold."""
    frontiers = []
    for t_idx in range(N_vt.shape[0]):
        active = [dist[v] for v in range(len(dist)) if N_vt[t_idx,v] > threshold]
        frontiers.append(max(active) if active else 0)
    return np.array(frontiers)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log',    required=True)
    ap.add_argument('--graph',  default='data/graphs/heavy_hex_3x3.json')
    ap.add_argument('--center', type=int, default=13)
    ap.add_argument('--threshold', type=float, default=1e-3)
    ap.add_argument('--out',    default=None)
    args = ap.parse_args()

    steps = parse_log(args.log)
    n_sites, adj = load_graph(args.graph)
    dist = bfs_distances(adj, args.center, n_sites)
    max_dist = max(dist)

    step_list = sorted(s for s in steps
                       if 'N_nl' in steps[s] and 'N_n' in steps[s]
                       and all(x is not None for x in steps[s]['N_n']))
    print(f"Loaded {len(step_list)} steps with spatial data")

    coeffs = poor_mans_coefficients(steps)

    # Build arrays: [n_steps, n_sites]
    N_nl  = np.array([steps[s]['N_nl'] for s in step_list])
    N_n   = [np.array([steps[s]['N_n'][g] for s in step_list]) for g in range(N_GAMMA)]
    N_mpf = np.zeros_like(N_nl)
    for i, s in enumerate(step_list):
        c = coeffs.get(s, np.array([1./N_GAMMA]*N_GAMMA))
        for g in range(N_GAMMA):
            N_mpf[i] += c[g] * steps[s]['N_n'][g]
        N_mpf[i] = np.clip(N_mpf[i], 0, None)

    # Frontier per cycle
    f_nl  = frontier(N_nl,  dist, max_dist, args.threshold)
    f_n   = [frontier(N_n[g], dist, max_dist, args.threshold) for g in range(N_GAMMA)]
    f_mpf = frontier(N_mpf, dist, max_dist, args.threshold)

    # Print table
    print(f"\n{'Cycle':>5}  {'F_nl':>5}  {'F_n1':>5}  {'F_n2':>5}  {'F_n3':>5}  {'F_mpf':>6}  {'c1':>6}  {'c2':>6}  {'c3':>6}")
    print("-"*75)
    for i, s in enumerate(step_list):
        c = coeffs.get(s, [0,0,0])
        print(f"{s:>5}  {f_nl[i]:>5}  {f_n[0][i]:>5}  {f_n[1][i]:>5}  {f_n[2][i]:>5}  "
              f"{f_mpf[i]:>6}  {c[0]:>6.3f}  {c[1]:>6.3f}  {c[2]:>6.3f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    # Heatmap: max n(v,t) per distance shell vs cycle
    def make_heatmap(N_vt, dist, max_dist, step_list):
        data = []
        for t in range(len(step_list)):
            row = np.zeros(max_dist+1)
            for d_val in range(max_dist+1):
                sites = [v for v in range(n_sites) if dist[v]==d_val]
                if sites: row[d_val] = max(N_vt[t,v] for v in sites)
            data.append(row)
        return np.array(data)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    titles = ['Noiseless', 'Noisy γ=1', 'Noisy γ=2', 'Noisy γ=3', 'MPF corrected', 'Frontier comparison']
    maps_to_plot = [N_nl, N_n[0], N_n[1], N_n[2], N_mpf]

    for k in range(5):
        ax = axes[k//3][k%3]
        hm = make_heatmap(maps_to_plot[k], dist, max_dist, step_list)
        # Log scale: clip small values to avoid log(0)
        hm_log = np.log10(np.clip(hm, 1e-6, None))
        im = ax.imshow(hm_log.T, aspect='auto', origin='lower', cmap='hot',
                       extent=[step_list[0]-0.5, step_list[-1]+0.5, -0.5, max_dist+0.5],
                       vmin=-6, vmax=0)
        ax.set_xlabel('Cycle'); ax.set_ylabel('BFS distance from center')
        ax.set_title(titles[k])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('log₁₀ n(v,t)')

    # Frontier comparison panel
    ax = axes[1][2]
    ax.plot(step_list, f_nl,  'k-o', lw=2,   label='Noiseless')
    ax.plot(step_list, f_n[0],'b-s', lw=1.5, label='Noisy γ=1', alpha=0.7)
    ax.plot(step_list, f_n[1],'r-s', lw=1.5, label='Noisy γ=2', alpha=0.7)
    ax.fill_between(step_list,
                    [min(f_n[g][i] for g in range(N_GAMMA)) for i in range(len(step_list))],
                    [max(f_n[g][i] for g in range(N_GAMMA)) for i in range(len(step_list))],
                    alpha=0.15, color='gray', label='Noisy band')
    ax.plot(step_list, f_mpf, 'm-^', lw=2, label='MPF corrected', zorder=5)
    ax.set_xlabel('Cycle'); ax.set_ylabel('Frontier distance (hops)')
    ax.set_title('Frontier: noiseless vs noisy vs MPF')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle(f'Spatial MPF correction — center={args.center}, threshold={args.threshold}', fontsize=12)
    plt.tight_layout()
    outfile = args.out or args.log.replace('.log', '_spatial_mpf.png')
    plt.savefig(outfile, dpi=150)
    print(f"\nFigure: {outfile}")

if __name__ == '__main__':
    main()
