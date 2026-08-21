#!/usr/bin/env python3
"""
plot_nvt_distribution.py — Compare N(v,t) BFS-distance distributions across runs.

For each step, shows weight(dist) as a curve. Healthy runs should show a smooth
spreading wavefront. Collapsing runs show loop-shell spikes and center drain.

Compares multiple runs (float/double, different chi_max/delta) side by side
to see if distributions stay in the "same family".

Usage:
  python3 scripts/plot_nvt_distribution.py \
      --logs logs/e0_f32_chi380_d1e4.log logs/e1_f32_chi200_d1e4.log \
              logs/e0_f32_chi380_d1e3.log logs/e1_f64_chi200_d1e4.log \
      --labels "f32 χ=380 δ=1e-4" "f32 χ=200 δ=1e-4" \
               "f32 χ=380 δ=1e-3" "f64 χ=200 δ=1e-4" \
      --center 13 --graph data/graphs/heavy_hex_3x3.json \
      --out logs/nvt_distribution_comparison.png
"""

import argparse, json, re
import numpy as np
from collections import deque, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── Graph ────────────────────────────────────────────────────────────────────

def load_adj(path):
    g = json.load(open(path))
    n = g.get('n_sites', len(g.get('vertices', [])))
    adj = [[] for _ in range(n)]
    for e in g['edges']:
        a = e['edge'][0][0] - 1
        b = e['edge'][1][0] - 1
        adj[a].append(b); adj[b].append(a)
    return adj, n

def bfs_dist(adj, center):
    n = len(adj)
    dist = [-1] * n
    dist[center] = 0
    q = deque([center])
    while q:
        v = q.popleft()
        for u in adj[v]:
            if dist[u] < 0:
                dist[u] = dist[v] + 1
                q.append(u)
    return np.array(dist)

def sites_at_dist(dist, d):
    return np.where(dist == d)[0]

# ── Parser ───────────────────────────────────────────────────────────────────

def parse_log(path):
    steps = []
    cur = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            m = re.match(r'STEP (\d+)\s+C_nl ([+-][\d.e+-]+).*?chi (\d+)', line)
            if m:
                if cur and cur.get('pauli') is not None: steps.append(cur)
                cur = {'step': int(m.group(1)),
                       'C_nl': float(m.group(2)),
                       'chi':  int(m.group(3)),
                       'pauli': None}
            m2 = re.match(r'PAULI_NL (\d+)\s+(.*)', line)
            if m2 and cur:
                cur['pauli'] = np.array(list(map(float, m2.group(2).split())))
    if cur and cur.get('pauli') is not None: steps.append(cur)
    return steps

# ── Distribution by BFS distance ─────────────────────────────────────────────

def weight_by_dist(pauli, dist):
    max_d = dist.max()
    profile = np.zeros(max_d + 1)
    count   = np.zeros(max_d + 1)
    for v, w in enumerate(pauli):
        d = dist[v]
        if d >= 0:
            profile[d] += w
            count[d]   += 1
    return profile, count

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs',   nargs='+', required=True)
    ap.add_argument('--labels', nargs='+', default=None)
    ap.add_argument('--center', type=int, default=13)
    ap.add_argument('--graph',  default='data/graphs/heavy_hex_3x3.json')
    ap.add_argument('--out',    default='logs/nvt_distribution.png')
    ap.add_argument('--collapse_thresh', type=float, default=0.3)
    ap.add_argument('--normalize', action='store_true',
                    help='Normalize each profile to sum=1 (shape comparison)')
    args = ap.parse_args()

    adj, n_sites = load_adj(args.graph)
    dist = bfs_dist(adj, args.center)
    max_d = int(dist.max())
    d_vals = np.arange(max_d + 1)

    labels = args.labels or [l.split('/')[-1].replace('.log','') for l in args.logs]
    all_data = [parse_log(l) for l in args.logs]
    n_runs   = len(all_data)
    n_steps  = max(len(d) for d in all_data)

    # Color each run distinctly; mark collapse steps with dashed line
    colors  = ['#e41a1c','#377eb8','#4daf4a','#984ea3',
                '#ff7f00','#a65628','#f781bf','#999999']
    markers = ['o','s','^','D','v','<','>','p']

    # ── Figure 1: per-step panels showing all runs ───────────────────────────
    # One panel per step; each panel shows weight-vs-BFS-dist curves for all runs
    ncols = 5
    nrows = int(np.ceil(n_steps / ncols))
    fig1, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.2, nrows * 2.8),
                              sharex=True)
    axes = axes.flatten()
    fig1.suptitle('N(v,t) by BFS distance — per step comparison', fontsize=13, y=1.01)

    for si in range(n_steps):
        ax = axes[si]
        for ri, (data, label, color, marker) in enumerate(
                zip(all_data, labels, colors, markers)):
            if si >= len(data): continue
            s = data[si]
            profile, _ = weight_by_dist(s['pauli'], dist)
            if args.normalize:
                tot = profile.sum()
                if tot > 0: profile = profile / tot
            collapsed = abs(s['C_nl']) < args.collapse_thresh
            ls = '--' if collapsed else '-'
            lw = 2.5 if collapsed else 1.5
            ax.plot(d_vals, profile, ls=ls, lw=lw, color=color,
                    marker=marker, ms=4, label=label if si == 0 else '')
            if collapsed:
                ax.set_facecolor('#fff0f0')

        ax.set_title(f'step {si+1}', fontsize=9)
        ax.set_xlabel('BFS dist', fontsize=7)
        ax.set_ylabel('weight' + (' (norm)' if args.normalize else ''), fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused panels
    for ax in axes[n_steps:]: ax.set_visible(False)

    handles = [plt.Line2D([0],[0], color=colors[i], marker=markers[i],
                           lw=1.5, ms=5, label=labels[i])
               for i in range(n_runs)]
    handles.append(plt.Line2D([0],[0], color='k', lw=2.5, ls='--',
                               label='collapsed'))
    fig1.legend(handles=handles, loc='lower center',
                ncol=min(n_runs+1, 5), fontsize=8,
                bbox_to_anchor=(0.5, -0.03))
    plt.tight_layout()
    out1 = args.out.replace('.png', '_per_step.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"Saved: {out1}")

    # ── Figure 2: per-run panels — time evolution of the distribution ─────────
    fig2, axes2 = plt.subplots(1, n_runs,
                               figsize=(n_runs * 3.8, 4.5),
                               sharey=True)
    if n_runs == 1: axes2 = [axes2]
    fig2.suptitle('N(v,t) distribution evolution — each run', fontsize=12)

    cmap = cm.get_cmap('plasma')

    for ri, (data, label, ax) in enumerate(zip(all_data, labels, axes2)):
        n = len(data)
        for si, s in enumerate(data):
            profile, _ = weight_by_dist(s['pauli'], dist)
            if args.normalize:
                tot = profile.sum()
                if tot > 0: profile = profile / tot
            collapsed = abs(s['C_nl']) < args.collapse_thresh
            color = cmap(si / max(n-1, 1))
            ls = '--' if collapsed else '-'
            lw = 2.5 if collapsed else 1.2
            alpha = 1.0 if collapsed else 0.6 + 0.4 * si / max(n-1, 1)
            ax.plot(d_vals, profile, color=color, ls=ls, lw=lw,
                    alpha=alpha, label=f'step {s["step"]}' + (' ✗' if collapsed else ''))
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('BFS dist from center')
        if ri == 0: ax.set_ylabel('weight' + (' (norm)' if args.normalize else ''))
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out2 = args.out.replace('.png', '_evolution.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")

    # ── Figure 3: collapse-step zoom — the family comparison ─────────────────
    # For each run: overlay the step BEFORE and AT collapse (or last 2 steps)
    fig3, axes3 = plt.subplots(1, n_runs,
                               figsize=(n_runs * 3.8, 4.0),
                               sharey=False)
    if n_runs == 1: axes3 = [axes3]
    fig3.suptitle('N(v,t) at saturation: step before vs at collapse', fontsize=12)

    for ri, (data, label, ax) in enumerate(zip(all_data, labels, axes3)):
        # Find collapse transition
        collapse_idx = None
        for i, s in enumerate(data):
            if abs(s['C_nl']) < args.collapse_thresh and i > 0 and abs(data[i-1]['C_nl']) >= args.collapse_thresh:
                collapse_idx = i
                break

        if collapse_idx is not None:
            pairs = [(collapse_idx - 1, 'before', '-', 2.0, '#2171b5'),
                     (collapse_idx,     'collapse', '--', 2.5, '#cb181d')]
            ax.set_title(f'{label}\n(collapse at step {data[collapse_idx]["step"]})', fontsize=8)
        else:
            # No collapse: show last two steps
            pairs = [(len(data)-2, 'step n-1', '-', 1.5, '#2171b5'),
                     (len(data)-1, 'step n',   '-', 2.0, '#238b45')]
            ax.set_title(f'{label}\n(no collapse — final steps)', fontsize=8)

        for idx, desc, ls, lw, color in pairs:
            if idx < 0 or idx >= len(data): continue
            s = data[idx]
            profile, cnt = weight_by_dist(s['pauli'], dist)
            if args.normalize:
                tot = profile.sum()
                if tot > 0: profile = profile / tot
            ax.plot(d_vals, profile, ls=ls, lw=lw, color=color,
                    marker='o', ms=4,
                    label=f'{desc} (step {s["step"]}, C={s["C_nl"]:+.3f}, χ={s["chi"]})')
            # Mark the cone boundary (v_hop=0.5 × step)
            cone_d = 0.5 * s['step']
            ax.axvline(cone_d, color=color, lw=0.8, ls=':', alpha=0.5)

        ax.set_xlabel('BFS dist from center')
        if ri == 0: ax.set_ylabel('weight' + (' (norm)' if args.normalize else ''))
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out3 = args.out.replace('.png', '_saturation_zoom.png')
    plt.savefig(out3, dpi=150, bbox_inches='tight')
    print(f"Saved: {out3}")

    plt.close('all')

if __name__ == '__main__':
    main()
