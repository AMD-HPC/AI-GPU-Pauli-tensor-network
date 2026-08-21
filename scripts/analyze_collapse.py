#!/usr/bin/env python3
"""
analyze_collapse.py — Diagnose the chi-saturation C(t) collapse via N(v,t).

Reads PAULI_NL + STEP lines from main_pauli_sweep output.
Computes BFS distances from center, bins weight by distance, checks
conservation and causal spread at each step — especially the collapse step.

Key diagnostic questions:
  1. Is Σ N(v,t) conserved across steps? (blow-up/vanish → ψ corrupted upstream)
  2. Does weight appear outside the causal cone? (yes → ψ corrupted, not BP)
  3. Is the transition to collapse sudden or gradual?
  4. Where does center-site weight drain TO at collapse?

Usage:
  python3 scripts/analyze_collapse.py --log logs/e0_float_380.log
  python3 scripts/analyze_collapse.py --log logs/e0_float_380.log --log2 logs/e0_double_380.log
  python3 scripts/analyze_collapse.py --log logs/e0_float_380.log --center 13 --graph data/graphs/heavy_hex_3x3.json
"""

import argparse, json, re, sys
import numpy as np
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Graph utilities ──────────────────────────────────────────────────────────

def load_adj(graph_path, n_sites):
    g = json.load(open(graph_path))
    adj = [[] for _ in range(n_sites)]
    for e in g['edges']:
        a = e['edge'][0][0] - 1  # 1-based in JSON
        b = e['edge'][1][0] - 1
        adj[a].append(b)
        adj[b].append(a)
    return adj

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
    return dist

# ── Parser ───────────────────────────────────────────────────────────────────

def parse_log(path):
    """Returns list of (step, C_nl, chi, pauli_nl) dicts."""
    steps = []
    cur = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            m = re.match(r'STEP (\d+)\s+C_nl ([+-][\d.e+-]+).*?chi (\d+)', line)
            if m:
                if cur: steps.append(cur)
                cur = {'step': int(m.group(1)),
                       'C_nl': float(m.group(2)),
                       'chi':  int(m.group(3)),
                       'pauli': None}
            m2 = re.match(r'PAULI_NL (\d+)\s+(.*)', line)
            if m2 and cur:
                weights = list(map(float, m2.group(2).split()))
                cur['pauli'] = np.array(weights)
    if cur: steps.append(cur)
    return steps

# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze(steps, dist, label, v_hop=0.5, collapse_thresh=0.3):
    """Print per-step diagnostics and return summary arrays."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"{'step':>4}  {'C_nl':>9}  {'chi':>5}  {'ΣN':>10}  {'Δ(ΣN)':>10}  "
          f"{'N_center':>10}  {'N_inside':>9}  {'N_outside':>10}  {'collapse?':>9}")
    print('-'*100)

    n_sites = len(dist)
    max_dist = max(d for d in dist if d >= 0)
    prev_sum = None
    results = []

    for s in steps:
        step = s['step']
        C_nl = s['C_nl']
        chi  = s['chi']
        w    = s['pauli']
        if w is None or len(w) != n_sites:
            continue

        total = float(w.sum())
        delta_sum = (total - prev_sum) if prev_sum is not None else 0.0
        prev_sum = total

        # Causal cone: sites reachable by lightcone at this step
        cone_mask = np.array([dist[v] <= v_hop * step for v in range(n_sites)])
        N_inside  = float(w[cone_mask].sum())
        N_outside = float(w[~cone_mask].sum())
        N_center  = float(w[dist == 0].sum()) if any(d == 0 for d in dist) else 0.0

        collapsed = abs(C_nl) < collapse_thresh

        print(f"{step:>4}  {C_nl:>+9.6f}  {chi:>5}  {total:>10.4e}  {delta_sum:>+10.4e}  "
              f"{N_center:>10.4e}  {N_inside:>9.4e}  {N_outside:>10.4e}  "
              f"{'*** COLLAPSE' if collapsed else '':>9}")

        # Per-distance profile
        by_dist = {}
        for v in range(n_sites):
            d = dist[v]
            by_dist[d] = by_dist.get(d, 0.0) + w[v]

        results.append({
            'step': step, 'C_nl': C_nl, 'chi': chi,
            'total': total, 'delta_sum': delta_sum,
            'N_center': N_center, 'N_inside': N_inside, 'N_outside': N_outside,
            'by_dist': by_dist, 'collapsed': collapsed
        })

    # Flag the transition
    print()
    for i, r in enumerate(results):
        if r['collapsed'] and i > 0 and not results[i-1]['collapsed']:
            prev = results[i-1]
            curr = r
            print(f">>> COLLAPSE TRANSITION: step {prev['step']}→{curr['step']}")
            print(f"    C_nl:     {prev['C_nl']:+.6f} → {curr['C_nl']:+.6f}")
            print(f"    chi:      {prev['chi']} → {curr['chi']}")
            print(f"    ΣN:       {prev['total']:.4e} → {curr['total']:.4e}  "
                  f"(ratio {curr['total']/max(prev['total'],1e-30):.3f})")
            print(f"    N_center: {prev['N_center']:.4e} → {curr['N_center']:.4e}  "
                  f"(ratio {curr['N_center']/max(prev['N_center'],1e-30):.3f})")
            print(f"    N_outside:{prev['N_outside']:.4e} → {curr['N_outside']:.4e}  "
                  f"(causal violation: {'YES ***' if curr['N_outside']>1e-4 else 'no'})")
            print(f"    Weight by BFS distance at collapse step {curr['step']}:")
            for d in sorted(curr['by_dist']):
                bar = '█' * int(curr['by_dist'][d] / max(curr['total'], 1e-30) * 40)
                print(f"      dist {d:2d}: {curr['by_dist'][d]:.4e}  {bar}")
            print(f"    Weight by BFS distance BEFORE collapse (step {prev['step']}):")
            for d in sorted(prev['by_dist']):
                bar = '█' * int(prev['by_dist'][d] / max(prev['total'], 1e-30) * 40)
                print(f"      dist {d:2d}: {prev['by_dist'][d]:.4e}  {bar}")

    return results

# ── Comparison plot ──────────────────────────────────────────────────────────

def plot_comparison(res1, res2, label1, label2, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Collapse diagnostic: {label1} vs {label2}', fontsize=12)

    steps1 = [r['step'] for r in res1]
    steps2 = [r['step'] for r in res2]

    def mark_collapse(ax, res, steps):
        for i, r in enumerate(res):
            if r['collapsed'] and (i == 0 or not res[i-1]['collapsed']):
                ax.axvline(r['step'], color='r', linestyle='--', alpha=0.5, label='collapse')

    # C(t)
    ax = axes[0, 0]
    ax.plot(steps1, [r['C_nl'] for r in res1], 'b-o', label=label1, ms=4)
    ax.plot(steps2, [r['C_nl'] for r in res2], 'g-s', label=label2, ms=4)
    mark_collapse(ax, res1, steps1); mark_collapse(ax, res2, steps2)
    ax.set_title('C(t)'); ax.set_xlabel('step'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ΣN(v,t) — conservation
    ax = axes[0, 1]
    ax.plot(steps1, [r['total'] for r in res1], 'b-o', label=label1, ms=4)
    ax.plot(steps2, [r['total'] for r in res2], 'g-s', label=label2, ms=4)
    mark_collapse(ax, res1, steps1); mark_collapse(ax, res2, steps2)
    ax.set_title('Σ N(v,t) — conservation'); ax.set_xlabel('step'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # N_center
    ax = axes[0, 2]
    ax.semilogy(steps1, [max(r['N_center'],1e-8) for r in res1], 'b-o', label=label1, ms=4)
    ax.semilogy(steps2, [max(r['N_center'],1e-8) for r in res2], 'g-s', label=label2, ms=4)
    mark_collapse(ax, res1, steps1); mark_collapse(ax, res2, steps2)
    ax.set_title('N_center (log)'); ax.set_xlabel('step'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # N_outside — causal probe
    ax = axes[1, 0]
    ax.semilogy(steps1, [max(r['N_outside'],1e-10) for r in res1], 'b-o', label=label1, ms=4)
    ax.semilogy(steps2, [max(r['N_outside'],1e-10) for r in res2], 'g-s', label=label2, ms=4)
    ax.axhline(1e-4, color='r', linestyle=':', label='causal threshold')
    mark_collapse(ax, res1, steps1); mark_collapse(ax, res2, steps2)
    ax.set_title('N_outside lightcone (causal probe)'); ax.set_xlabel('step'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # chi
    ax = axes[1, 1]
    ax.plot(steps1, [r['chi'] for r in res1], 'b-o', label=label1, ms=4)
    ax.plot(steps2, [r['chi'] for r in res2], 'g-s', label=label2, ms=4)
    mark_collapse(ax, res1, steps1); mark_collapse(ax, res2, steps2)
    ax.set_title('chi (bond dim)'); ax.set_xlabel('step'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # N_inside
    ax = axes[1, 2]
    ax.plot(steps1, [r['N_inside'] for r in res1], 'b-o', label=label1, ms=4)
    ax.plot(steps2, [r['N_inside'] for r in res2], 'g-s', label=label2, ms=4)
    mark_collapse(ax, res1, steps1); mark_collapse(ax, res2, steps2)
    ax.set_title('N_inside lightcone'); ax.set_xlabel('step'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved: {out_path}")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log',    required=True, help='Primary log (float)')
    ap.add_argument('--log2',   default=None,  help='Secondary log (double) for comparison')
    ap.add_argument('--graph',  default='data/graphs/heavy_hex_3x3.json')
    ap.add_argument('--center', type=int, default=13)
    ap.add_argument('--v_hop',  type=float, default=0.5, help='v_hop used in run (for causal cone)')
    ap.add_argument('--collapse_thresh', type=float, default=0.3, help='|C_nl| below this = collapsed')
    ap.add_argument('--out',    default=None)
    args = ap.parse_args()

    steps1 = parse_log(args.log)
    n_sites = len(steps1[0]['pauli']) if steps1 and steps1[0]['pauli'] is not None else 68

    adj = load_adj(args.graph, n_sites)
    dist = bfs_dist(adj, args.center)

    label1 = args.log.split('/')[-1].replace('.log', '')
    res1 = analyze(steps1, dist, label1, v_hop=args.v_hop, collapse_thresh=args.collapse_thresh)

    if args.log2:
        steps2 = parse_log(args.log2)
        label2 = args.log2.split('/')[-1].replace('.log', '')
        res2 = analyze(steps2, dist, label2, v_hop=args.v_hop, collapse_thresh=args.collapse_thresh)

        out = args.out or args.log.replace('.log', '_vs_double.png')
        plot_comparison(res1, res2, label1, label2, out)
    else:
        out = args.out or args.log.replace('.log', '_collapse.png')
        # Single-log plot
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle(label1)
        steps = [r['step'] for r in res1]
        axes[0].plot(steps, [r['C_nl'] for r in res1], 'b-o', ms=4); axes[0].set_title('C(t)'); axes[0].grid(True, alpha=0.3)
        axes[1].plot(steps, [r['total'] for r in res1], 'b-o', ms=4); axes[1].set_title('Σ N(v,t)'); axes[1].grid(True, alpha=0.3)
        axes[2].semilogy(steps, [max(r['N_outside'],1e-10) for r in res1], 'b-o', ms=4)
        axes[2].axhline(1e-4, color='r', linestyle=':', label='causal threshold')
        axes[2].set_title('N_outside (causal probe)'); axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(out, dpi=150)
        print(f"\nPlot saved: {out}")

if __name__ == '__main__':
    main()
