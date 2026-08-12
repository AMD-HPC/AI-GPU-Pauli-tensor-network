#!/usr/bin/env python3
"""
frontier_wave_map.py — Site-centric wave arrival analysis.

For each site v at lattice position (i,j):
  t_nl(v)    = first cycle where n_nl(v,t) > threshold   (noiseless arrival)
  t_gamma(v) = first cycle where n_gamma(v,t) > threshold (noisy arrival)
  l_gamma(v) = t_gamma(v) - t_nl(v)                      (delay)
  A_nl(v)    = n_nl(v, t_nl(v))                           (noiseless amplitude at arrival)
  A_gamma(v) = n_gamma(v, t_gamma(v))                     (noisy amplitude at arrival)
  dA_gamma(v)= A_nl(v) - A_gamma(v)                       (amplitude difference)

Visualized as 2D lattice maps (colored by delay or amplitude difference).
Sites outside noiseless lightcone: white.
Sites inside noiseless lightcone but noisy wave never arrives: black.

Usage:
  python3 scripts/frontier_wave_map.py \\
      --log logs/spatial_mpf_rect7x7_chi50_1e-3_v2.log \\
      --graph data/graphs/rect_7x7.json \\
      --center 24 --threshold 1e-4 --nx 7 --ny 7

  python3 scripts/frontier_wave_map.py \\
      --log logs/spatial_mpf_chi70_1e-3_v2.log \\
      --graph data/graphs/heavy_hex_3x3.json \\
      --center 13 --threshold 1e-4 --layout heavyhex
"""

import argparse
import re
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import deque

# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_log(path):
    data = {'NL': [], 'N1': [], 'N2': [], 'N3': []}
    with open(path) as f:
        for line in f:
            for g in ['NL', 'N1', 'N2', 'N3']:
                m = re.match(rf'PAULI_{g} (\d+)(.*)', line)
                if m:
                    vals = [float(x) for x in m.group(2).split()]
                    data[g].append(vals)
    return {g: np.array(v) for g, v in data.items() if v}

def bfs(path, center):
    with open(path) as f:
        d = json.load(f)
    ns = d.get('n_sites', len(d.get('vertices', [])))
    adj = [[] for _ in range(ns)]
    for e in d['edges']:
        u = e['edge'][0][0] - 1
        v = e['edge'][1][0] - 1
        adj[u].append(v)
        adj[v].append(u)
    dist = [-1] * ns
    dist[center] = 0
    q = deque([center])
    while q:
        v = q.popleft()
        for u in adj[v]:
            if dist[u] < 0:
                dist[u] = dist[v] + 1
                q.append(u)
    return ns, dist, adj

# ── Core computation ──────────────────────────────────────────────────────────

def arrival_times(N_vt, threshold):
    """
    For each site v, find first cycle t where N_vt[t,v] > threshold.
    Returns array of shape (n_sites,) with -1 where wave never arrives.
    """
    n_cycles, n_sites = N_vt.shape
    t_arr = np.full(n_sites, -1, dtype=int)
    amp   = np.zeros(n_sites)
    for t in range(n_cycles):
        for v in range(n_sites):
            if t_arr[v] < 0 and N_vt[t, v] > threshold:
                t_arr[v] = t + 1   # 1-based cycle
                amp[v]   = N_vt[t, v]
    return t_arr, amp

def site_wave_stats(data, threshold):
    """
    Per-site wave arrival analysis.

    Delay:
        l_gamma(v) = t_gamma(v) - t_nl(v)

    Natural amplitude ratio at noisy arrival:
        r_gamma(v) = n_gamma(v, t_gamma(v)) / n_nl(v, t_gamma(v))

    Both waves compared at the SAME cycle t_gamma(v) — the unique
    intersection where the noisy wave just crosses threshold.
    At that moment:
      n_gamma(v, t_gamma) ≈ threshold  (just arrived)
      n_nl(v, t_gamma)    >> threshold  (has grown for l_gamma cycles)
    The ratio measures how far the noiseless wave outran the noisy one
    during the delay — no free parameter, one natural value per site.
    """
    nl       = data['NL']
    n_cycles = nl.shape[0]
    n_sites  = nl.shape[1]
    t_nl, A_nl = arrival_times(nl, threshold)
    results = {'t_nl': t_nl, 'A_nl': A_nl}

    for g in ['N1', 'N2', 'N3']:
        if g not in data:
            continue
        ni = data[g]
        t_g, A_g = arrival_times(ni, threshold)
        nl_arrived = t_nl >= 0
        g_arrived  = t_g  >= 0

        delay = np.where(nl_arrived & g_arrived, t_g - t_nl, np.nan)

        # ratio at noisy arrival cycle: n_gamma(t_gamma) / n_nl(t_gamma)
        ratio = np.full(n_sites, np.nan)
        for v in range(n_sites):
            if not (nl_arrived[v] and g_arrived[v]):
                continue
            tg = t_g[v] - 1        # 0-based index of noisy arrival cycle
            if tg >= n_cycles:
                continue
            nl_val = nl[tg, v]     # noiseless at that same cycle
            ni_val = ni[tg, v]     # noisy just crossed threshold
            ratio[v] = ni_val / nl_val if nl_val > 0 else np.nan

        results[g] = {'t': t_g, 'A': A_g, 'delay': delay, 'ratio': ratio}
    return results

# ── Layout helpers ────────────────────────────────────────────────────────────

def rect_layout(ns, nx, ny):
    """Site index → (x, y) position for rectangular grid (row-major)."""
    assert nx * ny == ns, f"nx*ny={nx*ny} != n_sites={ns}"
    pos = {}
    for v in range(ns):
        pos[v] = (v % nx, v // nx)
    return pos

def graph_layout(graph_path, ns, dist=None):
    """
    Use (x,y) from graph JSON if available (dict-style vertices with 'coords').
    Falls back to BFS-distance layout using dist[] if provided.
    Returns dict v → (x, y).
    """
    with open(graph_path) as f:
        d = json.load(f)
    pos = {}
    verts = d.get('vertices', [])
    for entry in verts:
        # vertices may be plain lists [id] or dicts with 'coords'
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get('id', [None])
        idx = raw_id[0] if isinstance(raw_id, list) else raw_id
        if idx is None:
            continue
        v = int(idx) - 1
        coords = entry.get('coords', None)
        if coords and len(coords) >= 2:
            pos[v] = (float(coords[0]), float(coords[1]))
    if len(pos) == ns:
        return pos
    # Fallback: layout by BFS distance from center, adjacency order within layer
    if dist is None:
        return None
    max_d = max(dist)
    layer_count = [0] * (max_d + 2)
    pos = {}
    for v in range(ns):
        dv = dist[v] if dist[v] >= 0 else max_d + 1
        pos[v] = (dv, layer_count[dv])
        layer_count[dv] += 1
    return pos

# ── Plotting ──────────────────────────────────────────────────────────────────

def draw_map(ax, stats, key, field, pos, title, cmap, vmin, vmax,
             label_cb, ns):
    """
    Draw a colored scatter map of field values at site positions.
    White = noiseless wave never arrived.
    Black = noiseless arrived but noisy didn't.
    """
    xs, ys, cs = [], [], []
    xs_black, ys_black = [], []
    xs_white, ys_white = [], []

    t_nl = stats['t_nl']
    if key == 'NL':
        # show noiseless arrival time itself
        for v in range(ns):
            x, y = pos[v]
            if t_nl[v] < 0:
                xs_white.append(x); ys_white.append(y)
            else:
                xs.append(x); ys.append(y); cs.append(t_nl[v])
    else:
        g_stats = stats[key]
        vals = g_stats[field]
        t_g  = g_stats['t']
        for v in range(ns):
            x, y = pos[v]
            if t_nl[v] < 0:
                xs_white.append(x); ys_white.append(y)
            elif np.isnan(vals[v]):
                xs_black.append(x); ys_black.append(y)
            else:
                xs.append(x); ys.append(y); cs.append(vals[v])

    ax.scatter(xs_white, ys_white, c='white', edgecolors='lightgray',
               s=120, zorder=2, label='outside NL lightcone')
    if xs_black:
        ax.scatter(xs_black, ys_black, c='black', s=120, zorder=3,
                   label='NL arrived, noisy didn\'t')
    if xs:
        sc = ax.scatter(xs, ys, c=cs, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=120, zorder=4, edgecolors='gray', linewidths=0.3)
        plt.colorbar(sc, ax=ax, label=label_cb, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f0f0')
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)

def summary_stats(vals, name):
    v = vals[~np.isnan(vals)]
    if len(v) == 0:
        print(f"  {name}: no data")
        return
    print(f"  {name}: n={len(v)}  mu={v.mean():.3f}  "
          f"sigma={v.std():.3f}  min={v.min():.3f}  max={v.max():.3f}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log',       required=True)
    ap.add_argument('--graph',     required=True)
    ap.add_argument('--center',    type=int,   default=24)
    ap.add_argument('--threshold', type=float, default=1e-4)
    ap.add_argument('--nx',        type=int,   default=7,
                    help='Columns for rect layout')
    ap.add_argument('--ny',        type=int,   default=7,
                    help='Rows for rect layout')
    ap.add_argument('--layout',    default='rect',
                    choices=['rect', 'graph', 'heavyhex'],
                    help='Site position layout')
    ap.add_argument('--out',       default=None,
                    help='Output PNG path (default: auto)')
    args = ap.parse_args()

    data   = parse_log(args.log)
    ns, dist, adj = bfs(args.graph, args.center)

    # ── Layout ────────────────────────────────────────────────────────────────
    if args.layout == 'rect':
        pos = rect_layout(ns, args.nx, args.ny)
    else:
        pos = graph_layout(args.graph, ns, dist=dist)
        if pos is None:
            print("  WARNING: could not determine layout, using BFS distance")
            pos = {v: (dist[v], 0) for v in range(ns)}

    # ── Compute per-site stats ────────────────────────────────────────────────
    stats = site_wave_stats(data, args.threshold)

    print(f"\n{'='*60}")
    print(f"Log: {args.log}")
    print(f"Graph: {args.graph}, center={args.center}, "
          f"threshold={args.threshold:.0e}")
    print(f"{'='*60}")

    print(f"\nNoiseless arrival: "
          f"{(stats['t_nl'] >= 0).sum()}/{ns} sites reached")

    print(f"\nPer-site delay l_gamma(v) = t_gamma(v) - t_nl(v):")
    for g in ['N1', 'N2', 'N3']:
        if g not in stats: continue
        summary_stats(stats[g]['delay'], g)

    print(f"\nAmplitude ratio r_gamma(v) = n_gamma(v, t_gamma) / n_nl(v, t_gamma):")
    print(f"  (at noisy arrival cycle; ratio < 1 means noiseless outran noisy during delay)")
    for g in ['N1', 'N2', 'N3']:
        if g not in stats: continue
        summary_stats(stats[g]['ratio'], g)

    # ── Figures ───────────────────────────────────────────────────────────────
    gammas = [g for g in ['N1', 'N2', 'N3'] if g in stats]
    n_gamma = len(gammas)

    fig, axes = plt.subplots(2, n_gamma + 1,
                             figsize=(4 * (n_gamma + 1), 8))

    tag = 'rect7x7' if 'rect' in args.graph else 'heavyhex'

    # Row 0: delay maps
    # Panel 0,0: noiseless arrival time (reference)
    t_nl = stats['t_nl']
    t_max = int(np.nanmax([t_nl[t_nl >= 0].max()] +
                [stats[g]['t'][stats[g]['t'] >= 0].max()
                 for g in gammas if (stats[g]['t'] >= 0).any()]))

    draw_map(axes[0, 0], stats, 'NL', 't', pos,
             'NL arrival cycle $t_{\\rm nl}(v)$',
             'Blues', 1, t_max, 'cycle', ns)

    for col, g in enumerate(gammas, start=1):
        d_max = np.nanmax(stats[g]['delay']) if not np.all(np.isnan(stats[g]['delay'])) else 1
        draw_map(axes[0, col], stats, g, 'delay', pos,
                 f'Delay $l_{{{g}}}(v) = t_{{{g}}} - t_{{\\rm nl}}$',
                 'RdYlGn_r', 0, max(d_max, 1), 'cycles delay', ns)

    # Row 1: natural ratio r = n_gamma(t_gamma) / n_nl(t_gamma)
    axes[1, 0].axis('off')
    r_max = max(
        np.nanmax(stats[g]['ratio']) for g in gammas
        if g in stats and not np.all(np.isnan(stats[g]['ratio']))
    )
    for col, g in enumerate(gammas, start=1):
        draw_map(axes[1, col], stats, g, 'ratio', pos,
                 f'$r_{{{g}}}(v) = n_{{{g}}}(t_{{{g}}}) / n_{{\\rm nl}}(t_{{{g}}})$',
                 'RdYlGn', 0, min(r_max, 1.5), 'ratio', ns)

    # Summary text in axes[1,0]
    ax_txt = axes[1, 0]
    ax_txt.axis('off')
    lines = [f"Threshold: {args.threshold:.0e}",
             f"Sites reached (NL): {(t_nl >= 0).sum()}/{ns}", ""]
    for g in gammas:
        dl = stats[g]['delay']
        dl = dl[~np.isnan(dl)]
        rr = stats[g]['ratio']
        rr = rr[~np.isnan(rr)]
        lines.append(f"{g} delay: μ={dl.mean():.2f} σ={dl.std():.2f}" if len(dl) else f"{g}: no overlap")
        lines.append(f"{g} ratio: μ={rr.mean():.3f} σ={rr.std():.3f}" if len(rr) else "")
        lines.append("")
    ax_txt.text(0.05, 0.95, "\n".join(lines),
                transform=ax_txt.transAxes,
                va='top', ha='left', fontsize=9,
                fontfamily='monospace')

    fig.suptitle(f'Site-centric wave arrival — {tag}', fontsize=13, y=1.01)
    plt.tight_layout()

    out = args.out or args.log.replace('.log', f'_wavemap.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {out}")

if __name__ == '__main__':
    main()
