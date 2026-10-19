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

def log_ratio(ni_val, nl_val, floor=1e-7):
    """
    Pointwise entropy loss: -ln(n_gamma / n_nl).
    KL kernel: n_nl * (-ln(n_gamma/n_nl)) = relative entropy contribution.
    Both clipped to floor to avoid log(0).
    """
    ni_ = max(ni_val, floor)
    nl_ = max(nl_val, floor)
    return -np.log(ni_ / nl_)      # >= 0 when ni <= nl


def global_entropy_loss(data, threshold, floor=1e-7):
    """
    L_gamma(t) = sum_v  -ln(n_gamma(v,t) / n_nl(v,t))
    Restricted to sites inside the noiseless lightcone at cycle t.
    Scalar time series — total entropy loss across lattice per cycle.

    Also returns KL-weighted version:
    L_KL_gamma(t) = sum_v  n_nl(v,t) * (-ln(n_gamma(v,t)/n_nl(v,t)))
    which is the per-cycle KL divergence D_KL(n_nl || n_gamma).
    """
    nl = data['NL']
    nc, ns = nl.shape
    results = {}
    for g in ['N1', 'N2', 'N3']:
        if g not in data:
            continue
        ni = data[g]
        L    = np.zeros(nc)
        L_kl = np.zeros(nc)
        for t in range(nc):
            for v in range(ns):
                if nl[t, v] > threshold:
                    lr = log_ratio(ni[t, v], nl[t, v], floor)
                    L[t]    += lr
                    L_kl[t] += nl[t, v] * lr
        results[g] = {'L': L, 'L_kl': L_kl}
    return results


def site_wave_stats(data, threshold, dist, v_hop_gamma, floor=1e-7):
    """
    Per-site wave arrival analysis.

    Delay:
        l_gamma(v) = t_gamma(v) - t_nl(v)

    Per-site accumulated entropy loss in window [t_gamma - Delta, t_gamma + Delta]:
        Delta(v) = max(1, l_gamma(v))  — delay-adaptive window
        L_gamma(v) = sum_{t in window} -ln(n_gamma(v,t) / n_nl(v,t))

    Connection to information theory:
        -ln(n_gamma/n_nl) is the KL divergence kernel (cross-entropy loss).
        Weighted by n_nl: contribution to D_KL(n_nl || n_gamma) at site v.
        Integrated over time: accumulated relative entropy at v across the
        arrival event.  L_gamma(v) >= 0 always; zero only if n_gamma = n_nl
        throughout the window.
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

        delay   = np.where(nl_arrived & g_arrived, t_g - t_nl, np.nan)
        L_site  = np.full(n_sites, np.nan)   # entropy loss per site
        L_kl_site = np.full(n_sites, np.nan) # KL-weighted per site

        for v in range(n_sites):
            if not nl_arrived[v]:
                continue
            l_v   = int(t_g[v] - t_nl[v]) if g_arrived[v] else 0
            delta = max(1, l_v)
            # window centered on t_gamma (or t_nl if noisy never arrived)
            t_center = t_g[v] if g_arrived[v] else t_nl[v]
            t_lo = max(0,        t_center - delta - 1)   # 0-based
            t_hi = min(n_cycles, t_center + delta)        # 0-based exclusive
            nl_w  = nl[t_lo:t_hi, v]
            ni_w  = ni[t_lo:t_hi, v]
            lr    = np.array([log_ratio(ni_w[i], nl_w[i], floor)
                              for i in range(len(nl_w))])
            L_site[v]    = np.sum(lr)
            L_kl_site[v] = np.sum(nl_w * lr)

        results[g] = {'t': t_g, 'A': A_g, 'delay': delay,
                      'L': L_site, 'L_kl': L_kl_site}
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

GCOL = {'NL':'black','N1':'#2196F3','N2':'#FF9800','N3':'#E91E63'}

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
    dist = np.array(dist)

    # ── Layout ────────────────────────────────────────────────────────────────
    if args.layout == 'rect':
        pos = rect_layout(ns, args.nx, args.ny)
    else:
        pos = graph_layout(args.graph, ns, dist=dist)
        if pos is None:
            print("  WARNING: could not determine layout, using BFS distance")
            pos = {v: (dist[v], 0) for v in range(ns)}

    # ── Frontier velocity per gamma (needed for integration window) ───────────
    def compute_vhop(N_vt, dist_arr, thr):
        nc, ns_ = N_vt.shape
        F = []
        for t in range(nc):
            cur  = N_vt[t]
            prev = N_vt[t-1] if t > 0 else np.zeros(ns_)
            new  = [dist_arr[v] for v in range(ns_) if cur[v]>thr and prev[v]<=thr]
            act  = [dist_arr[v] for v in range(ns_) if cur[v]>thr]
            F.append(max(new) if new else (max(act) if act else 0))
        F = np.array(F)
        t_ = np.arange(1, len(F)+1)
        A_ = np.vstack([t_, np.ones(len(t_))]).T
        v_, _ = np.linalg.lstsq(A_, F, rcond=None)[0]
        return max(v_, 0.01)   # guard against zero/negative

    v_hop = {}
    for g in ['NL', 'N1', 'N2', 'N3']:
        if g in data:
            v_hop[g] = compute_vhop(data[g], dist, args.threshold)

    # ── Compute per-site stats ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Log: {args.log}")
    print(f"Graph: {args.graph}, center={args.center}, "
          f"threshold={args.threshold:.0e}")
    print(f"{'='*60}")

    print(f"\nFrontier velocity v_hop:")
    for g in ['NL','N1','N2','N3']:
        if g in v_hop:
            print(f"  {g}: {v_hop[g]:.4f} hops/cycle")

    gammas = [g for g in ['N1', 'N2', 'N3'] if g in data]
    tag = 'rect7x7' if 'rect' in args.graph else 'heavyhex'

    # ── LR-constrained MPF coefficients ──────────────────────────────────────
    from scipy.optimize import minimize as _minimize

    def frontier_per_cycle(N_vt, dist_arr, thr):
        nc, ns_ = N_vt.shape
        F = []
        for t in range(nc):
            cur  = N_vt[t]
            prev = N_vt[t-1] if t > 0 else np.zeros(ns_)
            new  = [dist_arr[v] for v in range(ns_) if cur[v]>thr and prev[v]<=thr]
            act  = [dist_arr[v] for v in range(ns_) if cur[v]>thr]
            F.append(max(new) if new else (max(act) if act else 0))
        return np.array(F)

    F_nl = frontier_per_cycle(data['NL'], dist, args.threshold)
    F_ni = {g: frontier_per_cycle(data[g], dist, args.threshold) for g in gammas}
    nc   = len(F_nl)
    A_lr = np.array([F_ni[g][:nc] for g in gammas]).T   # (nc, 3)

    def obj(c): return np.sum((A_lr@c - F_nl)**2)
    def jac(c): return 2*A_lr.T@(A_lr@c - F_nl)
    constraints = []
    for t in range(nc):
        constraints.append({'type':'ineq','fun': lambda c,t=t: (A_lr@c)[t]})
        constraints.append({'type':'ineq','fun': lambda c,t=t: F_nl[t]-(A_lr@c)[t]})
    res = _minimize(obj, x0=np.ones(3)/3, jac=jac, method='SLSQP',
                    constraints=constraints, options={'ftol':1e-12,'maxiter':2000})
    c_mpf = res.x
    print(f"\nMPF coefficients: c={np.round(c_mpf,3)}  "
          f"(sum={c_mpf.sum():.3f})")

    # Build MPF n(v,t) field: weighted combination of noisy states
    nl_vt  = data['NL']
    mpf_vt = sum(c_mpf[i] * data[gammas[i]] for i in range(len(gammas)))
    data['MPF'] = mpf_vt   # inject as a new "state"

    # per-site stats (delay-adaptive entropy window)
    stats = site_wave_stats(data, args.threshold, dist, v_hop.get('N1', 0.1))
    t_nl  = stats['t_nl']

    # global entropy loss time series
    glob  = global_entropy_loss(data, args.threshold)

    print(f"\nPer-site entropy loss — MPF:")
    if 'MPF' in stats: summary_stats(stats['MPF']['L'], 'MPF')

    print(f"\nNoiseless arrival: {(t_nl >= 0).sum()}/{ns} sites reached")

    print(f"\nPer-site delay l_gamma(v):")
    for g in gammas:
        if g in stats: summary_stats(stats[g]['delay'], g)

    print(f"\nPer-site entropy loss L_gamma(v) = sum_t -ln(n_gamma/n_nl)"
          f"  in window [t_gamma-Delta, t_gamma+Delta], Delta=max(1,delay):")
    for g in gammas:
        if g in stats: summary_stats(stats[g]['L'], g)

    print(f"\nPer-site KL loss L_KL_gamma(v) = sum_t n_nl * (-ln(n_gamma/n_nl)):")
    for g in gammas:
        if g in stats: summary_stats(stats[g]['L_kl'], g)

    # ── Figures ───────────────────────────────────────────────────────────────
    GCOL['MPF'] = '#2ECC40'   # green for MPF
    all_states  = gammas + ['MPF']
    max_d = int(dist[dist >= 0].max())
    ncols = len(all_states) + 1   # +1 for label column
    fig   = plt.figure(figsize=(4.5*ncols, 13))
    gs    = fig.add_gridspec(4, ncols, hspace=0.45, wspace=0.35)

    # Row 0: delay spatial maps (NL reference + γ1,2,3 + MPF)
    t_max = int(t_nl[t_nl >= 0].max()) if (t_nl >= 0).any() else 10
    ax00  = fig.add_subplot(gs[0, 0])
    draw_map(ax00, {'t_nl': t_nl, 'NL': {'t': t_nl}}, 'NL', 't', pos,
             'NL arrival $t_{\\rm nl}(v)$', 'Blues', 1, t_max, 'cycle', ns)
    d_global_max = max(
        np.nanmax(stats[g]['delay']) for g in gammas
        if g in stats and not np.all(np.isnan(stats[g]['delay'])))
    for col, g in enumerate(all_states, start=1):
        ax = fig.add_subplot(gs[0, col])
        if g == 'MPF':
            # MPF has no delay map — show entropy loss instead as preview
            if 'MPF' in stats:
                L_mpf = stats['MPF']['L']
                L_max_mpf = np.nanmax(L_mpf) if not np.all(np.isnan(L_mpf)) else 1
                draw_map(ax, stats, 'MPF', 'L', pos,
                         f'MPF $\\mathcal{{L}}$ (c=[{",".join(f"{c:.1f}" for c in c_mpf)}])',
                         'YlOrRd', 0, L_max_mpf, 'entropy loss', ns)
        else:
            draw_map(ax, stats, g, 'delay', pos,
                     f'Delay $l_{{{g}}}(v)$',
                     'RdYlGn_r', 0, max(d_global_max, 1), 'cycles', ns)

    # Row 1: entropy loss spatial maps
    L_max = max(np.nanmax(stats[g]['L'])
                for g in all_states
                if g in stats and not np.all(np.isnan(stats[g]['L'])))
    ax10 = fig.add_subplot(gs[1, 0]); ax10.axis('off')
    ax10.text(0.05, 0.95,
              "Entropy loss\n$\\mathcal{L}(v)$\n"
              "$=\\sum_t -\\ln\\frac{n}{n_{\\rm nl}}$\n"
              "window:\n$[t-\\Delta, t+\\Delta]$\n"
              "$\\Delta=\\max(1,l)$\n\n"
              "Green = MPF\nshould approach\nzero (recovery)",
              transform=ax10.transAxes, va='top', fontsize=9)
    for col, g in enumerate(all_states, start=1):
        if g not in stats: continue
        ax = fig.add_subplot(gs[1, col])
        title = (f'MPF $\\mathcal{{L}}(v)$\n$c$=[{",".join(f"{c:.1f}" for c in c_mpf)}]'
                 if g == 'MPF' else f'$\\mathcal{{L}}_{{{g}}}(v)$')
        draw_map(ax, stats, g, 'L', pos, title,
                 'YlGn' if g == 'MPF' else 'YlOrRd',
                 0, L_max, 'entropy loss', ns)

    # Row 2: global L(t) — all gammas + MPF on same axes
    ax20 = fig.add_subplot(gs[2, 0]); ax20.axis('off')
    ax20.text(0.05, 0.95,
              "Global entropy\nloss per cycle\n"
              "$L(t)=\\sum_v -\\ln\\frac{n}{n_{\\rm nl}}$\n\n"
              "Dashed:\n$D_{\\rm KL}(n_{\\rm nl}\\|n)(t)$\n\n"
              "Green = MPF:\nshould sit near\nzero across all\ncycles",
              transform=ax20.transAxes, va='top', fontsize=9)
    ax_L   = fig.add_subplot(gs[2, 1:])
    ax_L_r = ax_L.twinx()
    cycles = np.arange(1, data['NL'].shape[0]+1)
    for g in list(gammas) + ['MPF']:
        if g not in glob: continue
        lw  = 2.5 if g == 'MPF' else 1.8
        ax_L.plot(cycles,   glob[g]['L'],    '-',  color=GCOL[g], lw=lw,
                  label=f'{g}', zorder=5 if g=='MPF' else 3)
        ax_L_r.plot(cycles, glob[g]['L_kl'], '--', color=GCOL[g], lw=lw*0.7, alpha=0.6)
    ax_L.axhline(0, color='gray', lw=0.8, ls=':')
    ax_L.set_xlabel('Cycle', fontsize=9)
    ax_L.set_ylabel('Cross-entropy loss $L(t)$', fontsize=9)
    ax_L_r.set_ylabel('KL divergence (dashed)', fontsize=9, color='gray')
    ax_L.set_title('Global entropy loss — noisy vs MPF-corrected', fontsize=10)
    ax_L.legend(fontsize=9, loc='upper left')
    ax_L.grid(True, alpha=0.3)

    # Row 3: per-shell distribution — all gammas + MPF overlaid on same plot
    ax30 = fig.add_subplot(gs[3, 0]); ax30.axis('off')
    ax30.text(0.05, 0.95,
              "Per-BFS-shell\n$\\mathcal{L}(v)$ (mean±σ)\n\n"
              "All γ + MPF\noverlaid — MPF\n(green) should\ncollapse toward\nzero at all\nshells",
              transform=ax30.transAxes, va='top', fontsize=9)
    ax_sh = fig.add_subplot(gs[3, 1:])
    for g in list(gammas) + ['MPF']:
        if g not in stats: continue
        L_v = stats[g]['L']
        shells, mu_L, sig_L = [], [], []
        for d in range(max_d+1):
            idx  = np.where(dist == d)[0]
            vals = L_v[idx]; vals = vals[~np.isnan(vals)]
            if len(vals) == 0: continue
            shells.append(d); mu_L.append(vals.mean()); sig_L.append(vals.std())
        shells = np.array(shells); mu_L = np.array(mu_L); sig_L = np.array(sig_L)
        lw = 2.5 if g == 'MPF' else 1.8
        ax_sh.plot(shells, mu_L, 'o-', color=GCOL[g], lw=lw, ms=5,
                   label=g, zorder=5 if g=='MPF' else 3)
        ax_sh.fill_between(shells, mu_L-sig_L, mu_L+sig_L,
                            alpha=0.15, color=GCOL[g])
    ax_sh.axhline(0, color='gray', lw=0.8, ls='--')
    ax_sh.set_xlabel('BFS shell (distance from center)', fontsize=9)
    ax_sh.set_ylabel('$\\mathcal{L}$ (mean ± σ)', fontsize=9)
    ax_sh.set_title('Entropy loss per shell — MPF recovery', fontsize=10)
    ax_sh.legend(fontsize=9)
    ax_sh.grid(True, alpha=0.3)

    fig.suptitle(
        f'Entropy loss wavemap — {tag}\n'
        f'MPF: $c$=[{", ".join(f"{c:.2f}" for c in c_mpf)}]  '
        f'(sum={c_mpf.sum():.2f})',
        fontsize=12, y=1.01)
    out = args.out or args.log.replace('.log', '_entropy_wavemap.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {out}")

if __name__ == '__main__':
    main()
