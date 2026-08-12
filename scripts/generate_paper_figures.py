#!/usr/bin/env python3
"""
generate_paper_figures.py — Generate all 6 figures for frontier_mpf.tex

Run from /scratch/MPF/CppSim:
  python3 scripts/generate_paper_figures.py

Outputs to PaperFrontier/:
  fig_lightcone_heatmap.png
  fig_ptm_eigenvalues.png
  fig_wavemap_delay.png
  fig_wavemap_ratio.png
  fig_frontier_velocity.png
  fig_lr_mpf.png
"""

import re, json, sys, os
import numpy as np
from collections import deque
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

OUTDIR  = 'PaperFrontier'
LOG_HH  = 'logs/spatial_mpf_chi70_1e-3_v2.log'
LOG_R7  = 'logs/spatial_mpf_rect7x7_chi50_1e-3_v2.log'
GR_HH   = 'data/graphs/heavy_hex_3x3.json'
GR_R7   = 'data/graphs/rect_7x7.json'
CTR_HH  = 13
CTR_R7  = 24
NX, NY  = 7, 7
THR     = 1e-4
GAMMAS  = ['NL','N1','N2','N3']
COLORS  = {'NL':'black','N1':'#2196F3','N2':'#FF9800','N3':'#E91E63'}
LABELS  = {'NL':'Noiseless','N1':'γ=1','N2':'γ=2','N3':'γ=3'}

os.makedirs(OUTDIR, exist_ok=True)

# ── Shared helpers ────────────────────────────────────────────────────────────

def parse_log(path):
    data = {g: [] for g in GAMMAS}
    with open(path) as f:
        for line in f:
            for g in GAMMAS:
                m = re.match(rf'PAULI_{g} (\d+)(.*)', line)
                if m:
                    data[g].append([float(x) for x in m.group(2).split()])
    return {g: np.array(v) for g, v in data.items() if v}

def bfs(graph_path, center):
    with open(graph_path) as f:
        d = json.load(f)
    ns = d.get('n_sites', len(d.get('vertices', [])))
    adj = [[] for _ in range(ns)]
    for e in d['edges']:
        u = e['edge'][0][0]-1; v = e['edge'][1][0]-1
        adj[u].append(v); adj[v].append(u)
    dist = [-1]*ns; dist[center] = 0; q = deque([center])
    while q:
        v = q.popleft()
        for u in adj[v]:
            if dist[u]<0: dist[u]=dist[v]+1; q.append(u)
    return ns, np.array(dist), adj

def arrival_times(N_vt, thr):
    nc, ns = N_vt.shape
    t_arr = np.full(ns, -1, dtype=int)
    amp   = np.zeros(ns)
    for t in range(nc):
        mask = (t_arr < 0) & (N_vt[t] > thr)
        t_arr[mask] = t+1
        amp[mask]   = N_vt[t, mask]
    return t_arr, amp

def frontier_per_cycle(N_vt, dist, thr):
    nc, ns = N_vt.shape
    F = []
    for t in range(nc):
        cur  = N_vt[t]
        prev = N_vt[t-1] if t>0 else np.zeros(ns)
        new  = [dist[v] for v in range(ns) if cur[v]>thr and prev[v]<=thr]
        act  = [dist[v] for v in range(ns) if cur[v]>thr]
        F.append(max(new) if new else (max(act) if act else 0))
    return np.array(F)

def vhop(F):
    t = np.arange(1, len(F)+1)
    A = np.vstack([t, np.ones(len(t))]).T
    v, b = np.linalg.lstsq(A, F, rcond=None)[0]
    return v, b

def rect_pos(ns, nx, ny):
    return {v: (v%nx, v//nx) for v in range(ns)}

# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading logs...")
data_hh = parse_log(LOG_HH)
data_r7  = parse_log(LOG_R7)
ns_hh, dist_hh, _ = bfs(GR_HH, CTR_HH)
ns_r7,  dist_r7,  _ = bfs(GR_R7,  CTR_R7)
pos_r7 = rect_pos(ns_r7, NX, NY)
# BFS-layer layout for heavy-hex
max_d = int(dist_hh.max())
layer_cnt = np.zeros(max_d+2, dtype=int)
pos_hh = {}
for v in range(ns_hh):
    d = dist_hh[v] if dist_hh[v]>=0 else max_d+1
    pos_hh[v] = (d, layer_cnt[d])
    layer_cnt[d] += 1

# ── Fig 1: lightcone heatmap ──────────────────────────────────────────────────
print("Fig 1: lightcone heatmap...")

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for row, (data, dist, ns, title) in enumerate([
        (data_hh, dist_hh, ns_hh, 'Heavy-hex 3×3'),
        (data_r7,  dist_r7,  ns_r7,  'Rect 7×7')]):
    for col, g in enumerate(GAMMAS):
        ax = axes[row, col]
        if g not in data:
            ax.axis('off'); continue
        N_vt = data[g]
        nc   = N_vt.shape[0]
        # rows=sites, cols=cycles, sorted by BFS distance
        order = np.argsort(dist)
        mat   = np.log10(np.clip(N_vt[:, order].T, 1e-7, None))
        im = ax.imshow(mat, aspect='auto', origin='lower',
                       cmap='hot_r', vmin=-7, vmax=0,
                       extent=[0.5, nc+0.5, 0, ns])
        ax.set_title(f'{LABELS[g]}', fontsize=9)
        if col == 0:
            ax.set_ylabel(f'{title}\nsite (by dist)', fontsize=8)
        if row == 1:
            ax.set_xlabel('cycle', fontsize=8)
        ax.tick_params(labelsize=7)

fig.colorbar(im, ax=axes[:, -1], label='log₁₀ n(v,t)', fraction=0.03)
fig.suptitle('Non-identity Pauli weight $n(v,t)$ — wavefront heatmaps', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig_lightcone_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved fig_lightcone_heatmap.png")

# ── Fig 2: PTM eigenvalues ────────────────────────────────────────────────────
print("Fig 2: PTM eigenvalues...")

import re as _re

def parse_hpp_ptms(path, array_name, n_bonds):
    with open(path) as f:
        text = f.read()
    decl = _re.search(
        rf'static const float {_re.escape(array_name)}\[{n_bonds}\]\[256\]\s*=\s*\{{', text)
    if not decl:
        return None
    start = decl.end()-1
    depth=0; i=start
    while i<len(text):
        if text[i]=='{': depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0: break
        i+=1
    raw = text[start+1:i]
    blocks = _re.findall(r'\{([^{}]+)\}', raw)
    if len(blocks)!=n_bonds: return None
    frx = _re.compile(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?f')
    vals = []
    for blk in blocks:
        vals.extend(float(x.rstrip('f')) for x in frx.findall(blk))
    arr = np.array(vals).reshape(n_bonds, 256)
    return np.array([arr[b].reshape(16,16,order='F') for b in range(n_bonds)])

def get_ptms(topology, gamma):
    if topology == 'hh':
        nb = 76
        if gamma == 0:
            name = 'HEAVY_HEX_3x3_IDEAL_KICK_EPS010'
            path = 'include/heavyhex_ideal_kick_eps010_data.hpp'
        else:
            name = f'HEAVY_HEX_3x3_NOISY_EPS010_GAMMA{gamma}_GAMMA{gamma}_0'
            path = f'include/heavyhex_noisy_eps010_gamma{gamma}_data.hpp'
    else:
        nb = 84
        if gamma == 0:
            name = 'RECT_7x7_IDEAL_KICK_EPS010'
            path = 'include/rect_7x7_ideal_kick_eps010_data.hpp'
        else:
            # find actual array name
            with open(f'include/rect_7x7_noisy_eps010_gamma{gamma}_data.hpp') as f:
                txt = f.read()
            m = _re.search(r'static const float (\w+)\[84\]\[256\]', txt)
            if not m: return None
            name = m.group(1)
            path = f'include/rect_7x7_noisy_eps010_gamma{gamma}_data.hpp'
    return parse_hpp_ptms(path, name, nb)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for row, (topo, label, bond_idx) in enumerate([('hh','Heavy-hex',0),('r7','Rect 7×7',0)]):
    for col, (g_list, subtitle) in enumerate([
            ([0,1,2,3], 'All γ — eigenvalue positions'),
            ([0,1,2,3], 'Off-diagonal power vs γ')]):
        ax = axes[row, col]

        if col == 0:
            # Eigenvalue scatter in complex plane
            for gamma, color, lbl in [(0,'black','ideal'),(1,'#2196F3','γ=1'),
                                       (2,'#FF9800','γ=2'),(3,'#E91E63','γ=3')]:
                ptms = get_ptms(topo, gamma)
                if ptms is None: continue
                eigs = np.linalg.eigvals(ptms[bond_idx])
                ax.scatter(np.real(eigs), np.imag(eigs), s=40, color=color,
                           label=lbl, zorder=3, alpha=0.85)
            ax.axhline(0, color='gray', lw=0.5)
            ax.axvline(0, color='gray', lw=0.5)
            circ = plt.Circle((0,0),1,fill=False,color='gray',lw=0.5,ls='--')
            ax.add_patch(circ)
            ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.1,1.1)
            ax.set_xlabel('Re(λ)', fontsize=8); ax.set_ylabel('Im(λ)', fontsize=8)
            ax.set_title(f'{label} — bond {bond_idx} eigenvalues', fontsize=9)
            ax.legend(fontsize=7); ax.set_aspect('equal')

        else:
            # Off-diagonal power fraction vs gamma
            gammas_x = [0,1,2,3]
            fracs = []
            for gamma in gammas_x:
                ptms = get_ptms(topo, gamma)
                if ptms is None: fracs.append(np.nan); continue
                diag_p  = np.mean([np.sum(np.diag(ptms[b])**2) for b in range(ptms.shape[0])])
                offdiag = np.mean([np.sum((ptms[b]-np.diag(np.diag(ptms[b])))**2)
                                   for b in range(ptms.shape[0])])
                fracs.append(offdiag/(diag_p+offdiag))
            ax.plot(gammas_x, fracs, 'o-', color='steelblue', lw=2, ms=8)
            ax.set_ylim(0, 0.5)
            ax.set_xlabel('γ', fontsize=8)
            ax.set_ylabel('Off-diagonal power fraction', fontsize=8)
            ax.set_title(f'{label} — rotation content vs γ', fontsize=9)
            ax.axhline(fracs[0], color='gray', ls='--', lw=0.8, label='ideal')
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

plt.suptitle('Composed PTM eigenvalue analysis:\nnoise rescales magnitudes, gate sets eigenvectors',
             fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig_ptm_eigenvalues.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved fig_ptm_eigenvalues.png")

# ── Fig 3 & 4: wavemaps ───────────────────────────────────────────────────────
print("Fig 3&4: wavemaps...")

def compute_wavemap(data, ns):
    t_nl, _ = arrival_times(data['NL'], THR)
    result = {'t_nl': t_nl}
    for g in ['N1','N2','N3']:
        if g not in data: continue
        t_g, _ = arrival_times(data[g], THR)
        delay = np.where((t_nl>=0)&(t_g>=0), t_g-t_nl, np.nan)
        ratio = np.full(ns, np.nan)
        nc = data['NL'].shape[0]
        for v in range(ns):
            if t_nl[v]<0 or t_g[v]<0: continue
            tg = t_g[v]-1
            if tg>=nc: continue
            nl_val = data['NL'][tg, v]
            ni_val = data[g][tg, v]
            ratio[v] = ni_val/nl_val if nl_val>0 else np.nan
        result[g] = {'delay': delay, 'ratio': ratio, 't': t_g}
    return result

def scatter_map(ax, vals, pos, ns, cmap, vmin, vmax, t_nl, t_g=None, title='', cb_label=''):
    xs_w, ys_w, xs_b, ys_b, xs_c, ys_c, cs = [], [], [], [], [], [], []
    for v in range(ns):
        x, y = pos[v]
        if t_nl[v] < 0:
            xs_w.append(x); ys_w.append(y)
        elif np.isnan(vals[v]):
            xs_b.append(x); ys_b.append(y)
        else:
            xs_c.append(x); ys_c.append(y); cs.append(vals[v])
    ax.scatter(xs_w, ys_w, c='white', edgecolors='lightgray', s=100, zorder=2)
    if xs_b:
        ax.scatter(xs_b, ys_b, c='#222222', s=100, zorder=3)
    if xs_c:
        sc = ax.scatter(xs_c, ys_c, c=cs, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=100, zorder=4, edgecolors='gray', lw=0.3)
        plt.colorbar(sc, ax=ax, label=cb_label, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    ax.set_facecolor('#e8e8e8')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

for fig_name, field, cmap, vmin_fn, vmax_fn, cb_label, fig_title in [
    ('fig_wavemap_delay', 'delay', 'RdYlGn_r',
     lambda wm: 0,
     lambda wm: max(np.nanmax(wm[g]['delay']) for g in ['N1','N2','N3'] if g in wm),
     'cycles delay', 'Wavemap: arrival delay $l_\\gamma(v)$'),
    ('fig_wavemap_ratio', 'ratio', 'RdYlGn',
     lambda wm: 0, lambda wm: 1.0,
     'ratio', 'Wavemap: natural amplitude ratio $r_\\gamma(v)$'),
]:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for row, (data, dist, ns, pos, lbl) in enumerate([
            (data_hh, dist_hh, ns_hh, pos_hh, 'Heavy-hex'),
            (data_r7,  dist_r7,  ns_r7,  pos_r7,  'Rect 7×7')]):
        wm = compute_wavemap(data, ns)
        vmin = vmin_fn(wm); vmax = vmax_fn(wm)
        if vmax == 0: vmax = 1
        for col, g in enumerate(['N1','N2','N3']):
            ax = axes[row, col]
            if g not in wm:
                ax.axis('off'); continue
            vals = wm[g][field]
            scatter_map(ax, vals, pos, ns, cmap, vmin, vmax,
                        wm['t_nl'], wm[g]['t'],
                        title=f'{lbl} — {LABELS[g]}',
                        cb_label=cb_label)
    fig.suptitle(fig_title, fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/{fig_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved {fig_name}.png")

# ── Fig 5: frontier velocity ──────────────────────────────────────────────────
print("Fig 5: frontier velocity...")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (data, dist, ns, title) in zip(axes, [
        (data_hh, dist_hh, ns_hh, 'Heavy-hex 3×3'),
        (data_r7,  dist_r7,  ns_r7,  'Rect 7×7')]):
    for g in GAMMAS:
        if g not in data: continue
        F = frontier_per_cycle(data[g], dist, THR)
        t = np.arange(1, len(F)+1)
        v, b = vhop(F)
        ax.plot(t, F, 'o', color=COLORS[g], ms=5, alpha=0.7)
        ax.plot(t, v*t+b, '-', color=COLORS[g], lw=1.5,
                label=f'{LABELS[g]} $v$={v:.3f}')
    ax.set_xlabel('Cycle', fontsize=10)
    ax.set_ylabel('Frontier distance (hops)', fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle('Frontier velocity $v_{\\rm hop}$ — noiseless vs noisy', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig_frontier_velocity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved fig_frontier_velocity.png")

# ── Fig 6: LR-constrained MPF ─────────────────────────────────────────────────
print("Fig 6: LR-MPF...")

def lr_mpf(data, dist, ns):
    F = {}
    for g in GAMMAS:
        if g not in data:
            continue
        F[g] = frontier_per_cycle(data[g], dist, THR)
    nc = min(len(F[g]) for g in F)
    F = {g: F[g][:nc] for g in F}
    F_nl = F['NL']
    noisy = ['N1','N2','N3']
    A = np.array([F[g] for g in noisy]).T   # (nc, 3)
    b_vec = F_nl

    # SLSQP: minimize ||A·c - F_nl||^2 s.t. 0 <= A·c <= F_nl pointwise
    def obj(c): return np.sum((A@c - b_vec)**2)
    def grad(c): return 2*A.T@(A@c - b_vec)
    constraints = []
    for t in range(nc):
        constraints.append({'type':'ineq','fun': lambda c,t=t: (A@c)[t]})
        constraints.append({'type':'ineq','fun': lambda c,t=t: F_nl[t]-(A@c)[t]})
    from scipy.optimize import minimize as _min
    res = _min(obj, x0=np.ones(3)/3, jac=grad, method='SLSQP',
               constraints=constraints, options={'ftol':1e-10,'maxiter':1000})
    c = res.x
    F_lr = A @ c
    return F, F_nl, F_lr, c

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (data, dist, ns, title) in zip(axes, [
        (data_hh, dist_hh, ns_hh, 'Heavy-hex 3×3'),
        (data_r7,  dist_r7,  ns_r7,  'Rect 7×7')]):
    F, F_nl, F_lr, c = lr_mpf(data, dist, ns)
    t = np.arange(1, len(F_nl)+1)
    ax.plot(t, F_nl, 'k-', lw=2, label='NL (LR ceiling)')
    for g, col in zip(['N1','N2','N3'], ['#2196F3','#FF9800','#E91E63']):
        if g in F:
            ax.plot(t, F[g], '--', color=col, lw=1.2, alpha=0.7, label=LABELS[g])
    ax.plot(t, F_lr, 'g-', lw=2.5,
            label=f'MPF ($c$=[{c[0]:.1f},{c[1]:.1f},{c[2]:.1f}])')
    ax.fill_between(t, F_lr, F_nl, alpha=0.1, color='green', label='residual')
    ax.set_xlabel('Cycle', fontsize=10)
    ax.set_ylabel('Frontier distance (hops)', fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.suptitle('LR-constrained spatial MPF — frontier recovery', fontsize=11)
plt.tight_layout()
plt.savefig(f'{OUTDIR}/fig_lr_mpf.png', dpi=150, bbox_inches='tight')
plt.close()
print("  saved fig_lr_mpf.png")

print("\nAll figures generated in", OUTDIR)
