#!/usr/bin/env python3
"""
plot_mpf_entropy.py — Dedicated MPF vs NL entropy loss recovery figure.

Shows in one page:
  - Spatial map: entropy loss per site for NL, best noisy, and MPF
  - Global L(t): entropy loss vs cycle for all γ + MPF
  - Per-shell: mean ± σ of L(v) per BFS shell, all γ + MPF overlaid

Usage:
  python3 scripts/plot_mpf_entropy.py \
      --log logs/spatial_mpf_rect7x7_chi50_1e-3_v2.log \
      --graph data/graphs/rect_7x7.json \
      --center 24 --threshold 1e-4 --layout rect --nx 7 --ny 7

  python3 scripts/plot_mpf_entropy.py \
      --log logs/spatial_mpf_chi70_1e-3_v2.log \
      --graph data/graphs/heavy_hex_3x3.json \
      --center 13 --threshold 1e-4 --layout graph
"""

import argparse, re, json
import numpy as np
from collections import deque
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgrid

GCOL = {'NL':'black','N1':'#2196F3','N2':'#FF9800','N3':'#E91E63',
        'MPF':'#2ECC40','MPF_E':'#9B59B6','MPF_T':'#1A237E',
        'MPF_D1':'#8BC34A','MPF_D2':'#795548'}

# Dashed for noisy inputs, solid for MPF outputs
GLINE = {'NL':'-','N1':'--','N2':'--','N3':'--',
         'MPF':'-','MPF_E':'-','MPF_T':'-',
         'MPF_D1':'-','MPF_D2':'-'}
LABS = {'NL':'Noiseless','N1':'γ=1','N2':'γ=2','N3':'γ=3',
        'MPF':'MPF (frontier)','MPF_E':'MPF (entropy)',
        'MPF_T':'MPF α(t) time-proj',
        'MPF_D1':'MPF α(d) space-proj opt1',
        'MPF_D2':'MPF α(d) space-proj opt2'}
FLOOR = 1e-7

# ── helpers ───────────────────────────────────────────────────────────────────

def parse_log(path):
    data = {g: [] for g in ['NL','N1','N2','N3']}
    with open(path) as f:
        for line in f:
            for g in data:
                m = re.match(rf'PAULI_{g} (\d+)(.*)', line)
                if m:
                    data[g].append([float(x) for x in m.group(2).split()])
    return {g: np.array(v) for g,v in data.items() if v}

def bfs(path, center):
    with open(path) as f: d = json.load(f)
    ns  = d.get('n_sites', len(d.get('vertices',[])))
    adj = [[] for _ in range(ns)]
    for e in d['edges']:
        u=e['edge'][0][0]-1; v=e['edge'][1][0]-1
        adj[u].append(v); adj[v].append(u)
    dist=[-1]*ns; dist[center]=0; q=deque([center])
    while q:
        v=q.popleft()
        for u in adj[v]:
            if dist[u]<0: dist[u]=dist[v]+1; q.append(u)
    return ns, np.array(dist)

def rect_pos(ns, nx, ny):
    return {v:(v%nx, v//nx) for v in range(ns)}

def graph_pos(path, ns, dist):
    with open(path) as f: d=json.load(f)
    pos={}
    for e in d.get('vertices',[]):
        if not isinstance(e,dict): continue
        idx=(e.get('id',[None])[0])
        if idx is None: continue
        coords=e.get('coords',None)
        if coords and len(coords)>=2:
            pos[int(idx)-1]=(float(coords[0]),float(coords[1]))
    if len(pos)==ns: return pos
    # BFS fallback
    layer=[0]*(int(dist.max())+2)
    pos={}
    for v in range(ns):
        dv=dist[v] if dist[v]>=0 else int(dist.max())+1
        pos[v]=(dv, layer[dv]); layer[dv]+=1
    return pos

def frontier_per_cycle(N_vt, dist, thr):
    nc,ns=N_vt.shape; F=[]
    for t in range(nc):
        cur=N_vt[t]; prev=N_vt[t-1] if t>0 else np.zeros(ns)
        new=[dist[v] for v in range(ns) if cur[v]>thr and prev[v]<=thr]
        act=[dist[v] for v in range(ns) if cur[v]>thr]
        F.append(max(new) if new else (max(act) if act else 0))
    return np.array(F)

def arrival_time(N_vt, thr):
    nc,ns=N_vt.shape
    t_arr=np.full(ns,-1,dtype=int)
    for t in range(nc):
        mask=(t_arr<0)&(N_vt[t]>thr)
        t_arr[mask]=t+1
    return t_arr

def entropy_loss_site(ni_vt, nl_vt, t_nl, t_g, dist, thr):
    """
    Per-site accumulated entropy loss in delay-adaptive window.
    L(v) = sum_{t in [t_g-Delta, t_g+Delta]} -ln(n/n_nl)
    Delta = max(1, t_g - t_nl)
    """
    nc,ns = nl_vt.shape
    L = np.full(ns, np.nan)
    for v in range(ns):
        if t_nl[v] < 0: continue
        l_v    = max(0, int(t_g[v]-t_nl[v])) if t_g[v]>=0 else 0
        delta  = max(1, l_v)
        t_ctr  = t_g[v] if t_g[v]>=0 else t_nl[v]
        t_lo   = max(0,  t_ctr - delta - 1)
        t_hi   = min(nc, t_ctr + delta)
        nl_w   = nl_vt[t_lo:t_hi, v]
        ni_w   = ni_vt[t_lo:t_hi, v]
        lr     = -np.log(np.clip(ni_w,FLOOR,None)/np.clip(nl_w,FLOOR,None))
        L[v]   = np.sum(lr)
    return L

def entropy_loss_global(ni_vt, nl_vt, thr):
    """L(t) = sum_v -ln(n_gamma/n_nl) for sites inside NL lightcone."""
    nc,ns=nl_vt.shape; L=np.zeros(nc)
    for t in range(nc):
        mask=nl_vt[t]>thr
        L[t]=np.sum(-np.log(np.clip(ni_vt[t,mask],FLOOR,None)/
                            np.clip(nl_vt[t,mask],FLOOR,None)))
    return L

def fit_alpha_d(ni_stack, nl_vt, t_nl, t_gi_list, dist, thr, option=1):
    """
    Space projection: α(d) — for each BFS shell d, fit α over time.

    Option 1 (single snapshot):
      min_α  Σ_{v∈V_d} -ln( Σi αi n_γi(v, t_γi(v)) / n(v, t_n(v)) )
      Each noisy wave at its own arrival vs noiseless at its arrival.
      One (v,t) pair per site — no zero problem.

    Option 2 (spreading window [t_γ1, t_γ3]):
      min_α  Σ_{v∈V_d} Σ_{t∈[min_i t_γi(v), max_i t_γi(v)]}
                        -ln( Σi αi n_γi(v,t) / n(v,t) )
      Captures the period when noisy waves are spreading apart.
      Better conditioned than option 1 — more data per shell.

    Returns:
      alpha_d: dict {d: array(ngamma)} — coefficients per shell
      mpf_d_vt: (nc, ns) — reconstructed field
    """
    nc, ns  = nl_vt.shape
    ngamma  = ni_stack.shape[0]
    max_d   = int(dist[dist >= 0].max())
    alpha_d = {}
    mpf_d_vt = np.zeros((nc, ns))

    for d in range(max_d + 1):
        vset = np.where(dist == d)[0]
        # Only sites where noiseless and all noisy have arrived
        valid = [v for v in vset if t_nl[v] >= 0 and
                 all(t_gi_list[k][v] >= 0 for k in range(ngamma))]
        if len(valid) < ngamma:
            alpha_d[d] = np.ones(ngamma) / ngamma
            continue

        if option == 1:
            # Single snapshot: evaluate each noisy at its own arrival time
            rows = []
            targets = []
            for v in valid:
                row = np.array([ni_stack[k, t_gi_list[k][v], v]
                                for k in range(ngamma)])
                tgt = nl_vt[t_nl[v], v]
                if tgt > FLOOR and np.all(row > 0):
                    rows.append(row); targets.append(tgt)
            if len(rows) < ngamma:
                alpha_d[d] = np.ones(ngamma) / ngamma
                continue
            A = np.array(rows); b = np.array(targets)

            def obj1(c):
                mpf = np.clip(A @ c, FLOOR, None)
                return np.sum(-np.log(mpf / np.clip(b, FLOOR, None)))
            def jac1(c):
                mpf = np.clip(A @ c, FLOOR, None)
                return -np.sum(A / mpf[:, None], axis=0)

        else:  # option 2: spreading window [min t_γi, max t_γi]
            rows_vt = []   # list of (ni_vec, nl_val) per (v,t)
            for v in valid:
                t_lo = min(t_gi_list[k][v] for k in range(ngamma))
                t_hi = max(t_gi_list[k][v] for k in range(ngamma))
                for t in range(t_lo, t_hi + 1):
                    if t >= nc: break
                    nl_val = nl_vt[t, v]
                    ni_vec = np.array([ni_stack[k, t, v] for k in range(ngamma)])
                    if nl_val > FLOOR:
                        rows_vt.append((ni_vec, nl_val))
            if len(rows_vt) < ngamma:
                alpha_d[d] = np.ones(ngamma) / ngamma
                continue
            A = np.array([r[0] for r in rows_vt])
            b = np.array([r[1] for r in rows_vt])

            def obj1(c):
                mpf = np.clip(A @ c, FLOOR, None)
                return np.sum(-np.log(mpf / np.clip(b, FLOOR, None)))
            def jac1(c):
                mpf = np.clip(A @ c, FLOOR, None)
                return -np.sum(A / mpf[:, None], axis=0)

        # Shared solve
        x0 = alpha_d.get(d-1, np.ones(ngamma)/ngamma)
        cons_d = [
            {'type':'eq',   'fun': lambda c: np.sum(c) - 1.0},
            {'type':'ineq', 'fun': lambda c: A @ c},
            {'type':'ineq', 'fun': lambda c: np.clip(b, FLOOR, None) - A @ c}
        ]
        res = minimize(obj1, x0=x0, jac=jac1, method='SLSQP',
                       constraints=cons_d,
                       options={'ftol':1e-10, 'maxiter':500})
        alpha_d[d] = res.x

    # Reconstruct full field: apply α(d(v)) at each site v, all cycles
    alpha_site = np.array([alpha_d.get(dist[v], np.ones(ngamma)/ngamma)
                           for v in range(ns)])  # (ns, ngamma)
    for t in range(nc):
        for v in range(ns):
            mpf_d_vt[t, v] = np.dot(alpha_site[v], ni_stack[:, t, v])
    return alpha_d, mpf_d_vt

def fit_alpha_t(ni_stack, nl_vt, dist, thr):
    """
    Time projection: α(t) — for each cycle t, fit α over frontier sites F(t).

    F(t) = union of first-arrival sites across all states at time t.
    At these sites chi≈1 → simulation exact → fit operates on exact data.

    Returns:
      alpha_t: (nc, 3) — coefficients per cycle
      mpf_t_vt: (nc, ns) — reconstructed field applying α(t) at each cycle t
    """
    nc, ns = nl_vt.shape
    ngamma = ni_stack.shape[0]
    alpha_t  = np.zeros((nc, ngamma))
    mpf_t_vt = np.zeros((nc, ns))

    for t in range(nc):
        # F(t): first-arrival sites for NL or any noisy state at this cycle
        nl_prev = nl_vt[t-1] if t > 0 else np.zeros(ns)
        frontier = (nl_vt[t] > thr) & (nl_prev <= thr)
        for k in range(ngamma):
            ni_prev = ni_stack[k, t-1] if t > 0 else np.zeros(ns)
            frontier |= (ni_stack[k, t] > thr) & (ni_prev <= thr)

        # Also include any active frontier site (any wave nonzero, NL nonzero)
        active = nl_vt[t] > thr
        for k in range(ngamma):
            active &= (ni_stack[k, t] > thr)
        mask = frontier & active

        if mask.sum() < ngamma:
            # Not enough frontier sites — fall back to all active sites
            mask = active
        if mask.sum() < ngamma:
            # Still not enough — use previous α
            alpha_t[t] = alpha_t[t-1] if t > 0 else np.ones(ngamma)/ngamma
            mpf_t_vt[t] = np.einsum('k,kj->j', alpha_t[t], ni_stack[:, t, :])
            continue

        # Fit α(t): minimize MSE on frontier sites at this cycle
        # min ||Σk αk n_γk(v,t) - n_nl(v,t)||² for v ∈ mask
        A = ni_stack[:, t, :][:, mask].T   # (M, ngamma)
        b = nl_vt[t, mask]                  # (M,)

        def obj(c): return np.sum((A @ c - b)**2)
        def jac(c): return 2 * A.T @ (A @ c - b)
        cons = [{'type':'eq', 'fun': lambda c: np.sum(c) - 1.0},
                {'type':'ineq', 'fun': lambda c: A @ c},           # ≥ 0
                {'type':'ineq', 'fun': lambda c: b - A @ c}]       # ≤ n_nl

        x0 = alpha_t[t-1] if t > 0 else np.ones(ngamma)/ngamma
        res = minimize(obj, x0=x0, jac=jac, method='SLSQP',
                       constraints=cons, options={'ftol':1e-10,'maxiter':500})
        alpha_t[t] = res.x

        # Apply α(t) to ALL sites at cycle t
        mpf_t_vt[t] = np.einsum('k,kj->j', alpha_t[t], ni_stack[:, t, :])

    return alpha_t, mpf_t_vt

def scatter_map(ax, vals, pos, ns, t_nl, cmap, vmin, vmax, cb_label, title):
    xs_w,ys_w,xs_b,ys_b,xs_c,ys_c,cs=[],[],[],[],[],[],[]
    for v in range(ns):
        x,y=pos[v]
        if t_nl[v]<0: xs_w.append(x); ys_w.append(y)
        elif np.isnan(vals[v]): xs_b.append(x); ys_b.append(y)
        else: xs_c.append(x); ys_c.append(y); cs.append(vals[v])
    ax.scatter(xs_w,ys_w,c='white',edgecolors='lightgray',s=120,zorder=2)
    if xs_b: ax.scatter(xs_b,ys_b,c='#333',s=120,zorder=3)
    if xs_c:
        sc=ax.scatter(xs_c,ys_c,c=cs,cmap=cmap,vmin=vmin,vmax=vmax,
                      s=120,zorder=4,edgecolors='gray',lw=0.3)
        plt.colorbar(sc,ax=ax,label=cb_label,fraction=0.046,pad=0.04)
    ax.set_title(title,fontsize=10); ax.set_aspect('equal')
    ax.set_facecolor('#e8e8e8')
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--log',      required=True)
    ap.add_argument('--graph',    required=True)
    ap.add_argument('--center',   type=int,   default=24)
    ap.add_argument('--threshold',type=float, default=1e-4)
    ap.add_argument('--nx',       type=int,   default=7)
    ap.add_argument('--ny',       type=int,   default=7)
    ap.add_argument('--layout',   default='rect', choices=['rect','graph'])
    ap.add_argument('--out',      default=None)
    ap.add_argument('--qiml',     action='store_true',
                    help='Generate 2x2 QIML figure (red/green spatial + L(t)/L(d))')
    args=ap.parse_args()

    data=parse_log(args.log)
    ns,dist=bfs(args.graph,args.center)
    pos=(rect_pos(ns,args.nx,args.ny) if args.layout=='rect'
         else graph_pos(args.graph,ns,dist))
    tag='rect7x7' if 'rect' in args.graph else 'heavyhex'

    nl_vt  = data['NL']
    nc,_   = nl_vt.shape
    cycles = np.arange(1,nc+1)
    gammas = [g for g in ['N1','N2','N3'] if g in data]
    thr    = args.threshold

    # arrival times
    t_nl = arrival_time(nl_vt, thr)
    t_g  = {g: arrival_time(data[g], thr) for g in gammas}

    # ── LR-constrained MPF ────────────────────────────────────────────────────
    F_nl = frontier_per_cycle(nl_vt, dist, thr)
    F_ni = {g: frontier_per_cycle(data[g], dist, thr) for g in gammas}
    A    = np.array([F_ni[g][:nc] for g in gammas]).T
    def obj(c): return np.sum((A@c-F_nl)**2)
    def jac(c): return 2*A.T@(A@c-F_nl)
    cons=[{'type':'ineq','fun':lambda c,t=t:(A@c)[t]}          for t in range(nc)] + \
         [{'type':'ineq','fun':lambda c,t=t:F_nl[t]-(A@c)[t]} for t in range(nc)]
    res=minimize(obj,x0=np.ones(3)/3,jac=jac,method='SLSQP',
                 constraints=cons,options={'ftol':1e-12,'maxiter':2000})
    c_mpf  = res.x
    mpf_vt = sum(c_mpf[i]*data[gammas[i]] for i in range(len(gammas)))
    t_mpf  = arrival_time(mpf_vt, thr)
    print(f"MPF frontier coefficients: {np.round(c_mpf,3)}  sum={c_mpf.sum():.3f}")

    # ── Entropy-optimal MPF ───────────────────────────────────────────────────
    # Objective: minimize sum_{v,t in window} -ln(n_MPF(v,t)/n_nl(v,t))
    # Gradient:  d/dc_k = -sum_{v,t} n_gamma_k(v,t) / n_MPF(v,t)
    # Constraint: n_MPF(v,t) > 0 (LR: can't exceed noiseless, ensured by floor)
    # Also: sum(c) = 1 for normalization
    ni_stack = np.stack([data[g] for g in gammas], axis=0)  # (3, nc, ns)

    def mpf_field(c):
        return np.einsum('i,int->nt', c, ni_stack)

    def entropy_obj(c):
        n_mpf = np.clip(mpf_field(c), FLOOR, None)
        n_nl_ = np.clip(nl_vt, FLOOR, None)
        # only sites inside noiseless lightcone
        mask  = nl_vt > thr
        return np.sum(-np.log(n_mpf[mask] / n_nl_[mask]))

    def entropy_jac(c):
        n_mpf = np.clip(mpf_field(c), FLOOR, None)
        mask  = nl_vt > thr   # (nc, ns)
        grad  = np.zeros(len(c))
        for k in range(len(c)):
            # d/dc_k = -sum_{v,t in mask} n_gamma_k(v,t) / n_mpf(v,t)
            grad[k] = -np.sum(ni_stack[k][mask] / n_mpf[mask])
        return grad

    # Constraints:
    #   sum(c) = 1  (normalization)
    #   n_MPF(v,t) >= 0  for all (v,t) inside NL lightcone
    #   n_MPF(v,t) <= n_nl(v,t)  for all (v,t)  [Lieb-Robinson / causality]
    # Build constraint matrices over all (t,v) where nl > threshold
    mask_lr  = nl_vt > thr                         # (nc, ns) bool
    nl_flat  = nl_vt[mask_lr]                       # (M,)
    ni_flat  = ni_stack[:, mask_lr]                 # (3, M)

    cons_e = [
        {'type': 'eq',  'fun': lambda c: np.sum(c) - 1.0},
        # n_MPF >= 0: c . ni_flat >= 0 for each point
        {'type': 'ineq','fun': lambda c: ni_flat.T @ c},          # (M,) >= 0
        # n_MPF <= n_nl: n_nl - c . ni_flat >= 0 for each point
        {'type': 'ineq','fun': lambda c: nl_flat - ni_flat.T @ c} # (M,) >= 0
    ]
    res_e = minimize(entropy_obj, x0=c_mpf.copy(),
                     jac=entropy_jac, method='SLSQP',
                     constraints=cons_e,
                     options={'ftol':1e-10, 'maxiter':3000})
    c_mpf_e  = res_e.x
    mpf_e_vt = mpf_field(c_mpf_e)
    t_mpf_e  = arrival_time(mpf_e_vt, thr)
    print(f"MPF entropy coefficients:  {np.round(c_mpf_e,3)}  sum={c_mpf_e.sum():.3f}")

    # ── Time projection: α(t) ─────────────────────────────────────────────────
    # Fit one α per cycle over frontier sites F(t) — chi≈1, exact data.
    # α(t) is a "time projection": at cycle t, which combination of noisy
    # states best reconstructs the noiseless spatial field at the frontier?
    print("\n[Time projection α(t)]")
    alpha_t, mpf_t_vt = fit_alpha_t(ni_stack, nl_vt, dist, thr)
    t_mpf_t = arrival_time(mpf_t_vt, thr)
    print(f"  α(t) shape: {alpha_t.shape}  (one vector per cycle)")
    print(f"  α(t) range: [{alpha_t.min():.3f}, {alpha_t.max():.3f}]")
    for k, g in enumerate(gammas):
        print(f"  α_{g}(t): mean={alpha_t[:,k].mean():.3f}  "
              f"min={alpha_t[:,k].min():.3f}  max={alpha_t[:,k].max():.3f}")

    # ── Space projection: α(d) option 1 (single snapshot) ───────────────────
    print("\n[Space projection α(d) — option 1: single snapshot]")
    t_gi_list = [t_g[g] for g in gammas]
    alpha_d1, mpf_d1_vt = fit_alpha_d(ni_stack, nl_vt, t_nl, t_gi_list,
                                       dist, thr, option=1)
    t_mpf_d1 = arrival_time(mpf_d1_vt, thr)
    for d in sorted(alpha_d1.keys()):
        print(f"  α(d={d}): {np.round(alpha_d1[d],3)}")

    # ── Space projection: α(d) option 2 (spreading window [t_γ1, t_γ3]) ─────
    print("\n[Space projection α(d) — option 2: window [t_γ1, t_γ3]]")
    alpha_d2, mpf_d2_vt = fit_alpha_d(ni_stack, nl_vt, t_nl, t_gi_list,
                                       dist, thr, option=2)
    t_mpf_d2 = arrival_time(mpf_d2_vt, thr)
    for d in sorted(alpha_d2.keys()):
        print(f"  α(d={d}): {np.round(alpha_d2[d],3)}")

    # NOTE: α(d) + b(d) information latency (option 3) was implemented but
    # DISQUALIFIED: violates Lieb-Robinson (n_MPF > n_nl at some sites).
    # b(d) is a shell-level constant that cannot guarantee n_MPF ≤ n_nl at
    # every (v,t) without per-point constraints. hh: min L=-3.7 (40x overshoot),
    # r7: min L=-4.2 (67x overshoot). Concept is valid but needs reformulation.

    all_states = gammas + ['MPF', 'MPF_E', 'MPF_T', 'MPF_D1', 'MPF_D2']
    all_vt     = {**data, 'MPF': mpf_vt,  'MPF_E': mpf_e_vt,
                  'MPF_T': mpf_t_vt, 'MPF_D1': mpf_d1_vt, 'MPF_D2': mpf_d2_vt}
    all_tg     = {**t_g,  'MPF': t_mpf,   'MPF_E': t_mpf_e,
                  'MPF_T': t_mpf_t, 'MPF_D1': t_mpf_d1, 'MPF_D2': t_mpf_d2}

    # ── per-site entropy loss for all states ─────────────────────────────────
    L_site = {}
    for g in all_states:
        L_site[g] = entropy_loss_site(all_vt[g], nl_vt, t_nl, all_tg[g], dist, thr)
        vals = L_site[g][~np.isnan(L_site[g])]
        print(f"  L_site {g}: mu={vals.mean():.3f}  sigma={vals.std():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}")

    # global entropy loss
    L_glob = {g: entropy_loss_global(all_vt[g], nl_vt, thr) for g in all_states}

    # Best noisy mu for comparison
    best_noisy_mu = min(np.nanmean(L_site[g][~np.isnan(L_site[g])]) for g in gammas)
    mpf_mu   = np.nanmean(L_site['MPF'][~np.isnan(L_site['MPF'])])
    mpfe_mu  = np.nanmean(L_site['MPF_E'][~np.isnan(L_site['MPF_E'])])
    print(f"\nComparison (mean entropy loss):")
    print(f"  Best noisy:       {best_noisy_mu:.3f}")
    print(f"  MPF (frontier):   {mpf_mu:.3f}  {'BETTER' if mpf_mu < best_noisy_mu else 'worse'}")
    print(f"  MPF (entropy):    {mpfe_mu:.3f}  {'BETTER' if mpfe_mu < best_noisy_mu else 'worse'}")

    # ── Figure layout: 3 rows ─────────────────────────────────────────────────
    # Row 0: spatial maps (one per state)
    # Row 1: L(t) [left] | L(v) per BFS shell [right]  — loss in time and space
    # Row 2: α(t) [left] | α(d) opt1+opt2 [right]      — coefficients in time and space
    L_max = max(np.nanmax(L_site[g]) for g in all_states
                if not np.all(np.isnan(L_site[g])))
    max_d = int(dist[dist>=0].max())

    n_states = len(all_states)
    ncols = 1 + n_states
    fig = plt.figure(figsize=(4*n_states, 16))
    gs  = mgrid.GridSpec(3, ncols, figure=fig,
                         hspace=0.55, wspace=0.38,
                         left=0.04, right=0.98,
                         top=0.93,  bottom=0.05)

    # ── Row 0: spatial entropy loss maps ─────────────────────────────────────
    ax_lbl0 = fig.add_subplot(gs[0,0]); ax_lbl0.axis('off')
    ax_lbl0.text(0.5,0.5,
                 "Spatial\nentropy loss\n$\\mathcal{L}(v)$\n\n"
                 "$=\\sum_t -\\ln\\frac{n}{n_{\\rm nl}}$\n"
                 "window centered\non arrival,\nwidth=delay",
                 ha='center',va='center',fontsize=9,
                 transform=ax_lbl0.transAxes,
                 bbox=dict(boxstyle='round',fc='#f5f5f5',ec='gray',alpha=0.7))

    for col,g in enumerate(all_states,start=1):
        ax=fig.add_subplot(gs[0,col])
        cmap='YlGn' if g in ('MPF','MPF_E','MPF_T','MPF_D1','MPF_D2') else 'YlOrRd'
        vals_g = L_site[g][~np.isnan(L_site[g])]
        mu_g   = vals_g.mean() if len(vals_g) > 0 else np.nan
        better = mu_g < best_noisy_mu
        marker = '↓' if better else '↑'
        scatter_map(ax, L_site[g], pos, ns, t_nl,
                    cmap, 0, L_max, '$\\mathcal{L}$',
                    f'{LABS[g]}\nμ={mu_g:.3f} {marker}')

    # ── Row 1 left: L(t) global cross-entropy per cycle ──────────────────────
    half = ncols // 2
    ax_gl = fig.add_subplot(gs[1, 0:half])
    for g in all_states:
        lw = 2.5 if g not in ('N1','N2','N3') else 1.5
        ax_gl.plot(cycles, L_glob[g], GLINE[g], color=GCOL[g], lw=lw,
                   label=LABS[g], zorder=5 if 'MPF' in g else 3)
    ax_gl.axhline(0, color='gray', lw=0.8, ls=':')
    ax_gl.set_xlabel('Cycle', fontsize=10)
    ax_gl.set_ylabel('$L(t) = \\sum_v -\\ln(n/n_{\\rm nl})$', fontsize=10)
    ax_gl.set_title('$L(t)$ — cross-entropy loss per cycle', fontsize=11)
    ax_gl.legend(fontsize=8, loc='upper left', ncol=2)
    ax_gl.grid(True, alpha=0.3)

    # ── Row 1 right: L(v) per BFS shell ──────────────────────────────────────
    ax_sh = fig.add_subplot(gs[1, half:])
    for g in all_states:
        L_v = L_site[g]
        shells, mu_L, sig_L = [], [], []
        for d in range(max_d+1):
            idx  = np.where(dist==d)[0]
            vals = L_v[idx]; vals = vals[~np.isnan(vals)]
            if len(vals)==0: continue
            shells.append(d); mu_L.append(vals.mean()); sig_L.append(vals.std())
        shells = np.array(shells); mu_L = np.array(mu_L); sig_L = np.array(sig_L)
        lw = 2.5 if g not in ('N1','N2','N3') else 1.5
        ls = GLINE[g]
        ax_sh.plot(shells, mu_L, f'o{ls}', color=GCOL[g], lw=lw, ms=5,
                   label=LABS[g], zorder=5 if 'MPF' in g else 3)
        ax_sh.fill_between(shells, mu_L-sig_L, mu_L+sig_L, alpha=0.10, color=GCOL[g])
    ax_sh.axhline(0, color='gray', lw=1, ls='--')
    ax_sh.set_xlabel('BFS shell (distance from center)', fontsize=10)
    ax_sh.set_ylabel('$\\mathcal{L}(v)$ mean ± σ', fontsize=10)
    ax_sh.set_title('$L(v)$ — entropy loss per BFS shell', fontsize=11)
    ax_sh.legend(fontsize=8, ncol=2)
    ax_sh.grid(True, alpha=0.3)

    # ── Row 2 left: α(t) time projection coefficients ────────────────────────
    colors_g = {'N1':'#2196F3','N2':'#FF9800','N3':'#E91E63'}
    ax_at = fig.add_subplot(gs[2, 0:half])
    for k, g in enumerate(gammas):
        ax_at.plot(cycles, alpha_t[:, k], '-o', color=colors_g[g],
                   lw=1.8, ms=4, label=f'$\\alpha_{{{g[1]}}}(t)$ time-proj')
    ax_at.axhline(0, color='gray', lw=0.8, ls=':')
    ax_at.axhline(1, color='gray', lw=0.8, ls=':')
    for k, g in enumerate(gammas):
        ax_at.axhline(c_mpf_e[k], color=colors_g[g], lw=1.2, ls='--', alpha=0.5,
                      label=f'global $\\alpha_{{{g[1]}}}$')
    ax_at.set_xlabel('Cycle', fontsize=10)
    ax_at.set_ylabel('$\\alpha_i(t)$', fontsize=10)
    ax_at.set_title('Time projection $\\alpha(t)$ — solid=per cycle, dashed=global',
                    fontsize=11)
    ax_at.legend(fontsize=8, ncol=2)
    ax_at.grid(True, alpha=0.3)

    # ── Row 2 right: α(d) space projection — both options ────────────────────
    ax_ad = fig.add_subplot(gs[2, half:])
    shells_d = sorted(alpha_d1.keys())
    for k, g in enumerate(gammas):
        a1 = [alpha_d1[d][k] for d in shells_d]
        a2 = [alpha_d2[d][k] for d in shells_d]
        ax_ad.plot(shells_d, a1, 'o-',  color=colors_g[g], lw=1.5, ms=4,
                   label=f'$\\alpha_{{{g[1]}}}(d)$ opt1')
        ax_ad.plot(shells_d, a2, 's--', color=colors_g[g], lw=1.5, ms=4, alpha=0.6,
                   label=f'$\\alpha_{{{g[1]}}}(d)$ opt2')
    ax_ad.axhline(0, color='gray', lw=0.8, ls=':')
    ax_ad.axhline(1, color='gray', lw=0.8, ls=':')
    ax_ad.set_xlabel('BFS shell d', fontsize=10)
    ax_ad.set_ylabel('$\\alpha_i(d)$', fontsize=10)
    ax_ad.set_title('Space projection $\\alpha(d)$ — snapshot opt1 | window opt2',
                    fontsize=11)
    ax_ad.legend(fontsize=7, ncol=2)
    ax_ad.grid(True, alpha=0.3)

    # ── Title ─────────────────────────────────────────────────────────────────
    cf_str  = ', '.join(f'{c:.2f}' for c in c_mpf)
    ce_str  = ', '.join(f'{c:.2f}' for c in c_mpf_e)
    mpft_mu = np.nanmean(L_site['MPF_T'][~np.isnan(L_site['MPF_T'])])
    print(f"  MPF α(t) time-proj: {mpft_mu:.3f}  "
          f"{'BETTER' if mpft_mu < best_noisy_mu else 'worse'}")
    fig.suptitle(
        f'MPF entropy loss recovery — {tag}\n'
        f'Frontier MPF $c$=[{cf_str}]    Entropy MPF $c$=[{ce_str}]\n'
        f'Best noisy μ={best_noisy_mu:.2f} | '
        f'Frontier MPF μ={mpf_mu:.2f} | '
        f'Entropy MPF μ={mpfe_mu:.2f} | '
        f'Time-proj μ={mpft_mu:.2f}',
        fontsize=11, y=0.97)

    out=args.out or args.log.replace('.log','_mpf_entropy.png')
    plt.savefig(out,dpi=150,bbox_inches='tight')
    print(f"\nFigure saved: {out}")

    # ── QIML figure: 4-row layout ─────────────────────────────────────────────
    # Row 0: noisy spatial maps γ=1,2,3     (red/orange)
    # Row 1: MPF recovery maps entropy+α(t) (green)
    # Row 2: L(t) cross-entropy per cycle
    # Row 3: L(d) per BFS shell
    if args.qiml:
        qiml_out = out.replace('.png','_qiml.png')
        noisy_states = [g for g in all_states if g in ('N1','N2','N3')]
        mpf_states   = [g for g in all_states if g in ('MPF_E','MPF_T')]
        show_states  = noisy_states + mpf_states

        n_noisy = len(noisy_states)   # 3
        n_mpf   = len(mpf_states)     # 2
        ncols_q = max(n_noisy, n_mpf)  # 3

        fig2 = plt.figure(figsize=(4*ncols_q, 18))
        gs2  = mgrid.GridSpec(4, ncols_q, figure=fig2,
                              hspace=0.55, wspace=0.35,
                              left=0.05, right=0.97,
                              top=0.93,  bottom=0.05,
                              height_ratios=[1, 1, 1.2, 1.2])

        # Row 0: noisy (red)
        for i, g in enumerate(noisy_states):
            ax = fig2.add_subplot(gs2[0, i])
            mu_g = np.nanmean(L_site[g][~np.isnan(L_site[g])])
            scatter_map(ax, L_site[g], pos, ns, t_nl,
                        'YlOrRd', 0, L_max, '$\\mathcal{L}$',
                        f'{LABS[g]}  μ={mu_g:.2f}')

        # Row 1: MPF recovery (green) — centre panels if fewer than ncols_q
        offset = (ncols_q - n_mpf) // 2
        for i, g in enumerate(mpf_states):
            ax = fig2.add_subplot(gs2[1, offset + i])
            mu_g = np.nanmean(L_site[g][~np.isnan(L_site[g])])
            scatter_map(ax, L_site[g], pos, ns, t_nl,
                        'YlGn', 0, L_max, '$\\mathcal{L}$',
                        f'{LABS[g]}  μ={mu_g:.2f}')

        # Row 2: L(t)
        ax_lt = fig2.add_subplot(gs2[2, :])
        for g in show_states:
            lw = 2.2 if 'MPF' in g else 1.4
            ax_lt.plot(cycles, L_glob[g], GLINE[g], color=GCOL[g],
                       lw=lw, label=LABS[g], zorder=5 if 'MPF' in g else 3)
        ax_lt.axhline(0, color='gray', lw=0.7, ls=':')
        ax_lt.set_xlabel('Cycle', fontsize=10)
        ax_lt.set_ylabel('$L(t)$', fontsize=10)
        ax_lt.set_title('Cross-entropy loss $L(t)$ per cycle', fontsize=11)
        ax_lt.legend(fontsize=9, ncol=3, loc='upper left')
        ax_lt.grid(True, alpha=0.3)

        # Row 3: L(d)
        ax_ld = fig2.add_subplot(gs2[3, :])
        for g in show_states:
            L_v = L_site[g]
            shells, mu_L, sig_L = [], [], []
            for d in range(max_d+1):
                idx  = np.where(dist==d)[0]
                vals = L_v[idx]; vals = vals[~np.isnan(vals)]
                if len(vals)==0: continue
                shells.append(d); mu_L.append(vals.mean()); sig_L.append(vals.std())
            shells = np.array(shells); mu_L = np.array(mu_L); sig_L = np.array(sig_L)
            lw = 2.2 if 'MPF' in g else 1.4
            ax_ld.plot(shells, mu_L, f'o{GLINE[g]}', color=GCOL[g],
                       lw=lw, ms=5, label=LABS[g], zorder=5 if 'MPF' in g else 3)
            ax_ld.fill_between(shells, mu_L-sig_L, mu_L+sig_L,
                               alpha=0.12, color=GCOL[g])
        ax_ld.axhline(0, color='gray', lw=0.7, ls='--')
        ax_ld.set_xlabel('BFS shell $d$ (distance from center)', fontsize=10)
        ax_ld.set_ylabel('$\\mathcal{L}(v)$ mean ± σ', fontsize=10)
        ax_ld.set_title('Entropy loss $L(d)$ per BFS shell', fontsize=11)
        ax_ld.legend(fontsize=9, ncol=3)
        ax_ld.grid(True, alpha=0.3)

        mpft_mu = np.nanmean(L_site['MPF_T'][~np.isnan(L_site['MPF_T'])])
        fig2.suptitle(
            f'Wavemap entropy recovery — {tag}   '
            f'Best noisy μ={best_noisy_mu:.2f} | '
            f'Entropy MPF μ={mpfe_mu:.2f} | '
            f'α(t) μ={mpft_mu:.2f}',
            fontsize=12, y=0.97)

        plt.figure(fig2.number)
        plt.savefig(qiml_out, dpi=150, bbox_inches='tight')
        print(f"QIML figure saved: {qiml_out}")

if __name__=='__main__':
    main()
