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
        'MPF':'#2ECC40','MPF_E':'#9B59B6'}
LABS = {'NL':'Noiseless','N1':'γ=1','N2':'γ=2','N3':'γ=3',
        'MPF':'MPF (frontier)','MPF_E':'MPF (entropy)'}
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

    all_states = gammas + ['MPF', 'MPF_E']
    all_vt     = {**data, 'MPF': mpf_vt,  'MPF_E': mpf_e_vt}
    all_tg     = {**t_g,  'MPF': t_mpf,   'MPF_E': t_mpf_e}

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

    # ── Figure: 3 rows × (1 label + 6 panels) ────────────────────────────────
    L_max = max(np.nanmax(L_site[g]) for g in all_states
                if not np.all(np.isnan(L_site[g])))
    max_d = int(dist[dist>=0].max())

    fig = plt.figure(figsize=(22, 14))
    gs  = mgrid.GridSpec(3, 7, figure=fig,
                         hspace=0.50, wspace=0.38,
                         left=0.04, right=0.98,
                         top=0.92,  bottom=0.06)

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
        cmap='YlGn' if g=='MPF' else 'YlOrRd'
        scatter_map(ax, L_site[g], pos, ns, t_nl,
                    cmap, 0, L_max, '$\\mathcal{L}$',
                    f'{LABS[g]}\n$\\mathcal{{L}}(v)$')

    # ── Row 1: global L(t) ────────────────────────────────────────────────────
    ax_lbl1=fig.add_subplot(gs[1,0]); ax_lbl1.axis('off')
    ax_lbl1.text(0.5,0.5,
                 "Global\nentropy loss\nper cycle\n\n"
                 "$L(t)=\\sum_v$\n$-\\ln\\frac{n}{n_{\\rm nl}}$\n\n"
                 "MPF (green)\nshould sit\nnear zero",
                 ha='center',va='center',fontsize=9,
                 transform=ax_lbl1.transAxes,
                 bbox=dict(boxstyle='round',fc='#f5f5f5',ec='gray',alpha=0.7))

    ax_gl=fig.add_subplot(gs[1,1:])
    for g in all_states:
        lw=2.8 if g=='MPF' else 1.8
        ax_gl.plot(cycles, L_glob[g], '-', color=GCOL[g], lw=lw,
                   label=LABS[g], zorder=5 if g=='MPF' else 3)
    ax_gl.axhline(0,color='gray',lw=0.8,ls=':')
    ax_gl.set_xlabel('Cycle',fontsize=10)
    ax_gl.set_ylabel('$L(t) = \\sum_v -\\ln(n/n_{\\rm nl})$',fontsize=10)
    ax_gl.set_title('Global cross-entropy loss — noisy vs MPF-corrected',fontsize=11)
    ax_gl.legend(fontsize=9,loc='upper left')
    ax_gl.grid(True,alpha=0.3)

    # ── Row 2: per-shell distribution ─────────────────────────────────────────
    ax_lbl2=fig.add_subplot(gs[2,0]); ax_lbl2.axis('off')
    ax_lbl2.text(0.5,0.5,
                 "Per-BFS-shell\n$\\mathcal{L}(v)$\nmean ± σ\n\n"
                 "Removes\ndistance bias.\nMPF (green)\nshould collapse\ntoward zero\nat all shells",
                 ha='center',va='center',fontsize=9,
                 transform=ax_lbl2.transAxes,
                 bbox=dict(boxstyle='round',fc='#f5f5f5',ec='gray',alpha=0.7))

    ax_sh=fig.add_subplot(gs[2,1:])
    for g in all_states:
        L_v=L_site[g]
        shells,mu_L,sig_L=[],[],[]
        for d in range(max_d+1):
            idx=np.where(dist==d)[0]
            vals=L_v[idx]; vals=vals[~np.isnan(vals)]
            if len(vals)==0: continue
            shells.append(d); mu_L.append(vals.mean()); sig_L.append(vals.std())
        shells=np.array(shells); mu_L=np.array(mu_L); sig_L=np.array(sig_L)
        lw=2.8 if g=='MPF' else 1.8
        ax_sh.plot(shells,mu_L,'o-',color=GCOL[g],lw=lw,ms=6,
                   label=LABS[g],zorder=5 if g=='MPF' else 3)
        ax_sh.fill_between(shells,mu_L-sig_L,mu_L+sig_L,
                            alpha=0.12,color=GCOL[g])
    ax_sh.axhline(0,color='gray',lw=1,ls='--')
    ax_sh.set_xlabel('BFS shell (distance from center)',fontsize=10)
    ax_sh.set_ylabel('$\\mathcal{L}(v)$ mean ± σ',fontsize=10)
    ax_sh.set_title('Entropy loss per BFS shell — MPF recovery',fontsize=11)
    ax_sh.legend(fontsize=9)
    ax_sh.grid(True,alpha=0.3)

    # ── Title ─────────────────────────────────────────────────────────────────
    cf_str = ', '.join(f'{c:.2f}' for c in c_mpf)
    ce_str = ', '.join(f'{c:.2f}' for c in c_mpf_e)
    fig.suptitle(
        f'MPF entropy loss recovery — {tag}\n'
        f'Frontier MPF $c$=[{cf_str}]    Entropy MPF $c$=[{ce_str}]\n'
        f'Best noisy μ={best_noisy_mu:.2f} | '
        f'Frontier MPF μ={mpf_mu:.2f} | '
        f'Entropy MPF μ={mpfe_mu:.2f}',
        fontsize=11, y=0.97)

    out=args.out or args.log.replace('.log','_mpf_entropy.png')
    plt.savefig(out,dpi=150,bbox_inches='tight')
    print(f"\nFigure saved: {out}")

if __name__=='__main__':
    main()
