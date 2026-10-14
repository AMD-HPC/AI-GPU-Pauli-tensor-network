#!/usr/bin/env python3
"""
bfs_detect.py — BFS-guided Hamiltonian fault detection via threshold crossing.

Key insight: n_base(v,t) and n_pert(v,t) define their own BFS frontiers.
t*_base(v) = first cycle where n_base(v,t) > thr  — baseline wave arrives
t*_pert(v) = first cycle where n_pert(v,t) > thr  — perturbed wave arrives

Signal at the frontier:
  t*_pert > t*_base  → delayed arrival → RED (shadow behind broken bond)
  t*_pert = t*_base, n_pert < n_base → attenuated → RED
  t*_pert = t*_base, n_pert > n_base → accumulated → BLUE (wall face)
  t*_pert never crosses thr → complete shadow → strongly RED

Collision detection:
  n_base(v, t*) grows very fast (>2x per cycle) → multiple paths arriving
  simultaneously → collision node → unreliable signal, downweight

Broken bond identification:
  For each anomaly site v at t*_base(v):
    - Examine neighbors u of v
    - The broken bond is (u,v) where u crossed threshold BEFORE v
      (u is on the center side, v is in the shadow)

Multi-scan: collect all anomaly bonds across scans, rank by total score.

Usage:
  python3 scripts/bfs_detect.py --logdir logs/mawgm_f64_2b/trial_4 \
      --plan data/scans/rect_7x7_scan_3gpu.json \
      --graph data/graphs/rect_7x7.json
"""

import argparse
import json
import numpy as np
import sys
from collections import deque
sys.path.insert(0, 'scripts')
from searcher import parse_pauli_log, load_graph, build_adj, get_bond_sites

FLOOR = 1e-7


def bond_color_class(g, bi):
    """Color class of bond bi: 0,1=horizontal, 2,3=vertical."""
    from searcher import get_bond_sites
    a, b = get_bond_sites(g['edges'][bi])
    row_a = (a-1)//7; col_a = (a-1)%7
    row_b = (b-1)//7
    if row_a == row_b:  # horizontal
        return col_a % 2
    else:  # vertical
        return 2 + row_a % 2


def bfs_with_color(centers, adj, all_sites, g):
    """
    BFS from centers tracking (hop, color_class) for sub-cycle causality.
    Returns:
      hop[v]        = min hop distance from any center
      color[v]      = color class of last bond on BFS path
      parent_bond[v]= bond_idx on BFS path to v
      causal_order[v]= (hop, color_class) — true sub-cycle ordering
    """
    hop          = {s: -1 for s in all_sites}
    color        = {s: -1 for s in all_sites}
    parent_bond  = {s: -1 for s in all_sites}

    q = deque()
    for c in centers:
        if c in hop:
            hop[c] = 0; color[c] = -1
            q.append(c)

    while q:
        v = q.popleft()
        for u, bi in adj[v]:
            if hop[u] >= 0: continue
            c_bond = bond_color_class(g, bi)
            hop[u] = hop[v] + 1
            color[u] = c_bond
            parent_bond[u] = bi
            q.append(u)

    return hop, color, parent_bond


def threshold_crossing(data, thr, T_max):
    """
    For each site v: first cycle t* where data[t,v] > thr.
    Returns t_cross[v] (1-indexed), or -1 if never crosses.
    """
    nc = min(data.shape[0], T_max)
    ns = data.shape[1]
    t_cross = np.full(ns, -1, dtype=int)
    for v in range(ns):
        for t in range(nc):
            if data[t, v] > thr:
                t_cross[v] = t + 1  # 1-indexed
                break
    return t_cross


def is_collision(db, v, t_star, site_idx, s, thr, T_max):
    """
    Check if site v is a collision node at t_star:
    n_base grew very fast — two wavefronts arriving simultaneously.
    Heuristic: n_base(v, t_star) > 3 * n_base(v, t_star-1)
    """
    t = t_star - 1  # 0-indexed
    if t < 1: return False
    nb_now  = db[t,   site_idx[s]]
    nb_prev = db[t-1, site_idx[s]]
    if nb_prev < thr / 10: return False  # previous was essentially zero
    return nb_now > 3.0 * nb_prev


def bond_transfer_L(db, dp, si, ui, tb_s, tb_u, tp_s, tp_u):
    """
    Bond transfer coefficient ratio: ln(T_base / T_pert)
    where T = n(destination, t*_dest) / n(source, t*_source)

    When source is well established (n_base(source) >> thr), the
    threshold condition ensures n_base(source) ≈ n_pert(source),
    so this reduces to L(destination) = -ln(n_pert/n_base).
    But using explicit source normalization is more accurate.
    """
    if tb_s < 1 or tb_u < 1: return 0.0
    nb_s = db[tb_s-1, si]; nb_u = db[tb_u-1, ui]
    if nb_s < 1e-10 or nb_u < 1e-10: return 0.0

    # Perturbed: use t*_pert if available, else t*_base
    tp_s_use = tp_s if tp_s > 0 else tb_s
    tp_u_use = tp_u if tp_u > 0 else tb_u
    np_s = dp[min(tp_s_use-1, dp.shape[0]-1), si]
    np_u = dp[min(tp_u_use-1, dp.shape[0]-1), ui]
    if np_s < 1e-10: return 0.0

    T_base = nb_u / nb_s
    T_pert = max(np_u, FLOOR) / max(np_s, FLOOR)
    if T_base < 1e-10 or T_pert < 1e-10: return 0.0

    return np.log(T_base / T_pert)


def bfs_detect_scan(db, dp, centers, bonds_covered, g, adj, all_sites, site_idx,
                    v_hop, thr=1e-3, thr_L=0.05, T_max=5, verbose=False):
    """
    BFS fault detection with (hop, color_class) sub-cycle causal ordering.
    No Voronoi filter — all bonds scored across all scans.

    For each bond (u→v) where u is BFS upstream of v:
      Compute bond transfer ratio: ln(T_base/T_pert)
      This directly measures the coupling change, normalized by source.
      Score using: transfer ratio + delay + BFS causal order (hop, color_class).
      Earlier in BFS (smaller hop, earlier color) = more reliable.
    """
    t_base = threshold_crossing(db, thr, T_max)
    t_pert  = threshold_crossing(dp, thr, T_max)

    # BFS with color class for sub-cycle causal ordering
    hop_bfs, color_bfs, _ = bfs_with_color(centers, adj, all_sites, g)

    ns = db.shape[1]
    L_frontier = np.zeros(ns)
    for v in range(ns):
        t = t_base[v]
        if t < 0: continue
        nb = db[t-1, v]; np_ = dp[t-1, v]
        L_frontier[v] = -np.log(max(np_, FLOOR) / nb)

    candidates = []
    for si, s in enumerate(all_sites):
        t = t_base[si]
        if t < 0: continue
        L = L_frontier[si]
        if abs(L) < thr_L: continue

        is_col = is_collision(db, s, t, site_idx, s, thr, T_max)
        t_p = t_pert[si]
        delayed = (t_p < 0) or (t_p > t)

        hop_s   = hop_bfs.get(s, 999)
        color_s = color_bfs.get(s, 9)

        best_bond = -1; best_u = -1; best_score = -1

        for u, bond_idx in adj[s]:
            # No Voronoi filter — score ALL neighboring bonds
            ui = site_idx[u]
            tu_b = t_base[ui]; tu_p = t_pert[ui]
            if tu_b < 0: continue

            nb_u = db[tu_b-1,ui]; np_u = dp[tu_b-1,ui]
            L_u = -np.log(max(np_u,FLOOR)/nb_u) if nb_u>thr else 0.0

            sign_flip = (L > 0 and L_u < -0.05) or (L < 0 and L_u > 0.05)
            if not sign_flip: continue

            hop_u   = hop_bfs.get(u, 999)
            color_u = color_bfs.get(u, 9)

            # Causal order: earlier in BFS tree is center-side (wall face)
            u_upstream = (hop_u, color_u) < (hop_s, color_s)
            s_upstream = (hop_s, color_s) < (hop_u, color_u)

            u_delayed = (tu_p < 0) or (tu_p > tu_b)

            # Bond transfer ratio: ln(T_base/T_pert)
            # Source is BFS-upstream site, destination is downstream
            if u_upstream:
                src_i, dst_i = ui, si
                tb_src, tb_dst = tu_b, t
                tp_src, tp_dst = tu_p, t_p
            else:
                src_i, dst_i = si, ui
                tb_src, tb_dst = t, tu_b
                tp_src, tp_dst = t_p, tu_p

            transfer = bond_transfer_L(db, dp, src_i, dst_i,
                                       tb_src, tb_dst, tp_src, tp_dst)

            score = 10  # base for sign flip
            score += min(20, abs(transfer) * 10)  # transfer ratio magnitude
            if delayed or u_delayed: score += 5
            if u_upstream or s_upstream: score += 3
            # BFS proximity bonus: earlier = stronger evidence
            bfs_min = min(hop_s*10+color_s, hop_u*10+color_u)
            score += max(0, 30 - bfs_min)

            if score > best_score:
                best_score = score; best_u = u; best_bond = bond_idx

        if best_bond < 0: continue

        candidates.append({
            'site':         s,
            'bond':         best_bond,
            'L':            L,
            't_base':       t,
            't_pert':       t_p,
            'delayed':      delayed,
            'is_collision': is_col,
            'upstream':     best_u,
            'hop':          hop_s,
            'color':        color_s,
        })

    return candidates


def bfs_detect(logdir, plan, g, adj, all_sites, site_idx, v_hop,
               thr=1e-3, thr_L=0.05, T_max=5, verbose=True):
    """
    Multi-scan BFS fault detection via threshold crossing.
    Returns ranked list of candidate broken bonds.
    """
    bond_score  = {}
    bond_detail = {}

    for sc in plan['placements']:
        sid     = sc['scan']
        centers = sc['centers']
        # No Voronoi filter — pass ALL bonds to each scan
        all_bonds = set(range(len(g['edges'])))

        try:
            _, _, db = parse_pauli_log(f'{logdir}/scan{sid}_base.log', 'NL')
            _, _, dp = parse_pauli_log(f'{logdir}/scan{sid}_pert.log', 'NL')
        except Exception as e:
            if verbose: print(f"  Scan {sid}: ERROR {e}")
            continue

        cands = bfs_detect_scan(db, dp, centers, all_bonds, g, adj,
                                 all_sites, site_idx, v_hop, thr, thr_L, T_max,
                                 verbose=False)

        if verbose:
            print(f"\n  Scan {sid} centers={centers}:")
            print(f"  {'site':>5}  {'bond':>5}  {'L':>8}  {'t*b':>5}  {'t*p':>5}  {'delay':>6}  {'col?'}")
            print(f"  " + "-"*58)

        for c in cands:
            bi = c['bond']
            # Score: |L| weighted by 1/t*_base (earlier = more reliable)
            # Bonus if perturbed was delayed or never arrived
            weight = abs(c['L']) / c['t_base']
            if c['delayed']: weight *= 2.0   # delay = strong evidence
            if c['is_collision']: weight *= 0.3  # collision = less reliable

            if bi not in bond_score:
                bond_score[bi]  = 0.0
                bond_detail[bi] = []
            bond_score[bi] += weight
            c['scan'] = sid
            bond_detail[bi].append(c)

            if verbose:
                a, b = get_bond_sites(g['edges'][bi])
                col = 'YES' if c['is_collision'] else 'no'
                dly = 'YES' if c['delayed'] else 'no'
                print(f"  {c['site']:>5}  {bi:>3}({a}↔{b})  {c['L']:>+8.3f}  "
                      f"{c['t_base']:>5}  {c['t_pert']:>5}  {dly:>6}  {col}")

    ranked = sorted(bond_score.items(), key=lambda x: -x[1])

    if verbose and ranked:
        print(f"\n  Bond ranking:")
        for bi, score in ranked[:8]:
            a, b = get_bond_sites(g['edges'][bi])
            evs = bond_detail[bi]
            n_sc = len(set(e['scan'] for e in evs))
            print(f"    Bond {bi:3d} ({a}↔{b})  score={score:.4f}  "
                  f"n_scans={n_sc}  n_ev={len(evs)}")

    return ranked, bond_detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir',  required=True)
    ap.add_argument('--plan',    default='data/scans/rect_7x7_scan_3gpu.json')
    ap.add_argument('--graph',   default='data/graphs/rect_7x7.json')
    ap.add_argument('--v_hop',   type=float, default=0.448)
    ap.add_argument('--thr',     type=float, default=1e-3)
    ap.add_argument('--thr_L',   type=float, default=0.05)
    ap.add_argument('--T_max',   type=int,   default=5)
    ap.add_argument('--top_k',   type=int,   default=2)
    ap.add_argument('--broken',  default=None)
    args = ap.parse_args()

    g = load_graph(args.graph)
    adj, bond_map, all_sites = build_adj(g)
    site_idx = {s: i for i, s in enumerate(all_sites)}
    plan = json.load(open(args.plan))

    print(f"BFS threshold-crossing fault detection: {args.logdir}")
    ranked, detail = bfs_detect(
        args.logdir, plan, g, adj, all_sites, site_idx,
        v_hop=args.v_hop, thr=args.thr, thr_L=args.thr_L,
        T_max=args.T_max, verbose=True)

    declared = [bi for bi, _ in ranked[:args.top_k]]
    print(f"\nTop-{args.top_k} declared bonds: {declared}")
    for bi in declared:
        a, b = get_bond_sites(g['edges'][bi])
        print(f"  Bond {bi}: {a}↔{b}")

    if args.broken:
        br = json.load(open(args.broken))
        true_bonds = set(br.get('bonds', [br.get('bond', -1)]))
        found = set(declared)
        both    = true_bonds.issubset(found)
        partial = bool(true_bonds & found) and not both
        tag = '✓ BOTH' if both else ('~ PARTIAL' if partial else '✗ NONE')
        print(f"\nGround truth: {sorted(true_bonds)}")
        print(f"Result: {tag}")

if __name__ == '__main__':
    main()
