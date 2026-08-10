#!/usr/bin/env python3
"""
frontier_at_cycle.py — Report frontier distance per cycle for PAULI data.

Usage:
  python3 scripts/frontier_at_cycle.py --dat file.dat \
      --graph data/graphs/heavy_hex_3x3.json \
      --center 13 --threshold 1e-4
"""
import argparse, json, re, numpy as np
from collections import deque

def load_graph(path):
    with open(path) as f: d=json.load(f)
    ns = d.get('n_sites', len(d.get('vertices',[])))
    adj=[[] for _ in range(ns)]
    for e in d['edges']:
        u=e['edge'][0][0]-1; v=e['edge'][1][0]-1
        adj[u].append(v); adj[v].append(u)
    return ns,adj

def bfs(adj,center,ns):
    dist=[-1]*ns; dist[center]=0; q=deque([center])
    while q:
        v=q.popleft()
        for u in adj[v]:
            if dist[u]<0: dist[u]=dist[v]+1; q.append(u)
    return dist

def parse_dat(path):
    steps={}
    with open(path) as f:
        for line in f:
            m=re.match(r"PAULI (\d+)(.*)",line)
            if m:
                s=int(m.group(1))
                steps[s]=np.array([float(x) for x in m.group(2).split()])
    return steps

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dat',required=True)
    ap.add_argument('--graph',default='data/graphs/heavy_hex_3x3.json')
    ap.add_argument('--center',type=int,default=13)
    ap.add_argument('--threshold',type=float,default=1e-4)
    args=ap.parse_args()

    ns,adj=load_graph(args.graph)
    dist=bfs(adj,args.center,ns)
    steps=parse_dat(args.dat)

    print(f"{'Cycle':>5}  {'Frontier':>8}  {'MaxN':>10}")
    print("-"*30)
    for s in sorted(steps):
        w=steps[s]
        if len(w)!=ns: continue
        active=[dist[v] for v in range(ns) if w[v]>args.threshold]
        f=max(active) if active else 0
        print(f"{s:>5}  {f:>8}  {max(w):>10.4e}")

if __name__=='__main__':
    main()
