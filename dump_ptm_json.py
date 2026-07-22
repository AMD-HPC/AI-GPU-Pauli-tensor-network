"""
dump_ptm_json.py — Dump composed PTM for all bonds to JSON for Julia comparison

Produces: ptm_eps010.json
  { "bond_id": [[16x16 PTM col-major], ...] }

Julia will apply this as a single gate (CppSim style) and compare
with sequential Rxx·Ryy·Rzz (Julia style).

Usage: python3 dump_ptm_json.py
"""
import json, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_ptm import (PTM_gates1, PTM_gates2, PTM_gates3, PTM_gates4,
                       compose, load_graph, EDGES)

epsilon = 0.1
phi     = 1.41371669
GRAPH   = "data/graphs/heavy_hex_3x3.json"

alpha_map, delta_map = load_graph(GRAPH, epsilon)

result = {"epsilon": epsilon, "phi": phi, "bonds": []}

for bond_id, (u, v) in enumerate(EDGES):
    key   = (min(u,v), max(u,v))
    alpha = alpha_map[key]
    delta = delta_map[key]
    G1    = PTM_gates1()
    G2    = PTM_gates2(alpha)
    G3    = PTM_gates3(alpha)
    G4    = PTM_gates4(delta)
    PTM   = compose([G1, G2, G3, G4])  # 16×16 col-major
    result["bonds"].append({
        "bond_id": bond_id,
        "sites":   [u, v],
        "alpha":   alpha,
        "delta":   delta,
        "ptm":     PTM.flatten(order='F').tolist()  # col-major
    })

out = "data/ptm_eps010.json"
with open(out, 'w') as f:
    json.dump(result, f)
print(f"Written: {out}  ({len(result['bonds'])} bonds, epsilon={epsilon})")
