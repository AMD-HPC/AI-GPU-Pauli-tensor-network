#!/usr/bin/env python3
"""
check_ptm_eigenvalues.py — Inspect eigenvalues of M_gamma = gamma*M_hw + (1-gamma)*I
for a few representative heavy-hex bonds.

Usage:
  python3 scripts/check_ptm_eigenvalues.py
"""

import json
import numpy as np

NOISE_JSON = 'data/heavy_hex_3x3_gamma1.00-2.00-3.00_hardware.json'

with open(NOISE_JSON) as f:
    d = json.load(f)

# JSON structure: d[gamma_str][layer_idx][bond_id] = 16x16 PTM (row-major)
# Find the gamma=1.0 key
print("Top-level keys:", list(d.keys())[:6], "...")
gamma_keys = [k for k in d.keys() if k not in ('metadata', 'graph')]
print("Gamma keys:", gamma_keys)

# Pick a few bonds from layer 0 at gamma=1
layer_data = None
g1_key = None
for k in gamma_keys:
    try:
        val = float(k)
        if abs(val - 1.0) < 0.01:
            g1_key = k
    except:
        pass

print(f"\nUsing gamma key: {g1_key}")
layer_dict = d[g1_key]                          # keys '0','1','2'
layer0     = layer_dict['0']                     # dict: bond_key -> 256-element list
bond_keys  = list(layer0.keys())
print(f"Layers: {list(layer_dict.keys())}")
print(f"Bonds in layer 0: {len(bond_keys)}")
print(f"Sample bond keys: {bond_keys[:3]}")

for bond_key in bond_keys[:3]:
    ptm_hw = np.array(layer0[bond_key]).reshape(16, 16)   # column-major from JSON?

    print(f"\n{'='*60}")
    print(f"Bond '{bond_key}', layer 0")
    print(f"  PTM[0,0]={ptm_hw[0,0]:.4f}  (should be 1 for trace-preserving)")
    print(f"  Diagonal:      {np.round(np.diag(ptm_hw), 4)}")
    print(f"  Max off-diag:  {np.max(np.abs(ptm_hw - np.diag(np.diag(ptm_hw)))):.4f}")

    for gamma in [1.0, 2.0, 3.0]:
        Mg = gamma * ptm_hw + (1 - gamma) * np.eye(16)
        eigs = np.linalg.eigvals(Mg)
        er   = np.real(eigs)
        ei   = np.imag(eigs)
        idx  = np.argsort(er)[::-1]
        top5 = eigs[idx[:5]]
        print(f"\n  gamma={gamma:.1f}:  M = {gamma}*M_hw + ({1-gamma:.0f})*I")
        print(f"    re(eig): min={er.min():.4f}  max={er.max():.4f}  "
              f"n<0={( er<0).sum()}  n>1={(er>1).sum()}")
        print(f"    |im|max: {np.abs(ei).max():.4f}   |eig|>1: {(np.abs(eigs)>1).sum()}")
        print(f"    top 5:   {['({:.3f}{:+.3f}j)'.format(e.real,e.imag) for e in top5]}")
