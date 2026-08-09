"""
test_ptm_equivalence.py — Verify PTM composition equivalence

Part 1: Mathematical verification
  - Apply gates sequentially vs composed PTM to a known state vector
  - Should be identical (PTM composition = sequential application by definition)
  - Also verify Julia's gate decomposition matches CppSim's PTM

Part 2: Understand epsilon and delta in the gate sequence
  - Show what each sub-gate (G1,G2,G3,G4) does physically
  - Show how alpha (XX+YY) scales with epsilon but delta (ZZ) does not

Usage: python3 test_ptm_equivalence.py
"""

import numpy as np
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_ptm import (PTM_gates1, PTM_gates2, PTM_gates3, PTM_gates4,
                       PTM_kick, PTM_kick_site, compose, load_graph,
                       two_qubit_ptm, single_qubit_ptm, embed,
                       Rzz, Rx, Ry, I2, X, Y, Z, PAULIS)

print("="*70)
print("Part 1: PTM composition equivalence")
print("="*70)

# ── Setup: use bond 0 parameters ──────────────────────────────────────────────
GRAPH = "data/graphs/heavy_hex_3x3.json"
epsilon = 0.1
phi = 1.41371669

alpha_map, delta_map = load_graph(GRAPH, epsilon)
from build_ptm import EDGES
bond_id = 0
u, v = EDGES[bond_id]
key = (min(u,v), max(u,v))
alpha = alpha_map[key]
delta = delta_map[key]

print(f"\nBond {bond_id} ({u},{v}): coupling={-delta:.6f}")
print(f"  epsilon={epsilon}  alpha={alpha:.6f}  delta={delta:.6f}")
print(f"  alpha/delta = epsilon = {alpha/delta:.6f}")

# ── Test 1: Sequential application = composed PTM ─────────────────────────────
print("\n── Test 1: Sequential vs Composed ───────────────────────────────────")

G1 = PTM_gates1()
G2 = PTM_gates2(alpha)
G3 = PTM_gates3(alpha)
G4 = PTM_gates4(delta)

# Initial state: Z⊗I operator = [0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0,0,0]
# In Pauli basis: P_{ZI} component = 1, all others = 0
state = np.zeros(16); state[3] = 1.0  # ZI = index 3 (Z=3, I=0 → 3*4+0=12? No: I=0,X=1,Y=2,Z=3)
# Actually: P_i ⊗ P_j has index i*4+j. ZI = 3*4+0 = 12
state = np.zeros(16); state[12] = 1.0  # ZI

# Sequential: apply G1, then G2, then G3, then G4
s1 = G1 @ state
s2 = G2 @ s1
s3 = G3 @ s2
s4 = G4 @ s3

# Composed: single matrix multiplication
PTM_composed = compose([G1, G2, G3, G4])
s_composed = PTM_composed @ state

print(f"Initial state ZI: index 12 = {state[12]:.4f}")
print(f"After sequential G1·G2·G3·G4:")
nz = np.where(np.abs(s4) > 1e-6)[0]
for i in nz: print(f"  [{i:2d}] = {s4[i]:+.8f}")
print(f"After composed PTM:")
nz2 = np.where(np.abs(s_composed) > 1e-6)[0]
for i in nz2: print(f"  [{i:2d}] = {s_composed[i]:+.8f}")
print(f"Max |sequential - composed| = {np.max(np.abs(s4 - s_composed)):.2e}")
print("✓ Sequential == Composed" if np.allclose(s4, s_composed) else "✗ MISMATCH!")

# ── Test 2: What does each sub-gate do? ───────────────────────────────────────
print("\n── Test 2: Individual gate effects on ZI state ──────────────────────")
print(f"{'Gate':<8} {'ZI component':>14} {'ZZ component':>14} {'II component':>14}")
print(f"{'----':<8} {'-'*14:>14} {'-'*14:>14} {'-'*14:>14}")
s = state.copy()
for name, G in [("initial", np.eye(16)), ("G1", G1), ("G1·G2", G2),
                ("·G3", G3), ("·G4", G4)]:
    if name != "initial": s = G @ s
    else: s = G @ state
    s = G @ (state if name=="initial" else (G1@state if name=="G1" else s))

# Redo properly
s = state.copy()
print(f"{'initial':<8} {s[12]:+14.6f} {s[15]:+14.6f} {s[0]:+14.6f}")
for name, G in [("after G1", G1), ("after G2", G2), ("after G3", G3), ("after G4", G4)]:
    s = G @ s
    print(f"  {name:<10} {s[12]:+14.6f} {s[15]:+14.6f} {s[0]:+14.6f}")

# ── Test 3: Alpha scales with epsilon, delta does not ─────────────────────────
print("\n── Test 3: Epsilon scaling ───────────────────────────────────────────")
for eps in [0.05, 0.10, 0.15, 0.20]:
    am, dm = load_graph(GRAPH, eps)
    a = am[key]; d = dm[key]
    PTM = compose([PTM_gates1(), PTM_gates2(a), PTM_gates3(a), PTM_gates4(d)])
    s = PTM @ state
    print(f"  eps={eps:.2f}: alpha={a:.6f}  delta={d:.6f}  ZI→ZI={s[12]:+.6f}  ZI→ZZ={s[15]:+.6f}")

print(f"\n  Note: delta is constant ({dm[key]:.6f}) across all epsilon values")
print(f"        alpha scales linearly with epsilon")

# ── Test 4: Verify PTM matches Julia's Rxx·Ryy·Rzz decomposition ─────────────
print("\n── Test 4: Julia gate decomposition vs CppSim PTM ──────────────────")
# Julia applies: Rxx(α·dt), Ryy(α·dt), Rzz(δ·dt) with dt=1
# CppSim G1·G2·G3·G4 = hardware decomposition of exp(-i·α·(XX+YY)) · exp(-i·δ·ZZ)
# Let's compute Rxx·Ryy·Rzz directly and compare

def Rxx(theta): return embed(np.cos(theta)*np.eye(2) - 1j*np.sin(theta)*X, 0) @ \
                       embed(np.cos(theta)*np.eye(2) - 1j*np.sin(theta)*X, 1)
# Actually: Rxx(θ) = exp(-iθ XX/2) — standard definition
def rxx_gate(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c,0,0,-1j*s],[0,c,-1j*s,0],[0,-1j*s,c,0],[-1j*s,0,0,c]])
def ryy_gate(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c,0,0,1j*s],[0,c,-1j*s,0],[0,-1j*s,c,0],[1j*s,0,0,c]])
def rzz_gate(theta):
    e = np.exp(-1j*theta/2)
    return np.diag([e, e.conj(), e.conj(), e])

dt = 1.0
alpha_julia = alpha * dt  # = alpha (dt=1 for k=1)
delta_julia = delta * dt

Rxx_ptm = two_qubit_ptm(rxx_gate(alpha_julia))
Ryy_ptm = two_qubit_ptm(ryy_gate(alpha_julia))
Rzz_ptm = two_qubit_ptm(rzz_gate(delta_julia))

PTM_julia = compose([Rxx_ptm, Ryy_ptm, Rzz_ptm])
PTM_cppsim = compose([G1, G2, G3, G4])

s_julia  = PTM_julia  @ state
s_cppsim = PTM_cppsim @ state

print(f"Julia  Rxx·Ryy·Rzz → ZI component: {s_julia[12]:+.8f}")
print(f"CppSim G1·G2·G3·G4 → ZI component: {s_cppsim[12]:+.8f}")
print(f"Max |Julia - CppSim| = {np.max(np.abs(PTM_julia - PTM_cppsim)):.2e}")
if np.allclose(PTM_julia, PTM_cppsim, atol=1e-6):
    print("✓ Julia Rxx·Ryy·Rzz == CppSim G1·G2·G3·G4")
else:
    print("✗ MISMATCH — different gate decompositions!")
    # Show where they differ
    diff = np.abs(PTM_julia - PTM_cppsim)
    print(f"  Max diff at position {np.unravel_index(diff.argmax(), diff.shape)}")
    print(f"  Julia  PTM diagonal: {np.round(np.diag(PTM_julia),4)}")
    print(f"  CppSim PTM diagonal: {np.round(np.diag(PTM_cppsim),4)}")

print("\n" + "="*70)
print("Summary")
print("="*70)
print("1. Sequential application == Composed PTM: mathematically exact")
print("2. alpha = coupling × epsilon (scales with time step)")
print("3. delta = coupling (fixed, ZZ term independent of epsilon)")
print("4. Julia Rxx·Ryy·Rzz vs CppSim G1·G2·G3·G4: see above")
print("\nIf Test 4 shows MISMATCH → the hardware decomposition in CppSim")
print("implements a DIFFERENT gate than Julia's direct Rxx·Ryy·Rzz.")
print("This would explain the systematic ~1.6%% offset at cycle 1.")
