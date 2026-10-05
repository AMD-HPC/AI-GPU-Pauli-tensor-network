#!/usr/bin/env python3
"""
breaker.py — C_breaker: perturb one bond in a grid simulation.

The Hamiltonian Tomography Game:
  C_breaker chooses a bond and a physically meaningful perturbation.
  C_searcher must identify the broken bond from wavemap differences.

Perturbation modes:
  --coupling   δJ: scale the coupling constant on one bond (J → J*(1+delta))
  --depol      δp: increase depolarizing noise on one bond (p → p+delta)

Output: modified PTM data (.npy) and a run script for main_pauli_sweep_rect7x7
or main_pauli_sweep_hh_dep with the perturbed bond.

Usage:
  # List all bonds in a graph:
  python3 scripts/breaker.py --graph data/graphs/rect_7x7.json --list

  # Perturb bond 12 by +20% coupling:
  python3 scripts/breaker.py \\
      --graph data/graphs/rect_7x7.json \\
      --bond 12 --coupling 0.2 \\
      --out logs/broken_r7_bond12_dJ02.json

  # Perturb bond 12 by +5% depolarizing noise:
  python3 scripts/breaker.py \\
      --graph data/graphs/rect_7x7.json \\
      --bond 12 --depol 0.05 \\
      --out logs/broken_r7_bond12_dp005.json

The output JSON describes the perturbation and can be passed to searcher.py.
"""

import argparse
import json
import numpy as np
from pathlib import Path

# ── Pauli matrices ─────────────────────────────────────────────────────────────
I2 = np.eye(2, dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)
Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)
PAULIS_1Q = [I2, X, Y, Z]
PAULIS_2Q = [np.kron(a,b) for a in PAULIS_1Q for b in PAULIS_1Q]

def two_qubit_ptm(U4):
    """16×16 real PTM for two-qubit unitary U4 in Pauli⊗Pauli basis."""
    M = np.zeros((16,16))
    for i, Pi in enumerate(PAULIS_2Q):
        for j, Pj in enumerate(PAULIS_2Q):
            M[i,j] = np.real(np.trace(Pi.conj().T @ U4 @ Pj @ U4.conj().T)) / 4
    return M

def heisenberg_gate(J, eps):
    """Two-qubit PTM for exp(-i*eps*J*(XX+YY+ZZ))."""
    theta = eps * J
    c, s = np.cos(theta), np.sin(theta)
    # In computational basis:
    U = np.diag([np.exp(-1j*theta), 1, 1, 1])
    U[1,1] = c; U[1,2] = -1j*s
    U[2,1] = -1j*s; U[2,2] = c
    U[3,3] = np.exp(-1j*theta)
    return two_qubit_ptm(U)

def depol_ptm(p):
    """16×16 PTM for two-qubit depolarizing channel with rate p."""
    # Depolarizing: rho -> (1-p)*rho + p/16 * I
    # In Pauli basis: diagonal matrix, I component = 1, rest = (1-p)
    M = np.eye(16)
    M[1:, 1:] *= (1 - p)
    return M

def validate_ptm(M, name="PTM"):
    """Check PTM is physically valid: eigenvalues of transfer matrix in [-1,1]."""
    eigs = np.linalg.eigvals(M)
    max_eig = np.max(np.abs(eigs))
    if max_eig > 1.0 + 1e-6:
        print(f"  WARNING: {name} has |eigenvalue|={max_eig:.4f} > 1 (unphysical)")
        return False
    print(f"  {name}: max|eigenvalue|={max_eig:.4f} ✓")
    return True

def load_graph(path):
    with open(path) as f:
        g = json.load(f)
    return g

def get_bond_sites(e):
    """Extract (site_a, site_b) from edge dict or list."""
    if isinstance(e, dict):
        sites = e['edge']
        return sites[0][0] if isinstance(sites[0], list) else sites[0], \
               sites[1][0] if isinstance(sites[1], list) else sites[1]
    return e[0], e[1]

def list_bonds(g, graph_path=''):
    print(f"Graph: {g.get('name', graph_path)} — {g['n_sites']} sites, {len(g['edges'])} bonds")
    print(f"{'Bond':>4}  {'Site A':>6}  {'Site B':>6}  {'Coupling':>10}")
    print("-" * 35)
    for i, e in enumerate(g['edges']):
        a, b = get_bond_sites(e)
        coupling = e.get('coupling', 1.0) if isinstance(e, dict) else 1.0
        print(f"  {i:>3}  {a:>6}  {b:>6}  {coupling:>10.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph',    required=True, help='Graph JSON file')
    ap.add_argument('--bond',     type=int,      help='Bond index to perturb')
    ap.add_argument('--coupling', type=float,    help='δJ: coupling perturbation fraction (e.g. 0.2 = +20%%)')
    ap.add_argument('--depol',    type=float,    help='δp: depolarizing noise increase (e.g. 0.05)')
    ap.add_argument('--J',        type=float, default=1.0,   help='Base coupling J (default: 1.0)')
    ap.add_argument('--eps',      type=float, default=0.10,  help='Trotter step ε (default: 0.10)')
    ap.add_argument('--p0',       type=float, default=0.01,  help='Base depol rate p0 (default: 0.01)')
    ap.add_argument('--out',      default=None,  help='Output JSON describing perturbation')
    ap.add_argument('--list',     action='store_true', help='List all bonds and exit')
    args = ap.parse_args()

    g = load_graph(args.graph)

    if args.list:
        list_bonds(g, args.graph)
        return

    if args.bond is None:
        print("ERROR: --bond required (use --list to see bonds)")
        return

    if args.coupling is None and args.depol is None:
        print("ERROR: specify --coupling or --depol")
        return

    n_bonds = len(g['edges'])
    if args.bond >= n_bonds:
        print(f"ERROR: bond {args.bond} out of range (0..{n_bonds-1})")
        return

    bond = g['edges'][args.bond]
    site_a, site_b = get_bond_sites(bond)
    base_coupling = bond.get('coupling', args.J) if isinstance(bond, dict) else args.J

    print(f"\nBreaker configuration:")
    print(f"  Graph:  {args.graph}")
    print(f"  Bond:   {args.bond} ({site_a} ↔ {site_b})")

    result = {
        'graph':  args.graph,
        'bond':   args.bond,
        'site_a': site_a,
        'site_b': site_b,
        'J':      args.J,
        'eps':    args.eps,
        'p0':     args.p0,
        'perturbation': {}
    }

    # ── Baseline PTM ──────────────────────────────────────────────────────────
    M_gate_base = heisenberg_gate(args.J, args.eps)
    M_noise_base = depol_ptm(args.p0)
    M_base = M_noise_base @ M_gate_base
    print(f"\nBaseline PTM (J={args.J}, ε={args.eps}, p={args.p0}):")
    validate_ptm(M_base, "Baseline")

    # ── Perturbed PTM ─────────────────────────────────────────────────────────
    if args.coupling is not None:
        J_perturbed = args.J * (1 + args.coupling)
        M_gate_pert = heisenberg_gate(J_perturbed, args.eps)
        M_pert = M_noise_base @ M_gate_pert
        print(f"\nCoupling perturbation: J={args.J} → {J_perturbed:.4f} (δJ={args.coupling:+.2f})")
        validate_ptm(M_pert, "Perturbed")
        result['perturbation'] = {
            'type': 'coupling',
            'delta': args.coupling,
            'J_base': args.J,
            'J_perturbed': J_perturbed
        }

    elif args.depol is not None:
        p_perturbed = args.p0 + args.depol
        if p_perturbed > 1.0:
            print(f"ERROR: p_perturbed={p_perturbed} > 1 (unphysical)")
            return
        M_noise_pert = depol_ptm(p_perturbed)
        M_pert = M_noise_pert @ M_gate_base
        print(f"\nDepol perturbation: p={args.p0} → {p_perturbed:.4f} (δp={args.depol:+.4f})")
        validate_ptm(M_pert, "Perturbed")
        result['perturbation'] = {
            'type': 'depol',
            'delta': args.depol,
            'p_base': args.p0,
            'p_perturbed': p_perturbed
        }

    # ── PTM difference ────────────────────────────────────────────────────────
    diff = np.abs(M_pert - M_base)
    print(f"\nPTM difference: max={diff.max():.6f}  Frobenius={np.linalg.norm(diff):.6f}")

    # ── Save result ───────────────────────────────────────────────────────────
    result['ptm_base']      = M_base.tolist()
    result['ptm_perturbed'] = M_pert.tolist()
    result['ptm_diff_max']  = float(diff.max())
    result['ptm_diff_frob'] = float(np.linalg.norm(diff))

    out = args.out or f"logs/broken_bond{args.bond}.json"
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nBreaker config saved: {out}")
    print(f"\nNext: run searcher.py --broken {out} --centers 24,10,38 --log <baseline.log>")

if __name__ == '__main__':
    main()
