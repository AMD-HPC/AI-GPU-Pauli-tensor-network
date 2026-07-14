"""
build_exact_noise_ptm.py — Exact interleaved noise PTM for heavy-hex gates

Computes the physically correct effective PTM per bond by interleaving
hardware noise channels at the correct positions within each Trotter step:

    PTM_eff = M_ε · M_C · M_ε · M_B · M_ε · M_A

where M_A, M_B, M_C are gate PTMs (Ry, Rx, Rzz in Pauli basis) and
M_ε is the hardware noise PTM for this bond.

This is EXACT — no NOISE_POWER approximation.

The output format is identical to extract_noise_ptm.py so heavyhex_noise_data.hpp
can be used directly in CppSim without any code changes.

Usage:
    python3 build_exact_noise_ptm.py [noise_json] [graph_json] [output.hpp]

Defaults:
    noise_json:  data/heavy_hex_3x3_gamma1.00-1.40-1.80_hardware.json
    graph_json:  data/graphs/heavy_hex_3x3.json  (for coupling constants)
    output.hpp:  heavyhex_exact_noise_data.hpp    (note: different name!)

TODO: implement after experiment wave completes.
"""

import json
import sys
import os
import re
import numpy as np

# ── Pauli matrices ────────────────────────────────────────────────────────────
I2 = np.eye(2)
X  = np.array([[0,1],[1,0]], dtype=complex)
Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)
PAULIS = [I2, X, Y, Z]  # I=0, X=1, Y=2, Z=3

# ── Single-qubit PTM (4×4 real) ───────────────────────────────────────────────

def single_qubit_ptm(U):
    """Compute 4×4 real PTM for single-qubit unitary U in Pauli basis.
    PTM[i,j] = Re(Tr(Pi† U† Pj U)) / 2
    """
    M = np.zeros((4,4))
    for i, Pi in enumerate(PAULIS):
        for j, Pj in enumerate(PAULIS):
            M[i,j] = np.real(np.trace(Pi.conj().T @ U.conj().T @ Pj @ U)) / 2
    return M

def Ry(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c,-s],[s,c]], dtype=complex)

def Rx(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c,-1j*s],[-1j*s,c]], dtype=complex)

# ── Two-qubit PTM (16×16 real) ────────────────────────────────────────────────

def two_qubit_ptm(U4):
    """Compute 16×16 real PTM for two-qubit unitary U4 in Pauli⊗Pauli basis.
    PTM[i*4+j, k*4+l] = Re(Tr((Pi⊗Pj)† U4† (Pk⊗Pl) U4)) / 4
    """
    M = np.zeros((16,16))
    for i, Pi in enumerate(PAULIS):
        for j, Pj in enumerate(PAULIS):
            row = i*4+j
            PiPj = np.kron(Pi, Pj)
            for k, Pk in enumerate(PAULIS):
                for l, Pl in enumerate(PAULIS):
                    col = k*4+l
                    PkPl = np.kron(Pk, Pl)
                    M[row,col] = np.real(
                        np.trace(PiPj.conj().T @ U4.conj().T @ PkPl @ U4)) / 4
    return M

def Rzz(theta):
    """exp(-i theta/2 Z⊗Z)"""
    return np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2),
                    np.exp(1j*theta/2), np.exp(-1j*theta/2)])

def embed_single(U, site, n_sites=2):
    """Embed single-qubit gate U at given site in n_sites system."""
    ops = [np.eye(2)] * n_sites
    ops[site] = U
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

# ── Julia gate sequence per color class ──────────────────────────────────────
# From multi_xx_plus_yy_plus_zz_gate_new:
#   gates1 = Ry(π/2)⊗I, Rx(π)⊗I, I⊗Rx(-π), I⊗Ry(-π/2)
#   [Noise]
#   gates2 = Rzz(α), Ry(π/2)⊗I, Rx(-π/2)⊗I, I⊗Ry(π/2), I⊗Rx(-π/2)
#   [Noise]
#   gates3 = Rzz(α), Rx(-π/2)⊗I, I⊗Rx(-π/2)
#   [Noise]
#   gates4 = Rzz(Δ)

def julia_gate_sequence_ptm(alpha, delta, noise_ptm_16x16):
    """
    Compute exact effective PTM for one bond with interleaved noise.

    alpha, delta: Trotter parameters from graph JSON
    noise_ptm_16x16: 16×16 noise PTM for this bond (from hardware JSON)

    Returns: 16×16 col-major flat list (CppSim format)

    TODO: implement — placeholder returns identity for now.
    """
    # TODO: implement gate sequence PTM composition
    # Step 1: compute PTM for each gate group (gates1, gates2, gates3, gates4)
    # Step 2: interleave noise PTM between groups
    # Step 3: compose all PTMs via matrix multiplication
    raise NotImplementedError("TODO: implement after experiment wave completes")

# ── Parse edge and coupling data ──────────────────────────────────────────────

def parse_graph(graph_path):
    """Load coupling constants per edge from graph JSON."""
    with open(graph_path) as f:
        d = json.load(f)
    couplings = {}
    for e in d['edges']:
        u = e['edge'][0][0] - 1  # 1-based → 0-based
        v = e['edge'][1][0] - 1
        couplings[(min(u,v), max(u,v))] = e['coupling']
    return couplings

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    noise_json = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(script_dir, "data", "heavy_hex_3x3_gamma1.00-1.40-1.80_hardware.json")
    graph_json = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(script_dir, "data", "graphs", "heavy_hex_3x3.json")
    out_file   = sys.argv[3] if len(sys.argv) > 3 else \
        "heavyhex_exact_noise_data.hpp"  # note: different from extract_noise_ptm.py

    print(f"Noise JSON:  {noise_json}")
    print(f"Graph JSON:  {graph_json}")
    print(f"Output:      {out_file}")
    print()

    # TODO: implement full pipeline
    # 1. Load noise PTMs from JSON
    # 2. Load coupling constants from graph JSON
    # 3. For each bond, compute exact interleaved PTM via julia_gate_sequence_ptm()
    # 4. Write output in same format as extract_noise_ptm.py

    print("STATUS: Not yet implemented — see MPF_notes.md for design.")
    print("Run extract_noise_ptm.py with NOISE_POWER=25 as approximation.")

if __name__ == '__main__':
    main()
