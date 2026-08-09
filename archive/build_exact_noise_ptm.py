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
from numpy.linalg import matrix_power

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

def gate_group_ptm(gates):
    """Compose a list of (U4: 4×4 complex unitary) into a single 16×16 PTM."""
    M = np.eye(16)
    for U4 in gates:
        M = two_qubit_ptm(U4) @ M
    return M

def julia_gate_sequence_ptm(alpha, delta, noise_ptm_16x16):
    """
    Compute exact effective PTM for one bond with interleaved noise.

    From multi_xx_plus_yy_plus_zz_gate_new in tnqs_gates.jl:
      gates1: Ry(π/2)⊗I, Rx(π)⊗I, I⊗Rx(-π), I⊗Ry(-π/2)
      [Noise]
      gates2: Rzz(α), Ry(π/2)⊗I, Rx(-π/2)⊗I, I⊗Ry(π/2), I⊗Rx(-π/2)
      [Noise]
      gates3: Rzz(α), Rx(-π/2)⊗I, I⊗Rx(-π/2)
      [Noise]
      gates4: Rzz(Δ)

    alpha, delta: Trotter parameters (already scaled by dt) from graph JSON
    noise_ptm_16x16: 16×16 noise PTM in row-major format (from hardware JSON)

    Returns: 16×16 col-major flat list (CppSim format)
    """
    # Noise PTM in matrix form (row-major → numpy)
    Mn = np.array(noise_ptm_16x16, dtype=float)  # shape (16,16)

    # Single-qubit unitaries
    ry_pos = Ry(np.pi/2)
    rx_pos = Rx(np.pi)
    rx_neg = Rx(-np.pi)
    ry_neg_half = Ry(-np.pi/2)
    rx_neg_half = Rx(-np.pi/2)
    ry_pos_half = Ry(np.pi/2)

    # gates1: Ry(π/2)⊗I, Rx(π)⊗I, I⊗Rx(-π), I⊗Ry(-π/2)
    g1 = [
        np.kron(ry_pos, np.eye(2)),
        np.kron(rx_pos, np.eye(2)),
        np.kron(np.eye(2), rx_neg),
        np.kron(np.eye(2), ry_neg_half),
    ]

    # gates2: Rzz(α), Ry(π/2)⊗I, Rx(-π/2)⊗I, I⊗Ry(π/2), I⊗Rx(-π/2)
    g2 = [
        Rzz(alpha),
        np.kron(ry_pos_half, np.eye(2)),
        np.kron(rx_neg_half, np.eye(2)),
        np.kron(np.eye(2), ry_pos_half),
        np.kron(np.eye(2), rx_neg_half),
    ]

    # gates3: Rzz(α), Rx(-π/2)⊗I, I⊗Rx(-π/2)
    g3 = [
        Rzz(alpha),
        np.kron(rx_neg_half, np.eye(2)),
        np.kron(np.eye(2), rx_neg_half),
    ]

    # gates4: Rzz(Δ)
    g4 = [Rzz(delta)]

    # Compose: PTM_eff = Noise · gates4 · Noise · gates3 · Noise · gates2 · Noise · gates1
    PTM = np.eye(16)
    PTM = gate_group_ptm(g1) @ PTM
    PTM = Mn @ PTM                  # noise after gates1
    PTM = gate_group_ptm(g2) @ PTM
    PTM = Mn @ PTM                  # noise after gates2
    PTM = gate_group_ptm(g3) @ PTM
    PTM = Mn @ PTM                  # noise after gates3
    PTM = gate_group_ptm(g4) @ PTM
    # Note: no noise after gates4 in Julia's decomposition

    # Convert to col-major flat list
    return list(PTM.flatten(order='F'))

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
        "heavyhex_exact_noise_data.hpp"

    # Trotter parameters matching Julia's compute_overlaps_gpu.jl
    epsilon_ref = float(os.environ.get('EPSILON', '0.25'))
    k_value     = int(os.environ.get('K', '1'))
    dt          = 1.0 / k_value

    print(f"Noise JSON:  {noise_json}")
    print(f"Graph JSON:  {graph_json}")
    print(f"Output:      {out_file}")
    print(f"Trotter:     epsilon={epsilon_ref}  k={k_value}  dt={dt}")
    print()

    # Load noise PTMs
    with open(noise_json) as f:
        noise_data = json.load(f)
    gammas = sorted(noise_data.keys(), key=float)
    print(f"Gammas: {gammas}")

    # Load graph and coupling constants
    with open(graph_json) as f:
        graph_data = json.load(f)

    # Build coupling maps
    # α = coupling * epsilon_ref (per Julia: α = Δ * epsilon_ref)
    # Δ = coupling (raw)
    coupling_map = {}
    for e in graph_data['edges']:
        u = e['edge'][0][0] - 1  # 0-based
        v = e['edge'][1][0] - 1
        coupling_map[(min(u,v), max(u,v))] = e['coupling']

    # Build layer assignment per bond
    layers = graph_data['layers']
    bond_layer = {}
    bond_layer_edges = {}
    for layer_id, gate_list in layers.items():
        for gate in gate_list:
            u = gate[0][0] - 1
            v = gate[1][0] - 1
            key = (min(u,v), max(u,v))
            bond_layer[key] = int(layer_id)

    # Edge list (0-based)
    EDGES = [
        (0,30),(0,31),(1,30),(1,32),(2,32),(2,33),(2,34),(3,33),(3,35),
        (4,35),(4,36),(4,37),(5,36),(5,38),(6,38),(6,39),(7,40),(7,41),
        (8,31),(8,42),(9,42),(9,43),(9,44),(10,34),(10,43),(10,45),
        (11,45),(11,46),(11,47),(12,37),(12,46),(12,48),(13,48),(13,49),(13,50),
        (14,39),(14,49),(14,51),(15,51),(15,52),(16,53),(16,54),
        (17,44),(17,53),(17,55),(18,55),(18,56),(18,57),(19,47),(19,56),(19,58),
        (20,58),(20,59),(20,60),(21,50),(21,59),(21,61),(22,40),(22,61),(22,62),
        (23,52),(23,62),(24,54),(24,63),(25,63),(25,64),(26,57),(26,64),(26,65),
        (27,65),(27,66),(28,60),(28,66),(28,67),(29,41),(29,67),
    ]
    n_bonds = len(EDGES)

    # Parse noise PTMs per bond per gamma (same as extract_noise_ptm.py)
    def parse_edge_key(key):
        m = re.findall(r'\((\d+),\)', key)
        return int(m[0])-1, int(m[1])-1

    edge_to_bond = {(u,v): i for i,(u,v) in enumerate(EDGES)}
    edge_to_bond.update({(v,u): i for i,(u,v) in enumerate(EDGES)})

    # For each gamma, build exact interleaved PTMs
    gamma_exact_ptms = {}
    for gamma_str in gammas:
        layers_data = noise_data[gamma_str]
        noise_ptm_per_bond = {}

        # Collect raw noise PTMs per bond
        for layer_idx, layer_bonds in layers_data.items():
            for edge_key, ptm in layer_bonds.items():
                u, v = parse_edge_key(edge_key)
                bond_id = edge_to_bond.get((u,v))
                if bond_id is None: continue
                noise_ptm_per_bond[bond_id] = np.array(ptm, dtype=float)  # (16,16) row-major

        # Compute exact interleaved PTM for each bond
        exact_ptms = [None] * n_bonds
        for bond_id, (u, v) in enumerate(EDGES):
            key = (min(u,v), max(u,v))
            coup = coupling_map.get(key, 1.0)
            alpha = coup * epsilon_ref * dt  # α = coupling * epsilon * dt
            delta = coup * dt               # Δ = coupling * dt (last Rzz)

            if bond_id in noise_ptm_per_bond:
                Mn = noise_ptm_per_bond[bond_id]
                ptm_flat = julia_gate_sequence_ptm(alpha, delta, Mn)
            else:
                print(f"  WARNING: no noise PTM for bond {bond_id} gamma={gamma_str}")
                ptm_flat = list(np.eye(16).flatten(order='F'))

            exact_ptms[bond_id] = ptm_flat

        gamma_exact_ptms[gamma_str] = exact_ptms
        print(f"  gamma={gamma_str}: {n_bonds} exact PTMs computed")

    # Write output (same format as extract_noise_ptm.py)
    def gamma_to_cname(g):
        return g.replace('.', '_')

    lines = []
    lines.append("// heavyhex_exact_noise_data.hpp — auto-generated by build_exact_noise_ptm.py")
    lines.append("// Exact interleaved PTM: Noise·gates4·Noise·gates3·Noise·gates2·Noise·gates1")
    lines.append(f"// Source: {os.path.basename(noise_json)}")
    lines.append(f"// Gammas: {', '.join(gammas)}")
    lines.append(f"// Trotter: epsilon={epsilon_ref}  k={k_value}  dt={dt}")
    lines.append("#pragma once")
    lines.append("#include <array>")
    lines.append("")
    lines.append(f"static constexpr int HEAVY_HEX_3x3_EXACT_N_GAMMA = {len(gammas)};")
    lines.append(f"static constexpr float HEAVY_HEX_3x3_EXACT_GAMMAS[{len(gammas)}] = {{{', '.join(f'{float(g):.1f}f' for g in gammas)}}};")
    lines.append("")

    for gamma_str in gammas:
        cname = gamma_to_cname(gamma_str)
        ptms = gamma_exact_ptms[gamma_str]
        lines.append(f"static const float HEAVY_HEX_3x3_EXACT_GAMMA{cname}[{n_bonds}][256] = {{")
        for bond_id, ptm in enumerate(ptms):
            u, v = EDGES[bond_id]
            vals = ', '.join(f'{v:.8f}f' for v in ptm)
            lines.append(f"    /* bond {bond_id:2d} ({u:2d},{v:2d}) */ {{{vals}}},")
        lines.append("};")
        lines.append("")

    # Lookup function
    lines.append("inline const float* heavy_hex_3x3_exact_noise_ptm(int bond_id, float gamma) {")
    lines.append("    int best = 0;")
    lines.append("    float best_diff = fabsf(gamma - HEAVY_HEX_3x3_EXACT_GAMMAS[0]);")
    lines.append(f"    for (int i = 1; i < HEAVY_HEX_3x3_EXACT_N_GAMMA; i++) {{")
    lines.append("        float diff = fabsf(gamma - HEAVY_HEX_3x3_EXACT_GAMMAS[i]);")
    lines.append("        if (diff < best_diff) { best = i; best_diff = diff; }")
    lines.append("    }")
    for i, gamma_str in enumerate(gammas):
        cname = gamma_to_cname(gamma_str)
        if i < len(gammas)-1:
            lines.append(f"    if (best == {i}) return HEAVY_HEX_3x3_EXACT_GAMMA{cname}[bond_id];")
        else:
            lines.append(f"    return HEAVY_HEX_3x3_EXACT_GAMMA{cname}[bond_id];")
    lines.append("}")

    with open(out_file, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nWritten: {out_file}  ({os.path.getsize(out_file)/1e6:.1f} MB)")

if __name__ == '__main__':
    main()
