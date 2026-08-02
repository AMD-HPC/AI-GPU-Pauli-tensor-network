"""
build_ptm.py — Compositional PTM builder for heavy-hex gate sequences

Defines gate primitives and composes them into named models:

  IDEAL       : gates1 · gates2(α) · gates3(α) · gates4(δ)
  IDEAL_KICK  : gates1 · gates2(α) · gates3(α) · gates4(δ) · kick(φ)
  NOISY_KICK  : gates1 · noise(γ) · gates2(α) · noise(γ) · gates3(α) · noise(γ) · gates4(δ) · kick(φ)
  NOISY       : gates1 · noise(γ) · gates2(α) · noise(γ) · gates3(α) · noise(γ) · gates4(δ)

Parameters (from Julia compute_overlaps_gpu.jl):
  epsilon = 0.25   (Trotter step)
  phi     = 1.41371669  (kick angle)
  α = coupling × epsilon  (per bond)
  δ = coupling            (per bond)

Usage:
  python3 build_ptm.py [noise_json] [graph_json] [model] [output.hpp]

  model: IDEAL | IDEAL_KICK | NOISY | NOISY_KICK

Defaults:
  noise_json : data/heavy_hex_3x3_gamma1.00-1.40-1.80_hardware.json
  graph_json : data/graphs/heavy_hex_3x3.json
  model      : IDEAL_KICK
  output.hpp : include/heavyhex_<model>_data.hpp
"""

import json, sys, os, re
import numpy as np

# ── Pauli matrices ─────────────────────────────────────────────────────────────
I2 = np.eye(2, dtype=complex)
X  = np.array([[0,1],[1,0]], dtype=complex)
Y  = np.array([[0,-1j],[1j,0]], dtype=complex)
Z  = np.array([[1,0],[0,-1]], dtype=complex)
PAULIS = [I2, X, Y, Z]

# ── PTM conversion ─────────────────────────────────────────────────────────────

def single_qubit_ptm(U):
    """4×4 real PTM for single-qubit unitary U in Pauli basis.
    Schrodinger convention: M_S[i,j] = Tr(Pi U Pj U†) / 2
    """
    M = np.zeros((4,4))
    for i, Pi in enumerate(PAULIS):
        for j, Pj in enumerate(PAULIS):
            M[i,j] = np.real(np.trace(Pi.conj().T @ U @ Pj @ U.conj().T)) / 2
    return M

def two_qubit_ptm(U4):
    """16×16 real PTM for two-qubit unitary U4 in Pauli⊗Pauli basis.

    Julia's toitensor_heisenberg computes M_H then stores transpose(M_H).
    So Julia applies M_S (Schrodinger: O → U O U†) not M_H (O → U† O U).
    M_S[i,j] = Tr(Pi U Pj U†) / 4
    """
    M = np.zeros((16,16))
    for i, Pi in enumerate(PAULIS):
        for j, Pj in enumerate(PAULIS):
            PiPj = np.kron(Pi, Pj)
            for k, Pk in enumerate(PAULIS):
                for l, Pl in enumerate(PAULIS):
                    PkPl = np.kron(Pk, Pl)
                    M[i*4+j, k*4+l] = np.real(
                        np.trace(PiPj.conj().T @ U4 @ PkPl @ U4.conj().T)) / 4
    return M

# ── Single-qubit gates ─────────────────────────────────────────────────────────

def Rx(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)

def Ry(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def Rzz(theta):
    """exp(-i theta/2 Z⊗Z)"""
    return np.diag([np.exp(-1j*theta/2), np.exp(1j*theta/2),
                    np.exp(1j*theta/2), np.exp(-1j*theta/2)])

def embed(U, site, n=2):
    """Embed single-qubit gate U at given site in n-qubit system."""
    ops = [np.eye(2, dtype=complex)] * n
    ops[site] = U
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result

# ── Gate groups (16×16 PTMs) ───────────────────────────────────────────────────

def PTM_gates1():
    """Ry(π/2)⊗I · Rx(π)⊗I · I⊗Rx(-π) · I⊗Ry(-π/2)"""
    M = np.eye(16)
    for U in [embed(Ry(np.pi/2), 0),
              embed(Rx(np.pi),   0),
              embed(Rx(-np.pi),  1),
              embed(Ry(-np.pi/2),1)]:
        M = two_qubit_ptm(U) @ M
    return M

def PTM_gates2(alpha):
    """Rzz(α) · Ry(π/2)⊗I · Rx(-π/2)⊗I · I⊗Ry(π/2) · I⊗Rx(-π/2)"""
    M = np.eye(16)
    for U in [Rzz(alpha),
              embed(Ry(np.pi/2),  0),
              embed(Rx(-np.pi/2), 0),
              embed(Ry(np.pi/2),  1),
              embed(Rx(-np.pi/2), 1)]:
        M = two_qubit_ptm(U) @ M
    return M

def PTM_gates3(alpha):
    """Rzz(α) · Rx(-π/2)⊗I · I⊗Rx(-π/2)"""
    M = np.eye(16)
    for U in [Rzz(alpha),
              embed(Rx(-np.pi/2), 0),
              embed(Rx(-np.pi/2), 1)]:
        M = two_qubit_ptm(U) @ M
    return M

def PTM_gates4(delta):
    """Rzz(δ)"""
    return two_qubit_ptm(Rzz(delta))

def PTM_kick(phi):
    """
    Rx(2φ) on both sites — returns a 16×16 PTM.
    In Heisenberg picture this acts on the two-site operator.
    """
    Mk = single_qubit_ptm(Rx(2*phi))
    return np.kron(Mk, Mk)

def PTM_kick_site(phi):
    """Rx(2φ) on a single site — 4×4 PTM."""
    return single_qubit_ptm(Rx(2*phi))

# ── Gate sequence composer ─────────────────────────────────────────────────────

def compose(sequence):
    """Multiply a list of PTMs left-to-right (first gate applied first)."""
    M = np.eye(sequence[0].shape[0])
    for G in sequence:
        M = G @ M
    return M

# ── Named models ───────────────────────────────────────────────────────────────

def build_bond_ptm(model, alpha, delta, phi, noise_ptm_16x16):
    """
    Compute the effective 16×16 PTM for one bond.

    model: 'IDEAL' | 'IDEAL_KICK' | 'NOISY' | 'NOISY_KICK'
    alpha : coupling × epsilon  (Rzz angle for gates2, gates3)
    delta : coupling            (Rzz angle for gates4)
    phi   : kick angle
    noise_ptm_16x16: 16×16 numpy array for this bond's color class, or None for ideal
                     Applied as: G1 · Mn · G2 · Mn · G3 · Mn · G4
                     Same Mn used three times — it's the hardware noise for this
                     bond's color class, applied between each sub-gate pair
    """
    G1 = PTM_gates1()
    G2 = PTM_gates2(alpha)
    G3 = PTM_gates3(alpha)
    G4 = PTM_gates4(delta)
    Mn = np.array(noise_ptm_16x16, dtype=float) if noise_ptm_16x16 is not None else None

    # Note: kick is NOT baked into bond PTM — applied separately per site
    if model in ('IDEAL', 'IDEAL_KICK'):
        seq = [G1, G2, G3, G4]
    elif model in ('NOISY', 'NOISY_KICK'):
        assert Mn is not None
        seq = [G1, Mn, G2, Mn, G3, Mn, G4]
    else:
        raise ValueError(f"Unknown model: {model}")

    return compose(seq)

# ── Graph + noise data loading ─────────────────────────────────────────────────

# 76 bonds (0-based vertex ids) — from heavy_hex_3x3.json
# EDGES and edge_to_bond are populated from the graph JSON at runtime in main()
EDGES = []
n_bonds = 0
edge_to_bond = {}

def parse_edge_key(key):
    m = re.findall(r'\((\d+),\)', key)
    return int(m[0])-1, int(m[1])-1

def load_graph(graph_path, epsilon):
    global EDGES, n_bonds, edge_to_bond
    with open(graph_path) as f:
        d = json.load(f)
    coupling_map = {}
    edges_list = []
    for e in d['edges']:
        u = e['edge'][0][0] - 1
        v = e['edge'][1][0] - 1
        coupling_map[(min(u,v), max(u,v))] = e['coupling']
        edges_list.append((min(u,v), max(u,v)))
    # Populate global EDGES in consistent order
    EDGES = sorted(set(edges_list))
    n_bonds = len(EDGES)
    edge_to_bond = {(u,v): i for i,(u,v) in enumerate(EDGES)}
    edge_to_bond.update({(v,u): i for i,(u,v) in enumerate(EDGES)})
    # Julia convention: Δ = -coupling, α = Δ × epsilon = -coupling × epsilon
    alpha_map = {k: -v * epsilon for k, v in coupling_map.items()}
    delta_map  = {k: -v           for k, v in coupling_map.items()}
    return alpha_map, delta_map

def load_noise(noise_path):
    with open(noise_path) as f:
        d = json.load(f)
    gammas = sorted(d.keys(), key=float)
    # noise_per_gamma[gamma_str][layer_idx][bond_id] = 16×16 numpy array
    # Each bond has a separate noise PTM per color class (layer 0, 1, 2)
    noise_per_gamma = {}
    for gamma_str in gammas:
        layers_data = d[gamma_str]
        layers_dict = {}
        for layer_idx, layer_bonds in layers_data.items():
            ptm_per_bond = {}
            for edge_key, ptm in layer_bonds.items():
                u, v = parse_edge_key(edge_key)
                bond_id = edge_to_bond.get((u,v))
                if bond_id is not None:
                    ptm_per_bond[bond_id] = np.array(ptm, dtype=float)
            layers_dict[int(layer_idx)] = ptm_per_bond
        noise_per_gamma[gamma_str] = layers_dict
    return gammas, noise_per_gamma

# ── HPP writer ─────────────────────────────────────────────────────────────────

def write_hpp(out_path, model, gammas, ptms_by_gamma, epsilon, phi, graph_data=None):
    """
    ptms_by_gamma: dict gamma_str -> list of 76 flat col-major PTMs
    For ideal models: gammas=['ideal'], ptms_by_gamma={'ideal': [...]}
    """
    lines = []
    lines.append(f"// {os.path.basename(out_path)} — auto-generated by build_ptm.py")
    lines.append(f"// Model: {model}")
    lines.append(f"// epsilon={epsilon}  phi={phi}")
    if gammas != ['ideal']:
        lines.append(f"// Gammas: {', '.join(gammas)}")
    lines.append("#pragma once")
    lines.append("#include <cmath>")
    lines.append("")
    lines.append(f"static constexpr float HEAVY_HEX_3x3_{model}_EPSILON = {epsilon:.8f}f;")
    lines.append(f"static constexpr float HEAVY_HEX_3x3_{model}_PHI     = {phi:.8f}f;")
    lines.append(f"static constexpr float HEAVY_HEX_3x3_{model}_DT      = {epsilon:.8f}f;  // physical time step per cycle")
    lines.append("")

    # Initial state from graph JSON — +1=|0>, -1=|1>
    initial_state = graph_data.get('initial_state', {})
    if initial_state:
        # Build array indexed by 0-based site id (Julia 1-based keys like "(3,)" → site 2)
        import re as _re
        state_array = [0] * n_bonds  # n_bonds happens to equal n_sites=68 here, use 68
        state_array = [0] * 68
        for key, val in initial_state.items():
            m = _re.findall(r'\((\d+),\)', key)
            if m:
                site_0 = int(m[0]) - 1  # 1-based → 0-based
                state_array[site_0] = int(val)
        vals = ', '.join(str(v) for v in state_array)
        lines.append(f"// Initial device state: +1=|0> (I+Z)/2, -1=|1> (I-Z)/2")
        lines.append(f"static constexpr int HEAVY_HEX_3x3_{model}_INITIAL_STATE[68] = {{{vals}}};")
        lines.append("")

    # 4×4 single-site kick PTM (col-major) — only for kick models
    if 'KICK' in model:
        Mk = PTM_kick_site(phi)
        vals = ', '.join(f'{x:.8f}f' for x in Mk.flatten(order='F'))
        lines.append(f"// Rx(2*phi) single-site PTM [4x4 col-major] — applied once per site after bond gates")
        lines.append(f"static const float HEAVY_HEX_3x3_{model}_SITE_PTM[16] = {{{vals}}};")
        lines.append("")

    is_ideal = (gammas == ['ideal'])

    if is_ideal:
        ptms = ptms_by_gamma['ideal']
        lines.append(f"static const float HEAVY_HEX_3x3_{model}[{n_bonds}][256] = {{")
        for bond_id, ptm in enumerate(ptms):
            u, v = EDGES[bond_id]
            vals = ', '.join(f'{x:.8f}f' for x in ptm)
            lines.append(f"    /* bond {bond_id:2d} ({u:2d},{v:2d}) */ {{{vals}}},")
        lines.append("};")
        lines.append("")
        lines.append(f"inline const float* heavy_hex_3x3_{model.lower()}_ptm(int bond_id) {{")
        lines.append(f"    return HEAVY_HEX_3x3_{model}[bond_id];")
        lines.append("}")
    else:
        n_gamma = len(gammas)
        gamma_floats = [float(g) for g in gammas]
        lines.append(f"static constexpr int   HEAVY_HEX_3x3_{model}_N_GAMMA = {n_gamma};")
        lines.append(f"static constexpr float HEAVY_HEX_3x3_{model}_GAMMAS[{n_gamma}] = "
                     f"{{{', '.join(f'{g:.1f}f' for g in gamma_floats)}}};")
        lines.append("")
        for gamma_str in gammas:
            cname = gamma_str.replace('.', '_')
            ptms  = ptms_by_gamma[gamma_str]
            lines.append(f"static const float HEAVY_HEX_3x3_{model}_GAMMA{cname}[{n_bonds}][256] = {{")
            for bond_id, ptm in enumerate(ptms):
                u, v = EDGES[bond_id]
                vals = ', '.join(f'{x:.8f}f' for x in ptm)
                lines.append(f"    /* bond {bond_id:2d} ({u:2d},{v:2d}) */ {{{vals}}},")
            lines.append("};")
            lines.append("")
        lines.append(f"inline const float* heavy_hex_3x3_{model.lower()}_ptm(int bond_id, float gamma) {{")
        lines.append(f"    int best = 0;")
        lines.append(f"    float best_diff = fabsf(gamma - HEAVY_HEX_3x3_{model}_GAMMAS[0]);")
        lines.append(f"    for (int i = 1; i < HEAVY_HEX_3x3_{model}_N_GAMMA; i++) {{")
        lines.append(f"        float diff = fabsf(gamma - HEAVY_HEX_3x3_{model}_GAMMAS[i]);")
        lines.append(f"        if (diff < best_diff) {{ best = i; best_diff = diff; }}")
        lines.append(f"    }}")
        for i, gamma_str in enumerate(gammas):
            cname = gamma_str.replace('.', '_')
            if i < n_gamma - 1:
                lines.append(f"    if (best == {i}) return HEAVY_HEX_3x3_{model}_GAMMA{cname}[bond_id];")
            else:
                lines.append(f"    return HEAVY_HEX_3x3_{model}_GAMMA{cname}[bond_id];")
        lines.append("}")

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Written: {out_path}  ({os.path.getsize(out_path)//1024} KB)")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Parse --flag=value args first, then positional args
    epsilon = float(os.environ.get('EPSILON', '0.25'))
    phi     = float(os.environ.get('PHI',     '1.41371669'))
    gamma_filter = None  # None = all gammas; '1.0' = gamma=1.0 only
    graph_override = None
    positional = []
    for arg in sys.argv[1:]:
        if arg.startswith('--epsilon='):
            epsilon = float(arg[10:])
        elif arg.startswith('--phi='):
            phi = float(arg[6:])
        elif arg.startswith('--gamma='):
            gamma_filter = arg[8:]
        elif arg.startswith('--graph='):
            graph_override = arg[8:]
        else:
            positional.append(arg)

    noise_json = positional[0] if len(positional) > 0 else \
        os.path.join(script_dir, "data", "heavy_hex_3x3_gamma1.00-1.40-1.80_hardware.json")
    graph_json = graph_override if graph_override else \
        (positional[1] if len(positional) > 1 else \
        os.path.join(script_dir, "data", "graphs", "heavy_hex_3x3.json"))
    model      = positional[2] if len(positional) > 2 else "IDEAL_KICK"
    out_file   = positional[3] if len(positional) > 3 else None

    # Encode epsilon and gamma in model name to avoid HPP symbol conflicts
    # e.g. NOISY + epsilon=0.1 + gamma=2.0 → model_tag = "NOISY_EPS010_GAMMA2"
    eps_tag = str(epsilon).replace('.', '')[:4].ljust(3, '0')  # "0.1"→"010", "0.15"→"015"
    model_tag = f"{model}_EPS{eps_tag}"
    if gamma_filter is not None:
        gam_tag = gamma_filter.replace('.', '')[:4].rstrip('0').rstrip('.')
        model_tag = f"{model_tag}_GAMMA{gam_tag}"

    # Default output uses model_tag so files don't overwrite each other
    if out_file is None:
        out_file = os.path.join(script_dir, "include",
                                f"heavyhex_{model_tag.lower()}_data.hpp")

    print(f"Model:      {model}  (tag: {model_tag})")
    print(f"Noise JSON: {noise_json}")
    print(f"Graph JSON: {graph_json}")
    print(f"Output:     {out_file}")
    print(f"epsilon={epsilon}  phi={phi}")
    print()

    with open(graph_json) as f:
        graph_data = json.load(f)
    alpha_map, delta_map = load_graph(graph_json, epsilon)

    is_ideal = model in ('IDEAL', 'IDEAL_KICK')

    if is_ideal:
        ptms = []
        for bond_id, (u, v) in enumerate(EDGES):
            key = (min(u,v), max(u,v))
            alpha = alpha_map.get(key, epsilon)
            delta = delta_map.get(key, 1.0)
            ptm = build_bond_ptm(model, alpha, delta, phi, None)
            ptms.append(list(ptm.flatten(order='F')))
        print(f"  {model_tag}: {n_bonds} PTMs computed")
        write_hpp(out_file, model_tag, ['ideal'], {'ideal': ptms}, epsilon, phi, graph_data)
    else:
        gammas, noise_per_gamma = load_noise(noise_json)
        if gamma_filter is not None:
            # Keep only the closest gamma to the requested value
            target = float(gamma_filter)
            best = min(gammas, key=lambda g: abs(float(g)-target))
            gammas = [best]
            print(f"  Filtering to gamma={best} (requested {gamma_filter})")
        ptms_by_gamma = {}
        # Build bond → color class mapping from graph JSON
        # Each bond belongs to exactly one color class (0, 1, or 2)
        bond_color = {}
        for layer_idx, layer_bonds in noise_per_gamma[gammas[0]].items():
            for bond_id in layer_bonds:
                bond_color[bond_id] = layer_idx

        for gamma_str in gammas:
            ptms = []
            layers_dict = noise_per_gamma[gamma_str]  # {layer_idx: {bond_id: Mn}}
            for bond_id, (u, v) in enumerate(EDGES):
                key = (min(u,v), max(u,v))
                alpha = alpha_map.get(key, epsilon)
                delta = delta_map.get(key, 1.0)
                # Get the noise PTM for this bond's color class
                color = bond_color.get(bond_id, None)
                if color is not None and bond_id in layers_dict.get(color, {}):
                    Mn = layers_dict[color][bond_id]
                else:
                    print(f"  WARNING: no noise PTM for bond {bond_id} gamma={gamma_str}")
                    Mn = np.eye(16)
                ptm = build_bond_ptm(model, alpha, delta, phi, Mn)
                ptms.append(list(ptm.flatten(order='F')))
            ptms_by_gamma[gamma_str] = ptms
            print(f"  gamma={gamma_str}: {n_bonds} PTMs computed (per-bond color-class noise)")
        write_hpp(out_file, model_tag, gammas, ptms_by_gamma, epsilon, phi, graph_data)

if __name__ == '__main__':
    main()
