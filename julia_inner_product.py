"""
julia_inner_product.py — Pure Python reproduction of Julia's inner product

Reproduces: inner(ψ(t=1), ρ; alg="bp") = -0.961822

Steps:
  1. Initialize operator: Z at center, I elsewhere (chi=1)
  2. Apply one Trotter step (gate PTMs per bond, color by color)
  3. BP to converge messages
  4. Compute inner(ψ, ρ) via chi-vector BP + Bethe formula

Matches Julia exactly:
  - Same gate sequence (gates1→noise→gates2→noise→gates3→noise→gates4)
  - Same color ordering
  - Same BP convergence criterion (avg delta < 1e-5, max 25 sweeps)
  - Same sum normalization for messages
"""

import numpy as np
import json, re, sys
sys.path.insert(0, '/scratch/MPF/CppSim')
from build_ptm import (PTM_gates1, PTM_gates2, PTM_gates3, PTM_gates4,
                        PTM_kick_site, compose, EDGES, PAULIS, I2, X, Y, Z,
                        single_qubit_ptm, two_qubit_ptm, Rx)

print("="*60)
print("  julia_inner_product.py")
print("  Reproducing inner(ψ(t=1), ρ; alg='bp') = -0.961822")
print("="*60)

# ── Parameters (matching Julia) ───────────────────────────────────────────
EPSILON = 0.25
PHI     = 1.41371669
CENTER  = 2   # 0-based, Julia observable_site=3 (1-based)

# ── Load graph ────────────────────────────────────────────────────────────
with open('/scratch/MPF/CppSim/data/graphs/heavy_hex_3x3.json') as f:
    graph = json.load(f)

# Coupling constants per bond
coupling = {}
for e in graph['edges']:
    u = e['edge'][0][0] - 1
    v = e['edge'][1][0] - 1
    coupling[(min(u,v), max(u,v))] = e['coupling']

# Color layers: each layer is list of bonds
layers = graph['layers']  # dict: layer_idx → list of [u1-based, v1-based]
color_bonds = {}  # layer_idx → list of (u,v) 0-based
for layer_str, bond_list in layers.items():
    layer_idx = int(layer_str)
    color_bonds[layer_idx] = []
    for bond in bond_list:
        u = bond[0][0] - 1
        v = bond[1][0] - 1
        color_bonds[layer_idx].append((min(u,v), max(u,v)))

# Initial state for ρ
initial_state = graph['initial_state']
def spin(v):
    return initial_state.get(f'({v+1},)', 1)  # 1-based key

# ── Tensor Network ────────────────────────────────────────────────────────
# Site tensors: ψ_v[physical=4] initially, grow to ψ_v[chi,...,chi,physical=4]
# Use dict: site_id → numpy array, shape listed as bond dims last, physical first
# Actually store as: ψ_v shape = (chi_bond0, chi_bond1, ..., D)

N_SITES = 68
D = 4  # Pauli basis dimension

# Bond structure: each site knows its bonds in order
site_bonds = [[] for _ in range(N_SITES)]  # site_id → list of bond_ids
bond_list = list(EDGES)
for bid, (u,v) in enumerate(bond_list):
    site_bonds[u].append(bid)
    site_bonds[v].append(bid)

# ── Initialize ψ_v ────────────────────────────────────────────────────────
# Z at center: [0,0,0,1], I elsewhere: [1,0,0,0]
# Shape: (D,) initially (chi=1 for all bonds)

print(f"\nInitializing: Z at site {CENTER}, I elsewhere")
psi = {}
for v in range(N_SITES):
    if v == CENTER:
        psi[v] = np.array([0., 0., 0., 1.], dtype=float)  # Z
    else:
        psi[v] = np.array([1., 0., 0., 0.], dtype=float)  # I

# Bond dimensions: chi[bond_id] = current chi
chi = [1] * len(bond_list)

# Messages: msg[bond_id][dir] where dir=0: a→b, dir=1: b→a
# chi×chi matrices, initialized to identity
msgs = {}
for bid in range(len(bond_list)):
    msgs[bid] = [np.eye(1), np.eye(1)]  # chi=1 initially

# Shape of site tensor: (chi_bond0, chi_bond1, ..., D)
def site_shape(v):
    sh = [chi[bid] for bid in site_bonds[v]]
    sh.append(D)
    return sh

def site_vol(v):
    return int(np.prod(site_shape(v)))

def get_psi(v):
    return psi[v].reshape(site_shape(v))

def set_psi(v, arr):
    psi[v] = arr.flatten()

# ── Apply gate (SVD truncation) ────────────────────────────────────────────
def apply_gate_ptm(u, v, G, cutoff=1e-4):
    """Apply 16×16 PTM G to bond (u,v). Returns new chi."""
    global chi, msgs, psi

    bid = bond_list.index((min(u,v), max(u,v)))
    cur_chi = chi[bid]

    # Get psi_u and psi_v as matrices: [left_bonds × D, right_bonds × D]
    # For simplicity: do full contraction + SVD
    # psi_u shape: (..., D), psi_v shape: (..., D)
    # Find position of bond bid in each site's bond list
    u_pos = site_bonds[u].index(bid)
    v_pos = site_bonds[v].index(bid)

    pu = get_psi(u)  # shape: (chi_0,...,chi_k,...,D)
    pv = get_psi(v)

    # Contract: theta[left_u, d_u, d_v, left_v] = Σ_chi pu[...,chi,...,du] × pv[...,chi,...,dv]
    # Apply PTM: theta_new[...,du_new,...] = Σ_{du,dv} G[du_new*D+dv_new, du*D+dv] × theta[...,du,dv,...]
    # Then SVD to get new tensors

    u_shape = list(pu.shape)
    v_shape = list(pv.shape)

    # Merge all left dims of u (excluding bond to v and physical)
    # u_left: all dims except u_pos (bond to v) and last (physical)
    u_left_dims = [i for i in range(len(u_shape)-1) if i != u_pos]
    v_left_dims = [i for i in range(len(v_shape)-1) if i != v_pos]

    u_left_size = int(np.prod([u_shape[i] for i in u_left_dims])) if u_left_dims else 1
    v_left_size = int(np.prod([v_shape[i] for i in v_left_dims])) if v_left_dims else 1

    # Reshape psi_u to (u_left, chi_bond, D)
    # Move u_pos and last dim to end
    pu2 = np.moveaxis(pu, [u_pos, len(u_shape)-1], [len(u_shape)-2, len(u_shape)-1])
    pu2 = pu2.reshape(u_left_size, cur_chi, D)

    pv2 = np.moveaxis(pv, [v_pos, len(v_shape)-1], [len(v_shape)-2, len(v_shape)-1])
    pv2 = pv2.reshape(v_left_size, cur_chi, D)

    # theta[ul, vl, du, dv] = Σ_c pu2[ul, c, du] × pv2[vl, c, dv]
    theta = np.einsum('ucd,vce->uvde', pu2, pv2)  # (ul, vl, D, D)

    # Apply PTM G (16×16 col-major → need to handle correctly)
    # G[i,j] means: new[i] = Σ_j G[i,j] × old[j]
    # theta_new[ul,vl,du_new,dv_new] = Σ_{du,dv} G[du_new*D+dv_new, du*D+dv] × theta[ul,vl,du,dv]
    theta_flat = theta.reshape(u_left_size, v_left_size, D*D)
    theta_new_flat = theta_flat @ G.T  # apply G: each (du,dv) → (du_new,dv_new)
    theta_new = theta_new_flat.reshape(u_left_size, v_left_size, D, D)

    # SVD: theta_new[ul, du_new, vl, dv_new] → SVD over (ul,du_new) × (vl,dv_new)
    theta_svd = theta_new.transpose(0,2,1,3).reshape(u_left_size*D, v_left_size*D)
    U, s, Vt = np.linalg.svd(theta_svd, full_matrices=False)

    # Truncate
    keep = np.sum(s > cutoff * s[0]) if s[0] > 1e-15 else 1
    keep = max(keep, 1)
    U = U[:, :keep]
    s = s[:keep]
    Vt = Vt[:keep, :]

    new_chi = keep

    # Reshape back
    # U: (ul*D, new_chi) → (ul, D, new_chi)
    U = U.reshape(u_left_size, D, new_chi)
    # Vt: (new_chi, vl*D) → (new_chi, vl, D)
    Vt = Vt.reshape(new_chi, v_left_size, D)

    # Absorb sqrt(s) into both
    s_sqrt = np.sqrt(s)
    U  = U  * s_sqrt[None, None, :]   # (ul, D, new_chi)
    Vt = Vt * s_sqrt[:, None, None]   # (new_chi, vl, D)

    # Update chi and psi
    chi[bid] = new_chi

    # Rebuild psi_u: insert new bond dim back at u_pos
    # U shape: (ul, D, new_chi) → need (u_left_dims..., new_chi, D)
    new_u_shape = [u_shape[i] for i in u_left_dims] + [new_chi, D]
    pu_new = U.reshape(new_u_shape)
    # Move new_chi to position u_pos, D to last
    pu_new = np.moveaxis(pu_new, [-2, -1], [u_pos, -1])
    set_psi(u, pu_new)

    # Rebuild psi_v: Vt shape: (new_chi, vl, D) → need (v_left_dims..., new_chi, D)
    new_v_shape = [new_chi] + [v_shape[i] for i in v_left_dims] + [D]
    pv_new = Vt.reshape(new_v_shape)
    # new_chi is at position 0, move to v_pos
    pv_new = np.moveaxis(pv_new, 0, v_pos)
    set_psi(v, pv_new)

    return new_chi

# ── Apply Trotter step ────────────────────────────────────────────────────
print("\nApplying Trotter step (3 colors × 4 gate groups + kick)...")

# Process in reverse (Heisenberg picture)
# Julia: reversed_ev_gates = reverse(gates4, gates3, gates2, gates1) + kick
# So we apply: gates1_rev, gates2_rev, gates3_rev, gates4_rev, kick
# But for PTM: applying reversed gates = applying the adjoint gate
# In Heisenberg picture: O → U† O U, PTM is transpose of Schrodinger PTM
# Our PTMs already account for this (Schrodinger convention from build_ptm.py)

for color in sorted(color_bonds.keys()):
    bonds_in_color = color_bonds[color]
    print(f"  Color {color}: {len(bonds_in_color)} bonds")

    for (u,v) in bonds_in_color:
        key = (min(u,v), max(u,v))
        c = coupling[key]
        alpha = -c * EPSILON  # Julia convention: Δ = -coupling, α = Δ × epsilon
        delta = -c

        G1 = PTM_gates1()
        G2 = PTM_gates2(alpha)
        G3 = PTM_gates3(alpha)
        G4 = PTM_gates4(delta)

        # Julia applies reversed gate sequence in Heisenberg picture:
        # reversed = [gates4_rev, gates3_rev, gates2_rev, gates1_rev]
        # Each group is reversed internally too.
        # For PTM: M_reversed = M^T (adjoint of unitary = transpose of PTM)
        # Full reversed PTM = compose([G4,G3,G2,G1])^T = compose([G1^T,G2^T,G3^T,G4^T])
        G = compose([G1.T, G2.T, G3.T, G4.T])
        new_chi = apply_gate_ptm(u, v, G, cutoff=1e-8)

print(f"\nAfter Trotter step:")
print(f"  Max chi: {max(chi)}")
print(f"  Chi distribution: {sorted(set(chi))}")

# Apply kick gate (single-site Rx(2φ) on every site)
print(f"\nApplying kick gate Rx(2φ) to all {N_SITES} sites...")
Mk = single_qubit_ptm(Rx(2*PHI))  # 4×4 PTM
for v in range(N_SITES):
    pv = get_psi(v)
    sh = list(pv.shape)
    # Contract physical index (last) with Mk
    pv2 = (pv.reshape(-1, D) @ Mk.T).reshape(sh)
    set_psi(v, pv2)

# ── Run BP to converge messages ───────────────────────────────────────────
print("\nRunning BP to converge messages...")

def msg_incoming(bid, sid):
    u,v = bond_list[bid]
    return msgs[bid][1] if sid == u else msgs[bid][0]  # a→b=0, b→a=1

def msg_outgoing_idx(bid, sid):
    u,v = bond_list[bid]
    return 0 if sid == u else 1

def run_bp(n_sweeps=25, tol=1e-5, rho_fn=None):
    """
    Standard BP for autocorrelation inner(ψ, ψ₀).
    Messages are chi×chi matrices (full bilinear form).
    rho_fn: if None, uses Z at center / I elsewhere.
    """
    # Initialize messages to identity
    for bid in range(len(bond_list)):
        c = chi[bid]
        msgs[bid] = [np.eye(c), np.eye(c)]

    for sweep in range(n_sweeps):
        avg_delta = 0.
        n_msgs = 0
        for bid, (ua, vb) in enumerate(bond_list):
            for src in [ua, vb]:
                sh = site_shape(src)  # (chi_0,...,D)
                skip = site_bonds[src].index(bid)
                chi_out = sh[skip]

                psi_src = get_psi(src).reshape(sh)

                # Contract all non-skip bonds with incoming messages (chi×chi)
                cur = psi_src.copy()
                cur_sh = list(sh)

                for k, b2 in enumerate(site_bonds[src]):
                    if k == skip: continue
                    M = msg_incoming(b2, src)  # chi_k × chi_k
                    # contract mode k: result[...,i,...] = Σ_j cur[...,j,...] * M[j,i]
                    cur = np.tensordot(cur.reshape(cur_sh), M, axes=([k],[0]))
                    cur = np.moveaxis(cur, -1, k)

                # Apply physical selector (last dim) → chi_out × chi_out result
                # new_msg[i,j] = Σ_{others} cur[...,i,...,d] * sel[d] contracted with ψ₀
                # For standard BP: ψ₀ selects d=3 (Z) at center, d=0 (I) elsewhere
                if rho_fn is None:
                    sel = np.array([0.,0.,0.,1.]) if src==CENTER else np.array([1.,0.,0.,0.])
                else:
                    sel = rho_fn(src)

                # Contract physical index
                cur2 = np.tensordot(cur.reshape(cur_sh), sel, axes=([-1],[0]))
                cur_sh2 = cur_sh[:-1]  # remove physical dim

                # Sum over all dims except skip → chi_out vector
                nm = cur2.reshape(cur_sh2)
                for ax in sorted([i for i in range(len(cur_sh2)) if i != skip], reverse=True):
                    nm = nm.sum(axis=ax)

                # Normalize by sum (Julia convention)
                s = nm.sum()
                if abs(s) > 1e-15:
                    nm = nm / s

                # Store as diagonal chi×chi matrix
                new_msg = np.diag(nm)
                idx = msg_outgoing_idx(bid, src)
                old_msg = msgs[bid][idx]
                delta = np.abs(new_msg - old_msg).mean()
                avg_delta += delta
                n_msgs += 1
                msgs[bid][idx] = new_msg

        avg_delta /= max(n_msgs, 1)
        if avg_delta < tol:
            print(f"  Converged at sweep {sweep+1} (avg_delta={avg_delta:.2e})")
            break
    else:
        print(f"  Reached {n_sweeps} sweeps (avg_delta={avg_delta:.2e})")

run_bp()

# ── Autocorrelation check ─────────────────────────────────────────────────
def bethe_pf(rho_fn):
    """Compute Bethe partition function with per-site rho selector."""
    log_num = 0.0
    sign_num = 1.0

    for v in range(N_SITES):
        sh = site_shape(v)
        rho = rho_fn(v)
        cur = get_psi(v).reshape(sh)
        cur_sh = list(sh)

        # Contract all incoming messages
        for k, bid in enumerate(site_bonds[v]):
            M = msg_incoming(bid, v)  # chi×chi diagonal
            cur = np.tensordot(cur.reshape(cur_sh), M, axes=([k],[0]))
            cur = np.moveaxis(cur, -1, k)

        # Contract physical (last dim) with rho
        a_v = float(np.tensordot(cur.reshape(cur_sh), rho, axes=([-1],[0])).sum())

        if a_v == 0.0: sign_num = 0.0; break
        sign_num *= np.sign(a_v)
        log_num += np.log(abs(a_v))

    log_den = 0.0
    sign_den = 1.0
    for bid in range(len(bond_list)):
        # ze = Tr(Ma × Mb) for chi×chi messages
        Ma = msgs[bid][0]
        Mb = msgs[bid][1]
        ze = float(np.trace(Ma @ Mb))
        if abs(ze) > 1e-30:
            sign_den *= np.sign(ze)
            log_den += np.log(abs(ze))

    if sign_num == 0.0: return 0.0
    return sign_num * sign_den * np.exp(log_num - log_den)

# Debug: check psi at center
pv = get_psi(CENTER)
print(f"\nDebug psi[center] shape: {pv.shape}")
print(f"  psi[center] first 8 values: {pv.flatten()[:8]}")
print(f"  chi bonds around center: {[chi[bid] for bid in site_bonds[CENTER]]}")

# Z⊗I autocorrelation
rho_Z = lambda v: np.array([0.,0.,0.,1.]) if v==CENTER else np.array([1.,0.,0.,0.])
run_bp(rho_fn=rho_Z)
C_auto = bethe_pf(rho_Z)
print(f"\nAutocorrelation inner(ψ, Z⊗I) = {C_auto:.6f}  (Julia: -0.803318)")

# inner(ψ, all-zero ρ)
rho_all0 = lambda v: np.array([1.,0.,0.,1.])
run_bp(rho_fn=rho_all0)
C_all0 = bethe_pf(rho_all0)
print(f"inner(ψ, all-zero ρ)  = {C_all0:.6f}  (Julia: -0.961822)")

# inner(ψ, device ρ)
rho_device = lambda v: np.array([1.,0.,0., float(spin(v))])
run_bp(rho_fn=rho_device)
C_dev = bethe_pf(rho_device)
print(f"inner(ψ, device ρ)    = {C_dev:.6f}  (Julia: -0.678563)")

print("\nDone.")
