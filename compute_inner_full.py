"""
compute_inner_full.py — Full inner product using all 68 site phi values
from CppSim + Julia's converged messages.

This is the key validation: plug Julia's exact messages into the Bethe
formula with CppSim's phi_v. Should give -0.961822 if formula is correct.
"""

import numpy as np
import re

# ── Load CppSim phi dump ─────────────────────────────────────────────────
print("Loading phi dump from CppSim...")
sites = {}  # site_id → {spin, bvol, chis, phi}
with open('/tmp/phi_dump.txt') as f:
    for line in f:
        parts = line.strip().split()
        # site ID spin SPIN bvol BVOL chi CHI... phi PHI...
        sid = int(parts[1])
        spin = int(parts[3])
        bvol = int(parts[5])
        # find chi values and phi values
        chi_vals = []
        phi_vals = []
        in_chi = False; in_phi = False
        for p in parts[6:]:
            if p == 'chi': in_chi=True; in_phi=False; continue
            if p == 'phi': in_phi=True; in_chi=False; continue
            if in_chi: chi_vals.append(int(p))
            if in_phi: phi_vals.append(float(p))
        sites[sid] = {'spin': spin, 'bvol': bvol, 'chis': chi_vals,
                      'phi': np.array(phi_vals)}

print(f"Loaded {len(sites)} sites")
print(f"Site 2 (center): bvol={sites[2]['bvol']} chis={sites[2]['chis']}")
print(f"Site 2 phi sum: {sites[2]['phi'].sum():.6f}")

# ── Julia's converged chi-vector messages ─────────────────────────────────
# From test_inner_minimal.jl output (all-zero rho, 25 sweeps)
# Format: msg (julia_src,) => (julia_dst,): [values]
# Julia 1-based → CppSim 0-based: src_cpp = julia_src - 1
julia_msgs = {
    # (src_0based, dst_0based): chi-vector
    (0, 30): [1.0],   # (1,)=>(31,)
    (0, 31): [1.0],   # (1,)=>(32,)
    (1, 30): [1.0017206889942032],  # (2,)=>(31,)
    (1, 32): [0.7240152375458738],  # (2,)=>(33,)
    (2, 32): [1.0726611230806198, 0.030485325909600895, 0.1645486297855998, -0.2676950787758206],  # (3,)=>(33,)
    (2, 33): [0.995440236903479, -0.1564135891962512, 0.10830786311162433, 0.05266548918114787],  # (3,)=>(34,)
    (2, 34): [1.0877051195004392, -0.142544177574008, -0.053561101330993476, 0.1084001594045623],  # (3,)=>(35,)
    (3, 33): [2.5435852141865958, -1.3056619580456927, 0.004273873806013177, -0.24219712994691625],  # (4,)=>(34,)
    (3, 35): [1.0],  # (4,)=>(36,)
}
# All other messages = [1.0] (chi=1 bonds)

def get_julia_msg(src, dst):
    """Get Julia's converged message from src to dst (0-based)."""
    key = (src, dst)
    if key in julia_msgs:
        return np.array(julia_msgs[key])
    # Reverse direction
    key_rev = (dst, src)
    if key_rev in julia_msgs:
        return np.array(julia_msgs[key_rev])
    return np.array([1.0])  # default chi=1

# ── Bond structure from CppSim EDGES ─────────────────────────────────────
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

# For each site, list of (bond_id, neighbor_id) in tensor order
site_bonds = [[] for _ in range(68)]
for bid, (u,v) in enumerate(EDGES):
    site_bonds[u].append((bid, v))
    site_bonds[v].append((bid, u))

# ── Compute Bethe partition function ─────────────────────────────────────
print("\n── Chi×chi BilinearForm BP on phi_v ────────────────────────")

# Messages: chi×chi matrices, initialized to identity
bp_msgs = {}  # (bid, src) → chi×chi outgoing message
for bid, (u,v) in enumerate(EDGES):
    chi_u = max([c for c in sites[u]['chis']] or [1])
    chi_v = max([c for c in sites[v]['chis']] or [1])
    chi_bond = max(chi_u, chi_v, len(get_julia_msg(u,v)))
    # Actually use phi chi for this bond
    for sid, nbr in [(u,v),(v,u)]:
        c = sites[sid]['chis'][site_bonds[sid].index((bid,nbr))] if (bid,nbr) in site_bonds[sid] else 1
        bp_msgs[(bid,sid)] = np.eye(c)

for sweep in range(25):
    max_delta = 0.
    for bid, (ua, vb) in enumerate(EDGES):
        for src, dst in [(ua,vb),(vb,ua)]:
            s = sites[src]
            phi = s['phi']
            chis = s['chis']
            if not chis: continue
            skip_idx = [k for k,(b,n) in enumerate(site_bonds[src]) if b==bid]
            if not skip_idx: continue
            skip = skip_idx[0]
            chi_out = chis[skip]

            # Contract phi with all incoming messages except skip bond
            phi_t = phi.reshape(chis).astype(float)
            for k,(b2,n2) in enumerate(site_bonds[src]):
                if k==skip: continue
                M = bp_msgs.get((b2,n2), np.eye(chis[k]))
                # Contract axis k with M: result[...,j,...] = Σ_i phi[...,i,...]*M[i,j]
                phi_t = np.tensordot(phi_t, M, axes=([k],[0]))
                phi_t = np.moveaxis(phi_t, -1, k)

            # Sum all axes except skip → chi_out × chi_out message
            # phi_t still has shape chis, contract skip axis to get chi_out×chi_out
            # new_msg[i,j] = delta(i,j) * Σ_{others} phi_t[...,i,...]
            axes_to_sum = [k for k in range(len(chis)) if k!=skip]
            marginal = phi_t
            for ax in sorted(axes_to_sum, reverse=True):
                marginal = marginal.sum(axis=ax)
            # marginal shape: (chi_out,)
            new_msg = np.diag(marginal)

            # Normalize
            tr = np.trace(new_msg)
            if abs(tr) > 1e-20: new_msg /= tr

            old = bp_msgs.get((bid,src), np.eye(chi_out))
            delta = np.abs(new_msg - old).max()
            max_delta = max(max_delta, delta)
            bp_msgs[(bid,src)] = new_msg

    if max_delta < 1e-5:
        print(f"  Converged at sweep {sweep+1}")
        break

print(f"  Final max_delta: {max_delta:.2e}")

def get_msg_bp(bid, src):
    return bp_msgs.get((bid,src), np.eye(1))

print("\n── Bethe partition function ─────────────────────────────────")

def get_msg(bid, src):
    """Get incoming message to src on bond bid — from BilinearForm BP."""
    u, v = EDGES[bid]
    nbr = v if src == u else u
    return get_msg_bp(bid, nbr).diagonal()  # use diagonal of chi×chi msg

log_num = 0.0; sign_num = 1.0
for sid in range(68):
    s = sites[sid]
    phi = s['phi']
    bvol = s['bvol']
    chis = s['chis']  # chi for each bond

    if bvol == 1:
        # No bonds or all chi=1 — a_v = phi[0]
        a_v = phi[0]
    else:
        # Contract phi[i,j,...] with messages
        # phi shape: (chis[0], chis[1], ...) flattened
        phi_t = phi.reshape(chis if chis else [1])
        # Weight by messages
        weighted = phi_t.copy().astype(float)
        for k, (bid, nbr) in enumerate(site_bonds[sid]):
            m = get_msg(bid, sid)
            if len(m) == 1:
                weighted *= m[0]  # chi=1: scalar multiply
            else:
                # Multiply along axis k
                shape = [1]*len(chis)
                shape[k] = len(m)
                weighted = weighted * m.reshape(shape)
        a_v = weighted.sum()

    if abs(a_v) < 1e-100: sign_num = 0.0; break
    sign_num *= np.sign(a_v)
    log_num += np.log(abs(a_v))

    if sid in [2, 32, 33, 34]:
        print(f"  a_v[site={sid}] = {a_v:.6f}")

# Edge correction: ze_e = msg_fwd · msg_bwd
log_den = 0.0; sign_den = 1.0
for bid, (u,v) in enumerate(EDGES):
    Ma = get_msg_bp(bid, u)  # outgoing from u (= incoming to v from u)
    Mb = get_msg_bp(bid, v)  # outgoing from v (= incoming to u from v)
    ze = float(np.trace(Ma @ Mb))
    if abs(ze) > 1e-30:
        sign_den *= np.sign(ze)
        log_den += np.log(abs(ze))

if sign_num == 0.0:
    print("C = 0 (some a_v = 0)")
else:
    C = sign_num * sign_den * np.exp(log_num - log_den)
    print(f"\nC (full, Julia messages) = {C:.6f}  (Julia: -0.961822)")
    print(f"Ratio: {C/(-0.961822):.4f}")
