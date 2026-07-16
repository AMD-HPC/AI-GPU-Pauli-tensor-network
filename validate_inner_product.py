"""
validate_inner_product.py — Validate inner(ψ,ρ) in symmetric gauge

Transform CppSim's QR gauge tensors to symmetric (Vidal) gauge,
then compute inner product using Julia's chi-vector BP algorithm.

Golden targets (Julia):
    inner(ψ(t=1), all-zero ρ) = -0.961822
    inner(ψ(t=1), device ρ)   = -0.678563
    inner(ψ(t=1), Z⊗I...⊗I)  = -0.803318

Usage:
    python3 validate_inner_product.py

Input data comes from test_inner_op_state.cpp debug output.
"""

import numpy as np

print("="*60)
print("  validate_inner_product.py")
print("="*60)

def mat_sqrt(M):
    """Matrix square root via eigendecomposition. M must be symmetric."""
    vals, vecs = np.linalg.eigh(M)  # eigh for symmetric matrices
    sqrt_vals = np.sqrt(np.abs(vals))  # take abs to handle near-zero negatives
    return vecs @ np.diag(sqrt_vals) @ vecs.T

def phi_from_psi(psi, shape, rho):
    """φ_v = Σ_d ψ_v[bonds...,d] × ρ[d]"""
    D_phys = shape[-1]
    bvol = int(np.prod(shape[:-1]))
    psi_r = psi.reshape(bvol, D_phys)
    return psi_r @ rho

def mode_n_contract(tensor, shape, mode, M):
    """Contract mode `mode` of tensor with matrix M: out[...,i,...] = Σ_j tensor[...,j,...]*M[j,i]"""
    t = tensor.reshape(shape)
    t = np.tensordot(t, M, axes=([mode], [0]))
    # Move contracted axis to correct position
    t = np.moveaxis(t, -1, mode)
    return t.flatten()

# ── Data from CppSim test output (t=1, all-zero rho = [1,0,0,+1]) ─────────
print("\n── Loading CppSim data ─────────────────────────────────────")

# phi values from CppSim debug output
phi_center = np.array([
    -0.0000, 0.0000, 0.1927, -0.0000,
    -0.1042, 0.0000, -0.0000, -0.0000,
    -0.0000, -0.2996, -0.0000, 0.0000,
     0.0000, 0.0000, 0.0000, 0.0000,
     0.4670, -0.0000, 0.0000, 0.0000,
    -0.0000, 0.0000, 0.0634, 0.0000,
    -0.0000, 0.0000, -0.0000, -0.0634,
     0.0000, 0.0000, 0.0000, 0.0000,
     0.0000, 0.0000, -0.0000, 0.0444,
     0.0000, -0.0691, 0.0000, 0.0000,
     0.0240, -0.0000, 0.0000, -0.0000,
     0.0000, 0.0000, 0.0000, 0.0000,
     0.0000, 0.0000, 0.0000, 0.1368,
    -0.0000, -0.2126, 0.0000, 0.0000,
     0.0740, -0.0000, 0.0000, -0.0000,
     0.0000, 0.0000, 0.0000, 0.0000
]).reshape(4,4,4)  # [chi_32, chi_33, chi_34]

phi_32 = np.array([0.942828, 0.000000, 0.438473, -0.142468])   # shape [4]
phi_33 = np.array([0.953115, 0.000000, -0.458463, 0.148964])   # shape [4]
phi_34 = np.array([[-0.988485, 0.020862, 0.000000, 0.000000],
                   [ 0.000000, 0.000000, 0.000000,-0.212820],
                   [-0.067494, 0.000000,-0.202404, 0.000000],
                   [-0.207724, 0.000000, 0.065765, 0.000000]])  # shape [4,4]

print(f"phi_center shape: {phi_center.shape}  sum={phi_center.sum():.4f}")
print(f"phi_32 shape: {phi_32.shape}  sum={phi_32.sum():.4f}")
print(f"phi_33 shape: {phi_33.shape}  sum={phi_33.sum():.4f}")
print(f"phi_34 shape: {phi_34.shape}  sum={phi_34.sum():.4f}")

# ── Chi-vector BP (Julia's algorithm) ─────────────────────────────────────
print("\n── Chi-vector BP (Julia's algorithm) ───────────────────────")
print("Julia converged in 1 sweep. Messages:")
print("  (3,)→(33,) = [1.073, 0.030, 0.165, -0.268]  (center→32)")

# Site 34 has TWO chi=4 bonds: to center(2) and to another site (site3 in Julia)
# We need the phi of that other neighbor to complete the contraction
# For now: marginalize phi_34 over the second bond with messages=uniform
# This is what BP does when the other neighbor hasn't been reached yet

# Julia a_v values (from test_inner_minimal.jl):
# a_v[(3,)] = -6.618  (center, site 2)
# a_v[(31,)] = -0.892  (site 30)
# a_v[(33,)] = +0.903  (site 32)
# a_v[(35,)] = -0.928  (site 34)
# C = -0.962

print("\n── Computing messages via BP ────────────────────────────────")

# Initialize messages (chi-vectors)
# Bond 4: center(2) ↔ site32, chi=4
# Bond 5: center(2) ↔ site33, chi=4
# Bond 6: center(2) ↔ site34, chi=4
# Site34 also connects to another site via bond with chi=4

# For site 32 (leaf: one chi=4 bond to center):
# msg_32→center = phi_32 / sum(phi_32)
s32 = phi_32.sum()
msg_32_to_c = phi_32 / s32
print(f"msg_32→center = {msg_32_to_c}")

# For site 33 (leaf: one chi=4 bond to center):
s33 = phi_33.sum()
msg_33_to_c = phi_33 / s33
print(f"msg_33→center = {msg_33_to_c}")

# For site 34 (two chi=4 bonds):
# Need message from site34's other neighbor (site 3 in Julia = site 2... wait)
# phi_34[k,l] where k=chi toward center, l=chi toward other neighbor
# The other neighbor's message = uniform [0.25, 0.25, 0.25, 0.25] initially
msg_other_to_34 = np.ones(4) / 4

# msg_34→center[k] = Σ_l phi_34[k,l] × msg_other→34[l] / norm
msg_34_to_c = phi_34 @ msg_other_to_34
s34 = msg_34_to_c.sum()
if abs(s34) > 1e-10:
    msg_34_to_c = msg_34_to_c / s34
else:
    msg_34_to_c = msg_34_to_c / np.linalg.norm(msg_34_to_c)
print(f"msg_34→center (init) = {msg_34_to_c}")

# ── Compute a_center ─────────────────────────────────────────────
print("\n── Compute a_center ────────────────────────────────────────")
a_center = np.einsum('ijk,i,j,k->', phi_center, msg_32_to_c, msg_33_to_c, msg_34_to_c)
print(f"a_center = {a_center:.6f}  (Julia: -6.618)")

# ── Compute msg_center→32 ────────────────────────────────────────
msg_c_to_32 = np.einsum('ijk,j,k->i', phi_center, msg_33_to_c, msg_34_to_c)
s = msg_c_to_32.sum()
if abs(s) > 1e-10: msg_c_to_32 = msg_c_to_32 / s
print(f"msg_center→32 = {msg_c_to_32}")
print(f"Julia msg (3,)→(33,): [1.073, 0.030, 0.165, -0.268]")

# ── ze values ────────────────────────────────────────────────────
ze_32 = msg_32_to_c @ msg_c_to_32
print(f"\nze_32 = {ze_32:.6f}")

# ── Full C ────────────────────────────────────────────────────────
print("\n── Full C computation ──────────────────────────────────────")
# TODO: need all 68 sites' a_v values and all ze values
# For now compute just the chi=4 bond contribution
print("Need all site phi values to compute full C")
print("Next step: dump ALL phi values from CppSim and run full BP here")

print(f"\nTarget: C = -0.961822")
print(f"a_center / Julia_a_center = {a_center / (-6.618):.4f}  (should be ~1 if gauge matched)")
