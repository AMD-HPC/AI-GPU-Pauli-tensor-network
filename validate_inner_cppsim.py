"""
validate_inner_cppsim.py — Validate inner product using CppSim psi tensors

Instead of recomputing the Trotter step in Python, we use CppSim's
correctly-computed psi tensors (downloaded from GPU) and run the
chi-vector BP + Bethe formula in Python.

This isolates the inner product formula from the gate application.

Expected: inner(ψ, all-zero ρ) = -0.961822 (Julia golden)

To get psi data from CppSim: add dump code to test_inner_op_state.cpp
"""

import numpy as np

print("="*60)
print("  validate_inner_cppsim.py")
print("="*60)

# ── CppSim psi data (from test_inner_op_state debug output) ──────────────
# Center site 2: shape (4,4,4,4) = (chi_bond4, chi_bond5, chi_bond6, D)
# phi_center (all-zero rho=[1,0,0,1]):
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
]).reshape(4,4,4)  # phi_c[chi32, chi33, chi34]

phi_32 = np.array([0.942828, 0.000000, 0.438473, -0.142468])   # chi=4
phi_33 = np.array([0.953115, 0.000000, -0.458463, 0.148964])   # chi=4
phi_34 = np.array([[-0.988485, 0.020862, 0.000000, 0.000000],
                   [ 0.000000, 0.000000, 0.000000,-0.212820],
                   [-0.067494, 0.000000,-0.202404, 0.000000],
                   [-0.207724, 0.000000, 0.065765, 0.000000]])  # chi=4 × chi=4 (has other bond)

print(f"phi_center sum: {phi_center.sum():.4f}")
print(f"phi_32 sum: {phi_32.sum():.4f}")
print(f"phi_33 sum: {phi_33.sum():.4f}")

# ── Chi-vector BP on phi tensors ─────────────────────────────────────────
# Messages: chi-vectors (not chi×chi)
# Site 32: leaf with one chi=4 bond to center
# Site 33: leaf with one chi=4 bond to center
# Site 34: has two chi=4 bonds (to center and to site 3)

# Initialize uniform
m32_to_c = np.ones(4)/4   # msg site32 → center
m33_to_c = np.ones(4)/4   # msg site33 → center
m34_to_c = np.ones(4)/4   # msg site34 → center
mc_to_32 = np.ones(4)/4   # msg center → site32
mc_to_33 = np.ones(4)/4   # msg center → site33
mc_to_34 = np.ones(4)/4   # msg center → site34

print("\n── BP sweeps ────────────────────────────────────────────────")
for sweep in range(25):
    old = (m32_to_c.copy(), m33_to_c.copy(), m34_to_c.copy(),
           mc_to_32.copy(), mc_to_33.copy(), mc_to_34.copy())

    # Leaves → center (no incoming except from center which we skip)
    # msg_32→c[i] ∝ phi_32[i]  (leaf: no other incoming)
    nm = phi_32.copy()
    s = nm.sum()
    m32_to_c = nm/s if abs(s)>1e-15 else nm/np.linalg.norm(nm)

    nm = phi_33.copy()
    s = nm.sum()
    m33_to_c = nm/s if abs(s)>1e-15 else nm/np.linalg.norm(nm)

    # Site34 → center: phi_34[k,l] × m_other_to_34[l] summed over l
    # (m_other_to_34 = uniform [0.25,...] since site34's other bond has uniform msg)
    m_other = np.ones(4)/4  # message from site34's other neighbor (chi=1 or chi=4)
    nm = phi_34 @ m_other
    s = nm.sum()
    m34_to_c = nm/s if abs(s)>1e-15 else nm/np.linalg.norm(nm)

    # Center → leaf (skip the target leaf, use other two)
    # msg_c→32[i] ∝ Σ_{j,k} phi_c[i,j,k] × m33_to_c[j] × m34_to_c[k]
    nm = np.einsum('ijk,j,k->i', phi_center, m33_to_c, m34_to_c)
    s = nm.sum()
    mc_to_32 = nm/s if abs(s)>1e-15 else nm/np.linalg.norm(nm)

    nm = np.einsum('ijk,i,k->j', phi_center, m32_to_c, m34_to_c)
    s = nm.sum()
    mc_to_33 = nm/s if abs(s)>1e-15 else nm/np.linalg.norm(nm)

    nm = np.einsum('ijk,i,j->k', phi_center, m32_to_c, m33_to_c)
    s = nm.sum()
    mc_to_34 = nm/s if abs(s)>1e-15 else nm/np.linalg.norm(nm)

    # Check convergence
    delta = max(np.max(np.abs(m32_to_c - old[0])),
                np.max(np.abs(m33_to_c - old[1])),
                np.max(np.abs(m34_to_c - old[2])))
    if sweep < 3 or delta < 1e-6:
        print(f"  sweep {sweep+1}: m32→c={np.round(m32_to_c,4)}  delta={delta:.2e}")
    if delta < 1e-6: break

print(f"\nConverged messages:")
print(f"  m32→center: {np.round(m32_to_c,4)}")
print(f"  m33→center: {np.round(m33_to_c,4)}")
print(f"  m34→center: {np.round(m34_to_c,4)}")
print(f"  mc→32: {np.round(mc_to_32,4)}")
print(f"  Julia (3,)→(33,): [1.073, 0.030, 0.165, -0.268]")

# ── Bethe partition function ─────────────────────────────────────────────
print("\n── Bethe partition function ─────────────────────────────────")

# a_center = Σ_{i,j,k} phi_c[i,j,k] × m32→c[i] × m33→c[j] × m34→c[k]
a_center = np.einsum('ijk,i,j,k->', phi_center, m32_to_c, m33_to_c, m34_to_c)
print(f"a_center = {a_center:.6f}  (Julia: -6.618)")

# a_32 = Σ_i phi_32[i] × mc→32[i]
a_32 = np.dot(phi_32, mc_to_32)
a_33 = np.dot(phi_33, mc_to_33)
a_34 = np.dot(phi_34 @ m_other, mc_to_34)
print(f"a_32 = {a_32:.6f}  (Julia: 0.903)")
print(f"a_33 = {a_33:.6f}")
print(f"a_34 = {a_34:.6f}  (Julia: -0.928)")

# ze values (for the 3 chi=4 bonds near center)
ze_32 = np.dot(m32_to_c, mc_to_32)
ze_33 = np.dot(m33_to_c, mc_to_33)
ze_34 = np.dot(m34_to_c, mc_to_34)
print(f"\nze_32 = {ze_32:.6f}")
print(f"ze_33 = {ze_33:.6f}")
print(f"ze_34 = {ze_34:.6f}")

# For all other sites/bonds: phi_v is a scalar (chi=1), a_v = phi_v, ze=1
# These don't change C significantly for the chi=4 subgraph computation
# For now compute just the 4 sites near center

import math
# Approximate: C ≈ a_center × a_32 × a_33 × a_34 × (other_sites) / ze_32 × ze_33 × ze_34
# Other sites contribute phi_v × 1 (chi=1, messages=1)
# phi_v for all-zero rho at chi=1 sites: psi_v[0] + psi_v[3]
# Initially psi_v = [1,0,0,0] for I sites → phi_v = 1
# After gate, chi=1 sites are mostly unchanged → phi_v ≈ 1
# There are 68-4=64 such sites, all contributing a_v ≈ 1 (they don't touch center)

C_approx = (a_center * a_32 * a_33 * a_34) / (ze_32 * ze_33 * ze_34)
print(f"\nC (4-site approximation) = {C_approx:.6f}")
print(f"Target: -0.961822")
print(f"Ratio: {C_approx/(-0.961822):.4f}")

print("\n── Autocorrelation check (Z⊗I...⊗I) ───────────────────────")
# phi_Z_center[i,j,k] = psi_center[i,j,k,d=3]
# From debug: phi_Z first 4 = [0.300, ~0, ~0, ~0]
# phi_Z ≈ phi_allzero for Z selector (I component ~0 in QR gauge)
phi_Z_center = phi_center.copy()  # approximate: Z selector ≈ all-zero selector

# phi_I for non-center sites (Z selector gives 0, I selector gives psi[...,0])
# From debug: psi_32[0,0,d]=[0.9428,0,0,0] → phi_I_32[0]=0.9428
phi_I_32 = np.array([0.942828, 0.000000, 0.438473, -0.142468])  # same as phi_32 for all-zero!
phi_I_33 = np.array([0.953115, 0.000000,-0.458463, 0.148964])
phi_I_34 = phi_34.copy()

# Run same BP but with Z at center, I at non-center
# For leaves: msg_32→c[i] ∝ phi_I_32[i]
m32_z = phi_I_32 / phi_I_32.sum()
m33_z = phi_I_33 / phi_I_33.sum()
m34_z = (phi_I_34 @ np.ones(4)/4)
m34_z = m34_z / m34_z.sum()

# Center → leaves using phi_Z_center
mc32_z = np.einsum('ijk,j,k->i', phi_Z_center, m33_z, m34_z)
s = mc32_z.sum(); mc32_z = mc32_z/s if abs(s)>1e-15 else mc32_z

mc33_z = np.einsum('ijk,i,k->j', phi_Z_center, m32_z, m34_z)
s = mc33_z.sum(); mc33_z = mc33_z/s if abs(s)>1e-15 else mc33_z

mc34_z = np.einsum('ijk,i,j->k', phi_Z_center, m32_z, m33_z)
s = mc34_z.sum(); mc34_z = mc34_z/s if abs(s)>1e-15 else mc34_z

# a_center for autocorrelation
a_c_auto = np.einsum('ijk,i,j,k->', phi_Z_center, m32_z, m33_z, m34_z)
a_32_auto = np.dot(phi_I_32, mc32_z)
ze_32_auto = np.dot(m32_z, mc32_z)

print(f"a_center (autocorr) = {a_c_auto:.6f}")
print(f"a_32 (autocorr)     = {a_32_auto:.6f}")
print(f"ze_32 (autocorr)    = {ze_32_auto:.6f}")

# Note: other 64 sites all contribute a_v=1, ze=1 (chi=1, trivial)
# So C ≈ a_center × a_32 × a_33 × a_34 / ze_32 × ze_33 × ze_34
a_33_auto = np.dot(phi_I_33, mc33_z)
ze_33_auto = np.dot(m33_z, mc33_z)
a_34_auto = np.dot(phi_I_34 @ np.ones(4)/4, mc34_z)
ze_34_auto = np.dot(m34_z, mc34_z)

C_auto = (a_c_auto * a_32_auto * a_33_auto * a_34_auto) / (ze_32_auto * ze_33_auto * ze_34_auto)
print(f"C_autocorr (4-site) = {C_auto:.6f}  (Julia: -0.803318)")
print(f"C_inner    (4-site) = {C_approx:.6f}  (Julia: -0.961822)")
print(f"\nRatio C_inner/C_autocorr = {C_approx/C_auto:.4f}  (Julia: {-0.961822/-0.803318:.4f})")
