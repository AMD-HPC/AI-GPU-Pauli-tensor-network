"""
test_gauge_invariance.py — Is the two-state overlap gauge-invariant?

Minimal test: 2 sites, 1 bond, both with chi>1.
Compute ⟨ψ_a|ψ_b⟩ two ways:
  1. Exact contraction (gauge-invariant by definition)
  2. In gauge G, then in gauge G' — check they agree

If exact contraction is gauge-invariant (it must be), then the question
is whether BP reproduces it. On a SINGLE bond (tree), BP is exact,
so both gauges give the exact answer.

This isolates: does the double-layer construction work, independent of gauge?
"""
import numpy as np

np.random.seed(42)
D = 4      # physical (Pauli)
chi = 3    # bond dimension

print("="*60)
print("  test_gauge_invariance.py")
print("  2 sites, 1 bond, two-state overlap ⟨ψ_a|ψ_b⟩")
print("="*60)

# ── Two operators ψ_a and ψ_b, each as 2-site MPS ────────────────────────
# ψ_a: site0[chi, D], site1[chi, D]  (bond connects them)
a0 = np.random.randn(chi, D)
a1 = np.random.randn(chi, D)
b0 = np.random.randn(chi, D)
b1 = np.random.randn(chi, D)

# ── Exact overlap ⟨ψ_a|ψ_b⟩ ───────────────────────────────────────────────
# ⟨ψ_a|ψ_b⟩ = Σ_{d0,d1} [Σ_i a0[i,d0] a1[i,d1]] × [Σ_j b0[j,d0] b1[j,d1]]
# = Σ_{d0,d1} A[d0,d1] × B[d0,d1]  where A = a0ᵀa1 contracted on bond
# Actually: ψ_a as operator = Σ_i a0[i,:] ⊗ a1[i,:]  (i = bond index)
#           ψ_b as operator = Σ_j b0[j,:] ⊗ b1[j,:]
# ⟨ψ_a|ψ_b⟩ = Σ_{d0,d1} (Σ_i a0[i,d0]a1[i,d1])(Σ_j b0[j,d0]b1[j,d1])
A_op = np.einsum('id,ie->de', a0, a1)  # (D,D) — ψ_a as d0×d1 matrix
B_op = np.einsum('jd,je->de', b0, b1)  # (D,D)
exact = np.sum(A_op * B_op)
print(f"\nExact ⟨ψ_a|ψ_b⟩ = {exact:.6f}")

# ── Double-layer BP (single bond = tree, BP exact) ────────────────────────
# Double layer site tensor: T0[i,j] = Σ_d a0[i,d] b0[j,d]  (contract physical)
# bond dimension of double layer = chi_a × chi_b = 3×3 = 9
T0 = np.einsum('id,jd->ij', a0, b0)  # (chi_a, chi_b) at site 0
T1 = np.einsum('id,jd->ij', a1, b1)  # (chi_a, chi_b) at site 1

# BP on single bond: message from site0 to site1 = T0 marginalized (no other bonds)
# msg_0→1[i,j] = T0[i,j]  (only one bond, nothing to marginalize)
# a_1 = Σ_{i,j} T1[i,j] × msg_0→1[i,j]  = Σ T1 * T0
msg_01 = T0  # (chi_a, chi_b)
a1_scalar = np.sum(T1 * msg_01)
print(f"Double-layer contraction = {a1_scalar:.6f}")
print(f"Match exact: {np.isclose(a1_scalar, exact)}")

# ── Now apply a GAUGE transform to ψ_a's bond and recompute ──────────────
# Insert G G^-1 on the bond of ψ_a: a0 → a0·G, a1 → G^-1·a1
G = np.random.randn(chi, chi)
Ginv = np.linalg.inv(G)
a0_g = np.einsum('id,ij->jd', a0, G)      # a0·G on bond index
a1_g = np.einsum('ij,jd->id', Ginv, a1)   # G^-1·a1 on bond index

# Verify ψ_a unchanged (operator level)
A_op_g = np.einsum('id,ie->de', a0_g, a1_g)
print(f"\nAfter gauge on ψ_a: A_op unchanged? {np.allclose(A_op, A_op_g)}")

# Double-layer with gauged ψ_a
T0_g = np.einsum('id,jd->ij', a0_g, b0)
T1_g = np.einsum('id,jd->ij', a1_g, b1)
msg_01_g = T0_g
a1_g_scalar = np.sum(T1_g * msg_01_g)
print(f"Double-layer (ψ_a gauged) = {a1_g_scalar:.6f}")
print(f"Still matches exact: {np.isclose(a1_g_scalar, exact)}")

print("\n" + "="*60)
print("CONCLUSION:")
if np.isclose(a1_g_scalar, exact) and np.isclose(a1_scalar, exact):
    print("  Double-layer overlap IS gauge-invariant (on tree).")
    print("  → QR gauge works for two-state overlap IF we build double-layer.")
    print("  → Our failures: we never built the true double-layer network.")
else:
    print("  Double-layer overlap is NOT gauge-invariant — deeper issue.")
