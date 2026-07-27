#!/usr/bin/env python3
"""
test_phi_bp.py — Validate full-matrix message BP inner product on a tree.

Graph: 3-site chain A-B-C
  A (degree-1) -- bond 0 -- B (degree-2) -- bond 1 -- C (degree-1)

BP is exact on trees. Ground truth = direct contraction.

Messages: M[ia, ib] full chi_a × chi_b matrix (rank up to D=4).

Per message update at site B (degree-2), skipping bond 0 (to A):
  phi_a[ia_B, d] = sum_{ia_B} psi_B_a[ia_B, d]  (no other a-bonds to contract)
                 * (but we contract incoming M_1 for the b-side)
  Actually:
  phi[i_skip, d] = sum_{i_other} psi[i_skip, i_other, d]
                   * sum_{j_other} M_other[i_other, j_other]   ← row-sum of M

The b-phi uses column-sum of M. This is the marginal approach.

Ground truth for 3-site chain:
  <psi_a | psi_b> = sum_{ia,ib,ic,d} psi_A_a[ia,d]*psi_B_a[ib,d]*psi_C_a[ic,d]
                                    *psi_A_b[ia,d]*psi_B_b[ib,d]*psi_C_b[ic,d]
  (all sites share physical index d, inner product contracts over d per site,
   then product over sites)
Wait -- this is the OPERATOR inner product (Heisenberg picture).
The inner product <psi_a|psi_b> for TN states is:
  = sum_{ia,ib,ic} sum_d [psi_A_a[ia,d]*psi_A_b[ia,d]] * [T_B[ib,ib]] * [psi_C_a[ic,d]*psi_C_b[ic,d]]
where T_B[ib_a, ib_b] = sum_d psi_B_a[ib_a, d] * psi_B_b[ib_b, d]

For a chain A-B-C with bond dims chi_A=chi on bond A-B, chi_B=chi on bond B-C:
  Ground truth = sum_{ia,ib,ic,d} T_A[ia,ia]*T_B[ib,ib]*T_C[ic,ic] contracted along bonds
  = Tr(M_AB @ T_B @ M_BC)  -- but simpler for chain:

Actually:
  <ψ_a|ψ_b> = sum_{ia,ib_left,ib_right,ic}
               [sum_d ψ_A_a[ia,d]*ψ_A_b[ia,d]] *   -- T_A vector (chi_A)
               [sum_d ψ_B_a[ib_l,ib_r,d]*ψ_B_b[ib_l,ib_r,d]] *  -- not right

Let me be precise. In a TN state, the physical (Pauli) index is the last.
Site A: tensor ψ_A[ia, d] -- ia is bond to B, d is physical
Site B: tensor ψ_B[ib_l, ib_r, d] -- ib_l bond to A, ib_r bond to C
Site C: tensor ψ_C[ic, d] -- ic is bond to B

<ψ_a|ψ_b> = sum_{ia,ib_l,ib_r,ic,d}
             ψ_A_a[ia,d]*ψ_B_a[ib_l,ib_r,d]*ψ_C_a[ic,d] *
             ψ_A_b[ia,d]*ψ_B_b[ib_l,ib_r,d]*ψ_C_b[ic,d]
           (bond indices must match: ia connects A-B in ket AND bra,
            ib_l, ib_r, ic same)

Wait -- in the DOUBLE LAYER, bra bond index is INDEPENDENT of ket bond index.
The inner product contracts over physical d only per site, then BP propagates.

<ψ_a|ψ_b> = sum_{ia_a,ia_b, ib_la,ib_lb, ib_ra,ib_rb, ic_a,ic_b}
             T_A[ia_a,ia_b] * T_B[ib_la,ib_lb,ib_ra,ib_rb] * T_C[ic_a,ic_b]
             * delta(ia_a,ia_b)... NO -- bonds connect sites, not ket/bra.

Let me think again from scratch.

In the double-layer network:
- Each site v has tensor T_v = sum_d ψ_a[v,...,d] * ψ_b[v,...,d]
- Each bond carries a double index (ia, ib) where ia is ket bond, ib is bra bond
- The contraction is over all (ia,ib) pairs along each bond

For 3-site chain:
  T_A[ia_a, ia_b] = sum_d ψ_A_a[ia_a,d] * ψ_A_b[ia_b,d]   [chi_a × chi_b]
  T_B[ia_a,ia_b, ic_a,ic_b] = sum_d ψ_B_a[ia_a,ic_a,d]*ψ_B_b[ia_b,ic_b,d]
  T_C[ic_a,ic_b] = sum_d ψ_C_a[ic_a,d]*ψ_C_b[ic_b,d]

<ψ_a|ψ_b> = sum_{ia_a,ia_b,ic_a,ic_b} T_A[ia_a,ia_b] * T_B[ia_a,ia_b,ic_a,ic_b] * T_C[ic_a,ic_b]
           = Tr_AB(T_A . T_B_left) . T_C  where . means contraction over double bond

This is the Kronecker double-layer. Messages on bond AB: M_AB[ia_a,ia_b] chi_a×chi_b matrix.
"""
import numpy as np
import time

D = 4

def make_tensors(chi, seed=42):
    """Make random site tensors for 3-site chain."""
    rng = np.random.RandomState(seed)
    psi_A = rng.randn(chi, D).astype(np.float64)      # [chi, D]
    psi_B = rng.randn(chi, chi, D).astype(np.float64)  # [chi, chi, D]
    psi_C = rng.randn(chi, D).astype(np.float64)       # [chi, D]
    return psi_A, psi_B, psi_C

def exact_inner(psi_Aa, psi_Ba, psi_Ca, psi_Ab, psi_Bb, psi_Cb):
    """
    Exact inner product for 3-site chain via direct contraction.
    <ψ_a|ψ_b> = sum_{ia_a,ia_b,ic_a,ic_b} T_A[ia_a,ia_b]*T_B[ia_a,ia_b,ic_a,ic_b]*T_C[ic_a,ic_b]
    """
    chi = psi_Aa.shape[0]
    # T_A[ia_a, ia_b] = sum_d psi_Aa[ia_a,d]*psi_Ab[ia_b,d]
    T_A = psi_Aa @ psi_Ab.T  # [chi,chi]
    # T_C[ic_a, ic_b]
    T_C = psi_Ca @ psi_Cb.T  # [chi,chi]
    # T_B[ia_a,ia_b,ic_a,ic_b] = sum_d psi_Ba[ia_a,ic_a,d]*psi_Bb[ia_b,ic_b,d]
    # = einsum('iad,jbd->iajb', psi_Ba.reshape(chi,chi,D), psi_Bb.reshape(chi,chi,D))
    # Reshape: psi_Ba[ia,ic,d] -> [ia*ic, d], same for psi_Bb
    T_B_flat = psi_Ba.reshape(chi*chi, D) @ psi_Bb.reshape(chi*chi, D).T  # [chi^2, chi^2]
    T_B = T_B_flat.reshape(chi, chi, chi, chi)  # [ia_a, ic_a, ia_b, ic_b]... check order
    # Actually psi_Ba[ia,ic,d] linearizes as ia*chi+ic, so T_B_flat[ia_a*chi+ic_a, ia_b*chi+ic_b]
    # = T_B[ia_a, ic_a, ia_b, ic_b]
    T_B = T_B_flat.reshape(chi, chi, chi, chi)  # [ia_a, ic_a, ia_b, ic_b]

    # Contract: sum_{ia_a,ia_b} T_A[ia_a,ia_b] * T_B[ia_a,ic_a,ia_b,ic_b]
    # = M_left[ic_a, ic_b] = einsum('ab,acbd->cd', T_A, T_B)
    M_left = np.einsum('ab,acbd->cd', T_A, T_B)  # [chi,chi]

    # Contract with T_C: sum_{ic_a,ic_b} M_left[ic_a,ic_b]*T_C[ic_a,ic_b]
    result = np.sum(M_left * T_C)
    return float(result)

def bp_inner_chain(psi_Aa, psi_Ba, psi_Ca, psi_Ab, psi_Bb, psi_Cb, n_sweeps=25, tol=1e-8):
    """
    BP inner product for 3-site chain A-B-C.
    Full matrix messages M_AB[ia_a,ia_b], M_BA[ia_a,ia_b] (chi×chi)
    and M_BC, M_CB.

    Message update using phi (incremental contraction, no T_v build):

    At site A (degree-1), sending to B:
      T_A[ia_a,ia_b] = psi_Aa @ psi_Ab.T
      M_AB_new = T_A  (no other incoming messages at A)

    At site C (degree-1), sending to B:
      T_C[ic_a,ic_b] = psi_Ca @ psi_Cb.T
      M_CB_new = T_C

    At site B (degree-2), sending to A (skip bond to A, use incoming from C):
      phi_a[ia_a, d] = sum_{ic_a} psi_Ba[ia_a,ic_a,d] * r_C[ic_a]
          where r_C[ic_a] = sum_{ic_b} M_CB[ic_a,ic_b]  (row sum = a-marginal)
      phi_b[ia_b, d] = sum_{ic_b} psi_Bb[ia_b,ic_b,d] * c_C[ic_b]
          where c_C[ic_b] = sum_{ic_a} M_CB[ic_a,ic_b]  (col sum = b-marginal)
      M_BA_new[ia_a,ia_b] = sum_d phi_a[ia_a,d] * phi_b[ia_b,d]

    Bethe PF:
      a_A = sum_{ia_a,ia_b} T_A[ia_a,ia_b] * M_BA[ia_a,ia_b]
      a_B = sum_{ia,ic} T_B[ia_a,ia_b,ic_a,ic_b] * M_AB[ia_a,ia_b]*M_CB[ic_a,ic_b]... too expensive

    Actually on a tree after convergence:
      result = (a_A * a_C) / (ze_AB * ze_BC)  -- but this isn't right for chains either.

    Simplest correct Bethe for chain:
      After convergence, M_AB = T_A (from site A with no other incoming).
      At site B: a_B = sum_{ia_a,ia_b,ic_a,ic_b} T_B * M_AB * M_CB
                     = sum_{ia_a,ia_b} M_AB[ia_a,ia_b] * (phi_Ba @ phi_Bb.T)[ia_a,ia_b] ...
      Let me just use the standard Bethe formula:

      result = prod_v a_v / prod_e ze_e

      a_A = sum_{ia_a,ia_b} T_A[ia_a,ia_b] * M_BA[ia_a,ia_b]
      a_C = sum_{ic_a,ic_b} T_C[ic_a,ic_b] * M_BC[ic_a,ic_b]
      a_B = sum_{ia_a,ia_b,ic_a,ic_b} T_B[ia_a,ic_a,ia_b,ic_b]*M_AB[ia_a,ia_b]*M_CB[ic_a,ic_b]
          -- this IS O(chi^4) but T_B never built; use phi:
          a_B = sum_d phi_a_full[d] * phi_b_full[d]
          where phi_a_full[d] = sum_{ia_a,ic_a} psi_Ba[ia_a,ic_a,d]*r_AB[ia_a]*r_CB[ic_a]
                phi_b_full[d] = sum_{ia_b,ic_b} psi_Bb[ia_b,ic_b,d]*c_AB[ia_b]*c_CB[ic_b]
          with r_AB[ia_a]=sum_{ia_b}M_AB[ia_a,ia_b], c_AB[ia_b]=sum_{ia_a}M_AB[ia_a,ia_b]
               r_CB[ic_a]=sum_{ic_b}M_CB[ic_a,ic_b], c_CB[ic_b]=sum_{ic_a}M_CB[ic_a,ic_b]

      ze_AB = sum_{ia_a,ia_b} M_AB[ia_a,ia_b]*M_BA[ia_a,ia_b]
      ze_BC = sum_{ic_a,ic_b} M_BC[ic_a,ic_b]*M_CB[ic_a,ic_b]
    """
    chi = psi_Aa.shape[0]

    # T_A, T_C as chi×chi matrices
    T_A = psi_Aa @ psi_Ab.T  # [chi,chi]
    T_C = psi_Ca @ psi_Cb.T  # [chi,chi]

    # Initialize messages to identity/chi
    M_AB = np.eye(chi)/chi  # A→B
    M_BA = np.eye(chi)/chi  # B→A
    M_BC = np.eye(chi)/chi  # B→C
    M_CB = np.eye(chi)/chi  # C→B

    def normalize(M):
        s = np.max(np.abs(M))
        return M/s if s > 1e-20 else M

    for sweep in range(n_sweeps):
        old = [M_AB.copy(), M_BA.copy(), M_BC.copy(), M_CB.copy()]

        # Site A (degree-1) sends to B: M_AB_new = T_A (no other incoming)
        M_AB = normalize(T_A)

        # Site C (degree-1) sends to B: M_CB_new = T_C
        M_CB = normalize(T_C)

        # Site B sends to A (skip bond AB=bond0, use incoming M_CB from bond1):
        # nm[ia_a, ia_b] = sum_{ic_a,ic_b,d} psi_Ba[ia_a,ic_a,d]*M_CB[ic_a,ic_b]*psi_Bb[ia_b,ic_b,d]
        # Step 1: contract M_CB into psi_Ba: phi[ia_a,ic_b,d] = sum_{ic_a} psi_Ba[ia_a,ic_a,d]*M_CB[ic_a,ic_b]
        phi = np.einsum('iad,ij->jad', psi_Ba, M_CB)   # wait: psi_Ba[ia,ic,d], M_CB[ic_a,ic_b]
        # psi_Ba indices: 0=bond_to_A(ia), 1=bond_to_C(ic), 2=d
        # M_CB[ic_a, ic_b]: contract over ic (axis 1 of psi_Ba, axis 0 of M_CB)
        phi_a_mid = np.einsum('iad,ij->ajd', psi_Ba, M_CB)  # [ic_b, ia_a, D] -- wrong order
        # Correct: phi[ia_a, ic_b, d] = sum_ic psi_Ba[ia_a, ic, d] * M_CB[ic, ic_b]
        phi_mid = np.einsum('iad,ij->ajd', psi_Ba, M_CB)  # psi_Ba[ia,i,d]*M_CB[i,j] -> [ia=a? no]
        # Let me be explicit: psi_Ba[ia_a, ic_a, d], M_CB[ic_a, ic_b]
        # einsum('iad,ij->ajd', psi_Ba, M_CB): i=ia_a, a=ic_a, j=ic_b → result[a,j,d]=[ic_a=a,ic_b=j,d]
        # That's wrong. Let me use explicit axes:
        # phi[ia_a, ic_b, d] = sum_{ic_a} psi_Ba[ia_a, ic_a, d] * M_CB[ic_a, ic_b]
        phi_mid = np.tensordot(psi_Ba, M_CB, axes=([1],[0]))  # psi_Ba[ia_a,:,d] @ M_CB[:,ic_b] -> [ia_a,d,ic_b]
        phi_mid = phi_mid.transpose(0,2,1)  # [ia_a, ic_b, d]
        # Step 2: nm[ia_a, ia_b] = sum_{ic_b,d} phi_mid[ia_a,ic_b,d] * psi_Bb[ia_b,ic_b,d]
        nm_BA = np.einsum('acd,bcd->ab', phi_mid, psi_Bb)  # [ia_a, ia_b]
        M_BA = normalize(nm_BA)

        # Site B sends to C (skip bond BC=bond1, use incoming M_AB from bond0):
        # nm[ic_a, ic_b] = sum_{ia_a,ia_b,d} psi_Ba[ia_a,ic_a,d]*M_AB[ia_a,ia_b]*psi_Bb[ia_b,ic_b,d]
        # Step 1: contract M_AB into psi_Ba: phi[ia_b, ic_a, d] = sum_{ia_a} psi_Ba[ia_a,ic_a,d]*M_AB[ia_a,ia_b]
        phi_mid2 = np.tensordot(M_AB, psi_Ba, axes=([0],[0]))  # M_AB[ia_a,ia_b]*psi_Ba[ia_a,ic_a,d] -> [ia_b,ic_a,d]
        # Step 2: nm[ic_a, ic_b] = sum_{ia_b,d} phi_mid2[ia_b,ic_a,d] * psi_Bb[ia_b,ic_b,d]
        nm_BC = np.einsum('acd,aed->ce', phi_mid2, psi_Bb)  # [ic_a, ic_b]
        M_BC = normalize(nm_BC)

        # Convergence
        delta = max(np.max(np.abs(M_AB-old[0])), np.max(np.abs(M_BA-old[1])),
                    np.max(np.abs(M_BC-old[2])), np.max(np.abs(M_CB-old[3])))
        if delta < tol:
            break

    # Bethe PF
    a_A = float(np.sum(T_A * M_BA))
    a_C = float(np.sum(T_C * M_BC))

    # a_B = sum_{ia_a,ia_b,ic_a,ic_b,d} psi_Ba[ia_a,ic_a,d]*M_AB[ia_a,ia_b]*M_CB[ic_a,ic_b]*psi_Bb[ia_b,ic_b,d]
    # Step1: contract M_AB into psi_Ba: [ia_b, ic_a, d]
    tmp = np.tensordot(M_AB, psi_Ba, axes=([0],[0]))   # [ia_b, ic_a, d]
    # Step2: contract M_CB: [ia_b, ic_b, d] = sum_ic_a tmp[ia_b,ic_a,d]*M_CB[ic_a,ic_b]
    tmp2 = np.tensordot(tmp, M_CB, axes=([1],[0]))     # [ia_b, d, ic_b]
    tmp2 = tmp2.transpose(0,2,1)                        # [ia_b, ic_b, d]
    # Step3: contract psi_Bb[ia_b,ic_b,d]: a_B = sum_{ia_b,ic_b,d} tmp2*psi_Bb
    a_B = float(np.einsum('abd,abd->', tmp2, psi_Bb))

    ze_AB = float(np.sum(M_AB * M_BA))
    ze_BC = float(np.sum(M_BC * M_CB))

    if abs(ze_AB) < 1e-30 or abs(ze_BC) < 1e-30:
        return 0.0

    result = a_A * a_B * a_C / (ze_AB * ze_BC)
    return result

# ── Tests ─────────────────────────────────────────────────────────────────────
print("=== 3-site chain BP inner product test ===\n")

for chi in [4, 10, 18, 60]:
    psi_Aa, psi_Ba, psi_Ca = make_tensors(chi, seed=42)
    psi_Ab, psi_Bb, psi_Cb = make_tensors(chi, seed=99)

    gt = exact_inner(psi_Aa, psi_Ba, psi_Ca, psi_Ab, psi_Bb, psi_Cb)

    t0 = time.time()
    approx = bp_inner_chain(psi_Aa, psi_Ba, psi_Ca, psi_Ab, psi_Bb, psi_Cb)
    t1 = time.time()

    ratio = approx/gt if abs(gt) > 1e-20 else float('nan')
    print(f"chi={chi:3d}:  exact={gt:+.6f}  bp={approx:+.6f}  ratio={ratio:.4f}  t={t1-t0:.3f}s")

# Self-overlap test
print("\nSelf-overlap (expect ratio=1.0):")
for chi in [4, 18, 60]:
    psi_A, psi_B, psi_C = make_tensors(chi, seed=42)
    gt_self = exact_inner(psi_A, psi_B, psi_C, psi_A, psi_B, psi_C)
    approx_self = bp_inner_chain(psi_A, psi_B, psi_C, psi_A, psi_B, psi_C)
    ratio = approx_self/gt_self if abs(gt_self)>1e-20 else float('nan')
    print(f"chi={chi:3d}:  exact={gt_self:+.6f}  bp={approx_self:+.6f}  ratio={ratio:.4f}")
