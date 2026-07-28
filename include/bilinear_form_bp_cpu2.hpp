#pragma once
/*
 * bilinear_form_bp_cpu2.hpp — Correct O(chi^3*D) inner product via incremental contraction.
 *
 * Algorithm (validated in test_phi_bp.py):
 * Messages are full chi_a×chi_b matrices — never rank-1, never T_v built.
 *
 * Per message update at site src, skipping bond skip_k, using incoming messages M_k:
 *   For each non-skip bond k in order (say k1, then k2 for degree-3):
 *     Contract M_k into psi:
 *       phi_a[i_skip, contracted_others, d] = sum_{i_k} psi_a[i_skip,...,i_k,...,d] * M_k[i_k, j_k]
 *     After contracting all non-skip bonds:
 *       nm[i_skip_a, i_skip_b] = sum_d phi_a[i_skip_a,d] * phi_b[i_skip_b,d]
 *   Result nm is the new message M_skip (chi_a_skip × chi_b_skip matrix).
 *
 * Cost: O(chi^3 * D) per message update for degree-3 sites.
 * At chi=200: 200^3 * 4 * 76bonds * 2dirs * 25sweeps ≈ 30 billion ops — needs GPU.
 * At chi=18:  18^3  * 4 * 152 * 25 ≈ 90M ops — fast on CPU.
 *
 * Bethe PF: same incremental contraction with all bonds.
 *   a_v = sum_d phi_a_full[d] * phi_b_full[d]
 *   ze_e = sum_{ia,ib} M_fwd[ia,ib] * M_bwd[ia,ib]
 */

#include "grid.hpp"
#include "types.hpp"
#include "bilinear_form_bp.hpp"   // for StateSnapshot
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

// ── Matrix helpers (row-major) ────────────────────────────────────────────────

// C[m,n] += A[m,k] * B[k,n]  (SGEMM: row-major, no transpose)
static void matmul_nn(const float* A, const float* B, float* C, int m, int k, int n) {
    for (int i=0; i<m; i++)
        for (int p=0; p<k; p++) {
            float a = A[i*k+p];
            for (int j=0; j<n; j++) C[i*n+j] += a * B[p*n+j];
        }
}

// C[m,n] += A[k,m]^T * B[k,n]  (A transposed)
static void matmul_tn(const float* A, const float* B, float* C, int k, int m, int n) {
    for (int p=0; p<k; p++)
        for (int i=0; i<m; i++) {
            float a = A[p*m+i];
            for (int j=0; j<n; j++) C[i*n+j] += a * B[p*n+j];
        }
}

// ── BLF2 — inner product via correct incremental message contraction ──────────

struct BLF2 {
    // Compute <psi_a|psi_b> via full-matrix message BP.
    // psi_a, psi_b: CPU copies (from StateSnapshot).
    // msgs_a, msgs_b: single-layer messages for warm-start initialization.
    static float compute(
        StateSnapshot sa,   // by value — we normalize psi in-place
        StateSnapshot sb,
        int   n_sweeps = 25,
        float tol      = 1e-5f,
        bool  verbose  = false)
    {
        const Grid& ga = *sa.grid;
        const Grid& gb = *sb.grid;
        int n_sites = ga.n_sites;
        int n_bonds = ga.n_bonds;

        if (verbose) {
            float na=0.f, nb_=0.f;
            if (!sa.psi.empty() && sa.psi.size()>13) for(auto v:sa.psi[13]) na=std::max(na,std::abs(v));
            if (!sb.psi.empty() && sb.psi.size()>13) for(auto v:sb.psi[13]) nb_=std::max(nb_,std::abs(v));
            printf("  [BLF2] site13 psi_a max=%.4e psi_b max=%.4e  n_bonds=%d\n",na,nb_,n_bonds);
            if (n_bonds>0) printf("  [BLF2] bond0 chi_a=%d chi_b=%d\n",
                ga.bonds[0].chi, gb.bonds[0].chi);
        }

        // Normalize psi per site — accumulate log scale for Bethe correction.
        double log_scale = 0.0;
        for (int sid=0; sid<n_sites; sid++) {
            float s=0.f; for(auto v:sa.psi[sid]) s=std::max(s,std::abs(v));
            if(s>1e-30f){for(auto& v:sa.psi[sid])v/=s; log_scale+=std::log((double)s);}
            s=0.f; for(auto v:sb.psi[sid]) s=std::max(s,std::abs(v));
            if(s>1e-30f){for(auto& v:sb.psi[sid])v/=s; log_scale+=std::log((double)s);}
        }
        // Note: if psi is already normalized (max=1 per site), log_scale≈0 and this is a no-op.

        // Per-site metadata
        struct SiteMeta {
            int nb;
            int chi_a[MAX_DEGREE], chi_b[MAX_DEGREE];
            int vola, volb;
            int bond_id[MAX_DEGREE];
            int msg_idx[MAX_DEGREE];   // incoming msg index for this site
        };
        std::vector<SiteMeta> smeta(n_sites);
        for (auto& site : ga.sites) {
            int sid = site.id;
            auto& m = smeta[sid];
            m.nb = 0; m.vola = 1; m.volb = 1;
            for (int d=0; d<MAX_DEGREE; d++) {
                int bid = site.bond_id[d];
                if (bid < 0) continue;
                int k = m.nb++;
                m.chi_a[k] = ga.bonds[bid].chi;
                m.chi_b[k] = gb.bonds[bid].chi;
                m.vola *= m.chi_a[k];
                m.volb *= m.chi_b[k];
                m.bond_id[k] = bid;
                // Incoming message to this site from bond bid
                m.msg_idx[k] = (sid == ga.bonds[bid].site_a)
                    ? ga.bonds[bid].msg_from_b()   // site_a receives from_b
                    : ga.bonds[bid].msg_from_a();  // site_b receives from_a
            }
        }

        // Messages: M_fwd[bid][ia*cb+ib] (site_a→site_b), M_bwd[bid][ia*cb+ib] (site_b→site_a)
        // Size: chi_a × chi_b per bond
        auto chi_a_bond = [&](int bid) { return ga.bonds[bid].chi; };
        auto chi_b_bond = [&](int bid) { return gb.bonds[bid].chi; };

        std::vector<std::vector<float>> M_fwd(n_bonds), M_bwd(n_bonds);
        for (auto& bond : ga.bonds) {
            int bid = bond.id;
            int ca = chi_a_bond(bid), cb = chi_b_bond(bid);
            int mn = std::min(ca, cb);
            M_fwd[bid].assign(ca*cb, 0.f);
            M_bwd[bid].assign(ca*cb, 0.f);
            // Identity warm-start: diagonal = 1/min(ca,cb)
            // Kronecker diagonal warm-start from single-layer messages is unreliable
            // because QR-gauge messages may have near-zero diagonal entries.
            for (int i=0; i<mn; i++) {
                M_fwd[bid][i*cb+i] = 1.f/mn;
                M_bwd[bid][i*cb+i] = 1.f/mn;
            }
        }

        // Helper: get incoming message matrix for site src, bond slot k
        // Returns pointer to M_fwd or M_bwd depending on direction
        auto get_msg = [&](int src, int k) -> const float* {
            int bid = smeta[src].bond_id[k];
            return (src == ga.bonds[bid].site_a) ? M_bwd[bid].data() : M_fwd[bid].data();
        };
        auto get_out = [&](int src, int skip_k) -> std::vector<float>& {
            int bid = smeta[src].bond_id[skip_k];
            return (src == ga.bonds[bid].site_a) ? M_fwd[bid] : M_bwd[bid];
        };

        // Normalization
        auto normalize = [](std::vector<float>& M) {
            float s = 0.f; for (auto v:M) s = std::max(s, std::abs(v));
            if (s > 1e-20f) for (auto& v:M) v /= s;
        };

        // BP sweeps
        for (int sweep=0; sweep<n_sweeps; sweep++) {
            float max_delta = 0.f;

            for (auto& bond : ga.bonds) {
                for (int dir=0; dir<2; dir++) {
                    int src    = (dir==0) ? bond.site_a : bond.site_b;
                    auto& sm   = smeta[src];
                    int nb     = sm.nb;
                    int skip_k = -1;
                    for (int k=0; k<nb; k++) if (sm.bond_id[k]==bond.id) { skip_k=k; break; }
                    if (skip_k < 0) continue;

                    int ca_skip = sm.chi_a[skip_k];
                    int cb_skip = sm.chi_b[skip_k];

                    const float* psi_a = sa.psi[src].data();
                    const float* psi_b = sb.psi[src].data();

                    // ── Compute nm[ca_skip, cb_skip] ────────────────────────
                    // Contract all non-skip incoming messages into psi_a and psi_b separately,
                    // then outer product over d.
                    //
                    // phi_a[i_skip, d]: contract all non-skip bonds into psi_a
                    // For degree-2 (one non-skip bond k1):
                    //   phi_a[i_skip, d] = sum_{i_k1} sum_{j_k1} psi_a[i_skip,i_k1,d]*M_k1[i_k1,j_k1]
                    //   Note: the j_k1 dimension is summed out (contracted with psi_b side later)
                    //   But for phi_a we only contract over the a-indices of incoming messages.
                    //   The correct formula (from Python validation):
                    //   phi_a[i_skip,d] result when we contract over BOTH i_k1 and j_k1 doesn't work.
                    //
                    // The correct incremental approach (from Python):
                    //   intermediate[i_skip, j_k1, d] = sum_{i_k1} psi_a[i_skip,i_k1,d]*M_k1[i_k1,j_k1]
                    //   nm[i_skip_a, i_skip_b] = sum_{j_k1,d} intermediate[i_skip_a,j_k1,d]*psi_b[i_skip_b,j_k1,d]
                    //
                    // For degree-3 (two non-skip bonds k1, k2), contract k1 first:
                    //   inter1[i_skip, i_k2, j_k1, d] = sum_{i_k1} psi_a[...]*M_k1[i_k1,j_k1]
                    //   inter2[i_skip, j_k1, j_k2, d] = sum_{i_k2} inter1[...]*M_k2[i_k2,j_k2]  ← WRONG order
                    // Actually psi_b side contracts k1 and k2 out of psi_b.
                    // Let's implement for degree-2 first (the common case), degree-3 separately.

                    std::vector<float> nm(ca_skip * cb_skip, 0.f);

                    if (nb == 1) {
                        // Degree-1 site (leaf): nm = psi_a @ psi_b.T  directly
                        // psi_a[i_skip, d], psi_b[i_skip, d]
                        // nm[ia, ib] = sum_d psi_a[ia,d]*psi_b[ib,d]
                        matmul_tn(psi_a, psi_b, nm.data(), D, ca_skip, cb_skip);
                        // Wait: psi_a[ia,d] row-major, psi_b[ib,d] row-major
                        // nm[ia,ib] = sum_d psi_a[ia,d]*psi_b[ib,d] = psi_a @ psi_b.T
                        // matmul_tn(A^T, B): C[m,n] += A[k,m]^T * B[k,n]
                        // We want C[ia,ib] = sum_d A[ia,d]*B[ib,d] -- this is (psi_a)(psi_b)^T
                        // Better: directly
                        std::fill(nm.begin(), nm.end(), 0.f);
                        for (int ia=0; ia<ca_skip; ia++)
                            for (int ib=0; ib<cb_skip; ib++)
                                for (int d=0; d<D; d++)
                                    nm[ia*cb_skip+ib] += psi_a[ia*D+d] * psi_b[ib*D+d];

                    } else if (nb == 2) {
                        // Degree-2 site: one non-skip bond k1
                        int k1 = (skip_k==0) ? 1 : 0;
                        int ca1 = sm.chi_a[k1], cb1 = sm.chi_b[k1];
                        const float* M1 = get_msg(src, k1);

                        // psi_a layout: if skip_k=0: psi_a[i_skip, i_k1, d] — i_skip outermost
                        //               if skip_k=1: psi_a[i_k1, i_skip, d] — i_k1 outermost
                        // Step 1: inter_a[i_skip, j_k1, d] = sum_{i_k1} psi_a[i_skip,i_k1,d]*M1[i_k1,j_k1]
                        //         where M1 is incoming to src for bond k1, so M1[ia_k1, ib_k1]
                        // For phi_a we contract over ia_k1 using M1:
                        //   inter_a[i_skip_a, ib_k1, d] = sum_{ia_k1} psi_a[i_skip_a, ia_k1, d] * M1[ia_k1, ib_k1]
                        std::vector<float> inter_a(ca_skip * cb1 * D, 0.f);
                        std::vector<float> inter_b(cb_skip * ca1 * D, 0.f);

                        if (skip_k == 0) {
                            // psi_a[i_skip_a, ia_k1, d]: stride = ca1*D, D
                            // inter_a[i_skip_a, ib_k1, d] = sum_{ia_k1} psi_a[i_skip_a,ia_k1,d]*M1[ia_k1,ib_k1]
                            for (int ia=0; ia<ca_skip; ia++)
                                for (int jb=0; jb<cb1; jb++)
                                    for (int ib=0; ib<ca1; ib++)
                                        for (int d=0; d<D; d++)
                                            inter_a[(ia*cb1+jb)*D+d] += psi_a[(ia*ca1+ib)*D+d] * M1[ib*cb1+jb];
                            // psi_b[i_skip_b, ib_k1, d]:
                            // inter_b[i_skip_b, ia_k1, d] = sum_{ib_k1} psi_b[i_skip_b,ib_k1,d]*M1[ia_k1,ib_k1]^T
                            //   = sum_{ib_k1} psi_b[..]*M1[ia_k1,ib_k1]
                            for (int ib=0; ib<cb_skip; ib++)
                                for (int ja=0; ja<ca1; ja++)
                                    for (int jb=0; jb<cb1; jb++)
                                        for (int d=0; d<D; d++)
                                            inter_b[(ib*ca1+ja)*D+d] += psi_b[(ib*cb1+jb)*D+d] * M1[ja*cb1+jb];
                        } else {
                            // psi_a[ia_k1, i_skip_a, d]:
                            // inter_a[i_skip_a, ib_k1, d] = sum_{ia_k1} psi_a[ia_k1,i_skip_a,d]*M1[ia_k1,ib_k1]
                            for (int ia=0; ia<ca1; ia++)
                                for (int jb=0; jb<cb1; jb++)
                                    for (int is=0; is<ca_skip; is++)
                                        for (int d=0; d<D; d++)
                                            inter_a[(is*cb1+jb)*D+d] += psi_a[(ia*ca_skip+is)*D+d] * M1[ia*cb1+jb];
                            for (int ib=0; ib<cb1; ib++)
                                for (int ja=0; ja<ca1; ja++)
                                    for (int js=0; js<cb_skip; js++)
                                        for (int d=0; d<D; d++)
                                            inter_b[(js*ca1+ja)*D+d] += psi_b[(ib*cb_skip+js)*D+d] * M1[ja*cb1+ib];
                        }

                        // nm[i_skip_a, i_skip_b] = sum_{j_k1, d} inter_a[i_skip_a,j_k1,d]*inter_b[i_skip_b,j_k1,d]
                        for (int ia=0; ia<ca_skip; ia++)
                            for (int ib=0; ib<cb_skip; ib++)
                                for (int j=0; j<cb1; j++)  // j = ib_k1
                                    for (int d=0; d<D; d++)
                                        nm[ia*cb_skip+ib] += inter_a[(ia*cb1+j)*D+d] * inter_b[(ib*ca1+j)*D+d];
                        // Note: inter_b is indexed by [i_skip_b, ia_k1] and inter_a by [i_skip_a, ib_k1]
                        // The j in inter_a is ib_k1, in inter_b is ia_k1 -- these must match via M1
                        // Actually on reflection the correct contraction is:
                        // nm[ia_skip, ib_skip] = sum_d inter_a[ia_skip, :, d] . inter_b[ib_skip, :, d]
                        // where inter_a[ia_skip, ib_k1, d] and inter_b[ib_skip, ia_k1, d]
                        // and ib_k1 matches ia_k1 only if the message M1 makes them the same index
                        // This is correct: the bond k1 carries (ia_k1, ib_k1) pair,
                        // inter_a contracted M1 over ia_k1 leaving ib_k1,
                        // inter_b contracted M1^T over ib_k1 leaving ia_k1... NO
                        // Let me reconsider from the Python validation.

                        // From Python (working code):
                        // phi_mid[ia_a, ic_b, d] = sum_{ic_a} psi_Ba[ia_a,ic_a,d] * M_CB[ic_a,ic_b]
                        // nm[ia_a, ia_b] = sum_{ic_b,d} phi_mid[ia_a,ic_b,d] * psi_Bb[ia_b,ic_b,d]
                        // So: contract M into psi_a to get intermediate indexed by (i_skip_a, ib_k1, d)
                        //     then contract that with psi_b[i_skip_b, ib_k1, d] -- psi_b is NOT modified

                        // Redo correctly:
                        std::fill(nm.begin(), nm.end(), 0.f);
                        // Recompute inter_a correctly
                        std::fill(inter_a.begin(), inter_a.end(), 0.f);
                        if (skip_k == 0) {
                            // psi_a[i_skip_a, ia_k1, d], M1[ia_k1, ib_k1]
                            // inter_a[i_skip_a, ib_k1, d] = sum_{ia_k1} psi_a[i_skip_a,ia_k1,d]*M1[ia_k1,ib_k1]
                            for (int ia=0; ia<ca_skip; ia++)
                                for (int ib=0; ib<ca1; ib++)
                                    for (int jb=0; jb<cb1; jb++)
                                        for (int d=0; d<D; d++)
                                            inter_a[(ia*cb1+jb)*D+d] += psi_a[(ia*ca1+ib)*D+d]*M1[ib*cb1+jb];
                            // nm[ia_skip, ib_skip] = sum_{ib_k1,d} inter_a[ia_skip,ib_k1,d]*psi_b[ib_skip,ib_k1,d]
                            for (int ia=0; ia<ca_skip; ia++)
                                for (int ib_s=0; ib_s<cb_skip; ib_s++)
                                    for (int jb=0; jb<cb1; jb++)
                                        for (int d=0; d<D; d++)
                                            nm[ia*cb_skip+ib_s] += inter_a[(ia*cb1+jb)*D+d]*psi_b[(ib_s*cb1+jb)*D+d];
                        } else {
                            // psi_a[ia_k1, i_skip_a, d], M1[ia_k1, ib_k1]
                            // inter_a[i_skip_a, ib_k1, d] = sum_{ia_k1} psi_a[ia_k1,i_skip_a,d]*M1[ia_k1,ib_k1]
                            for (int ia=0; ia<ca1; ia++)
                                for (int is=0; is<ca_skip; is++)
                                    for (int jb=0; jb<cb1; jb++)
                                        for (int d=0; d<D; d++)
                                            inter_a[(is*cb1+jb)*D+d] += psi_a[(ia*ca_skip+is)*D+d]*M1[ia*cb1+jb];
                            // psi_b[ib_k1, i_skip_b, d]
                            // nm[ia_skip, ib_skip] = sum_{ib_k1,d} inter_a[ia_skip,ib_k1,d]*psi_b[ib_k1,ib_skip,d]
                            for (int ia=0; ia<ca_skip; ia++)
                                for (int ib_s=0; ib_s<cb_skip; ib_s++)
                                    for (int jb=0; jb<cb1; jb++)
                                        for (int d=0; d<D; d++)
                                            nm[ia*cb_skip+ib_s] += inter_a[(ia*cb1+jb)*D+d]*psi_b[(jb*cb_skip+ib_s)*D+d];
                        }

                    } else {
                        // Degree-3: two non-skip bonds k1, k2
                        // Same two-step approach as degree-2, validated in test_phi_bp.py:
                        // Step 1: inter[ia_skip, ia_k1, ib_k2, d] =
                        //           sum_{ia_k2} psi_a[ia_skip,ia_k1,ia_k2,d] * M_k2[ia_k2,ib_k2]
                        // Step 2: inter2[ia_skip, ib_k1, ib_k2, d] =
                        //           sum_{ia_k1} inter[ia_skip,ia_k1,ib_k2,d] * M_k1[ia_k1,ib_k1]
                        // Step 3: nm[ia_skip, ib_skip] =
                        //           sum_{ib_k1,ib_k2,d} inter2[ia_skip,ib_k1,ib_k2,d]
                        //                              * psi_b[ib_skip,ib_k1,ib_k2,d]
                        // Cost: O(chi^3*D) — same as degree-2

                        int k1=-1, k2=-1;
                        for (int k=0; k<nb; k++) if(k!=skip_k) { if(k1<0) k1=k; else k2=k; }

                        int ca1=sm.chi_a[k1], cb1=sm.chi_b[k1];
                        int ca2=sm.chi_a[k2], cb2=sm.chi_b[k2];
                        const float* M1 = get_msg(src, k1);
                        const float* M2 = get_msg(src, k2);

                        // Step 1: inter[ia_skip, ia_k1, ib_k2, d] — size ca_skip*ca1*cb2*D
                        std::vector<float> inter(ca_skip*ca1*cb2*D, 0.f);
                        for (int is=0; is<ca_skip; is++)
                        for (int i1=0; i1<ca1; i1++)
                        for (int i2=0; i2<ca2; i2++) {
                            int idx[3]; idx[skip_k]=is; idx[k1]=i1; idx[k2]=i2;
                            int fa=(idx[0]*sm.chi_a[1]+idx[1])*sm.chi_a[2]+idx[2];
                            for (int jb2=0; jb2<cb2; jb2++)
                                for (int d=0; d<D; d++)
                                    inter[((is*ca1+i1)*cb2+jb2)*D+d] +=
                                        psi_a[fa*D+d] * M2[i2*cb2+jb2];
                        }

                        // Step 2: inter2[ia_skip, ib_k1, ib_k2, d] — size ca_skip*cb1*cb2*D
                        std::vector<float> inter2(ca_skip*cb1*cb2*D, 0.f);
                        for (int is=0; is<ca_skip; is++)
                        for (int i1=0; i1<ca1; i1++)
                        for (int jb2=0; jb2<cb2; jb2++)
                        for (int jb1=0; jb1<cb1; jb1++)
                            for (int d=0; d<D; d++)
                                inter2[((is*cb1+jb1)*cb2+jb2)*D+d] +=
                                    inter[((is*ca1+i1)*cb2+jb2)*D+d] * M1[i1*cb1+jb1];

                        // Step 3: nm[ia_skip, ib_skip] — contract inter2 with psi_b
                        for (int is=0; is<ca_skip; is++)
                        for (int ib_s=0; ib_s<cb_skip; ib_s++)
                        for (int jb1=0; jb1<cb1; jb1++)
                        for (int jb2=0; jb2<cb2; jb2++) {
                            int jdx[3]; jdx[skip_k]=ib_s; jdx[k1]=jb1; jdx[k2]=jb2;
                            int fb=(jdx[0]*sm.chi_b[1]+jdx[1])*sm.chi_b[2]+jdx[2];
                            for (int d=0; d<D; d++)
                                nm[is*cb_skip+ib_s] +=
                                    inter2[((is*cb1+jb1)*cb2+jb2)*D+d] * psi_b[fb*D+d];
                        }
                    }

                    // Normalize
                    float s=0.f; for(auto v:nm) s=std::max(s,std::abs(v));
                    if(s>1e-20f) for(auto& v:nm) v/=s;

                    // Convergence delta
                    auto& old = get_out(src, skip_k);
                    for (int i=0; i<(int)nm.size(); i++)
                        max_delta = std::max(max_delta, std::abs(nm[i]-old[i]));
                    old = nm;
                }
            }
            if (verbose) printf("  [BLF2] sweep %d max_delta=%.2e\n", sweep, max_delta);
            if (max_delta < tol) { if (verbose) printf("  [BLF2] converged sweep %d\n", sweep); break; }
        }

        // Bethe PF
        double log_num=0.0, sign_num=1.0;
        double log_sa=0.0, log_sb=0.0;
        // (psi already normalized in StateSnapshot — use log_scale from snapshot if available)

        for (auto& site : ga.sites) {
            int sid = site.id;
            auto& sm = smeta[sid];
            int nb = sm.nb;
            const float* psi_a = sa.psi[sid].data();
            const float* psi_b = sb.psi[sid].data();

            // a_v = sum_{ia_flat, ib_flat, d} psi_a[ia,d]*psi_b[ib,d] * prod_k M_k[ia_k, ib_k]
            // Joint enumeration over (ia_flat, ib_flat) — correct, no marginal factorization.
            int vola = sm.vola, volb = sm.volb;
            double a_v = 0.0;
            for (int ia=0; ia<vola; ia++) {
                int tmp=ia;
                int ia_k[MAX_DEGREE];
                for (int k=nb-1; k>=0; k--) { ia_k[k]=tmp%sm.chi_a[k]; tmp/=sm.chi_a[k]; }

                for (int ib=0; ib<volb; ib++) {
                    int tmp2=ib;
                    int ib_k[MAX_DEGREE];
                    for (int k=nb-1; k>=0; k--) { ib_k[k]=tmp2%sm.chi_b[k]; tmp2/=sm.chi_b[k]; }

                    // Joint message product M_k[ia_k, ib_k] for all bonds
                    float msg_prod = 1.f;
                    for (int k=0; k<nb; k++) {
                        const float* Mk = get_msg(sid, k);
                        msg_prod *= Mk[ia_k[k]*sm.chi_b[k] + ib_k[k]];
                    }

                    for (int d=0; d<D; d++)
                        a_v += (double)psi_a[ia*D+d] * (double)psi_b[ib*D+d] * msg_prod;
                }
            }
            sign_num *= (a_v>0?1.0:(a_v<0?-1.0:0.0));
            log_num  += (a_v!=0?std::log(std::abs(a_v)):-1e30);
        }

        double log_den=0.0, sign_den=1.0;
        for (auto& bond : ga.bonds) {
            int bid = bond.id;
            int ca = chi_a_bond(bid), cb = chi_b_bond(bid);
            double ze=0.0;
            for (int i=0; i<ca*cb; i++) ze += (double)M_fwd[bid][i]*(double)M_bwd[bid][i];
            sign_den *= (ze>0?1.0:(ze<0?-1.0:0.0));
            log_den  += (std::abs(ze)>1e-30?std::log(std::abs(ze)):-1e30);
        }

        if (sign_num==0.0) return 0.f;
        double result = sign_num*sign_den*std::exp(log_num-log_den+log_scale);
        if (verbose) printf("  [BLF2] overlap=%.6f\n",(float)result);
        return (float)result;
    }
};
