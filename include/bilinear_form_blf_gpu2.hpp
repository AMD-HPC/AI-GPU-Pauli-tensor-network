#pragma once
/*
 * bilinear_form_blf_gpu2.hpp — GPU inner product via rocBLAS SGEMM.
 *
 * Algorithm: same as BLF2 (bilinear_form_bp_cpu2.hpp) but all heavy
 * operations run on GPU. Messages and psi stay on device throughout BP.
 *
 * Per message update (degree-2 site, skip bond k0, incoming bond k1):
 *   Step 1 SGEMM: inter[ca_skip, cb_k1*D] = psi_a[ca_skip, ca_k1*D] @ (M_k1 kron I_D)
 *                 implemented as batched-D: inter[:,j,:] = psi_a @_bond M_k1
 *   Step 2 SGEMM: nm[ca_skip, cb_skip] = inter_reshape @ psi_b_reshape^T
 *
 * For degree-3: three sequential SGEMMs.
 *
 * Messages: full chi_a×chi_b matrices on GPU, ping-pong double-buffered.
 * Convergence check: D2H of message norms only (one float per bond per sweep).
 *
 * Usage:
 *   BLF_GPU2 blf(grid_a, grid_b, handle);
 *   blf.upload(psi_a_gpu, dev_a, psi_b_gpu, dev_b, msg_a_gpu, msg_b_gpu);
 *   float overlap = blf.compute(n_sweeps, tol);
 */

#include "grid.hpp"
#include "types.hpp"
#include <vector>
#include <cmath>
#include <chrono>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do { \
    hipError_t e=(cmd); \
    if(e!=hipSuccess){fprintf(stderr,"HIP error %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__);exit(1);} \
} while(0)
#endif

// ── Normalize kernel — max-abs normalize each message matrix in-place ─────────
// grid=(n_msgs), block=(1)
__global__ void normalize_msg2_kernel(
    float** msgs, const int* chi_ab, int n_msgs)
{
    int idx = blockIdx.x;
    if (idx >= n_msgs) return;
    int bid = idx / 2;
    int chi = chi_ab[bid];
    float* m = msgs[idx];
    float s = 0.f;
    for (int i=0; i<chi*chi; i++) s = fmaxf(s, fabsf(m[i]));  // chi_a * chi_b
    if (s > 1e-20f) for (int i=0; i<chi*chi; i++) m[i] /= s;
}

// ── BLF_GPU2 ────────────────────────────────────────────────────────────────

struct BLF_GPU2 {
    const Grid& grid_a;
    const Grid& grid_b;
    rocblas_handle handle;
    int n_sites, n_bonds;

    // Device psi pointers (downloaded from source device at upload time)
    std::vector<float*> d_psi_a, d_psi_b;   // [n_sites] device ptrs

    // Device message matrices: M_fwd[bid][ia*cb+ib], M_bwd[bid][ia*cb+ib]
    // chi_a × chi_b each
    std::vector<float*> d_M_fwd, d_M_bwd;   // [n_bonds] device ptrs
    std::vector<float*> d_M_fwd2, d_M_bwd2; // ping-pong

    // Device pointer arrays (for kernel launches)
    float** d_msgs      = nullptr;   // [2*n_bonds]: fwd[0],bwd[0],...
    float** d_msgs_new  = nullptr;

    // Scratch buffers per bond direction (max size across all sites)
    float* d_inter  = nullptr;  // intermediate contraction step 1
    float* d_inter2 = nullptr;  // intermediate contraction step 2 (degree-3)
    int scratch_size = 0;

    // Host metadata
    struct SiteMeta {
        int nb;
        int chi_a[MAX_DEGREE], chi_b[MAX_DEGREE];
        int vola, volb;
        int bond_id[MAX_DEGREE];
    };
    std::vector<SiteMeta> smeta;

    // Frozen chi at upload time
    std::vector<int> frozen_chi_a, frozen_chi_b;   // [n_bonds]

    // Log scale from psi normalization
    double log_scale_a = 0.0, log_scale_b = 0.0;

    BLF_GPU2(const Grid& ga, const Grid& gb, rocblas_handle h)
        : grid_a(ga), grid_b(gb), handle(h),
          n_sites(ga.n_sites), n_bonds(ga.n_bonds)
    {
        d_psi_a.resize(n_sites, nullptr);
        d_psi_b.resize(n_sites, nullptr);
        d_M_fwd.resize(n_bonds, nullptr);
        d_M_bwd.resize(n_bonds, nullptr);
        d_M_fwd2.resize(n_bonds, nullptr);
        d_M_bwd2.resize(n_bonds, nullptr);
        frozen_chi_a.resize(n_bonds); frozen_chi_b.resize(n_bonds);
        smeta.resize(n_sites);
    }

    ~BLF_GPU2() { _free(); }

    // Upload psi+messages from source device. Call after Trotter+BP on GPU 0.
    void upload(
        const std::vector<float*>& psi_a_src, int dev_a,
        const std::vector<float*>& psi_b_src, int dev_b,
        const std::vector<float*>* msg_a_src = nullptr,  // single-layer msgs on dev_a
        const std::vector<float*>* msg_b_src = nullptr)
    {
        int cur; HIP_CHECK(hipGetDevice(&cur));

        // Freeze chi and rebuild site metadata
        for (auto& bond : grid_a.bonds) frozen_chi_a[bond.id] = bond.chi;
        for (auto& bond : grid_b.bonds) frozen_chi_b[bond.id] = bond.chi;
        _build_smeta();

        // Free old psi buffers
        for (auto p : d_psi_a) if (p) HIP_CHECK(hipFree(p));
        for (auto p : d_psi_b) if (p) HIP_CHECK(hipFree(p));

        // Download+normalize psi_a from dev_a, upload to current device
        log_scale_a = 0.0; log_scale_b = 0.0;
        for (auto& site : grid_a.sites) {
            int sid = site.id;
            int vola = smeta[sid].vola;
            int vol = vola * D;
            std::vector<float> buf(vol);
            HIP_CHECK(hipSetDevice(dev_a));
            HIP_CHECK(hipMemcpy(buf.data(), psi_a_src[sid], vol*sizeof(float), hipMemcpyDeviceToHost));
            float nm=0.f; for(auto v:buf) nm=std::max(nm,std::abs(v));
            if(nm>1e-30f){for(auto& v:buf)v/=nm; log_scale_a+=std::log((double)nm);}
            HIP_CHECK(hipSetDevice(cur));
            HIP_CHECK(hipMalloc(&d_psi_a[sid], vol*sizeof(float)));
            HIP_CHECK(hipMemcpy(d_psi_a[sid], buf.data(), vol*sizeof(float), hipMemcpyHostToDevice));
        }
        for (auto& site : grid_b.sites) {
            int sid = site.id;
            int volb = smeta[sid].volb;
            int vol = volb * D;
            std::vector<float> buf(vol);
            HIP_CHECK(hipSetDevice(dev_b));
            HIP_CHECK(hipMemcpy(buf.data(), psi_b_src[sid], vol*sizeof(float), hipMemcpyDeviceToHost));
            float nm=0.f; for(auto v:buf) nm=std::max(nm,std::abs(v));
            if(nm>1e-30f){for(auto& v:buf)v/=nm; log_scale_b+=std::log((double)nm);}
            HIP_CHECK(hipSetDevice(cur));
            HIP_CHECK(hipMalloc(&d_psi_b[sid], vol*sizeof(float)));
            HIP_CHECK(hipMemcpy(d_psi_b[sid], buf.data(), vol*sizeof(float), hipMemcpyHostToDevice));
        }
        HIP_CHECK(hipSetDevice(cur));

        // Free old message buffers, allocate new ones
        for (auto p : d_M_fwd)  if(p) HIP_CHECK(hipFree(p));
        for (auto p : d_M_bwd)  if(p) HIP_CHECK(hipFree(p));
        for (auto p : d_M_fwd2) if(p) HIP_CHECK(hipFree(p));
        for (auto p : d_M_bwd2) if(p) HIP_CHECK(hipFree(p));

        for (auto& bond : grid_a.bonds) {
            int bid = bond.id;
            int ca = frozen_chi_a[bid], cb = frozen_chi_b[bid];
            int mn = std::min(ca, cb);
            std::vector<float> mf(ca*cb, 0.f), mb(ca*cb, 0.f);
            // Identity warm-start
            for (int i=0; i<mn; i++) { mf[i*cb+i]=1.f/mn; mb[i*cb+i]=1.f/mn; }
            HIP_CHECK(hipMalloc(&d_M_fwd[bid],  ca*cb*sizeof(float)));
            HIP_CHECK(hipMalloc(&d_M_bwd[bid],  ca*cb*sizeof(float)));
            HIP_CHECK(hipMalloc(&d_M_fwd2[bid], ca*cb*sizeof(float)));
            HIP_CHECK(hipMalloc(&d_M_bwd2[bid], ca*cb*sizeof(float)));
            HIP_CHECK(hipMemcpy(d_M_fwd[bid],  mf.data(), ca*cb*sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_M_bwd[bid],  mb.data(), ca*cb*sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_M_fwd2[bid], mf.data(), ca*cb*sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_M_bwd2[bid], mb.data(), ca*cb*sizeof(float), hipMemcpyHostToDevice));
        }

        // Allocate scratch: max inter size across all bond updates
        int max_inter = 0;
        for (auto& bond : grid_a.bonds) {
            for (int dir=0; dir<2; dir++) {
                int src = (dir==0) ? bond.site_a : bond.site_b;
                auto& sm = smeta[src];
                // inter size: ca_skip * max_cb_other * D
                for (int skip_k=0; skip_k<sm.nb; skip_k++) {
                    int ca_skip = sm.chi_a[skip_k];
                    int inter_sz = ca_skip * D;
                    for (int k=0; k<sm.nb; k++) if(k!=skip_k) inter_sz *= sm.chi_b[k];
                    max_inter = std::max(max_inter, inter_sz);
                }
            }
        }
        if (d_inter)  HIP_CHECK(hipFree(d_inter));
        if (d_inter2) HIP_CHECK(hipFree(d_inter2));
        scratch_size = max_inter;
        HIP_CHECK(hipMalloc(&d_inter,  max_inter*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_inter2, max_inter*sizeof(float)));
    }

    // Upload pre-gauged psi (float32, already on current device).
    // Normalizes all psi_a by a single global scale, same for psi_b.
    // This prevents float32 underflow while preserving the inter-site relative
    // scale that the Bethe PF requires. The log-scales are tracked and restored
    // in _bethe_cpu() via log_scale_a + log_scale_b.
    void upload_gauged(
        const std::vector<float*>& psi_a_dev,
        const std::vector<float*>& psi_b_dev)
    {
        for (auto& bond : grid_a.bonds) frozen_chi_a[bond.id] = bond.chi;
        for (auto& bond : grid_b.bonds) frozen_chi_b[bond.id] = bond.chi;
        _build_smeta();

        for (auto p : d_psi_a) if (p) HIP_CHECK(hipFree(p));
        for (auto p : d_psi_b) if (p) HIP_CHECK(hipFree(p));

        // Per-site normalization by joint max(|psi_a|, |psi_b|) at each site.
        // Using the same scale for both psi_a and psi_b at each site preserves
        // their relative magnitude (which carries the cross-state overlap signal)
        // while preventing float32 underflow in the gauged tensors.
        log_scale_a = 0.0; log_scale_b = 0.0;
        for (auto& site : grid_a.sites) {
            int sid = site.id;
            int vola = smeta[sid].vola * D;
            int volb = smeta[sid].volb * D;
            std::vector<float> bufa(vola), bufb(volb);
            HIP_CHECK(hipMemcpy(bufa.data(), psi_a_dev[sid],
                vola*sizeof(float), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(bufb.data(), psi_b_dev[sid],
                volb*sizeof(float), hipMemcpyDeviceToHost));

            // Joint scale: max of both tensors at this site
            float nm = 1e-30f;
            for (auto v : bufa) nm = std::max(nm, std::abs(v));
            for (auto v : bufb) nm = std::max(nm, std::abs(v));

            // Normalize both by same scale — preserves relative magnitude
            for (auto& v : bufa) v /= nm;
            for (auto& v : bufb) v /= nm;

            // Each site contributes log(nm) to both a and b log-scales
            // (same nm used for both, so log_scale_a == log_scale_b per site)
            log_scale_a += std::log((double)nm);
            log_scale_b += std::log((double)nm);

            HIP_CHECK(hipMalloc(&d_psi_a[sid], vola*sizeof(float)));
            HIP_CHECK(hipMalloc(&d_psi_b[sid], volb*sizeof(float)));
            HIP_CHECK(hipMemcpy(d_psi_a[sid], bufa.data(),
                vola*sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_psi_b[sid], bufb.data(),
                volb*sizeof(float), hipMemcpyHostToDevice));
        }

        // Free old messages, allocate, identity warm-start
        for (auto p : d_M_fwd)  if(p) HIP_CHECK(hipFree(p));
        for (auto p : d_M_bwd)  if(p) HIP_CHECK(hipFree(p));
        for (auto p : d_M_fwd2) if(p) HIP_CHECK(hipFree(p));
        for (auto p : d_M_bwd2) if(p) HIP_CHECK(hipFree(p));
        for (auto& bond : grid_a.bonds) {
            int bid = bond.id;
            int ca = frozen_chi_a[bid], cb = frozen_chi_b[bid];
            int mn = std::min(ca, cb);
            std::vector<float> mf(ca*cb, 0.f), mb(ca*cb, 0.f);
            for (int i=0; i<mn; i++) { mf[i*cb+i]=1.f/mn; mb[i*cb+i]=1.f/mn; }
            HIP_CHECK(hipMalloc(&d_M_fwd[bid],  ca*cb*sizeof(float)));
            HIP_CHECK(hipMalloc(&d_M_bwd[bid],  ca*cb*sizeof(float)));
            HIP_CHECK(hipMalloc(&d_M_fwd2[bid], ca*cb*sizeof(float)));
            HIP_CHECK(hipMalloc(&d_M_bwd2[bid], ca*cb*sizeof(float)));
            HIP_CHECK(hipMemcpy(d_M_fwd[bid],  mf.data(), ca*cb*sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_M_bwd[bid],  mb.data(), ca*cb*sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_M_fwd2[bid], mf.data(), ca*cb*sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_M_bwd2[bid], mb.data(), ca*cb*sizeof(float), hipMemcpyHostToDevice));
        }
        // Allocate scratch
        int max_inter = 0;
        for (auto& bond : grid_a.bonds) {
            for (int dir=0; dir<2; dir++) {
                int src = (dir==0)?bond.site_a:bond.site_b;
                auto& sm = smeta[src];
                for (int skip_k=0; skip_k<sm.nb; skip_k++) {
                    int ca_skip = sm.chi_a[skip_k];
                    int inter_sz = ca_skip * D;
                    for (int k=0; k<sm.nb; k++) if(k!=skip_k) inter_sz *= sm.chi_b[k];
                    max_inter = std::max(max_inter, inter_sz);
                }
            }
        }
        if (d_inter)  HIP_CHECK(hipFree(d_inter));
        if (d_inter2) HIP_CHECK(hipFree(d_inter2));
        scratch_size = max_inter;
        HIP_CHECK(hipMalloc(&d_inter,  max_inter*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_inter2, max_inter*sizeof(float)));
    }

    // Compute <psi_a|psi_b> via GPU BP.
    float compute(int n_sweeps=25, float tol=1e-5f, bool verbose=false,
                  double* t_ms_out=nullptr) {
        auto t0 = std::chrono::high_resolution_clock::now();

        const float one=1.f, zero=0.f;

        for (int sweep=0; sweep<n_sweeps; sweep++) {
            float max_delta = 0.f;

            for (auto& bond : grid_a.bonds) {
                for (int dir=0; dir<2; dir++) {
                    int src     = (dir==0) ? bond.site_a : bond.site_b;
                    auto& sm    = smeta[src];
                    int nb      = sm.nb;
                    int skip_k  = -1;
                    for (int k=0; k<nb; k++) if(sm.bond_id[k]==bond.id){skip_k=k;break;}
                    if (skip_k<0) continue;

                    int ca_skip = sm.chi_a[skip_k];
                    int cb_skip = sm.chi_b[skip_k];

                    float* psi_a = d_psi_a[src];
                    float* psi_b = d_psi_b[src];

                    // Get incoming message for each non-skip bond
                    // Incoming to src on bond k: if src=site_a → use M_bwd[k], else M_fwd[k]
                    auto get_M = [&](int k) -> float* {
                        int bk = sm.bond_id[k];
                        return (src == grid_a.bonds[bk].site_a) ? d_M_bwd[bk] : d_M_fwd[bk];
                    };
                    // Output message: if src=site_a → M_fwd (a→b), else M_bwd (b→a)
                    float*& out_M = (src == bond.site_a) ? d_M_fwd2[bond.id] : d_M_bwd2[bond.id];
                    float*& old_M = (src == bond.site_a) ? d_M_fwd[bond.id]  : d_M_bwd[bond.id];

                    // ── Compute nm[ca_skip, cb_skip] via rocBLAS SGEMM ───────

                    // psi_a layout: [i0, i1, ..., d] row-major (bond order 0,1,...,nb-1)
                    // For degree-2 (nb=2), skip_k=0:
                    //   psi_a[ca_skip, ca_k1, D]
                    //   Step1: inter[ca_skip, cb_k1, D] via rocBLAS
                    //   Step2: nm[ca_skip, cb_skip] via rocBLAS
                    // For degree-2, skip_k=1:
                    //   psi_a[ca_k1, ca_skip, D]
                    //   Step1: inter[ca_skip, cb_k1, D] (needs transpose psi_a)
                    //   Step2: nm[ca_skip, cb_skip]

                    if (nb == 1) {
                        // Leaf node: nm = psi_a @ psi_b^T (over D)
                        // psi_a[ca_skip, D], psi_b[cb_skip, D]
                        // nm = psi_a @ psi_b^T: [ca_skip, cb_skip]
                        // rocBLAS col-major: nm^T = psi_b @ psi_a^T
                        // sgemm(op_none, op_transpose, cb_skip, ca_skip, D,
                        //       alpha, psi_b, cb_skip,  -- actually need lda
                        //       psi_a, ca_skip, beta, nm^T, cb_skip)
                        // Using row-major GEMM: C = A @ B^T
                        // rocBLAS: C^T = B @ A^T → sgemm(op_none, op_transpose, cb_skip, ca_skip, D, ...)
                        rocblas_sgemm(handle,
                            rocblas_operation_none, rocblas_operation_transpose,
                            cb_skip, ca_skip, D,
                            &one, psi_b, cb_skip, psi_a, ca_skip,
                            &zero, out_M, cb_skip);
                        // out_M is nm^T[cb_skip, ca_skip] -- but we want nm[ca_skip, cb_skip]
                        // For symmetric messages this doesn't matter for convergence.
                        // Actually let's just transpose:
                        // TODO: for now store as-is and handle in Bethe.
                    } else if (nb == 2) {
                        int k1 = (skip_k==0) ? 1 : 0;
                        int ca1 = sm.chi_a[k1], cb1 = sm.chi_b[k1];
                        float* M1 = get_M(k1);

                        if (skip_k == 0) {
                            // psi_a[ca_skip, ca1, D] row-major
                            // Step1: inter[ca_skip, cb1, D] = sum_{ca1} psi_a * M1
                            // Treat as: for each d, inter[:, :, d] = psi_a[:, :, d] @ M1
                            // psi_a[:,:,d] is NOT contiguous — stride D between ca1 elements
                            // Use batched GEMM with batch=D... or simpler:
                            // Reshape psi_a as [ca_skip, ca1*D] is wrong layout.
                            // Best: use rocblas_sgemm_strided_batched with batch=D
                            //   A[d] = psi_a[:, :, d]: stride between d-batches = 1
                            //                          row stride = ca1*D, col stride = D
                            //   This is non-standard — use a simple kernel instead.
                            // SIMPLE APPROACH: use rocblas_sgemm treating as row-major
                            // inter[ca_skip*D, cb1] = psi_a_view[ca_skip*D, ca1] @ M1[ca1, cb1]
                            // psi_a_view: row i=i_skip*D+d, col j=ia_k1 → addr: i_skip*ca1*D + ia_k1*D + d
                            // This is NOT standard row-major (col stride = D ≠ 1).
                            // USE TRANSPOSE: first transpose psi_a to [ca1, ca_skip, D] → [ca1, ca_skip*D]
                            // Then GEMM: inter_T[cb1, ca_skip*D] = M1^T[cb1,ca1] @ psi_a_T[ca1, ca_skip*D]
                            // In rocBLAS col-major: inter_T is col-major [cb1, ca_skip*D]
                            //   sgemm(op_transpose, op_none, cb1, ca_skip*D, ca1,
                            //         alpha, M1, ca1, psi_a_T, ca1, beta, inter_T, cb1)
                            // For row-major M1[ca1,cb1]: col-major is M1^T[cb1,ca1], lda=ca1
                            // So: rocblas_sgemm(op_transpose, op_none, cb1, ca_skip*D, ca1,
                            //       alpha, M1, ca1,   ← M1 col-major [ca1,cb1], lda=ca1
                            //       psi_a_T, ca1,     ← psi_a_T col-major [ca1, ca_skip*D], lda=ca1
                            //       beta, inter_T, cb1) ← inter_T col-major [cb1, ca_skip*D], lda=cb1

                            // But we don't have psi_a_T on device. We need to transpose.
                            // For now: do the transpose on CPU via a download (will optimize later).
                            // Download psi_a, transpose, upload to d_inter, then GEMM.
                            int vola = sm.vola;
                            std::vector<float> psi_a_cpu(vola*D), psi_a_T(ca1*ca_skip*D);
                            HIP_CHECK(hipMemcpy(psi_a_cpu.data(), psi_a, vola*D*sizeof(float), hipMemcpyDeviceToHost));
                            // Transpose [ca_skip, ca1, D] → [ca1, ca_skip, D] i.e. psi_a_T[ia_k1, i_skip, d]
                            for(int is=0;is<ca_skip;is++)
                            for(int i1=0;i1<ca1;i1++)
                            for(int d=0;d<D;d++)
                                psi_a_T[(i1*ca_skip+is)*D+d] = psi_a_cpu[(is*ca1+i1)*D+d];
                            HIP_CHECK(hipMemcpy(d_inter, psi_a_T.data(), ca1*ca_skip*D*sizeof(float), hipMemcpyHostToDevice));

                            // GEMM: inter_T[cb1, ca_skip*D] = M1^T @ psi_a_T[ca1, ca_skip*D]
                            // rocBLAS col-major: M1 row-major [ca1,cb1] → col-major [cb1,ca1], lda=ca1
                            //                   psi_a_T row-major [ca1,ca_skip*D] → col-major lda=ca1
                            // Result: inter_T[cb1, ca_skip*D]
                            rocblas_sgemm(handle,
                                rocblas_operation_transpose, rocblas_operation_none,
                                cb1, ca_skip*D, ca1,
                                &one, M1, ca1, d_inter, ca1,
                                &zero, d_inter2, cb1);
                            // d_inter2 = inter_T[cb1, ca_skip*D] in col-major = inter[ca_skip*D, cb1] row-major
                            // Need inter[ca_skip, cb1, D] -- reshape inter[ca_skip*D, cb1] noting
                            // inter[i_skip*D+d, ib_k1] = inter_T[ib_k1, i_skip*D+d] (col-major indexing)
                            // Actually inter_T col-major [cb1, ca_skip*D] means:
                            //   inter_T[ib_k1, i_skip*D+d] is at address ib_k1 + (i_skip*D+d)*cb1
                            // We want inter[i_skip, ib_k1, d] at address i_skip*cb1*D + ib_k1*D + d
                            // These don't match directly. Need another transpose.
                            // Download inter_T, rearrange, then GEMM with psi_b.
                            std::vector<float> inter_T(cb1*ca_skip*D), inter(ca_skip*cb1*D);
                            HIP_CHECK(hipMemcpy(inter_T.data(), d_inter2, cb1*ca_skip*D*sizeof(float), hipMemcpyDeviceToHost));
                            // inter_T col-major [cb1, ca_skip*D]: element [ib, (is*D+d)] at addr ib + (is*D+d)*cb1
                            for(int ib=0;ib<cb1;ib++)
                            for(int is=0;is<ca_skip;is++)
                            for(int d=0;d<D;d++)
                                inter[(is*cb1+ib)*D+d] = inter_T[ib + (is*D+d)*cb1];

                            // Step2: nm[ca_skip, cb_skip] = sum_{cb1,D} inter * psi_b
                            // psi_b[cb_skip, cb1, D] row-major (skip_k=0 → psi_b[cb_skip, cb1, D])
                            // nm[is, ib_s] = sum_{ib1,d} inter[is,ib1,d] * psi_b[ib_s,ib1,d]
                            // = inter_view[ca_skip, cb1*D] @ psi_b_view[cb_skip, cb1*D]^T
                            std::vector<float> psi_b_cpu(sm.volb*D);
                            HIP_CHECK(hipMemcpy(psi_b_cpu.data(), psi_b, sm.volb*D*sizeof(float), hipMemcpyDeviceToHost));

                            std::vector<float> nm(ca_skip*cb_skip, 0.f);
                            // inter[ca_skip, cb1*D] @ psi_b[cb_skip, cb1*D]^T = nm[ca_skip, cb_skip]
                            // rocBLAS: nm^T[cb_skip, ca_skip] = psi_b[cb_skip, cb1*D] @ inter[ca_skip, cb1*D]^T
                            HIP_CHECK(hipMemcpy(d_inter, inter.data(), ca_skip*cb1*D*sizeof(float), hipMemcpyHostToDevice));
                            HIP_CHECK(hipMemcpy(d_inter2, psi_b_cpu.data(), sm.volb*D*sizeof(float), hipMemcpyHostToDevice));

                            // sgemm: nm^T[cb_skip, ca_skip] = psi_b_view[cb_skip, cb1*D] @ inter_view^T[cb1*D, ca_skip]
                            // col-major: nm^T is [cb_skip, ca_skip], psi_b_view is [cb1*D, cb_skip] col-major
                            rocblas_sgemm(handle,
                                rocblas_operation_none, rocblas_operation_transpose,
                                cb_skip, ca_skip, cb1*D,
                                &one, d_inter2, cb_skip, d_inter, ca_skip,
                                &zero, out_M, cb_skip);
                            // out_M = nm^T[cb_skip, ca_skip] — need to transpose to nm[ca_skip, cb_skip]
                            // For now: download, transpose, upload
                            std::vector<float> nm_T(ca_skip*cb_skip);
                            HIP_CHECK(hipMemcpy(nm_T.data(), out_M, ca_skip*cb_skip*sizeof(float), hipMemcpyDeviceToHost));
                            // nm_T col-major [cb_skip, ca_skip]: element [ib_s, is] at addr ib_s + is*cb_skip
                            for(int is=0;is<ca_skip;is++)
                            for(int ib_s=0;ib_s<cb_skip;ib_s++)
                                nm[is*cb_skip+ib_s] = nm_T[ib_s + is*cb_skip];

                            // Normalize
                            float s=0.f; for(auto v:nm) s=std::max(s,std::abs(v));
                            if(s>1e-20f) for(auto& v:nm) v/=s;
                            // Delta
                            std::vector<float> old_m(ca_skip*cb_skip);
                            HIP_CHECK(hipMemcpy(old_m.data(), old_M, ca_skip*cb_skip*sizeof(float), hipMemcpyDeviceToHost));
                            for(int i=0;i<ca_skip*cb_skip;i++) max_delta=std::max(max_delta,std::abs(nm[i]-old_m[i]));
                            HIP_CHECK(hipMemcpy(out_M, nm.data(), ca_skip*cb_skip*sizeof(float), hipMemcpyHostToDevice));

                        } else {
                            // skip_k=1: psi_a[ca_k1, ca_skip, D]
                            // Delegate to CPU for now (TODO: GPU path)
                            _cpu_msg_update_deg2_skip1(src, skip_k, k1, ca_skip, cb_skip, ca1, cb1,
                                psi_a, psi_b, get_M(k1), old_M, out_M, max_delta);
                        }
                    } else {
                        // Degree-3: delegate to CPU (TODO: GPU path)
                        _cpu_msg_update_deg3(src, skip_k, psi_a, psi_b, get_M, old_M, out_M, max_delta);
                    }
                }
            }

            // Swap ping-pong
            for (int bid=0; bid<n_bonds; bid++) {
                std::swap(d_M_fwd[bid], d_M_fwd2[bid]);
                std::swap(d_M_bwd[bid], d_M_bwd2[bid]);
            }

            if (verbose) printf("  [BLF_GPU2] sweep %d max_delta=%.2e\n", sweep, max_delta);
            if (max_delta < tol) { if (verbose) printf("  [BLF_GPU2] converged sweep %d\n", sweep); break; }
        }

        // Bethe PF on CPU (68 sites — fast)
        float result = _bethe_cpu();

        if (t_ms_out) {
            auto t1 = std::chrono::high_resolution_clock::now();
            *t_ms_out = std::chrono::duration<double,std::milli>(t1-t0).count();
        }
        return result;
    }

    // Damped BP: after each sweep, blend new messages with old:
    //   M_e ← alpha * M_new + (1-alpha) * M_old
    // This stabilizes loop correction sweeps starting from identity warm-start.
    // Implemented as a wrapper: run compute() for one sweep at a time, then
    // apply the damping blend before the next sweep.
    float compute_damped(int n_sweeps=10, float tol=1e-5f, float alpha=0.5f,
                         bool verbose=false, double* t_ms_out=nullptr) {
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int sweep=0; sweep<n_sweeps; sweep++) {
            // Save current messages (before update)
            std::vector<std::vector<float>> old_fwd(n_bonds), old_bwd(n_bonds);
            for (auto& bond : grid_a.bonds) {
                int bid=bond.id;
                int ca=frozen_chi_a[bid], cb=frozen_chi_b[bid];
                old_fwd[bid].resize(ca*cb);
                old_bwd[bid].resize(ca*cb);
                HIP_CHECK(hipMemcpy(old_fwd[bid].data(), d_M_fwd[bid],
                    ca*cb*sizeof(float), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(old_bwd[bid].data(), d_M_bwd[bid],
                    ca*cb*sizeof(float), hipMemcpyDeviceToHost));
            }

            // One undamped sweep (reuses existing compute() infrastructure)
            // Run exactly 1 sweep with high tol so it never exits early
            float max_delta_sweep = 0.f;
            // Call existing sweep logic inline via compute(1, 1e30f) trick
            // Actually: just do one pass of the message loop manually
            const float one=1.f, zero=0.f;
            for (auto& bond : grid_a.bonds) {
                for (int dir=0; dir<2; dir++) {
                    int src    = (dir==0) ? bond.site_a : bond.site_b;
                    auto& sm   = smeta[src];
                    int nb     = sm.nb;
                    int skip_k = -1;
                    for (int k=0;k<nb;k++) if(sm.bond_id[k]==bond.id){skip_k=k;break;}
                    if (skip_k<0) continue;
                    int ca_skip=sm.chi_a[skip_k], cb_skip=sm.chi_b[skip_k];
                    float*& out_M=(src==bond.site_a)?d_M_fwd2[bond.id]:d_M_bwd2[bond.id];
                    float*& old_M=(src==bond.site_a)?d_M_fwd[bond.id] :d_M_bwd[bond.id];
                    float* psi_a=d_psi_a[src]; float* psi_b=d_psi_b[src];
                    auto get_M=[&](int k)->float*{
                        int bk=sm.bond_id[k];
                        return (src==grid_a.bonds[bk].site_a)?d_M_bwd[bk]:d_M_fwd[bk];
                    };
                    if(nb==1){
                        rocblas_sgemm(handle,rocblas_operation_none,rocblas_operation_transpose,
                            cb_skip,ca_skip,D,&one,psi_b,cb_skip,psi_a,ca_skip,&zero,out_M,cb_skip);
                    } else if(nb==2){
                        int k1=(skip_k==0)?1:0;
                        int ca1=sm.chi_a[k1],cb1=sm.chi_b[k1];
                        if(skip_k==0)
                            _cpu_msg_update_deg2_skip1(src,1,0,ca_skip,cb_skip,ca1,cb1,
                                psi_a,psi_b,get_M(k1),old_M,out_M,max_delta_sweep);
                        else
                            _cpu_msg_update_deg2_skip1(src,skip_k,k1,ca_skip,cb_skip,ca1,cb1,
                                psi_a,psi_b,get_M(k1),old_M,out_M,max_delta_sweep);
                    } else {
                        _cpu_msg_update_deg3(src,skip_k,psi_a,psi_b,get_M,old_M,out_M,max_delta_sweep);
                    }
                }
            }
            for(int bid=0;bid<n_bonds;bid++){
                std::swap(d_M_fwd[bid],d_M_fwd2[bid]);
                std::swap(d_M_bwd[bid],d_M_bwd2[bid]);
            }

            // Apply damping: M ← alpha*M_new + (1-alpha)*M_old
            float max_delta=0.f;
            for (auto& bond : grid_a.bonds) {
                int bid=bond.id;
                int ca=frozen_chi_a[bid], cb=frozen_chi_b[bid];
                int n=ca*cb;
                std::vector<float> nf(n), nb_v(n);
                HIP_CHECK(hipMemcpy(nf.data(), d_M_fwd[bid], n*sizeof(float), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(nb_v.data(), d_M_bwd[bid], n*sizeof(float), hipMemcpyDeviceToHost));
                for(int i=0;i<n;i++){
                    float df=alpha*nf[i]+(1.f-alpha)*old_fwd[bid][i];
                    float db=alpha*nb_v[i]+(1.f-alpha)*old_bwd[bid][i];
                    max_delta=std::max(max_delta,std::abs(df-old_fwd[bid][i]));
                    max_delta=std::max(max_delta,std::abs(db-old_bwd[bid][i]));
                    nf[i]=df; nb_v[i]=db;
                }
                // normalize
                float sf=0.f,sb=0.f;
                for(auto v:nf)sf=std::max(sf,std::abs(v));
                for(auto v:nb_v)sb=std::max(sb,std::abs(v));
                if(sf>1e-20f)for(auto&v:nf)v/=sf;
                if(sb>1e-20f)for(auto&v:nb_v)v/=sb;
                HIP_CHECK(hipMemcpy(d_M_fwd[bid],nf.data(),n*sizeof(float),hipMemcpyHostToDevice));
                HIP_CHECK(hipMemcpy(d_M_bwd[bid],nb_v.data(),n*sizeof(float),hipMemcpyHostToDevice));
            }

            if(verbose) printf("  [BLF_GPU2 damped a=%.2f] sweep %d max_delta=%.2e\n",alpha,sweep,max_delta);
            if(max_delta < tol){ if(verbose) printf("  converged sweep %d\n",sweep); break; }
        }

        float result = _bethe_cpu();
        if(t_ms_out){
            auto t1=std::chrono::high_resolution_clock::now();
            *t_ms_out=std::chrono::duration<double,std::milli>(t1-t0).count();
        }
        return result;
    }

private:
    void _build_smeta() {
        for (auto& site : grid_a.sites) {
            int sid = site.id; auto& sm = smeta[sid];
            sm.nb=0; sm.vola=1; sm.volb=1;
            for (int d=0; d<MAX_DEGREE; d++) {
                int bid = site.bond_id[d]; if(bid<0) continue;
                int k = sm.nb++;
                sm.chi_a[k] = frozen_chi_a[bid];
                sm.chi_b[k] = frozen_chi_b[bid];
                sm.vola *= sm.chi_a[k];
                sm.volb *= sm.chi_b[k];
                sm.bond_id[k] = bid;
            }
        }
    }

    // CPU fallback for deg-2 skip_k=1 (psi_a[ca_k1, ca_skip, D])
    void _cpu_msg_update_deg2_skip1(
        int src, int skip_k, int k1,
        int ca_skip, int cb_skip, int ca1, int cb1,
        float* psi_a_dev, float* psi_b_dev,
        float* M1_dev, float* old_M_dev, float*& out_M_dev,
        float& max_delta)
    {
        auto& sm = smeta[src];
        int vola = sm.vola, volb = sm.volb;
        std::vector<float> pa(vola*D), pb(volb*D), M1(ca1*cb1);
        HIP_CHECK(hipMemcpy(pa.data(), psi_a_dev, vola*D*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(pb.data(), psi_b_dev, volb*D*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(M1.data(), M1_dev, ca1*cb1*sizeof(float), hipMemcpyDeviceToHost));

        // inter[ca_skip, cb1, D] = sum_{ca1} psi_a[ca1, ca_skip, D] * M1[ca1, cb1]
        std::vector<float> inter(ca_skip*cb1*D, 0.f);
        for(int i1=0;i1<ca1;i1++)
        for(int is=0;is<ca_skip;is++)
        for(int jb1=0;jb1<cb1;jb1++)
        for(int d=0;d<D;d++)
            inter[(is*cb1+jb1)*D+d] += pa[(i1*ca_skip+is)*D+d]*M1[i1*cb1+jb1];

        // nm[ca_skip, cb_skip] = sum_{cb1,D} inter * psi_b[cb1, cb_skip, D]
        std::vector<float> nm(ca_skip*cb_skip, 0.f);
        for(int is=0;is<ca_skip;is++)
        for(int ib_s=0;ib_s<cb_skip;ib_s++)
        for(int jb1=0;jb1<cb1;jb1++)
        for(int d=0;d<D;d++)
            nm[is*cb_skip+ib_s] += inter[(is*cb1+jb1)*D+d]*pb[(jb1*cb_skip+ib_s)*D+d];

        float s=0.f; for(auto v:nm) s=std::max(s,std::abs(v));
        if(s>1e-20f) for(auto& v:nm) v/=s;

        std::vector<float> old_m(ca_skip*cb_skip);
        HIP_CHECK(hipMemcpy(old_m.data(), old_M_dev, ca_skip*cb_skip*sizeof(float), hipMemcpyDeviceToHost));
        for(int i=0;i<ca_skip*cb_skip;i++) max_delta=std::max(max_delta,std::abs(nm[i]-old_m[i]));
        HIP_CHECK(hipMemcpy(out_M_dev, nm.data(), ca_skip*cb_skip*sizeof(float), hipMemcpyHostToDevice));
    }

    // CPU fallback for degree-3
    void _cpu_msg_update_deg3(
        int src, int skip_k,
        float* psi_a_dev, float* psi_b_dev,
        std::function<float*(int)> get_M,
        float* old_M_dev, float*& out_M_dev,
        float& max_delta)
    {
        auto& sm = smeta[src];
        int nb = sm.nb;
        int ca_skip = sm.chi_a[skip_k], cb_skip = sm.chi_b[skip_k];
        int k1=-1, k2=-1;
        for(int k=0;k<nb;k++) if(k!=skip_k){if(k1<0)k1=k;else k2=k;}
        int ca1=sm.chi_a[k1],cb1=sm.chi_b[k1],ca2=sm.chi_a[k2],cb2=sm.chi_b[k2];

        std::vector<float> pa(sm.vola*D), pb(sm.volb*D);
        std::vector<float> M1(ca1*cb1), M2(ca2*cb2);
        HIP_CHECK(hipMemcpy(pa.data(), psi_a_dev, sm.vola*D*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(pb.data(), psi_b_dev, sm.volb*D*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(M1.data(), get_M(k1), ca1*cb1*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(M2.data(), get_M(k2), ca2*cb2*sizeof(float), hipMemcpyDeviceToHost));

        // Step 1: inter[ca_skip, ca1, cb2, D]
        std::vector<float> inter(ca_skip*ca1*cb2*D,0.f);
        for(int is=0;is<ca_skip;is++)
        for(int i1=0;i1<ca1;i1++)
        for(int i2=0;i2<ca2;i2++){
            int idx[3]; idx[skip_k]=is;idx[k1]=i1;idx[k2]=i2;
            int fa=(idx[0]*sm.chi_a[1]+idx[1])*sm.chi_a[2]+idx[2];
            for(int jb2=0;jb2<cb2;jb2++)
            for(int d=0;d<D;d++)
                inter[((is*ca1+i1)*cb2+jb2)*D+d]+=pa[fa*D+d]*M2[i2*cb2+jb2];
        }
        // Step 2: inter2[ca_skip, cb1, cb2, D]
        std::vector<float> inter2(ca_skip*cb1*cb2*D,0.f);
        for(int is=0;is<ca_skip;is++)
        for(int i1=0;i1<ca1;i1++)
        for(int jb2=0;jb2<cb2;jb2++)
        for(int jb1=0;jb1<cb1;jb1++)
        for(int d=0;d<D;d++)
            inter2[((is*cb1+jb1)*cb2+jb2)*D+d]+=inter[((is*ca1+i1)*cb2+jb2)*D+d]*M1[i1*cb1+jb1];
        // Step 3: nm[ca_skip, cb_skip]
        std::vector<float> nm(ca_skip*cb_skip,0.f);
        for(int is=0;is<ca_skip;is++)
        for(int ib_s=0;ib_s<cb_skip;ib_s++)
        for(int jb1=0;jb1<cb1;jb1++)
        for(int jb2=0;jb2<cb2;jb2++){
            int jdx[3];jdx[skip_k]=ib_s;jdx[k1]=jb1;jdx[k2]=jb2;
            int fb=(jdx[0]*sm.chi_b[1]+jdx[1])*sm.chi_b[2]+jdx[2];
            for(int d=0;d<D;d++)
                nm[is*cb_skip+ib_s]+=inter2[((is*cb1+jb1)*cb2+jb2)*D+d]*pb[fb*D+d];
        }
        float s=0.f;for(auto v:nm)s=std::max(s,std::abs(v));
        if(s>1e-20f)for(auto& v:nm)v/=s;
        std::vector<float> old_m(ca_skip*cb_skip);
        HIP_CHECK(hipMemcpy(old_m.data(),old_M_dev,ca_skip*cb_skip*sizeof(float),hipMemcpyDeviceToHost));
        for(int i=0;i<ca_skip*cb_skip;i++)max_delta=std::max(max_delta,std::abs(nm[i]-old_m[i]));
        HIP_CHECK(hipMemcpy(out_M_dev,nm.data(),ca_skip*cb_skip*sizeof(float),hipMemcpyHostToDevice));
    }

    float _bethe_cpu() {
        double log_num=0.0,sign_num=1.0;
        for(auto& site:grid_a.sites){
            int sid=site.id; auto& sm=smeta[sid];
            int nb=sm.nb;
            std::vector<float> pa(sm.vola*D),pb(sm.volb*D);
            HIP_CHECK(hipMemcpy(pa.data(),d_psi_a[sid],sm.vola*D*sizeof(float),hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(pb.data(),d_psi_b[sid],sm.volb*D*sizeof(float),hipMemcpyDeviceToHost));

            auto get_M_bethe=[&](int k)->std::vector<float> {
                int bk=sm.bond_id[k];
                int ca=frozen_chi_a[bk],cb=frozen_chi_b[bk];
                float* ptr=(sid==grid_a.bonds[bk].site_a)?d_M_bwd[bk]:d_M_fwd[bk];
                std::vector<float> m(ca*cb);
                HIP_CHECK(hipMemcpy(m.data(),ptr,ca*cb*sizeof(float),hipMemcpyDeviceToHost));
                return m;
            };
            std::vector<std::vector<float>> Ms(nb);
            for(int k=0;k<nb;k++) Ms[k]=get_M_bethe(k);

            double a_v=0.0;
            for(int ia=0;ia<sm.vola;ia++){
                int tmp=ia; int ia_k[MAX_DEGREE];
                for(int k=nb-1;k>=0;k--){ia_k[k]=tmp%sm.chi_a[k];tmp/=sm.chi_a[k];}
                for(int ib=0;ib<sm.volb;ib++){
                    int tmp2=ib; int ib_k[MAX_DEGREE];
                    for(int k=nb-1;k>=0;k--){ib_k[k]=tmp2%sm.chi_b[k];tmp2/=sm.chi_b[k];}
                    float mp=1.f;
                    for(int k=0;k<nb;k++) mp*=Ms[k][ia_k[k]*sm.chi_b[k]+ib_k[k]];
                    for(int d=0;d<D;d++) a_v+=(double)pa[ia*D+d]*(double)pb[ib*D+d]*mp;
                }
            }
            sign_num*=(a_v>0?1.0:(a_v<0?-1.0:0.0));
            log_num+=(a_v!=0?std::log(std::abs(a_v)):-1e30);
        }
        double log_den=0.0,sign_den=1.0;
        for(auto& bond:grid_a.bonds){
            int bid=bond.id;
            int ca=frozen_chi_a[bid],cb=frozen_chi_b[bid];
            std::vector<float> mf(ca*cb),mb(ca*cb);
            HIP_CHECK(hipMemcpy(mf.data(),d_M_fwd[bid],ca*cb*sizeof(float),hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(mb.data(),d_M_bwd[bid],ca*cb*sizeof(float),hipMemcpyDeviceToHost));
            double ze=0.0;
            for(int i=0;i<ca*cb;i++) ze+=(double)mf[i]*(double)mb[i];
            sign_den*=(ze>0?1.0:(ze<0?-1.0:0.0));
            log_den+=(std::abs(ze)>1e-30?std::log(std::abs(ze)):-1e30);
        }
        if(sign_num==0.0) return 0.f;
        return (float)(sign_num*sign_den*std::exp(log_num-log_den+log_scale_a+log_scale_b));
    }

    void _free() {
        for(auto p:d_psi_a)  if(p) HIP_CHECK(hipFree(p));
        for(auto p:d_psi_b)  if(p) HIP_CHECK(hipFree(p));
        for(auto p:d_M_fwd)  if(p) HIP_CHECK(hipFree(p));
        for(auto p:d_M_bwd)  if(p) HIP_CHECK(hipFree(p));
        for(auto p:d_M_fwd2) if(p) HIP_CHECK(hipFree(p));
        for(auto p:d_M_bwd2) if(p) HIP_CHECK(hipFree(p));
        if(d_inter)  HIP_CHECK(hipFree(d_inter));
        if(d_inter2) HIP_CHECK(hipFree(d_inter2));
    }
};
