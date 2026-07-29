#pragma once
/*
 * bilinear_form_bp_gpu3.hpp — GPU inner product via rocBLAS SGEMM.
 *
 * Direct GPU port of bilinear_form_bp_cpu2.hpp.
 * All SGEMM operations validated in test_blf_sgemm.cpp.
 *
 * Per message update (degree-2, skip_k=0):
 *   Step 1: inter[ca_skip, cb1, D] = psi_a[ca_skip, ca1, D] @ M[ca1, cb1]
 *           → rocblas_sgemm_strided_batched(batch=D)
 *   Step 2: nm[ca_skip, cb_skip] = inter @ psi_b^T
 *           → D sequential rocblas_sgemm calls with accumulation
 *
 * Degree-3: two sequential step-1 SGEMMs then one step-2.
 * Bethe PF: CPU (68 sites, fast).
 * Messages: full chi_a×chi_b matrices, ping-pong on GPU.
 */

#include "grid.hpp"
#include "types.hpp"
#include "bilinear_form_bp.hpp"   // for StateSnapshot
#include <vector>
#include <cmath>
#include <chrono>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do { \
    hipError_t e=(cmd); \
    if(e!=hipSuccess){fprintf(stderr,"HIP %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__);exit(1);} \
} while(0)
#endif

static constexpr int D_BLF_G3 = 4;  // Pauli basis dimension

// ── Normalize kernel ─────────────────────────────────────────────────────────
__global__ void normalize_msg_gpu3(float* m, int n) {
    if (threadIdx.x != 0) return;
    float s = 0.f;
    for (int i=0; i<n; i++) s = fmaxf(s, fabsf(m[i]));
    if (s > 1e-20f) for (int i=0; i<n; i++) m[i] /= s;
}

// ── Transpose kernel: psi[n1, n2, D] → out[D, n1, n2] ───────────────────────
// Launched as grid=(n1*n2*D / 256 + 1), block=(256)
__global__ void transpose_to_batch_kernel(
    const float* __restrict__ psi, float* __restrict__ out,
    int n1, int n2, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    // idx = i1*n2*D + i2*D + d
    int d  = idx % D_BLF_G3;
    int i2 = (idx / D_BLF_G3) % n2;
    int i1 = idx / (D_BLF_G3 * n2);
    // out[d, i1, i2] = out[d*n1*n2 + i1*n2 + i2]
    out[d*n1*n2 + i1*n2 + i2] = psi[idx];
}

// ── Rearrange kernel: psi[a,b,c,D] → out[skip,k1,k2,D] permuting bond axes ──
// Handles deg-3 reordering entirely on GPU.
// All three bond orders: skip_k ∈ {0,1,2}
__global__ void rearrange_deg3_kernel(
    const float* __restrict__ psi, float* __restrict__ out,
    int c0, int c1, int c2,          // original chi_a[0..2]
    int skip_k, int k1_idx, int k2_idx,  // target order: skip, k1, k2
    int ca_skip, int ca1, int ca2, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d    = idx % D_BLF_G3;
    int rem  = idx / D_BLF_G3;
    int i2   = rem % ca2; rem /= ca2;
    int i1   = rem % ca1; rem /= ca1;
    int i_sk = rem;
    // Original indices in bond order 0,1,2
    int orig[3]; orig[skip_k]=i_sk; orig[k1_idx]=i1; orig[k2_idx]=i2;
    int src_flat = (orig[0]*c1 + orig[1])*c2 + orig[2];
    out[idx] = psi[src_flat*D_BLF_G3 + d];
}

// ── BLF_GPU3 ─────────────────────────────────────────────────────────────────

struct BLF_GPU3 {
    rocblas_handle handle;
    int n_sites, n_bonds;

    // Per-site metadata (frozen at upload time)
    struct SiteMeta {
        int nb;
        int chi_a[MAX_DEGREE], chi_b[MAX_DEGREE];
        int vola, volb;
        int bond_id[MAX_DEGREE];
    };
    std::vector<SiteMeta> smeta;
    std::vector<int> frozen_chi_a, frozen_chi_b;  // [n_bonds]

    // Device psi pointers (one per site, normalized)
    std::vector<float*> d_psi_a, d_psi_b;

    // Device messages: M_fwd[bid] = site_a→site_b, M_bwd[bid] = site_b→site_a
    // Each chi_a×chi_b floats. Ping-pong for race-free updates.
    std::vector<float*> d_M_fwd,  d_M_bwd;   // current (ping)
    std::vector<float*> d_M_fwd2, d_M_bwd2;  // scratch  (pong)

    // Scratch buffer for intermediate contraction
    float* d_scratch  = nullptr;
    float* d_scratch2 = nullptr;
    int scratch_elems = 0;

    // Log scale from psi normalization
    double log_scale_a = 0.0, log_scale_b = 0.0;

    // Grid references (for Bethe PF topology)
    const Grid* grid_a_ptr = nullptr;
    const Grid* grid_b_ptr = nullptr;

    BLF_GPU3(rocblas_handle h) : handle(h) {}
    ~BLF_GPU3() { _free(); }

    // ── upload ────────────────────────────────────────────────────────────────
    void upload(
        const Grid& ga, const std::vector<float*>& psi_a_src, int dev_a,
        const Grid& gb, const std::vector<float*>& psi_b_src, int dev_b)
    {
        grid_a_ptr = &ga; grid_b_ptr = &gb;
        n_sites = ga.n_sites; n_bonds = ga.n_bonds;

        // Freeze chi
        frozen_chi_a.assign(n_bonds, 1); frozen_chi_b.assign(n_bonds, 1);
        for (auto& bond : ga.bonds) frozen_chi_a[bond.id] = bond.chi;
        for (auto& bond : gb.bonds) frozen_chi_b[bond.id] = bond.chi;

        // Build site metadata
        smeta.resize(n_sites);
        for (auto& site : ga.sites) {
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

        int cur; HIP_CHECK(hipGetDevice(&cur));

        // Free old psi, alloc new
        for (auto p : d_psi_a) if(p) HIP_CHECK(hipFree(p));
        for (auto p : d_psi_b) if(p) HIP_CHECK(hipFree(p));
        d_psi_a.assign(n_sites, nullptr);
        d_psi_b.assign(n_sites, nullptr);
        log_scale_a = log_scale_b = 0.0;

        for (auto& site : ga.sites) {
            int sid=site.id; int vol=smeta[sid].vola*D;
            std::vector<float> buf(vol);
            HIP_CHECK(hipSetDevice(dev_a));
            HIP_CHECK(hipMemcpy(buf.data(), psi_a_src[sid], vol*sizeof(float), hipMemcpyDeviceToHost));
            float s=0.f; for(auto v:buf) s=std::max(s,std::abs(v));
            if(s>1e-30f){for(auto& v:buf)v/=s; log_scale_a+=std::log((double)s);}
            HIP_CHECK(hipSetDevice(cur));
            HIP_CHECK(hipMalloc(&d_psi_a[sid], vol*sizeof(float)));
            HIP_CHECK(hipMemcpy(d_psi_a[sid], buf.data(), vol*sizeof(float), hipMemcpyHostToDevice));
        }
        for (auto& site : gb.sites) {
            int sid=site.id; int vol=smeta[sid].volb*D;
            std::vector<float> buf(vol);
            HIP_CHECK(hipSetDevice(dev_b));
            HIP_CHECK(hipMemcpy(buf.data(), psi_b_src[sid], vol*sizeof(float), hipMemcpyDeviceToHost));
            float s=0.f; for(auto v:buf) s=std::max(s,std::abs(v));
            if(s>1e-30f){for(auto& v:buf)v/=s; log_scale_b+=std::log((double)s);}
            HIP_CHECK(hipSetDevice(cur));
            HIP_CHECK(hipMalloc(&d_psi_b[sid], vol*sizeof(float)));
            HIP_CHECK(hipMemcpy(d_psi_b[sid], buf.data(), vol*sizeof(float), hipMemcpyHostToDevice));
        }
        HIP_CHECK(hipSetDevice(cur));

        // Free old messages, alloc new
        for (auto p : d_M_fwd)  if(p) HIP_CHECK(hipFree(p));
        for (auto p : d_M_bwd)  if(p) HIP_CHECK(hipFree(p));
        for (auto p : d_M_fwd2) if(p) HIP_CHECK(hipFree(p));
        for (auto p : d_M_bwd2) if(p) HIP_CHECK(hipFree(p));
        d_M_fwd.assign(n_bonds,nullptr); d_M_bwd.assign(n_bonds,nullptr);
        d_M_fwd2.assign(n_bonds,nullptr); d_M_bwd2.assign(n_bonds,nullptr);

        for (auto& bond : ga.bonds) {
            int bid=bond.id; int ca=frozen_chi_a[bid], cb=frozen_chi_b[bid];
            int mn=std::min(ca,cb);
            std::vector<float> mf(ca*cb,0.f), mb(ca*cb,0.f);
            for(int i=0;i<mn;i++){mf[i*cb+i]=1.f/mn; mb[i*cb+i]=1.f/mn;}
            for(int k=0;k<4;k++){
                float** ptr = k==0?&d_M_fwd[bid]:k==1?&d_M_bwd[bid]:k==2?&d_M_fwd2[bid]:&d_M_bwd2[bid];
                HIP_CHECK(hipMalloc(ptr, ca*cb*sizeof(float)));
                HIP_CHECK(hipMemcpy(*ptr, (k<2?&mf:&mb)[0].data()? (k%2==0?mf.data():mb.data()) : mf.data(),
                    ca*cb*sizeof(float), hipMemcpyHostToDevice));
            }
            // Simpler init:
            HIP_CHECK(hipMemcpy(d_M_fwd[bid],  mf.data(), ca*cb*sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_M_bwd[bid],  mb.data(), ca*cb*sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_M_fwd2[bid], mf.data(), ca*cb*sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_M_bwd2[bid], mb.data(), ca*cb*sizeof(float), hipMemcpyHostToDevice));
        }

        // Allocate transposed psi and inter buffers (GPU, no D2H in compute)
        _alloc_psi_T_buffers();

        // Allocate scratch: max inter size = max(ca_skip * cb1 * D) across all bond updates
        int max_sz = 0;
        for (auto& site : ga.sites) {
            auto& sm = smeta[site.id];
            for (int sk=0; sk<sm.nb; sk++) {
                int ca_skip = sm.chi_a[sk];
                for (int k=0; k<sm.nb; k++) if(k!=sk) {
                    int sz = ca_skip * sm.chi_b[k] * D;
                    max_sz = std::max(max_sz, sz);
                    // degree-3: second intermediate also needs ca_skip * cb1 * cb2 * D
                    if (sm.nb==3) {
                        int k2=-1; for(int kk=0;kk<sm.nb;kk++) if(kk!=sk&&kk!=k){k2=kk;break;}
                        if(k2>=0) max_sz = std::max(max_sz, ca_skip*sm.chi_b[k]*sm.chi_b[k2]*D);
                    }
                }
            }
        }
        if(d_scratch)  HIP_CHECK(hipFree(d_scratch));
        if(d_scratch2) HIP_CHECK(hipFree(d_scratch2));
        scratch_elems = max_sz;
        HIP_CHECK(hipMalloc(&d_scratch,  max_sz*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_scratch2, max_sz*sizeof(float)));
    }

    // ── compute ───────────────────────────────────────────────────────────────
    float compute(int n_sweeps=25, float tol=1e-5f, bool verbose=false,
                  double* t_ms_out=nullptr)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        const float one=1.f, zero=0.f;

        for (int sweep=0; sweep<n_sweeps; sweep++) {
            float max_delta = 0.f;

            for (int bid=0; bid<n_bonds; bid++) {
                const Bond& bond = grid_a_ptr->bonds[bid];
                for (int dir=0; dir<2; dir++) {
                    int src     = (dir==0) ? bond.site_a : bond.site_b;
                    auto& sm    = smeta[src];
                    int nb      = sm.nb;
                    int skip_k  = -1;
                    for(int k=0;k<nb;k++) if(sm.bond_id[k]==bid){skip_k=k;break;}
                    if(skip_k<0) continue;

                    int ca_skip = sm.chi_a[skip_k];
                    int cb_skip = sm.chi_b[skip_k];

                    float* psi_a = d_psi_a[src];
                    float* psi_b = d_psi_b[src];

                    // Incoming message for non-skip bond k: if src=site_a → M_bwd, else M_fwd
                    auto get_M = [&](int k) -> float* {
                        int bk = sm.bond_id[k];
                        return (src==grid_a_ptr->bonds[bk].site_a) ? d_M_bwd[bk] : d_M_fwd[bk];
                    };
                    // Output message (pong buffer)
                    float*& out_M = (src==bond.site_a) ? d_M_fwd2[bid] : d_M_bwd2[bid];
                    float*& old_M = (src==bond.site_a) ? d_M_fwd[bid]  : d_M_bwd[bid];

                    if (nb == 1) {
                        // Degree-1: nm = psi_a @ psi_b^T  (over D)
                        // sgemm: nm_col[cb_skip,ca_skip] = psi_b_col[D,cb_skip]^T @ psi_a_col[D,ca_skip]^T^T...
                        // Use: nm_col[cb_skip,ca_skip] = psi_b[cb_skip,D] @ psi_a[ca_skip,D]^T
                        // row-major A[cb_skip,D] @ B[ca_skip,D]^T = C[cb_skip,ca_skip]
                        // col-major: C_col[ca_skip,cb_skip] = B_col^T @ A_col^T... use op tricks
                        // Simple: sgemm(op_T,op_none, ca_skip,cb_skip,D, psi_a,D, psi_b,D, out_M,ca_skip)
                        // This gives: out_M col[ca_skip,cb_skip] = psi_a^T @ psi_b  where col-major
                        // psi_a col[D,ca_skip] lda=D, op_T → [ca_skip,D]
                        // psi_b col[D,cb_skip] lda=D, op_N → [D,cb_skip]
                        // Result: [ca_skip,cb_skip] col-major = nm_row[cb_skip,ca_skip]^T...
                        // Just use the same pattern as step2:
                        for(int d=0;d<D;d++){
                            rocblas_sgemm(handle,
                                rocblas_operation_transpose, rocblas_operation_none,
                                cb_skip, ca_skip, 1,
                                &one,
                                psi_b+d, D,    // psi_b[cb_skip,1] batch at offset d, stride D
                                psi_a+d, D,
                                d==0?&zero:&one,
                                out_M, cb_skip);
                        }
                        // out_M col[cb_skip,ca_skip] = nm_row[ca_skip,cb_skip] ✓

                    } else if (nb == 2) {
                        int k1 = (skip_k==0) ? 1 : 0;
                        int ca1=sm.chi_a[k1], cb1=sm.chi_b[k1];
                        float* M1 = get_M(k1);

                        if (skip_k==0) {
                            // psi_a[ca_skip, ca1, D] — skip_k outermost, correct layout
                            _step1_gpu(psi_a, ca_skip, ca1, M1, cb1);
                            // psi_b[cb_skip, cb1, D] — skip_k outermost
                            _step2_gpu(psi_b, ca_skip, cb1, cb_skip, out_M);
                        } else {
                            // skip_k=1: psi_a[ca1, ca_skip, D] — transpose to [ca_skip, ca1, D]
                            // Use GPU kernel: treat as [ca1, ca_skip, D] → transpose_to_batch
                            // gives [D, ca1, ca_skip] — not what we want.
                            // Simpler: D2H transpose on CPU (skip_k=1 is minority, optimize later)
                            auto cpu_transpose = [&](float* d_src, int n1, int n2, float* d_dst){
                                int sz=n1*n2*D_BLF_G3;
                                std::vector<float> h(sz),ht(sz);
                                HIP_CHECK(hipMemcpy(h.data(),d_src,sz*sizeof(float),hipMemcpyDeviceToHost));
                                for(int i=0;i<n1;i++)for(int j=0;j<n2;j++)for(int d=0;d<D_BLF_G3;d++)
                                    ht[(j*n1+i)*D_BLF_G3+d]=h[(i*n2+j)*D_BLF_G3+d];
                                HIP_CHECK(hipMemcpy(d_dst,ht.data(),sz*sizeof(float),hipMemcpyHostToDevice));
                            };
                            cpu_transpose(psi_a, ca1, ca_skip, d_scratch2); // [ca_skip,ca1,D]
                            _step1_gpu(d_scratch2, ca_skip, ca1, M1, cb1);
                            cpu_transpose(psi_b, cb1, cb_skip, d_scratch2); // [cb_skip,cb1,D]
                            _step2_gpu(d_scratch2, ca_skip, cb1, cb_skip, out_M);
                        }

                    } else {
                        // Degree-3: two step-1 then one step-2
                        int k1=-1, k2=-1;
                        for(int k=0;k<nb;k++) if(k!=skip_k){if(k1<0)k1=k;else k2=k;}
                        int ca1=sm.chi_a[k1],cb1=sm.chi_b[k1];
                        int ca2=sm.chi_a[k2],cb2=sm.chi_b[k2];
                        float* M1=get_M(k1); float* M2=get_M(k2);

                        // Degree-3: GPU SGEMM path
                        // Rearrange psi_a so skip_k is outermost index
                        _rearrange_deg3_gpu(psi_a, sm.chi_a, skip_k, k1, k2,
                                            ca_skip, ca1, ca2, d_scratch2);
                        // d_scratch2: psi_a_r[ca_skip, ca1, ca2, D]

                        // Step A: inter1[ca_skip*ca1, cb2, D] = psi_a_r[:,: ,ca2,:] @ M2[ca2,cb2]
                        // Treat psi_a_r as [ca_skip*ca1, ca2, D] → step1 with ca_skip*ca1 rows
                        _step1_gpu(d_scratch2, ca_skip*ca1, ca2, M2, cb2);
                        // d_inter_T: [D, ca_skip*ca1, cb2]

                        // Step B: inter2[ca_skip, cb1, cb2*D] = inter1[ca_skip,ca1,cb2*D] @ M1[ca1,cb1]
                        // Treat inter1 as [ca_skip, ca1, cb2*D]: need M1 contracted over ca1
                        // inter1 is stored as d_inter_T[D, ca_skip*ca1, cb2]
                        // Rearrange to [ca_skip, ca1, D*cb2] → use scratch
                        // Actually: for each (d,cb2 pair), inter1_d[ca_skip,ca1] @ M1 → inter2_d[ca_skip,cb1]
                        // This is strided_batched with batch = D*cb2
                        // inter1 in d_inter_T[D, ca_skip*ca1, cb2]:
                        //   batch db = d*cb2+jb2, A_db = d_inter_T[d*ca_skip*ca1*cb2 + ... ]
                        // This requires a rearrangement. Simpler: download inter_T, rearrange, re-upload
                        // then do strided_batched over D*cb2 batches.
                        {
                            int sz = D_BLF_G3*ca_skip*ca1*cb2;
                            std::vector<float> iT(sz), i2(ca_skip*ca1*cb2*D_BLF_G3);
                            HIP_CHECK(hipMemcpy(iT.data(), d_inter_T, sz*sizeof(float), hipMemcpyDeviceToHost));
                            // iT[D, ca_skip*ca1, cb2] → rearrange to [ca_skip, ca1, cb2, D]
                            for(int d=0;d<D_BLF_G3;d++)
                            for(int is=0;is<ca_skip;is++)
                            for(int i1=0;i1<ca1;i1++)
                            for(int jb2=0;jb2<cb2;jb2++)
                                i2[((is*ca1+i1)*cb2+jb2)*D_BLF_G3+d] = iT[d*ca_skip*ca1*cb2 + (is*ca1+i1)*cb2 + jb2];
                            HIP_CHECK(hipMemcpy(d_scratch2, i2.data(), sz*sizeof(float), hipMemcpyHostToDevice));
                        }
                        // d_scratch2: inter1_r[ca_skip, ca1, cb2*D] (treating cb2*D as one dim)
                        // Step B: inter2[ca_skip, cb1, cb2*D] = inter1_r[ca_skip, ca1, cb2*D] @_{ca1} M1
                        // Same as step1 but contracting ca1, with cb2*D "D-like" batches
                        // Use step1 treating cb2*D as the batch dimension:
                        // inter1_r[ca_skip, ca1, cb2*D] → _step1 with ca_skip rows, ca1 contract, cb1 output
                        // but "D" here is cb2*D which changes the kernel. Use strided_batched directly:
                        {
                            // inter1_r[ca_skip, ca1, cb2*D] — transpose to [cb2*D, ca_skip, ca1]
                            int sz=ca_skip*ca1*cb2*D_BLF_G3;
                            std::vector<float> i2(sz), i2T(sz);
                            HIP_CHECK(hipMemcpy(i2.data(), d_scratch2, sz*sizeof(float), hipMemcpyDeviceToHost));
                            // [ca_skip,ca1,cb2*D] → [cb2*D, ca_skip, ca1]
                            int BD = cb2*D_BLF_G3;
                            for(int is=0;is<ca_skip;is++)
                            for(int i1=0;i1<ca1;i1++)
                            for(int bd=0;bd<BD;bd++)
                                i2T[bd*ca_skip*ca1 + is*ca1 + i1] = i2[(is*ca1+i1)*BD+bd];
                            HIP_CHECK(hipMemcpy(d_scratch2, i2T.data(), sz*sizeof(float), hipMemcpyHostToDevice));
                        }
                        // strided_batched: inter2_T[cb2*D, ca_skip, cb1] = M1 @ i2T[cb2*D, ca_skip, ca1]
                        {
                            float alpha=1.f, beta=0.f;
                            int BD=cb2*D_BLF_G3;
                            rocblas_sgemm_strided_batched(handle,
                                rocblas_operation_none, rocblas_operation_none,
                                cb1, ca_skip, ca1,
                                &alpha,
                                M1,          cb1, 0,
                                d_scratch2,  ca1, ca_skip*ca1,
                                &beta,
                                d_inter_T,   cb1, ca_skip*cb1,
                                BD);
                        }
                        // d_inter_T[cb2*D, ca_skip, cb1] = inter2_T

                        // Step C: rearrange inter2_T and psi_b, then nm via step2
                        // inter2_T[cb2*D, ca_skip, cb1] → inter2[ca_skip, cb1*cb2*D] for step2
                        // psi_b_r[cb_skip, cb1, cb2, D] (rearranged) → psi_b_flat[cb_skip, cb1*cb2*D]
                        // nm[ca_skip, cb_skip] = inter2[ca_skip, cb1*cb2*D] @ psi_b_flat[cb_skip, cb1*cb2*D]^T
                        // This is one SGEMM (no batching needed since cb1*cb2*D is the K dimension)
                        {
                            // Rearrange inter2_T[cb2*D, ca_skip, cb1] → [ca_skip, cb1*cb2*D]
                            int sz_i=D_BLF_G3*ca_skip*cb1*cb2;
                            std::vector<float> iT2(sz_i), i2flat(ca_skip*cb1*cb2*D_BLF_G3);
                            HIP_CHECK(hipMemcpy(iT2.data(), d_inter_T, sz_i*sizeof(float), hipMemcpyDeviceToHost));
                            int BD=cb2*D_BLF_G3;
                            // iT2[cb2*D, ca_skip, cb1] → i2flat[ca_skip, cb1, cb2, D]
                            for(int bd=0;bd<BD;bd++)
                            for(int is=0;is<ca_skip;is++)
                            for(int jb1=0;jb1<cb1;jb1++){
                                int jb2=bd/D_BLF_G3, d=bd%D_BLF_G3;
                                i2flat[((is*cb1+jb1)*cb2+jb2)*D_BLF_G3+d] = iT2[bd*ca_skip*cb1 + is*cb1 + jb1];
                            }
                            HIP_CHECK(hipMemcpy(d_scratch2, i2flat.data(), sz_i*sizeof(float), hipMemcpyHostToDevice));
                            // d_scratch2: inter2[ca_skip, cb1*cb2*D] (flat K dimension)

                            // Rearrange psi_b so skip_k is outermost
                            int sz_pb=sm.volb*D_BLF_G3;
                            std::vector<float> pb_h(sz_pb), pb_r(sz_pb);
                            HIP_CHECK(hipMemcpy(pb_h.data(), psi_b, sz_pb*sizeof(float), hipMemcpyDeviceToHost));
                            // psi_b original order [bond0,bond1,bond2,D], rearrange to [skip,k1,k2,D]
                            for(int ib_s=0;ib_s<cb_skip;ib_s++)
                            for(int jb1=0;jb1<cb1;jb1++)
                            for(int jb2=0;jb2<cb2;jb2++)
                            for(int d=0;d<D_BLF_G3;d++){
                                int jdx[3]; jdx[skip_k]=ib_s; jdx[k1]=jb1; jdx[k2]=jb2;
                                int fb=(jdx[0]*sm.chi_b[1]+jdx[1])*sm.chi_b[2]+jdx[2];
                                pb_r[((ib_s*cb1+jb1)*cb2+jb2)*D_BLF_G3+d] = pb_h[fb*D_BLF_G3+d];
                            }
                            // pb_r[cb_skip, cb1*cb2*D] flat
                            float* d_pb_r; HIP_CHECK(hipMalloc(&d_pb_r, sz_pb*sizeof(float)));
                            HIP_CHECK(hipMemcpy(d_pb_r, pb_r.data(), sz_pb*sizeof(float), hipMemcpyHostToDevice));

                            // nm col[cb_skip,ca_skip] = pb_r[cb_skip,K]^T_col @ inter2[ca_skip,K]^T_col
                            // = sgemm(op_T,op_none, cb_skip,ca_skip, K=cb1*cb2*D,
                            //         pb_r,cb1*cb2*D, inter2_col,cb1*cb2*D, nm,cb_skip)
                            int K=cb1*cb2*D_BLF_G3;
                            float alpha=1.f, beta=0.f;
                            rocblas_sgemm(handle,
                                rocblas_operation_transpose, rocblas_operation_none,
                                cb_skip, ca_skip, K,
                                &alpha, d_pb_r, K, d_scratch2, K,
                                &beta, out_M, cb_skip);
                            HIP_CHECK(hipDeviceSynchronize());
                            HIP_CHECK(hipFree(d_pb_r));
                        }
                        // Normalize
                        hipLaunchKernelGGL(normalize_msg_gpu3, 1, 1, 0, 0,
                            out_M, ca_skip*cb_skip);
                        continue;
                    }

                    // Normalize (GPU kernel)
                    hipLaunchKernelGGL(normalize_msg_gpu3, 1, 1, 0, 0,
                        out_M, ca_skip*cb_skip);
                }
            }

            HIP_CHECK(hipDeviceSynchronize());

            // Swap ping-pong
            for(int bid=0;bid<n_bonds;bid++){
                std::swap(d_M_fwd[bid], d_M_fwd2[bid]);
                std::swap(d_M_bwd[bid], d_M_bwd2[bid]);
            }

            // Convergence check: download all messages once per sweep
            for(int bid=0;bid<n_bonds;bid++){
                int ca=frozen_chi_a[bid], cb=frozen_chi_b[bid], sz=ca*cb;
                std::vector<float> mf(sz), mb(sz), mf2(sz), mb2(sz);
                HIP_CHECK(hipMemcpy(mf.data(),  d_M_fwd[bid],  sz*sizeof(float), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(mb.data(),  d_M_bwd[bid],  sz*sizeof(float), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(mf2.data(), d_M_fwd2[bid], sz*sizeof(float), hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(mb2.data(), d_M_bwd2[bid], sz*sizeof(float), hipMemcpyDeviceToHost));
                for(int i=0;i<sz;i++){
                    max_delta=std::max(max_delta,std::abs(mf[i]-mf2[i]));
                    max_delta=std::max(max_delta,std::abs(mb[i]-mb2[i]));
                }
            }

            if(verbose) printf("  [BLF3] sweep %d max_delta=%.2e\n", sweep, max_delta);
            if(max_delta < tol){ if(verbose) printf("  [BLF3] converged sweep %d\n",sweep); break; }
        }

        float result = _bethe_cpu();

        if(t_ms_out){
            auto t1=std::chrono::high_resolution_clock::now();
            *t_ms_out=std::chrono::duration<double,std::milli>(t1-t0).count();
        }
        return result;
    }

private:
    // ── GPU scratch buffers for transposed psi (pre-allocated) ───────────────
    // Avoids malloc/free per message update.
    float* d_psi_T_a = nullptr;   // transpose buffer for psi_a
    float* d_psi_T_b = nullptr;   // transpose buffer for psi_b
    float* d_inter_T = nullptr;   // inter in [D, ca_skip, cb1] layout
    int    psi_T_elems = 0;
    int    inter_T_elems = 0;

    void _alloc_psi_T_buffers() {
        // Max psi size across all sites
        int max_psi = 0, max_inter = 0;
        for (int sid=0; sid<n_sites; sid++) {
            auto& sm = smeta[sid];
            max_psi = std::max(max_psi, sm.vola * D_BLF_G3);
            max_psi = std::max(max_psi, sm.volb * D_BLF_G3);
            for (int sk=0; sk<sm.nb; sk++) {
                int ca_skip = sm.chi_a[sk];
                for (int k=0; k<sm.nb; k++) if(k!=sk)
                    max_inter = std::max(max_inter, D_BLF_G3 * ca_skip * sm.chi_b[k]);
            }
        }
        if (d_psi_T_a) HIP_CHECK(hipFree(d_psi_T_a));
        if (d_psi_T_b) HIP_CHECK(hipFree(d_psi_T_b));
        if (d_inter_T) HIP_CHECK(hipFree(d_inter_T));
        psi_T_elems = max_psi; inter_T_elems = max_inter;
        HIP_CHECK(hipMalloc(&d_psi_T_a, max_psi*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_psi_T_b, max_psi*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_inter_T, max_inter*sizeof(float)));
    }

    // ── SGEMM helpers — all GPU, no D2H ──────────────────────────────────────

    // Transpose psi[n1, n2, D] → d_out[D, n1, n2] on GPU
    void _transpose_to_batch(float* d_psi, int n1, int n2, float* d_out) {
        int total = n1*n2*D_BLF_G3;
        int blocks = (total+255)/256;
        hipLaunchKernelGGL(transpose_to_batch_kernel, blocks, 256, 0, 0,
            d_psi, d_out, n1, n2, total);
    }

    // Step 1: inter_T[D, ca_skip, cb1] = psi_a[ca_skip, ca1, D] @ M[ca1, cb1]
    // Output: d_inter_T (class member, [D, ca_skip, cb1] layout)
    void _step1_gpu(float* psi_a, int ca_skip, int ca1, float* M, int cb1) {
        // Transpose psi_a[ca_skip,ca1,D] → d_psi_T_a[D,ca_skip,ca1]
        _transpose_to_batch(psi_a, ca_skip, ca1, d_psi_T_a);

        // SGEMM strided batched: d_inter_T[D,ca_skip,cb1] = M @ d_psi_T_a (batched)
        float alpha=1.f, beta=0.f;
        rocblas_sgemm_strided_batched(handle,
            rocblas_operation_none, rocblas_operation_none,
            cb1, ca_skip, ca1,
            &alpha,
            M,          cb1, 0,
            d_psi_T_a,  ca1, ca_skip*ca1,
            &beta,
            d_inter_T,  cb1, ca_skip*cb1,
            D_BLF_G3);
        // d_inter_T is [D, ca_skip, cb1] — same layout as psi_T, ready for step 2
    }

    // Step 2: nm_col[cb_skip,ca_skip] accumulated from inter_T and psi_b
    // inter_T[D,ca_skip,cb1] (in d_inter_T), psi_b[cb_skip,cb1,D] (device ptr)
    void _step2_gpu(float* psi_b, int ca_skip, int cb1, int cb_skip, float* d_out) {
        // Transpose psi_b[cb_skip,cb1,D] → d_psi_T_b[D,cb_skip,cb1] — all on GPU
        _transpose_to_batch(psi_b, cb_skip, cb1, d_psi_T_b);

        // For each d: nm_col[cb_skip,ca_skip] += psi_b_T_d[cb_skip,cb1]^T_col @ inter_T_d[ca_skip,cb1]^T_col^T
        // = sgemm(op_T, op_none, cb_skip, ca_skip, cb1,
        //         psi_b_T_d, lda=cb1,
        //         inter_T_d, lda=cb1,
        //         nm, ldc=cb_skip)
        float alpha=1.f, beta0=0.f, beta1=1.f;
        for(int d=0;d<D_BLF_G3;d++){
            rocblas_sgemm(handle,
                rocblas_operation_transpose, rocblas_operation_none,
                cb_skip, ca_skip, cb1,
                &alpha,
                d_psi_T_b + d*cb_skip*cb1, cb1,
                d_inter_T  + d*ca_skip*cb1, cb1,
                d==0?&beta0:&beta1,
                d_out, cb_skip);
        }
        // d_out: col-major [cb_skip, ca_skip] = nm_row[ca_skip, cb_skip] ✓
    }

    // Rearrange deg-3 psi on GPU: psi[bond0,bond1,bond2,D] → out[skip,k1,k2,D]
    void _rearrange_deg3_gpu(float* d_psi, int chi_a[3], int skip_k, int k1, int k2,
                              int ca_skip, int ca1, int ca2, float* d_out) {
        int total = ca_skip*ca1*ca2*D_BLF_G3;
        int blocks = (total+255)/256;
        hipLaunchKernelGGL(rearrange_deg3_kernel, blocks, 256, 0, 0,
            d_psi, d_out, chi_a[0], chi_a[1], chi_a[2],
            skip_k, k1, k2, ca_skip, ca1, ca2, total);
    }

    // CPU fallback for degree-3 (correct, from BLF2)
    void _deg3_cpu_fallback(int src, int skip_k, int k1, int k2,
                             int ca_skip, int cb_skip, int ca1, int cb1, int ca2, int cb2,
                             float* psi_a_dev, float* psi_b_dev,
                             float* M1_dev, float* M2_dev,
                             float* old_M_dev, float*& out_M_dev, float& max_delta)
    {
        auto& sm = smeta[src];
        std::vector<float> pa(sm.vola*D), pb(sm.volb*D), M1(ca1*cb1), M2(ca2*cb2);
        HIP_CHECK(hipMemcpy(pa.data(), psi_a_dev, sm.vola*D*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(pb.data(), psi_b_dev, sm.volb*D*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(M1.data(), M1_dev, ca1*cb1*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(M2.data(), M2_dev, ca2*cb2*sizeof(float), hipMemcpyDeviceToHost));

        std::vector<float> inter(ca_skip*ca1*cb2*D,0.f);
        for(int is=0;is<ca_skip;is++) for(int i1=0;i1<ca1;i1++) for(int i2=0;i2<ca2;i2++){
            int idx[3]; idx[skip_k]=is;idx[k1]=i1;idx[k2]=i2;
            int fa=(idx[0]*sm.chi_a[1]+idx[1])*sm.chi_a[2]+idx[2];
            for(int jb2=0;jb2<cb2;jb2++) for(int d=0;d<D;d++)
                inter[((is*ca1+i1)*cb2+jb2)*D+d]+=pa[fa*D+d]*M2[i2*cb2+jb2];
        }
        std::vector<float> inter2(ca_skip*cb1*cb2*D,0.f);
        for(int is=0;is<ca_skip;is++) for(int i1=0;i1<ca1;i1++)
        for(int jb2=0;jb2<cb2;jb2++) for(int jb1=0;jb1<cb1;jb1++) for(int d=0;d<D;d++)
            inter2[((is*cb1+jb1)*cb2+jb2)*D+d]+=inter[((is*ca1+i1)*cb2+jb2)*D+d]*M1[i1*cb1+jb1];

        std::vector<float> nm(ca_skip*cb_skip,0.f);
        for(int is=0;is<ca_skip;is++) for(int ib_s=0;ib_s<cb_skip;ib_s++)
        for(int jb1=0;jb1<cb1;jb1++) for(int jb2=0;jb2<cb2;jb2++){
            int jdx[3];jdx[skip_k]=ib_s;jdx[k1]=jb1;jdx[k2]=jb2;
            int fb=(jdx[0]*sm.chi_b[1]+jdx[1])*sm.chi_b[2]+jdx[2];
            for(int d=0;d<D;d++)
                nm[is*cb_skip+ib_s]+=inter2[((is*cb1+jb1)*cb2+jb2)*D+d]*pb[fb*D+d];
        }
        float s=0.f; for(auto v:nm) s=std::max(s,std::abs(v));
        if(s>1e-20f) for(auto& v:nm) v/=s;

        std::vector<float> old_h(ca_skip*cb_skip);
        HIP_CHECK(hipMemcpy(old_h.data(), old_M_dev, ca_skip*cb_skip*sizeof(float), hipMemcpyDeviceToHost));
        for(int i=0;i<ca_skip*cb_skip;i++) max_delta=std::max(max_delta,std::abs(nm[i]-old_h[i]));
        HIP_CHECK(hipMemcpy(out_M_dev, nm.data(), ca_skip*cb_skip*sizeof(float), hipMemcpyHostToDevice));
    }

    // ── Bethe PF (CPU) ────────────────────────────────────────────────────────
    float _bethe_cpu() {
        double log_num=0.0, sign_num=1.0;
        for(auto& site : grid_a_ptr->sites){
            int sid=site.id; auto& sm=smeta[sid]; int nb=sm.nb;
            std::vector<float> pa(sm.vola*D), pb(sm.volb*D);
            HIP_CHECK(hipMemcpy(pa.data(), d_psi_a[sid], sm.vola*D*sizeof(float), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(pb.data(), d_psi_b[sid], sm.volb*D*sizeof(float), hipMemcpyDeviceToHost));
            auto get_Mh=[&](int k)->std::vector<float>{
                int bk=sm.bond_id[k]; int ca=frozen_chi_a[bk], cb=frozen_chi_b[bk];
                float* ptr=(sid==grid_a_ptr->bonds[bk].site_a)?d_M_bwd[bk]:d_M_fwd[bk];
                std::vector<float> m(ca*cb);
                HIP_CHECK(hipMemcpy(m.data(),ptr,ca*cb*sizeof(float),hipMemcpyDeviceToHost));
                return m;
            };
            std::vector<std::vector<float>> Ms(nb);
            for(int k=0;k<nb;k++) Ms[k]=get_Mh(k);
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
        double log_den=0.0, sign_den=1.0;
        for(auto& bond : grid_a_ptr->bonds){
            int bid=bond.id; int ca=frozen_chi_a[bid],cb=frozen_chi_b[bid];
            std::vector<float> mf(ca*cb), mb(ca*cb);
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

    void _free(){
        for(auto p:d_psi_a)  if(p)HIP_CHECK(hipFree(p));
        for(auto p:d_psi_b)  if(p)HIP_CHECK(hipFree(p));
        for(auto p:d_M_fwd)  if(p)HIP_CHECK(hipFree(p));
        for(auto p:d_M_bwd)  if(p)HIP_CHECK(hipFree(p));
        for(auto p:d_M_fwd2) if(p)HIP_CHECK(hipFree(p));
        for(auto p:d_M_bwd2) if(p)HIP_CHECK(hipFree(p));
        if(d_scratch)   HIP_CHECK(hipFree(d_scratch));
        if(d_scratch2)  HIP_CHECK(hipFree(d_scratch2));
        if(d_psi_T_a)   HIP_CHECK(hipFree(d_psi_T_a));
        if(d_psi_T_b)   HIP_CHECK(hipFree(d_psi_T_b));
        if(d_inter_T)   HIP_CHECK(hipFree(d_inter_T));
    }
};
