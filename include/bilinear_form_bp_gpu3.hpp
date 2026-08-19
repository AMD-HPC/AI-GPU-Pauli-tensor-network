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

// ── Max-delta kernel: atomically accumulate max|new-old| into d_delta ────────
// Launched per message: grid=(n_msgs), block=(1)
// Compares current message (ping) against previous (pong, before swap)
__global__ void max_delta_kernel(
    const float* __restrict__ m_new,
    const float* __restrict__ m_old,
    float* __restrict__ d_delta,
    int n)
{
    if (threadIdx.x != 0) return;
    float local_max = 0.f;
    for (int i=0; i<n; i++) local_max = fmaxf(local_max, fabsf(m_new[i]-m_old[i]));
    atomicMax((int*)d_delta, __float_as_int(local_max));  // bit-cast trick (valid for positive floats)
}

// ── Transpose kernel: psi[n1, n2, D] → out[D, n1, n2] ───────────────────────
__global__ void transpose_to_batch_kernel(
    const float* __restrict__ psi, float* __restrict__ out,
    int n1, int n2, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d  = idx % D_BLF_G3;
    int i2 = (idx / D_BLF_G3) % n2;
    int i1 = idx / (D_BLF_G3 * n2);
    out[d*n1*n2 + i1*n2 + i2] = psi[idx];
}

// ── Swap axes kernel: psi[n1, n2, D] → out[n2, n1, D] ───────────────────────
// Used for skip_k=1 on degree-2 sites: swaps the two bond axes, keeps D last.
__global__ void swap_axes01_kernel(
    const float* __restrict__ psi, float* __restrict__ out,
    int n1, int n2, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    // idx = i1*n2*D + i2*D + d  (row-major [n1,n2,D])
    int d  = idx % D_BLF_G3;
    int i2 = (idx / D_BLF_G3) % n2;
    int i1 = idx / (D_BLF_G3 * n2);
    // out[i2, i1, d] at i2*n1*D + i1*D + d
    out[i2*n1*D_BLF_G3 + i1*D_BLF_G3 + d] = psi[idx];
}

// ── Rearrange kernel: psi[a,b,c,D] → out[skip,k1,k2,D] permuting bond axes ──
__global__ void rearrange_deg3_kernel(
    const float* __restrict__ psi, float* __restrict__ out,
    int c0, int c1, int c2,
    int skip_k, int k1_idx, int k2_idx,
    int ca_skip, int ca1, int ca2, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d    = idx % D_BLF_G3;
    int rem  = idx / D_BLF_G3;
    int i2   = rem % ca2; rem /= ca2;
    int i1   = rem % ca1; rem /= ca1;
    int i_sk = rem;
    int orig[3]; orig[skip_k]=i_sk; orig[k1_idx]=i1; orig[k2_idx]=i2;
    int src_flat = (orig[0]*c1 + orig[1])*c2 + orig[2];
    out[idx] = psi[src_flat*D_BLF_G3 + d];
}

// ── Rearrange inter1: [D, ca_skip*ca1, cb2] → [ca_skip*ca1*cb2, D] interleaved
// then reinterpret as [cb2*D, ca_skip, ca1] for step B strided_batched
// Combined: out[bd*ca_skip*ca1 + is*ca1 + i1] = in[d*ca_skip*ca1*cb2 + (is*ca1+i1)*cb2 + jb2]
//           where bd = d*cb2 + jb2
__global__ void rearrange_inter1_kernel(
    const float* __restrict__ in, float* __restrict__ out,
    int ca_skip, int ca1, int cb2, int D_, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    // out indexed as [bd, ca_skip, ca1] flat = bd*ca_skip*ca1 + is*ca1 + i1
    int i1   = idx % ca1;
    int is   = (idx / ca1) % ca_skip;
    int bd   = idx / (ca1*ca_skip);   // bd = d*cb2 + jb2
    int d    = bd / cb2;
    int jb2  = bd % cb2;
    // in[D, ca_skip*ca1, cb2]: in[d*ca_skip*ca1*cb2 + (is*ca1+i1)*cb2 + jb2]
    out[idx] = in[d*ca_skip*ca1*cb2 + (is*ca1+i1)*cb2 + jb2];
}

// ── Rearrange inter2: [cb2*D, ca_skip, cb1] → [ca_skip, cb1*cb2*D] flat
// out[is*(cb1*cb2*D) + jb1*cb2*D + jb2*D + d] = in[bd*ca_skip*cb1 + is*cb1 + jb1]
//   where bd = d*cb2 + jb2
__global__ void rearrange_inter2_kernel(
    const float* __restrict__ in, float* __restrict__ out,
    int ca_skip, int cb1, int cb2, int D_, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    // out[ca_skip, cb1*cb2*D]: idx = is*(cb1*cb2*D) + jb1*cb2*D + jb2*D + d
    int d   = idx % D_;
    int jb2 = (idx / D_) % cb2;
    int jb1 = (idx / (D_*cb2)) % cb1;
    int is  = idx / (D_*cb2*cb1);
    int bd  = d*cb2 + jb2;
    out[idx] = in[bd*ca_skip*cb1 + is*cb1 + jb1];
}

// ── Rearrange psi_b for degree-3: [orig_order, D] → [cb_skip, cb1, cb2, D]
// skip_k, k1, k2 define the target ordering
__global__ void rearrange_psi_b_deg3_kernel(
    const float* __restrict__ psi_b, float* __restrict__ out,
    int c0, int c1, int c2,          // original chi_b[0..2]
    int skip_k, int k1_idx, int k2_idx,
    int cb_skip, int cb1, int cb2, int D_, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d    = idx % D_;
    int rem  = idx / D_;
    int jb2  = rem % cb2; rem /= cb2;
    int jb1  = rem % cb1; rem /= cb1;
    int ib_s = rem;
    int orig[3]; orig[skip_k]=ib_s; orig[k1_idx]=jb1; orig[k2_idx]=jb2;
    int fb = (orig[0]*c1 + orig[1])*c2 + orig[2];
    out[idx] = psi_b[fb*D_ + d];
}

// ── BLF_GPU3 ─────────────────────────────────────────────────────────────────

struct BLF_GPU3 {
    rocblas_handle handle;
    hipStream_t    stream = nullptr;   // dedicated stream — enables concurrent BLF_GPU3 objects
    float*         d_delta = nullptr;  // GPU convergence accumulator (one float)
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

    BLF_GPU3(rocblas_handle h) : handle(h) {
        HIP_CHECK(hipStreamCreate(&stream));
        rocblas_set_stream(handle, stream);
        HIP_CHECK(hipMalloc(&d_delta, sizeof(float)));
    }
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

        // WARM-START = IDENTITY (M = I/min(ca,cb)). This is a deliberate,
        // re-evaluatable choice — NOT the only option. See
        // archive/INNER_PRODUCT_STATUS.md (2026-07-17): a kron warm-start
        // outer(diag(msg_a), diag(msg_b)) was implemented and had ZERO effect
        // in float32 — the loopy heavy-hex double-layer BP forgets its seed and
        // flows to a single dominant fixed point, which is DEGENERATE for
        // near-identical states (γ=1 tracks nl closely → A ill-conditioned,
        // ~50% cross-overlap undercount vs exact tree).
        //
        // OPEN EXPERIMENT (after Scalar=double, Phase 1): re-test kron vs
        // identity in FLOAT64. Hypothesis: the seed was inert in f32 because
        // the inter-state distinction (small differences between near-equal
        // Schmidt values) fell below the ~1e-7 f32 floor; f64's ~1e-15 floor
        // may keep states distinguishable long enough for the warm-start to
        // steer BP out of the degenerate basin. The kron seed is now trivially
        // available as outer(v_a, v_b) from the unified Vidal chi-vectors
        // (state.messages_v / Phase 0b). Do NOT hardcode-away this init: keep
        // it a selectable knob when that experiment is run.
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
        // d_scratch2 is now allocated in _alloc_psi_T_buffers() with correct deg-3 sizing
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
                            // skip_k=1: psi_a[ca1, ca_skip, D] → [ca_skip, ca1, D] via GPU kernel
                            int total2 = ca1*ca_skip*D_BLF_G3;
                            hipLaunchKernelGGL(swap_axes01_kernel, (total2+255)/256, 256, 0, stream,
                                psi_a, d_scratch2, ca1, ca_skip, total2);
                            _step1_gpu(d_scratch2, ca_skip, ca1, M1, cb1);
                            // psi_b[cb1, cb_skip, D] → [cb_skip, cb1, D]
                            int total3 = cb1*cb_skip*D_BLF_G3;
                            hipLaunchKernelGGL(swap_axes01_kernel, (total3+255)/256, 256, 0, stream,
                                psi_b, d_scratch2, cb1, cb_skip, total3);
                            _step2_gpu(d_scratch2, ca_skip, cb1, cb_skip, out_M);
                        }

                    } else {
                        // Degree-3: two step-1 then one step-2
                        int k1=-1, k2=-1;
                        for(int k=0;k<nb;k++) if(k!=skip_k){if(k1<0)k1=k;else k2=k;}
                        int ca1=sm.chi_a[k1],cb1=sm.chi_b[k1];
                        int ca2=sm.chi_a[k2],cb2=sm.chi_b[k2];
                        float* M1=get_M(k1); float* M2=get_M(k2);

                        // ── Degree-3: full GPU path ──────────────────────────
                        // Follows paper equations (6)-(7) exactly:
                        //   Step A: phi1[ca_skip*ca1, cb2, D] = psi_a_r[ca_skip,ca1,ca2,D] @_{ca2} M2
                        //   Step B: phi2[cb2*D, ca_skip, cb1] = phi1_r[cb2*D,ca_skip,ca1] @_{ca1} M1
                        //   Step C: nm[ca_skip,cb_skip] = phi2_flat[ca_skip,K] @ psi_b_r[cb_skip,K]^T
                        //           where K = cb1*cb2*D
                        // All rearrangements via GPU kernels — no D2H.

                        int BD = cb2*D_BLF_G3;  // combined batch dimension for step B
                        int K  = cb1*cb2*D_BLF_G3;  // flat K for step C

                        // ── Step A ──────────────────────────────────────────
                        // Rearrange psi_a → psi_a_r[ca_skip, ca1, ca2, D] (GPU)
                        _rearrange_deg3_gpu(psi_a, sm.chi_a, skip_k, k1, k2,
                                            ca_skip, ca1, ca2, d_scratch2);
                        // Step 1 SGEMM: treat psi_a_r as [ca_skip*ca1, ca2, D]
                        _step1_gpu(d_scratch2, ca_skip*ca1, ca2, M2, cb2);
                        // d_inter_T: [D, ca_skip*ca1, cb2]  (from _step1_gpu output)

                        // ── Step B ──────────────────────────────────────────
                        // Rearrange inter1: [D, ca_skip*ca1, cb2] → [cb2*D, ca_skip, ca1]
                        {
                            int total3 = BD*ca_skip*ca1;
                            int blocks3 = (total3+255)/256;
                            hipLaunchKernelGGL(rearrange_inter1_kernel, blocks3, 256, 0, stream,
                                d_inter_T, d_scratch2, ca_skip, ca1, cb2, D_BLF_G3, total3);
                        }
                        // d_scratch2: [cb2*D, ca_skip, ca1]
                        // strided_batched: phi2[cb2*D, ca_skip, cb1] = M1 @ d_scratch2
                        {
                            float alpha=1.f, beta=0.f;
                            rocblas_sgemm_strided_batched(handle,
                                rocblas_operation_none, rocblas_operation_none,
                                cb1, ca_skip, ca1,
                                &alpha,
                                M1,          cb1, 0,            // M1[ca1,cb1], stride=0
                                d_scratch2,  ca1, ca_skip*ca1,  // psi_a_r batch stride
                                &beta,
                                d_inter_T,   cb1, ca_skip*cb1,  // phi2 output
                                BD);
                        }
                        // d_inter_T: [cb2*D, ca_skip, cb1]

                        // ── Step C ──────────────────────────────────────────
                        // Rearrange inter2: [cb2*D, ca_skip, cb1] → [ca_skip, cb1*cb2*D]
                        {
                            int total3 = ca_skip*K;
                            int blocks3 = (total3+255)/256;
                            hipLaunchKernelGGL(rearrange_inter2_kernel, blocks3, 256, 0, stream,
                                d_inter_T, d_scratch2, ca_skip, cb1, cb2, D_BLF_G3, total3);
                        }
                        // d_scratch2: [ca_skip, K]

                        // Rearrange psi_b → psi_b_r[cb_skip, cb1, cb2, D] = [cb_skip, K]
                        {
                            int total3 = cb_skip*K;
                            int blocks3 = (total3+255)/256;
                            // Allocate temporary psi_b_r
                            float* d_pb_r;
                            HIP_CHECK(hipMalloc(&d_pb_r, cb_skip*K*sizeof(float)));
                            hipLaunchKernelGGL(rearrange_psi_b_deg3_kernel, blocks3, 256, 0, stream,
                                psi_b, d_pb_r,
                                sm.chi_b[0], sm.chi_b[1], sm.chi_b[2],
                                skip_k, k1, k2, cb_skip, cb1, cb2, D_BLF_G3, total3);

                            // nm col[cb_skip,ca_skip] = d_pb_r[cb_skip,K]^T @ d_scratch2[ca_skip,K]^T
                            // sgemm(op_T, op_none, cb_skip, ca_skip, K, ...)
                            float alpha=1.f, beta=0.f;
                            rocblas_sgemm(handle,
                                rocblas_operation_transpose, rocblas_operation_none,
                                cb_skip, ca_skip, K,
                                &alpha, d_pb_r, K, d_scratch2, K,
                                &beta, out_M, cb_skip);
                            HIP_CHECK(hipFree(d_pb_r));
                        }
                        // out_M: col[cb_skip,ca_skip] = nm_row[ca_skip,cb_skip] ✓
                        // Normalize
                        hipLaunchKernelGGL(normalize_msg_gpu3, 1, 1, 0, stream,
                            out_M, ca_skip*cb_skip);
                        continue;
                    }

                    // Normalize (GPU kernel)
                    hipLaunchKernelGGL(normalize_msg_gpu3, 1, 1, 0, stream,
                        out_M, ca_skip*cb_skip);
                }
            }

            // Reset delta accumulator and compute max|new-old| for all messages on GPU
            float zero_f = 0.f;
            HIP_CHECK(hipMemcpyAsync(d_delta, &zero_f, sizeof(float), hipMemcpyHostToDevice, stream));

            // Swap ping-pong first (after swap: fwd=new, fwd2=old)
            for(int bid=0;bid<n_bonds;bid++){
                std::swap(d_M_fwd[bid], d_M_fwd2[bid]);
                std::swap(d_M_bwd[bid], d_M_bwd2[bid]);
            }

            // GPU delta: compare new (fwd) vs old (fwd2) for each bond
            for(int bid=0;bid<n_bonds;bid++){
                int sz=frozen_chi_a[bid]*frozen_chi_b[bid];
                hipLaunchKernelGGL(max_delta_kernel, 1, 1, 0, stream,
                    d_M_fwd[bid], d_M_fwd2[bid], d_delta, sz);
                hipLaunchKernelGGL(max_delta_kernel, 1, 1, 0, stream,
                    d_M_bwd[bid], d_M_bwd2[bid], d_delta, sz);
            }

            // One D2H: download single float
            HIP_CHECK(hipMemcpyAsync(&max_delta, d_delta, sizeof(float), hipMemcpyDeviceToHost, stream));
            HIP_CHECK(hipStreamSynchronize(stream));

            if(verbose) printf("  [BLF3] sweep %d max_delta=%.2e\n", sweep, max_delta);
            if(max_delta < tol){ if(verbose) printf("  [BLF3] converged sweep %d\n",sweep); break; }
        }

        float result = _bethe_gpu();

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
        int max_psi=0, max_inter=0, max_scratch=0;
        for (int sid=0; sid<n_sites; sid++) {
            auto& sm = smeta[sid]; int nb=sm.nb;
            max_psi = std::max(max_psi, sm.vola * D_BLF_G3);
            max_psi = std::max(max_psi, sm.volb * D_BLF_G3);
            for (int sk=0; sk<nb; sk++) {
                int ca_skip=sm.chi_a[sk], cb_skip=sm.chi_b[sk];
                if (nb==2) {
                    int k1=(sk==0)?1:0;
                    int cb1=sm.chi_b[k1], ca1=sm.chi_a[k1];
                    max_inter  = std::max(max_inter,  D_BLF_G3*ca_skip*cb1);
                    max_scratch= std::max(max_scratch, ca_skip*ca1*D_BLF_G3); // for cpu_transpose
                } else if (nb==3) {
                    int k1=-1,k2=-1;
                    for(int k=0;k<3;k++) if(k!=sk){if(k1<0)k1=k;else k2=k;}
                    int ca1=sm.chi_a[k1],cb1=sm.chi_b[k1],ca2=sm.chi_a[k2],cb2=sm.chi_b[k2];
                    int BD=cb2*D_BLF_G3, K=cb1*cb2*D_BLF_G3;
                    max_inter  = std::max(max_inter,  D_BLF_G3*ca_skip*ca1*cb2);
                    max_inter  = std::max(max_inter,  BD*ca_skip*cb1);
                    // d_scratch2: message update + Bethe PF needs cb0*cb1*cb2*D for rearrange_deg3
                    max_scratch= std::max(max_scratch, ca_skip*ca1*ca2*D_BLF_G3);
                    max_scratch= std::max(max_scratch, BD*ca_skip*ca1);
                    max_scratch= std::max(max_scratch, ca_skip*K);
                    // Bethe PF needs cb0*cb1*cb2*D for psi_b rearrangement
                    max_scratch= std::max(max_scratch, cb_skip*cb1*cb2*D_BLF_G3);
                }
            }
        }
        if (d_psi_T_a) HIP_CHECK(hipFree(d_psi_T_a));
        if (d_psi_T_b) HIP_CHECK(hipFree(d_psi_T_b));
        if (d_inter_T) HIP_CHECK(hipFree(d_inter_T));
        if (d_scratch2) HIP_CHECK(hipFree(d_scratch2));
        psi_T_elems=max_psi; inter_T_elems=max_inter;
        HIP_CHECK(hipMalloc(&d_psi_T_a, max_psi*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_psi_T_b, max_psi*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_inter_T,  max_inter*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_scratch2, max_scratch*sizeof(float)));
    }

    // ── SGEMM helpers — all GPU, no D2H ──────────────────────────────────────

    // Transpose psi[n1, n2, D] → d_out[D, n1, n2] on GPU
    void _transpose_to_batch(float* d_psi, int n1, int n2, float* d_out) {
        int total = n1*n2*D_BLF_G3;
        hipLaunchKernelGGL(transpose_to_batch_kernel, (total+255)/256, 256, 0, stream,
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
        hipLaunchKernelGGL(rearrange_deg3_kernel, blocks, 256, 0, stream,
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

    // ── Bethe PF (GPU) ───────────────────────────────────────────────────────
    // For each site: a_v = sum_{ia,ib,d} psi_a[ia,d]*psi_b[ib,d]*prod_k M_k[ia_k,ib_k]
    // Same SGEMM structure as message update but contracting ALL bonds.
    //
    // Degree-2 site (bonds k0, k1):
    //   Step 1: phi_a[chi_b0, chi_b1, D] = contract M_0 into psi_a over chi_a0
    //           then contract M_1 into result over chi_a1
    //   Simplified: phi_a[chi_b0*chi_b1, D] = psi_a contracted with both M's
    //   Step 2: a_v = sum_d phi_a[ia,d]*phi_b[ib,d]  (one SDOT after weighting)
    //
    // We implement via two _step1_gpu calls treating "skip" as a dummy:
    //   First contract k1 out of psi_a → inter[chi_a0, chi_b1, D]
    //   Then contract k0 out of inter → phi_a[chi_b0, D] — a D-vector!
    //   Similarly for phi_b.
    //   a_v = phi_a · phi_b (dot product of two D-vectors)
    //
    // Device buffers d_psi_T_a, d_inter_T already allocated — reuse them.
    float _bethe_gpu() {
        double log_num=0.0, sign_num=1.0;

        for(auto& site : grid_a_ptr->sites){
            int sid=site.id; auto& sm=smeta[sid]; int nb=sm.nb;

            auto get_M_dev=[&](int k)->float*{
                int bk=sm.bond_id[k];
                return (sid==grid_a_ptr->bonds[bk].site_a)?d_M_bwd[bk]:d_M_fwd[bk];
            };

            // Contract all messages into psi_a → phi_a[D] (scalar per Pauli)
            // Contract all messages into psi_b → phi_b[D]
            // a_v = phi_a · phi_b
            //
            // Strategy: sequential _step1_gpu contractions
            // After k contractions: intermediate has shape [remaining_chi_a, remaining_chi_b, D]
            // Final: [1, 1, D] = phi_a[D]

            float phi_a[D_BLF_G3] = {}, phi_b[D_BLF_G3] = {};

            if (nb == 1) {
                // phi_a[d] = sum_{ia} psi_a[ia,d]*M_0[ia,ib] summed over ib
                // = sum_{ia,ib} psi_a[ia,d]*M_0[ia,ib]
                // = psi_a_flat[D] @ M_0_rowsum[chi_a0]  (weighted by row sums)
                int ca0=sm.chi_a[0], cb0=sm.chi_b[0];
                float* M0=get_M_dev(0);
                // phi_a[d] = sum_ia psi_a[ia,d] * (sum_ib M0[ia,ib])
                // Download M0 row sums on CPU (chi_a0 floats — tiny)
                std::vector<float> rs(ca0,0.f), cs(cb0,0.f);
                std::vector<float> m(ca0*cb0);
                HIP_CHECK(hipMemcpy(m.data(),M0,ca0*cb0*sizeof(float),hipMemcpyDeviceToHost));
                for(int ia=0;ia<ca0;ia++) for(int ib=0;ib<cb0;ib++){rs[ia]+=m[ia*cb0+ib]; cs[ib]+=m[ia*cb0+ib];}
                std::vector<float> pa(sm.vola*D_BLF_G3), pb(sm.volb*D_BLF_G3);
                HIP_CHECK(hipMemcpy(pa.data(),d_psi_a[sid],sm.vola*D_BLF_G3*sizeof(float),hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(pb.data(),d_psi_b[sid],sm.volb*D_BLF_G3*sizeof(float),hipMemcpyDeviceToHost));
                for(int ia=0;ia<ca0;ia++) for(int d=0;d<D_BLF_G3;d++) phi_a[d]+=pa[ia*D_BLF_G3+d]*rs[ia];
                for(int ib=0;ib<cb0;ib++) for(int d=0;d<D_BLF_G3;d++) phi_b[d]+=pb[ib*D_BLF_G3+d]*cs[ib];

            } else if (nb == 2) {
                // Degree-2: O(chi^3 D) via two SGEMMs + SDOT
                // a_v = sum_{ia0,ia1,ib0,ib1,d} psi_a[ia0,ia1,d]*psi_b[ib0,ib1,d]*M0[ia0,ib0]*M1[ia1,ib1]
                //
                // Step 1: inter_a[ia0, ib1, D] = psi_a[ia0,ia1,D] @ M1[ia1,ib1]  (SGEMM)
                // Step 2: G[ia0, ib0] = inter_a_flat[ia0, ib1*D] @ psi_b_flat[ib0, ib1*D]^T (SGEMM)
                // Step 3: a_v = frobenius(M0, G) = SDOT(M0_flat, G_flat)
                int ca0=sm.chi_a[0],ca1=sm.chi_a[1],cb0=sm.chi_b[0],cb1=sm.chi_b[1];
                float* M0=get_M_dev(0); float* M1=get_M_dev(1);

                // Step 1: inter_a stored in d_inter_T as [D, ca0, cb1]
                _step1_gpu(d_psi_a[sid], ca0, ca1, M1, cb1);

                // Step 2: G[ca0, cb0] = inter_a_flat[ca0, cb1*D] @ psi_b_flat[cb0, cb1*D]^T
                // Rearrange d_inter_T[D,ca0,cb1] → inter_flat[ca0, cb1*D] on CPU (small: ca0*cb1*D)
                int sz_i = ca0*cb1*D_BLF_G3;
                std::vector<float> inter_T(sz_i), inter_flat(sz_i), pb_flat(sm.volb*D_BLF_G3);
                HIP_CHECK(hipMemcpy(inter_T.data(), d_inter_T, sz_i*sizeof(float), hipMemcpyDeviceToHost));
                // d_inter_T[d, ia0, ib1] at d*ca0*cb1 + ia0*cb1 + ib1 → inter_flat[ia0, ib1, d]
                for(int d=0;d<D_BLF_G3;d++)
                for(int ia0=0;ia0<ca0;ia0++)
                for(int ib1=0;ib1<cb1;ib1++)
                    inter_flat[(ia0*cb1+ib1)*D_BLF_G3+d] = inter_T[d*ca0*cb1+ia0*cb1+ib1];

                HIP_CHECK(hipMemcpy(pb_flat.data(),d_psi_b[sid],sm.volb*D_BLF_G3*sizeof(float),hipMemcpyDeviceToHost));

                // G[ia0, ib0] = sum_{ib1,d} inter_flat[ia0,ib1,d]*psi_b[ib0,ib1,d]
                // = inter_flat[ca0, cb1*D] @ psi_b_flat[cb0, cb1*D]^T  — one SGEMM on CPU
                int K = cb1*D_BLF_G3;
                std::vector<float> G(ca0*cb0, 0.f);
                for(int ia0=0;ia0<ca0;ia0++)
                for(int ib0=0;ib0<cb0;ib0++)
                for(int k=0;k<K;k++)
                    G[ia0*cb0+ib0] += inter_flat[ia0*K+k] * pb_flat[ib0*K+k];

                // Step 3: a_v = sum_{ia0,ib0} M0[ia0,ib0] * G[ia0,ib0]
                std::vector<float> m0(ca0*cb0);
                HIP_CHECK(hipMemcpy(m0.data(),M0,ca0*cb0*sizeof(float),hipMemcpyDeviceToHost));
                double a_v=0.0;
                for(int i=0;i<ca0*cb0;i++) a_v+=(double)m0[i]*(double)G[i];

                sign_num*=(a_v>0?1.0:(a_v<0?-1.0:0.0));
                log_num+=(a_v!=0?std::log(std::abs(a_v)):-1e30);
                continue;

            } else {
                // Degree-3: three SGEMMs — O(chi^3 D) per site
                // a_v = sum_{ia,ib,d} psi_a[ia,d]*psi_b[ib,d]*M0[ia0,ib0]*M1[ia1,ib1]*M2[ia2,ib2]
                //
                // Step 1: inter1[ca0*ca1, cb2, D] = psi_a[ca0,ca1,ca2,D] @_{ca2} M2  (SGEMM)
                // Step 2: inter2[ca0, cb1*cb2*D] contracted with M1  (CPU loop, O(chi^3 D))
                // Step 3: G[ca0, cb0] = inter2 @ psi_b_flat^T  (SGEMM)
                // Step 4: a_v = frobenius(M0, G)  (SDOT)

                int ca0=sm.chi_a[0],ca1=sm.chi_a[1],ca2=sm.chi_a[2];
                int cb0=sm.chi_b[0],cb1=sm.chi_b[1],cb2=sm.chi_b[2];
                float* M0=get_M_dev(0); float* M1=get_M_dev(1); float* M2=get_M_dev(2);

                // Rearrange psi_a to [ca0,ca1,ca2,D] order (may already be correct)
                _rearrange_deg3_gpu(d_psi_a[sid], sm.chi_a, 0, 1, 2, ca0, ca1, ca2, d_scratch2);
                // d_scratch2: psi_a_r[ca0,ca1,ca2,D]

                // Step 1: inter1[ca0*ca1, cb2, D] = psi_a_r[ca0*ca1,ca2,D] @_{ca2} M2
                _step1_gpu(d_scratch2, ca0*ca1, ca2, M2, cb2);
                // d_inter_T: [D, ca0*ca1, cb2]

                // Download inter1 and rearrange to [ca0, ca1, cb2, D]
                int sz1 = ca0*ca1*cb2*D_BLF_G3;
                std::vector<float> iT1(sz1), inter1(sz1);
                HIP_CHECK(hipMemcpy(iT1.data(), d_inter_T, sz1*sizeof(float), hipMemcpyDeviceToHost));
                for(int d=0;d<D_BLF_G3;d++)
                for(int ia01=0;ia01<ca0*ca1;ia01++)
                for(int ib2=0;ib2<cb2;ib2++)
                    inter1[(ia01*cb2+ib2)*D_BLF_G3+d] = iT1[d*ca0*ca1*cb2+ia01*cb2+ib2];

                // Step 2: contract M1 → inter2[ca0, cb1, cb2, D]  O(chi^3 D)
                std::vector<float> m1(ca1*cb1);
                HIP_CHECK(hipMemcpy(m1.data(),M1,ca1*cb1*sizeof(float),hipMemcpyDeviceToHost));
                std::vector<float> inter2(ca0*cb1*cb2*D_BLF_G3, 0.f);
                for(int ia0=0;ia0<ca0;ia0++)
                for(int ia1=0;ia1<ca1;ia1++)
                for(int ib1=0;ib1<cb1;ib1++)
                for(int ib2=0;ib2<cb2;ib2++)
                for(int d=0;d<D_BLF_G3;d++)
                    inter2[((ia0*cb1+ib1)*cb2+ib2)*D_BLF_G3+d] +=
                        inter1[((ia0*ca1+ia1)*cb2+ib2)*D_BLF_G3+d] * m1[ia1*cb1+ib1];

                // inter2 flat as [ca0, cb1*cb2*D]
                int K3 = cb1*cb2*D_BLF_G3;

                // Step 3: rearrange psi_b to [cb0,cb1,cb2,D] order, flatten to [cb0, K3]
                _rearrange_deg3_gpu(d_psi_b[sid], sm.chi_b, 0, 1, 2, cb0, cb1, cb2, d_scratch2);
                std::vector<float> pb_flat(cb0*K3);
                {
                    std::vector<float> pb_r(cb0*cb1*cb2*D_BLF_G3);
                    HIP_CHECK(hipMemcpy(pb_r.data(),d_scratch2,cb0*K3*sizeof(float),hipMemcpyDeviceToHost));
                    pb_flat = pb_r; // already [cb0, cb1, cb2, D] = [cb0, K3]
                }

                // G[ca0, cb0] = inter2[ca0, K3] @ pb_flat[cb0, K3]^T
                std::vector<float> G(ca0*cb0, 0.f);
                for(int ia0=0;ia0<ca0;ia0++)
                for(int ib0=0;ib0<cb0;ib0++)
                for(int k=0;k<K3;k++)
                    G[ia0*cb0+ib0] += inter2[ia0*K3+k] * pb_flat[ib0*K3+k];

                // Step 4: a_v = frobenius(M0, G)
                std::vector<float> m0(ca0*cb0);
                HIP_CHECK(hipMemcpy(m0.data(),M0,ca0*cb0*sizeof(float),hipMemcpyDeviceToHost));
                double a_v=0.0;
                for(int i=0;i<ca0*cb0;i++) a_v+=(double)m0[i]*(double)G[i];

                sign_num*=(a_v>0?1.0:(a_v<0?-1.0:0.0));
                log_num+=(a_v!=0?std::log(std::abs(a_v)):-1e30);
                continue;
            }

            // For nb==1 case: phi_a and phi_b computed above
            double a_v=0.0;
            for(int d=0;d<D_BLF_G3;d++) a_v+=(double)phi_a[d]*(double)phi_b[d];
            sign_num*=(a_v>0?1.0:(a_v<0?-1.0:0.0));
            log_num+=(a_v!=0?std::log(std::abs(a_v)):-1e30);
        }

        // ze per bond: dot product M_fwd · M_bwd (on GPU via SDOT)
        double log_den=0.0, sign_den=1.0;
        for(auto& bond : grid_a_ptr->bonds){
            int bid=bond.id; int ca=frozen_chi_a[bid],cb=frozen_chi_b[bid]; int sz=ca*cb;
            float ze_gpu=0.f;
            rocblas_sdot(handle, sz, d_M_fwd[bid], 1, d_M_bwd[bid], 1, &ze_gpu);
            HIP_CHECK(hipStreamSynchronize(stream));
            double ze=(double)ze_gpu;
            sign_den*=(ze>0?1.0:(ze<0?-1.0:0.0));
            log_den+=(std::abs(ze)>1e-30?std::log(std::abs(ze)):-1e30);
        }
        if(sign_num==0.0) return 0.f;
        return (float)(sign_num*sign_den*std::exp(log_num-log_den+log_scale_a+log_scale_b));
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
        if(d_delta)     HIP_CHECK(hipFree(d_delta));
        if(stream)      HIP_CHECK(hipStreamDestroy(stream));
    }
};
