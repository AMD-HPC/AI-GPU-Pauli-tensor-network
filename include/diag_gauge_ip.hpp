#pragma once
/*
 * diag_gauge_ip.hpp — DG gauge inner product, fully GPU.
 *
 * Two-phase design to share eigendecomposition across pairs:
 *
 *   DG_StateGauge: per-state, computed once
 *     - Eigendecompose M_a per bond: M_a = A D A^T (rocsolver_ssyevd)
 *     - Build W_a = sqrt(|D|)*A  (for psi_a: scaled rotation)
 *     - Build W_rot = A          (for psi_b: unscaled rotation)
 *     - Apply W_a^T to psi_a → store d_gauged_psi_a on GPU
 *
 *   DG_Pair: per pair (i,j), reuses gauge from state_i
 *     - Apply W_rot^T (from gauge_i) to psi_j → d_tpsi_b
 *     - sdot(d_gauged_psi_a, d_tpsi_b) per site
 *     - Bethe PF with D_a edge correction
 *
 * Cost: 4 eigendecompositions (1 nl + 3 noisy) instead of 9.
 *       Per pair: only GEMM + sdot — no eigendecomposition.
 *
 * Build: link -lrocsolver -lrocblas
 */

#include "grid.hpp"
#include "types.hpp"
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do { \
    hipError_t e=(cmd); \
    if(e!=hipSuccess){fprintf(stderr,"HIP %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__);exit(1);} \
} while(0)
#endif
#define RS_CHECK(cmd) do { \
    rocblas_status s=(cmd); \
    if(s!=rocblas_status_success){fprintf(stderr,"rocSOLVER %d at %s:%d\n",(int)s,__FILE__,__LINE__);exit(1);} \
} while(0)

static double _dg_now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

// ── Kernel: build W_a = sqrt(|D|)*A and W_rot = A ────────────────────────────
__global__ void build_W_kernel(
    const float* __restrict__ A,
    const float* __restrict__ D,
    float* __restrict__ W_a,
    float* __restrict__ W_rot,
    const int* __restrict__ chi_per_bond,
    int chi_max, float rel_cutoff, int n_bonds)
{
    int bid = blockIdx.x;
    if (bid >= n_bonds) return;
    int chi = chi_per_bond[bid];
    const float* Ak = A    + (size_t)bid*chi_max*chi_max;
    const float* Dk = D    + (size_t)bid*chi_max;
    float*       Wa = W_a  + (size_t)bid*chi_max*chi_max;
    float*       Wr = W_rot+ (size_t)bid*chi_max*chi_max;
    float max_d = 0.f;
    for (int j=0;j<chi;j++) max_d=fmaxf(max_d,fabsf(Dk[j]));
    float thresh = rel_cutoff * max_d;
    for (int idx=threadIdx.x; idx<chi_max*chi_max; idx+=blockDim.x) {
        int col=idx/chi_max, row=idx%chi_max;
        if (row>=chi||col>=chi){Wa[idx]=0.f;Wr[idx]=0.f;continue;}
        float a=Ak[idx];
        if (fabsf(Dk[col])<thresh){Wa[idx]=0.f;Wr[idx]=0.f;}
        else {Wa[idx]=sqrtf(fabsf(Dk[col]))*a; Wr[idx]=a;}
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DG_StateGauge: eigendecompose one state's messages, store gauged psi_a
// ─────────────────────────────────────────────────────────────────────────────
struct DG_StateGauge {
    const Grid& grid;
    rocblas_handle handle;
    int n_sites, n_bonds, chi_max, own_dev;
    float eig_cutoff;

    // Per-bond gauge (GPU)
    float* d_A   = nullptr;  // eigenvectors [n_bonds, chi_max, chi_max]
    float* d_D   = nullptr;  // eigenvalues  [n_bonds, chi_max]
    float* d_E   = nullptr;  // syevd workspace
    float* d_W_a = nullptr;  // sqrt(|D|)*A  (for psi_a)
    float* d_Wr  = nullptr;  // A only       (for psi_b rotation)
    int*   d_info= nullptr;
    int*   d_chi = nullptr;
    std::vector<int> h_chi;

    // Gauged psi_a: psi after W_a^T applied, stored per-site (GPU)
    // One buffer per site — size chi^deg*D (variable)
    std::vector<float*> d_gpsi_a;  // [n_sites]

    // Scratch for GEMM during precompute
    float* d_scratch1 = nullptr;
    float* d_scratch2 = nullptr;
    size_t scratch_size = 0;

    // Per-site dot product accumulators (for self-overlap A[i,i])
    std::vector<double> h_log_av, h_sign_av;
    float* d_site_dots = nullptr;

    // Timing flag
    bool verbose = false;

    DG_StateGauge(const Grid& g, rocblas_handle h, int chi_max_, float eig_cut=1e-2f)
        : grid(g), handle(h), n_sites(g.n_sites), n_bonds(g.n_bonds),
          chi_max(chi_max_), eig_cutoff(eig_cut)
    {
        HIP_CHECK(hipGetDevice(&own_dev));
        size_t mat_sz=(size_t)n_bonds*chi_max*chi_max;
        size_t vec_sz=(size_t)n_bonds*chi_max;
        HIP_CHECK(hipMalloc(&d_A,    mat_sz*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_D,    vec_sz*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_E,    vec_sz*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_W_a,  mat_sz*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_Wr,   mat_sz*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_info, n_bonds*sizeof(int)));
        HIP_CHECK(hipMalloc(&d_chi,  n_bonds*sizeof(int)));
        scratch_size=(size_t)chi_max*chi_max*chi_max*D;
        HIP_CHECK(hipMalloc(&d_scratch1, scratch_size*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_scratch2, scratch_size*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_site_dots, n_sites*sizeof(float)));
        d_gpsi_a.resize(n_sites, nullptr);
        h_chi.resize(n_bonds);
        h_log_av.assign(n_sites, 0.0);
        h_sign_av.assign(n_sites, 1.0);
    }

    ~DG_StateGauge() {
        if(d_A) HIP_CHECK(hipFree(d_A));
        if(d_D) HIP_CHECK(hipFree(d_D));
        if(d_E) HIP_CHECK(hipFree(d_E));
        if(d_W_a) HIP_CHECK(hipFree(d_W_a));
        if(d_Wr)  HIP_CHECK(hipFree(d_Wr));
        if(d_info) HIP_CHECK(hipFree(d_info));
        if(d_chi)  HIP_CHECK(hipFree(d_chi));
        if(d_scratch1) HIP_CHECK(hipFree(d_scratch1));
        if(d_scratch2) HIP_CHECK(hipFree(d_scratch2));
        if(d_site_dots) HIP_CHECK(hipFree(d_site_dots));
        for(auto p:d_gpsi_a) if(p) HIP_CHECK(hipFree(p));
    }

    // Compute eigendecomposition + gauged psi_a for one state
    void precompute(
        const std::vector<float*>& psi_src,   // state psi on dev_src
        const std::vector<float*>& msg_src,   // state messages on dev_src
        int dev_src)
    {
        int cur; HIP_CHECK(hipGetDevice(&cur));
        HIP_CHECK(hipSetDevice(own_dev));
        double t0=_dg_now_ms(), t1;

        // ── 1. P2P copy messages, symmetrize, H2D ──────────────────────────
        size_t mat_bytes=(size_t)chi_max*chi_max*sizeof(float);
        std::vector<float> h_fwd(n_bonds*chi_max*chi_max);
        std::vector<float> h_bwd(n_bonds*chi_max*chi_max);
        float *d_mf=nullptr, *d_mb=nullptr;
        HIP_CHECK(hipMalloc(&d_mf, n_bonds*mat_bytes));
        HIP_CHECK(hipMalloc(&d_mb, n_bonds*mat_bytes));
        for(auto& bond:grid.bonds){
            int bid=bond.id; h_chi[bid]=bond.chi;
            HIP_CHECK(hipMemcpyPeer(d_mf+bid*chi_max*chi_max, own_dev,
                msg_src[bond.msg_from_a()], dev_src, mat_bytes));
            HIP_CHECK(hipMemcpyPeer(d_mb+bid*chi_max*chi_max, own_dev,
                msg_src[bond.msg_from_b()], dev_src, mat_bytes));
        }
        HIP_CHECK(hipMemcpy(h_fwd.data(), d_mf, n_bonds*mat_bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_bwd.data(), d_mb, n_bonds*mat_bytes, hipMemcpyDeviceToHost));
        HIP_CHECK(hipFree(d_mf)); HIP_CHECK(hipFree(d_mb));

        std::vector<float> h_A(n_bonds*chi_max*chi_max, 0.f);
        for(auto& bond:grid.bonds){
            int bid=bond.id, chi=bond.chi;
            const float* mf=h_fwd.data()+bid*chi_max*chi_max;
            const float* mb=h_bwd.data()+bid*chi_max*chi_max;
            float* dst=h_A.data()+bid*chi_max*chi_max;
            for(int i=0;i<chi;i++)
                for(int j=0;j<chi;j++)
                    dst[i+j*chi_max]=0.5f*(mf[i+j*chi_max]+mb[j+i*chi_max]);
        }
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), n_bonds*mat_bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_chi, h_chi.data(), n_bonds*sizeof(int), hipMemcpyHostToDevice));
        t1=_dg_now_ms();
        if(verbose) fprintf(stderr,"[DG] msg P2P+sym+H2D %.1fms\n",t1-t0); t0=t1;

        // ── 2. Per-bond eigendecomposition ─────────────────────────────────
        for(auto& bond:grid.bonds){
            int bid=bond.id, chi=h_chi[bid]; if(chi<=0)continue;
            RS_CHECK(rocsolver_ssyevd(handle, rocblas_evect_original,
                rocblas_fill_upper, chi,
                d_A+bid*(size_t)chi_max*chi_max, chi_max,
                d_D+bid*chi_max, d_E+bid*chi_max, d_info+bid));
        }
        HIP_CHECK(hipDeviceSynchronize());
        t1=_dg_now_ms();
        if(verbose) fprintf(stderr,"[DG] syevd %.1fms\n",t1-t0); t0=t1;

        // ── 3. Build W_a and W_rot ──────────────────────────────────────────
        build_W_kernel<<<n_bonds,256>>>(d_A,d_D,d_W_a,d_Wr,d_chi,chi_max,eig_cutoff,n_bonds);
        HIP_CHECK(hipDeviceSynchronize());
        t1=_dg_now_ms();
        if(verbose) fprintf(stderr,"[DG] build_W %.1fms\n",t1-t0); t0=t1;

        // ── 4. Apply W_a^T to psi_a, compute self-overlap per site ─────────
        const float one=1.f, zero=0.f;
        rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

        for(auto& site:grid.sites){
            int sid=site.id;
            std::vector<int> bids,chis;
            for(int d=0;d<MAX_DEGREE;d++){
                int bid=site.bond_id[d]; if(bid<0)continue;
                bids.push_back(bid); chis.push_back(h_chi[bid]);
            }
            int nb=(int)bids.size(), vol=1;
            for(auto c:chis) vol*=c;
            int totD=vol*D;

            // Allocate gauged psi_a buffer (per site, exact size)
            if(d_gpsi_a[sid]) HIP_CHECK(hipFree(d_gpsi_a[sid]));
            HIP_CHECK(hipMalloc(&d_gpsi_a[sid], totD*sizeof(float)));

            // P2P copy psi
            HIP_CHECK(hipMemcpyPeer(d_gpsi_a[sid], own_dev,
                psi_src[sid], dev_src, totD*sizeof(float)));

            // Apply W_a^T per bond mode (ping-pong scratch)
            float* cur=d_gpsi_a[sid], *nxt=d_scratch1;
            int stride_outer=1, vol_remain=vol;
            for(int k=0;k<nb;k++){
                int chi_k=chis[k]; vol_remain/=chi_k;
                int stride_inner=vol_remain;
                int m=chi_k, n_col=stride_inner*D, kk=chi_k;
                float* Wk=d_W_a+(size_t)bids[k]*chi_max*chi_max;
                for(int outer=0;outer<stride_outer;outer++){
                    int off=outer*chi_k*stride_inner*D;
                    rocblas_sgemm(handle,
                        rocblas_operation_none, rocblas_operation_none,
                        n_col, m, kk, &one,
                        cur+off, n_col, Wk, chi_max,
                        &zero, nxt+off, n_col);
                }
                std::swap(cur, nxt);
                stride_outer*=chi_k;
            }
            // Sync GEMMs then copy result to stable d_gpsi_a[sid]
            HIP_CHECK(hipDeviceSynchronize());
            if(cur != d_gpsi_a[sid])
                HIP_CHECK(hipMemcpy(d_gpsi_a[sid], cur,
                    totD*sizeof(float), hipMemcpyDeviceToDevice));

            // Sync sdot for self-overlap
            float dot_self = 0.f;
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
            rocblas_sdot(handle, totD,
                d_gpsi_a[sid], 1, d_gpsi_a[sid], 1, &dot_self);
            HIP_CHECK(hipDeviceSynchronize());
            double a_self = (double)dot_self;
            h_sign_av[sid] = (a_self>0.0)?1.0:(a_self<0.0)?-1.0:0.0;
            h_log_av[sid]  = (a_self!=0.0)?std::log(std::abs(a_self)):-1e30;
        }
        t1=_dg_now_ms();
        if(verbose) fprintf(stderr,"[DG] psi P2P+GEMM+sdot %.1fms\n",t1-t0);

        HIP_CHECK(hipSetDevice(cur));
    }

    // Self-overlap ⟨psi_a|psi_a⟩ using gauged psi_a (A[i,i] diagonal)
    float self_overlap() {
        int cur; HIP_CHECK(hipGetDevice(&cur));
        HIP_CHECK(hipSetDevice(own_dev));

        double log_num=0.0, sign_num=1.0;
        for(int s=0;s<n_sites;s++){
            if(h_sign_av[s]==0.0){sign_num=0.0;break;}
            sign_num*=h_sign_av[s]; log_num+=h_log_av[s];
        }
        float result=0.f;
        if(sign_num!=0.0){
            // Edge correction: z_e = sum_i |D[i]| (with cutoff)
            std::vector<float> h_D(n_bonds*chi_max);
            HIP_CHECK(hipMemcpy(h_D.data(),d_D,n_bonds*chi_max*sizeof(float),hipMemcpyDeviceToHost));
            double log_den=0.0, sign_den=1.0;
            for(auto& bond:grid.bonds){
                int bid=bond.id, chi=h_chi[bid];
                const float* Dk=h_D.data()+bid*chi_max;
                float max_d=0.f; for(int i=0;i<chi;i++) max_d=std::max(max_d,std::abs(Dk[i]));
                float thresh=eig_cutoff*max_d;
                double ze_p=0.0,ze_n=0.0;
                for(int i=0;i<chi;i++){
                    if(std::abs(Dk[i])<thresh)continue;
                    if(Dk[i]>0.f)ze_p+=Dk[i]; else ze_n-=Dk[i];
                }
                double ze=ze_p-ze_n;
                if(std::abs(ze)>1e-30){sign_den*=(ze>0?1.0:-1.0);log_den+=std::log(std::abs(ze));}
            }
            result=(float)(sign_num*sign_den*std::exp(log_num-log_den));
        }
        HIP_CHECK(hipSetDevice(cur));
        return result;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// DG_Pair: compute ⟨psi_a|psi_b⟩ using precomputed gauge from state_a
// ─────────────────────────────────────────────────────────────────────────────
struct DG_Pair {
    const Grid& grid;
    rocblas_handle handle;
    int n_sites, n_bonds, chi_max, own_dev;

    float* d_tpsi_b   = nullptr;  // W_rot^T @ psi_b (scratch, reused)
    float* d_scratch   = nullptr;  // GEMM ping-pong
    float* d_site_dots = nullptr;  // [n_sites] async sdot
    size_t scratch_size = 0;

    std::vector<double> h_log_av, h_sign_av;

    DG_Pair(const Grid& g, rocblas_handle h, int chi_max_)
        : grid(g), handle(h), n_sites(g.n_sites), n_bonds(g.n_bonds), chi_max(chi_max_)
    {
        HIP_CHECK(hipGetDevice(&own_dev));
        scratch_size=(size_t)chi_max*chi_max*chi_max*D;
        HIP_CHECK(hipMalloc(&d_tpsi_b,   scratch_size*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_scratch,   scratch_size*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_site_dots, n_sites*sizeof(float)));
        h_log_av.assign(n_sites,0.0);
        h_sign_av.assign(n_sites,1.0);
    }

    ~DG_Pair(){
        if(d_tpsi_b)    HIP_CHECK(hipFree(d_tpsi_b));
        if(d_scratch)    HIP_CHECK(hipFree(d_scratch));
        if(d_site_dots)  HIP_CHECK(hipFree(d_site_dots));
    }

    // Apply W_rot^T from gauge_a to psi_b, dot with gauge_a's stored psi_a
    void compute_pair(
        const DG_StateGauge& gauge_a,
        const std::vector<float*>& psi_b_src, int dev_b)
    {
        int cur; HIP_CHECK(hipGetDevice(&cur));
        HIP_CHECK(hipSetDevice(own_dev));
        const float one=1.f, zero=0.f;
        rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

        for(auto& site:grid.sites){
            int sid=site.id;
            std::vector<int> bids,chis;
            for(int d=0;d<MAX_DEGREE;d++){
                int bid=site.bond_id[d]; if(bid<0)continue;
                bids.push_back(bid); chis.push_back(gauge_a.h_chi[bid]);
            }
            int nb=(int)bids.size(), vol=1;
            for(auto c:chis) vol*=c;
            int totD=vol*D;

            // P2P copy psi_b
            HIP_CHECK(hipMemcpyPeer(d_tpsi_b, own_dev,
                psi_b_src[sid], dev_b, totD*sizeof(float)));

            // Apply W_rot^T per bond mode (ping-pong scratch)
            float* cur_b=d_tpsi_b, *nxt_b=d_scratch;
            int stride_outer=1, vol_remain=vol;
            for(int k=0;k<nb;k++){
                int chi_k=chis[k]; vol_remain/=chi_k;
                int stride_inner=vol_remain;
                int m=chi_k, n_col=stride_inner*D, kk=chi_k;
                float* Wk=gauge_a.d_Wr+(size_t)bids[k]*chi_max*chi_max;
                for(int outer=0;outer<stride_outer;outer++){
                    int off=outer*chi_k*stride_inner*D;
                    rocblas_sgemm(handle,
                        rocblas_operation_none, rocblas_operation_none,
                        n_col, m, kk, &one,
                        cur_b+off, n_col, Wk, chi_max,
                        &zero, nxt_b+off, n_col);
                }
                std::swap(cur_b,nxt_b);
                stride_outer*=chi_k;
            }
            // Ensure result is in d_tpsi_b (stable buffer) before sdot
            if (cur_b != d_tpsi_b)
                HIP_CHECK(hipMemcpy(d_tpsi_b, cur_b,
                    totD*sizeof(float), hipMemcpyDeviceToDevice));
            // Synchronous sdot — avoids race on shared scratch buffer
            float dot_val = 0.f;
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
            rocblas_sdot(handle, totD,
                gauge_a.d_gpsi_a[sid], 1, d_tpsi_b, 1, &dot_val);
            HIP_CHECK(hipDeviceSynchronize());
            double a_v = (double)dot_val;
            h_sign_av[sid] = (a_v>0.0)?1.0:(a_v<0.0)?-1.0:0.0;
            h_log_av[sid]  = (a_v!=0.0)?std::log(std::abs(a_v)):-1e30;
        }
        HIP_CHECK(hipSetDevice(cur));
    }

    // Bethe PF using precomputed h_log_av/h_sign_av and gauge_a's D for edge correction
    float bethe(const DG_StateGauge& gauge_a) {
        int cur; HIP_CHECK(hipGetDevice(&cur));
        HIP_CHECK(hipSetDevice(own_dev));

        double log_num=0.0, sign_num=1.0;
        for(int s=0;s<n_sites;s++){
            if(h_sign_av[s]==0.0){sign_num=0.0;break;}
            sign_num*=h_sign_av[s]; log_num+=h_log_av[s];
        }
        float result=0.f;
        if(sign_num!=0.0){
            std::vector<float> h_D(n_bonds*chi_max);
            HIP_CHECK(hipMemcpy(h_D.data(),gauge_a.d_D,
                n_bonds*chi_max*sizeof(float),hipMemcpyDeviceToHost));
            double log_den=0.0, sign_den=1.0;
            for(auto& bond:grid.bonds){
                int bid=bond.id, chi=gauge_a.h_chi[bid];
                const float* Dk=h_D.data()+bid*chi_max;
                float max_d=0.f; for(int i=0;i<chi;i++) max_d=std::max(max_d,std::abs(Dk[i]));
                float thresh=gauge_a.eig_cutoff*max_d;
                double ze_p=0.0,ze_n=0.0;
                for(int i=0;i<chi;i++){
                    if(std::abs(Dk[i])<thresh)continue;
                    if(Dk[i]>0.f)ze_p+=Dk[i]; else ze_n-=Dk[i];
                }
                double ze=ze_p-ze_n;
                if(std::abs(ze)>1e-30){sign_den*=(ze>0?1.0:-1.0);log_den+=std::log(std::abs(ze));}
            }
            result=(float)(sign_num*sign_den*std::exp(log_num-log_den));
        }
        HIP_CHECK(hipSetDevice(cur));
        return result;
    }
};

// Legacy DIAG_GAUGE_IP wrapper for backward compatibility with main_gauge_compare
// Uses the new two-phase design internally
struct DIAG_GAUGE_IP {
    DG_StateGauge gauge_a;
    DG_Pair       pair;
    const Grid&   grid_b_ref;
    std::vector<float*>* psi_b_ptr = nullptr;
    int dev_b_saved = 0;

    DIAG_GAUGE_IP(const Grid& g, rocblas_handle h, int chi_max_, float eig_cut=1e-2f)
        : gauge_a(g, h, chi_max_, eig_cut),
          pair(g, h, chi_max_),
          grid_b_ref(g)
    {}

    void upload(
        const std::vector<float*>& psi_a, int dev_a,
        const std::vector<float*>& msg_a,
        const std::vector<float*>& psi_b, int dev_b,
        const std::vector<float*>& /*msg_b*/)
    {
        gauge_a.precompute(psi_a, msg_a, dev_a);
        // Store psi_b reference for compute()
        psi_b_ptr = const_cast<std::vector<float*>*>(&psi_b);
        dev_b_saved = dev_b;
    }

    float compute() {
        if (!psi_b_ptr) return 0.f;
        pair.compute_pair(gauge_a, *psi_b_ptr, dev_b_saved);
        return pair.bethe(gauge_a);
    }
};
