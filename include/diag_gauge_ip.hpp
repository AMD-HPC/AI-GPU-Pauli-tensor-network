#pragma once
/*
 * diag_gauge_ip.hpp — DG gauge inner product, fully GPU.
 *
 * Algorithm (Julia L-MPF approach, asymmetric gauge):
 *   1. GPU: eigendecompose M_a per bond: M_a = A_a D_a A_a^T
 *           via rocsolver_ssyevd_strided_batched (float32, all 76 bonds at once)
 *   2. GPU: compute W_a[i,j] = sqrt(|D_a[j]|) * A_a[i,j]  (scale columns)
 *           compute W_r[i,j] =             1  * A_a[i,j]  (rotation only, for psi_b)
 *   3. GPU: for each site, apply W_a^T to psi_a and W_r^T to psi_b
 *           via rocblas_sgemm (one GEMM per bond mode per site)
 *   4. GPU: vertex factors a_v = sdot(tpsi_a[v], tpsi_b[v]) per site
 *   5. CPU: log-sum Bethe PF (68 floats — negligible)
 *
 * Edge correction: z_e = sum_i sign(D_a[i]) * |D_a[i]|  (= Tr(M_a) with signs)
 *
 * Why asymmetric (not symmetric):
 *   Absorbing sqrt(D_a)*A_a^T into psi_a and A_a^T into psi_b preserves
 *   the state-a-specific structure in A[i,j] = <psi_i|psi_j>.
 *   Symmetric gauge (same transform both sides) collapses A to near rank-1
 *   when states are similar — Lagrange solve explodes.
 *
 * Cost: O(chi^3 * n_bonds) eigensolve + O(chi^2 * D * n_sites) GEMM — GPU.
 *       Much cheaper than 25 × O(chi^3 * D * n_bonds) QR-gauge BP sweeps.
 *
 * Build: link -lrocsolver -lrocblas
 * Include: rocsolver/rocsolver.h
 */

#include "grid.hpp"
#include "types.hpp"
#include <vector>
#include <cmath>
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
    if(s!=rocblas_status_success){fprintf(stderr,"rocSOLVER error %d at %s:%d\n",(int)s,__FILE__,__LINE__);exit(1);} \
} while(0)

// ── Kernel: scale columns of A by sqrt(|D|), apply eig_cutoff ─────────────────
// W[i,j] = (|D[j]| >= thresh) ? sqrt(|D[j]|) * A[i,j] : 0
// W_rot[i,j] = (|D[j]| >= thresh) ? A[i,j] : 0   (rotation only, no sqrt)
// One block per bond, threads over chi×chi elements.
__global__ void build_W_kernel(
    const float* __restrict__ A,      // [n_bonds, chi_max, chi_max] col-major
    const float* __restrict__ D,      // [n_bonds, chi_max] eigenvalues
    float* __restrict__ W,            // [n_bonds, chi_max, chi_max] sqrt-scaled
    float* __restrict__ W_rot,        // [n_bonds, chi_max, chi_max] rotation only
    const int* __restrict__ chi_per_bond, // actual chi per bond
    int chi_max, float rel_cutoff, int n_bonds)
{
    int bid = blockIdx.x;
    if (bid >= n_bonds) return;
    int chi = chi_per_bond[bid];

    const float* Ak = A    + (size_t)bid * chi_max * chi_max;
    const float* Dk = D    + (size_t)bid * chi_max;
    float*       Wk = W    + (size_t)bid * chi_max * chi_max;
    float*       Rk = W_rot+ (size_t)bid * chi_max * chi_max;

    // Find max |D| for relative cutoff
    float max_d = 0.f;
    for (int j = 0; j < chi; j++) max_d = fmaxf(max_d, fabsf(Dk[j]));
    float thresh = rel_cutoff * max_d;

    for (int idx = threadIdx.x; idx < chi_max * chi_max; idx += blockDim.x) {
        int col = idx / chi_max;  // j
        int row = idx % chi_max;  // i
        if (row >= chi || col >= chi) {
            Wk[idx] = 0.f; Rk[idx] = 0.f; continue;
        }
        float a = Ak[idx];  // A[i,j] col-major: A[row + col*chi_max]
        if (fabsf(Dk[col]) < thresh) {
            Wk[idx] = 0.f; Rk[idx] = 0.f;
        } else {
            float s = sqrtf(fabsf(Dk[col]));
            Wk[idx] = s * a;
            Rk[idx] = a;
        }
    }
}

// ── DIAG_GAUGE_IP ─────────────────────────────────────────────────────────────
struct DIAG_GAUGE_IP {
    const Grid& grid;
    rocblas_handle handle;
    int n_sites, n_bonds, chi_max;
    float eig_cutoff;  // relative: truncate |D[i]| < eig_cutoff * max|D|

    // ── GPU buffers (on current device when constructed) ───────────────────────
    // Strided batch layout: [n_bonds, chi_max, chi_max] contiguous
    float* d_A_a   = nullptr;  // message matrices for state_a (input/output of syevd)
    float* d_D_a   = nullptr;  // eigenvalues [n_bonds, chi_max]
    float* d_E_a   = nullptr;  // syevd workspace [n_bonds, chi_max]
    float* d_W_a   = nullptr;  // sqrt(|D|)*A columns (for psi_a)
    float* d_W_rot = nullptr;  // A columns only (for psi_b)
    int*   d_info  = nullptr;  // syevd info [n_bonds]
    int*   d_chi   = nullptr;  // chi per bond [n_bonds]

    // Gauged psi tensors — allocated per-site on demand
    std::vector<float*> d_tpsi_a, d_tpsi_b;  // [n_sites]

    // Host chi values
    std::vector<int> h_chi;

    DIAG_GAUGE_IP(const Grid& g, rocblas_handle h, int chi_max_, float eig_cut=1e-2f)
        : grid(g), handle(h),
          n_sites(g.n_sites), n_bonds(g.n_bonds),
          chi_max(chi_max_), eig_cutoff(eig_cut)
    {
        HIP_CHECK(hipGetDevice(&own_dev));  // record device at construction time
        size_t mat_sz = (size_t)n_bonds * chi_max * chi_max;
        size_t vec_sz = (size_t)n_bonds * chi_max;
        HIP_CHECK(hipMalloc(&d_A_a,   mat_sz * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_D_a,   vec_sz * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_E_a,   vec_sz * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_W_a,   mat_sz * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_W_rot, mat_sz * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_info,  n_bonds * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_chi,   n_bonds * sizeof(int)));
        d_tpsi_a.resize(n_sites, nullptr);
        d_tpsi_b.resize(n_sites, nullptr);
        h_chi.resize(n_bonds);
    }

    ~DIAG_GAUGE_IP() {
        if (d_A_a)   HIP_CHECK(hipFree(d_A_a));
        if (d_D_a)   HIP_CHECK(hipFree(d_D_a));
        if (d_E_a)   HIP_CHECK(hipFree(d_E_a));
        if (d_W_a)   HIP_CHECK(hipFree(d_W_a));
        if (d_W_rot) HIP_CHECK(hipFree(d_W_rot));
        if (d_info)  HIP_CHECK(hipFree(d_info));
        if (d_chi)   HIP_CHECK(hipFree(d_chi));
        for (auto p : d_tpsi_a) if (p) HIP_CHECK(hipFree(p));
        for (auto p : d_tpsi_b) if (p) HIP_CHECK(hipFree(p));
    }

    // ── Upload: copy messages from source device, eigendecompose, build W ─────
    // Device on which this object was constructed (owns all GPU buffers)
    int own_dev = 0;

    void upload(
        const std::vector<float*>& psi_a_src, int dev_a,
        const std::vector<float*>& msg_a_src,   // state_a messages (on dev_a)
        const std::vector<float*>& psi_b_src, int dev_b,
        const std::vector<float*>& /*msg_b_src*/)  // state_b messages (unused)
    {
        int cur; HIP_CHECK(hipGetDevice(&cur));
        HIP_CHECK(hipSetDevice(own_dev));  // all ops on our device

        // ── Step 1: copy message matrices to d_A_a, symmetrize ────────────────
        // Use average of msg_from_a and msg_from_b for symmetry
        HIP_CHECK(hipMemset(d_A_a, 0, (size_t)n_bonds*chi_max*chi_max*sizeof(float)));
        for (auto& bond : grid.bonds) {
            int bid = bond.id;
            int chi = bond.chi;
            h_chi[bid] = chi;

            // Download fwd and bwd messages from dev_a via host
            std::vector<float> mf(chi_max*chi_max), mb(chi_max*chi_max);
            HIP_CHECK(hipSetDevice(dev_a));
            HIP_CHECK(hipMemcpy(mf.data(), msg_a_src[bond.msg_from_a()],
                chi_max*chi_max*sizeof(float), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(mb.data(), msg_a_src[bond.msg_from_b()],
                chi_max*chi_max*sizeof(float), hipMemcpyDeviceToHost));
            HIP_CHECK(hipSetDevice(own_dev));

            // Symmetrize: A[i,j] = 0.5*(mf[i,j]+mb[j,i]) col-major
            std::vector<float> sym(chi_max*chi_max, 0.f);
            for (int i = 0; i < chi; i++)
                for (int j = 0; j < chi; j++)
                    sym[i + j*chi_max] = 0.5f*(mf[i + j*chi_max] + mb[j + i*chi_max]);

            // Upload to strided slot [bid, :, :]
            HIP_CHECK(hipMemcpy(d_A_a + (size_t)bid*chi_max*chi_max,
                sym.data(), chi_max*chi_max*sizeof(float), hipMemcpyHostToDevice));
        }

        // Upload chi per bond
        HIP_CHECK(hipMemcpy(d_chi, h_chi.data(), n_bonds*sizeof(int), hipMemcpyHostToDevice));

        // ── Step 2: per-bond eigendecomposition on GPU ────────────────────────
        // Use individual rocsolver_ssyevd calls with actual chi per bond.
        // Batched version requires same n — unsafe when chi varies (1 at step 1).
        for (auto& bond : grid.bonds) {
            int bid = bond.id;
            int chi = h_chi[bid];
            if (chi <= 0) continue;
            RS_CHECK(rocsolver_ssyevd(
                handle,
                rocblas_evect_original,
                rocblas_fill_upper,
                chi,                                          // actual chi
                d_A_a + (size_t)bid*chi_max*chi_max,         // A slot
                chi_max,                                      // lda = chi_max (padded)
                d_D_a + (size_t)bid*chi_max,                 // D slot
                d_E_a + (size_t)bid*chi_max,                 // E workspace
                d_info + bid));
        }

        // Synchronize and check info
        HIP_CHECK(hipDeviceSynchronize());
        {
            std::vector<int> h_info_check(n_bonds);
            HIP_CHECK(hipMemcpy(h_info_check.data(), d_info,
                n_bonds*sizeof(int), hipMemcpyDeviceToHost));
            for (int b=0; b<n_bonds; b++)
                if (h_info_check[b] != 0)
                    fprintf(stderr,"[DG] syevd bond %d info=%d chi=%d\n",
                            b, h_info_check[b], h_chi[b]);
        }

        // ── Step 3: build W_a and W_rot on GPU ───────────────────────────────
        build_W_kernel<<<n_bonds, 256>>>(
            d_A_a, d_D_a, d_W_a, d_W_rot,
            d_chi, chi_max, eig_cutoff, n_bonds);
        HIP_CHECK(hipDeviceSynchronize());

        // ── Step 4: apply gauge to psi_a and psi_b ───────────────────────────
        // For each site: tpsi_a = W_a^T @ psi_a, tpsi_b = W_rot^T @ psi_b
        // W_a^T [chi, chi_max] @ psi_a [chi_max, D*outer] = tpsi_a [chi, D*outer]
        // Applied sequentially per bond mode (same as Trotter gate application).
        const float one=1.f, zero=0.f;

        for (auto& site : grid.sites) {
            int sid = site.id;
            std::vector<int> bids, chis;
            for (int d = 0; d < MAX_DEGREE; d++) {
                int bid = site.bond_id[d]; if (bid < 0) continue;
                bids.push_back(bid);
                chis.push_back(h_chi[bid]);
            }
            int nb  = (int)bids.size();
            int vol = 1; for (auto c : chis) vol *= c;
            int totD = vol * D;

            // Free old tpsi buffers, allocate fresh
            if (d_tpsi_a[sid]) { HIP_CHECK(hipFree(d_tpsi_a[sid])); d_tpsi_a[sid]=nullptr; }
            if (d_tpsi_b[sid]) { HIP_CHECK(hipFree(d_tpsi_b[sid])); d_tpsi_b[sid]=nullptr; }
            HIP_CHECK(hipMalloc(&d_tpsi_a[sid], totD*sizeof(float)));
            HIP_CHECK(hipMalloc(&d_tpsi_b[sid], totD*sizeof(float)));

            // Copy psi_a and psi_b to own_dev via host
            std::vector<float> bufa(totD), bufb(totD);
            HIP_CHECK(hipSetDevice(dev_a));
            HIP_CHECK(hipMemcpy(bufa.data(), psi_a_src[sid],
                totD*sizeof(float), hipMemcpyDeviceToHost));
            HIP_CHECK(hipSetDevice(dev_b));
            HIP_CHECK(hipMemcpy(bufb.data(), psi_b_src[sid],
                totD*sizeof(float), hipMemcpyDeviceToHost));
            HIP_CHECK(hipSetDevice(own_dev));
            HIP_CHECK(hipMemcpy(d_tpsi_a[sid], bufa.data(),
                totD*sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_tpsi_b[sid], bufb.data(),
                totD*sizeof(float), hipMemcpyHostToDevice));

            // Apply W^T per bond mode using SGEMM
            // tpsi layout: [i_0, ..., i_{nb-1}, d] row-major
            // For bond k: new_tpsi[..., j_k, ..., d] = sum_i W[i,j] * tpsi[..., i_k, ..., d]
            //           = W^T [j,i] contracted with mode k of tpsi
            // SGEMM: treat outer*inner*D as batch/columns, chi as the contracted index
            // This is a mode-k multiplication: each slice is a GEMM W^T @ slice

            // Apply W^T per bond mode using SGEMM — separate scratch for a and b
            float* scratch_a = nullptr, *scratch_b = nullptr;
            HIP_CHECK(hipMalloc(&scratch_a, totD*sizeof(float)));
            HIP_CHECK(hipMalloc(&scratch_b, totD*sizeof(float)));

            int stride_outer = 1;
            int vol_remain   = vol;
            for (int k = 0; k < nb; k++) {
                int chi_k = chis[k];
                vol_remain /= chi_k;
                int stride_inner = vol_remain;

                int m     = chi_k;
                int n_col = stride_inner * D;
                int kk    = chi_k;

                float* Wk_a = d_W_a   + (size_t)bids[k]*chi_max*chi_max;
                float* Wk_r = d_W_rot + (size_t)bids[k]*chi_max*chi_max;

                for (int outer = 0; outer < stride_outer; outer++) {
                    int offset = outer * chi_k * stride_inner * D;
                    // Row-major: new_psi[m=chi_k, n_col] = W^T[chi_k,chi_k] @ psi[chi_k, n_col]
                    // rocBLAS col-major: C^T[n_col,m] = psi^T[n_col,kk] @ W[kk,m]
                    // same flat layout: scratch[r + j*n_col] = new_psi[j*n_col + r]
                    rocblas_sgemm(handle,
                        rocblas_operation_none, rocblas_operation_none,
                        n_col, m, kk, &one,
                        d_tpsi_a[sid]+offset, n_col,  // psi [n_col, kk] lda=n_col
                        Wk_a, chi_max,                 // W   [kk,  m]   ldb=chi_max
                        &zero, scratch_a+offset, n_col); // result [n_col, m] ldc=n_col
                    rocblas_sgemm(handle,
                        rocblas_operation_none, rocblas_operation_none,
                        n_col, m, kk, &one,
                        d_tpsi_b[sid]+offset, n_col,
                        Wk_r, chi_max,
                        &zero, scratch_b+offset, n_col);
                }

                // Swap: tpsi ← scratch for next bond mode
                std::swap(d_tpsi_a[sid], scratch_a);
                std::swap(d_tpsi_b[sid], scratch_b);
                stride_outer *= chi_k;
            }
            HIP_CHECK(hipDeviceSynchronize());  // wait for all GEMMs
            HIP_CHECK(hipFree(scratch_a));
            HIP_CHECK(hipFree(scratch_b));
        }
        HIP_CHECK(hipSetDevice(cur));  // restore caller's device
    }

    // ── Compute: Bethe PF ─────────────────────────────────────────────────────
    float compute(float /*tol*/=1e-5f, bool /*verbose*/=false)
    {
        int cur; HIP_CHECK(hipGetDevice(&cur));
        HIP_CHECK(hipSetDevice(own_dev));

        // Vertex factors: a_v = sdot(tpsi_a[v], tpsi_b[v])
        double log_num=0.0, sign_num=1.0;
        const float one=1.f;
        for (auto& site : grid.sites) {
            int sid = site.id;
            int vol = 1;
            for (int d=0;d<MAX_DEGREE;d++) {
                int bid=site.bond_id[d]; if(bid<0) continue;
                vol *= h_chi[bid];
            }
            int totD = vol * D;

            // sdot on GPU
            float dot_val = 0.f;
            // Use host pointer mode — result written directly to host float
            rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host);
            rocblas_sdot(handle, totD, d_tpsi_a[sid], 1, d_tpsi_b[sid], 1, &dot_val);
            HIP_CHECK(hipDeviceSynchronize());  // wait for sdot to complete

            double a_v = (double)dot_val;
            if (a_v==0.0){sign_num=0.0;break;}
            sign_num *= (a_v>0.0)?1.0:-1.0;
            log_num  += std::log(std::abs(a_v));
        }
        if (sign_num==0.0) return 0.f;

        // Edge correction
        std::vector<float> h_D(n_bonds*chi_max);
        HIP_CHECK(hipMemcpy(h_D.data(), d_D_a,
            n_bonds*chi_max*sizeof(float), hipMemcpyDeviceToHost));

        double log_den=0.0, sign_den=1.0;
        for (auto& bond : grid.bonds) {
            int bid=bond.id, chi=h_chi[bid];
            const float* Dk = h_D.data() + bid*chi_max;
            float max_d=0.f;
            for(int i=0;i<chi;i++) max_d=std::max(max_d,std::abs(Dk[i]));
            float thresh=eig_cutoff*max_d;
            double ze_pos=0.0, ze_neg=0.0;
            for(int i=0;i<chi;i++){
                if(std::abs(Dk[i])<thresh) continue;
                if(Dk[i]>0.f) ze_pos+=Dk[i];
                else          ze_neg-=Dk[i];
            }
            double ze=ze_pos-ze_neg;
            if(std::abs(ze)>1e-30){
                sign_den*=(ze>0.0)?1.0:-1.0;
                log_den+=std::log(std::abs(ze));
            }
        }

        HIP_CHECK(hipSetDevice(cur));  // restore caller's device
        return (float)(sign_num*sign_den*std::exp(log_num-log_den));
    }
};
