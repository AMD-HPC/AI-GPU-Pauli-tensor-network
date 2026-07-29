/*
 * test_blf_sgemm.cpp — Validate GPU SGEMM for degree-2 message update.
 *
 * Tests step 1 of the message update:
 *   inter[ca_skip, cb1, D] = sum_{ca1} psi_a[ca_skip, ca1, D] * M[ca1, cb1]
 *
 * GPU approach: rocblas_sgemm_strided_batched with batch=D
 *   For each d: inter[:,jb1,d] = psi_a[:,:,d] @ M
 *   Transpose psi_a to [D, ca_skip, ca1] so each batch is contiguous.
 *
 * Then step 2 on CPU to validate the full result matches BLF2 CPU.
 *
 * Build: make PHYS_DIM=4 test_blf_sgemm
 * Run:   ./test_blf_sgemm [chi]
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do { \
    hipError_t e=(cmd); \
    if(e!=hipSuccess){fprintf(stderr,"HIP %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__);exit(1);} \
} while(0)
#endif

static constexpr int D = 4;

// CPU reference (from BLF2)
// psi_a[ca_skip,ca1,D], M[ca1,cb1], psi_b[cb_skip,cb1,D]
std::vector<float> cpu_msg(
    const float* psi_a, int ca_skip, int ca1,
    const float* M, int cb1,
    const float* psi_b, int cb_skip)
{
    std::vector<float> inter(ca_skip*cb1*D, 0.f);
    for(int is=0;is<ca_skip;is++)
    for(int i1=0;i1<ca1;i1++)
    for(int jb1=0;jb1<cb1;jb1++)
    for(int d=0;d<D;d++)
        inter[(is*cb1+jb1)*D+d] += psi_a[(is*ca1+i1)*D+d]*M[i1*cb1+jb1];

    std::vector<float> nm(ca_skip*cb_skip, 0.f);
    for(int is=0;is<ca_skip;is++)
    for(int ib_s=0;ib_s<cb_skip;ib_s++)
    for(int jb1=0;jb1<cb1;jb1++)
    for(int d=0;d<D;d++)
        nm[is*cb_skip+ib_s] += inter[(is*cb1+jb1)*D+d]*psi_b[(ib_s*cb1+jb1)*D+d];
    return nm;
}

// GPU step 1: inter[ca_skip, cb1, D] = psi_a[ca_skip,ca1,D] @ M[ca1,cb1] (over ca1)
// Uses strided_batched SGEMM with batch=D.
std::vector<float> gpu_step1(
    rocblas_handle handle,
    const float* psi_a_h, int ca_skip, int ca1,
    const float* M_h, int cb1)
{
    // Transpose psi_a[ca_skip,ca1,D] → psi_a_T[D, ca_skip, ca1]
    // so batch d is at d*ca_skip*ca1, contiguous [ca_skip,ca1]
    std::vector<float> psi_a_T(D*ca_skip*ca1);
    for(int is=0;is<ca_skip;is++)
    for(int i1=0;i1<ca1;i1++)
    for(int d=0;d<D;d++)
        psi_a_T[d*ca_skip*ca1 + is*ca1 + i1] = psi_a_h[(is*ca1+i1)*D+d];

    float *d_A, *d_M, *d_C;
    HIP_CHECK(hipMalloc(&d_A, D*ca_skip*ca1*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_M, ca1*cb1*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_C, D*ca_skip*cb1*sizeof(float)));
    HIP_CHECK(hipMemcpy(d_A, psi_a_T.data(), D*ca_skip*ca1*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_M, M_h, ca1*cb1*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C, 0, D*ca_skip*cb1*sizeof(float)));

    // For each batch d:
    //   C_d[ca_skip, cb1] = A_d[ca_skip, ca1] @ M[ca1, cb1]   (all row-major)
    //
    // rocBLAS is col-major. For row-major A@B=C:
    //   Equivalent col-major: C^T = B^T @ A^T
    //   sgemm(op_none, op_none, m=cb1, n=ca_skip, k=ca1,
    //         B_colmaj_ptr, ldb=cb1,   ← M row-major [ca1,cb1]: col-major [cb1,ca1], lda=cb1
    //         A_colmaj_ptr, lda=ca1,   ← A row-major [ca_skip,ca1]: col-major [ca1,ca_skip], lda=ca1
    //         C_colmaj_ptr, ldc=cb1)   ← C row-major [ca_skip,cb1]: col-major [cb1,ca_skip], lda=cb1
    //   Result stored as C_colmaj[cb1,ca_skip] = C_rowmaj[ca_skip,cb1] ✓
    float alpha=1.f, beta=0.f;
    rocblas_sgemm_strided_batched(handle,
        rocblas_operation_none, rocblas_operation_none,
        cb1,        // m
        ca_skip,    // n
        ca1,        // k
        &alpha,
        d_M, cb1, 0,                    // B=M, ldb=cb1, no stride (same M each batch)
        d_A, ca1, ca_skip*ca1,          // A=psi_a_T batch d, lda=ca1, stride=ca_skip*ca1
        &beta,
        d_C, cb1, ca_skip*cb1,          // C batch d, ldc=cb1, stride=ca_skip*cb1
        D);
    HIP_CHECK(hipDeviceSynchronize());

    // Download C[D, ca_skip, cb1] col-major → rearrange to inter[ca_skip, cb1, D]
    // C batch d stored as col-major [cb1, ca_skip] with ldc=cb1:
    //   C_d[jb1, is] at addr d*ca_skip*cb1 + jb1 + is*cb1 = d*ca_skip*cb1 + is*cb1 + jb1
    // Wait: col-major [cb1,ca_skip] means element (row=jb1, col=is) at jb1 + is*cb1
    // So: C_ptr[d*ca_skip*cb1 + is*cb1 + jb1] = C_d[jb1,is] = C_row_d[is,jb1] = inter[is,jb1,d] ✓
    std::vector<float> C_gpu(D*ca_skip*cb1);
    HIP_CHECK(hipMemcpy(C_gpu.data(), d_C, D*ca_skip*cb1*sizeof(float), hipMemcpyDeviceToHost));

    std::vector<float> inter(ca_skip*cb1*D);
    for(int d=0;d<D;d++)
    for(int is=0;is<ca_skip;is++)
    for(int jb1=0;jb1<cb1;jb1++)
        inter[(is*cb1+jb1)*D+d] = C_gpu[d*ca_skip*cb1 + is*cb1 + jb1];

    HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_M)); HIP_CHECK(hipFree(d_C));
    return inter;
}

int main(int argc, char** argv) {
    int chi = argc>1 ? atoi(argv[1]) : 4;
    printf("=== SGEMM step-1 test chi=%d ===\n", chi);

    rocblas_handle handle; rocblas_create_handle(&handle);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.f,1.f);

    int ca_skip=chi, ca1=chi, cb1=chi, cb_skip=chi;

    std::vector<float> psi_a(ca_skip*ca1*D), M(ca1*cb1), psi_b(cb_skip*cb1*D);
    for(auto& v:psi_a) v=dist(rng);
    for(auto& v:M)     v=dist(rng);
    for(auto& v:psi_b) v=dist(rng);

    // CPU step 1 (from BLF2)
    std::vector<float> inter_cpu(ca_skip*cb1*D, 0.f);
    for(int is=0;is<ca_skip;is++)
    for(int i1=0;i1<ca1;i1++)
    for(int jb1=0;jb1<cb1;jb1++)
    for(int d=0;d<D;d++)
        inter_cpu[(is*cb1+jb1)*D+d] += psi_a[(is*ca1+i1)*D+d]*M[i1*cb1+jb1];

    // GPU step 1
    auto inter_gpu = gpu_step1(handle, psi_a.data(), ca_skip, ca1, M.data(), cb1);

    float max_err=0.f, max_val=0.f;
    for(int i=0;i<ca_skip*cb1*D;i++){
        max_err=std::max(max_err,std::abs(inter_cpu[i]-inter_gpu[i]));
        max_val=std::max(max_val,std::abs(inter_cpu[i]));
    }
    printf("Step 1: max_err=%.2e max_val=%.2e rel=%.2e  %s\n",
           max_err, max_val, max_err/(max_val+1e-30f),
           max_err < 1e-4f*(max_val+1e-30f) ? "PASS" : "FAIL");

    auto nm_cpu = cpu_msg(psi_a.data(),ca_skip,ca1,M.data(),cb1,psi_b.data(),cb_skip);

    // Step 2 GPU:
    // nm[ca_skip, cb_skip] = sum_{cb1,D} inter[ca_skip,cb1,D] * psi_b[cb_skip,cb1,D]
    // Transpose inter → inter_T[D, ca_skip, cb1] and psi_b → psi_b_T[D, cb_skip, cb1]
    // For each d: nm_d[ca_skip,cb_skip] += inter_T_d[ca_skip,cb1] @ psi_b_T_d[cb_skip,cb1]^T
    // Same col-major trick as step 1:
    //   nm_d[ca_skip,cb_skip] = inter_d @ psi_b_d^T
    //   col-major: nm_d_col[cb_skip,ca_skip] = psi_b_d_col[cb1,cb_skip]^T^T @ inter_d_col[cb1,ca_skip]^T
    //   = sgemm(op_none, op_none, m=cb_skip, n=ca_skip, k=cb1,
    //           psi_b_d_col, lda=cb_skip,  ← psi_b row[cb_skip,cb1] → col[cb1,cb_skip] lda=cb_skip? NO.
    //   psi_b_T_d[cb_skip,cb1] row-major: col-major [cb1,cb_skip] lda=cb1
    //   But we want psi_b^T contracted with inter → different sgemm:
    //   nm_d = inter_d[ca_skip,cb1] @ psi_b_d[cb_skip,cb1]^T
    //   col-major: nm_d_col[cb_skip,ca_skip] = psi_b_d_col[cb_skip,cb1]^T @ inter_d_col[ca_skip,cb1]^T^T
    //   = psi_b_d_col^T @ inter_d_col  where psi_b_d_col[cb1,cb_skip] and inter_d_col[cb1,ca_skip]
    //   Hmm — psi_b row[cb_skip,cb1] as col = [cb1,cb_skip] lda=cb1 → this is psi_b^T in row-major
    //   inter row[ca_skip,cb1] as col = [cb1,ca_skip] lda=cb1
    //   sgemm(op_T, op_none, cb_skip, ca_skip, cb1,
    //         psi_b_col, lda=cb1,   ← psi_b_col[cb1,cb_skip], op_T gives [cb_skip,cb1] → psi_b_row
    //         inter_col, lda=cb1,   ← inter_col[cb1,ca_skip], op_none
    //         nm_col, ldc=cb_skip)  ← nm_col[cb_skip,ca_skip] = nm_row[ca_skip,cb_skip] ✓

    // Transpose inter and psi_b to [D, *, *] layout for strided batched
    std::vector<float> inter_T2(D*ca_skip*cb1), psi_b_T(D*cb_skip*cb1);
    for(int d=0;d<D;d++)
    for(int is=0;is<ca_skip;is++)
    for(int jb1=0;jb1<cb1;jb1++)
        inter_T2[d*ca_skip*cb1 + is*cb1 + jb1] = inter_gpu[(is*cb1+jb1)*D+d];
    for(int d=0;d<D;d++)
    for(int ib_s=0;ib_s<cb_skip;ib_s++)
    for(int jb1=0;jb1<cb1;jb1++)
        psi_b_T[d*cb_skip*cb1 + ib_s*cb1 + jb1] = psi_b[(ib_s*cb1+jb1)*D+d];

    float *d_inter2, *d_psi_b2, *d_nm;
    HIP_CHECK(hipMalloc(&d_inter2, D*ca_skip*cb1*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_psi_b2, D*cb_skip*cb1*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_nm,     ca_skip*cb_skip*sizeof(float)));
    HIP_CHECK(hipMemcpy(d_inter2, inter_T2.data(), D*ca_skip*cb1*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_psi_b2, psi_b_T.data(),  D*cb_skip*cb1*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_nm, 0, ca_skip*cb_skip*sizeof(float)));

    float alpha=1.f, beta_acc=1.f;  // accumulate across D batches
    // First batch with beta=0, rest with beta=1
    float alpha1=1.f, beta0=0.f, beta1=1.f;
    // Use strided_batched with beta=alpha for accumulation — but rocBLAS doesn't support
    // varying beta per batch. Instead: run D separate sgemm calls accumulating.
    for(int d=0;d<D;d++){
        rocblas_sgemm(handle,
            rocblas_operation_transpose, rocblas_operation_none,
            cb_skip, ca_skip, cb1,
            &alpha1,
            d_psi_b2 + d*cb_skip*cb1, cb1,   // psi_b_T_d[cb_skip,cb1] row-major → col[cb1,cb_skip] lda=cb1, op_T→[cb_skip,cb1]
            d_inter2  + d*ca_skip*cb1, cb1,   // inter_T_d[ca_skip,cb1] row-major → col[cb1,ca_skip] lda=cb1, op_N
            d==0 ? &beta0 : &beta1,
            d_nm, cb_skip);                   // nm_col[cb_skip,ca_skip] ldc=cb_skip
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Download nm_col[cb_skip,ca_skip] → nm_row[ca_skip,cb_skip]
    std::vector<float> nm_col(cb_skip*ca_skip);
    HIP_CHECK(hipMemcpy(nm_col.data(), d_nm, cb_skip*ca_skip*sizeof(float), hipMemcpyDeviceToHost));
    std::vector<float> nm_gpu_full(ca_skip*cb_skip);
    // nm_col is [cb_skip, ca_skip] col-major: element [ib_s, is] at ib_s + is*cb_skip
    for(int is=0;is<ca_skip;is++)
    for(int ib_s=0;ib_s<cb_skip;ib_s++)
        nm_gpu_full[is*cb_skip+ib_s] = nm_col[ib_s + is*cb_skip];

    HIP_CHECK(hipFree(d_inter2)); HIP_CHECK(hipFree(d_psi_b2)); HIP_CHECK(hipFree(d_nm));

    max_err=0.f; max_val=0.f;
    for(int i=0;i<ca_skip*cb_skip;i++){
        max_err=std::max(max_err,std::abs(nm_cpu[i]-nm_gpu_full[i]));
        max_val=std::max(max_val,std::abs(nm_cpu[i]));
    }
    printf("Step 2: max_err=%.2e max_val=%.2e rel=%.2e  %s\n",
           max_err, max_val, max_err/(max_val+1e-30f),
           max_err < 1e-4f*(max_val+1e-30f) ? "PASS" : "FAIL");
    printf("CPU nm[0,0]=%.6f  GPU nm[0,0]=%.6f\n", nm_cpu[0], nm_gpu_full[0]);

    rocblas_destroy_handle(handle);
    return 0;
}
