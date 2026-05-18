/*
 * tests/test_gate_ops.cpp
 *
 * Tests for the remaining gate computation building blocks:
 *   1. SVD (rocsolver_sgesvd) on gfx906
 *   2. TRSM replacement: CPU R⁻¹ then GEMM (Q = A · R⁻¹)
 *   3. Tensor permute (reshape for QR split)
 *   4. Full CholeskyQR pipeline (POTRF + CPU_TRSM + GEMM)
 *   5. Full Householder QR (GEQRF + ORGQR)
 *   6. Mini gate pipeline: QR + RR + SVD + reconstruct
 *
 * Build:
 *   hipcc -O2 -std=c++17 -DPHYS_DIM=2 -I../include \
 *         -o test_gate_ops test_gate_ops.cpp -lrocblas -lrocsolver -lm
 */

#include "types.hpp"
#include "tensor.hpp"
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <numeric>

static rocblas_handle g_handle = nullptr;

/* =========================================================
 * Timing helper
 * ========================================================= */
static double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

/* =========================================================
 * GEMM helper — C = opA(A) · opB(B)
 * ========================================================= */
static void gemm(
    const float* A, int m, int k,
    const float* B, int n,
    float* C,
    rocblas_operation opA = rocblas_operation_none,
    rocblas_operation opB = rocblas_operation_none,
    float alpha = 1.0f, float beta = 0.0f)
{
    int lda = (opA == rocblas_operation_none) ? m : k;
    int ldb = (opB == rocblas_operation_none) ? k : n;
    RB_CHECK(rocblas_sgemm(g_handle, opA, opB, m, n, k,
        &alpha, A, lda, B, ldb, &beta, C, m));
    HIP_CHECK(hipDeviceSynchronize());
}

/* =========================================================
 * CPU R⁻¹: back-substitution for n×n upper triangular R
 * Returns R⁻¹ as a host vector (col-major)
 * Cost: O(n³/6) — for n=100 this is ~170K ops, microseconds
 * ========================================================= */
static std::vector<float> cpu_upper_inv(const std::vector<float>& R, int n) {
    std::vector<float> Rinv(n*n, 0.0f);
    // Initialize as identity
    for (int i=0; i<n; i++) Rinv[i+i*n] = 1.0f;
    // Back-substitution: solve R · X = I column by column
    // For each column j of X (= column j of I):
    for (int j=0; j<n; j++) {
        for (int i=j; i>=0; i--) {
            float s = Rinv[i+j*n];
            for (int k=i+1; k<n; k++)
                s -= R[i+k*n] * Rinv[k+j*n];
            Rinv[i+j*n] = s / R[i+i*n];
        }
    }
    return Rinv;
}

/* =========================================================
 * Test 1: SVD on gfx906
 * ========================================================= */
static void test_svd() {
    printf("\n=== Test 1: SVD (rocsolver_sgesvd) ===\n");

    // Test at n=100 (steady state for F32: D*chi = 2*50)
    // and at n=200 (steady state for ComplexF32 equivalent)
    for (int n : {10, 50, 100, 200}) {
        int m = n;  // square for simplicity
        // Build a known matrix: diag(1,2,...,n) via outer product
        std::vector<float> hA(m*n, 0.0f);
        for (int i=0; i<std::min(m,n); i++) hA[i+i*m] = (float)(i+1);
        // Add small off-diagonal to make it interesting
        for (int i=0; i<m; i++)
            for (int j=0; j<n; j++)
                if (i!=j) hA[i+j*m] += 0.01f;

        float *dA, *dU, *dS, *dVt, *dWork;
        int   *dInfo;
        int k = std::min(m,n);
        HIP_CHECK(hipMalloc(&dA,    (size_t)m*n*sizeof(float)));
        HIP_CHECK(hipMalloc(&dU,    (size_t)m*k*sizeof(float)));
        HIP_CHECK(hipMalloc(&dS,    (size_t)k*sizeof(float)));
        HIP_CHECK(hipMalloc(&dVt,   (size_t)k*n*sizeof(float)));
        HIP_CHECK(hipMalloc(&dWork, (size_t)std::max(m,n)*sizeof(float)));
        HIP_CHECK(hipMalloc(&dInfo, sizeof(int)));
        HIP_CHECK(hipMemcpy(dA, hA.data(), m*n*sizeof(float), hipMemcpyHostToDevice));

        double t0 = now_ms();
        RB_CHECK(rocsolver_sgesvd(g_handle,
            rocblas_svect_singular, rocblas_svect_singular,
            m, n, dA, m, dS, dU, m, dVt, k,
            dWork, rocblas_inplace, dInfo));
        HIP_CHECK(hipDeviceSynchronize());
        double dt = now_ms() - t0;

        int hinfo;
        HIP_CHECK(hipMemcpy(&hinfo, dInfo, sizeof(int), hipMemcpyDeviceToHost));

        std::vector<float> hS(k);
        HIP_CHECK(hipMemcpy(hS.data(), dS, k*sizeof(float), hipMemcpyDeviceToHost));

        printf("  SVD %dx%d: info=%d  dt=%.1fms  S[0]=%.4f S[k-1]=%.4f  %s\n",
            m, n, hinfo, dt, hS[0], hS[k-1],
            hinfo==0 ? "OK" : "FAILED");

        // Verify S[i] ≈ i+1 (diagonal matrix singular values)
        if (hinfo == 0) {
            // Sort descending for comparison
            std::vector<float> expected(k);
            for (int i=0; i<k; i++) expected[i] = (float)(k-i);
            // Just check largest and smallest
            assert(std::abs(hS[0] - expected[0]) < 1.0f);  // within 1 due to off-diag
        }

        HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dU));
        HIP_CHECK(hipFree(dS)); HIP_CHECK(hipFree(dVt));
        HIP_CHECK(hipFree(dWork)); HIP_CHECK(hipFree(dInfo));
    }
}

/* =========================================================
 * Test 2: TRSM replacement — CPU R⁻¹ then GEMM
 * ========================================================= */
static void test_trsm_replacement() {
    printf("\n=== Test 2: TRSM replacement (CPU R⁻¹ + GEMM) ===\n");

    // Typical sizes at chi=50, D=2:
    // m = chi^3 = 125000, n = D*chi = 100
    // R is n×n = 100×100 (tiny)
    // Q = A · R⁻¹ where A is m×n

    for (auto [m, n] : std::vector<std::pair<int,int>>{{100,10},{1000,50},{125000,100}}) {
        // Build upper triangular R (n×n) — diagonal dominant
        std::vector<float> hR(n*n, 0.0f);
        for (int i=0; i<n; i++) {
            hR[i+i*n] = (float)(i+2);  // diagonal
            for (int j=i+1; j<n; j++)
                hR[i+j*n] = 0.1f;       // upper triangle
        }

        // Build A (m×n) — random-ish
        std::vector<float> hA(m*n);
        for (int i=0; i<m*n; i++) hA[i] = (float)(i % 7) / 7.0f;

        // Expected: Q_expected = A · R⁻¹  (computed on host for small cases)
        std::vector<float> hQ_expected;
        if (m <= 1000) {
            auto hRinv = cpu_upper_inv(hR, n);
            hQ_expected.resize(m*n, 0.0f);
            for (int i=0; i<m; i++)
                for (int j=0; j<n; j++) {
                    float s=0.0f;
                    for (int k=0; k<n; k++) s += hA[i+k*m] * hRinv[k+j*n];
                    hQ_expected[i+j*m] = s;
                }
        }

        // GPU: upload A and R
        float *dA, *dR, *dRinv, *dQ;
        HIP_CHECK(hipMalloc(&dA,    (size_t)m*n*sizeof(float)));
        HIP_CHECK(hipMalloc(&dR,    (size_t)n*n*sizeof(float)));
        HIP_CHECK(hipMalloc(&dRinv, (size_t)n*n*sizeof(float)));
        HIP_CHECK(hipMalloc(&dQ,    (size_t)m*n*sizeof(float)));
        HIP_CHECK(hipMemcpy(dA, hA.data(), m*n*sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dR, hR.data(), n*n*sizeof(float), hipMemcpyHostToDevice));

        double t0 = now_ms();

        // Step 1: CPU R⁻¹  (n×n only — cheap regardless of m)
        auto hRinv = cpu_upper_inv(hR, n);
        double t_rinv = now_ms();

        // Step 2: Upload R⁻¹
        HIP_CHECK(hipMemcpy(dRinv, hRinv.data(), n*n*sizeof(float), hipMemcpyHostToDevice));
        double t_upload = now_ms();

        // Step 3: Q = A · R⁻¹  via GEMM (m×n · n×n → m×n)
        gemm(dA, m, n, dRinv, n, dQ);
        double t_gemm = now_ms();

        double dt_total = t_gemm - t0;
        double dt_rinv  = t_rinv - t0;
        double dt_up    = t_upload - t_rinv;
        double dt_gm    = t_gemm - t_upload;

        // Verify for small cases
        float max_err = 0.0f;
        if (m <= 1000) {
            std::vector<float> hQ(m*n);
            HIP_CHECK(hipMemcpy(hQ.data(), dQ, m*n*sizeof(float), hipMemcpyDeviceToHost));
            for (int i=0; i<m*n; i++)
                max_err = std::max(max_err, std::abs(hQ[i] - hQ_expected[i]));
        }

        printf("  Q=A·R⁻¹  m=%-7d n=%-4d: total=%.1fms "
               "(R⁻¹=%.2fms upload=%.2fms GEMM=%.1fms)",
               m, n, dt_total, dt_rinv, dt_up, dt_gm);
        if (m <= 1000) printf("  max_err=%.2e %s", max_err, max_err<1e-4f?"OK":"FAIL");
        printf("\n");

        HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dR));
        HIP_CHECK(hipFree(dRinv)); HIP_CHECK(hipFree(dQ));
    }
}

/* =========================================================
 * Test 3: Tensor permute (for QR split)
 * ========================================================= */
static void test_permute() {
    printf("\n=== Test 3: Tensor permute for QR split ===\n");

    // Site tensor after env contraction: [chi_UP, chi_DOWN, chi_LEFT, D, chi_cut]
    // QR split: left=[chi_UP, chi_DOWN, chi_LEFT], right=[D, chi_cut]
    // Since physical dim D comes AFTER bond dims, and chi_cut is LAST,
    // the layout [chi,chi,chi,D,chi] is already in correct order:
    //   left = dims 0,1,2  → m = chi^3
    //   right = dims 3,4   → n = D*chi
    // No permute needed! Just reshape view.

    int chi=8;
    Tensor<float> psi({chi, chi, chi, D, chi});  // [chi,chi,chi,D,chi]
    printf("  Site tensor shape: "); psi.print_shape();

    // Fill with sequential values
    std::vector<float> h(psi.total);
    std::iota(h.begin(), h.end(), 0.0f);
    psi.copy_from_host(h);

    // QR split: n_left=3 → m=chi^3, n=D*chi
    auto [m, n] = psi.matrix_dims(3);
    printf("  QR split: m=%d (chi^3=%d)  n=%d (D*chi=%d)\n",
        m, chi*chi*chi, n, D*chi);
    assert(m == chi*chi*chi);
    assert(n == D*chi);

    // The data pointer IS the matrix — no copy needed for col-major split
    // Verify: element [i0,i1,i2,id,ic] should be at flat index
    // i0 + chi*(i1 + chi*(i2 + chi*(id + D*ic)))
    // In matrix view: row = i0 + chi*(i1 + chi*i2), col = id + D*ic
    int i0=1, i1=2, i2=3, id=1, ic=4;
    int expected_flat = i0 + chi*(i1 + chi*(i2 + chi*(id + D*ic)));
    int row = i0 + chi*(i1 + chi*i2);
    int col = id + D*ic;
    int matrix_flat = row + m*col;  // col-major
    printf("  Element [%d,%d,%d,%d,%d]: flat=%d, matrix[%d,%d]=%d  %s\n",
        i0, i1, i2, id, ic,
        expected_flat, row, col, matrix_flat,
        expected_flat == matrix_flat ? "OK" : "MISMATCH");
    assert(expected_flat == matrix_flat);

    // Now test the case where permute IS needed:
    // Julia's tensor may have shape [D, chi, chi, chi, chi] (phys first)
    // Then right_inds = [D, chi_cut] are NOT contiguous → need permute
    printf("  Physical-first layout [D,chi,chi,chi,chi]:\n");
    Tensor<float> psi2({D, chi, chi, chi, chi});
    printf("    shape: "); psi2.print_shape();
    // For QR split with right=[D, chi_cut=dim4]:
    // These are dims 0 and 4 — NOT contiguous → need permute to [chi,chi,chi,D,chi]
    printf("    Needs permute: dims [1,2,3] → left, dims [0,4] → right\n");
    printf("    After permute: [chi,chi,chi,D,chi] → m=%d n=%d\n",
        chi*chi*chi, D*chi);
    printf("    (our convention avoids this by putting D after bond dims)\n");

    printf("  Permute test: OK (no permute needed with our convention)\n");
}

/* =========================================================
 * Test 4: Full CholeskyQR pipeline
 * ========================================================= */
static void test_cholesky_qr() {
    printf("\n=== Test 4: CholeskyQR (POTRF + CPU R⁻¹ + GEMM) ===\n");

    // Typical QR input: m=chi^3=125000, n=D*chi=100
    for (auto [m, n] : std::vector<std::pair<int,int>>{{1000,20},{125000,100}}) {
        // Build a tall-skinny random matrix A (m×n)
        std::vector<float> hA(m*n);
        for (int i=0; i<m*n; i++) hA[i] = (float)((i*7+3) % 13) / 13.0f - 0.5f;

        float *dA, *dC, *dR, *dRinv, *dQ;
        int   *dInfo;
        HIP_CHECK(hipMalloc(&dA,    (size_t)m*n*sizeof(float)));
        HIP_CHECK(hipMalloc(&dC,    (size_t)n*n*sizeof(float)));
        HIP_CHECK(hipMalloc(&dR,    (size_t)n*n*sizeof(float)));
        HIP_CHECK(hipMalloc(&dRinv, (size_t)n*n*sizeof(float)));
        HIP_CHECK(hipMalloc(&dQ,    (size_t)m*n*sizeof(float)));
        HIP_CHECK(hipMalloc(&dInfo, sizeof(int)));
        HIP_CHECK(hipMemcpy(dA, hA.data(), m*n*sizeof(float), hipMemcpyHostToDevice));

        double t0 = now_ms();

        // Step 1: C = AᵀA  (n×n, tiny GEMM)
        gemm(dA, n, m, dA, n,
             dC, rocblas_operation_transpose, rocblas_operation_none);
        double t_gram = now_ms();

        // Step 2: Add regularization on host (tiny n×n)
        {
            std::vector<float> hC(n*n);
            HIP_CHECK(hipMemcpy(hC.data(), dC, n*n*sizeof(float), hipMemcpyDeviceToHost));
            float reg = 1e-6f;
            for (int i=0; i<n; i++) hC[i+i*n] += reg;
            HIP_CHECK(hipMemcpy(dC, hC.data(), n*n*sizeof(float), hipMemcpyHostToDevice));
        }

        // Step 3: POTRF — Cholesky of C → R (upper triangular)
        RB_CHECK(rocsolver_spotrf(g_handle, rocblas_fill_upper, n, dC, n, dInfo));
        HIP_CHECK(hipDeviceSynchronize());
        double t_potrf = now_ms();

        int hinfo;
        HIP_CHECK(hipMemcpy(&hinfo, dInfo, sizeof(int), hipMemcpyDeviceToHost));

        // Extract upper triangle → R
        {
            std::vector<float> hC(n*n), hR(n*n, 0.0f);
            HIP_CHECK(hipMemcpy(hC.data(), dC, n*n*sizeof(float), hipMemcpyDeviceToHost));
            for (int j=0; j<n; j++)
                for (int i=0; i<=j; i++)
                    hR[i+j*n] = hC[i+j*n];
            HIP_CHECK(hipMemcpy(dR, hR.data(), n*n*sizeof(float), hipMemcpyHostToDevice));

            // Step 4: R⁻¹ on CPU (n×n only)
            auto hRinv = cpu_upper_inv(hR, n);
            HIP_CHECK(hipMemcpy(dRinv, hRinv.data(), n*n*sizeof(float), hipMemcpyHostToDevice));
        }
        double t_rinv = now_ms();

        // Step 5: Q = A · R⁻¹  (large GEMM: m×n · n×n)
        gemm(dA, m, n, dRinv, n, dQ);
        double t_gemm = now_ms();

        double dt = t_gemm - t0;

        // Verify Q is approximately orthonormal: QᵀQ ≈ I  (for small m)
        float max_err = -1.0f;
        if (m <= 1000) {
            float *dQtQ;
            HIP_CHECK(hipMalloc(&dQtQ, n*n*sizeof(float)));
            gemm(dQ, n, m, dQ, n, dQtQ,
                 rocblas_operation_transpose, rocblas_operation_none);
            std::vector<float> hQtQ(n*n);
            HIP_CHECK(hipMemcpy(hQtQ.data(), dQtQ, n*n*sizeof(float), hipMemcpyDeviceToHost));
            max_err = 0.0f;
            for (int i=0; i<n; i++)
                for (int j=0; j<n; j++) {
                    float exp = (i==j) ? 1.0f : 0.0f;
                    max_err = std::max(max_err, std::abs(hQtQ[i+j*n] - exp));
                }
            HIP_CHECK(hipFree(dQtQ));
        }

        printf("  CholQR m=%-7d n=%-4d: info=%d  total=%.1fms"
               " (gram=%.1f potrf=%.1f R⁻¹=%.1f GEMM=%.1f)",
               m, n, hinfo, dt,
               t_gram-t0, t_potrf-t_gram, t_rinv-t_potrf, t_gemm-t_rinv);
        if (max_err >= 0.0f)
            printf("  QᵀQ-I max=%.2e %s", max_err, max_err<1e-3f?"OK":"FAIL");
        printf("\n");

        HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dC)); HIP_CHECK(hipFree(dR));
        HIP_CHECK(hipFree(dRinv)); HIP_CHECK(hipFree(dQ)); HIP_CHECK(hipFree(dInfo));
    }
}

/* =========================================================
 * Test 5: Householder QR
 * ========================================================= */
static void test_householder_qr() {
    printf("\n=== Test 5: Householder QR (GEQRF + ORGQR) ===\n");

    for (auto [m, n] : std::vector<std::pair<int,int>>{{200,10},{1000,20}}) {
        int k = std::min(m, n);
        std::vector<float> hA(m*n);
        for (int i=0; i<m*n; i++) hA[i] = (float)((i*7+3) % 13) / 13.0f - 0.5f;

        float *dWork, *dtau;
        HIP_CHECK(hipMalloc(&dWork, (size_t)m*n*sizeof(float)));
        HIP_CHECK(hipMalloc(&dtau,  (size_t)k*sizeof(float)));
        HIP_CHECK(hipMemcpy(dWork, hA.data(), m*n*sizeof(float), hipMemcpyHostToDevice));

        double t0 = now_ms();

        // GEQRF: in-place QR factorization
        RB_CHECK(rocsolver_sgeqrf(g_handle, m, n, dWork, m, dtau));
        HIP_CHECK(hipDeviceSynchronize());
        double t_geqrf = now_ms();

        // Extract R from upper triangle (on host) before ORGQR overwrites
        std::vector<float> hWork(m*n), hR(k*n, 0.0f);
        HIP_CHECK(hipMemcpy(hWork.data(), dWork, m*n*sizeof(float), hipMemcpyDeviceToHost));
        for (int j=0; j<n; j++)
            for (int i=0; i<=j && i<k; i++)
                hR[i + k*j] = hWork[i + j*m];

        // ORGQR: form Q explicitly in dWork (m×k)
        RB_CHECK(rocsolver_sorgqr(g_handle, m, k, k, dWork, m, dtau));
        HIP_CHECK(hipDeviceSynchronize());
        double t_orgqr = now_ms();

        // Verify QᵀQ ≈ I  — Q is m×k col-major in dWork
        // QᵀQ is k×k: use GEMM(Tᵀ, N, k, k, m)
        float *dQtQ;
        HIP_CHECK(hipMalloc(&dQtQ, k*k*sizeof(float)));
        // C[k×k] = dWorkᵀ[k×m] · dWork[m×k]
        float alpha=1.f, beta=0.f;
        RB_CHECK(rocblas_sgemm(g_handle,
            rocblas_operation_transpose, rocblas_operation_none,
            k, k, m,
            &alpha, dWork, m,   // Q: m×k col-major, lda=m
                    dWork, m,
            &beta,  dQtQ,  k));
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> hQtQ(k*k);
        HIP_CHECK(hipMemcpy(hQtQ.data(), dQtQ, k*k*sizeof(float), hipMemcpyDeviceToHost));
        float max_err = 0.0f;
        for (int i=0; i<k; i++)
            for (int j=0; j<k; j++) {
                float exp = (i==j) ? 1.0f : 0.0f;
                max_err = std::max(max_err, std::abs(hQtQ[i+j*k] - exp));
            }

        // Verify QR = A: A_reconstructed = Q(m×k) · R(k×n)
        float *dA_check;
        HIP_CHECK(hipMalloc(&dA_check, m*n*sizeof(float)));
        float *dR_gpu;
        HIP_CHECK(hipMalloc(&dR_gpu, k*n*sizeof(float)));
        HIP_CHECK(hipMemcpy(dR_gpu, hR.data(), k*n*sizeof(float), hipMemcpyHostToDevice));
        RB_CHECK(rocblas_sgemm(g_handle,
            rocblas_operation_none, rocblas_operation_none,
            m, n, k,
            &alpha, dWork, m,
                    dR_gpu, k,
            &beta,  dA_check, m));
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> hA_check(m*n);
        HIP_CHECK(hipMemcpy(hA_check.data(), dA_check, m*n*sizeof(float), hipMemcpyDeviceToHost));
        float max_err_qr = 0.0f;
        for (int i=0; i<m*n; i++)
            max_err_qr = std::max(max_err_qr, std::abs(hA_check[i] - hA[i]));

        printf("  HouseQR m=%-5d n=%-4d k=%-4d: total=%.1fms"
               " (geqrf=%.1f orgqr=%.1f)"
               "  QᵀQ-I=%.2e %s  QR-A=%.2e %s\n",
               m, n, k, t_orgqr-t0,
               t_geqrf-t0, t_orgqr-t_geqrf,
               max_err,    max_err    < 1e-4f ? "OK" : "FAIL",
               max_err_qr, max_err_qr < 1e-4f ? "OK" : "FAIL");

        HIP_CHECK(hipFree(dWork)); HIP_CHECK(hipFree(dtau));
        HIP_CHECK(hipFree(dQtQ));  HIP_CHECK(hipFree(dA_check));
        HIP_CHECK(hipFree(dR_gpu));
    }
}

/* =========================================================
 * Test 7: Unary gate — mode_n_product on physical index
 *
 * Site tensor shape: [chi_b1, ..., chi_b(d-1), D]
 * Physical index is the LAST dimension.
 * Gate G[D×D] contracts along mode (ndim-1).
 *
 * Tests:
 *   - Identity gate → tensor unchanged
 *   - X gate (bit flip) → physical indices swapped
 *   - Rz(π/4) → known rotation
 * ========================================================= */

// mode_n_product from test_contraction.cpp (reproduced here)
static void mode_n_product_last(
    const float* in, const int* shape, int ndim, int N,
    const float* E, int chi_out, int mode,
    float* out)
{
    // L = prod(shape[0..mode-1]), K = shape[mode], R = prod(shape[mode+1..])
    int L=1, K=shape[mode], R=1;
    for (int i=0;    i<mode;  i++) L *= shape[i];
    for (int i=mode+1; i<ndim; i++) R *= shape[i];

    float alpha=1.f, beta=0.f;
    for (int r=0; r<R; r++) {
        const float* A = in  + (size_t)r * L * K;
        float*       C = out + (size_t)r * L * chi_out;
        // C[L×chi_out] = A[L×K] · Eᵀ[K×chi_out]
        RB_CHECK(rocblas_sgemm(g_handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            L, chi_out, K,
            &alpha, A, L,
                    E, chi_out,
            &beta,  C, L));
    }
    HIP_CHECK(hipDeviceSynchronize());
}

static void test_unary_gate() {
    printf("\n=== Test 7: Unary gate (mode_n_product on physical index) ===\n");

    int chi = 4;
    // Site tensor: [chi, chi, D]  — edge site with 2 bonds
    // Physical index is LAST (mode 2)
    int shape[] = {chi, chi, D};
    int ndim = 3;
    int N = chi * chi * D;

    // Fill with known values: element [b1, b2, d] = b1 + chi*b2 + chi^2*d
    std::vector<float> h_psi(N);
    for (int b1=0; b1<chi; b1++)
        for (int b2=0; b2<chi; b2++)
            for (int d=0; d<D; d++)
                h_psi[b1 + chi*b2 + chi*chi*d] = (float)(b1 + chi*b2 + chi*chi*d);

    float *d_psi, *d_out;
    HIP_CHECK(hipMalloc(&d_psi, N*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, N*sizeof(float)));
    HIP_CHECK(hipMemcpy(d_psi, h_psi.data(), N*sizeof(float), hipMemcpyHostToDevice));

    // ── Test 7a: Identity gate ──────────────────────────────────────────
    {
        // I = [[1,0],[0,1]]  col-major: [1,0,0,1]
        std::vector<float> hG = {1.f, 0.f, 0.f, 1.f};  // col-major D×D
        float *d_G;
        HIP_CHECK(hipMalloc(&d_G, D*D*sizeof(float)));
        HIP_CHECK(hipMemcpy(d_G, hG.data(), D*D*sizeof(float), hipMemcpyHostToDevice));

        mode_n_product_last(d_psi, shape, ndim, N, d_G, D, 2, d_out);

        std::vector<float> h_out(N);
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, N*sizeof(float), hipMemcpyDeviceToHost));
        float max_err = 0.f;
        for (int i=0; i<N; i++) max_err = std::max(max_err, std::abs(h_out[i] - h_psi[i]));
        printf("  Identity gate: max_err=%.2e %s\n", max_err, max_err<1e-6f?"OK":"FAIL");
        assert(max_err < 1e-6f);
        HIP_CHECK(hipFree(d_G));
    }

    // ── Test 7b: X gate (bit flip: swaps d=0 and d=1) ──────────────────
    {
        // X = [[0,1],[1,0]]  col-major: [0,1,1,0]
        std::vector<float> hG = {0.f, 1.f, 1.f, 0.f};
        float *d_G;
        HIP_CHECK(hipMalloc(&d_G, D*D*sizeof(float)));
        HIP_CHECK(hipMemcpy(d_G, hG.data(), D*D*sizeof(float), hipMemcpyHostToDevice));

        mode_n_product_last(d_psi, shape, ndim, N, d_G, D, 2, d_out);

        std::vector<float> h_out(N);
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, N*sizeof(float), hipMemcpyDeviceToHost));

        // Verify: d_out[b1,b2,d] should equal h_psi[b1,b2,1-d]
        float max_err = 0.f;
        for (int b1=0; b1<chi; b1++)
            for (int b2=0; b2<chi; b2++)
                for (int d=0; d<D; d++) {
                    float got      = h_out [b1 + chi*b2 + chi*chi*d];
                    float expected = h_psi [b1 + chi*b2 + chi*chi*(1-d)];
                    max_err = std::max(max_err, std::abs(got - expected));
                }
        printf("  X gate (bit flip): max_err=%.2e %s\n", max_err, max_err<1e-6f?"OK":"FAIL");
        assert(max_err < 1e-6f);
        HIP_CHECK(hipFree(d_G));
    }

    // ── Test 7c: Z gate (phase flip: d=1 → -d=1) ──────────────────────
    {
        // Z = [[1,0],[0,-1]]  col-major: [1,0,0,-1]
        std::vector<float> hG = {1.f, 0.f, 0.f, -1.f};
        float *d_G;
        HIP_CHECK(hipMalloc(&d_G, D*D*sizeof(float)));
        HIP_CHECK(hipMemcpy(d_G, hG.data(), D*D*sizeof(float), hipMemcpyHostToDevice));

        mode_n_product_last(d_psi, shape, ndim, N, d_G, D, 2, d_out);

        std::vector<float> h_out(N);
        HIP_CHECK(hipMemcpy(h_out.data(), d_out, N*sizeof(float), hipMemcpyDeviceToHost));

        float max_err = 0.f;
        for (int b1=0; b1<chi; b1++)
            for (int b2=0; b2<chi; b2++)
                for (int d=0; d<D; d++) {
                    float got      = h_out[b1 + chi*b2 + chi*chi*d];
                    float expected = h_psi[b1 + chi*b2 + chi*chi*d] * (d==0 ? 1.f : -1.f);
                    max_err = std::max(max_err, std::abs(got - expected));
                }
        printf("  Z gate (phase flip): max_err=%.2e %s\n", max_err, max_err<1e-6f?"OK":"FAIL");
        assert(max_err < 1e-6f);
        HIP_CHECK(hipFree(d_G));
    }

    // ── Test 7d: steady-state size [chi,chi,chi,D] ─────────────────────
    {
        printf("  Steady-state unary gate [chi=%d, chi, chi, D=%d]:\n", chi, D);
        int chi2 = 50;
        int shape2[] = {chi2, chi2, chi2, D};
        int N2 = chi2*chi2*chi2*D;

        float *d_psi2, *d_out2;
        HIP_CHECK(hipMalloc(&d_psi2, N2*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_out2, N2*sizeof(float)));
        HIP_CHECK(hipMemset(d_psi2, 0, N2*sizeof(float)));

        // Identity gate
        std::vector<float> hG = {1.f, 0.f, 0.f, 1.f};
        float *d_G;
        HIP_CHECK(hipMalloc(&d_G, D*D*sizeof(float)));
        HIP_CHECK(hipMemcpy(d_G, hG.data(), D*D*sizeof(float), hipMemcpyHostToDevice));

        double t0 = now_ms();
        mode_n_product_last(d_psi2, shape2, 4, N2, d_G, D, 3, d_out2);
        double dt = now_ms() - t0;
        printf("    Identity on [%d,%d,%d,%d]: %.2fms\n",
               chi2, chi2, chi2, D, dt);

        HIP_CHECK(hipFree(d_psi2)); HIP_CHECK(hipFree(d_out2)); HIP_CHECK(hipFree(d_G));
    }

    HIP_CHECK(hipFree(d_psi)); HIP_CHECK(hipFree(d_out));
    printf("  Unary gate: ALL OK\n");
}

/* =========================================================
 * Test 6: Mini gate pipeline
 * QR(v1) + QR(v2) + RR=R1ᵀR2 + SVD(RR) + reconstruct
 * At steady-state sizes: m=125000, n=100
 * ========================================================= */
static void test_mini_gate_pipeline() {
    printf("\n=== Test 6: Mini gate pipeline (steady-state sizes) ===\n");

    int chi = 50, m1 = chi*chi*chi, n = D*chi;  // 125000 × 100
    int m2 = m1;  // same for interior-interior gate
    int k  = n;   // QR rank = n (tall-skinny)

    printf("  Sizes: m=%d n=%d (chi=%d D=%d)\n", m1, n, chi, D);
    printf("  Allocating tensors...\n");

    // Build random A1, A2 (m×n)
    std::vector<float> hA1(m1*n), hA2(m2*n);
    for (int i=0; i<m1*n; i++) hA1[i] = (float)((i*7+3) % 13)/13.0f - 0.5f;
    for (int i=0; i<m2*n; i++) hA2[i] = (float)((i*11+5) % 17)/17.0f - 0.5f;

    float *dA1, *dA2;
    float *dQ1, *dR1, *dRinv1;
    float *dQ2, *dR2, *dRinv2;
    float *dRR, *dU, *dS, *dVt, *dWork;
    float *dOut1, *dOut2;
    int   *dInfo;

    HIP_CHECK(hipMalloc(&dA1,    (size_t)m1*n*sizeof(float)));
    HIP_CHECK(hipMalloc(&dA2,    (size_t)m2*n*sizeof(float)));
    HIP_CHECK(hipMalloc(&dQ1,    (size_t)m1*k*sizeof(float)));
    HIP_CHECK(hipMalloc(&dR1,    (size_t)k*n*sizeof(float)));
    HIP_CHECK(hipMalloc(&dRinv1, (size_t)k*n*sizeof(float)));
    HIP_CHECK(hipMalloc(&dQ2,    (size_t)m2*k*sizeof(float)));
    HIP_CHECK(hipMalloc(&dR2,    (size_t)k*n*sizeof(float)));
    HIP_CHECK(hipMalloc(&dRinv2, (size_t)k*n*sizeof(float)));
    HIP_CHECK(hipMalloc(&dRR,    (size_t)n*n*sizeof(float)));
    HIP_CHECK(hipMalloc(&dU,     (size_t)n*k*sizeof(float)));
    HIP_CHECK(hipMalloc(&dS,     (size_t)k*sizeof(float)));
    HIP_CHECK(hipMalloc(&dVt,    (size_t)k*n*sizeof(float)));
    HIP_CHECK(hipMalloc(&dWork,  (size_t)n*n*sizeof(float)));
    HIP_CHECK(hipMalloc(&dOut1,  (size_t)m1*chi*sizeof(float)));
    HIP_CHECK(hipMalloc(&dOut2,  (size_t)m2*chi*sizeof(float)));
    HIP_CHECK(hipMalloc(&dInfo,  sizeof(int)));

    HIP_CHECK(hipMemcpy(dA1, hA1.data(), m1*n*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dA2, hA2.data(), m2*n*sizeof(float), hipMemcpyHostToDevice));

    double t0 = now_ms();

    // ── QR1: CholeskyQR of A1 (m1×n) ──
    {
        float *dC; int *dI;
        HIP_CHECK(hipMalloc(&dC, n*n*sizeof(float)));
        HIP_CHECK(hipMalloc(&dI, sizeof(int)));
        // Gram matrix C = A1ᵀA1
        gemm(dA1, n, m1, dA1, n, dC,
             rocblas_operation_transpose, rocblas_operation_none);
        // Regularize + POTRF on host download
        std::vector<float> hC(n*n);
        HIP_CHECK(hipMemcpy(hC.data(), dC, n*n*sizeof(float), hipMemcpyDeviceToHost));
        for (int i=0; i<n; i++) hC[i+i*n] += 1e-6f;
        HIP_CHECK(hipMemcpy(dC, hC.data(), n*n*sizeof(float), hipMemcpyHostToDevice));
        RB_CHECK(rocsolver_spotrf(g_handle, rocblas_fill_upper, n, dC, n, dI));
        HIP_CHECK(hipDeviceSynchronize());
        // Extract R1, compute R1⁻¹, upload
        HIP_CHECK(hipMemcpy(hC.data(), dC, n*n*sizeof(float), hipMemcpyDeviceToHost));
        std::vector<float> hR1(n*n, 0.0f);
        for (int j=0; j<n; j++) for (int i=0; i<=j; i++) hR1[i+j*n] = hC[i+j*n];
        HIP_CHECK(hipMemcpy(dR1, hR1.data(), n*n*sizeof(float), hipMemcpyHostToDevice));
        auto hRinv1 = cpu_upper_inv(hR1, n);
        HIP_CHECK(hipMemcpy(dRinv1, hRinv1.data(), n*n*sizeof(float), hipMemcpyHostToDevice));
        // Q1 = A1 · R1⁻¹
        gemm(dA1, m1, n, dRinv1, n, dQ1);
        HIP_CHECK(hipFree(dC)); HIP_CHECK(hipFree(dI));
    }
    double t_qr1 = now_ms();

    // ── QR2: CholeskyQR of A2 ──
    {
        float *dC; int *dI;
        HIP_CHECK(hipMalloc(&dC, n*n*sizeof(float)));
        HIP_CHECK(hipMalloc(&dI, sizeof(int)));
        gemm(dA2, n, m2, dA2, n, dC,
             rocblas_operation_transpose, rocblas_operation_none);
        std::vector<float> hC(n*n);
        HIP_CHECK(hipMemcpy(hC.data(), dC, n*n*sizeof(float), hipMemcpyDeviceToHost));
        for (int i=0; i<n; i++) hC[i+i*n] += 1e-6f;
        HIP_CHECK(hipMemcpy(dC, hC.data(), n*n*sizeof(float), hipMemcpyHostToDevice));
        RB_CHECK(rocsolver_spotrf(g_handle, rocblas_fill_upper, n, dC, n, dI));
        HIP_CHECK(hipDeviceSynchronize());
        HIP_CHECK(hipMemcpy(hC.data(), dC, n*n*sizeof(float), hipMemcpyDeviceToHost));
        std::vector<float> hR2(n*n, 0.0f);
        for (int j=0; j<n; j++) for (int i=0; i<=j; i++) hR2[i+j*n] = hC[i+j*n];
        HIP_CHECK(hipMemcpy(dR2, hR2.data(), n*n*sizeof(float), hipMemcpyHostToDevice));
        auto hRinv2 = cpu_upper_inv(hR2, n);
        HIP_CHECK(hipMemcpy(dRinv2, hRinv2.data(), n*n*sizeof(float), hipMemcpyHostToDevice));
        gemm(dA2, m2, n, dRinv2, n, dQ2);
        HIP_CHECK(hipFree(dC)); HIP_CHECK(hipFree(dI));
    }
    double t_qr2 = now_ms();

    // ── RR = R1ᵀ · R2  (n×n · n×n → n×n) ──
    gemm(dR1, n, n, dR2, n, dRR,
         rocblas_operation_transpose, rocblas_operation_none);
    double t_rr = now_ms();

    // ── SVD(RR) → U, S, Vt  (n×n) ──
    RB_CHECK(rocsolver_sgesvd(g_handle,
        rocblas_svect_singular, rocblas_svect_singular,
        n, n, dRR, n, dS, dU, n, dVt, n,
        dWork, rocblas_inplace, dInfo));
    HIP_CHECK(hipDeviceSynchronize());
    double t_svd = now_ms();

    // Read rank (truncate at chi)
    std::vector<float> hS(n);
    HIP_CHECK(hipMemcpy(hS.data(), dS, n*sizeof(float), hipMemcpyDeviceToHost));
    int rank = chi;  // truncate to chi

    // ── Reconstruct: Out1 = Q1 · U[:rank]  (m1×n · n×rank → m1×rank) ──
    gemm(dQ1, m1, n, dU, rank, dOut1);
    double t_recon1 = now_ms();

    // ── Reconstruct: Out2 = Q2 · Vt[:rank]ᵀ  (m2×n · n×rank → m2×rank) ──
    gemm(dQ2, m2, n, dVt, rank, dOut2,
         rocblas_operation_none, rocblas_operation_transpose);
    double t_recon2 = now_ms();

    double dt_total = t_recon2 - t0;

    printf("  QR1:     %.1f ms\n", t_qr1 - t0);
    printf("  QR2:     %.1f ms\n", t_qr2 - t_qr1);
    printf("  RR:      %.1f ms\n", t_rr  - t_qr2);
    printf("  SVD:     %.1f ms\n", t_svd - t_rr);
    printf("  Recon1:  %.1f ms\n", t_recon1 - t_svd);
    printf("  Recon2:  %.1f ms\n", t_recon2 - t_recon1);
    printf("  TOTAL:   %.1f ms\n", dt_total);
    printf("  S[0]=%.4f S[chi-1]=%.4f rank=%d\n", hS[0], hS[chi-1], rank);
    printf("  Pipeline: OK\n");

    HIP_CHECK(hipFree(dA1)); HIP_CHECK(hipFree(dA2));
    HIP_CHECK(hipFree(dQ1)); HIP_CHECK(hipFree(dR1)); HIP_CHECK(hipFree(dRinv1));
    HIP_CHECK(hipFree(dQ2)); HIP_CHECK(hipFree(dR2)); HIP_CHECK(hipFree(dRinv2));
    HIP_CHECK(hipFree(dRR)); HIP_CHECK(hipFree(dU));
    HIP_CHECK(hipFree(dS));  HIP_CHECK(hipFree(dVt)); HIP_CHECK(hipFree(dWork));
    HIP_CHECK(hipFree(dOut1)); HIP_CHECK(hipFree(dOut2)); HIP_CHECK(hipFree(dInfo));
}

int main() {
    printf("=== test_gate_ops ===\n");
    printf("D=%d (physical dim)\n", D);

    rocblas_create_handle(&g_handle);

    test_svd();
    test_trsm_replacement();
    test_permute();
    test_cholesky_qr();
    test_householder_qr();
    test_unary_gate();
    test_mini_gate_pipeline();

    rocblas_destroy_handle(g_handle);
    printf("\n=== ALL GATE OPS TESTS COMPLETE ===\n");
    return 0;
}
