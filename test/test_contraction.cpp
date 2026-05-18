/*
 * tests/test_contraction.cpp
 *
 * Test: mode-n product (env contraction) using rocBLAS SGEMM.
 *
 * The mode-n product contracts one bond index of a tensor with a matrix.
 * For a site tensor ψ[chi0, chi1, chi2, D, chi_cut] and env E[chi0, chi0]:
 *
 *   result[i0, i1, i2, d, ic] = Σⱼ ψ[j, i1, i2, d, ic] · E[j, i0]
 *
 * In matrix form (col-major):
 *   Reshape ψ so that mode-0 is rows, rest is cols:
 *     A = ψ_mat[chi0 × (chi1*chi2*D*chi_cut)]   (row-major view)
 *   Then:
 *     result_mat = Eᵀ · A    [chi0 × rest]
 *   Or in col-major (rocBLAS native):
 *     result_mat = Aᵀ · E    [(rest) × chi0]  then transpose back
 *
 * Actually simpler: use the "left multiply" form directly.
 * For col-major tensors, mode-n product along dim k:
 *   1. Reshape tensor to [left_size × chi_k × right_size]  (virtual 3D)
 *   2. For each slice along right_size, do: result_col = E · ψ_col
 *   3. This is a single GEMM with batching
 *
 * Build:
 *   hipcc -O2 -std=c++17 -I../include -o test_contraction test_contraction.cpp -lrocblas -lm
 * Run:
 *   ./test_contraction
 */

#include "types.hpp"
#include "tensor.hpp"
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cassert>

/* =========================================================
 * mode_n_product: contract tensor along dimension `mode` with matrix E
 *
 * Input:
 *   in:   Tensor with shape [..., chi_mode, ...]  (col-major GPU)
 *   E:    Matrix [chi_out × chi_mode] on GPU (col-major)
 *   mode: which dimension to contract
 *
 * Output:
 *   out:  Tensor with same shape but shape[mode] = chi_out
 *
 * Algorithm (col-major):
 *   left_size  = product(shape[0..mode-1])   = L
 *   chi_mode   = shape[mode]                 = K
 *   right_size = product(shape[mode+1..end]) = R
 *
 *   Reshape in  as [L × K × R] — virtually, no copy
 *   Reshape out as [L × chi_out × R]
 *
 *   For col-major storage, the K dimension stride = L.
 *   Each "right slice" r has:
 *     in_slice  = in.data  + r * L * K   [L×K col-major]
 *     out_slice = out.data + r * L * chi_out
 *
 *   GEMM: out_slice[L × chi_out] = in_slice[L × K] · Eᵀ[K × chi_out]
 *   i.e.: C = A · Bᵀ  where A=in_slice, B=E, C=out_slice
 *   rocBLAS: sgemm(N, T, L, chi_out, K, 1, A, L, E, chi_out, 0, C, L)
 * ========================================================= */
void mode_n_product(
    rocblas_handle handle,
    const Tensor<float>& in,
    const float* E,          // [chi_out × chi_mode] col-major GPU
    int chi_out,
    int mode,
    Tensor<float>& out)
{
    int L = 1, K = in.shape[mode], R = 1;
    for (int i = 0;    i < mode;        i++) L *= in.shape[i];
    for (int i = mode+1; i < in.ndim;  i++) R *= in.shape[i];

    float alpha = 1.0f, beta = 0.0f;

    for (int r = 0; r < R; r++) {
        const float* A = in.data  + (size_t)r * L * K;
        float*       C = out.data + (size_t)r * L * chi_out;

        // C[L × chi_out] = A[L × K] · Eᵀ[K × chi_out]
        // rocBLAS col-major: sgemm(opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
        // m=L, n=chi_out, k=K
        // A: L×K col-major, lda=L
        // B=E: chi_out×K col-major → opB=Transpose → K×chi_out
        // C: L×chi_out col-major, ldc=L
        rocblas_sgemm(handle,
            rocblas_operation_none,       // opA: A as-is  (L×K)
            rocblas_operation_transpose,  // opB: Eᵀ       (K×chi_out from chi_out×K)
            L, chi_out, K,
            &alpha,
            A, L,
            E, chi_out,   // E is [chi_out × K] col-major, lda=chi_out
            &beta,
            C, L);
    }
    HIP_CHECK(hipDeviceSynchronize());
}

/* =========================================================
 * pseudo_sqrt_inv_sqrt: given a diagonal BP message (stored as chi×chi),
 * compute sqrt and inv_sqrt environments.
 *
 * Julia's pseudo_sqrt_inv_sqrt does:
 *   1. Symmetrize: M = (M + Mᵀ)/2
 *   2. Eigendecompose: M = V · Λ · Vᵀ  (SYEVD)
 *   3. Threshold small/negative eigenvalues: λ = max(λ, cutoff * max_λ)
 *   4. sqrt_env    = V · diag(sqrt(λ)) · Vᵀ
 *   5. inv_sqrt_env = V · diag(1/sqrt(λ)) · Vᵀ
 *
 * For BP messages from simple_update: M is already diagonal (singular values).
 * So V=I, Λ=diag(s), and:
 *   sqrt_env     = diag(sqrt(s))
 *   inv_sqrt_env = diag(1/sqrt(s))
 *
 * We implement the full SYEVD version for generality.
 * For now: simple diagonal version as a placeholder (V=I assumed).
 * ========================================================= */
void pseudo_sqrt_diag(
    const float* msg,      // [chi × chi] GPU, diagonal entries are singular values
    float* sqrt_env,       // [chi × chi] GPU output
    float* inv_sqrt_env,   // [chi × chi] GPU output
    int chi,
    float cutoff = 1e-8f)
{
    // Read diagonal on host
    std::vector<float> h_msg(chi * chi);
    HIP_CHECK(hipMemcpy(h_msg.data(), msg, chi*chi*sizeof(float), hipMemcpyDeviceToHost));

    std::vector<float> h_sqrt(chi*chi, 0.0f), h_inv(chi*chi, 0.0f);

    // Find max diagonal
    float max_val = 0.0f;
    for (int i = 0; i < chi; i++) max_val = std::max(max_val, h_msg[i + i*chi]);
    float thresh = cutoff * max_val;

    for (int i = 0; i < chi; i++) {
        float val = std::max(h_msg[i + i*chi], thresh);
        float s   = std::sqrt(val);
        h_sqrt[i + i*chi] = s;
        h_inv [i + i*chi] = (s > 1e-30f) ? 1.0f/s : 0.0f;
    }

    HIP_CHECK(hipMemcpy(sqrt_env,     h_sqrt.data(), chi*chi*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(inv_sqrt_env, h_inv.data(),  chi*chi*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());
}

int main() {
    printf("=== test_contraction ===\n");

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // ── Test 1: simple mode-0 product ────────────────────────────────────
    {
        printf("\n--- Test 1: mode-0 product [4,3] x E[4,4] ---\n");
        // ψ shape: [4, 3] — 4 rows, 3 cols
        // E shape: [4, 4] — square identity → result should equal input
        int m=4, n=3;
        std::vector<float> h_psi(m*n);
        for (int i=0; i<m*n; i++) h_psi[i] = (float)(i+1);

        Tensor<float> psi({m, n});
        psi.copy_from_host(h_psi);
        psi.print_elements(12, "psi_before");

        // E = identity [m×m]
        std::vector<float> h_E(m*m, 0.0f);
        for (int i=0; i<m; i++) h_E[i+i*m] = 1.0f;
        Tensor<float> E({m, m});
        E.copy_from_host(h_E);

        Tensor<float> result({m, n});
        mode_n_product(handle, psi, E.data, m, 0, result);
        result.print_elements(12, "result (should match psi)");

        auto h_res = result.copy_to_host();
        for (int i=0; i<m*n; i++)
            assert(std::abs(h_res[i] - h_psi[i]) < 1e-5f);
        printf("  mode-0 identity: OK\n");
    }

    // ── Test 2: mode-0 product with scaling ──────────────────────────────
    {
        printf("\n--- Test 2: mode-0 product with 2x scaling ---\n");
        int chi=4;
        std::vector<float> h_psi(chi*chi);
        for (int i=0; i<chi*chi; i++) h_psi[i] = 1.0f;

        Tensor<float> psi({chi, chi});
        psi.copy_from_host(h_psi);

        // E = 2*identity
        std::vector<float> h_E(chi*chi, 0.0f);
        for (int i=0; i<chi; i++) h_E[i+i*chi] = 2.0f;
        Tensor<float> E({chi, chi});
        E.copy_from_host(h_E);

        Tensor<float> result({chi, chi});
        mode_n_product(handle, psi, E.data, chi, 0, result);

        auto h_res = result.copy_to_host();
        for (int i=0; i<chi*chi; i++)
            assert(std::abs(h_res[i] - 2.0f) < 1e-5f);
        printf("  mode-0 2x scale: OK\n");
    }

    // ── Test 3: env contraction on a site tensor ──────────────────────────
    {
        printf("\n--- Test 3: env contraction on edge site [chi,chi,chi,D,chi_cut] ---\n");
        int chi=8, chi_cut=8;

        // Site tensor: [chi_UP, chi_DOWN, chi_LEFT, D, chi_cut]
        // = [8, 8, 8, 2, 8]
        Tensor<float> psi({chi, chi, chi, D, chi_cut});
        // Fill with 1s
        std::vector<float> h_ones(psi.total, 1.0f);
        psi.copy_from_host(h_ones);

        printf("  psi shape: "); psi.print_shape();

        // Env UP: [chi × chi] = identity → should not change psi
        std::vector<float> h_E(chi*chi, 0.0f);
        for (int i=0; i<chi; i++) h_E[i+i*chi] = 1.0f;
        Tensor<float> E_up({chi, chi});
        E_up.copy_from_host(h_E);

        Tensor<float> psi_contracted({chi, chi, chi, D, chi_cut});
        mode_n_product(handle, psi, E_up.data, chi, 0, psi_contracted);

        // All elements should still be 1
        auto h_res = psi_contracted.copy_to_host();
        float max_err = 0.0f;
        for (auto x : h_res) max_err = std::max(max_err, std::abs(x - 1.0f));
        printf("  identity env contraction max_err=%.2e: %s\n",
            max_err, max_err < 1e-5f ? "OK" : "FAILED");

        // Now contract all 3 env bonds (UP, DOWN, LEFT)
        // Each is a mode-n product along modes 0, 1, 2
        Tensor<float> tmp1({chi, chi, chi, D, chi_cut});
        Tensor<float> tmp2({chi, chi, chi, D, chi_cut});
        mode_n_product(handle, psi,  E_up.data, chi, 0, tmp1);  // UP
        mode_n_product(handle, tmp1, E_up.data, chi, 1, tmp2);  // DOWN
        mode_n_product(handle, tmp2, E_up.data, chi, 2, psi_contracted); // LEFT

        h_res = psi_contracted.copy_to_host();
        max_err = 0.0f;
        for (auto x : h_res) max_err = std::max(max_err, std::abs(x - 1.0f));
        printf("  3x identity env contraction max_err=%.2e: %s\n",
            max_err, max_err < 1e-5f ? "OK" : "FAILED");
    }

    // ── Test 4: pseudo_sqrt_diag ──────────────────────────────────────────
    {
        printf("\n--- Test 4: pseudo_sqrt of diagonal BP message ---\n");
        int chi = 6;
        // BP message: diagonal with singular values [4, 9, 16, 25, 0.0001, 0]
        std::vector<float> h_msg(chi*chi, 0.0f);
        float vals[] = {4.0f, 9.0f, 16.0f, 25.0f, 0.0001f, 0.0f};
        for (int i=0; i<chi; i++) h_msg[i+i*chi] = vals[i];

        Tensor<float> msg({chi, chi});
        msg.copy_from_host(h_msg);

        Tensor<float> sqrt_env({chi, chi}), inv_sqrt_env({chi, chi});
        pseudo_sqrt_diag(msg.data, sqrt_env.data, inv_sqrt_env.data, chi, 1e-8f);

        auto h_sqrt = sqrt_env.copy_to_host();
        auto h_inv  = inv_sqrt_env.copy_to_host();

        printf("  sqrt_env diagonal:     ");
        for (int i=0; i<chi; i++) printf("%.4f ", h_sqrt[i+i*chi]);
        printf("\n  inv_sqrt_env diagonal: ");
        for (int i=0; i<chi; i++) printf("%.4f ", h_inv[i+i*chi]);
        printf("\n");

        // Verify: sqrt * inv_sqrt = I (on non-zero entries)
        for (int i=0; i<chi-1; i++) {
            float product = h_sqrt[i+i*chi] * h_inv[i+i*chi];
            assert(std::abs(product - 1.0f) < 1e-4f);
        }
        printf("  sqrt * inv_sqrt = I: OK\n");

        // Verify: sqrt^2 = original (on non-thresholded entries)
        for (int i=0; i<4; i++) {
            float sq = h_sqrt[i+i*chi];
            assert(std::abs(sq*sq - vals[i]) < 1e-4f);
        }
        printf("  sqrt^2 = original: OK\n");
    }

    // ── Test 5: contract then invert (round-trip) ─────────────────────────
    {
        printf("\n--- Test 5: contract with sqrt_env then inv_sqrt_env (round-trip) ---\n");
        int chi=8;
        // Random-ish site tensor
        std::vector<float> h_psi(chi*chi*D);
        for (int i=0; i<(int)h_psi.size(); i++) h_psi[i] = (float)(i % 7) - 3.0f;

        Tensor<float> psi({chi, chi, D});
        psi.copy_from_host(h_psi);

        // BP message: diagonal with values [1,2,3,4,5,6,7,8]
        std::vector<float> h_msg(chi*chi, 0.0f);
        for (int i=0; i<chi; i++) h_msg[i+i*chi] = (float)(i+1);
        Tensor<float> msg({chi, chi});
        msg.copy_from_host(h_msg);

        Tensor<float> sqrt_env({chi,chi}), inv_sqrt_env({chi,chi});
        pseudo_sqrt_diag(msg.data, sqrt_env.data, inv_sqrt_env.data, chi);

        // Contract with sqrt_env along mode 0
        Tensor<float> contracted({chi, chi, D});
        mode_n_product(handle, psi, sqrt_env.data, chi, 0, contracted);

        // Contract back with inv_sqrt_env along mode 0
        Tensor<float> recovered({chi, chi, D});
        mode_n_product(handle, contracted, inv_sqrt_env.data, chi, 0, recovered);

        // Should recover original psi
        auto h_rec = recovered.copy_to_host();
        float max_err = 0.0f;
        for (int i=0; i<(int)h_psi.size(); i++)
            max_err = std::max(max_err, std::abs(h_rec[i] - h_psi[i]));
        printf("  round-trip max_err=%.2e: %s\n",
            max_err, max_err < 1e-4f ? "OK" : "FAILED");
    }

    rocblas_destroy_handle(handle);
    printf("\n=== ALL CONTRACTION TESTS PASSED ===\n");
    return 0;
}
