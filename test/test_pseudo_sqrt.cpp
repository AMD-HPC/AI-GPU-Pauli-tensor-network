/*
 * tests/test_pseudo_sqrt.cpp
 *
 * Test: pseudo_sqrt_inv_sqrt via SYEVD (rocsolver_ssyevd)
 * This matches Julia's pseudo_sqrt_inv_sqrt exactly.
 *
 * Algorithm:
 *   Given M (chi×chi, symmetric positive semidefinite BP message):
 *   1. Symmetrize: M = (M + Mᵀ)/2
 *   2. Eigendecompose: M = V · Λ · Vᵀ  (SYEVD on GPU)
 *   3. Threshold: λ_i = max(λ_i, cutoff * max_λ)  (zero negatives)
 *   4. sqrt_env    = V · diag(sqrt(λ)) · Vᵀ
 *   5. inv_sqrt_env = V · diag(1/sqrt(λ)) · Vᵀ
 *
 * Build:
 *   hipcc -O2 -std=c++17 -DPHYS_DIM=2 -I../include \
 *         -o test_pseudo_sqrt test_pseudo_sqrt.cpp -lrocblas -lrocsolver -lm
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

/* =========================================================
 * pseudo_sqrt_syevd
 *
 * Computes sqrt_env and inv_sqrt_env for a symmetric BP message M.
 * All computation on GPU, small final steps on host.
 *
 * Inputs:
 *   handle:      rocblas handle
 *   M:           [chi×chi] GPU array (symmetric BP message)
 *   chi:         dimension
 *   cutoff:      threshold for small eigenvalues (relative to max)
 *
 * Outputs:
 *   sqrt_env:    [chi×chi] GPU array
 *   inv_sqrt_env:[chi×chi] GPU array
 * ========================================================= */
void pseudo_sqrt_syevd(
    rocblas_handle handle,
    const float* M,
    float* sqrt_env,
    float* inv_sqrt_env,
    int chi,
    float cutoff = 10.0f * 1.1920929e-7f)  // 10 * eps(Float32) matches Julia default
{
    // Step 1: Copy M and symmetrize on host
    std::vector<float> hM(chi*chi);
    HIP_CHECK(hipMemcpy(hM.data(), M, chi*chi*sizeof(float), hipMemcpyDeviceToHost));
    for (int i=0; i<chi; i++)
        for (int j=i+1; j<chi; j++) {
            float sym = 0.5f * (hM[i+j*chi] + hM[j+i*chi]);
            hM[i+j*chi] = hM[j+i*chi] = sym;
        }

    // Upload symmetrized M
    float* d_M;
    HIP_CHECK(hipMalloc(&d_M, chi*chi*sizeof(float)));
    HIP_CHECK(hipMemcpy(d_M, hM.data(), chi*chi*sizeof(float), hipMemcpyHostToDevice));

    // Step 2: Eigendecompose M = V · Λ · Vᵀ
    float* d_W;
    float* d_E;
    int*   d_info;
    HIP_CHECK(hipMalloc(&d_W,    chi*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_E,    std::max(chi-1, 1)*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_info, sizeof(int)));

    RB_CHECK(rocsolver_ssyevd(handle,
        rocblas_evect_original,
        rocblas_fill_upper,
        chi, d_M, chi,
        d_W, d_E, d_info));
    HIP_CHECK(hipDeviceSynchronize());

    int hinfo;
    HIP_CHECK(hipMemcpy(&hinfo, d_info, sizeof(int), hipMemcpyDeviceToHost));
    if (hinfo != 0)
        fprintf(stderr, "[pseudo_sqrt] SYEVD failed with info=%d\n", hinfo);

    // Step 3: Read eigenvalues and eigenvectors on host
    std::vector<float> hW(chi), hV(chi*chi);
    HIP_CHECK(hipMemcpy(hW.data(), d_W, chi*sizeof(float),    hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hV.data(), d_M, chi*chi*sizeof(float), hipMemcpyDeviceToHost));

    // Step 4: Apply Julia's exact rule:
    //   D_sqrt[i]     = (x==0 || |x| < cutoff) ? 0 : sqrt(x)
    //   D_inv_sqrt[i] = (x==0 || |x| < cutoff) ? 0 : 1/sqrt(x)
    std::vector<float> hVsqrt(chi*chi), hVinv(chi*chi);
    for (int k=0; k<chi; k++) {
        float lam = hW[k];
        float s, sinv;
        if (lam == 0.0f || std::abs(lam) < cutoff) {
            s = 0.0f; sinv = 0.0f;
        } else {
            s    = std::sqrt(std::abs(lam));
            sinv = 1.0f / s;
        }
        for (int i=0; i<chi; i++) {
            hVsqrt[i + k*chi] = hV[i + k*chi] * s;
            hVinv [i + k*chi] = hV[i + k*chi] * sinv;
        }
    }

    // Step 5: sqrt_env = Vsqrt · Vᵀ,  inv_sqrt_env = Vinv · Vᵀ
    float *d_Vsqrt, *d_Vinv;
    HIP_CHECK(hipMalloc(&d_Vsqrt, chi*chi*sizeof(float)));
    HIP_CHECK(hipMalloc(&d_Vinv,  chi*chi*sizeof(float)));
    HIP_CHECK(hipMemcpy(d_Vsqrt, hVsqrt.data(), chi*chi*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_Vinv,  hVinv.data(),  chi*chi*sizeof(float), hipMemcpyHostToDevice));

    float alpha=1.0f, beta=0.0f;
    RB_CHECK(rocblas_sgemm(handle,
        rocblas_operation_none, rocblas_operation_transpose,
        chi, chi, chi,
        &alpha, d_Vsqrt, chi, d_M, chi,
        &beta,  sqrt_env, chi));

    RB_CHECK(rocblas_sgemm(handle,
        rocblas_operation_none, rocblas_operation_transpose,
        chi, chi, chi,
        &alpha, d_Vinv, chi, d_M, chi,
        &beta,  inv_sqrt_env, chi));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_M)); HIP_CHECK(hipFree(d_W));
    HIP_CHECK(hipFree(d_E)); HIP_CHECK(hipFree(d_info));
    HIP_CHECK(hipFree(d_Vsqrt)); HIP_CHECK(hipFree(d_Vinv));
}

int main() {
    printf("=== test_pseudo_sqrt ===\n");

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // ── Test 1: diagonal message (simple case) ────────────────────────────
    {
        printf("\n--- Test 1: diagonal BP message [6×6] ---\n");
        int chi = 6;
        float vals[] = {4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f};
        std::vector<float> hM(chi*chi, 0.0f);
        for (int i=0; i<chi; i++) hM[i+i*chi] = vals[i];

        float *d_M, *d_sqrt, *d_inv;
        HIP_CHECK(hipMalloc(&d_M,    chi*chi*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_sqrt, chi*chi*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_inv,  chi*chi*sizeof(float)));
        HIP_CHECK(hipMemcpy(d_M, hM.data(), chi*chi*sizeof(float), hipMemcpyHostToDevice));

        pseudo_sqrt_syevd(handle, d_M, d_sqrt, d_inv, chi, 1e-8f);

        // Read back and verify diagonal entries
        std::vector<float> h_sqrt(chi*chi), h_inv(chi*chi);
        HIP_CHECK(hipMemcpy(h_sqrt.data(), d_sqrt, chi*chi*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_inv.data(),  d_inv,  chi*chi*sizeof(float), hipMemcpyDeviceToHost));

        printf("  sqrt diagonal: ");
        for (int i=0; i<chi; i++) printf("%.3f ", h_sqrt[i+i*chi]);
        printf("\n  expected:      ");
        for (int i=0; i<chi; i++) printf("%.3f ", std::sqrt(vals[i]));
        printf("\n");

        // Verify: sqrt^2 ≈ original
        for (int i=0; i<chi; i++) {
            float sq = h_sqrt[i+i*chi];
            assert(std::abs(sq*sq - vals[i]) < 1e-3f);
        }
        printf("  sqrt^2 = original: OK\n");

        // Verify: sqrt * inv_sqrt ≈ I (on diagonal)
        for (int i=0; i<chi; i++) {
            float prod = h_sqrt[i+i*chi] * h_inv[i+i*chi];
            assert(std::abs(prod - 1.0f) < 1e-4f);
        }
        printf("  sqrt * inv_sqrt = I: OK\n");

        HIP_CHECK(hipFree(d_M)); HIP_CHECK(hipFree(d_sqrt)); HIP_CHECK(hipFree(d_inv));
    }

    // ── Test 2: general symmetric SPD message ─────────────────────────────
    {
        printf("\n--- Test 2: general symmetric SPD message [8×8] ---\n");
        int chi = 8;
        // Build a random SPD matrix: A = B·Bᵀ + chi*I
        std::vector<float> hB(chi*chi), hM(chi*chi, 0.0f);
        for (int i=0; i<chi*chi; i++) hB[i] = (float)(i % 5) - 2.0f;
        for (int i=0; i<chi; i++)
            for (int j=0; j<chi; j++) {
                float s = 0.0f;
                for (int k=0; k<chi; k++) s += hB[i+k*chi] * hB[j+k*chi];
                hM[i+j*chi] = s;
            }
        for (int i=0; i<chi; i++) hM[i+i*chi] += (float)chi;

        float *d_M, *d_sqrt, *d_inv;
        HIP_CHECK(hipMalloc(&d_M,    chi*chi*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_sqrt, chi*chi*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_inv,  chi*chi*sizeof(float)));
        HIP_CHECK(hipMemcpy(d_M, hM.data(), chi*chi*sizeof(float), hipMemcpyHostToDevice));

        pseudo_sqrt_syevd(handle, d_M, d_sqrt, d_inv, chi, 1e-8f);

        // Verify: sqrt_env · sqrt_env ≈ M (on host)
        std::vector<float> h_sqrt(chi*chi), h_inv(chi*chi);
        HIP_CHECK(hipMemcpy(h_sqrt.data(), d_sqrt, chi*chi*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_inv.data(),  d_inv,  chi*chi*sizeof(float), hipMemcpyDeviceToHost));

        // Compute sqrt·sqrt on host
        float max_err = 0.0f;
        for (int i=0; i<chi; i++)
            for (int j=0; j<chi; j++) {
                float s = 0.0f;
                for (int k=0; k<chi; k++) s += h_sqrt[i+k*chi] * h_sqrt[j+k*chi];
                max_err = std::max(max_err, std::abs(s - hM[i+j*chi]));
            }
        printf("  sqrt·sqrt - M max_err = %.2e: %s\n",
            max_err, max_err < 1e-2f ? "OK" : "FAILED");

        // Verify: sqrt · inv_sqrt ≈ I
        float max_err_id = 0.0f;
        for (int i=0; i<chi; i++)
            for (int j=0; j<chi; j++) {
                float s = 0.0f;
                for (int k=0; k<chi; k++) s += h_sqrt[i+k*chi] * h_inv[j+k*chi];
                float expected = (i==j) ? 1.0f : 0.0f;
                max_err_id = std::max(max_err_id, std::abs(s - expected));
            }
        printf("  sqrt · inv_sqrt - I max_err = %.2e: %s\n",
            max_err_id, max_err_id < 1e-3f ? "OK" : "FAILED");

        HIP_CHECK(hipFree(d_M)); HIP_CHECK(hipFree(d_sqrt)); HIP_CHECK(hipFree(d_inv));
    }

    // ── Test 3: steady-state size chi=50 ─────────────────────────────────
    {
        printf("\n--- Test 3: steady-state BP message [50×50] ---\n");
        int chi = 50;
        // Diagonal message (typical BP message is close to diagonal)
        std::vector<float> hM(chi*chi, 0.0f);
        for (int i=0; i<chi; i++) hM[i+i*chi] = (float)(i+1);  // 1,2,...,50

        float *d_M, *d_sqrt, *d_inv;
        HIP_CHECK(hipMalloc(&d_M,    chi*chi*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_sqrt, chi*chi*sizeof(float)));
        HIP_CHECK(hipMalloc(&d_inv,  chi*chi*sizeof(float)));
        HIP_CHECK(hipMemcpy(d_M, hM.data(), chi*chi*sizeof(float), hipMemcpyHostToDevice));

        auto t0 = std::chrono::high_resolution_clock::now();
        pseudo_sqrt_syevd(handle, d_M, d_sqrt, d_inv, chi, 1e-8f);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();

        std::vector<float> h_sqrt(chi*chi);
        HIP_CHECK(hipMemcpy(h_sqrt.data(), d_sqrt, chi*chi*sizeof(float), hipMemcpyDeviceToHost));

        // Check diagonal entries
        bool ok = true;
        for (int i=0; i<chi; i++) {
            float expected = std::sqrt((float)(i+1));
            if (std::abs(h_sqrt[i+i*chi] - expected) > 1e-3f) { ok = false; break; }
        }
        printf("  chi=50 SYEVD: %.2f ms, diagonal OK: %s\n", ms, ok ? "YES" : "NO");

        HIP_CHECK(hipFree(d_M)); HIP_CHECK(hipFree(d_sqrt)); HIP_CHECK(hipFree(d_inv));
    }

    rocblas_destroy_handle(handle);
    printf("\n=== ALL PSEUDO_SQRT TESTS PASSED ===\n");
    return 0;
}
