/*
 * tests/test_measures.cpp
 *
 * Test: BP partition function and measurements.
 *
 * Test strategy: use chi=1 tensors where exact answer is known.
 *
 * At chi=1:
 *   - Site tensor ψ_v[D] — just a D-vector
 *   - Messages msg[1×1] — just scalars = 1.0 (trivial env)
 *   - vertex_scalar(v) = Σ_d O[d,d] * ψ_v[d]  (for diagonal O)
 *   - edge_scalar(e) = 1.0
 *   - Z = ∏_v vertex_scalar(v)
 *
 * For identity state (all sites = [1,1]/sqrt(2), msg=1):
 *   vertex_scalar with I = Σ_d ψ_v[d] = 1+1 = 2  (unnormalized)
 *   vertex_scalar with Z = ψ_v[0] - ψ_v[1] = 0
 *
 * For Z state at center (ψ_center = [1,0]):
 *   vertex_scalar with I = 1
 *   vertex_scalar with Z = 1
 *   → autocorrelation = 1.0
 *
 * Build:
 *   hipcc -O2 -std=c++17 -DPHYS_DIM=2 -I../include \
 *         -o test_measures test_measures.cpp -lrocblas -lrocsolver -lm
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "measures.hpp"
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>

int main() {
    printf("=== test_measures ===\n");
    printf("D=%d\n\n", D);

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Use a small 2×2 grid with chi=1 for exact verification
    Grid g(2, 2, 1);
    Workspace ws(1);  // chi_max=1

    // ── Setup: chi=1 tensors and messages ────────────────────────────
    // At chi=1: each site tensor has shape [1,...,1,D] = just D elements
    // Each message is 1×1 = scalar = 1.0

    int n_sites = g.n_sites;   // 4
    int n_bonds = g.n_bonds;   // 4

    // Allocate site tensors (chi=1, so each is just D floats)
    std::vector<float*> psi(n_sites);
    for (int i=0; i<n_sites; i++) {
        HIP_CHECK(hipMalloc(&psi[i], D*sizeof(float)));
    }

    // Allocate messages (1×1 each = scalar)
    std::vector<float*> msgs(n_bonds);
    for (int i=0; i<n_bonds; i++) {
        HIP_CHECK(hipMalloc(&msgs[i], 1*sizeof(float)));
        float one = 1.0f;
        HIP_CHECK(hipMemcpy(msgs[i], &one, sizeof(float),
                            hipMemcpyHostToDevice));
    }

    // Update bond chis to 1
    for (auto& b : g.bonds) b.chi = 1;

    Measures meas(handle, ws);

    // ── Test 1: all I state ───────────────────────────────────────────
    {
        printf("--- Test 1: All-identity state ---\n");
        // Each site = [1, 1] (unnormalized)
        std::vector<float> h_psi_I = {1.f, 1.f};
        for (int i=0; i<n_sites; i++)
            HIP_CHECK(hipMemcpy(psi[i], h_psi_I.data(),
                D*sizeof(float), hipMemcpyHostToDevice));

        // norm = ∏_v (ψ_v[0] + ψ_v[1]) / ∏_e msg_trace
        //      = ∏_v 2 / ∏_e 1 = 2^4 = 16
        Scalar norm = meas.norm_sqr(g, psi, msgs);
        printf("  norm = %.4f (expected %.4f)\n", norm, std::pow(2.f, n_sites));
        assert(std::abs(norm - std::pow(2.f, n_sites)) < 1e-3f);
        printf("  norm: OK\n");

        // Z at any site = ψ[0] - ψ[1] = 1 - 1 = 0
        // numerator = 0 * (product of other vertex_scalars) = 0
        Scalar z0 = meas.autocorrelation(g, psi, msgs, 0);
        printf("  Z(0,0) = %.4f (expected 0.0)\n", z0);
        assert(std::abs(z0) < 1e-4f);
        printf("  Z at identity site: OK\n");
    }

    // ── Test 2: Z state at center ─────────────────────────────────────
    {
        printf("\n--- Test 2: Z state at center site ---\n");
        // Center site (0 for 2×2) = [1, 0] (Z eigenstate with eigenvalue +1)
        // All other sites = [1, 0] (also Z eigenstate)
        std::vector<float> h_psi_Z = {1.f, 0.f};
        for (int i=0; i<n_sites; i++)
            HIP_CHECK(hipMemcpy(psi[i], h_psi_Z.data(),
                D*sizeof(float), hipMemcpyHostToDevice));

        // norm = ∏_v (ψ[0]+ψ[1]) = ∏_v 1 = 1
        Scalar norm = meas.norm_sqr(g, psi, msgs);
        printf("  norm = %.4f (expected 1.0)\n", norm);
        assert(std::abs(norm - 1.f) < 1e-3f);

        // Z at center: ψ[0] - ψ[1] = 1 - 0 = 1
        // numerator = 1 × (other vertex_scalars=1) = 1
        // autocorr = 1/1 = 1
        int center = g.site_id(0,0);
        Scalar ac = meas.autocorrelation(g, psi, msgs, center);
        printf("  autocorrelation = %.4f (expected 1.0)\n", ac);
        assert(std::abs(ac - 1.f) < 1e-3f);
        printf("  Z state autocorrelation: OK\n");
    }

    // ── Test 3: mixed state ───────────────────────────────────────────
    {
        printf("\n--- Test 3: Mixed state ---\n");
        // Center site = [1, 0] (Z=+1), all others = [1, 1]
        int center = g.site_id(0,0);
        std::vector<float> h_psi_Z = {1.f, 0.f};
        std::vector<float> h_psi_I = {1.f, 1.f};

        for (int i=0; i<n_sites; i++) {
            auto& h = (i == center) ? h_psi_Z : h_psi_I;
            HIP_CHECK(hipMemcpy(psi[i], h.data(),
                D*sizeof(float), hipMemcpyHostToDevice));
        }

        // norm: center contributes 1, others contribute 2
        // norm = 1 × 2^3 = 8
        Scalar norm = meas.norm_sqr(g, psi, msgs);
        printf("  norm = %.4f (expected 8.0)\n", norm);
        assert(std::abs(norm - 8.f) < 1e-2f);

        // Z at center: num = (1-0) × 2^3 = 8, den = 8
        // autocorr = 8/8 = 1.0
        Scalar ac = meas.autocorrelation(g, psi, msgs, center);
        printf("  autocorrelation = %.4f (expected 1.0)\n", ac);
        assert(std::abs(ac - 1.f) < 1e-3f);

        // Z at non-center site (1): num = 0 × ... = 0
        // autocorr = 0
        Scalar ac_other = meas.autocorrelation(g, psi, msgs, 1);
        printf("  Z at identity site = %.4f (expected 0.0)\n", ac_other);
        assert(std::abs(ac_other) < 1e-4f);
        printf("  Mixed state: OK\n");
    }

    // ── Test 4: non-trivial messages ──────────────────────────────────
    {
        printf("\n--- Test 4: Non-trivial messages (chi=1, msg=2.0) ---\n");
        // With chi=1, message is scalar=2.0
        // edge_scalar = Tr(msg) = 2.0
        // vertex_scalar uses msg as multiplicative factor

        float msg_val = 2.0f;
        for (int i=0; i<n_bonds; i++)
            HIP_CHECK(hipMemcpy(msgs[i], &msg_val, sizeof(float),
                                hipMemcpyHostToDevice));

        // All Z eigenstates
        std::vector<float> h_psi = {1.f, 0.f};
        for (int i=0; i<n_sites; i++)
            HIP_CHECK(hipMemcpy(psi[i], h_psi.data(),
                D*sizeof(float), hipMemcpyHostToDevice));

        // vertex_scalar: ψ contracted with messages then summed
        // At chi=1: message just multiplies the tensor
        // For corner site (2 bonds): multiply by msg^2 = 4
        // vertex_scalar = 4 * (ψ[0]+ψ[1]) = 4 * 1 = 4
        // edge_scalar = Tr(msg) = 2

        // log Z = Σ_v log(vertex_scalar) - Σ_e log(edge_scalar)
        // For 2×2: 4 corners, all same
        // Each vertex: contract with 2 messages (chi=1 → multiply twice)
        // vertex_scalar = msg^degree × Σ_d ψ[d]
        // Corner degree=2: 2^2 * 1 = 4
        // log Z = 4*log(4) - 4*log(2) = 4*(2log2) - 4*log2 = 4*log2
        // Z = 2^4 = 16

        Scalar norm = meas.norm_sqr(g, psi, msgs);
        printf("  norm = %.4f\n", norm);

        // autocorr should still be 1 (Z state)
        int center = g.site_id(0,0);
        Scalar ac = meas.autocorrelation(g, psi, msgs, center);
        printf("  autocorrelation = %.4f (expected 1.0)\n", ac);
        assert(std::abs(ac - 1.f) < 1e-3f);
        printf("  Non-trivial messages: OK\n");
    }

    // ── Test 5: 4×4 grid at chi=1 ────────────────────────────────────
    {
        printf("\n--- Test 5: 4×4 grid at chi=1 ---\n");

        // Cleanup 2×2
        for (auto p : psi)  HIP_CHECK(hipFree(p));
        for (auto p : msgs) HIP_CHECK(hipFree(p));

        Grid g4(4, 4, 1);
        for (auto& b : g4.bonds) b.chi = 1;

        int ns = g4.n_sites, nb = g4.n_bonds;
        std::vector<float*> psi4(ns), msgs4(nb);
        for (int i=0; i<ns; i++) {
            HIP_CHECK(hipMalloc(&psi4[i], D*sizeof(float)));
        }
        for (int i=0; i<nb; i++) {
            HIP_CHECK(hipMalloc(&msgs4[i], sizeof(float)));
            float one = 1.0f;
            HIP_CHECK(hipMemcpy(msgs4[i], &one, sizeof(float),
                                hipMemcpyHostToDevice));
        }

        // All Z eigenstates
        std::vector<float> h_psi = {1.f, 0.f};
        for (int i=0; i<ns; i++)
            HIP_CHECK(hipMemcpy(psi4[i], h_psi.data(),
                D*sizeof(float), hipMemcpyHostToDevice));

        // norm = ∏_v 1 = 1, Z = 1 everywhere
        Scalar norm = meas.norm_sqr(g4, psi4, msgs4);
        printf("  norm = %.4f (expected 1.0)\n", norm);
        assert(std::abs(norm - 1.f) < 1e-2f);

        // Center site of 4×4 grid
        int center = g4.site_id(1,1);
        Scalar ac = meas.autocorrelation(g4, psi4, msgs4, center);
        printf("  autocorrelation at center = %.4f (expected 1.0)\n", ac);
        assert(std::abs(ac - 1.f) < 1e-3f);
        printf("  4×4 grid: OK\n");

        for (auto p : psi4)  HIP_CHECK(hipFree(p));
        for (auto p : msgs4) HIP_CHECK(hipFree(p));
    }

    rocblas_destroy_handle(handle);
    printf("\n=== ALL MEASURES TESTS PASSED ===\n");
    return 0;
}
