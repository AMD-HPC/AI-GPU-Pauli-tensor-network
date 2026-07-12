/*
 * test_overlap_cpu.cpp — Validation tests for CPU inner product
 *
 * Tests:
 *   1. At t=0: ⟨ψ_nl|ψ_n⟩ = 1 for all gammas (same initial state)
 *   2. After evolution: ⟨ψ_nl|ψ_nl⟩ = C_nl (self-overlap = autocorrelation)
 *   3. After evolution: ⟨ψ_nl|ψ_n(γ)⟩ differs from ⟨ψ_nl|ψ_nl⟩
 *
 * Build: make PHYS_DIM=4 test_overlap_cpu
 */

#include "types.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"
#include "measures.hpp"
#include "workspace.hpp"
#include "overlap_cpu.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <rocblas/rocblas.h>

static int n_pass = 0, n_fail = 0;

#define CHECK_NEAR(a, b, tol, msg) do { \
    float diff = std::fabs((float)(a) - (float)(b)); \
    if (diff <= tol) { printf("  PASS: %s  (%.6f ≈ %.6f)\n", msg, (float)(a), (float)(b)); n_pass++; } \
    else             { printf("  FAIL: %s  (|%.6f - %.6f| = %.2e > %.2e)\n", msg, (float)(a), (float)(b), diff, tol); n_fail++; } \
} while(0)

#define CHECK_LT(a, b, msg) do { \
    if ((float)(a) < (float)(b)) { printf("  PASS: %s\n", msg); n_pass++; } \
    else { printf("  FAIL: %s  (%.6f >= %.6f)\n", msg, (float)(a), (float)(b)); n_fail++; } \
} while(0)

#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); n_pass++; } \
    else      { printf("  FAIL: %s\n", msg); n_fail++; } \
} while(0)

int main() {
    printf("═══════════════════════════════════════\n");
    printf("  test_overlap_cpu — PHYS_DIM=%d\n", D);
    printf("═══════════════════════════════════════\n");

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    int chi_max = 20;
    int center  = 2;

    // ── Test 1: t=0, overlap = 1 for all gammas ──────────────────
    printf("\n[Test 1] At t=0: overlap(nl, n_gamma) = 1\n");
    {
        FlatHeavyHexGrid  grid_nl(chi_max);
        HeavyHexState     state_nl(grid_nl);
        Workspace         ws_nl(chi_max, 3);
        state_nl.init_heisenberg(grid_nl, center);

        // Download noiseless at t=0
        hipSetDevice(0);
        auto h_psi_nl  = download_psi(state_nl.psi, grid_nl, chi_max);
        auto h_msg_nl  = download_messages_diag(state_nl.messages, grid_nl, chi_max);

        for (float gamma : {1.0f, 1.4f, 1.8f}) {
            NoisyHeavyHexGrid grid_n(chi_max, gamma);
            HeavyHexState     state_n(grid_n);
            state_n.init_heisenberg(grid_n, center);

            auto h_psi_n = download_psi(state_n.psi, grid_n, chi_max);
            float ov = cpu_bethe_inner(h_psi_nl, h_psi_n, h_msg_nl, grid_nl, chi_max);
            char msg[64];
            snprintf(msg, sizeof(msg), "overlap(nl, n_%.1f) at t=0 ≈ 1", gamma);
            CHECK_NEAR(ov, 1.0f, 0.01f, msg);
        }
    }

    // ── Test 2: self-overlap = autocorrelation ────────────────────
    printf("\n[Test 2] Self-overlap(nl, nl) ≈ C_nl after 3 steps\n");
    {
        FlatHeavyHexGrid   grid_nl(chi_max);
        HeavyHexState      state_nl(grid_nl);
        Workspace          ws_nl(chi_max, 3);
        Measures           meas_nl(handle, ws_nl);
        HeavyHexSimulator  sim_nl(handle, 1.0f, 0.1f);
        state_nl.init_heisenberg(grid_nl, center);

        for (int i = 0; i < 3; i++)
            sim_nl.trotter_layer(state_nl, grid_nl, ws_nl, chi_max, 1e-4f);

        // GPU autocorrelation
        double tb=0, tp=0; int sw=0; float dl=0.f;
        Scalar C_nl = meas_nl.autocorrelation_with_bp(
            static_cast<Grid&>(grid_nl), state_nl.psi, state_nl.messages,
            center, 100, &tb, &tp, &sw, &dl);

        printf("  GPU C_nl = %.6f\n", C_nl);

        // CPU self-overlap
        auto h_psi_nl = download_psi(state_nl.psi, grid_nl, chi_max);
        auto h_msg_nl = download_messages_diag(state_nl.messages, grid_nl, chi_max);
        float ov_self = cpu_bethe_inner(h_psi_nl, h_psi_nl, h_msg_nl, grid_nl, chi_max);
        printf("  CPU self-overlap = %.6f\n", ov_self);

        // They should be close (same Bethe approximation, different implementation)
        CHECK_NEAR(ov_self, C_nl, 0.05f, "CPU self-overlap ≈ GPU C_nl");
    }

    // ── Test 3: overlap(nl, n_gamma) differs between gammas ───────
    printf("\n[Test 3] Overlaps differ between gammas after 5 steps\n");
    {
        FlatHeavyHexGrid   grid_nl(chi_max);
        HeavyHexState      state_nl(grid_nl);
        Workspace          ws_nl(chi_max, 3);
        HeavyHexSimulator  sim_nl(handle, 1.0f, 0.1f);
        state_nl.init_heisenberg(grid_nl, center);
        for (int i = 0; i < 5; i++)
            sim_nl.trotter_layer(state_nl, grid_nl, ws_nl, chi_max, 1e-4f);

        auto h_psi_nl = download_psi(state_nl.psi, grid_nl, chi_max);
        auto h_msg_nl = download_messages_diag(state_nl.messages, grid_nl, chi_max);

        float prev_ov = 2.0f;
        for (float gamma : {1.0f, 1.4f, 1.8f}) {
            NoisyHeavyHexGrid    grid_n(chi_max, gamma);
            HeavyHexState        state_n(grid_n);
            Workspace            ws_n(chi_max, 3);
            NoisyHeavyHexSimulator sim_n(handle, 1.0f, 0.1f);
            state_n.init_heisenberg(grid_n, center);
            for (int i = 0; i < 5; i++)
                sim_n.trotter_layer(state_n, grid_n, ws_n, chi_max, 1e-4f);

            auto h_psi_n = download_psi(state_n.psi, grid_n, chi_max);
            float ov = cpu_bethe_inner(h_psi_nl, h_psi_n, h_msg_nl, grid_nl, chi_max);
            printf("  gamma=%.1f  overlap=%.6f\n", gamma, ov);

            char msg[64];
            snprintf(msg, sizeof(msg), "overlap(gamma=%.1f) != overlap(prev)", gamma);
            CHECK(std::fabs(ov - prev_ov) > 1e-4f, msg);
            prev_ov = ov;
        }
    }

    rocblas_destroy_handle(handle);

    printf("\n═══════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", n_pass, n_fail);
    printf("═══════════════════════════════════════\n");
    return n_fail > 0 ? 1 : 0;
}
