/*
 * test_overlap_gpu.cpp — Validation tests for GPU P2P inner product
 *
 * Tests:
 *   1. At t=0: overlap(nl, n_gamma) = 1 for all gammas
 *   2. Self-overlap(nl, nl) ≈ C_nl (autocorrelation)
 *   3. After evolution: overlaps differ between gammas
 *
 * Build: make PHYS_DIM=4 test_overlap_gpu
 * Run:   ./test/test_overlap_gpu  (requires 2+ GPUs with P2P)
 */

#include "types.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"
#include "measures.hpp"
#include "workspace.hpp"
#include "overlap_gpu.hpp"
#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

static int n_pass = 0, n_fail = 0;

#define CHECK_NEAR(a, b, tol, msg) do { \
    float diff = std::fabs((float)(a) - (float)(b)); \
    if (diff <= tol) { printf("  PASS: %s  (%.6f ≈ %.6f)\n", msg, (float)(a), (float)(b)); n_pass++; } \
    else { printf("  FAIL: %s  (|%.6f - %.6f| = %.2e > %.2e)\n", msg, (float)(a), (float)(b), diff, tol); n_fail++; } \
} while(0)

#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); n_pass++; } \
    else      { printf("  FAIL: %s\n", msg); n_fail++; } \
} while(0)

int main() {
    printf("═══════════════════════════════════════\n");
    printf("  test_overlap_gpu — PHYS_DIM=%d\n", D);
    printf("═══════════════════════════════════════\n");

    int n_gpus = 0;
    hipGetDeviceCount(&n_gpus);
    if (n_gpus < 2) {
        printf("  SKIP: need 2+ GPUs, found %d\n", n_gpus);
        return 0;
    }

    enable_p2p(n_gpus);

    int chi_max = 20;
    int chi_n   = 20;
    int center  = 2;

    // ── Setup GPU 0: noiseless ────────────────────────────────────
    hipSetDevice(0);
    rocblas_handle handle0;
    rocblas_create_handle(&handle0);
    FlatHeavyHexGrid  grid_nl(chi_max);
    Workspace         ws_nl(chi_max, 3);
    HeavyHexState     state_nl(grid_nl);
    Measures          meas_nl(handle0, ws_nl);
    HeavyHexSimulator sim_nl(handle0, 1.0f, 0.1f);
    state_nl.init_heisenberg(grid_nl, center);

    // ── Setup GPU 1: 3 noisy states ───────────────────────────────
    hipSetDevice(1);
    rocblas_handle handle1;
    rocblas_create_handle(&handle1);
    static constexpr float GAMMAS[3] = {1.0f, 1.4f, 1.8f};
    static constexpr int N_GAMMAS = 3;

    NoisyHeavyHexGrid*     grid_n[N_GAMMAS];
    HeavyHexState*         state_n[N_GAMMAS];
    Workspace*             ws_n[N_GAMMAS];
    NoisyHeavyHexSimulator* sim_n[N_GAMMAS];

    for (int g = 0; g < N_GAMMAS; g++) {
        grid_n[g]  = new NoisyHeavyHexGrid(chi_n, GAMMAS[g]);
        state_n[g] = new HeavyHexState(*grid_n[g]);
        ws_n[g]    = new Workspace(chi_n, 3);
        sim_n[g]   = new NoisyHeavyHexSimulator(handle1, 1.0f, 0.1f);
        state_n[g]->init_heisenberg(*grid_n[g], center);
    }

    const NoisyHeavyHexGrid* cgrid_n[N_GAMMAS];
    const HeavyHexState*     cstate_n[N_GAMMAS];
    for (int g = 0; g < N_GAMMAS; g++) {
        cgrid_n[g]  = grid_n[g];
        cstate_n[g] = state_n[g];
    }

    // ── Test 1: t=0, overlap = 1 ──────────────────────────────────
    printf("\n[Test 1] At t=0: overlap(nl, n_gamma) = 1\n");
    {
        // Run BP on noiseless to get messages
        hipSetDevice(0);
        double tb=0,tp=0; int sw=0; float dl=0.f;
        float C_nl0 = meas_nl.autocorrelation_with_bp(
            static_cast<Grid&>(grid_nl), state_nl.psi, state_nl.messages,
            center, 100, &tb, &tp, &sw, &dl);
        printf("  C_nl(t=0) = %.6f\n", C_nl0);

        float overlaps[N_GAMMAS];
        compute_overlaps_p2p(meas_nl,
            static_cast<Grid&>(grid_nl), state_nl.psi, state_nl.messages,
            center, cgrid_n, cstate_n, chi_n, N_GAMMAS, 0, 1,
            overlaps, nullptr);

        for (int g = 0; g < N_GAMMAS; g++) {
            char msg[64];
            snprintf(msg, sizeof(msg), "overlap(nl, n_%.1f) at t=0 ≈ 1", GAMMAS[g]);
            CHECK_NEAR(overlaps[g], 1.0f, 0.01f, msg);
        }
    }

    // ── Test 2: self-overlap = C_nl after evolution ───────────────
    printf("\n[Test 2] Self-overlap(nl,nl) ≈ C_nl after 3 steps\n");
    {
        // Evolve noiseless 3 steps
        hipSetDevice(0);
        for (int i = 0; i < 3; i++)
            sim_nl.trotter_layer(state_nl, grid_nl, ws_nl, chi_max, 1e-4f);

        double tb=0,tp=0; int sw=0; float dl=0.f;
        float C_nl = meas_nl.autocorrelation_with_bp(
            static_cast<Grid&>(grid_nl), state_nl.psi, state_nl.messages,
            center, 100, &tb, &tp, &sw, &dl);
        printf("  GPU C_nl = %.6f\n", C_nl);

        // Self-overlap: copy noiseless psi to GPU 0 as if it were "noisy"
        // (should give same result as autocorrelation)
        auto psi_self = p2p_copy_psi(state_nl.psi, grid_nl, chi_max, 0, 0);
        float ov_self = meas_nl.autocorrelation_with_bp(
            static_cast<Grid&>(grid_nl), psi_self, state_nl.messages,
            center, 100, &tb, &tp, &sw, &dl);
        free_psi(psi_self, 0);
        printf("  GPU self-overlap = %.6f\n", ov_self);
        CHECK_NEAR(ov_self, C_nl, 1e-4f, "Self-overlap == C_nl exactly");
    }

    // ── Test 3: overlaps differ between gammas ────────────────────
    printf("\n[Test 3] Overlaps differ between gammas after 5 steps\n");
    {
        // Evolve noisy 5 steps (noiseless already at step 3, evolve 2 more)
        hipSetDevice(0);
        for (int i = 0; i < 2; i++)
            sim_nl.trotter_layer(state_nl, grid_nl, ws_nl, chi_max, 1e-4f);

        hipSetDevice(1);
        for (int g = 0; g < N_GAMMAS; g++)
            for (int i = 0; i < 5; i++)
                sim_n[g]->trotter_layer(*state_n[g], *grid_n[g], *ws_n[g], chi_n, 1e-4f);

        // Rerun BP on noiseless
        hipSetDevice(0);
        double tb=0,tp=0; int sw=0; float dl=0.f;
        float C_nl;
        float overlaps[N_GAMMAS];
        compute_overlaps_p2p(meas_nl,
            static_cast<Grid&>(grid_nl), state_nl.psi, state_nl.messages,
            center, cgrid_n, cstate_n, chi_n, N_GAMMAS, 0, 1,
            overlaps, &C_nl);

        printf("  C_nl = %.6f\n", C_nl);
        float prev = 2.0f;
        for (int g = 0; g < N_GAMMAS; g++) {
            printf("  gamma=%.1f  overlap=%.6f\n", GAMMAS[g], overlaps[g]);
            char msg[64];
            snprintf(msg, sizeof(msg), "overlap(gamma=%.1f) != overlap(prev)", GAMMAS[g]);
            CHECK(std::fabs(overlaps[g] - prev) > 1e-4f, msg);
            prev = overlaps[g];
        }
    }

    // ── Cleanup ───────────────────────────────────────────────────
    hipSetDevice(0);
    rocblas_destroy_handle(handle0);
    hipSetDevice(1);
    for (int g = 0; g < N_GAMMAS; g++) {
        delete sim_n[g]; delete ws_n[g];
        delete state_n[g]; delete grid_n[g];
    }
    rocblas_destroy_handle(handle1);

    printf("\n═══════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", n_pass, n_fail);
    printf("═══════════════════════════════════════\n");
    return n_fail > 0 ? 1 : 0;
}
