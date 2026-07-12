/*
 * test_noisy_heavyhex.cpp — Validation tests for NoisyHeavyHexGrid
 *
 * Tests:
 *   1. Noise PTM structure — diagonal dominance (near-identity for small gamma)
 *   2. Noisy C(t=0) = 1 before any gates
 *   3. Noisy C(t=1) < Noiseless C(t=1) — noise reduces autocorrelation
 *   4. Higher gamma → lower C(t=1) — stronger noise = more decay
 *
 * Build: make PHYS_DIM=4 test_noisy_heavyhex
 * Run:   ./test/test_noisy_heavyhex
 */

#include "types.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"
#include "workspace.hpp"
#include "measures.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <rocblas/rocblas.h>

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); n_pass++; } \
    else      { printf("  FAIL: %s\n", msg); n_fail++; } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) do { \
    float diff = std::fabs((float)(a) - (float)(b)); \
    if (diff <= tol) { printf("  PASS: %s  (%.6f ≈ %.6f)\n", msg, (float)(a), (float)(b)); n_pass++; } \
    else             { printf("  FAIL: %s  (|%.6f - %.6f| = %.2e > %.2e)\n", msg, (float)(a), (float)(b), diff, tol); n_fail++; } \
} while(0)

#define CHECK_LT(a, b, msg) do { \
    if ((float)(a) < (float)(b)) { printf("  PASS: %s  (%.6f < %.6f)\n", msg, (float)(a), (float)(b)); n_pass++; } \
    else                          { printf("  FAIL: %s  (%.6f >= %.6f)\n", msg, (float)(a), (float)(b)); n_fail++; } \
} while(0)

/* ── Test 1: Noise PTM structure ─────────────────────────────────── */
void test_noise_ptm_structure() {
    printf("\n[Test 1] Noise PTM structure\n");

    // For small gamma=1.0, noise is weak → PTM close to identity
    // Diagonal elements should be close to 1, off-diagonal small
    NoisyHeavyHexGrid grid(10, 1.0f);

    const float* ptm = grid.noise_ptm(0);  // bond 0

    // II element (0,0) should be ~1
    CHECK_NEAR(ptm[0 + 16*0], 1.0f, 1e-3f, "noise PTM[II,II] ≈ 1 (gamma=1.0)");

    // Check diagonal dominance: each diagonal element > sum of off-diagonal in its column
    bool diag_dominant = true;
    for (int col = 0; col < 16; col++) {
        float diag = std::fabs(ptm[col + 16*col]);
        float offdiag = 0.f;
        for (int row = 0; row < 16; row++)
            if (row != col) offdiag += std::fabs(ptm[row + 16*col]);
        if (diag < offdiag) diag_dominant = false;
    }
    CHECK(diag_dominant, "noise PTM is diagonally dominant (near-identity)");

    // gamma=1.8 should have larger off-diagonal elements than gamma=1.0
    NoisyHeavyHexGrid grid18(10, 1.8f);
    const float* ptm18 = grid18.noise_ptm(0);
    float offdiag10 = 0.f, offdiag18 = 0.f;
    for (int col = 0; col < 16; col++)
        for (int row = 0; row < 16; row++)
            if (row != col) {
                offdiag10 += std::fabs(ptm[row + 16*col]);
                offdiag18 += std::fabs(ptm18[row + 16*col]);
            }
    CHECK_LT(offdiag10, offdiag18, "gamma=1.0 has smaller off-diagonal than gamma=1.8");

    printf("  off-diagonal sum: gamma=1.0: %.4f  gamma=1.8: %.4f\n",
           offdiag10, offdiag18);
}

/* ── Test 2: All three gammas available ──────────────────────────── */
void test_gamma_lookup() {
    printf("\n[Test 2] Gamma lookup\n");

    for (float g : {1.0f, 1.4f, 1.8f}) {
        NoisyHeavyHexGrid grid(10, g);
        const float* ptm = grid.noise_ptm(0);
        CHECK(ptm != nullptr, (std::string("gamma=") + std::to_string(g) + " PTM not null").c_str());
        CHECK_NEAR(ptm[0], 1.0f, 0.1f, (std::string("gamma=") + std::to_string(g) + " PTM[0,0] ≈ 1").c_str());
    }

    // Nearest gamma: 1.2 → should give gamma=1.0 PTM (closer than 1.4)
    NoisyHeavyHexGrid g12(10, 1.2f);
    NoisyHeavyHexGrid g10(10, 1.0f);
    CHECK(g12.noise_ptm(0) == g10.noise_ptm(0), "gamma=1.2 → nearest is gamma=1.0");

    // Nearest gamma: 1.3 → should give gamma=1.4 PTM
    NoisyHeavyHexGrid g13(10, 1.3f);
    NoisyHeavyHexGrid g14(10, 1.4f);
    CHECK(g13.noise_ptm(0) == g14.noise_ptm(0), "gamma=1.3 → nearest is gamma=1.4");
}

/* ── Test 3: Noisy C(t=1) < Noiseless C(t=1) ────────────────────── */
void test_noisy_reduces_ct(rocblas_handle handle) {
    printf("\n[Test 3] Noisy C(t=1) < Noiseless C(t=1)\n");

    int chi_max = 20;
    int center  = 13;

    // Noiseless
    FlatHeavyHexGrid   grid_nl(chi_max);
    HeavyHexState      state_nl(grid_nl);
    Workspace          ws_nl(chi_max, 3);
    Measures           meas_nl(handle, ws_nl);
    HeavyHexSimulator  sim_nl(handle, 1.0f, 0.1f);
    state_nl.init_heisenberg(grid_nl, center);
    LayerMetrics lm = {};
    sim_nl.trotter_layer(state_nl, grid_nl, ws_nl, chi_max, 1e-10f, &lm);
    double t_bp=0, t_pf=0; int sweeps=0; float delta=0.f;
    Scalar C_nl = meas_nl.autocorrelation_with_bp(
        static_cast<Grid&>(grid_nl), state_nl.psi, state_nl.messages,
        center, 100, &t_bp, &t_pf, &sweeps, &delta);
    printf("  Noiseless C(t=1) = %.6f\n", C_nl);

    // Noisy gamma=1.0
    NoisyHeavyHexGrid    grid_n(chi_max, 1.0f);
    HeavyHexState        state_n(grid_n);
    Workspace            ws_n(chi_max, 3);
    Measures             meas_n(handle, ws_n);
    NoisyHeavyHexSimulator sim_n(handle, 1.0f, 0.1f);
    state_n.init_heisenberg(grid_n, center);
    sim_n.trotter_layer(state_n, grid_n, ws_n, chi_max, 1e-10f);
    Scalar C_n = meas_n.autocorrelation_with_bp(
        static_cast<Grid&>(grid_n), state_n.psi, state_n.messages,
        center, 100, &t_bp, &t_pf, &sweeps, &delta);
    printf("  Noisy   C(t=1) = %.6f  (gamma=1.0)\n", C_n);

    CHECK_LT(C_n, C_nl, "Noisy C(t=1) < Noiseless C(t=1)");
    CHECK_NEAR(C_nl, 0.886f, 0.05f, "Noiseless C(t=1) ≈ 0.886");
}

/* ── Test 4: Higher gamma → lower C(t=1) ────────────────────────── */
void test_gamma_ordering(rocblas_handle handle) {
    printf("\n[Test 4] Higher gamma → lower C(t=1)\n");

    int chi_max = 10;
    int center  = 13;
    float prev_C = 1.0f;

    for (float g : {1.0f, 1.4f, 1.8f}) {
        NoisyHeavyHexGrid    grid(chi_max, g);
        HeavyHexState        state(grid);
        Workspace            ws(chi_max, 3);
        Measures             meas(handle, ws);
        NoisyHeavyHexSimulator sim(handle, 1.0f, 0.1f);
        state.init_heisenberg(grid, center);
        sim.trotter_layer(state, grid, ws, chi_max, 1e-10f);
        double t_bp=0, t_pf=0; int sweeps=0; float delta=0.f;
        Scalar C = meas.autocorrelation_with_bp(
            static_cast<Grid&>(grid), state.psi, state.messages,
            center, 100, &t_bp, &t_pf, &sweeps, &delta);
        printf("  gamma=%.1f  C(t=1) = %.6f\n", g, C);
        char msg[64];
        snprintf(msg, sizeof(msg), "C(gamma=%.1f) < C(gamma=prev)", g);
        CHECK_LT(C, prev_C, msg);
        prev_C = C;
    }
}

/* ── main ──────────────────────────────────────────────────────────── */
int main() {
    printf("═══════════════════════════════════════\n");
    printf("  test_noisy_heavyhex — PHYS_DIM=%d\n", D);
    printf("═══════════════════════════════════════\n");

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    test_noise_ptm_structure();
    test_gamma_lookup();
    test_noisy_reduces_ct(handle);
    test_gamma_ordering(handle);

    rocblas_destroy_handle(handle);

    printf("\n═══════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", n_pass, n_fail);
    printf("═══════════════════════════════════════\n");

    return n_fail > 0 ? 1 : 0;
}
