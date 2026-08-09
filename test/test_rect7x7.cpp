/*
 * test_rect7x7.cpp — Validate FlatRect7x7Grid construction and Trotter step.
 *
 * Tests:
 *   1. Grid construction: 49 sites, 84 bonds, 4 colors, 21 gates/color
 *   2. Color independence: no two gates in same color share a site
 *   3. HeavyHexState allocation from generic Grid
 *   4. One Trotter step completes without crash
 *   5. Autocorrelation C(t) is finite after Trotter
 *
 * Build: make PHYS_DIM=4 test_rect7x7
 * Run:   ./test/test_rect7x7
 */

#include "types.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "measures.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"
#include "trotter.hpp"
#include "rect_7x7.hpp"
#include "rect_7x7_ideal_kick_eps010_data.hpp"

#include <cstdio>
#include <cmath>
#include <set>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); n_pass++; } \
    else      { printf("  FAIL: %s\n", msg); n_fail++; } \
} while(0)

int main() {
    printf("=== test_rect7x7 ===\n\n");

    // ── Test 1: Grid construction ─────────────────────────────────────────────
    printf("1. Grid construction\n");
    FlatRect7x7Grid grid(20);
    CHECK(grid.n_sites == 49,    "n_sites == 49");
    CHECK(grid.n_bonds == 84,    "n_bonds == 84");
    CHECK((int)grid.colors.size() == 4, "4 color classes");
    for (int c = 0; c < 4; c++) {
        char msg[64]; snprintf(msg, sizeof(msg), "color %d has 21 gates", c);
        CHECK(grid.colors[c].size() == 21, msg);
    }

    // ── Test 2: Color independence ────────────────────────────────────────────
    printf("\n2. Color independence (no shared sites within color)\n");
    bool all_independent = true;
    for (int c = 0; c < 4; c++) {
        std::set<int> used;
        for (auto& g : grid.colors[c]) {
            if (used.count(g.site_a) || used.count(g.site_b)) {
                all_independent = false;
                printf("  FAIL: color %d has conflict at bond %d (sites %d,%d)\n",
                       c, g.bond_id, g.site_a, g.site_b);
            }
            used.insert(g.site_a);
            used.insert(g.site_b);
        }
    }
    CHECK(all_independent, "all colors are independent (no shared sites)");

    // ── Test 3: Max degree ────────────────────────────────────────────────────
    printf("\n3. Site degrees\n");
    int max_deg = 0, n_deg4 = 0, n_deg3 = 0, n_deg2 = 0;
    for (auto& site : grid.sites) {
        int deg = 0;
        for (int d = 0; d < MAX_DEGREE; d++) if (site.bond_id[d] >= 0) deg++;
        max_deg = std::max(max_deg, deg);
        if (deg==4) n_deg4++;
        else if (deg==3) n_deg3++;
        else if (deg==2) n_deg2++;
    }
    CHECK(max_deg == 4, "max degree = 4 (interior sites)");
    CHECK(n_deg4 == 25, "25 interior sites (degree 4)");   // 5×5 interior
    CHECK(n_deg3 == 20, "20 edge sites (degree 3)");        // 4×5 edges
    CHECK(n_deg2 == 4,  "4 corner sites (degree 2)");

    // ── Test 4: HeavyHexState allocation ──────────────────────────────────────
    printf("\n4. HeavyHexState allocation\n");
    int max_deg_ws = max_deg;
    Workspace ws(20, max_deg_ws);
    rocblas_handle handle; rocblas_create_handle(&handle);

    bool state_ok = true;
    HeavyHexState* state = nullptr;
    try {
        state = new HeavyHexState(static_cast<Grid&>(grid));
        state->init_heisenberg(static_cast<Grid&>(grid), 24);  // center of 7×7
    } catch (...) {
        state_ok = false;
    }
    CHECK(state_ok, "HeavyHexState allocated and initialized");

    // ── Test 5: Trotter step ──────────────────────────────────────────────────
    printf("\n5. Trotter step (chi=20, cutoff=1e-3)\n");
    if (state_ok) {
        GateOps gate_ops(handle);
        PtmFn ptm = [](int b) -> const float* { return RECT_7x7_IDEAL_KICK_EPS010[b]; };
        bool trotter_ok = true;
        try {
            do_trotter(state->psi, state->messages,
                       static_cast<Grid&>(grid),
                       ptm, nullptr,
                       RECT_7x7_IDEAL_KICK_EPS010_SITE_PTM,
                       ws, gate_ops, 20, 1e-3f);
        } catch (...) {
            trotter_ok = false;
        }
        CHECK(trotter_ok, "do_trotter completed without exception");

        // ── Test 6: Autocorrelation ───────────────────────────────────────────
        printf("\n6. Autocorrelation C(t)\n");
        if (trotter_ok) {
            Measures meas(handle, ws);
            float C = meas.autocorrelation_with_bp(
                static_cast<Grid&>(grid), state->psi, state->messages, 24, 10);
            printf("  C(t=1) = %+.6f\n", C);
            CHECK(std::isfinite(C),      "C(t) is finite");
            CHECK(std::abs(C) <= 1.01f, "C(t) in [-1, 1]");
        }
        delete state;
    }

    rocblas_destroy_handle(handle);

    // ── Summary ───────────────────────────────────────────────────────────────
    printf("\n=== Summary: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}
