/*
 * test_heavyhex.cpp — Validation tests for heavy-hex grid and gates
 *
 * Tests:
 *   1. PTM correctness: XxYyZz(0.1) vs Julia's numerical PTM
 *   2. Grid topology: 68 sites, 76 bonds, degree distribution, 3 colors
 *   3. State initialization: center=Z, others=I, chi=1
 *   4. C(t=0) = 1 before any gates
 *   5. Bond dim growth after 1 Trotter step
 *
 * Build: make PHYS_DIM=4 test_heavyhex
 * Run:   ./test/test_heavyhex
 */

#include "types.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"
#include "workspace.hpp"
#include "measures.hpp"
#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>
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

/* ── Test 1: PTM correctness ──────────────────────────────────────── */
void test_ptm() {
    printf("\n[Test 1] XxYyZz PTM at theta=0.1\n");

    // Julia's numerical PTM at theta=0.1
    // a=cos²(0.2)=0.9605305, b=sin²(0.2)=0.0394695, g=sin(0.2)cos(0.2)=0.1947092
    float theta = 0.1f;
    float c2 = std::cos(2.f*theta), s2 = std::sin(2.f*theta);
    float a = c2*c2, b = s2*s2, g = s2*c2;

    auto M = HeavyHexGates::XxYyZz(theta);
    auto get = [&](int row, int col) { return M[row + 16*col]; };

    // Invariant elements
    CHECK_NEAR(get( 0, 0), 1.f,  1e-6f, "II → II");
    CHECK_NEAR(get( 5, 5), 1.f,  1e-6f, "XX → XX");
    CHECK_NEAR(get(10,10), 1.f,  1e-6f, "YY → YY");
    CHECK_NEAR(get(15,15), 1.f,  1e-6f, "ZZ → ZZ");

    // Block A spot checks (from Julia output)
    CHECK_NEAR(get( 1, 1),  a,   1e-5f, "PTM[IX,IX] = a");
    CHECK_NEAR(get( 4, 1),  b,   1e-5f, "PTM[XI,IX] = b");
    CHECK_NEAR(get(11, 1), -g,   1e-5f, "PTM[YZ,IX] = -g");
    CHECK_NEAR(get(14, 1),  g,   1e-5f, "PTM[ZY,IX] = +g");

    // Block B spot checks
    CHECK_NEAR(get( 2, 2),  a,   1e-5f, "PTM[IY,IY] = a");
    CHECK_NEAR(get( 8, 2),  b,   1e-5f, "PTM[YI,IY] = b");
    CHECK_NEAR(get( 7, 2),  g,   1e-5f, "PTM[XZ,IY] = +g");
    CHECK_NEAR(get(13, 2), -g,   1e-5f, "PTM[ZX,IY] = -g");

    // Block C spot checks
    CHECK_NEAR(get( 3, 3),  a,   1e-5f, "PTM[IZ,IZ] = a");
    CHECK_NEAR(get(12, 3),  b,   1e-5f, "PTM[ZI,IZ] = b");
    CHECK_NEAR(get( 6, 3), -g,   1e-5f, "PTM[XY,IZ] = -g");
    CHECK_NEAR(get( 9, 3),  g,   1e-5f, "PTM[YX,IZ] = +g");

    // Zero elements (off-block)
    CHECK_NEAR(get( 0, 1), 0.f,  1e-6f, "PTM[II,IX] = 0 (off-block)");
    CHECK_NEAR(get( 5, 1), 0.f,  1e-6f, "PTM[XX,IX] = 0 (off-block)");
    CHECK_NEAR(get( 2, 1), 0.f,  1e-6f, "PTM[IY,IX] = 0 (cross-block)");

    // Verify PTM is column-stochastic in L1 norm (each col sums to 1 for unitary)
    // Actually for a unitary channel: Tr(PTM) = sum of diagonal = preserved trace
    // Simpler check: column norms
    for (int col = 0; col < 16; col++) {
        float col_sq = 0.f;
        for (int row = 0; row < 16; row++) col_sq += get(row,col)*get(row,col);
        if (std::fabs(col_sq - 1.f) > 1e-4f) {
            printf("  FAIL: col %d L2-norm = %.6f (expected 1.0)\n", col, std::sqrt(col_sq));
            n_fail++;
            return;
        }
    }
    printf("  PASS: all 16 columns have unit L2-norm (unitary PTM)\n");
    n_pass++;
}

/* ── Test 2: Grid topology ────────────────────────────────────────── */
void test_grid() {
    printf("\n[Test 2] FlatHeavyHexGrid topology\n");

    FlatHeavyHexGrid grid(100);

    CHECK(grid.n_sites == 68, "n_sites = 68");
    CHECK(grid.n_bonds == 76, "n_bonds = 76");

    int d2 = 0, d3 = 0;
    for (auto& s : grid.sites) {
        if (s.degree == 2) d2++;
        else if (s.degree == 3) d3++;
    }
    CHECK(d2 == 52, "degree-2 sites = 52");
    CHECK(d3 == 16, "degree-3 sites = 16");
    CHECK(d2 + d3 == 68, "all sites accounted for");

    CHECK(grid.colors.size() == 3, "3 color classes");
    CHECK((int)grid.colors[0].size() == 25, "color 0: 25 gates");
    CHECK((int)grid.colors[1].size() == 25, "color 1: 25 gates");
    CHECK((int)grid.colors[2].size() == 26, "color 2: 26 gates");

    int total_gates = 0;
    for (auto& c : grid.colors) total_gates += (int)c.size();
    CHECK(total_gates == 76, "total gates = n_bonds = 76");

    // Verify no two gates in same color share a site
    for (int c = 0; c < 3; c++) {
        std::vector<bool> used(68, false);
        bool ok = true;
        for (auto& gate : grid.colors[c]) {
            if (used[gate.site_a] || used[gate.site_b]) { ok = false; break; }
            used[gate.site_a] = used[gate.site_b] = true;
        }
        char msg[64];
        snprintf(msg, sizeof(msg), "color %d: no shared sites", c);
        CHECK(ok, msg);
    }

    // Verify bond_local is set for all sites
    bool local_ok = true;
    for (auto& s : grid.sites) {
        int local = 0;
        for (int d = 0; d < MAX_DEGREE; d++)
            if (s.bond_id[d] >= 0 && s.bond_local[d] != local++) local_ok = false;
    }
    CHECK(local_ok, "bond_local indices consistent");

    grid.print();
}

/* ── Test 3: State initialization ────────────────────────────────── */
void test_state_init(rocblas_handle handle) {
    printf("\n[Test 3] State initialization\n");

    FlatHeavyHexGrid grid(10);
    HeavyHexState state(grid);

    int center = 2;  // site index 2 (Julia's obs_site=3, 0-based = 2)
    state.init_heisenberg(grid, center);

    // Read back center site
    std::vector<Scalar> buf(D);
    HIP_CHECK(hipMemcpy(buf.data(), state.psi[center],
        D*sizeof(Scalar), hipMemcpyDeviceToHost));
    CHECK_NEAR(buf[0], 0.f, 1e-6f, "center site: a_I = 0");
    CHECK_NEAR(buf[1], 0.f, 1e-6f, "center site: a_X = 0");
    CHECK_NEAR(buf[2], 0.f, 1e-6f, "center site: a_Y = 0");
    CHECK_NEAR(buf[3], 1.f, 1e-6f, "center site: a_Z = 1");

    // Read back a non-center site
    int other = (center == 0) ? 1 : 0;
    HIP_CHECK(hipMemcpy(buf.data(), state.psi[other],
        D*sizeof(Scalar), hipMemcpyDeviceToHost));
    CHECK_NEAR(buf[0], 1.f, 1e-6f, "other site: a_I = 1");
    CHECK_NEAR(buf[1], 0.f, 1e-6f, "other site: a_X = 0");
    CHECK_NEAR(buf[2], 0.f, 1e-6f, "other site: a_Y = 0");
    CHECK_NEAR(buf[3], 0.f, 1e-6f, "other site: a_Z = 0");

    // All chi = 1
    bool chi_ok = true;
    for (auto& b : grid.bonds) if (b.chi != 1) chi_ok = false;
    CHECK(chi_ok, "all bond chi = 1 after init");
}

/* ── Test 4: C(t=0) = 1 ──────────────────────────────────────────── */
// NOTE: Measures is currently tied to Grid& (rectangular lattice).
// FlatHeavyHexGrid support requires templating Measures or a common base class.
// This test is a placeholder — will be enabled after the interface is unified.
void test_ct0(rocblas_handle handle) {
    printf("\n[Test 4] C(t=0) = 1  [SKIPPED — Measures/Grid interface pending]\n");
    printf("  SKIP: Measures::autocorrelation_with_bp requires Grid& not FlatHeavyHexGrid\n");
}

/* ── Test 5: Bond dim growth ─────────────────────────────────────── */
void test_bond_growth(rocblas_handle handle) {
    printf("\n[Test 5] Bond dim growth after 1 Trotter step\n");

    FlatHeavyHexGrid grid(20);
    HeavyHexState state(grid);
    Workspace ws(20);

    int center = 2;  // 0-based; Julia's obs_site=3 (1-based) → site id 2
    state.init_heisenberg(grid, center);

    HeavyHexSimulator sim(handle, 1.0f, 0.1f);
    LayerMetrics lm = {};
    int max_chi = sim.trotter_layer(state, grid, ws, 20, 1e-10f, &lm);

    printf("  max_chi after 1 step = %d\n", max_chi);
    CHECK(max_chi > 1, "chi grew > 1 after 1 Trotter step");
    CHECK(max_chi <= 20, "chi <= chi_max");
    printf("  gate_err = %.2e  n_gates = %d\n", lm.gate_err, lm.n_gates);
    CHECK(lm.n_gates > 0, "gates were applied");
}

/* ── main ──────────────────────────────────────────────────────────── */
int main() {
    printf("═══════════════════════════════════════\n");
    printf("  test_heavyhex — PHYS_DIM=%d\n", D);
    printf("═══════════════════════════════════════\n");

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    test_ptm();
    test_grid();
    test_state_init(handle);
    test_ct0(handle);
    test_bond_growth(handle);

    rocblas_destroy_handle(handle);

    printf("\n═══════════════════════════════════════\n");
    printf("  Results: %d passed, %d failed\n", n_pass, n_fail);
    printf("═══════════════════════════════════════\n");

    return n_fail > 0 ? 1 : 0;
}
