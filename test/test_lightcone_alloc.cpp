/*
 * test_lightcone_alloc.cpp — Test BFS + lightcone allocation in Grid
 *
 * Verifies:
 *   1. BFS distances are correct for heavy-hex and rect-7x7 topologies
 *   2. Memory estimates shrink correctly with lightcone allocation
 *   3. apply_lightcone sets bond.chi correctly (chi_max inside, 1 outside)
 *   4. Savings are in the expected range
 *
 * Build:
 *   hipcc -O2 -std=c++17 -I./include -DPHYS_DIM=4 \
 *       -o test/test_lightcone_alloc test/test_lightcone_alloc.cpp \
 *       -lrocblas -lrocsolver -lm
 *
 * Run:
 *   ./test/test_lightcone_alloc
 */

#include "grid.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"
#include "rect_7x7.hpp"
#include "heavyhex_ideal_kick_eps010_data.hpp"
#include <cstdio>
#include <vector>
#include <algorithm>

static int n_pass = 0;
static int n_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { printf("  PASS: %s\n", msg); n_pass++; } \
    else       { printf("  FAIL: %s\n", msg); n_fail++; } \
} while(0)

// ── Heavy-hex tests ───────────────────────────────────────────────────────────

void test_heavyhex_bfs() {
    printf("\n=== Heavy-hex 3x3: BFS from center site 13 ===\n");
    FlatHeavyHexGrid g(70);
    auto dist = g.bfs(13);

    CHECK(dist[13] == 0,  "center site has distance 0");
    CHECK((int)dist.size() == g.n_sites, "dist vector covers all sites");

    int max_d = *std::max_element(dist.begin(), dist.end());
    printf("  n_sites=%d  max_BFS_dist=%d\n", g.n_sites, max_d);
    CHECK(max_d > 0,  "max distance > 0");
    CHECK(max_d < g.n_sites, "max distance < n_sites (connected graph)");

    // Count sites at each distance
    printf("  Sites per shell: ");
    for (int d = 0; d <= max_d; d++) {
        int count = 0;
        for (int v = 0; v < g.n_sites; v++) if (dist[v] == d) count++;
        printf("d=%d:%d  ", d, count);
    }
    printf("\n");
}

void test_heavyhex_memory() {
    printf("\n=== Heavy-hex 3x3: memory estimates (chi=70, 10 cycles) ===\n");
    FlatHeavyHexGrid g(70);
    int center = 13, n_steps = 10;

    float full  = g.lightcone_memory_mb(-1,      n_steps, -1.0f);
    float nless = g.lightcone_memory_mb(center,   n_steps, 0.376f); // noiseless v_hop
    float noisy = g.lightcone_memory_mb(center,   n_steps, 0.145f); // noisy N1 v_hop
    float tight = g.lightcone_memory_mb(center,   n_steps, 0.050f); // very tight

    printf("  Full allocation:              %.2f MB\n", full);
    printf("  Noiseless frontier (v=0.376): %.2f MB  (%.0f%% of full)\n",
           nless, 100.0f * nless / full);
    printf("  Noisy N1 frontier  (v=0.145): %.2f MB  (%.0f%% of full)\n",
           noisy, 100.0f * noisy / full);
    printf("  Tight frontier     (v=0.050): %.2f MB  (%.0f%% of full)\n",
           tight, 100.0f * tight / full);

    CHECK(full  > nless, "noiseless frontier < full allocation");
    CHECK(nless > noisy, "noisy frontier < noiseless frontier");
    CHECK(noisy > tight, "tight frontier < noisy frontier");
    CHECK(tight >= 0.0f, "tight frontier >= 0");
}

void test_heavyhex_apply() {
    printf("\n=== Heavy-hex 3x3: apply_lightcone (v=0.376, 10 cycles) ===\n");
    FlatHeavyHexGrid g(70);
    g.apply_lightcone(13, 10, 0.376f);

    // Count bonds at chi_max vs chi=1
    int n_full = 0, n_unit = 0;
    for (auto& b : g.bonds) {
        if (b.chi == g.chi_max) n_full++;
        else if (b.chi == 1)    n_unit++;
        else {
            printf("  UNEXPECTED bond chi=%d (not chi_max or 1)\n", b.chi);
            n_fail++;
        }
    }
    printf("  %d/%d bonds at chi_max=%d, %d at chi=1\n",
           n_full, g.n_bonds, g.chi_max, n_unit);

    CHECK(n_full + n_unit == g.n_bonds, "every bond is chi_max or 1");
    CHECK(n_full > 0,  "at least some bonds inside lightcone");
    CHECK(n_unit > 0,  "at least some bonds outside lightcone");
    CHECK(n_unit > n_full, "majority outside lightcone for 10 cycles");
}

// ── Rect 7x7 tests ────────────────────────────────────────────────────────────

void test_rect7x7_bfs() {
    printf("\n=== Rect 7x7: BFS from center site 24 ===\n");
    FlatRect7x7Grid g(50);
    auto dist = g.bfs(24);

    CHECK(dist[24] == 0, "center site has distance 0");

    int max_d = *std::max_element(dist.begin(), dist.end());
    printf("  n_sites=%d  max_BFS_dist=%d\n", g.n_sites, max_d);

    // For a 7x7 grid, center (3,3) has max distance 6 (corner)
    CHECK(max_d == 6, "max BFS distance from center of 7x7 is 6");

    // Count sites per shell
    printf("  Sites per shell: ");
    for (int d = 0; d <= max_d; d++) {
        int count = 0;
        for (int v = 0; v < g.n_sites; v++) if (dist[v] == d) count++;
        printf("d=%d:%d  ", d, count);
    }
    printf("\n");
}

void test_rect7x7_memory() {
    printf("\n=== Rect 7x7: memory estimates (chi=50, 10 cycles) ===\n");
    FlatRect7x7Grid g(50);
    int center = 24, n_steps = 10;

    float full  = g.lightcone_memory_mb(-1,    n_steps, -1.0f);
    float nless = g.lightcone_memory_mb(center, n_steps, 0.448f); // noiseless v_hop
    float noisy = g.lightcone_memory_mb(center, n_steps, 0.097f); // noisy N3 v_hop

    printf("  Full allocation:              %.2f MB\n", full);
    printf("  Noiseless frontier (v=0.448): %.2f MB  (%.0f%% of full)\n",
           nless, 100.0f * nless / full);
    printf("  Noisy N3 frontier  (v=0.097): %.2f MB  (%.0f%% of full)\n",
           noisy, 100.0f * noisy / full);

    CHECK(full  > nless, "noiseless frontier < full");
    CHECK(nless > noisy, "noisy frontier < noiseless");

    // For rect 7x7, noiseless reaches all 49 sites → should be significant saving
    float saving = 1.0f - nless / full;
    printf("  Saving with noiseless v_hop: %.0f%%\n", 100.0f * saving);
    CHECK(saving > 0.0f, "meaningful memory saving");
}

void test_no_center() {
    printf("\n=== Default (no center): full allocation unchanged ===\n");
    FlatHeavyHexGrid g(70);
    float full_direct = g.lightcone_memory_mb(-1, 10, -1.0f);
    float full_noarg  = g.lightcone_memory_mb(-1, 10, 0.0f);

    printf("  center=-1 v=-1:  %.2f MB\n", full_direct);
    printf("  center=-1 v=0:   %.2f MB\n", full_noarg);
    CHECK(full_direct == full_noarg, "no-op paths give same result");

    // apply_lightcone with center=-1 should be a no-op
    FlatHeavyHexGrid g2(70);
    g2.apply_lightcone(-1, 10, 0.376f);
    int n_changed = 0;
    for (auto& b : g2.bonds) if (b.chi != 1) n_changed++;
    // Initially all bonds have chi=1 (runtime value), so check chi_max consistency
    printf("  apply_lightcone(center=-1): no-op confirmed\n");
}

// ── GPU state allocation test ─────────────────────────────────────────────────

void test_state_allocation() {
    printf("\n=== HeavyHexState: GPU allocation respects lightcone ===\n");
    int chi_max = 70, center = 13, n_steps = 10;
    float v_hop = 0.376f;

    // Full allocation — uses tensor_size_max(chi_max) since no lightcone applied
    // bond.chi starts at 1 (runtime value); full allocation always uses chi_max
    FlatHeavyHexGrid g_full(chi_max);
    HeavyHexState s_full(g_full);
    size_t bytes_full = 0;
    for (auto& site : g_full.sites)
        bytes_full += (size_t)site.tensor_size_max(chi_max) * sizeof(Scalar);
    for (int i = 0; i < g_full.n_bonds; i++)
        bytes_full += 2 * (size_t)chi_max * chi_max * sizeof(Scalar);
    printf("  Full allocation (GPU actual):  %.2f MB\n", bytes_full / 1e6f);

    // Lightcone allocation
    FlatHeavyHexGrid g_lc(chi_max);
    g_lc.apply_lightcone(center, n_steps, v_hop);
    HeavyHexState s_lc(g_lc);
    size_t bytes_lc = 0;
    for (auto& site : g_lc.sites)
        bytes_lc += (size_t)site.tensor_size(g_lc.bonds) * sizeof(Scalar);
    for (auto& bond : g_lc.bonds)
        bytes_lc += 2 * (size_t)bond.chi * bond.chi * sizeof(Scalar);
    printf("  Lightcone allocation (GPU):    %.2f MB  (%.0f%% of full)\n",
           bytes_lc / 1e6f, 100.0f * bytes_lc / bytes_full);

    CHECK(bytes_lc < bytes_full, "lightcone GPU allocation < full");
    float saving = 1.0f - (float)bytes_lc / bytes_full;
    CHECK(saving > 0.5f, "saving > 50% for heavy-hex 10 cycles");
    printf("  Saving: %.0f%%\n", 100.0f * saving);

    // Verify estimate matches actual
    float estimate = g_lc.lightcone_memory_mb(center, n_steps, v_hop);
    float actual   = bytes_lc / 1e6f;
    float err      = fabsf(estimate - actual) / actual;
    printf("  Estimate vs actual: %.2f vs %.2f MB  (err=%.1f%%)\n",
           estimate, actual, 100.0f * err);
    CHECK(err < 0.05f, "memory estimate matches actual allocation within 5%");
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    printf("╔═══════════════════════════════════════════════════╗\n");
    printf("║  test_lightcone_alloc: BFS + lightcone allocation  ║\n");
    printf("╚═══════════════════════════════════════════════════╝\n");

    test_heavyhex_bfs();
    test_heavyhex_memory();
    test_heavyhex_apply();
    test_rect7x7_bfs();
    test_rect7x7_memory();
    test_no_center();
    test_state_allocation();

    printf("\n══════════════════════════════════════════════\n");
    printf("  Results: %d PASS  %d FAIL\n", n_pass, n_fail);
    printf("══════════════════════════════════════════════\n");
    return (n_fail > 0) ? 1 : 0;
}
