/*
 * tests/test_grid.cpp
 *
 * Test: Grid construction, site types, bond structure, edge coloring.
 *
 * Build:
 *   hipcc -O2 -std=c++17 -DPHYS_DIM=2 -I../include \
 *         -o test_grid test_grid.cpp -lrocblas -lm
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include <cstdio>
#include <cassert>
#include <set>

int main() {
    printf("=== test_grid ===\n");
    printf("D=%d\n", D);

    // ── Test 1: 4×4 grid structure ───────────────────────────────────
    {
        printf("\n--- Test 1: 4×4 grid structure ---\n");
        Grid g(4, 4, 50);

        // Sites
        assert(g.n_sites == 16);

        // Bonds: horizontal = 4*(4-1)=12, vertical = (4-1)*4=12 → 24 total
        assert(g.n_bonds == 24);
        printf("  Sites: %d  Bonds: %d  OK\n", g.n_sites, g.n_bonds);

        // Corner sites have degree 2
        assert(g.site(0,0).degree == 2);
        assert(g.site(0,3).degree == 2);
        assert(g.site(3,0).degree == 2);
        assert(g.site(3,3).degree == 2);
        printf("  Corner degrees: 2  OK\n");

        // Edge sites (non-corner) have degree 3
        assert(g.site(0,1).degree == 3);  // top edge
        assert(g.site(1,0).degree == 3);  // left edge
        assert(g.site(3,2).degree == 3);  // bottom edge
        assert(g.site(2,3).degree == 3);  // right edge
        printf("  Edge degrees: 3  OK\n");

        // Interior sites have degree 4
        assert(g.site(1,1).degree == 4);
        assert(g.site(2,2).degree == 4);
        assert(g.site(1,2).degree == 4);
        assert(g.site(2,1).degree == 4);
        printf("  Interior degrees: 4  OK\n");
    }

    // ── Test 2: Bond connectivity ─────────────────────────────────────
    {
        printf("\n--- Test 2: Bond connectivity ---\n");
        Grid g(4, 4, 50);

        // Site (1,1) is interior — should have all 4 bonds
        const Site& s = g.site(1,1);
        assert(s.bond_id[UP]    >= 0);
        assert(s.bond_id[DOWN]  >= 0);
        assert(s.bond_id[LEFT]  >= 0);
        assert(s.bond_id[RIGHT] >= 0);

        // Site (0,0) is top-left corner — only DOWN and RIGHT
        const Site& corner = g.site(0,0);
        assert(corner.bond_id[UP]    == -1);
        assert(corner.bond_id[LEFT]  == -1);
        assert(corner.bond_id[DOWN]  >= 0);
        assert(corner.bond_id[RIGHT] >= 0);
        printf("  Corner (0,0) bonds: DOWN=%d RIGHT=%d  OK\n",
               corner.bond_id[DOWN], corner.bond_id[RIGHT]);

        // Verify bond is shared: site(1,1).bond[LEFT] == site(1,0).bond[RIGHT]
        const Site& s10 = g.site(1,0);
        assert(g.site(1,1).bond_id[LEFT] == s10.bond_id[RIGHT]);
        printf("  Bond sharing (1,1)LEFT == (1,0)RIGHT: OK\n");

        // Verify neighbor links
        assert(g.site(1,1).neighbor_id[UP]    == g.site_id(0,1));
        assert(g.site(1,1).neighbor_id[DOWN]  == g.site_id(2,1));
        assert(g.site(1,1).neighbor_id[LEFT]  == g.site_id(1,0));
        assert(g.site(1,1).neighbor_id[RIGHT] == g.site_id(1,2));
        printf("  Neighbor links (1,1): OK\n");
    }

    // ── Test 3: Local bond indices (tensor shape positions) ───────────
    {
        printf("\n--- Test 3: Local bond indices ---\n");
        Grid g(4, 4, 50);

        // Interior site (1,1): bonds UP, DOWN, LEFT, RIGHT → local 0,1,2,3
        // Tensor shape: [chi_UP, chi_DOWN, chi_LEFT, chi_RIGHT, D]
        const Site& s = g.site(1,1);
        assert(s.bond_local[UP]    == 0);
        assert(s.bond_local[DOWN]  == 1);
        assert(s.bond_local[LEFT]  == 2);
        assert(s.bond_local[RIGHT] == 3);
        printf("  Interior (1,1) local: UP=0 DOWN=1 LEFT=2 RIGHT=3  OK\n");

        // Corner (0,0): bonds DOWN, RIGHT → local 0,1
        // Tensor shape: [chi_DOWN, chi_RIGHT, D]
        const Site& corner = g.site(0,0);
        assert(corner.bond_local[UP]    == -1);
        assert(corner.bond_local[LEFT]  == -1);
        assert(corner.bond_local[DOWN]  == 0);
        assert(corner.bond_local[RIGHT] == 1);
        printf("  Corner (0,0) local: DOWN=0 RIGHT=1  OK\n");

        // Top edge (0,1): bonds DOWN, LEFT, RIGHT → local 0,1,2
        const Site& top = g.site(0,1);
        assert(top.bond_local[DOWN]  == 0);
        assert(top.bond_local[LEFT]  == 1);
        assert(top.bond_local[RIGHT] == 2);
        printf("  Top edge (0,1) local: DOWN=0 LEFT=1 RIGHT=2  OK\n");
    }

    // ── Test 4: Tensor shapes ─────────────────────────────────────────
    {
        printf("\n--- Test 4: Tensor shapes at chi_max=50 ---\n");
        Grid g(4, 4, 50);

        // Corner (0,0): shape = [chi_DOWN, chi_RIGHT, D] = [50,50,2]
        auto sh_corner = g.site(0,0).tensor_shape(50);
        assert(sh_corner.size() == 3);
        assert(sh_corner[0] == 50);
        assert(sh_corner[1] == 50);
        assert(sh_corner[2] == D);
        printf("  Corner shape: [%d,%d,%d]  OK\n",
               sh_corner[0], sh_corner[1], sh_corner[2]);

        // Edge (0,1): shape = [chi_DOWN, chi_LEFT, chi_RIGHT, D] = [50,50,50,2]
        auto sh_edge = g.site(0,1).tensor_shape(50);
        assert(sh_edge.size() == 4);
        assert(sh_edge[3] == D);
        printf("  Edge shape: [%d,%d,%d,%d]  OK\n",
               sh_edge[0], sh_edge[1], sh_edge[2], sh_edge[3]);

        // Interior (1,1): shape = [chi_UP,chi_DOWN,chi_LEFT,chi_RIGHT,D] = [50,50,50,50,2]
        auto sh_int = g.site(1,1).tensor_shape(50);
        assert(sh_int.size() == 5);
        assert(sh_int[4] == D);
        printf("  Interior shape: [%d,%d,%d,%d,%d]  OK\n",
               sh_int[0], sh_int[1], sh_int[2], sh_int[3], sh_int[4]);

        // Memory summary
        printf("  Total tensor memory: %.1f MB\n",
               g.total_tensor_bytes()/1e6);
        printf("  Total message memory: %.1f MB\n",
               g.total_message_bytes()/1e6);
    }

    // ── Test 5: Edge coloring ─────────────────────────────────────────
    {
        printf("\n--- Test 5: Edge coloring ---\n");
        Grid g(4, 4, 50);

        // Total gates across all colors = total bonds
        int total = 0;
        for (int c=0; c<4; c++) total += g.colors[c].size();
        assert(total == g.n_bonds);
        printf("  Total gates across colors: %d == %d bonds  OK\n",
               total, g.n_bonds);

        // Each color class: gates must not share any site
        for (int c=0; c<4; c++) {
            std::set<int> used_sites;
            for (auto& gate : g.colors[c]) {
                assert(used_sites.find(gate.site_a) == used_sites.end());
                assert(used_sites.find(gate.site_b) == used_sites.end());
                used_sites.insert(gate.site_a);
                used_sites.insert(gate.site_b);
            }
            printf("  Color %d (%s): %zu gates, no site conflicts  OK\n",
                   c, Grid::color_name(c), g.colors[c].size());
        }

        // For 4×4: horizontal bonds = 12, vertical bonds = 12
        // H color 0 (col_a even: c=0,2): 4 rows × 2 = 8 bonds
        // H color 1 (col_a odd:  c=1):   4 rows × 1 = 4 bonds
        // V color 2 (row_a even: r=0,2): 4 cols × 2 = 8 bonds
        // V color 3 (row_a odd:  r=1):   4 cols × 1 = 4 bonds
        printf("  Color sizes: %zu %zu %zu %zu\n",
               g.colors[0].size(), g.colors[1].size(),
               g.colors[2].size(), g.colors[3].size());
        assert(g.colors[0].size() + g.colors[1].size() == 12);  // all H bonds
        assert(g.colors[2].size() + g.colors[3].size() == 12);  // all V bonds
        printf("  H bonds: %zu+%zu=12  V bonds: %zu+%zu=12  OK\n",
               g.colors[0].size(), g.colors[1].size(),
               g.colors[2].size(), g.colors[3].size());
    }

    // ── Test 6: 2×2 grid (sanity check) ──────────────────────────────
    {
        printf("\n--- Test 6: 2×2 grid sanity check ---\n");
        Grid g(2, 2, 4);

        assert(g.n_sites == 4);
        assert(g.n_bonds == 4);  // 2 horizontal + 2 vertical

        // All sites are corners (degree 2)
        for (auto& s : g.sites) assert(s.degree == 2);
        printf("  2×2: 4 sites, 4 bonds, all corners  OK\n");

        // 2×2: all horizontal bonds have col_a=0 (even) → color 0
        //      all vertical bonds have row_a=0 (even) → color 2
        // colors 1 and 3 are empty
        assert(g.colors[0].size() == 2);  // 2 horizontal bonds
        assert(g.colors[1].size() == 0);
        assert(g.colors[2].size() == 2);  // 2 vertical bonds
        assert(g.colors[3].size() == 0);
        printf("  2×2 colors: %zu %zu %zu %zu  OK\n",
               g.colors[0].size(), g.colors[1].size(),
               g.colors[2].size(), g.colors[3].size());

        g.print();
    }

    // ── Test 7: Full 4×4 print ────────────────────────────────────────
    {
        printf("\n--- Test 7: Full 4×4 grid ---\n");
        Grid g(4, 4, 50);
        g.print_tensor_shapes();
    }

    printf("\n=== ALL GRID TESTS PASSED ===\n");
    return 0;
}
