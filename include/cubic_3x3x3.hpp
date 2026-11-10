/* cubic_3x3x3.hpp — 3×3×3 cubic lattice for 3D Heisenberg tomography
 *
 * 64 sites, 144 bonds (48 per axis), 6 color classes:
 *   color 0: X-bonds, even x  (x=0,2)
 *   color 1: X-bonds, odd  x  (x=1,3)
 *   color 2: Y-bonds, even y  (y=0,2)
 *   color 3: Y-bonds, odd  y  (y=1,3)
 *   color 4: Z-bonds, even z  (z=0,2)
 *   color 5: Z-bonds, odd  z  (z=1,3)
 *
 * Site indexing (0-based): s = z*16 + y*4 + x
 * MAX_DEGREE must be 6 for 3D (interior sites have 6 neighbors)
 */

#pragma once
#include "grid.hpp"

static constexpr int CUBIC_3x3x3_NX      = 3;
static constexpr int CUBIC_3x3x3_NY      = 3;
static constexpr int CUBIC_3x3x3_NZ      = 3;
static constexpr int CUBIC_3x3x3_N_SITES  = 27;
static constexpr int CUBIC_3x3x3_N_BONDS  = 54;
static constexpr int CUBIC_3x3x3_N_COLORS = 6;

struct FlatCubic3x3x3Grid : Grid {

    static int site_id(int x, int y, int z) {
        return z * CUBIC_3x3x3_NX * CUBIC_3x3x3_NY + y * CUBIC_3x3x3_NX + x;
    }

    explicit FlatCubic3x3x3Grid(int chi_max_)
        : Grid(chi_max_, CUBIC_3x3x3_N_COLORS)
    {
        n_sites = CUBIC_3x3x3_N_SITES;
        n_bonds = CUBIC_3x3x3_N_BONDS;

        sites.resize(n_sites);
        bonds.resize(n_bonds);

        for (int i = 0; i < n_sites; i++) {
            sites[i].id     = i;
            sites[i].row    = 0;   // not used for 3D
            sites[i].col    = i;
            sites[i].degree = 0;
            for (int d = 0; d < MAX_DEGREE; d++) {
                sites[i].bond_id[d]     = -1;
                sites[i].neighbor_id[d] = -1;
                sites[i].bond_local[d]  = -1;
            }
        }

        int bid = 0;

        // X-bonds: color 0 (even x) / color 1 (odd x)
        for (int z = 0; z < CUBIC_3x3x3_NZ; z++)
        for (int y = 0; y < CUBIC_3x3x3_NY; y++)
        for (int x = 0; x < CUBIC_3x3x3_NX - 1; x++) {
            int color = (x % 2 == 0) ? 0 : 1;
            _add_bond(bid++, site_id(x, y, z), site_id(x+1, y, z), color);
        }

        // Y-bonds: color 2 (even y) / color 3 (odd y)
        for (int z = 0; z < CUBIC_3x3x3_NZ; z++)
        for (int y = 0; y < CUBIC_3x3x3_NY - 1; y++)
        for (int x = 0; x < CUBIC_3x3x3_NX; x++) {
            int color = (y % 2 == 0) ? 2 : 3;
            _add_bond(bid++, site_id(x, y, z), site_id(x, y+1, z), color);
        }

        // Z-bonds: color 4 (even z) / color 5 (odd z)
        for (int z = 0; z < CUBIC_3x3x3_NZ - 1; z++)
        for (int y = 0; y < CUBIC_3x3x3_NY; y++)
        for (int x = 0; x < CUBIC_3x3x3_NX; x++) {
            int color = (z % 2 == 0) ? 4 : 5;
            _add_bond(bid++, site_id(x, y, z), site_id(x, y, z+1), color);
        }

        if (bid != CUBIC_3x3x3_N_BONDS) {
            fprintf(stderr, "Bond count mismatch: %d != %d\n", bid, CUBIC_3x3x3_N_BONDS);
            exit(1);
        }

        printf("[FlatCubic3x3x3Grid] %d sites, %d bonds, %d colors, chi_max=%d\n",
               n_sites, n_bonds, (int)colors.size(), chi_max_);
    }

private:
    void _add_bond(int bid, int u, int v, int color) {
        Bond b;
        b.id = bid; b.chi = 1; b.chi_alloc = chi_max;
        b.site_a = u; b.site_b = v;

        int k_u = _next_slot(u);
        sites[u].bond_id[k_u]     = bid;
        sites[u].bond_local[k_u]  = k_u;
        sites[u].neighbor_id[k_u] = v;
        sites[u].degree++;
        b.dir_a = k_u;

        int k_v = _next_slot(v);
        sites[v].bond_id[k_v]     = bid;
        sites[v].bond_local[k_v]  = k_v;
        sites[v].neighbor_id[k_v] = u;
        sites[v].degree++;
        b.dir_b = k_v;

        bonds[bid] = b;

        Gate g;
        g.bond_id    = bid;
        g.site_a     = u;
        g.site_b     = v;
        g.dir_a_to_b = k_u;
        g.dir_b_to_a = k_v;
        colors[color].push_back(g);
    }

    int _next_slot(int sid) {
        for (int d = 0; d < MAX_DEGREE; d++)
            if (sites[sid].bond_id[d] < 0) return d;
        fprintf(stderr, "Site %d: MAX_DEGREE=%d exceeded\n", sid, MAX_DEGREE);
        exit(1);
        return 0;
    }
};
