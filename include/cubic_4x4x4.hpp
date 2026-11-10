/* cubic_4x4x4.hpp — 4×4×4 cubic lattice for 3D Heisenberg tomography
 *
 * 64 sites, 144 bonds (48 per axis), 6 color classes:
 *   color 0: X-bonds, even x-start  (x=0,2)
 *   color 1: X-bonds, odd  x-start  (x=1,3)
 *   color 2: Y-bonds, even y-start  (y=0,2)
 *   color 3: Y-bonds, odd  y-start  (y=1,3)
 *   color 4: Z-bonds, even z-start  (z=0,2)
 *   color 5: Z-bonds, odd  z-start  (z=1,3)
 *
 * Site indexing (1-based): s = z*16 + y*4 + x + 1
 */

#pragma once
#include "grid.hpp"

static constexpr int CUBIC_4x4x4_NX     = 4;
static constexpr int CUBIC_4x4x4_NY     = 4;
static constexpr int CUBIC_4x4x4_NZ     = 4;
static constexpr int CUBIC_4x4x4_N_SITES  = 64;
static constexpr int CUBIC_4x4x4_N_BONDS  = 144;
static constexpr int CUBIC_4x4x4_N_COLORS = 6;

struct FlatCubic4x4x4Grid : Grid {

    static int site_id(int x, int y, int z) {
        return z * CUBIC_4x4x4_NX * CUBIC_4x4x4_NY
             + y * CUBIC_4x4x4_NX
             + x;  // 0-based internal
    }

    explicit FlatCubic4x4x4Grid(int chi_max_)
        : Grid(chi_max_, CUBIC_4x4x4_N_COLORS)
    {
        n_sites = CUBIC_4x4x4_N_SITES;
        n_bonds = CUBIC_4x4x4_N_BONDS;

        sites.resize(n_sites);
        bonds.resize(n_bonds);

        for (int i = 0; i < n_sites; i++) {
            sites[i].id = i;
            sites[i].degree = 0;
        }

        int bid = 0;

        // X-bonds: color 0 (even x) / color 1 (odd x)
        for (int z = 0; z < CUBIC_4x4x4_NZ; z++)
        for (int y = 0; y < CUBIC_4x4x4_NY; y++)
        for (int x = 0; x < CUBIC_4x4x4_NX - 1; x++) {
            int u = site_id(x,   y, z);
            int v = site_id(x+1, y, z);
            int color = (x % 2 == 0) ? 0 : 1;
            _add_bond(bid++, u, v, color);
        }

        // Y-bonds: color 2 (even y) / color 3 (odd y)
        for (int z = 0; z < CUBIC_4x4x4_NZ; z++)
        for (int y = 0; y < CUBIC_4x4x4_NY - 1; y++)
        for (int x = 0; x < CUBIC_4x4x4_NX; x++) {
            int u = site_id(x, y,   z);
            int v = site_id(x, y+1, z);
            int color = (y % 2 == 0) ? 2 : 3;
            _add_bond(bid++, u, v, color);
        }

        // Z-bonds: color 4 (even z) / color 5 (odd z)
        for (int z = 0; z < CUBIC_4x4x4_NZ - 1; z++)
        for (int y = 0; y < CUBIC_4x4x4_NY; y++)
        for (int x = 0; x < CUBIC_4x4x4_NX; x++) {
            int u = site_id(x, y, z);
            int v = site_id(x, y, z+1);
            int color = (z % 2 == 0) ? 4 : 5;
            _add_bond(bid++, u, v, color);
        }

        if (bid != CUBIC_4x4x4_N_BONDS) {
            fprintf(stderr, "Bond count mismatch: %d != %d\n", bid, CUBIC_4x4x4_N_BONDS);
            exit(1);
        }

        printf("[FlatCubic4x4x4Grid] %d sites, %d bonds, %d colors, chi_max=%d\n",
               n_sites, n_bonds, (int)colors.size(), chi_max_);
    }

    void _add_bond(int bid, int u, int v, int color) {
        Bond b;
        b.id = bid;
        b.sites[0] = u;
        b.sites[1] = v;
        b.color = color;
        b.coupling = 1.0f;
        bonds[bid] = b;
        colors[color].push_back(b);
        sites[u].neighbors.push_back({v, bid});
        sites[v].neighbors.push_back({u, bid});
        sites[u].degree++;
        sites[v].degree++;
    }
};
