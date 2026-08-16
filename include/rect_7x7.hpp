#pragma once
/*
 * rect_7x7.hpp — FlatRect7x7Grid: 7×7 rectangular lattice
 *
 * 49 sites, 84 bonds, 2 color classes:
 *   color 0: horizontal bonds (|u-v|=1, 42 bonds)
 *   color 1: vertical bonds   (|u-v|=7, 42 bonds)
 *
 * Site numbering: row-major, 0-indexed
 *   site(r,c) = r*7 + c   for r,c in 0..6
 *
 * Interior sites: degree 4
 * Edge sites:     degree 2 or 3
 *
 * Build: make PHYS_DIM=4 DFLAGS=-DGRAPH_rect_7x7 main_generic_lightcone
 */

#include "grid.hpp"
#include "types.hpp"
#include <cstdio>

static constexpr int RECT_7x7_N_SITES  = 49;
static constexpr int RECT_7x7_N_BONDS  = 84;
static constexpr int RECT_7x7_N_COLORS = 4;
static constexpr int RECT_7x7_NX       = 7;
static constexpr int RECT_7x7_NY       = 7;

struct FlatRect7x7Grid : public Grid {

    explicit FlatRect7x7Grid(int chi_max_) : Grid(chi_max_, RECT_7x7_N_COLORS) {
        n_sites = RECT_7x7_N_SITES;
        n_bonds = RECT_7x7_N_BONDS;

        sites.resize(n_sites);
        bonds.resize(n_bonds);

        for (int i = 0; i < n_sites; i++) {
            sites[i].id     = i;
            sites[i].degree = 0;
            sites[i].row    = i / RECT_7x7_NX;
            sites[i].col    = i % RECT_7x7_NX;
            for (int d = 0; d < MAX_DEGREE; d++) {
                sites[i].bond_id[d]     = -1;
                sites[i].bond_local[d]  = d;
                sites[i].neighbor_id[d] = -1;
            }
        }

        // 4-color scheme (same as 2D Ising in grid.hpp):
        //   color 0: horizontal, even-column start  (c=0,2,4,6)
        //   color 1: horizontal, odd-column start   (c=1,3,5)
        //   color 2: vertical,   even-row start     (r=0,2,4,6)
        //   color 3: vertical,   odd-row start      (r=1,3,5)
        int bid = 0;

        // Horizontal bonds
        for (int r = 0; r < RECT_7x7_NY; r++) {
            for (int c = 0; c < RECT_7x7_NX-1; c++) {
                int u = r*RECT_7x7_NX + c;
                int v = r*RECT_7x7_NX + c + 1;
                int color = (c % 2 == 0) ? 0 : 1;
                _add_bond(bid, u, v, color);
                bid++;
            }
        }

        // Vertical bonds
        for (int r = 0; r < RECT_7x7_NY-1; r++) {
            for (int c = 0; c < RECT_7x7_NX; c++) {
                int u = r*RECT_7x7_NX + c;
                int v = (r+1)*RECT_7x7_NX + c;
                int color = (r % 2 == 0) ? 2 : 3;
                _add_bond(bid, u, v, color);
                bid++;
            }
        }

        if (bid != RECT_7x7_N_BONDS) {
            fprintf(stderr, "FlatRect7x7Grid: expected %d bonds, got %d\n",
                    RECT_7x7_N_BONDS, bid);
        }

        printf("[FlatRect7x7Grid] %d sites, %d bonds, %d colors, chi_max=%d\n",
               n_sites, n_bonds, (int)colors.size(), chi_max_);
    }

private:
    void _add_bond(int bid, int u, int v, int color) {
        Bond b;
        b.id = bid; b.chi = 1; b.chi_alloc = chi_max;
        b.site_a = u; b.site_b = v;

        // Assign to site u
        int k_u = _next_slot(u);
        sites[u].bond_id[k_u]     = bid;
        sites[u].bond_local[k_u]  = k_u;
        sites[u].neighbor_id[k_u] = v;
        sites[u].degree++;
        b.dir_a = k_u;

        // Assign to site v
        int k_v = _next_slot(v);
        sites[v].bond_id[k_v]     = bid;
        sites[v].bond_local[k_v]  = k_v;
        sites[v].neighbor_id[k_v] = u;
        sites[v].degree++;
        b.dir_b = k_v;

        bonds[bid] = b;

        // Add gate to color class — dir_a_to_b/dir_b_to_a = local slot indices
        Gate g;
        g.bond_id    = bid;
        g.site_a     = u;
        g.site_b     = v;
        g.dir_a_to_b = k_u;  // local slot of this bond in site u
        g.dir_b_to_a = k_v;  // local slot of this bond in site v
        colors[color].push_back(g);
    }

    int _next_slot(int sid) {
        for (int d = 0; d < MAX_DEGREE; d++)
            if (sites[sid].bond_id[d] < 0) return d;
        fprintf(stderr, "Site %d: MAX_DEGREE=%d exceeded\n", sid, MAX_DEGREE);
        return 0;
    }
};
