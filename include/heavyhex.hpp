#pragma once
/*
 * heavyhex.hpp — Heavy-hex lattice for CppSim
 *
 * ═══════════════════════════════════════════════════════════════════
 * HEAVY-HEX TOPOLOGY
 * ═══════════════════════════════════════════════════════════════════
 *
 * The heavy-hex lattice (used in IBM quantum hardware) is a degree-3
 * graph arranged in a hexagonal pattern. For an nx×ny unit cell:
 *
 *   - "junction" sites (degree 3): (nx+1)*(ny+1) = 16 for 3×3
 *   - "arm" sites (degree 2): connect junctions along hex edges
 *   - Total sites: 68 for 3×3  (52 arm + 16 junction)
 *   - Total bonds: 76 for 3×3
 *
 * Each hexagon has 2 arm sites on horizontal edges and 1 on
 * vertical edges. The backbone of junction sites forms a rectangular
 * (nx+1)×(ny+1) grid with the hexagonal arms between them.
 *
 * CONSTRUCTION STRATEGY:
 *
 *   We label junction sites first (0..(nx+1)*(ny+1)-1), then arm
 *   sites. Bonds come in 3 natural color classes matching the
 *   heavy-hex gate layers:
 *     Color 0: horizontal arms — top edge of each hex row
 *     Color 1: horizontal arms — bottom edge of each hex row
 *     Color 2: vertical arms  — between rows
 *
 *   This matches the Julia JSON layer structure exactly.
 *
 * COUPLING STRENGTHS:
 *
 *   HeavyHexGrid supports two modes:
 *     uniform:     all bonds get coupling J (default 1.0)
 *     from_data:   use HeavyHexData (hardcoded from JSON) for 3×3
 *
 * BUILD: make PHYS_DIM=4 (Pauli basis, D=4 mandatory)
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include <vector>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <algorithm>

/* =========================================================
 * HeavyHexGrid — heavy-hex lattice with degree-2/3 sites
 *
 * Compatible with the existing Site/Bond/Gate/Grid interface.
 * Reuses gate.hpp, measures.hpp, workspace.hpp unchanged.
 * ========================================================= */
struct HeavyHexGrid {
    int nx, ny;          // unit cell dimensions (3×3 for heavy_hex_3x3)
    int n_sites;         // total sites
    int n_bonds;         // total bonds
    int chi_max;

    std::vector<Site>  sites;
    std::vector<Bond>  bonds;
    std::vector<float> couplings;  // one coupling per bond (indexed by bond_id)

    // 3 color classes matching Julia's gate layers 0,1,2
    std::vector<std::vector<Gate>> colors;

    HeavyHexGrid(int nx_, int ny_, int chi_max_, float J_uniform = 1.0f)
        : nx(nx_), ny(ny_), chi_max(chi_max_), colors(3)
    {
        _build(J_uniform);
    }

    // Memory estimates (same interface as Grid)
    size_t total_tensor_bytes() const {
        size_t total = 0;
        for (auto& s : sites)
            total += (size_t)s.tensor_size_max(chi_max) * sizeof(Scalar);
        return total;
    }
    size_t total_message_bytes() const {
        return (size_t)n_bonds * 2 * chi_max * chi_max * sizeof(Scalar);
    }

    void print() const {
        printf("HeavyHexGrid %dx%d: %d sites, %d bonds, chi_max=%d\n",
               nx, ny, n_sites, n_bonds, chi_max);
        printf("  Degree-2 (arm):      %d\n",
               (int)std::count_if(sites.begin(), sites.end(),
                   [](const Site& s){ return s.degree == 2; }));
        printf("  Degree-3 (junction): %d\n",
               (int)std::count_if(sites.begin(), sites.end(),
                   [](const Site& s){ return s.degree == 3; }));
        for (int c = 0; c < 3; c++)
            printf("  Color %d: %zu gates\n", c, colors[c].size());
    }

private:
    /* ── Build the full lattice ──────────────────────────────────── */
    void _build(float J_uniform) {
        // ── Step 1: assign site ids ───────────────────────────────
        // Junction sites: (nx+1)*(ny+1) arranged in a grid
        //   junction_id(r,c) = r*(nx+1) + c   r∈[0,ny], c∈[0,nx]
        // Arm sites come after junctions.
        //
        // Arm layout:
        //   Horizontal arms: 2 per horizontal junction pair per row
        //     Between junction(r,c) and junction(r,c+1): 2 arms
        //     arm_h_top(r,c): top arm (color 0) — part of upper hex
        //     arm_h_bot(r,c): bot arm (color 1) — part of lower hex
        //     ...except boundary rows have only one arm
        //   Vertical arms: 1 per vertical junction pair per col
        //     Between junction(r,c) and junction(r+1,c): 1 arm (color 2)

        int n_junctions = (nx+1)*(ny+1);

        // Count arm sites:
        //   Horizontal: nx pairs per row × (ny+1) rows × 2 arms
        //               but top row (r=0) and bottom row (r=ny) only 1 arm each
        //   Actually simpler: each horizontal bond (junction pair) gets:
        //     2 arms if it has hexagons both above and below (1 ≤ r ≤ ny-1)
        //     1 arm  if it's on the top (r=0) or bottom (r=ny) boundary
        //   Vertical: nx+1 columns × ny pairs = ny*(nx+1) arm sites (color 2)

        // We'll build all bonds explicitly and count as we go.
        // Sites are pre-allocated with enough slots.

        sites.clear();
        bonds.clear();
        couplings.clear();

        // Pre-allocate junction sites
        sites.resize(n_junctions);
        for (int r = 0; r <= ny; r++) {
            for (int c = 0; c <= nx; c++) {
                int id = r*(nx+1)+c;
                Site& s = sites[id];
                s.id  = id;
                s.row = r; s.col = c;
                s.degree = 0;
                s.bond_id.fill(-1);
                s.neighbor_id.fill(-1);
                s.bond_local.fill(-1);
            }
        }

        int next_site_id = n_junctions;
        int bond_id      = 0;

        auto add_arm_site = [&](int r, int c) -> int {
            Site s;
            s.id  = next_site_id++;
            s.row = r; s.col = c;
            s.degree = 2;
            s.bond_id.fill(-1);
            s.neighbor_id.fill(-1);
            s.bond_local.fill(-1);
            sites.push_back(s);
            return s.id;
        };

        auto add_bond = [&](int sa, int sb, int dir_a, int dir_b,
                             float coupling, int color) {
            Bond b;
            b.id = bond_id;
            b.site_a = std::min(sa, sb);
            b.site_b = std::max(sa, sb);
            // dir_a/dir_b are stored per original orientation
            b.dir_a = (sa < sb) ? dir_a : dir_b;
            b.dir_b = (sa < sb) ? dir_b : dir_a;
            b.chi = 1;
            bonds.push_back(b);
            couplings.push_back(coupling);

            // Wire sites
            sites[sa].bond_id[dir_a]      = bond_id;
            sites[sa].neighbor_id[dir_a]  = sb;
            sites[sa].degree++;
            sites[sb].bond_id[dir_b]      = bond_id;
            sites[sb].neighbor_id[dir_b]  = sa;
            sites[sb].degree++;

            // Add gate to color class
            Gate g;
            g.bond_id    = bond_id;
            g.site_a     = b.site_a;
            g.site_b     = b.site_b;
            g.dir_a_to_b = b.dir_a;
            g.dir_b_to_a = b.dir_b;
            colors[color].push_back(g);

            bond_id++;
        };

        // ── Step 2: build bonds ───────────────────────────────────
        //
        // Heavy-hex structure for 3×3 (each hex = 6 edges):
        //
        //  J--a--J--a--J--a--J     row 0 junctions (r=0)
        //  |     |     |     |
        //  b     b     b     b     vertical arms (color 2)
        //  |     |     |     |
        //  J--a--J--a--J--a--J     row 1 junctions (r=1)
        //  |     |     |     |
        //  b     b     b     b
        //  |     |     |     |
        //  J--a--J--a--J--a--J     row 2 junctions (r=2)
        //  ...
        //
        // Each "a" (horizontal arm) is 2 sites between junctions:
        //   J - arm_top - arm_bot - J   (but only one visible in diagram above)
        //
        // Actually in heavy-hex each horizontal junction pair has
        // exactly ONE arm site between them (degree-2 node on the edge).
        // The "2 colors" come from which hex face the arm belongs to:
        //   arms on even hex rows → color 0
        //   arms on odd hex rows  → color 1
        //
        // More precisely, for junction pair (r,c)-(r,c+1):
        //   if r is even (0,2,4,...): color 0 (top of hex)
        //   if r is odd  (1,3,5,...): color 1 (bottom of hex)
        //
        // Vertical arms between junction(r,c) and junction(r+1,c):
        //   always color 2

        // Horizontal arms (one arm site per horizontal junction pair)
        for (int r = 0; r <= ny; r++) {
            for (int c = 0; c < nx; c++) {
                int ja = r*(nx+1) + c;       // junction (r,c)
                int jb = r*(nx+1) + c + 1;   // junction (r,c+1)
                int arm = add_arm_site(r, c); // arm between them

                // color: even row → 0, odd row → 1
                int color = r % 2;

                // ja → arm: RIGHT/LEFT
                add_bond(ja, arm, RIGHT, LEFT, J_uniform, color);
                // arm → jb: RIGHT/LEFT
                add_bond(arm, jb, RIGHT, LEFT, J_uniform, color);
            }
        }

        // Vertical arms (one arm site per vertical junction pair)
        for (int r = 0; r < ny; r++) {
            for (int c = 0; c <= nx; c++) {
                int ja  = r*(nx+1) + c;       // junction (r,c)
                int jb  = (r+1)*(nx+1) + c;   // junction (r+1,c)
                int arm = add_arm_site(r, c);  // arm between them

                // ja → arm: DOWN/UP
                add_bond(ja, arm, DOWN, UP, J_uniform, 2);
                // arm → jb: DOWN/UP
                add_bond(arm, jb, DOWN, UP, J_uniform, 2);
            }
        }

        n_sites = (int)sites.size();
        n_bonds = (int)bonds.size();

        // ── Step 3: assign bond_local indices ─────────────────────
        // bond_local[d] = position of direction d in tensor shape
        for (auto& s : sites) {
            int local = 0;
            for (int d = 0; d < MAX_DEGREE; d++) {
                if (s.bond_id[d] >= 0)
                    s.bond_local[d] = local++;
            }
        }
    }
};

/* =========================================================
 * HeavyHexData — hardcoded 3×3 topology from heavy_hex_3x3.json
 *
 * Use this to override the uniform coupling with the actual
 * hardware-derived coupling strengths for validation against Julia.
 *
 * Sites are 0-based (Julia 1-based → subtract 1).
 * ========================================================= */
struct HeavyHexEdge {
    int u, v;
    float coupling;
};

// 76 edges from heavy_hex_3x3.json (0-based vertex ids)
static const HeavyHexEdge HEAVY_HEX_3x3_EDGES[] = {
    {0, 30, 1.0788355602f}, {0, 31, 0.9120141404f},
    {1, 30, 0.9126206518f}, {1, 32, 0.9266745036f},
    {2, 32, 0.9297804801f}, {2, 33, 0.9927418132f}, {2, 34, 0.9087076614f},
    {3, 33, 1.0631620831f}, {3, 35, 1.1438683276f},
    {4, 35, 0.9645830157f}, {4, 36, 1.0796126919f}, {4, 37, 0.8824628867f},
    {5, 36, 0.9406604961f}, {5, 38, 1.0429606365f},
    {6, 38, 1.0952328463f}, {6, 39, 1.0129139212f},
    {7, 40, 0.9705226106f}, {7, 41, 0.9625731565f},
    {8, 31, 1.1169511740f}, {8, 42, 1.0256848496f},
    {9, 42, 1.0200925089f}, {9, 43, 1.1133899549f}, {9, 44, 1.0289788394f},
    {10, 34, 0.8469160937f}, {10, 43, 1.0218401311f}, {10, 45, 0.9390884104f},
    {11, 45, 1.0648947361f}, {11, 46, 1.1335869464f}, {11, 47, 0.9334079458f},
    {12, 37, 0.8994402010f}, {12, 46, 1.0808491754f}, {12, 48, 0.9619254224f},
    {13, 48, 0.9793386886f}, {13, 49, 0.9689256127f}, {13, 50, 0.9959526600f},
    {14, 39, 1.0104770792f}, {14, 49, 0.9034509513f}, {14, 51, 1.0965660750f},
    {15, 51, 1.1311017356f}, {15, 52, 1.0404175401f},
    {16, 53, 1.1197959670f}, {16, 54, 0.8770911287f},
    {17, 44, 1.1460076415f}, {17, 53, 0.9962598969f}, {17, 55, 0.9284062687f},
    {18, 55, 0.9040971061f}, {18, 56, 1.0954845286f}, {18, 57, 1.0378568404f},
    {19, 47, 0.9742639147f}, {19, 56, 0.8829446686f}, {19, 58, 1.0930732350f},
    {20, 58, 1.0526993989f}, {20, 59, 0.9906889689f}, {20, 60, 1.0155249664f},
    {21, 50, 0.9719658946f}, {21, 59, 0.9157202345f}, {21, 61, 0.9797093499f},
    {22, 40, 0.9713468554f}, {22, 61, 0.9768239321f}, {22, 62, 1.1055754068f},
    {23, 52, 0.8852311316f}, {23, 62, 0.8641222403f},
    {24, 54, 0.8897853237f}, {24, 63, 0.9476480274f},
    {25, 63, 0.9882130696f}, {25, 64, 0.9736584856f},
    {26, 57, 1.1382598722f}, {26, 64, 0.9491636467f}, {26, 65, 0.8918565241f},
    {27, 65, 1.1841894256f}, {27, 66, 0.9537136817f},
    {28, 60, 0.9834704354f}, {28, 66, 1.0720224181f}, {28, 67, 0.9817405804f},
    {29, 41, 0.9438291785f}, {29, 67, 0.8537121307f},
};
static constexpr int HEAVY_HEX_3x3_N_EDGES = 76;

// Color classes from JSON layers (0-based vertex ids)
// Layer 0 (color 0): 25 gates
static const int HEAVY_HEX_3x3_COLOR0[][2] = {
    {4,35},{12,37},{25,63},{14,51},{21,50},{26,65},{11,45},{23,52},
    {24,54},{20,58},{17,55},{9,42},{13,49},{18,56},{10,43},{29,41},
    {0,31},{28,67},{5,36},{1,30},{22,62},{19,47},{7,40},{2,34},{3,33}
};
static constexpr int HEAVY_HEX_3x3_COLOR0_N = 25;

// Layer 1 (color 1): 25 gates
static const int HEAVY_HEX_3x3_COLOR1[][2] = {
    {29,67},{27,65},{13,48},{14,39},{23,62},{3,35},{16,53},{19,58},
    {17,44},{6,38},{10,34},{28,66},{11,47},{2,33},{4,36},{8,42},
    {22,61},{26,64},{15,51},{9,43},{21,59},{12,46},{18,57},{1,32},{20,60}
};
static constexpr int HEAVY_HEX_3x3_COLOR1_N = 25;

// Layer 2 (color 2): 26 gates
static const int HEAVY_HEX_3x3_COLOR2[][2] = {
    {5,38},{24,63},{0,30},{27,66},{15,52},{21,61},{2,32},{9,44},
    {13,50},{22,40},{10,45},{7,41},{8,31},{19,56},{12,48},{18,55},
    {6,39},{14,49},{20,59},{25,64},{26,57},{28,60},{16,54},{11,46},
    {17,53},{4,37}
};
static constexpr int HEAVY_HEX_3x3_COLOR2_N = 26;

/* =========================================================
 * build_heavyhex_3x3 — builds exact heavy_hex_3x3 grid
 *
 * Uses the hardcoded edge list and color classes from JSON.
 * Sites and bonds are indexed to match Julia's vertex ordering.
 * ========================================================= */
struct FlatHeavyHexGrid : public Grid {
    std::vector<float> couplings;

    explicit FlatHeavyHexGrid(int chi_max_) : Grid(chi_max_, 3) {
        n_sites = 68;
        n_bonds = 76;
        _build();
    }

    void print() const {
        printf("FlatHeavyHexGrid: %d sites, %d bonds, chi_max=%d\n",
               n_sites, n_bonds, chi_max);
        int d2 = 0, d3 = 0;
        for (auto& s : sites) { if (s.degree==2) d2++; else if(s.degree==3) d3++; }
        printf("  Degree-2 (arm):      %d\n", d2);
        printf("  Degree-3 (junction): %d\n", d3);
        for (int c = 0; c < 3; c++)
            printf("  Color %d: %zu gates\n", c, colors[c].size());
    }

private:
    void _build() {
        // Init sites
        sites.resize(n_sites);
        for (int i = 0; i < n_sites; i++) {
            sites[i].id = i;
            sites[i].row = 0; sites[i].col = i;  // flat layout
            sites[i].degree = 0;
            sites[i].bond_id.fill(-1);
            sites[i].neighbor_id.fill(-1);
            sites[i].bond_local.fill(-1);
        }

        // Build bonds from edge list
        // We assign directions based on bond index order (not geometry):
        // Each bond is a pair (u,v) with u<v.
        // We use the 4 direction slots to pack up to 3 bonds per site.
        // Direction assignment: first bond → UP slot, second → DOWN, third → LEFT
        // (RIGHT is never used since max degree is 3)

        bonds.reserve(n_bonds);
        couplings.reserve(n_bonds);

        for (int i = 0; i < HEAVY_HEX_3x3_N_EDGES; i++) {
            int u = HEAVY_HEX_3x3_EDGES[i].u;
            int v = HEAVY_HEX_3x3_EDGES[i].v;
            float c = HEAVY_HEX_3x3_EDGES[i].coupling;

            Bond b;
            b.id = i;
            b.site_a = u; b.site_b = v;
            b.chi = 1;
            couplings.push_back(c);

            // Assign direction slots greedily: UP, DOWN, LEFT (max degree 3)
            static const Direction dirs[3] = {UP, DOWN, LEFT};
            int dir_u = -1, dir_v = -1;
            for (int d : dirs) {
                if (sites[u].bond_id[d] < 0 && dir_u < 0) dir_u = d;
            }
            for (int d : dirs) {
                if (sites[v].bond_id[d] < 0 && dir_v < 0) dir_v = d;
            }
            assert(dir_u >= 0 && dir_v >= 0);

            b.dir_a = dir_u;
            b.dir_b = dir_v;

            sites[u].bond_id[dir_u]     = i;
            sites[u].neighbor_id[dir_u] = v;
            sites[u].degree++;
            sites[v].bond_id[dir_v]     = i;
            sites[v].neighbor_id[dir_v] = u;
            sites[v].degree++;

            bonds.push_back(b);
        }

        // Build bond_local indices
        for (auto& s : sites) {
            int local = 0;
            for (int d = 0; d < MAX_DEGREE; d++)
                if (s.bond_id[d] >= 0)
                    s.bond_local[d] = local++;
        }

        // Build color classes from hardcoded layer data
        // Helper: find bond_id for edge (u,v)
        auto find_bond = [&](int u, int v) -> int {
            for (auto& b : bonds)
                if ((b.site_a==u && b.site_b==v) || (b.site_a==v && b.site_b==u))
                    return b.id;
            return -1;
        };

        auto add_color = [&](int color, const int pairs[][2], int n) {
            for (int i = 0; i < n; i++) {
                int u = pairs[i][0], v = pairs[i][1];
                int bid = find_bond(u, v);
                assert(bid >= 0);
                const Bond& b = bonds[bid];
                Gate g;
                g.bond_id    = bid;
                g.site_a     = b.site_a;
                g.site_b     = b.site_b;
                g.dir_a_to_b = b.dir_a;
                g.dir_b_to_a = b.dir_b;
                colors[color].push_back(g);
            }
        };

        add_color(0, HEAVY_HEX_3x3_COLOR0, HEAVY_HEX_3x3_COLOR0_N);
        add_color(1, HEAVY_HEX_3x3_COLOR1, HEAVY_HEX_3x3_COLOR1_N);
        add_color(2, HEAVY_HEX_3x3_COLOR2, HEAVY_HEX_3x3_COLOR2_N);
    }
};

/* =========================================================
 * NoisyHeavyHexGrid — heavy-hex grid with per-bond noise PTMs
 *
 * Extends FlatHeavyHexGrid with hardware noise from the PTM data.
 * The noise PTM for each bond is looked up by bond_id and gamma.
 * No runtime JSON loading — data is embedded as static C++ arrays.
 *
 * Usage:
 *   NoisyHeavyHexGrid grid(chi_max, 1.0f);
 *   const float* ptm = grid.noise_ptm(bond_id);  // 16×16 col-major
 * ========================================================= */
#include "heavyhex_noise_data.hpp"

struct NoisyHeavyHexGrid : public FlatHeavyHexGrid {
    float gamma;

    NoisyHeavyHexGrid(int chi_max_, float gamma_)
        : FlatHeavyHexGrid(chi_max_), gamma(gamma_) {}

    // Return pointer to 256-element col-major 16×16 noise PTM for this bond
    const float* noise_ptm(int bond_id) const {
        return heavy_hex_3x3_noise_ptm(bond_id, gamma);
    }
};
