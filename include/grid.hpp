#pragma once
/*
 * grid.hpp
 *
 * ═══════════════════════════════════════════════════════════════════
 * THE LATTICE — TOPOLOGY OF THE TENSOR NETWORK
 * ═══════════════════════════════════════════════════════════════════
 *
 * This file defines the spatial structure of the simulation:
 * which sites exist, how they connect, and in what order to apply gates.
 *
 * WHAT IS THE GRID?
 *
 *   A 2D rectangular lattice of nrows × ncols sites.
 *   Each site corresponds to one qubit (or qutrit if D=3).
 *   Each site has a tensor ψ_v stored on the GPU.
 *   Adjacent sites are connected by bonds (virtual indices).
 *
 *   4×4 example (site ids):
 *     0  1  2  3      ← row 0
 *     4  5  6  7      ← row 1
 *     8  9  10 11     ← row 2
 *    12 13  14 15     ← row 3
 *
 *   site_id = row * ncols + col   (0-based)
 *
 * SITE TYPES AND DEGREES:
 *
 *   Degree = number of bonds (neighbors).
 *   In a 2D square lattice:
 *     Corner sites (4 of them): degree 2  → shape [chi,chi,D]
 *     Edge sites (4*(n-2)):     degree 3  → shape [chi,chi,chi,D]
 *     Interior sites ((n-2)²):  degree 4  → shape [chi,chi,chi,chi,D]
 *
 *   The degree determines the tensor size:
 *     Corner:   chi^2 * D   =   5,000 floats at chi=50,D=2
 *     Edge:     chi^3 * D   = 250,000 floats at chi=50,D=2
 *     Interior: chi^4 * D   = 12.5M  floats at chi=50,D=2
 *
 * BONDS:
 *
 *   Each bond connects exactly two adjacent sites.
 *   Bond id is assigned sequentially:
 *     horizontal bonds first (row by row, left to right)
 *     then vertical bonds (col by col, top to bottom)
 *
 *   For a 4×4 grid:
 *     12 horizontal bonds (4 rows × 3 bonds/row)
 *     12 vertical bonds   (4 cols × 3 bonds/col)
 *     = 24 bonds total
 *
 *   Each bond has a current chi (bond dimension), which starts at 1
 *   and grows as gates are applied. The BP messages live on bonds.
 *
 * LOCAL BOND INDEX:
 *
 *   Within a site tensor, bonds appear in direction order:
 *     UP first (if exists), then DOWN, LEFT, RIGHT, then D
 *
 *   bond_local[d] = position of direction d's bond in the tensor shape.
 *   This is essential for mode_n_contract to know which axis to contract.
 *
 *   Example: interior site (1,1)
 *     All 4 bonds exist → shape = [chi_UP, chi_DOWN, chi_LEFT, chi_RIGHT, D]
 *     bond_local[UP]=0, bond_local[DOWN]=1, bond_local[LEFT]=2, bond_local[RIGHT]=3
 *
 *   Example: corner site (0,0)
 *     Only DOWN and RIGHT exist → shape = [chi_DOWN, chi_RIGHT, D]
 *     bond_local[UP]=-1, bond_local[LEFT]=-1
 *     bond_local[DOWN]=0, bond_local[RIGHT]=1
 *
 * EDGE COLORING — HOW GATES ARE SCHEDULED:
 *
 *   Problem: we can't apply gates on two bonds that share a site
 *   simultaneously (they'd both try to modify the same tensor).
 *
 *   Solution: color the bonds so that same-color bonds share no sites.
 *   Then we can apply all gates of one color in parallel.
 *
 *   For a 2D square lattice, 4 colors suffice:
 *
 *     Color 0 (H-even): horizontal bonds starting at even columns
 *       e.g. (r,0)-(r,1), (r,2)-(r,3), (r,4)-(r,5), ...
 *       These share no sites (col 0,1 don't overlap with col 2,3).
 *
 *     Color 1 (H-odd): horizontal bonds starting at odd columns
 *       e.g. (r,1)-(r,2), (r,3)-(r,4), ...
 *
 *     Color 2 (V-even): vertical bonds starting at even rows
 *       e.g. (0,c)-(1,c), (2,c)-(3,c), ...
 *
 *     Color 3 (V-odd): vertical bonds starting at odd rows
 *       e.g. (1,c)-(2,c), (3,c)-(4,c), ...
 *
 *   One Trotter layer applies gates in all 4 colors sequentially,
 *   with gates within each color running in parallel on HIP streams.
 *
 *   Julia uses a graph coloring algorithm (optimal) that mixes
 *   horizontal and vertical — we use this simpler 4-color scheme.
 *   Both are correct; ours is slightly simpler to implement.
 */

#include "types.hpp"
#include "tensor.hpp"
#include <vector>
#include <array>
#include <cassert>
#include <cstdio>

/* =========================================================
 * Bond — one virtual index connecting two adjacent sites
 * ========================================================= */
struct Bond {
    int site_a;   // first site id (always the one with lower id)
    int site_b;   // second site id (always higher than site_a)
    int dir_a;    // direction from site_a to site_b (e.g. RIGHT)
    int dir_b;    // direction from site_b to site_a (e.g. LEFT)
    int chi;      // current bond dimension (grows from 1 to chi_max)
    int id;       // global bond index (used to index messages[])

    // Directed message indices:
    // messages[msg_from_a()] = μ_{site_a → site_b}
    // messages[msg_from_b()] = μ_{site_b → site_a}
    int msg_from_a() const { return 2*id; }
    int msg_from_b() const { return 2*id + 1; }
    // Incoming message index for a given site
    int msg_incoming(int site_id) const {
        return (site_id == site_a) ? msg_from_b() : msg_from_a();
    }
};

/* =========================================================
 * Site — one node of the lattice (one qubit)
 * ========================================================= */
struct Site {
    int id;       // global index = row * ncols + col
    int row, col; // grid coordinates (0-based)
    int degree;   // number of bonds: 2 (corner), 3 (edge), 4 (interior)

    // bond_id[d]:    global bond index for direction d, or -1 if no bond
    // neighbor_id[d]: site_id of neighbor in direction d, or -1 if no neighbor
    // bond_local[d]: position of this bond in the site's tensor shape (0-based)
    //                -1 if no bond in this direction
    std::array<int, MAX_DEGREE> bond_id;
    std::array<int, MAX_DEGREE> neighbor_id;
    std::array<int, MAX_DEGREE> bond_local;

    // Tensor shape at chi_max: [chi_b0, chi_b1, ..., D]
    // Bond dims in direction order (UP→DOWN→LEFT→RIGHT, skipping missing)
    // D (physical) always last.
    std::vector<int> tensor_shape(int chi_max) const {
        std::vector<int> shape;
        for (int d = 0; d < MAX_DEGREE; d++)
            if (bond_id[d] >= 0) shape.push_back(chi_max);
        shape.push_back(D);
        return shape;
    }

    // Tensor shape at current bond dimensions (from bonds array)
    std::vector<int> tensor_shape(const std::vector<Bond>& bonds) const {
        std::vector<int> shape;
        for (int d = 0; d < MAX_DEGREE; d++)
            if (bond_id[d] >= 0)
                shape.push_back(bonds[bond_id[d]].chi);
        shape.push_back(D);
        return shape;
    }

    // Max tensor size (number of floats) at chi_max
    int tensor_size_max(int chi_max) const {
        int s = D;
        for (int d = 0; d < MAX_DEGREE; d++)
            if (bond_id[d] >= 0) s *= chi_max;
        return s;
    }

    // Current tensor size (using actual chi per bond)
    int tensor_size(const std::vector<Bond>& bonds) const {
        int s = D;
        for (int d = 0; d < MAX_DEGREE; d++)
            if (bond_id[d] >= 0)
                s *= bonds[bond_id[d]].chi;
        return s;
    }

    void print() const {
        printf("  Site(%d,%d) id=%d degree=%d bonds=[", row, col, id, degree);
        for (int d=0; d<MAX_DEGREE; d++)
            if (bond_id[d] >= 0)
                printf("%s:%d ", dir_name((Direction)d), bond_id[d]);
        printf("]\n");
    }
};

/* =========================================================
 * Gate — a two-site operation on one bond
 *
 * Specifies which sites and in which direction the gate acts.
 * Created from bond information during edge coloring.
 * Passed to apply_gate() which does the actual computation.
 * ========================================================= */
struct Gate {
    int bond_id;      // which bond this gate acts on
    int site_a;       // first site (lower id, "left" or "top")
    int site_b;       // second site (higher id, "right" or "bottom")
    int dir_a_to_b;   // direction from site_a to site_b (e.g. RIGHT)
    int dir_b_to_a;   // direction from site_b to site_a (e.g. LEFT)
};

/* =========================================================
 * Grid — the full 2D lattice
 *
 * Created once at startup. All structural information is
 * computed in the constructor and never changes during simulation.
 * The bond chi values (in grid.bonds[i].chi) DO change as gates run.
 * ========================================================= */
struct Grid {
    int nrows, ncols;   // lattice dimensions (arbitrary, not hardcoded)
    int n_sites;        // nrows * ncols
    int n_bonds;        // horizontal + vertical bonds
    int chi_max;        // pre-allocated max bond dimension

    std::vector<Site> sites;   // indexed by site_id
    std::vector<Bond> bonds;   // indexed by bond_id

    // 4 color classes for parallel gate scheduling.
    // colors[c] = list of independent gates in color class c.
    // All gates in one color share no sites → safe to run in parallel.
    std::vector<std::vector<Gate>> colors;

    /* ── Constructor ─────────────────────────────────────────────────── */
    Grid(int nrows_, int ncols_, int chi_max_)
        : nrows(nrows_), ncols(ncols_),
          n_sites(nrows_*ncols_), chi_max(chi_max_),
          colors(4)
    {
        _build_sites();
        _build_bonds();
        _build_colors();
    }

protected:
    // Protected no-op constructor for subclasses that build their own topology
    Grid(int chi_max_, int n_colors)
        : nrows(0), ncols(0), n_sites(0), n_bonds(0),
          chi_max(chi_max_), colors(n_colors) {}

public:

    /* ── Accessors ───────────────────────────────────────────────────── */
    int site_id(int row, int col) const { return row * ncols + col; }
    const Site& site(int row, int col) const { return sites[site_id(row, col)]; }

    // Opposite direction: UP↔DOWN, LEFT↔RIGHT
    static int opposite(int d) {
        switch(d) {
            case UP:    return DOWN;
            case DOWN:  return UP;
            case LEFT:  return RIGHT;
            case RIGHT: return LEFT;
        }
        return -1;
    }

    /* ── Memory estimates ────────────────────────────────────────────── */

    // Total GPU memory for all site tensors at chi_max
    size_t total_tensor_bytes() const {
        size_t total = 0;
        for (auto& s : sites)
            total += (size_t)s.tensor_size_max(chi_max) * sizeof(Scalar);
        return total;
    }

    // Total GPU memory for all BP messages (2 per bond: one each direction)
    // Each message is chi_max × chi_max floats
    size_t total_message_bytes() const {
        return (size_t)n_bonds * 2 * chi_max * chi_max * sizeof(Scalar);
    }

    /* ── Print utilities ─────────────────────────────────────────────── */
    void print() const {
        printf("Grid %dx%d: %d sites, %d bonds, chi_max=%d\n",
               nrows, ncols, n_sites, n_bonds, chi_max);
        for (auto& s : sites) s.print();
        printf("  Edge coloring:\n");
        for (int c=0; c<4; c++) {
            printf("    Color %d (%s): %zu gates → [",
                   c, color_name(c), colors[c].size());
            for (auto& g : colors[c])
                printf("(%d-%d) ", g.site_a, g.site_b);
            printf("]\n");
        }
    }

    void print_tensor_shapes() const {
        printf("Tensor shapes (chi_max=%d, D=%d):\n", chi_max, D);
        for (auto& s : sites) {
            auto shape = s.tensor_shape(chi_max);
            printf("  Site(%d,%d): [", s.row, s.col);
            for (int i=0; i<(int)shape.size(); i++) {
                printf("%d", shape[i]);
                if (i<(int)shape.size()-1) printf(",");
            }
            int total = 1; for (int x : shape) total *= x;
            printf("]  total=%d  %.2fMB\n", total, total*4.0/1e6);
        }
    }

    static const char* color_name(int c) {
        switch(c) {
            case 0: return "H-even";  // horizontal, even col start
            case 1: return "H-odd";   // horizontal, odd col start
            case 2: return "V-even";  // vertical, even row start
            case 3: return "V-odd";   // vertical, odd row start
        }
        return "?";
    }

private:
    /* ── Build sites ─────────────────────────────────────────────────── */
    void _build_sites() {
        sites.resize(n_sites);
        for (int r=0; r<nrows; r++) {
            for (int c=0; c<ncols; c++) {
                Site& s = sites[site_id(r,c)];
                s.id  = site_id(r,c);
                s.row = r; s.col = c;
                s.bond_id.fill(-1);
                s.neighbor_id.fill(-1);
                s.bond_local.fill(-1);

                // Set neighbors (bounds-checked)
                if (r > 0)       s.neighbor_id[UP]    = site_id(r-1, c);
                if (r < nrows-1) s.neighbor_id[DOWN]   = site_id(r+1, c);
                if (c > 0)       s.neighbor_id[LEFT]   = site_id(r, c-1);
                if (c < ncols-1) s.neighbor_id[RIGHT]  = site_id(r, c+1);

                // Degree = number of existing neighbors
                s.degree = 0;
                for (int d=0; d<MAX_DEGREE; d++)
                    if (s.neighbor_id[d] >= 0) s.degree++;
            }
        }
    }

    /* ── Build bonds ─────────────────────────────────────────────────── */
    void _build_bonds() {
        int bid = 0;

        // Horizontal bonds first (row-major order)
        // Bond goes from site (r,c) to (r,c+1) in direction RIGHT/LEFT
        for (int r=0; r<nrows; r++) {
            for (int c=0; c<ncols-1; c++) {
                int sa = site_id(r, c);
                int sb = site_id(r, c+1);
                Bond b;
                b.id=bid; b.site_a=sa; b.site_b=sb;
                b.dir_a=RIGHT; b.dir_b=LEFT; b.chi=1;
                bonds.push_back(b);
                sites[sa].bond_id[RIGHT] = bid;
                sites[sb].bond_id[LEFT]  = bid;
                bid++;
            }
        }

        // Vertical bonds second (col-major order)
        // Bond goes from site (r,c) to (r+1,c) in direction DOWN/UP
        for (int r=0; r<nrows-1; r++) {
            for (int c=0; c<ncols; c++) {
                int sa = site_id(r, c);
                int sb = site_id(r+1, c);
                Bond b;
                b.id=bid; b.site_a=sa; b.site_b=sb;
                b.dir_a=DOWN; b.dir_b=UP; b.chi=1;
                bonds.push_back(b);
                sites[sa].bond_id[DOWN] = bid;
                sites[sb].bond_id[UP]   = bid;
                bid++;
            }
        }

        n_bonds = bid;

        // Build local bond indices:
        // bond_local[d] = which position in the tensor shape this bond occupies.
        // We assign indices in direction order: UP first, then DOWN, LEFT, RIGHT.
        // Skipped directions don't consume an index.
        for (auto& s : sites) {
            int local = 0;
            for (int d=0; d<MAX_DEGREE; d++) {
                if (s.bond_id[d] >= 0)
                    s.bond_local[d] = local++;
            }
        }
    }

    /* ── Build edge coloring ─────────────────────────────────────────── */
    void _build_colors() {
        // Color assignment rule:
        //   Horizontal bonds: color = col_a % 2  (0 or 1)
        //   Vertical bonds:   color = 2 + row_a % 2  (2 or 3)
        //
        // Why this works:
        //   Two horizontal bonds with the same col_a parity are separated
        //   by at least 2 columns — they share no sites.
        //   Same logic for vertical bonds and row parity.
        for (auto& b : bonds) {
            Gate g;
            g.bond_id    = b.id;
            g.site_a     = b.site_a;
            g.site_b     = b.site_b;
            g.dir_a_to_b = b.dir_a;
            g.dir_b_to_a = b.dir_b;

            if (b.dir_a == RIGHT) {
                // Horizontal: color by starting column parity
                int col_a = sites[b.site_a].col;
                colors[col_a % 2].push_back(g);
            } else {
                // Vertical (dir_a == DOWN): color by starting row parity
                int row_a = sites[b.site_a].row;
                colors[2 + row_a % 2].push_back(g);
            }
        }
    }
};
