#pragma once
/*
 * types.hpp
 *
 * ═══════════════════════════════════════════════════════════════════
 * FUNDAMENTAL TYPE DEFINITIONS FOR CppSim
 * ═══════════════════════════════════════════════════════════════════
 *
 * This file defines the basic building blocks that everything else
 * depends on. Keep it minimal — no heavy includes, no logic.
 *
 * KEY DESIGN DECISIONS:
 *
 * 1. REAL float32 instead of ComplexFloat32
 *    The Ising Hamiltonian H = h·ΣZ + J·ΣXX has real matrix elements.
 *    In the Heisenberg picture, operators that start real stay real.
 *    Julia uses ComplexF32 defensively — this wastes 2× memory and
 *    halves GEMM throughput. We use float throughout:
 *      Julia ComplexF32 at chi=50: ~400MB workspace
 *      Our float32 at chi=50:      ~200MB workspace
 *      → we can push to chi=70-80 in the same 32GB
 *
 * 2. D as compile-time constant (default 2 for qubits)
 *    The physical dimension D is known at compile time for a given
 *    simulation. This lets the compiler optimize inner loops.
 *    Change at build time: make PHYS_DIM=3 for qutrits.
 *    The code never assumes D==2 — all loops use the D constant.
 *
 * 3. Col-major storage throughout
 *    rocBLAS (AMD's GPU BLAS) uses col-major by default.
 *    Element [row, col] of an m×n matrix is at index row + m*col.
 *    All tensors follow this convention — no transposition overhead.
 *
 * 4. MAX_DEGREE=4 for 2D square lattice
 *    Each site has at most 4 neighbors: UP, DOWN, LEFT, RIGHT.
 *    This is a compile-time constant for array sizing.
 *    It doesn't assume 4 neighbors exist — it's just the maximum.
 */

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

/* =========================================================
 * Scalar type — float32 for all tensor elements
 *
 * To switch to float64: change to double and rebuild.
 * float32 is sufficient because:
 *   - Truncation error in SVD is ~1e-6 to 1e-8
 *   - float32 precision is ~1e-7
 *   - Going to float64 doesn't improve physics, just wastes memory
 * ========================================================= */
using Scalar = float;

/* =========================================================
 * Physical dimension D
 *
 * D=2: qubits (spin-1/2), Pauli operators {I, X, Y, Z}
 * D=3: qutrits (spin-1), relevant for solid-state systems
 * D=4: two-qubit local Hilbert space (for certain encodings)
 *
 * Set at compile time: hipcc -DPHYS_DIM=2 ...
 * ========================================================= */
#ifndef PHYS_DIM
#define PHYS_DIM 2
#endif
static constexpr int D = PHYS_DIM;

/* =========================================================
 * Maximum coordination number (bonds per site)
 *
 * For a 2D square lattice: UP, DOWN, LEFT, RIGHT = 4.
 * Corner sites only have 2, edge sites have 3, interior have 4.
 * This constant sizes arrays — not all entries will be used.
 * ========================================================= */
static constexpr int MAX_DEGREE = 4;

/* =========================================================
 * Error-check macros
 *
 * These wrap every HIP/rocBLAS call to catch errors immediately.
 * In production you might make them no-ops for speed, but during
 * development they are essential.
 *
 * HIP_CHECK: checks hipError_t return codes
 * RB_CHECK:  checks rocblas_status return codes (also used for rocsolver)
 * ========================================================= */
#define HIP_CHECK(cmd) do {                                         \
    hipError_t _e = (cmd);                                          \
    if (_e != hipSuccess) {                                         \
        fprintf(stderr, "[HIP] %s at %s:%d\n",                     \
            hipGetErrorString(_e), __FILE__, __LINE__);             \
        exit(1);                                                    \
    }                                                               \
} while(0)

#define RB_CHECK(cmd) do {                                          \
    rocblas_status _s = (cmd);                                      \
    if (_s != rocblas_status_success) {                             \
        fprintf(stderr, "[rocBLAS] error %d at %s:%d\n",           \
            (int)_s, __FILE__, __LINE__);                           \
    }                                                               \
} while(0)

/* =========================================================
 * Bond dimension tracking
 *
 * chi_max: the maximum bond dimension we pre-allocate for.
 *          Set at startup based on available GPU memory.
 *          Never changes during simulation.
 *
 * chi[bond]: the CURRENT bond dimension for each bond.
 *            Starts at 1 (product state) and grows toward chi_max
 *            as the Trotter layers run and entanglement builds up.
 *            After chi reaches chi_max, SVD truncation keeps it there.
 *
 * Why track per-bond chi?
 *   Early layers have chi << chi_max → much cheaper computation.
 *   A chi=5 gate takes microseconds; a chi=50 gate takes 37ms.
 *   By using actual chi (not chi_max) in computations, early
 *   layers are fast and we pay the full cost only at steady state.
 * ========================================================= */
struct BondDims {
    int chi_max;
    int* chi;    // current chi per bond (host array, indexed by bond_id)
    int  n_bonds;

    BondDims(int chi_max_, int n_bonds_)
        : chi_max(chi_max_), n_bonds(n_bonds_) {
        chi = new int[n_bonds_];
        for (int i=0; i<n_bonds_; i++) chi[i] = 1;  // start at chi=1
    }
    ~BondDims() { delete[] chi; }

    int max_chi() const {
        int m = 0;
        for (int i=0; i<n_bonds; i++) m = std::max(m, chi[i]);
        return m;
    }
    int min_chi() const {
        int m = chi_max;
        for (int i=0; i<n_bonds; i++) m = std::min(m, chi[i]);
        return m;
    }
};

/* =========================================================
 * Direction enum — cardinal directions on the 2D lattice
 *
 * Each site has up to 4 bonds, one per direction.
 * We use UP=0, DOWN=1, LEFT=2, RIGHT=3 consistently:
 *   - For indexing bond_id[] and neighbor_id[] arrays in Site
 *   - For the tensor shape order: [chi_UP, chi_DOWN, chi_LEFT, chi_RIGHT, D]
 *   - The physical index D is ALWAYS last
 *
 * Note: UP means row-1 (lower row index), DOWN means row+1.
 * ========================================================= */
enum Direction { UP=0, DOWN=1, LEFT=2, RIGHT=3 };

static const char* dir_name(Direction d) {
    switch(d) {
        case UP:    return "UP";
        case DOWN:  return "DOWN";
        case LEFT:  return "LEFT";
        case RIGHT: return "RIGHT";
    }
    return "?";
}
