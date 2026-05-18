#pragma once
/*
 * tensor.hpp
 *
 * ═══════════════════════════════════════════════════════════════════
 * THE TENSOR — THE WORKHORSE OF THE SIMULATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * A Tensor<T> is a multi-dimensional array stored as a flat GPU buffer.
 * All the physics lives in tensors. Understanding this class is the
 * foundation for understanding everything else.
 *
 * STORAGE CONVENTION: COL-MAJOR (FORTRAN ORDER)
 *
 *   For a 2D matrix A[m×n], element A[row,col] is stored at:
 *     flat_index = row + m * col
 *
 *   For a general tensor T[d0, d1, d2, ...]:
 *     flat_index = i0 + d0*(i1 + d1*(i2 + d2*(...)))
 *
 *   This matches rocBLAS convention — no transposition needed for GEMM.
 *
 * SHAPE CONVENTION FOR SITE TENSORS:
 *
 *   A site tensor always has the physical index D LAST:
 *     [chi_b0, chi_b1, ..., chi_b{degree-1}, D]
 *
 *   Bond indices appear in direction order (UP=0, DOWN=1, LEFT=2, RIGHT=3),
 *   but only for bonds that actually exist at that site.
 *
 *   Examples for chi=50, D=2:
 *     Corner site (2 bonds):   [50, 50, 2]         = 5,000 floats = 20KB
 *     Edge site   (3 bonds):   [50, 50, 50, 2]     = 250,000 floats = 1MB
 *     Interior    (4 bonds):   [50, 50, 50, 50, 2] = 12,500,000 floats = 50MB
 *
 * WHY PHYSICAL INDEX LAST?
 *
 *   The QR decomposition splits the tensor into:
 *     LEFT indices (bond dims without cut bond) → rows of the matrix
 *     RIGHT indices (cut bond + physical dim D) → cols of the matrix
 *
 *   With D last, the right side is [chi_cut, D] which is contiguous
 *   in col-major storage — no permutation needed before QR!
 *
 *   Julia uses [D, chi, chi, ...] (physical first) which DOES require
 *   a permutation before QR. Our convention avoids this entirely.
 *
 * MATRIX VIEW FOR QR/SVD:
 *
 *   Before QR on a bond (say RIGHT for an interior site):
 *     shape = [chi_UP, chi_DOWN, chi_LEFT, chi_RIGHT, D]
 *     split at n_left=3:
 *       m (rows) = chi_UP * chi_DOWN * chi_LEFT = chi^3
 *       n (cols) = chi_RIGHT * D                = chi*D
 *     Matrix view: data pointer is ALREADY the m×n col-major matrix!
 *     No copy, no permutation.
 *
 *   This is the key insight: the flat array IS the QR input matrix.
 *
 * GPU MEMORY OWNERSHIP:
 *
 *   Tensor owns its GPU memory by default (owned=true).
 *   When owned=true, destructor calls hipFree(data).
 *   Non-owning views (owned=false) can point into pre-allocated workspace.
 *   No copy constructor — tensors are large, copies must be explicit.
 */

#include "types.hpp"
#include <vector>
#include <numeric>
#include <cassert>
#include <cstring>
#include <cstdio>

template<typename T = Scalar>
struct Tensor {
    T*               data   = nullptr;  // GPU device pointer (col-major flat)
    std::vector<int> shape;             // dimension sizes [d0, d1, ..., dn]
    int              ndim   = 0;        // number of dimensions
    int              total  = 0;        // total elements = product of shape
    bool             owned  = true;     // if true, hipFree in destructor

    /* ── Constructors ──────────────────────────────────────────────────── */

    Tensor() = default;

    // Allocate a new GPU tensor with given shape, initialized to zero
    explicit Tensor(std::vector<int> shape_)
        : shape(std::move(shape_)), ndim((int)shape.size()), owned(true)
    {
        total = 1;
        for (int s : shape) total *= s;
        HIP_CHECK(hipMalloc(&data, (size_t)total * sizeof(T)));
        HIP_CHECK(hipMemset(data,  0, (size_t)total * sizeof(T)));
    }

    // Non-owning view into existing GPU memory (e.g. workspace buffer)
    // The caller is responsible for keeping the memory alive
    Tensor(T* ptr, std::vector<int> shape_, bool owned_=false)
        : data(ptr), shape(std::move(shape_)),
          ndim((int)shape.size()), owned(owned_)
    {
        total = 1;
        for (int s : shape) total *= s;
    }

    // No copy — tensors are large GPU arrays, copies must be intentional
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // Move constructor — transfers ownership
    Tensor(Tensor&& o) noexcept
        : data(o.data), shape(std::move(o.shape)),
          ndim(o.ndim), total(o.total), owned(o.owned)
    { o.data = nullptr; o.owned = false; }

    Tensor& operator=(Tensor&& o) noexcept {
        if (this != &o) {
            free_data();
            data  = o.data;  shape = std::move(o.shape);
            ndim  = o.ndim;  total = o.total;  owned = o.owned;
            o.data = nullptr; o.owned = false;
        }
        return *this;
    }

    ~Tensor() { free_data(); }

    /* ── Memory operations ─────────────────────────────────────────────── */

    void free_data() {
        if (owned && data) { HIP_CHECK(hipFree(data)); data = nullptr; }
    }

    size_t bytes() const { return (size_t)total * sizeof(T); }

    // Upload from host std::vector (must have correct size)
    void copy_from_host(const std::vector<T>& h) {
        assert((int)h.size() == total);
        HIP_CHECK(hipMemcpy(data, h.data(), bytes(), hipMemcpyHostToDevice));
    }

    // Download to host std::vector
    std::vector<T> copy_to_host() const {
        std::vector<T> h(total);
        HIP_CHECK(hipMemcpy(h.data(), data, bytes(), hipMemcpyDeviceToHost));
        return h;
    }

    // Zero out all elements
    void zero() { HIP_CHECK(hipMemset(data, 0, bytes())); }

    /* ── Shape utilities ───────────────────────────────────────────────── */

    // Product of shape[i..j] inclusive
    int size_range(int i, int j) const {
        int s = 1;
        for (int k = i; k <= j; k++) s *= shape[k];
        return s;
    }

    // Matrix view dimensions for QR/SVD split.
    // Split point n_left: first n_left dims become rows, rest become cols.
    //
    // Example: shape=[50,50,50,2], n_left=3
    //   m = 50*50*50 = 125000  (rows = product of first 3 dims)
    //   n = 2        = 2       (cols = product of last 1 dim)
    //
    // The data pointer is ALREADY a valid m×n col-major matrix — no copy!
    std::pair<int,int> matrix_dims(int n_left) const {
        assert(n_left > 0 && n_left < ndim);
        int m = size_range(0, n_left - 1);
        int n = size_range(n_left, ndim - 1);
        return {m, n};
    }

    // Flat index for a multi-index (col-major)
    // idx = {i0, i1, ..., in} → i0 + d0*(i1 + d1*(i2 + ...))
    int flat_index(const std::vector<int>& idx) const {
        assert((int)idx.size() == ndim);
        int flat = 0, stride = 1;
        for (int k = 0; k < ndim; k++) {
            flat   += idx[k] * stride;
            stride *= shape[k];
        }
        return flat;
    }

    /* ── Debug utilities ───────────────────────────────────────────────── */

    void print_shape(const char* name = "") const {
        printf("Tensor %s: [", name);
        for (int k = 0; k < ndim; k++) {
            printf("%d", shape[k]);
            if (k < ndim-1) printf(", ");
        }
        printf("] total=%d (%.2f MB)\n", total, bytes()/1e6);
    }

    // Print first n elements (downloads from GPU — use sparingly)
    void print_elements(int n = 8, const char* name = "") const {
        auto h = copy_to_host();
        printf("Tensor %s first %d elements: ", name, n);
        for (int i = 0; i < std::min(n, total); i++)
            printf("%.4f ", (float)h[i]);
        printf("\n");
    }

    // Frobenius norm squared (downloads — use for debugging only)
    double norm_sq_host() const {
        auto h = copy_to_host();
        double s = 0;
        for (auto x : h) s += (double)x * x;
        return s;
    }
};

/* =========================================================
 * MatrixView — lightweight 2D view of a Tensor
 *
 * Does NOT own memory — just holds a pointer + dimensions.
 * Used to pass tensor data to rocBLAS GEMM calls without copying.
 *
 * For a tensor of shape [d0, d1, ..., dn] split at n_left:
 *   m  = d0 * d1 * ... * d_{n_left-1}   (rows)
 *   n  = d_{n_left} * ... * d_n          (cols)
 *   ld = m  (leading dimension = rows for col-major, no padding)
 *
 * The data pointer of the Tensor IS the matrix data — zero overhead.
 * ========================================================= */
template<typename T = Scalar>
struct MatrixView {
    T*  data;   // GPU pointer (not owned)
    int m;      // rows
    int n;      // cols
    int ld;     // leading dimension = m for col-major

    // Construct from a Tensor with a given split point
    MatrixView(Tensor<T>& t, int n_left_dims)
        : data(t.data), ld(0)
    {
        auto [m_, n_] = t.matrix_dims(n_left_dims);
        m = m_; n = n_; ld = m;
    }

    // Direct construction from raw pointer and dimensions
    MatrixView(T* ptr, int m_, int n_)
        : data(ptr), m(m_), n(n_), ld(m_) {}

    size_t bytes() const { return (size_t)m * n * sizeof(T); }
};
