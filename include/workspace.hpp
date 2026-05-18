#pragma once
/*
 * workspace.hpp
 *
 * ═══════════════════════════════════════════════════════════════════
 * PRE-ALLOCATED GPU WORKSPACE FOR GATE COMPUTATION
 * ═══════════════════════════════════════════════════════════════════
 *
 * The golden rule: ZERO hipMalloc/hipFree during simulation.
 *
 * Every GPU allocation is done ONCE at startup and reused for every
 * gate in every layer. This eliminates:
 *   - GC pressure (Julia's biggest problem)
 *   - Memory fragmentation
 *   - Pool cap issues (the 463MB cap we fought for weeks)
 *   - Synchronization overhead from allocator locks
 *
 * WHY THIS DESIGN WORKS:
 *
 *   Gate computation is stateless: given the same inputs, it always
 *   produces the same outputs, and the intermediate buffers can be
 *   reused between gates. We just need to ensure one gate finishes
 *   before the next one starts (or use separate workspaces per stream
 *   for parallel gates within a color class).
 *
 * BUFFER LAYOUT (for one gate, worst case: two interior sites):
 *
 *   LARGE BUFFERS (chi^3 × D*chi each = 50MB at chi=50):
 *     cv1, cv2    — contracted site tensors after env absorption
 *                   Shape: [chi^3, D*chi] = [125000, 100] at chi=50
 *     Q1, Q2      — orthogonal factors from CholeskyQR
 *                   Same shape as cv1, cv2
 *
 *   MEDIUM BUFFERS (D²*chi × D²*chi = 400×400 at chi=50):
 *     RR4         — A1 × A2ᵀ: bond matrix with D-indices explicit [k1D × k2D]
 *     oR4         — gate(RR4): gate-applied bond matrix, SVD input [k1D × k2D]
 *     U           — SVD left singular vectors [k1D × kfull], kfull ≤ D²*chi
 *     Vt          — SVD right singular vectors [kfull × k2D]
 *
 * BUFFER REUSE ANALYSIS (sequential execution):
 *
 *   Six of the nine named buffers serve dual roles in the gate pipeline.
 *   The lifetime diagram (Steps 1–10 as in gate.hpp):
 *
 *     cv1, cv2  : QR input (Step 4) → reconstruction GEMM output (Step 8)
 *     A1,  A2   : reshape R (Step 5) → build M from U/Vt (Step 8)
 *     Q1,  Q2   : QR output (Step 4) → GEMM factor (Step 8) + permute scratch (Step 9)
 *
 *   The remaining three medium buffers have non-overlapping lifetimes:
 *
 *     RR4 : written Step 5, read Step 6, DEAD after gate application
 *     U   :                              written Step 7, read Step 8, DEAD
 *
 *   → RR4 and U could be aliased (one fewer medium buffer = D²χ² floats).
 *     oR4 and Vt CANNOT be aliased: oR4 is the SVD input consumed in-place
 *     (rocblas_inplace flag) while Vt is written simultaneously by the same call.
 *
 *   WHY WE DO NOT ALIAS RR4 ↔ U:
 *     The saving is ~6 MB at chi=50, ~100 MB at chi=100 — meaningful, but
 *     naming RR4 as U (or vice versa) would break the self-documenting layout
 *     and require a comment explaining the aliasing at every use site.
 *     The current layout is already O-optimal: total allocation is
 *     4·large + 4·medium + small + env, and no further reduction is possible
 *     without aliasing buffers that are live in overlapping GPU streams
 *     (parallel gates in the same color class each hold independent workspaces,
 *     so inter-gate aliasing is not applicable here).
 *
 *   SMALL BUFFERS (D*chi × D*chi = 100×100 at chi=50):
 *     R1, R2      — upper triangular factors from CholeskyQR
 *     A1, A2      — M for reconstruction GEMM: M[r,c+new_chi*d] = scale(U or Vt)
 *     RR          — reserved (legacy)
 *     gram        — Gram matrix AᵀA for POTRF
 *     S           — singular values [D²*chi]
 *
 *   ENV BUFFERS (chi×chi per direction = 50×50 per bond):
 *     sqrt_env[d]     — √(BP message) for direction d
 *     inv_sqrt_env[d] — 1/√(BP message) for direction d
 *     Used to absorb/desorb environment before/after QR.
 *
 *   SCRATCH (for rocsolver calls):
 *     syevd_W, syevd_E, syevd_info — SYEVD eigenvalue computation
 *     svd_work, svd_info            — GESVD singular value decomposition
 *     potrf_info                    — POTRF Cholesky factorization
 *
 * MEMORY BUDGET (float32):
 *   chi=50: 4×50MB + small + env ≈ 200MB per workspace
 *   chi=70: 4×192MB + small + env ≈ 770MB per workspace
 *   With 4 streams: 4× the above
 *   Plus state tensors (4×4 grid at chi=50): ~208MB
 *   Total at chi=50 with 4 streams: ~1GB — fits easily in 32GB
 *
 * WORKSPACEPOOL:
 *   For parallel gate execution (gates in same color class are
 *   independent), we create N_STREAMS workspaces — one per HIP stream.
 *   Gates in the same color class can run concurrently on different
 *   streams, each using their own workspace.
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include <cstdio>
#include <vector>
#include <cassert>

struct Workspace {
    int chi_max;    // maximum bond dimension (set at startup, never changes)
    int m_max;      // max left dims = chi_max^(MAX_DEGREE-1) = chi_max^3
    int n_max;      // max right dims = chi_max * D

    /* ── Large buffers — dominate memory usage ───────────────────────── */

    // cv1, cv2: contracted site tensors after absorbing sqrt_env
    // Shape: [m_max × n_max] = [chi^3 × D*chi] col-major
    // These are the inputs to QR decomposition.
    Scalar* cv1    = nullptr;
    Scalar* cv2    = nullptr;

    // Q1, Q2: orthogonal factors from QR decomposition
    // Shape: [m_max × n_max] same as cv1, cv2
    // Q has orthonormal columns: Q^T · Q = I_{n×n}
    Scalar* Q1     = nullptr;
    Scalar* Q2     = nullptr;

    /* ── Medium buffers — (D²*chi)² each ────────────────────────────── */

    // RR4: bond matrix A1×A2ᵀ [k1D × k2D], k1D/k2D ≤ n_max*D
    // oR4: gate-applied bond matrix G(RR4), SVD input — same shape as RR4
    Scalar* RR4    = nullptr;
    Scalar* oR4    = nullptr;

    // U, Vt: SVD factors of oR4 on-device — hold full [k1D×kfull] and [kfull×k2D]
    // k1D, k2D, kfull ≤ n_max*D → medium buffer size exactly right.
    Scalar* U      = nullptr;
    Scalar* Vt     = nullptr;

    /* ── Small buffers — D*chi × D*chi each ─────────────────────────── */

    // R1, R2: upper triangular factors from QR
    // ψ_v ≈ Q · R  where R is n×n upper triangular
    Scalar* R1     = nullptr;
    Scalar* R2     = nullptr;

    // A1, A2: reused in Steps 5 and 8.
    // Step 5: R reshaped for bond GEMM: A[(r*D+d)+kD*b] = R[r+k*(b+chi_cut*d)]
    // Step 8: M for reconstruction GEMM: M[r+k*(c+new_chi*d)] = scale(U or Vt)
    Scalar* A1     = nullptr;
    Scalar* A2     = nullptr;

    // RR: reserved (legacy)
    Scalar* RR     = nullptr;

    // S: singular values from SVD [n_max*D] — kfull ≤ n_max*D = D²*chi_max
    // Sorted descending. Truncation: keep S[0..rank-1], discard rest.
    // The BP messages for next layer are set to diag(S[:rank]).
    Scalar* S      = nullptr;

    // gram: Gram matrix AᵀA for POTRF (Cholesky QR)
    // C = AᵀA is n×n positive definite.
    // POTRF gives C = R^T · R where R is upper triangular → that's our R.
    Scalar* gram   = nullptr;

    /* ── Env buffers — one chi×chi matrix per bond direction ─────────── */

    // For each bond direction d ∈ {UP, DOWN, LEFT, RIGHT}:
    //   sqrt_env[d][chi×chi]:     √(message_d)  — used BEFORE QR
    //   inv_sqrt_env[d][chi×chi]: 1/√(message_d) — used AFTER reconstruction
    //
    // The env absorption works as:
    //   contracted_v = ψ_v × sqrt_env[b0] × sqrt_env[b1] × ...
    //   reconstructed_v = Q_v × inv_sqrt_env[b0] × inv_sqrt_env[b1] × ...
    //
    // This "gauges" the tensors into a canonical form for QR.
    Scalar* sqrt_env    [MAX_DEGREE] = {};
    Scalar* inv_sqrt_env[MAX_DEGREE] = {};

    /* ── SYEVD scratch — for pseudo_sqrt ────────────────────────────── */

    // rocsolver_ssyevd computes eigendecomposition of a symmetric matrix.
    // Used to compute √(msg) and 1/√(msg) from BP messages.
    Scalar* syevd_W    = nullptr;   // eigenvalues [chi_max]
    Scalar* syevd_E    = nullptr;   // off-diagonal scratch [chi_max-1]
    int*    syevd_info = nullptr;   // convergence flag (0 = success)
    Scalar* syevd_V    = nullptr;   // eigenvector temp [chi_max²]: copy of V before sqE overwrite

    /* ── SVD scratch — for rocsolver_sgesvd ─────────────────────────── */
    Scalar* svd_work   = nullptr;   // rocsolver SVD E buffer [n_max*D]: SVD is on [k1D×k2D] where k1D≤n_max*D
    int*    svd_info   = nullptr;   // convergence flag

    /* ── POTRF info — for Cholesky factorization ─────────────────────── */
    int*    potrf_info = nullptr;   // 0=success, k>0=not positive definite

    size_t total_bytes = 0;         // total GPU memory allocated

    // ── Cholesky-QR diagnostic counters ──────────────────────────────────
    // Useful for cross-architecture comparison (e.g. MI60 vs MI100 vs MI300X):
    // unexpected fallbacks explain performance non-uniformity across problem sizes.
    int chol_attempted = 0;   // Cholesky path entered (thresholds met)
    int chol_fallback  = 0;   // SPOTRF failed → fell back to Householder
    int chol_skipped   = 0;   // matrix too small to attempt Cholesky

    /* ── Constructor: allocate everything at startup ─────────────────── */
    Workspace(int chi_max_) : chi_max(chi_max_) {
        // m_max = chi_max^(MAX_DEGREE-1) = chi^3 for 2D lattice
        m_max = 1;
        for (int i=0; i<MAX_DEGREE-1; i++) m_max *= chi_max;
        m_max += 1;  // one-element safety margin against boundary overrun in rocsolver internals

        // n_max = chi_max * D
        n_max = chi_max * D;

        size_t large  = (size_t)m_max * n_max * sizeof(Scalar);
        size_t medium = (size_t)n_max * D * n_max * D * sizeof(Scalar);
        size_t small  = (size_t)n_max * n_max * sizeof(Scalar);
        size_t env    = (size_t)chi_max * chi_max * sizeof(Scalar);

        // Large buffers
        _alloc(cv1,  large);
        _alloc(cv2,  large);
        _alloc(Q1,   large);
        _alloc(Q2,   large);

        // Medium buffers
        _alloc(RR4,  medium);
        _alloc(oR4,  medium);
        _alloc(U,    medium);
        _alloc(Vt,   medium);

        // Small buffers
        _alloc(R1,    small);
        _alloc(R2,    small);
        _alloc(A1,    small);
        _alloc(A2,    small);
        _alloc(RR,    small);
        _alloc(gram,  small);
        _alloc_scalar(S, n_max * D);

        // Env buffers
        for (int d=0; d<MAX_DEGREE; d++) {
            _alloc(sqrt_env[d],     env);
            _alloc(inv_sqrt_env[d], env);
        }

        // SYEVD scratch
        _alloc_scalar(syevd_W, chi_max);
        _alloc_scalar(syevd_E, std::max(chi_max-1, 1));
        _alloc_int(syevd_info, 1);
        _alloc(syevd_V, env);   // eigenvector temp (chi_max² = env size)

        // SVD scratch: E buffer needs min(k1D,k2D) elements where k1D = k1*D ≤ n_max*D
        _alloc_scalar(svd_work, n_max * D);
        _alloc_int(svd_info, 1);

        // POTRF info
        _alloc_int(potrf_info, 1);

        HIP_CHECK(hipDeviceSynchronize());
        printf("[Workspace] allocated %.1f MB (chi_max=%d m_max=%d n_max=%d)\n",
               total_bytes/1e6, chi_max, m_max, n_max);
    }

    ~Workspace() {
        _free(cv1);  _free(cv2);  _free(Q1);   _free(Q2);
        _free(RR4);  _free(oR4);  _free(U);    _free(Vt);
        _free(R1);   _free(R2);
        _free(A1);   _free(A2);
        _free(RR);   _free(S);    _free(gram);
        for (int d=0; d<MAX_DEGREE; d++) {
            _free(sqrt_env[d]);
            _free(inv_sqrt_env[d]);
        }
        _free(syevd_W); _free(syevd_E);
        _free_int(syevd_info);
        _free(syevd_V);
        _free(svd_work);
        _free_int(svd_info);
        _free_int(potrf_info);
    }

    Workspace(const Workspace&) = delete;
    Workspace& operator=(const Workspace&) = delete;

    /* ── Zero large buffers between layers ───────────────────────────── */
    void zero_large() {
        size_t large = (size_t)m_max * n_max * sizeof(Scalar);
        HIP_CHECK(hipMemset(cv1, 0, large));
        HIP_CHECK(hipMemset(cv2, 0, large));
        HIP_CHECK(hipMemset(Q1,  0, large));
        HIP_CHECK(hipMemset(Q2,  0, large));
    }

    /* ── Compute actual gate dimensions (may be < chi_max early on) ──── */

    // m for site v: product of all bond dims EXCEPT the cut bond
    // This is the number of rows in the QR matrix for this site
    static int compute_m(const Site& s, const std::vector<Bond>& bonds,
                         int cut_dir) {
        int m = 1;
        for (int d=0; d<MAX_DEGREE; d++) {
            if (d == cut_dir) continue;
            if (s.bond_id[d] >= 0)
                m *= bonds[s.bond_id[d]].chi;
        }
        return m;
    }

    // Overload with frozen chi
    static int compute_m(const Site& s, const std::vector<Bond>& bonds,
                         int cut_dir, const std::vector<int>& fchi) {
        int m = 1;
        for (int d=0; d<MAX_DEGREE; d++) {
            if (d == cut_dir) continue;
            if (s.bond_id[d] >= 0)
                m *= fchi[bonds[s.bond_id[d]].id];
        }
        return m;
    }

    static int compute_n(const Site& s, const std::vector<Bond>& bonds,
                         int cut_dir) {
        int chi_cut = (s.bond_id[cut_dir] >= 0) ?
                      bonds[s.bond_id[cut_dir]].chi : 1;
        return chi_cut * D;
    }

    /* ── Print buffer layout ─────────────────────────────────────────── */
    void print() const {
        printf("Workspace: chi_max=%d m_max=%d n_max=%d\n",
               chi_max, m_max, n_max);
        size_t large = (size_t)m_max * n_max * sizeof(Scalar);
        size_t small = (size_t)n_max * n_max * sizeof(Scalar);
        size_t env   = (size_t)chi_max * chi_max * sizeof(Scalar);
        size_t medium = (size_t)n_max * D * n_max * D * sizeof(Scalar);
        printf("  Large buffers  (×4): %.1f MB each\n", large/1e6);
        printf("  Medium buffers (×4): %.1f MB each\n", medium/1e6);
        printf("  Small buffers  (×6): %.3f MB each\n", small/1e6);
        printf("  Env buffers  (×%d): %.3f MB each\n", 2*MAX_DEGREE, env/1e6);
        printf("  Total: %.1f MB\n", total_bytes/1e6);
        printf("  Cholesky-QR: attempted=%d  fallback=%d  skipped=%d\n",
               chol_attempted, chol_fallback, chol_skipped);
    }

private:
    void _alloc(Scalar*& ptr, size_t bytes) {
        HIP_CHECK(hipMalloc(&ptr, bytes));
        HIP_CHECK(hipMemset(ptr, 0, bytes));
        total_bytes += bytes;
    }
    void _alloc_scalar(Scalar*& ptr, int n) {
        _alloc(ptr, (size_t)n * sizeof(Scalar));
    }
    void _alloc_int(int*& ptr, int n) {
        size_t bytes = (size_t)n * sizeof(int);
        HIP_CHECK(hipMalloc(&ptr, bytes));
        HIP_CHECK(hipMemset(ptr, 0, bytes));
        total_bytes += bytes;
    }
    void _free(Scalar*& ptr) {
        if (ptr) { HIP_CHECK(hipFree(ptr)); ptr = nullptr; }
    }
    void _free_int(int*& ptr) {
        if (ptr) { HIP_CHECK(hipFree(ptr)); ptr = nullptr; }
    }
};

/* =========================================================
 * WorkspacePool — N independent workspaces, one per HIP stream
 *
 * Within a color class, all gates are independent (no shared sites).
 * We can run them concurrently on separate HIP streams, each using
 * its own workspace to avoid conflicts.
 *
 * Usage:
 *   WorkspacePool pool(chi_max, n_streams=4);
 *   for (int i=0; i<gates.size(); i++) {
 *       apply_gate(gates[i], pool.ws(i), pool.stream(i));
 *   }
 *   hipDeviceSynchronize();  // wait for all streams
 *
 * Stream wrapping: pool.ws(i) == pool.ws(i + n_streams)
 * So we can index by gate number and it wraps automatically.
 * ========================================================= */
struct WorkspacePool {
    std::vector<Workspace*> workspaces;
    std::vector<hipStream_t> streams;
    int n_streams;

    WorkspacePool(int chi_max, int n_streams_) : n_streams(n_streams_) {
        streams.resize(n_streams);
        for (int i=0; i<n_streams; i++) {
            HIP_CHECK(hipStreamCreate(&streams[i]));
            workspaces.push_back(new Workspace(chi_max));
        }
        printf("[WorkspacePool] %d streams, %.1f MB total\n",
               n_streams,
               n_streams * workspaces[0]->total_bytes / 1e6);
    }

    ~WorkspacePool() {
        for (auto* w : workspaces) delete w;
        for (auto s : streams) HIP_CHECK(hipStreamDestroy(s));
    }

    // Get workspace for gate i (wraps around)
    Workspace& ws(int i) { return *workspaces[i % n_streams]; }

    // Get HIP stream for gate i (wraps around)
    hipStream_t stream(int i) { return streams[i % n_streams]; }

    size_t total_bytes() const {
        return n_streams * workspaces[0]->total_bytes;
    }
};
