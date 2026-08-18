#pragma once
/*
 * measures.hpp
 *
 * ═══════════════════════════════════════════════════════════════════
 * WHAT THIS FILE DOES — READ THIS FIRST
 * ═══════════════════════════════════════════════════════════════════
 *
 * We want to measure physical quantities from the tensor network
 * state O(t) — specifically the autocorrelation function:
 *
 *   C(t) = inner(ψ(t), ψ0)
 *
 * where ψ0 is the initial state (Z at center, I elsewhere) and
 * inner(ψ, ψ0) contracts the physical index d at each site:
 *   center site: picks d=3 (Z-component)
 *   other sites: picks d=0 (I-component)
 *
 * The denominator is 1 by normalization of Pauli operators:
 *   inner(ψ(0), ψ(0)) = inner(Z, Z) = Tr(Z²) = 1.
 *
 * C(t=0) = 1 by construction.
 * C(t→∞) → 0 as the operator spreads and loses memory of Z.
 *
 * ═══════════════════════════════════════════════════════════════════
 * WHY WE NEED BP (BELIEF PROPAGATION) FOR THE TRACE
 * ═══════════════════════════════════════════════════════════════════
 *
 * Exact trace: Tr(O) = Σ_{all bond indices} ∏_v ψ_v[b1,b2,...,d,d]
 * This is exponentially expensive — 2^n_bonds tensor contractions.
 *
 * BP approximation: factorise the trace using the BP messages
 * (which we already maintain during the simulation for the gates).
 * The BP messages encode the "environment" seen by each site.
 *
 * The Bethe approximation gives:
 *
 *   Tr(O) ≈ Z_BP = (∏_v vertex_scalar(v)) / (∏_e edge_scalar(e))
 *
 * where:
 *   vertex_scalar(v) = local contribution from site v
 *                    = Tr_local(ψ_v · incoming_messages · O_v)
 *   edge_scalar(e)   = correction term for each bond
 *                    = Tr(message(e))  [prevents double-counting]
 *
 * This is exact for tree graphs and a good approximation for 2D.
 * Julia uses exactly this: partitionfunction(bp_cache) = exp(freenergy)
 *
 * ═══════════════════════════════════════════════════════════════════
 * HOW vertex_scalar(v) IS COMPUTED
 * ═══════════════════════════════════════════════════════════════════
 *
 * For a site v with tensor ψ_v of shape [chi_UP, chi_DOWN, chi_LEFT, D]:
 *
 * Step 1: Start with ψ_v
 *
 * Step 2: For each bond direction d (UP, DOWN, LEFT, ...):
 *   Absorb the BP message msg_d[chi×chi] into the bond index:
 *   ψ_v[..., b_d, ...] → Σ_{b'} ψ_v[...,b',,...] × msg_d[b',b_d]
 *   This is a mode-n product (one GEMM per bond).
 *
 * Step 3: Contract physical index with operator O[D×D]:
 *   ψ_v[..., d] → Σ_{d'} ψ_v[...,d'] × O[d',d]
 *   For O=I: no change (identity)
 *   For O=Z: flips sign of d=1 component
 *
 * Step 4: Sum all remaining elements → scalar
 *   This is the local trace: Σ_{all indices} ψ_contracted[...]
 *
 * ═══════════════════════════════════════════════════════════════════
 * HOW edge_scalar(e) IS COMPUTED
 * ═══════════════════════════════════════════════════════════════════
 *
 * Each BP message msg_e[chi×chi] is counted once in each adjacent
 * vertex_scalar, so it's double-counted. We divide by edge_scalar(e)
 * to correct for this:
 *
 *   edge_scalar(e) = Tr(msg_e) = Σ_i msg_e[i,i]
 *
 * This is just the trace of the message matrix.
 *
 * ═══════════════════════════════════════════════════════════════════
 * MEMORY LAYOUT
 * ═══════════════════════════════════════════════════════════════════
 *
 * - Messages: msg[bond_id] → float* on GPU, shape [chi×chi] col-major
 * - Site tensors: psi[site_id] → float* on GPU, shape [chi_b0,...,D]
 * - Operators: O[D×D] col-major host vector, uploaded per call
 * - Scratch: reuses workspace.cv1 and workspace.cv2 (no new allocations)
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <cmath>
#include <vector>
#include <cstdio>
#include <cassert>
#include <numeric>
#include <limits>
#include <chrono>

/* ── write_diagonal_message ─────────────────────────────────────────────────
 * Writes a chi×chi diagonal matrix from src[chi] (already normalized by
 * inv_sum) into dst[chi×chi], zeroing all off-diagonal entries.
 * Must run on the same HIP stream as the rocBLAS handle to preserve
 * ordering with the sgemv calls that produced d_mu (src). */
__global__ void write_diagonal_message(const Scalar* src, Scalar* dst,
                                        int chi, Scalar inv_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chi * chi) return;
    int row = idx % chi;
    int col = idx / chi;
    dst[idx] = (row == col) ? src[row] * inv_sum : 0.f;
}

/* ── upcast_f2d ──────────────────────────────────────────────────────────────
 * Upcasts float→double element-wise on GPU. Used to feed f32 psi/messages
 * into the f64 BP path without changing any storage format. */
__global__ void upcast_f2d(const float* __restrict__ src,
                            double*      __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = (double)src[idx];
}

/* =========================================================
 * Measurement operators in Pauli basis (D=4)
 *
 * Pauli basis: ψ_v = [a_I, a_X, a_Y, a_Z]
 *
 * We measure C(t) = inner(ψ(t), ψ0) matching Julia's:
 *   tr_ψtψ0 = inner(ψ, ψ0; alg="bp")   # ψ0=Z at center, I elsewhere
 *
 * In Julia's BilinearForm, inner(ψ, ϕ) contracts ψ_v[d]*ϕ_v[d]
 * at each site (summing over physical index d).
 *
 * For ϕ = ψ0 (Z at center=[0,0,0,1], I elsewhere=[1,0,0,0]):
 *   center: ψ_center[3]  (Z-component)
 *   others: ψ_v[0]       (I-component)
 *
 * Implemented as diagonal selector matrices:
 *   I selector: diag(1,0,0,0) → picks a_I = ψ[0]  (non-center sites)
 *   Z selector: diag(0,0,0,1) → picks a_Z = ψ[3]  (center site)
 *
 * C(0) = 1 by normalization of Pauli operators (hardcoded in main.cpp).
 * ========================================================= */
struct Operators {
    // I_selector: picks a_I = ψ[0] component (I-component of Pauli expansion)
    // Matches Julia's inner(ψ, id_state) where id_state = [1,0,0,0] at each site.
    // inner(ψ,ϕ) contracts ψ[d]*ϕ[d] — for ϕ=[1,0,0,0] that selects ψ[0] only.
    static std::vector<Scalar> I() {
        std::vector<Scalar> m(D*D, 0.f);
        m[0+D*0] = 1.f;  // only picks index 0 (I-component)
        return m;
    }
    // Z_selector: picks a_Z = ψ[3] component
    // Matches Julia's inner(ψ, ψ0) where ψ0 center = [0,0,0,1].
    static std::vector<Scalar> Z() {
        std::vector<Scalar> m(D*D, 0.f);
        m[3+D*3] = 1.f;  // Z is index 3 in {I,X,Y,Z}
        return m;
    }
    // X_selector: picks a_X = ψ[1] component
    static std::vector<Scalar> X() {
        std::vector<Scalar> m(D*D, 0.f);
        m[1+D*1] = 1.f;
        return m;
    }
    // Y_selector: picks a_Y = ψ[2] component
    static std::vector<Scalar> Y() {
        std::vector<Scalar> m(D*D, 0.f);
        m[2+D*2] = 1.f;
        return m;
    }
    // rho_selector for ρ_v = (I + s·Z)/2
    // In Pauli basis vector form: [1, 0, 0, s] (normalization absorbed)
    // s=+1 for |0>, s=-1 for |1>
    static std::vector<Scalar> rho(int spin) {  // spin = +1 or -1
        std::vector<Scalar> m(D*D, 0.f);
        m[0+D*0] = 1.f;           // I component
        m[3+D*3] = 1.f * spin;    // Z component with sign
        return m;
    }
    // Trace: full D×D identity — sums ALL Pauli components.
    // Used in autocorrelation_with_bp to compute per-site Frobenius
    // vertex_scalars for rescaling (brings psi tensors to O(1) numerically).
    static std::vector<Scalar> Trace() {
        std::vector<Scalar> m(D*D, 0.f);
        for (int d=0; d<D; d++) m[d+D*d] = 1.f;
        return m;
    }
};

/* =========================================================
 * mode_n_scale
 *
 * Scales one index of a tensor by a chi-vector (diagonal message).
 * Equivalent to mode_n_contract with a diagonal matrix, but O(chi×volume)
 * instead of O(chi²×volume) — exact for diagonal messages, no SGEMM.
 *
 * Given tensor `in` with shape [s0,...,s_{mode}=chi,...,s_{n-1}]
 * and vector `scale` of length chi:
 *   out[i0,...,k,...,i_{n-1}] = in[i0,...,k,...,i_{n-1}] × scale[k]
 *
 * in-place: out may alias in.
 * ========================================================= */
__global__ void mode_n_scale_kernel(
    const Scalar* __restrict__ in,
    const Scalar* __restrict__ scale,  // chi-vector
    Scalar*                    out,
    int L,    // product of dims before mode
    int chi,  // size of scaled mode
    int R)    // product of dims after mode
{
    // Memory layout matches rocBLAS col-major strided-batched:
    // R batches of [L × chi] col-major matrices.
    // Element at batch r, row l, col k: index = r*(L*chi) + k*L + l
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = L * chi * R;
    if (idx >= total) return;
    int l =  idx % L;
    int k = (idx / L) % chi;
    int r =  idx / (L * chi);
    int pos = r * (L * chi) + k * L + l;
    out[pos] = in[pos] * scale[k];
}

static void mode_n_scale(
    const Scalar* in, const int* shape, int ndim,
    const Scalar* scale, int chi, int mode,
    Scalar* out)
{
    int L=1, R=1;
    for (int i=0;      i<mode;  i++) L *= shape[i];
    for (int i=mode+1; i<ndim;  i++) R *= shape[i];
    int total = L * chi * R;
    int blocks = (total + 255) / 256;
    mode_n_scale_kernel<<<blocks, 256>>>(in, scale, out, L, chi, R);
}

/* =========================================================
 * mode_n_contract
 *
 * Contracts one index of a tensor with a matrix E.
 *
 * Given tensor `in` with shape [s0, s1, ..., s_{mode}, ..., s_{n-1}]
 * and matrix E of shape [chi_out × chi_in]:
 *
 *   out[i0,...,j,...,i_{n-1}] = Σ_k in[i0,...,k,...,i_{n-1}] × E[j,k]
 *
 * where the sum is over index k at position `mode`.
 *
 * Implementation:
 *   L = product of dimensions before `mode`   (left block size)
 *   K = shape[mode]                            (contracted dimension)
 *   R = product of dimensions after `mode`     (right block size)
 *
 *   The tensor is viewed as R slices of [L × K] matrices.
 *   For each slice r: out_slice[L×chi_out] = in_slice[L×K] · E^T[K×chi_out]
 *   This is one GEMM call per right-slice.
 *
 * For our case: messages are chi×chi (square), so chi_in == chi_out.
 * For operators: chi_in == chi_out == D.
 * ========================================================= */
static void mode_n_contract(
    rocblas_handle handle,
    const Scalar* in,
    const int* shape, int ndim,  // current tensor shape
    const Scalar* E,              // matrix [chi_out × chi_in] col-major
    int chi_in, int chi_out,      // dimensions of E
    int mode,                     // which tensor index to contract
    Scalar* out)                  // result (same shape except shape[mode]=chi_out)
{
    // Compute left and right block sizes
    int L=1, R=1;
    for (int i=0;      i<mode;  i++) L *= shape[i];
    for (int i=mode+1; i<ndim;  i++) R *= shape[i];

    float alpha=1.f, beta=0.f;
    // Batched GEMM: R slices of [L×chi_in] × E^T, E shared (stride=0).
    // No explicit sync — default-stream ops are ordered; D2H copies below will wait.
    RB_CHECK(rocblas_sgemm_strided_batched(handle,
        rocblas_operation_none,
        rocblas_operation_transpose,
        L, chi_out, chi_in,
        &alpha,
        in,  L,       (rocblas_stride)(L * chi_in),
        E,   chi_out, (rocblas_stride)(0),
        &beta,
        out, L,       (rocblas_stride)(L * chi_out),
        R));
}

/* ── mode_n_contract_d ───────────────────────────────────────────────────────
 * Double-precision version of mode_n_contract — identical logic, rocblas_d* calls. */
static void mode_n_contract_d(
    rocblas_handle handle,
    const double* in,
    const int* shape, int ndim,
    const double* E,
    int chi_in, int chi_out,
    int mode,
    double* out)
{
    int L=1, R=1;
    for (int i=0;      i<mode;  i++) L *= shape[i];
    for (int i=mode+1; i<ndim;  i++) R *= shape[i];
    double alpha=1.0, beta=0.0;
    RB_CHECK(rocblas_dgemm_strided_batched(handle,
        rocblas_operation_none,
        rocblas_operation_transpose,
        L, chi_out, chi_in,
        &alpha,
        in,  L,       (rocblas_stride)(L * chi_in),
        E,   chi_out, (rocblas_stride)(0),
        &beta,
        out, L,       (rocblas_stride)(L * chi_out),
        R));
}

/* =========================================================
 * edge_scalar
 *
 * Computes Tr(message) = Σ_i msg[i,i] for a chi×chi message.
 *
 * Purpose: the denominator in the Bethe approximation.
 * Each bond message is absorbed into BOTH adjacent vertex_scalars,
 * so it gets counted twice. Dividing by edge_scalar corrects this.
 *
 * At chi=1: message is a scalar, Tr = just that scalar value.
 * At chi=50: message is 50×50, Tr = sum of diagonal = 50 numbers.
 *
 * This is done on host (messages are small: chi² floats).
 * ========================================================= */
static Scalar edge_scalar(const Scalar* msg, int chi) {
    std::vector<Scalar> h(chi*chi);
    HIP_CHECK(hipMemcpy(h.data(), msg, chi*chi*sizeof(Scalar),
                        hipMemcpyDeviceToHost));
    Scalar tr = 0.f;
    for (int i=0; i<chi; i++) tr += h[i + i*chi];  // diagonal elements only
    return tr;
}

/* =========================================================
 * vertex_scalar
 *
 * Local contribution of one site to the BP partition function.
 *
 * Conceptually: "what does site v contribute to the trace,
 * given the environment described by its incoming BP messages?"
 *
 * For site v with tensor ψ_v[chi_b0, chi_b1, ..., D]:
 *
 *   vertex_scalar(v) = Σ_{all indices} (ψ_v · msg_b0 · msg_b1 · ... · O)
 *
 * where each msg_bd is contracted along the bond dimension bd,
 * and O is contracted along the physical index D.
 *
 * Algorithm (ping-pong between two scratch buffers):
 *
 *   cur = ψ_v                          [chi_b0, chi_b1, ..., D]
 *   for each bond d:
 *     next = cur contracted with msg_d  [same shape — msg is square]
 *     swap(cur, next)
 *   next = cur contracted with O        [same shape — O is D×D]
 *   sum all elements of next → scalar
 *
 * Ping-pong means we alternate writing to scratch_a and scratch_b.
 * No allocation needed — both buffers pre-exist in workspace.
 *
 * Example for corner site (2 bonds, chi=50, D=2):
 *   ψ_v shape: [50, 50, 2]  (DOWN, RIGHT, physical)
 *   After msg_DOWN:  [50, 50, 2]  (still same shape)
 *   After msg_RIGHT: [50, 50, 2]
 *   After O:         [50, 50, 2]
 *   Sum all 5000 elements → one number
 * ========================================================= */
static Scalar vertex_scalar(
    rocblas_handle handle,
    const Site& site,
    const Scalar* psi,                       // site tensor on GPU
    const Grid& grid,
    const std::vector<Scalar*>& messages,    // messages[bond_id] on GPU
    const Scalar* O_gpu,                     // operator [D×D] on GPU
    Scalar* scratch_a,                       // pre-allocated scratch buffer A
    Scalar* scratch_b,                       // pre-allocated scratch buffer B
    bool full_bp = false)                    // true = norm-message quadratic form
{
    std::vector<int> shape;
    for (int d=0; d<MAX_DEGREE; d++)
        if (site.bond_id[d] >= 0)
            shape.push_back(grid.bonds[site.bond_id[d]].chi);
    shape.push_back(D);
    int ndim = (int)shape.size();

    int total = D;
    for (int d=0; d<MAX_DEGREE; d++)
        if (site.bond_id[d] >= 0)
            total *= grid.bonds[site.bond_id[d]].chi;

    HIP_CHECK(hipMemcpy(scratch_a, psi,
        total*sizeof(Scalar), hipMemcpyDeviceToDevice));

    Scalar* cur  = scratch_a;
    Scalar* next = scratch_b;
    std::vector<int> cur_shape = shape;

    // Full chi×chi mode: contract physical FIRST so T_sel is saved before
    // the message ping-pong overwrites both scratch buffers.
    // T_sel[bonds] = ψ_v contracted with O (physical selector, d→1).
    // We save it to host; after message contractions T_eff is on device
    // and we return dot(T_sel, T_eff) — the quadratic Bethe vertex form.
    std::vector<Scalar> T_sel_host;
    if (full_bp) {
        int last_mode = (int)cur_shape.size() - 1;
        mode_n_contract(handle, cur,
            cur_shape.data(), (int)cur_shape.size(),
            O_gpu, D, D, last_mode, next);
        std::swap(cur, next);
        int n_tsel = 1; for (auto s : cur_shape) n_tsel *= s;
        T_sel_host.resize(n_tsel);
        HIP_CHECK(hipMemcpy(T_sel_host.data(), cur,
            n_tsel*sizeof(Scalar), hipMemcpyDeviceToHost));
    }

    // Contract all incoming bond messages (both modes).
    for (int d=0; d<MAX_DEGREE; d++) {
        if (site.bond_id[d] < 0) continue;
        int local_idx = site.bond_local[d];
        int bond_id   = site.bond_id[d];
        const Bond& b = grid.bonds[bond_id];
        int chi       = b.chi;
        mode_n_contract(handle, cur,
            cur_shape.data(), (int)cur_shape.size(),
            messages[b.msg_incoming(site.id)], chi, chi,
            local_idx, next);
        std::swap(cur, next);
    }

    // Diagonal mode: contract physical LAST, then sum linearly.
    if (!full_bp) {
        int last_mode = (int)cur_shape.size() - 1;
        mode_n_contract(handle, cur,
            cur_shape.data(), (int)cur_shape.size(),
            O_gpu, D, D, last_mode, next);
        std::swap(cur, next);
    }

    int remaining = 1;
    for (auto s : cur_shape) remaining *= s;
    std::vector<Scalar> h(remaining);
    HIP_CHECK(hipMemcpy(h.data(), cur,
        remaining*sizeof(Scalar), hipMemcpyDeviceToHost));

    double sd = 0.0;
    if (full_bp) {
        // Quadratic form: dot(T_sel, M_gram · T_sel).
        // T_sel has O applied (physical selected); T_eff = Gram-messages · T_sel.
        for (int i=0; i<remaining; i++)
            sd += (double)T_sel_host[i] * (double)h[i];
    } else {
        for (auto x : h) sd += (double)x;
    }
    Scalar s = (Scalar)sd;
    if (!std::isfinite(s)) {
        for(int i=0;i<remaining;i++)
            if(!std::isfinite(h[i])){
                printf("[DBG vertex_scalar] NaN at site%d element[%d]=%.6g (total=%d)\n",
                       site.id, i, h[i], remaining);
                break;
            }
    }
    return s;
}

/* =========================================================
 * vertex_scalar_d
 *
 * Double-precision version of vertex_scalar (diagonal mode only).
 * Upcasts psi (float) and each bond message (float) to double before
 * contracting, accumulates in double, returns double.
 * O_gpu_d must be a D×D double matrix already on GPU.
 * dbl_a/dbl_b are pre-allocated double scratch (≥ m_max*n_max doubles).
 * dbl_msg is scratch for upcasting one message at a time (≥ chi_max² doubles).
 * ========================================================= */
static double vertex_scalar_d(
    rocblas_handle handle,
    const Site& site,
    const float* psi,
    const Grid& grid,
    const std::vector<float*>& messages,
    const double* O_gpu_d,
    double* dbl_a,
    double* dbl_b,
    double* dbl_msg)
{
    std::vector<int> shape;
    for (int d=0; d<MAX_DEGREE; d++)
        if (site.bond_id[d] >= 0)
            shape.push_back(grid.bonds[site.bond_id[d]].chi);
    shape.push_back(D);
    int ndim = (int)shape.size();

    int total = D;
    for (int d=0; d<MAX_DEGREE; d++)
        if (site.bond_id[d] >= 0)
            total *= grid.bonds[site.bond_id[d]].chi;

    hipStream_t rbstream; rocblas_get_stream(handle, &rbstream);

    // Upcast psi (float→double) into dbl_a
    int rblocks = (total + 255) / 256;
    upcast_f2d<<<rblocks, 256, 0, rbstream>>>(psi, dbl_a, total);

    double* cur  = dbl_a;
    double* next = dbl_b;
    std::vector<int> cur_shape = shape;

    // Contract all incoming bond messages (upcast each to double)
    for (int d=0; d<MAX_DEGREE; d++) {
        if (site.bond_id[d] < 0) continue;
        int local_idx = site.bond_local[d];
        int bond_id   = site.bond_id[d];
        const Bond& b = grid.bonds[bond_id];
        int chi       = b.chi;
        int msg_blocks = (chi*chi + 255) / 256;
        upcast_f2d<<<msg_blocks, 256, 0, rbstream>>>(
            messages[b.msg_incoming(site.id)], dbl_msg, chi*chi);
        mode_n_contract_d(handle, cur, cur_shape.data(), ndim,
                          dbl_msg, chi, chi, local_idx, next);
        std::swap(cur, next);
    }

    // Contract physical with O (D×D double), then sum linearly
    {
        int last_mode = (int)cur_shape.size() - 1;
        mode_n_contract_d(handle, cur, cur_shape.data(), ndim,
                          O_gpu_d, D, D, last_mode, next);
        std::swap(cur, next);
    }

    int remaining = 1;
    for (auto s : cur_shape) remaining *= s;
    std::vector<double> h(remaining);
    HIP_CHECK(hipMemcpy(h.data(), cur, remaining*sizeof(double), hipMemcpyDeviceToHost));

    double sd = 0.0;
    for (auto x : h) sd += x;
    if (!std::isfinite(sd)) {
        for (int i=0; i<remaining; i++)
            if (!std::isfinite(h[i])) {
                printf("[DBG vertex_scalar_d] NaN at site%d element[%d]=%.6g\n",
                       site.id, i, h[i]);
                break;
            }
    }
    return sd;
}

/* =========================================================
 * bp_partitionfunction
 *
 * Computes the Bethe approximation to Tr(O · ψ):
 *
 *   Z = (∏_v vertex_scalar(v)) / (∏_e edge_scalar(e))
 *
 * Arguments:
 *   grid        — lattice structure (sites, bonds)
 *   psi         — site tensors on GPU (indexed by site_id)
 *   messages    — BP messages on GPU (indexed by bond_id)
 *   O_local     — operator applied at center_site (host vector [D×D])
 *   center_site — site where O_local is applied; -1 means I everywhere
 *   scratch_a/b — pre-allocated GPU scratch buffers (from workspace)
 *
 * Notes:
 *   - O=I at all sites gives the norm Tr(ψ) (used for debug / norm_sqr)
 *   - O=Z at center gives inner(ψ, ψ0) = C(t) directly
 *
 *   - We use direct product (not log-sum) to correctly handle zeros.
 *     Zero numerator means the observable is exactly zero.
 *   - For large grids where overflow is a concern, switch to log-sum
 *     with explicit zero detection.
 * ========================================================= */
static Scalar bp_partitionfunction(
    rocblas_handle handle,
    const Grid& grid,
    const std::vector<Scalar*>& psi,
    const std::vector<Scalar*>& messages,
    const std::vector<Scalar>& O_local,
    int center_site,
    Scalar* scratch_a,
    Scalar* scratch_b,
    bool full_bp = false,
    double* dbl_a  = nullptr,   // float64 scratch (≥ m_max*n_max doubles)
    double* dbl_b  = nullptr,   // float64 scratch
    double* dbl_msg= nullptr)   // float64 scratch (≥ chi_max² doubles)
{
    // Upload identity and measurement operator to GPU in float64.
    // vertex_scalar_d requires double operators to avoid float32 overflow
    // when contracting chi=400 tensors (vertex values can be O(1e±10)).
    std::vector<Scalar> I_op = Operators::I();
    // Float32 operators (kept for full_bp path which still uses vertex_scalar)
    float *d_I, *d_O;
    HIP_CHECK(hipMalloc(&d_I, D*D*sizeof(Scalar)));
    HIP_CHECK(hipMalloc(&d_O, D*D*sizeof(Scalar)));
    HIP_CHECK(hipMemcpy(d_I, I_op.data(),    D*D*sizeof(Scalar), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_O, O_local.data(), D*D*sizeof(Scalar), hipMemcpyHostToDevice));

    // Float64 operators for vertex_scalar_d
    double *d_I_d = nullptr, *d_O_d = nullptr;
    bool use_f64 = (dbl_a && dbl_b && dbl_msg && !full_bp);
    if (use_f64) {
        HIP_CHECK(hipMalloc(&d_I_d, D*D*sizeof(double)));
        HIP_CHECK(hipMalloc(&d_O_d, D*D*sizeof(double)));
        std::vector<double> I_op_d(I_op.begin(), I_op.end());
        std::vector<double> O_op_d(O_local.begin(), O_local.end());
        HIP_CHECK(hipMemcpy(d_I_d, I_op_d.data(), D*D*sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_O_d, O_op_d.data(), D*D*sizeof(double), hipMemcpyHostToDevice));
    }

    // Numerator: log-sum of vertex scalars.
    // Use vertex_scalar_d (float64) when scratch buffers available — prevents
    // float32 overflow/underflow at large chi (vertex values can be O(1e±10)).
    double log_num = 0.0;
    double sign_num = 1.0;
    for (auto& site : grid.sites) {
        double vs_d;
        if (use_f64) {
            const double* op_d = (site.id == center_site) ? d_O_d : d_I_d;
            vs_d = vertex_scalar_d(handle, site, psi[site.id],
                                   grid, messages, op_d,
                                   dbl_a, dbl_b, dbl_msg);
        } else {
            const Scalar* op = (site.id == center_site) ? d_O : d_I;
            vs_d = (double)vertex_scalar(handle, site, psi[site.id],
                                         grid, messages, op,
                                         scratch_a, scratch_b, full_bp);
        }
        if (!std::isfinite(vs_d)) {
            if (use_f64) { HIP_CHECK(hipFree(d_I_d)); HIP_CHECK(hipFree(d_O_d)); }
            HIP_CHECK(hipFree(d_I)); HIP_CHECK(hipFree(d_O));
            return std::numeric_limits<Scalar>::quiet_NaN();
        }
        if (vs_d == 0.0) { sign_num = 0.0; break; }
        sign_num *= (vs_d > 0.0 ? 1.0 : -1.0);
        log_num  += std::log(std::abs(vs_d));
    }

    // Bethe edge correction in log-space.
    // Ze = Σ_i μ_{a→b}[i] * μ_{b→a}[i]. After BP sweeps on a signed
    // observable, messages can have negative elements, making Ze < 0.
    // We must track sign_den = Π_e sign(Ze) separately; ignoring it
    // (old: skip Ze < 0) silently flips the sign of C(t).
    double log_den  = 0.0;
    double sign_den = 1.0;
    for (auto& bond : grid.bonds) {
        if (bond.chi_alloc == 1) continue;  // outside lightcone: trivial message, contributes 0 to log Z
        int chi  = bond.chi;
        int chi2 = chi * chi;
        // Bethe edge scalar: Ze = <μ_a, μ_b>_F = Σ_{i,j} μ_a[i,j]*μ_b[i,j].
        // For diagonal messages (diag mode) off-diagonals are zero so this
        // reduces to the old diagonal dot product.  For full chi×chi messages
        // all elements are non-zero and must be included; using only the
        // diagonal here (the old code) produces a wrong denominator while
        // vertex_scalars use the full matrix — mismatch causes C(t) to blow up.
        std::vector<Scalar> ma(chi2), mb(chi2);
        HIP_CHECK(hipMemcpy(ma.data(), messages[bond.msg_from_a()],
            chi2*sizeof(Scalar), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(mb.data(), messages[bond.msg_from_b()],
            chi2*sizeof(Scalar), hipMemcpyDeviceToHost));
        double ze = 0.0;
        for(int k=0; k<chi2; k++) ze += (double)ma[k] * mb[k];
        if(std::abs(ze) > 1e-30) {
            sign_den *= (ze > 0 ? 1.0 : -1.0);
            log_den  += std::log(std::abs(ze));
        }
    }

    if (use_f64) { HIP_CHECK(hipFree(d_I_d)); HIP_CHECK(hipFree(d_O_d)); }
    HIP_CHECK(hipFree(d_I));
    HIP_CHECK(hipFree(d_O));

    if (sign_num == 0.0) return 0.f;
    Scalar result = (Scalar)(sign_num * sign_den * std::exp(log_num - log_den));
    return result;
}

/* ── normalize_and_delta ─────────────────────────────────────────────────────
 * Normalizes new BP message, computes max |new−old| on-device, writes result.
 * d_s must be filled by rocblas_sdot (device pointer mode) before launch.
 * alpha=1.0: standard update (no damping). alpha<1: new = (1-α)·old + α·normalized.
 * atomicMax trick: valid for non-negative floats (IEEE 754 bit-order ≡ int). */
__global__ void normalize_and_delta(
    const Scalar* __restrict__ d_mu,    // new unnormalized diagonal [chi]
    const Scalar* __restrict__ old_msg, // old message [chi×chi] col-major
    Scalar*                    new_msg, // output message [chi×chi] (may alias old_msg)
    const Scalar* __restrict__ d_s,     // Σ d_mu — written by rocblas_sdot
    Scalar*                    d_delta, // running max delta (atomicMax, int trick)
    int chi,
    float alpha)                         // damping factor ∈ (0,1]; 1.0 = no damping
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chi * chi) return;
    int row = idx % chi;
    int col = idx / chi;
    Scalar inv_s = (fabsf(*d_s) > 1e-12f) ? 1.f / *d_s : 1.f;
    if (row == col) {
        Scalar normalized = d_mu[row] * inv_s;
        Scalar old_val    = old_msg[idx];
        Scalar new_val    = (alpha >= 1.f) ? normalized
                                           : ((1.f - alpha) * old_val + alpha * normalized);
        Scalar diff       = fabsf(new_val - old_val);
        atomicMax(reinterpret_cast<int*>(d_delta), __float_as_int(diff));
        new_msg[idx] = new_val;
    } else {
        new_msg[idx] = 0.f;
    }
}

/* ── square_elements ─────────────────────────────────────────────────────────
 * Elementwise square: out[i] = in[i]². Used by norm BP to pre-process ψ so
 * the subsequent linear env contraction sums ψ² over all physical components. */
__global__ void square_elements(const Scalar* __restrict__ in, Scalar* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { Scalar v = in[idx]; out[idx] = v * v; }
}

/* ── reshape_for_bp ──────────────────────────────────────────────────────────
 * Transposes cur[L, chi, R] (col-major, leading dim L) into
 * A[chi, L*R] (col-major, leading dim chi).
 * Used by the full chi×chi BP path before sgemm(A, Aᵀ).
 * Element mapping: cur[l + L*c + L*chi*r] → A[c + chi*(l + L*r)] */
__global__ void reshape_for_bp(
    const Scalar* __restrict__ cur,  // [L × chi × R] col-major
    Scalar*                    A,    // [chi × L*R]   col-major  (output)
    int L, int chi, int R)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= L * chi * R) return;
    int l =  idx % L;
    int c = (idx / L) % chi;
    int r =  idx / (L * chi);
    A[c + chi * (l + L * r)] = cur[l + L * c + L * chi * r];
}

/* ── normalize_and_delta_full ────────────────────────────────────────────────
 * Full chi×chi variant: normalizes d_M[chi×chi] by trace (d_s), computes
 * max|new−old| over ALL chi² elements, writes result to new_msg.
 * d_s must contain trace(d_M) = Σ_i d_M[i + chi*i] before launch. */
__global__ void normalize_and_delta_full(
    const Scalar* __restrict__ d_M,     // new unnormalized message [chi×chi]
    const Scalar* __restrict__ old_msg, // old message [chi×chi] col-major
    Scalar*                    new_msg, // output message [chi×chi] (may alias old_msg)
    const Scalar* __restrict__ d_s,     // trace(d_M) — written by rocblas_sdot
    Scalar*                    d_delta, // running max delta (atomicMax, int trick)
    int chi)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chi * chi) return;
    Scalar inv_s = (fabsf(*d_s) > 1e-12f) ? 1.f / *d_s : 1.f;
    Scalar new_val = d_M[idx] * inv_s;
    Scalar diff    = fabsf(new_val - old_msg[idx]);
    atomicMax(reinterpret_cast<int*>(d_delta), __float_as_int(diff));
    new_msg[idx] = new_val;
}

/* ── normalize_and_delta_d ───────────────────────────────────────────────────
 * Double-precision reduction → normalize d_mu_d (double), write float message.
 * Convergence delta is tracked in float (sufficient for the stopping criterion). */
__global__ void normalize_and_delta_d(
    const double* __restrict__ d_mu,    // new unnormalized diagonal [chi] doubles
    const float*  __restrict__ old_msg, // old message [chi×chi] col-major float
    float*                     new_msg, // output message [chi×chi] float (may alias old)
    const double* __restrict__ d_s,     // Σ d_mu — written by rocblas_ddot
    float*                     d_delta, // running max delta (atomicMax, float trick)
    int chi)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chi * chi) return;
    int row = idx % chi;
    int col = idx / chi;
    double inv_s = (fabs(*d_s) > 1e-12) ? 1.0 / *d_s : 1.0;
    if (row == col) {
        float new_val = (float)(d_mu[row] * inv_s);
        float old_val = old_msg[idx];
        float diff    = fabsf(new_val - old_val);
        atomicMax(reinterpret_cast<int*>(d_delta), __float_as_int(diff));
        new_msg[idx]  = new_val;
    } else {
        new_msg[idx] = 0.f;
    }
}

/* ── BpResult ────────────────────────────────────────────────────────────────
 * Return value of bp_converge: sweep count + final max-delta.
 * Distinguishes hard layers (sweeps==n_sweeps, delta>>tol) from easy ones. */
struct BpResult { int sweeps; float delta; };

/* =========================================================
 * Measures — public measurement interface
 *
 * Usage:
 *   Measures meas(handle, workspace);
 *
 *   // After each Trotter layer:
 *   Scalar C = meas.autocorrelation(grid, psi, messages, center_site);
 *   printf("C(t) = %f\n", C);
 *
 * All methods reuse workspace.cv1 and workspace.cv2 as scratch.
 * No GPU allocations during measurement — safe to call every layer.
 * ========================================================= */
struct Measures {
    rocblas_handle handle;
    Workspace& ws;
    GateOps gate_ops;
    bool    full_bp   = false;  // false=diagonal (default), true=full chi×chi (wrong for bilinear form)
    bool    damped_bp = false;  // true=damped diagonal BP (--bp=damped)
    float   bp_alpha  = 1.0f;  // damping factor α ∈ (0,1]; 1.0=no damping (--alpha=)
    const int* initial_state_ = nullptr;  // per-site initial state: +1=|0>, -1=|1>, nullptr=all |0>

    // Norm BP messages — chi-vector (same layout as C(t) diagonal messages).
    // Populated by bp_converge_norm; used by compute_norm.
    std::vector<Scalar*> norm_bp_msg;
    std::vector<int>     norm_bp_msg_chi;  // allocated chi per slot for realloc detection

    Scalar* d_sel_full    = nullptr; // [D] ones — sums ALL physical components (norm BP)
    Scalar* d_psi_sq      = nullptr; // scratch for ψ² in compute_norm; sized chi_max^4*D
    Scalar* d_ones        = nullptr;  // GPU ones vector for GEMV reductions in bp_update
    Scalar* d_sel_I       = nullptr;  // [1×D] selector for I-component (d=0)
    Scalar* d_sel_Z       = nullptr;  // [1×D] selector for Z-component (d=3)
    Scalar* d_sel_rho_p   = nullptr;  // [1×D] = [1,0,0,+1] for spin=+1 (|0>)
    Scalar* d_sel_rho_m   = nullptr;  // [1×D] = [1,0,0,-1] for spin=-1 (|1>)
    Scalar* d_mu    = nullptr;  // GPU output buffer for reduced message (chi_max floats)
    Scalar* d_M     = nullptr;  // GPU scratch for full chi×chi message (chi_max² floats)
    Scalar* d_s     = nullptr;  // 1 float on GPU — sdot result (Σ d_mu or trace(M))
    Scalar* d_delta = nullptr;  // 1 float on GPU — running max delta per sweep

    // ── f64 diagonal BP buffers ───────────────────────────────────────────
    double* dbl_a    = nullptr; // double scratch A — size m_max*n_max doubles
    double* dbl_b    = nullptr; // double scratch B (ping-pong partner of dbl_a)
    double* dbl_msg  = nullptr; // chi_max² doubles — upcast one message at a time
    double* d_mu_d   = nullptr; // chi_max doubles — diagonal reduction result
    double* d_s_d    = nullptr; // 1 double — ddot result for normalization
    double* d_ones_d = nullptr; // m_max doubles of 1.0 — for dgemv reductions
    double* d_sel_I_d = nullptr;// [D] doubles — I selector (d=0)
    double* d_sel_Z_d = nullptr;// [D] doubles — Z selector (d=3)
    double* d_O_d    = nullptr; // D×D double — measurement operator upload slot
    double* d_I_d    = nullptr; // D×D double — identity operator (pre-loaded)

    Measures(rocblas_handle h, Workspace& ws_) : handle(h), ws(ws_), gate_ops(h) {
        // ones vector: max size = m_max = chi_max^3 (for GEMV over L or R_other)
        HIP_CHECK(hipMalloc(&d_ones, (size_t)ws.m_max * sizeof(Scalar)));
        std::vector<Scalar> ones(ws.m_max, 1.f);
        HIP_CHECK(hipMemcpy(d_ones, ones.data(), ws.m_max*sizeof(Scalar), hipMemcpyHostToDevice));

        // Physical-dim selectors [1×D] col-major: E[0 + 1*d] = 1 at d_sel
        std::vector<Scalar> sel(D, 0.f);
        HIP_CHECK(hipMalloc(&d_sel_I, D*sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_sel_Z, D*sizeof(Scalar)));
        sel[0] = 1.f;
        HIP_CHECK(hipMemcpy(d_sel_I, sel.data(), D*sizeof(Scalar), hipMemcpyHostToDevice));
        sel[0] = 0.f; sel[3] = 1.f;
        HIP_CHECK(hipMemcpy(d_sel_Z, sel.data(), D*sizeof(Scalar), hipMemcpyHostToDevice));

        HIP_CHECK(hipMalloc(&d_sel_full, D*sizeof(Scalar)));
        std::vector<Scalar> ones_D(D, 1.f);
        HIP_CHECK(hipMemcpy(d_sel_full, ones_D.data(), D*sizeof(Scalar), hipMemcpyHostToDevice));

        // ρ selectors for inner_op_state: [1,0,0,±1]
        HIP_CHECK(hipMalloc(&d_sel_rho_p, D*sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_sel_rho_m, D*sizeof(Scalar)));
        std::vector<Scalar> rho_p = {1.f, 0.f, 0.f, +1.f};
        std::vector<Scalar> rho_m = {1.f, 0.f, 0.f, -1.f};
        HIP_CHECK(hipMemcpy(d_sel_rho_p, rho_p.data(), D*sizeof(Scalar), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_sel_rho_m, rho_m.data(), D*sizeof(Scalar), hipMemcpyHostToDevice));

        // d_psi_sq: worst-case site at chi_max: chi_max^max_degree * D elements
        // Use ws.max_degree (not MAX_DEGREE=4) for heavy-hex and other degree-3 lattices
        {
            size_t psi_sq_sz = (size_t)D;
            for (int i = 0; i < ws.max_degree; i++) psi_sq_sz *= (size_t)ws.chi_max;
            HIP_CHECK(hipMalloc(&d_psi_sq, psi_sq_sz * sizeof(Scalar)));
        }

        HIP_CHECK(hipMalloc(&d_mu,    (size_t)ws.chi_max * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_M,     (size_t)ws.chi_max * ws.chi_max * sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_s,     sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_delta, sizeof(Scalar)));

        // ── f64 diagonal BP buffers ───────────────────────────────────────
        size_t dbl_large = (size_t)ws.m_max * ws.n_max * sizeof(double);
        HIP_CHECK(hipMalloc(&dbl_a,   dbl_large));
        HIP_CHECK(hipMalloc(&dbl_b,   dbl_large));
        HIP_CHECK(hipMalloc(&dbl_msg, (size_t)ws.chi_max * ws.chi_max * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_mu_d,  (size_t)ws.chi_max * sizeof(double)));
        HIP_CHECK(hipMalloc(&d_s_d,   sizeof(double)));
        HIP_CHECK(hipMalloc(&d_ones_d,(size_t)ws.m_max * sizeof(double)));
        std::vector<double> ones_d(ws.m_max, 1.0);
        HIP_CHECK(hipMemcpy(d_ones_d, ones_d.data(),
            ws.m_max*sizeof(double), hipMemcpyHostToDevice));

        std::vector<double> dsel_d(D, 0.0);
        HIP_CHECK(hipMalloc(&d_sel_I_d, D*sizeof(double)));
        HIP_CHECK(hipMalloc(&d_sel_Z_d, D*sizeof(double)));
        dsel_d[0] = 1.0;
        HIP_CHECK(hipMemcpy(d_sel_I_d, dsel_d.data(), D*sizeof(double), hipMemcpyHostToDevice));
        dsel_d[0] = 0.0; dsel_d[3] = 1.0;
        HIP_CHECK(hipMemcpy(d_sel_Z_d, dsel_d.data(), D*sizeof(double), hipMemcpyHostToDevice));

        HIP_CHECK(hipMalloc(&d_O_d, D*D*sizeof(double)));
        HIP_CHECK(hipMalloc(&d_I_d, D*D*sizeof(double)));
        std::vector<double> I_op_d(D*D, 0.0);
        for (int i=0; i<D; i++) I_op_d[i+D*i] = 1.0;
        HIP_CHECK(hipMemcpy(d_I_d, I_op_d.data(), D*D*sizeof(double), hipMemcpyHostToDevice));

        printf("[Measures] f64 BP scratch: %.1f MB (chi_max=%d)\n",
               2*dbl_large/1e6, ws.chi_max);
    }

    ~Measures() {
        if (d_ones)  HIP_CHECK(hipFree(d_ones));
        if (d_sel_I)     HIP_CHECK(hipFree(d_sel_I));
        if (d_sel_Z)     HIP_CHECK(hipFree(d_sel_Z));
        if (d_sel_rho_p) HIP_CHECK(hipFree(d_sel_rho_p));
        if (d_sel_rho_m) HIP_CHECK(hipFree(d_sel_rho_m));
        if (d_mu)    HIP_CHECK(hipFree(d_mu));
        if (d_M)     HIP_CHECK(hipFree(d_M));
        if (d_s)     HIP_CHECK(hipFree(d_s));
        if (d_delta) HIP_CHECK(hipFree(d_delta));
        if (dbl_a)    HIP_CHECK(hipFree(dbl_a));
        if (dbl_b)    HIP_CHECK(hipFree(dbl_b));
        if (dbl_msg)  HIP_CHECK(hipFree(dbl_msg));
        if (d_mu_d)   HIP_CHECK(hipFree(d_mu_d));
        if (d_s_d)    HIP_CHECK(hipFree(d_s_d));
        if (d_ones_d) HIP_CHECK(hipFree(d_ones_d));
        if (d_sel_I_d) HIP_CHECK(hipFree(d_sel_I_d));
        if (d_sel_Z_d) HIP_CHECK(hipFree(d_sel_Z_d));
        if (d_O_d)      HIP_CHECK(hipFree(d_O_d));
        if (d_I_d)      HIP_CHECK(hipFree(d_I_d));
        if (d_sel_full) HIP_CHECK(hipFree(d_sel_full));
        if (d_psi_sq)   HIP_CHECK(hipFree(d_psi_sq));
        for (auto p : norm_bp_msg)  if (p) HIP_CHECK(hipFree(p));
    }

    /* ── bp_partitionfunction_d ──────────────────────────────────────
     * Double-precision Bethe partition function (diagonal mode only).
     * Psi and messages are still stored as float; vertex_scalar_d upcasts
     * each on-the-fly. Edge scalars are already accumulated in double.
     * Returns double so callers can downcast only at the final step. */
    double bp_partitionfunction_d(
        const Grid& grid,
        const std::vector<Scalar*>& psi,
        const std::vector<Scalar*>& messages,
        const std::vector<Scalar>& O_local,
        int center_site)
    {
        // Upload O operator (float→double) into d_O_d
        std::vector<double> O_op_d(D*D);
        for (int i=0; i<D*D; i++) O_op_d[i] = (double)O_local[i];
        HIP_CHECK(hipMemcpy(d_O_d, O_op_d.data(), D*D*sizeof(double), hipMemcpyHostToDevice));
        // d_I_d is pre-loaded as identity in the constructor

        double log_num  = 0.0;
        double sign_num = 1.0;
        for (auto& site : grid.sites) {
            const double* op = (site.id == center_site) ? d_O_d : d_I_d;
            double vs = vertex_scalar_d(handle, site, psi[site.id],
                                        grid, messages, op,
                                        dbl_a, dbl_b, dbl_msg);
            if (!std::isfinite(vs)) return std::numeric_limits<double>::quiet_NaN();
            if (vs == 0.0) { sign_num = 0.0; break; }
            sign_num *= (vs > 0.0 ? 1.0 : -1.0);
            log_num  += std::log(std::abs(vs));
        }

        // Edge corrections — messages are float, accumulated in double
        double log_den  = 0.0;
        double sign_den = 1.0;
        for (auto& bond : grid.bonds) {
            if (bond.chi_alloc == 1) continue;  // outside lightcone: trivial, log Z contribution = 0
            int chi  = bond.chi;
            int chi2 = chi * chi;
            std::vector<Scalar> ma(chi2), mb(chi2);
            HIP_CHECK(hipMemcpy(ma.data(), messages[bond.msg_from_a()],
                chi2*sizeof(Scalar), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(mb.data(), messages[bond.msg_from_b()],
                chi2*sizeof(Scalar), hipMemcpyDeviceToHost));
            double ze = 0.0;
            for (int k=0; k<chi2; k++) ze += (double)ma[k] * (double)mb[k];
            if (std::abs(ze) > 1e-30) {
                sign_den *= (ze > 0 ? 1.0 : -1.0);
                log_den  += std::log(std::abs(ze));
            }
        }

        if (sign_num == 0.0) return 0.0;
        return sign_num * sign_den * std::exp(log_num - log_den);
    }

    /* ── bp_update ───────────────────────────────────────────────────
     * One sweep of BP updates for the bilinear form inner(ψ, ψ0).
     *
     * Julia's BP runs on the double-layer network ψ⊗ψ0, where the
     * physical index D is contracted at each site with ψ0_v[d]:
     *   - center site: ψ0 = Z = [0,0,0,1] → select d=3
     *   - all others:  ψ0 = I = [1,0,0,0] → select d=0
     *
     * Summing all d components mixes I/X/Y/Z and corrupts convergence
     * (gives 0.091 instead of 0.922). Selecting the correct d component
     * makes messages consistent with the specific measurement.
     *
     * Updated message from src→dst:
     *   μ[i_dst] ∝ Σ_{j≠i_dst} [Π_k μ_{k→src}[j_k]] * ψ_src[j0,..,i_dst,..,d_sel]
     */
    /* bp_update — fully GPU reduction, D2H only chi floats per message.
     *
     * After contracting all non-cut neighbor bonds on GPU, we select the
     * d_sel physical component with a [1×D] GEMM (contracts D→1), then
     * reduce the remaining [L × chi_cut × R_other] tensor to chi_cut values
     * using two rocBLAS sgemv calls (sum over L, then over R_other).
     * Only chi floats cross PCIe instead of the old 100MB tensor copy.
     */
    float bp_update(const Grid& grid,
                    const std::vector<Scalar*>& psi,
                    std::vector<Scalar*>& messages,
                    int center_site,
                    float alpha = 1.0f,
                    std::vector<Scalar*>* messages_v = nullptr)  // chi-vector output
    {
        float one=1.f, zero=0.f;
        hipStream_t rbstream; rocblas_get_stream(handle, &rbstream);

        // Reset per-sweep max delta to 0.0f (IEEE: all-zero bytes).
        HIP_CHECK(hipMemsetAsync(d_delta, 0, sizeof(Scalar), rbstream));

        for(auto& bond : grid.bonds){
            // Skip outside-lightcone bonds (chi_alloc=1): trivial identity messages,
            // no information to propagate, convergence immediate.
            if (bond.chi_alloc == 1) continue;
            int chi = bond.chi;
            for(int dir=0; dir<2; dir++){
                int src_id  = (dir==0) ? bond.site_a : bond.site_b;
                int src_dir = (dir==0) ? bond.dir_a  : bond.dir_b;
                const Site& src = grid.sites[src_id];

                std::vector<int> shape;
                for(int d=0;d<MAX_DEGREE;d++)
                    if(src.bond_id[d]>=0)
                        shape.push_back(grid.bonds[src.bond_id[d]].chi);
                shape.push_back(D);
                int ndim=(int)shape.size();
                int total=1; for(auto s:shape) total*=s;

                HIP_CHECK(hipMemcpy(ws.cv1,psi[src_id],
                    total*sizeof(Scalar),hipMemcpyDeviceToDevice));
                Scalar* cur=ws.cv1, *nxt=ws.cv2;

                // Contract all non-cut neighbor bonds on GPU.
                // Use mode_n_scale (chi-vector) when messages_v available — O(chi×vol)
                // instead of mode_n_contract chi×chi SGEMM O(chi²×vol). Exact for
                // diagonal messages; no precision loss from chi×chi matrix multiply.
                for(int d=0;d<MAX_DEGREE;d++){
                    if(src.bond_id[d]<0 || d==src_dir) continue;
                    int li=src.bond_local[d];
                    const Bond& eb=grid.bonds[src.bond_id[d]];
                    int bchi=eb.chi;
                    int msg_idx = eb.msg_incoming(src_id);
                    if(messages_v && (*messages_v)[msg_idx]) {
                        // Fast diagonal path: scale mode li by chi-vector
                        mode_n_scale(cur, shape.data(), ndim,
                                     (*messages_v)[msg_idx], bchi, li, nxt);
                    } else {
                        // Fallback: full chi×chi contraction
                        mode_n_contract(handle,cur,shape.data(),ndim,
                                       messages[msg_idx],bchi,bchi,li,nxt);
                    }
                    std::swap(cur,nxt);
                }

                int src_local = src.bond_local[src_dir];
                int L=1; for(int d=0;d<src_local;d++) L*=shape[d];
                int R_full=1; for(int d=src_local+1;d<ndim;d++) R_full*=shape[d];
                int R_other = R_full / D;

                int msg_idx = (dir==0) ? bond.msg_from_a() : bond.msg_from_b();
                int blocks  = (chi*chi + 255) / 256;

                if (!full_bp) {
                    // ── Diagonal path: select one physical component, reduce to chi vector ──
                    const Scalar* sel = (src_id == center_site) ? d_sel_Z : d_sel_I;
                    mode_n_contract(handle, cur, shape.data(), ndim,
                                    sel, D, 1, ndim-1, nxt);
                    // nxt[L, chi, R_other] — physical dim collapsed to 1.
                    if(L == 1){
                        RB_CHECK(rocblas_sgemv(handle, rocblas_operation_none,
                            chi, R_other, &one, nxt, chi, d_ones, 1, &zero, d_mu, 1));
                    } else if(R_other == 1){
                        RB_CHECK(rocblas_sgemv(handle, rocblas_operation_transpose,
                            L, chi, &one, nxt, L, d_ones, 1, &zero, d_mu, 1));
                    } else {
                        RB_CHECK(rocblas_sgemv(handle, rocblas_operation_transpose,
                            L, chi*R_other, &one, nxt, L, d_ones, 1, &zero, cur, 1));
                        RB_CHECK(rocblas_sgemv(handle, rocblas_operation_none,
                            chi, R_other, &one, cur, chi, d_ones, 1, &zero, d_mu, 1));
                    }
                    RB_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
                    RB_CHECK(rocblas_sdot(handle, chi, d_mu, 1, d_ones, 1, d_s));
                    RB_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
                    normalize_and_delta<<<blocks, 256, 0, rbstream>>>(
                        d_mu, messages[msg_idx], messages[msg_idx], d_s, d_delta, chi, alpha);
                    // Also extract diagonal to chi-vector if provided
                    if (messages_v && (*messages_v)[msg_idx]) {
                        int vblocks = (chi + 255) / 256;
                        extract_diagonal<<<vblocks, 256, 0, rbstream>>>(
                            messages[msg_idx], (*messages_v)[msg_idx], chi);
                    }
                } else {
                    // ── Full chi×chi bilinear-form path ───────────────────────────────
                    // Select the correct physical component first (same d_sel as
                    // diagonal path), then compute M = A_sel · A_sel^T.
                    // This gives the Gram matrix of the selected Pauli component:
                    //   M[i,j] = Σ_{L,R} ψ[L,i,R,d_sel] * ψ[L,j,R,d_sel]
                    // which is the natural chi×chi message for the bilinear form
                    // inner(ψ, ψ0) — consistent with the quadratic vertex formula.
                    const Scalar* sel = (src_id == center_site) ? d_sel_Z : d_sel_I;
                    mode_n_contract(handle, cur, shape.data(), ndim,
                                    sel, D, 1, ndim-1, nxt);
                    // nxt[L, chi, R_other] — physical dim collapsed to 1.
                    Scalar* A_buf = (nxt == ws.cv1) ? ws.cv2 : ws.cv1;
                    int LR_other = L * R_other;
                    int total_sel = L * chi * R_other;
                    int rblocks = (total_sel + 255) / 256;
                    reshape_for_bp<<<rblocks, 256, 0, rbstream>>>(nxt, A_buf, L, chi, R_other);
                    RB_CHECK(rocblas_sgemm(handle,
                        rocblas_operation_none, rocblas_operation_transpose,
                        chi, chi, LR_other,
                        &one,  A_buf, chi,
                               A_buf, chi,
                        &zero, d_M,   chi));
                    RB_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
                    RB_CHECK(rocblas_sdot(handle, chi, d_M, chi+1, d_ones, 1, d_s));
                    RB_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
                    normalize_and_delta_full<<<blocks, 256, 0, rbstream>>>(
                        d_M, messages[msg_idx], messages[msg_idx], d_s, d_delta, chi);
                }
            }
        }

        // D2H one float per sweep for convergence check.
        float max_change;
        HIP_CHECK(hipMemcpy(&max_change, d_delta, sizeof(float), hipMemcpyDeviceToHost));
        return max_change;
    }

    /* ── bp_update_with_rho ──────────────────────────────────────────
     * Same as bp_update but uses per-site ρ selector instead of d_sel_I.
     * Used by inner_op_state to run BP on the bilinear form ⟨ψ(t)|ρ⟩.
     * rho_spins[v]: +1 → [1,0,0,+1], -1 → [1,0,0,-1], center → Z=[0,0,0,1]
     */
    float bp_update_with_rho(const Grid& grid,
                             const std::vector<Scalar*>& psi,
                             std::vector<Scalar*>& messages,
                             int center_site,
                             const int* rho_spins,
                             float alpha = 1.0f)
    {
        float one=1.f, zero=0.f;
        hipStream_t rbstream; rocblas_get_stream(handle, &rbstream);
        HIP_CHECK(hipMemsetAsync(d_delta, 0, sizeof(Scalar), rbstream));

        for (auto& bond : grid.bonds) {
            if (bond.chi_alloc == 1) continue;  // skip outside-lightcone bonds
            int chi = bond.chi;
            for (int dir = 0; dir < 2; dir++) {
                int src_id  = (dir==0) ? bond.site_a : bond.site_b;
                int src_dir = (dir==0) ? bond.dir_a  : bond.dir_b;
                const Site& src = grid.sites[src_id];

                std::vector<int> shape;
                for (int d = 0; d < MAX_DEGREE; d++)
                    if (src.bond_id[d] >= 0)
                        shape.push_back(grid.bonds[src.bond_id[d]].chi);
                shape.push_back(D);
                int ndim = (int)shape.size();
                int total = 1; for (auto s : shape) total *= s;

                HIP_CHECK(hipMemcpy(ws.cv1, psi[src_id],
                    total*sizeof(Scalar), hipMemcpyDeviceToDevice));
                Scalar* cur = ws.cv1, *nxt = ws.cv2;

                // Contract all non-cut neighbor bonds
                for (int d = 0; d < MAX_DEGREE; d++) {
                    if (src.bond_id[d] < 0 || d == src_dir) continue;
                    int li = src.bond_local[d];
                    const Bond& eb = grid.bonds[src.bond_id[d]];
                    int bchi = eb.chi;
                    mode_n_contract(handle, cur, shape.data(), ndim,
                                   messages[eb.msg_incoming(src_id)],
                                   bchi, bchi, li, nxt);
                    std::swap(cur, nxt);
                }

                // Select physical component with ρ selector per site
                const Scalar* sel;
                if (src_id == center_site)         sel = d_sel_Z;
                else if (rho_spins[src_id] >= 0)   sel = d_sel_rho_p;
                else                               sel = d_sel_rho_m;

                mode_n_contract(handle, cur, shape.data(), ndim,
                                sel, D, 1, ndim-1, nxt);

                int src_local = src.bond_local[src_dir];
                int L = 1; for (int d = 0; d < src_local; d++) L *= shape[d];
                int R_full = 1; for (int d = src_local+1; d < ndim; d++) R_full *= shape[d];
                int R_other = R_full / D;

                int msg_idx = (dir==0) ? bond.msg_from_a() : bond.msg_from_b();
                int blocks  = (chi*chi + 255) / 256;

                if (L == 1) {
                    RB_CHECK(rocblas_sgemv(handle, rocblas_operation_none,
                        chi, R_other, &one, nxt, chi, d_ones, 1, &zero, d_mu, 1));
                } else if (R_other == 1) {
                    RB_CHECK(rocblas_sgemv(handle, rocblas_operation_transpose,
                        L, chi, &one, nxt, L, d_ones, 1, &zero, d_mu, 1));
                } else {
                    RB_CHECK(rocblas_sgemv(handle, rocblas_operation_transpose,
                        L, chi*R_other, &one, nxt, L, d_ones, 1, &zero, cur, 1));
                    RB_CHECK(rocblas_sgemv(handle, rocblas_operation_none,
                        chi, R_other, &one, cur, chi, d_ones, 1, &zero, d_mu, 1));
                }
                RB_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
                RB_CHECK(rocblas_sdot(handle, chi, d_mu, 1, d_ones, 1, d_s));
                RB_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
                normalize_and_delta<<<blocks, 256, 0, rbstream>>>(
                    d_mu, messages[msg_idx], messages[msg_idx], d_s, d_delta, chi, alpha);
            }
        }

        HIP_CHECK(hipStreamSynchronize(rbstream));
        Scalar delta_h = 0.f;
        HIP_CHECK(hipMemcpy(&delta_h, d_delta, sizeof(Scalar), hipMemcpyDeviceToHost));
        return delta_h;
    }

    BpResult bp_converge_with_rho(const Grid& grid,
                                  const std::vector<Scalar*>& psi,
                                  std::vector<Scalar*>& messages,
                                  int center_site,
                                  const int* rho_spins,
                                  int n_sweeps = 100,
                                  float tol = 1e-5f)
    {
        float last_delta = 1e30f;
        for (int i = 0; i < n_sweeps; i++) {
            last_delta = bp_update_with_rho(grid, psi, messages, center_site, rho_spins);
            if (last_delta < tol) return {i+1, last_delta};
        }
        return {n_sweeps, last_delta};
    }

    BpResult bp_converge(const Grid& grid,
                         const std::vector<Scalar*>& psi,
                         std::vector<Scalar*>& messages,
                         int n_sweeps, int center_site,
                         float tol=1e-5f,
                         float alpha=1.0f,
                         std::vector<Scalar*>* messages_v=nullptr)
    {
        float last_delta = 1e30f;
        int i = 0;
        for(; i<n_sweeps; i++){
            last_delta = bp_update(grid,psi,messages,center_site,alpha);
            if(last_delta < tol) { i++; break; }
        }
        // Populate messages_v by extracting diagonals from converged messages.
        // Do NOT do an extra bp_update — that would shift the fixed point.
        if(messages_v) {
            for(auto& bond : grid.bonds) {
                if(bond.chi_alloc == 1) continue;
                int chi = bond.chi;
                for(int d=0; d<2; d++) {
                    int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                    if((*messages_v)[idx]) {
                        int blocks = (chi+255)/256;
                        extract_diagonal<<<blocks,256>>>(
                            messages[idx], (*messages_v)[idx], chi);
                    }
                }
            }
        }
        return {i, last_delta};
    }

    /* ── bp_update_norm ─────────────────────────────────────────────────
     * One sweep of norm BP: produces chi-vector messages for Tr(O†O).
     *
     * Key difference from bp_update: ψ is squared elementwise before
     * the env contraction, and ALL D physical components are summed
     * (no center_site, no d_sel_Z/I). This makes messages self-consistent
     * for the quadratic form Σ_{d,L,R} ψ[L,i,R,d]² × env rather than the
     * bilinear form inner(ψ, ψ₀) computed by bp_update.
     */
    float bp_update_norm(const Grid& grid,
                         const std::vector<Scalar*>& psi,
                         std::vector<Scalar*>& messages)
    {
        float one=1.f, zero=0.f;
        hipStream_t rbstream; rocblas_get_stream(handle, &rbstream);
        HIP_CHECK(hipMemsetAsync(d_delta, 0, sizeof(Scalar), rbstream));

        for (auto& bond : grid.bonds) {
            if (bond.chi_alloc == 1) continue;  // skip outside-lightcone bonds
            int chi = bond.chi;
            for (int dir = 0; dir < 2; dir++) {
                int src_id  = (dir==0) ? bond.site_a : bond.site_b;
                int src_dir = (dir==0) ? bond.dir_a  : bond.dir_b;
                const Site& src = grid.sites[src_id];

                std::vector<int> shape;
                for (int d = 0; d < MAX_DEGREE; d++)
                    if (src.bond_id[d] >= 0)
                        shape.push_back(grid.bonds[src.bond_id[d]].chi);
                shape.push_back(D);
                int ndim  = (int)shape.size();
                int total = 1; for (auto s : shape) total *= s;

                // Square ψ elementwise into cv1 — turns linear env sum into ψ²-weighted sum
                int sq_blocks = (total + 255) / 256;
                square_elements<<<sq_blocks, 256, 0, rbstream>>>(psi[src_id], ws.cv1, total);
                Scalar* cur = ws.cv1, *nxt = ws.cv2;

                // Contract all non-cut neighbor bonds with squared ψ
                for (int d = 0; d < MAX_DEGREE; d++) {
                    if (src.bond_id[d] < 0 || d == src_dir) continue;
                    int li = src.bond_local[d];
                    const Bond& eb = grid.bonds[src.bond_id[d]];
                    int bchi = eb.chi;
                    mode_n_contract(handle, cur, shape.data(), ndim,
                                    messages[eb.msg_incoming(src_id)],
                                    bchi, bchi, li, nxt);
                    std::swap(cur, nxt);
                }

                // Sum ALL D physical components using d_sel_full ([1,1,1,1] · ψ²_env)
                mode_n_contract(handle, cur, shape.data(), ndim,
                                d_sel_full, D, 1, ndim-1, nxt);

                int src_local = src.bond_local[src_dir];
                int L       = 1; for (int d = 0; d < src_local; d++) L *= shape[d];
                int R_full  = 1; for (int d = src_local+1; d < ndim; d++) R_full *= shape[d];
                int R_other = R_full / D;

                int msg_idx = (dir==0) ? bond.msg_from_a() : bond.msg_from_b();
                int blocks  = (chi*chi + 255) / 256;

                if (L == 1) {
                    RB_CHECK(rocblas_sgemv(handle, rocblas_operation_none,
                        chi, R_other, &one, nxt, chi, d_ones, 1, &zero, d_mu, 1));
                } else if (R_other == 1) {
                    RB_CHECK(rocblas_sgemv(handle, rocblas_operation_transpose,
                        L, chi, &one, nxt, L, d_ones, 1, &zero, d_mu, 1));
                } else {
                    RB_CHECK(rocblas_sgemv(handle, rocblas_operation_transpose,
                        L, chi*R_other, &one, nxt, L, d_ones, 1, &zero, cur, 1));
                    RB_CHECK(rocblas_sgemv(handle, rocblas_operation_none,
                        chi, R_other, &one, cur, chi, d_ones, 1, &zero, d_mu, 1));
                }

                RB_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
                RB_CHECK(rocblas_sdot(handle, chi, d_mu, 1, d_ones, 1, d_s));
                RB_CHECK(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
                normalize_and_delta<<<blocks, 256, 0, rbstream>>>(
                    d_mu, messages[msg_idx], messages[msg_idx], d_s, d_delta, chi, 1.0f);
            }
        }

        float max_change;
        HIP_CHECK(hipMemcpy(&max_change, d_delta, sizeof(float), hipMemcpyDeviceToHost));
        return max_change;
    }

    /* Runs norm BP to convergence and stores messages in norm_bp_msg. */
    BpResult bp_converge_norm(const Grid& grid,
                              const std::vector<Scalar*>& psi,
                              int n_sweeps, float tol=1e-5f)
    {
        // Allocate / reallocate norm_bp_msg when chi grows
        int nm = 2 * grid.n_bonds;
        if ((int)norm_bp_msg.size() != nm) {
            for (auto p : norm_bp_msg) if (p) HIP_CHECK(hipFree(p));
            norm_bp_msg.assign(nm, nullptr);
            norm_bp_msg_chi.assign(nm, 0);
        }
        for (auto& bond : grid.bonds) {
            for (int d = 0; d < 2; d++) {
                int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                if (norm_bp_msg_chi[idx] != bond.chi) {
                    if (norm_bp_msg[idx]) HIP_CHECK(hipFree(norm_bp_msg[idx]));
                    HIP_CHECK(hipMalloc(&norm_bp_msg[idx], (size_t)bond.chi*bond.chi*sizeof(Scalar)));
                    norm_bp_msg_chi[idx] = bond.chi;
                    // Initialize to identity (diagonal 1/chi) so first sweep is meaningful
                    std::vector<Scalar> init(bond.chi * bond.chi, 0.f);
                    for (int i = 0; i < bond.chi; i++) init[i + i*bond.chi] = 1.f / bond.chi;
                    HIP_CHECK(hipMemcpy(norm_bp_msg[idx], init.data(),
                        bond.chi*bond.chi*sizeof(Scalar), hipMemcpyHostToDevice));
                }
            }
        }

        float last_delta = 1e30f;
        for (int i = 0; i < n_sweeps; i++) {
            last_delta = bp_update_norm(grid, psi, norm_bp_msg);
            if (last_delta < tol) return {i+1, last_delta};
        }
        return {n_sweeps, last_delta};
    }

    Scalar norm_sqr(
        const Grid& grid,
        const std::vector<Scalar*>& psi,
        const std::vector<Scalar*>& messages)
    {
        return bp_partitionfunction(handle, grid, psi, messages,
            Operators::I(), -1, ws.cv1, ws.cv2);
    }

    /* ── expect_z: C(t) = Tr(Z_v · O(t)) / Tr(I) ───────────────────
     * Numerator: BP partition function with Z at site_id.
     * Denominator: D^n_sites (constant Hilbert space dimension).
     * At t=0 with O(0)=Z at center: Tr(Z·Z)=Tr(I)=D^n → C=1. */
    Scalar expect_z(
        const Grid& grid,
        const std::vector<Scalar*>& psi,
        const std::vector<Scalar*>& messages,
        int site_id)
    {
        Scalar num = bp_partitionfunction(handle, grid, psi, messages,
            Operators::Z(), site_id, ws.cv1, ws.cv2);
        Scalar den = norm_sqr(grid, psi, messages);
        return (std::abs(den) > 0.f) ? num/den : 0.f;
    }

    Scalar autocorrelation(
        const Grid& grid,
        const std::vector<Scalar*>& psi,
        const std::vector<Scalar*>& messages,
        int center_site)
    {
        return bp_partitionfunction(handle, grid, psi, messages,
            Operators::Z(), center_site, ws.cv1, ws.cv2);
    }

    /* ── autocorrelation_with_bp ─────────────────────────────────────
     * Runs BP on a COPY of messages (state.messages unchanged), then
     * measures C(t) = bp_partitionfunction(Z, center) in log space.
     * bp_converge stops early when max message change < tol (default 1e-5).
     *
     * Optional timing outputs (milliseconds):
     *   t_bp_ms  — time spent in bp_converge (belief propagation sweeps)
     *   t_pf_ms  — time spent in bp_partitionfunction (Bethe contraction)
     */
    Scalar autocorrelation_with_bp(
        const Grid& grid,
        const std::vector<Scalar*>& psi,
        const std::vector<Scalar*>& messages,
        int center_site, int n_sweeps=100,
        double* t_bp_ms=nullptr, double* t_pf_ms=nullptr,
        int* n_sweeps_out=nullptr, float* delta_out=nullptr,
        float* spatial_out=nullptr,
        const std::vector<Scalar*>* messages_v_in=nullptr)  // chi-vector gate messages
    {
        using clk = std::chrono::high_resolution_clock;
        using ms  = std::chrono::duration<double, std::milli>;

        // Allocate temporary BP messages — warm start from gate messages.
        int nm = 2 * grid.n_bonds;
        std::vector<Scalar*> bp_msg(nm, nullptr);
        std::vector<Scalar*> bp_msg_v(nm, nullptr);  // chi-vector companion
        for (auto& bond : grid.bonds) {
            for (int d=0; d<2; d++) {
                int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                int chi = bond.chi;
                HIP_CHECK(hipMalloc(&bp_msg[idx], (size_t)chi*chi*sizeof(Scalar)));
                HIP_CHECK(hipMemcpy(bp_msg[idx], messages[idx],
                    (size_t)chi*chi*sizeof(Scalar), hipMemcpyDeviceToDevice));
                // Chi-vector: warm start from gate messages_v if available
                HIP_CHECK(hipMalloc(&bp_msg_v[idx], (size_t)chi*sizeof(Scalar)));
                if (messages_v_in && (*messages_v_in)[idx])
                    HIP_CHECK(hipMemcpy(bp_msg_v[idx], (*messages_v_in)[idx],
                        (size_t)chi*sizeof(Scalar), hipMemcpyDeviceToDevice));
                else
                    HIP_CHECK(hipMemset(bp_msg_v[idx], 0, (size_t)chi*sizeof(Scalar)));
            }
        }

        float alpha = (damped_bp) ? bp_alpha : 1.0f;
        auto t0 = clk::now();
        BpResult bpr = bp_converge(grid, psi, bp_msg, n_sweeps, center_site, 1e-5f, alpha,
                                   &bp_msg_v);
        auto t1 = clk::now();
        if (t_bp_ms)    *t_bp_ms    = ms(t1 - t0).count();
        if (n_sweeps_out) *n_sweeps_out = bpr.sweeps;
        if (delta_out)    *delta_out    = bpr.delta;

        Scalar C = bp_partitionfunction(handle, grid, psi, bp_msg,
            Operators::Z(), center_site, ws.cv1, ws.cv2, full_bp,
            dbl_a, dbl_b, dbl_msg);
        auto t2 = clk::now();
        if (t_pf_ms) *t_pf_ms = ms(t2 - t1).count();

        // Optional spatial C(r,t): reuse converged messages, one call per site
        if (spatial_out) {
            for (int s = 0; s < (int)grid.n_sites; s++)
                spatial_out[s] = bp_partitionfunction(handle, grid, psi, bp_msg,
                    Operators::Z(), s, ws.cv1, ws.cv2, full_bp,
                    dbl_a, dbl_b, dbl_msg);
        }

        for (auto p : bp_msg)   if (p) HIP_CHECK(hipFree(p));
        for (auto p : bp_msg_v) if (p) HIP_CHECK(hipFree(p));
        return C;
    }

    /* ── inner_op_state ──────────────────────────────────────────────
     * Computes C(t) = Tr(O(t) · ρ) = inner(ψ(t), ρ) via BP.
     *
     * Mirrors Julia's inner(ψ, ρ; alg="bp"):
     *   1. BilinearForm(ψ, ρ) — fresh BPC with identity messages
     *   2. update(bpc) — BP sweeps on ψ⊗ρ bilinear form
     *   3. partition function
     *
     * Since ρ is a product state (chi=1), the bilinear form bond dim
     * equals ψ's bond dim. BP uses per-site ρ selectors:
     *   rho_spins[v]=+1 → sel=[1,0,0,+1], -1 → sel=[1,0,0,-1]
     *   center_site   → sel=[0,0,0,1] (Z observable)
     *
     * Fresh identity messages (not reusing ψ's BPC messages).
     * Sign tracked in log-space — no sqrt(-1) issue.
     */
    Scalar inner_op_state(
        const Grid& grid,
        const std::vector<Scalar*>& psi,
        const std::vector<Scalar*>& messages,
        const int* rho_spins,
        int center_site,
        int n_sweeps = 100)
    {
        // Step 1: Allocate fresh identity messages for ⟨ψ|ρ⟩ bilinear form
        int nm = 2 * grid.n_bonds;
        std::vector<Scalar*> bp_msg(nm, nullptr);
        for (auto& bond : grid.bonds) {
            for (int d = 0; d < 2; d++) {
                int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                int chi = bond.chi;
                HIP_CHECK(hipMalloc(&bp_msg[idx], (size_t)chi*chi*sizeof(Scalar)));
                // Julia uses delta(linds) = plain chi×chi identity (no 1/chi)
                std::vector<Scalar> eye(chi*chi, 0.f);
                for (int i = 0; i < chi; i++) eye[i*chi+i] = 1.f;
                HIP_CHECK(hipMemcpy(bp_msg[idx], eye.data(),
                    (size_t)chi*chi*sizeof(Scalar), hipMemcpyHostToDevice));
            }
        }

        // Step 2: Run BP on ψ using per-site ρ selectors
        // bp_converge_with_rho selects [1,0,0,±1] per site instead of [1,0,0,0]
        bp_converge_with_rho(grid, psi, bp_msg, center_site, rho_spins, n_sweeps);

        // Step 3: Compute Bethe partition function with ρ selectors
        // Use bp_partitionfunction with per-site ρ operators
        // Upload per-site operators to GPU once via vertex_scalar loop
        // (reuse bp_partitionfunction with rho ops)

        // Build per-site D×D operator matrices: diag([1,0,0,±1]) or diag([0,0,0,1])
        // These are the same as d_sel_rho_p/m/Z but as D×D diagonal matrices
        std::vector<Scalar> op_rho_p(D*D, 0.f); op_rho_p[0]   = 1.f; op_rho_p[3+D*3] = 1.f;
        std::vector<Scalar> op_rho_m(D*D, 0.f); op_rho_m[0]   = 1.f; op_rho_m[3+D*3] = -1.f;
        std::vector<Scalar> op_Z(D*D, 0.f);     op_Z[3+D*3]   = 1.f;

        float *d_op_rho_p, *d_op_rho_m, *d_op_Z;
        HIP_CHECK(hipMalloc(&d_op_rho_p, D*D*sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_op_rho_m, D*D*sizeof(Scalar)));
        HIP_CHECK(hipMalloc(&d_op_Z,     D*D*sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_op_rho_p, op_rho_p.data(), D*D*sizeof(Scalar), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_op_rho_m, op_rho_m.data(), D*D*sizeof(Scalar), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_op_Z,     op_Z.data(),     D*D*sizeof(Scalar), hipMemcpyHostToDevice));

        // Bethe partition function: Π_v a_v / Π_e ze_e
        double log_num  = 0.0;
        double sign_num = 1.0;

        for (auto& site : grid.sites) {
            const float* op = (site.id == center_site)       ? d_op_Z :
                              (rho_spins[site.id] >= 0)      ? d_op_rho_p : d_op_rho_m;
            Scalar vs = vertex_scalar(handle, site, psi[site.id], grid, bp_msg,
                                      op, ws.cv1, ws.cv2, false);
            if (!std::isfinite(vs)) { sign_num = 0.0; break; }
            if (vs == 0.f)          { sign_num = 0.0; break; }
            sign_num *= (vs > 0.f ? 1.0 : -1.0);
            log_num  += std::log(std::abs((double)vs));
        }

        // Bethe edge correction
        double log_den  = 0.0;
        double sign_den = 1.0;
        for (auto& bond : grid.bonds) {
            int chi = bond.chi, chi2 = chi*chi;
            std::vector<Scalar> ma(chi2), mb(chi2);
            HIP_CHECK(hipMemcpy(ma.data(), bp_msg[bond.msg_from_a()],
                chi2*sizeof(Scalar), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(mb.data(), bp_msg[bond.msg_from_b()],
                chi2*sizeof(Scalar), hipMemcpyDeviceToHost));
            double ze = 0.0;
            for (int k = 0; k < chi2; k++) ze += (double)ma[k] * mb[k];
            if (std::abs(ze) > 1e-30) {
                sign_den *= (ze > 0 ? 1.0 : -1.0);
                log_den  += std::log(std::abs(ze));
            }
        }

        // Cleanup
        for (auto p : bp_msg) if (p) HIP_CHECK(hipFree(p));
        HIP_CHECK(hipFree(d_op_rho_p));
        HIP_CHECK(hipFree(d_op_rho_m));
        HIP_CHECK(hipFree(d_op_Z));

        if (sign_num == 0.0) return 0.f;
        return (Scalar)(sign_num * sign_den * std::exp(log_num - log_den));
    }

    /* ── rescale ─────────────────────────────────────────────────────
     * Normalises the tensor network so the Frobenius "partition function"
     * equals 1, matching Julia's rescale(ψ_bpc) called after each layer.
     *
     * Z_full = bp_partitionfunction with the full D×D identity (sums ALL
     * Pauli components), so it measures Σ_d (ψ_v[d])² · messages — the
     * true Frobenius norm of the operator network.
     *
     * Each site tensor is divided by Z_full^(1/n_sites) so that after
     * rescaling Z_full = 1.  Without this, norms grow to ~18 by layer 2
     * and vertex_scalars go to zero, making C(t)=0 for all t>1. */
    void rescale(const Grid& grid,
                 std::vector<Scalar*>& psi,
                 std::vector<Scalar*>& messages,
                 Workspace& ws_)
    {
        // Match Julia's rescale_vertices!: T_v → T_v * sign(vs_v) / sqrt(|vs_v|)
        // where vs_v = vertex_scalar with full D×D identity (summing all Paulis).
        // This normalises each site independently so vs_v → sign(vs_v) after rescale.
        // Julia also rescales messages (rescale_messages!) but we skip that for now
        // since our messages come from SVD and are already normalised.
        std::vector<Scalar> Id_full(D*D, 0.f);
        for (int d=0; d<D; d++) Id_full[d + D*d] = 1.f;
        float *d_Id;
        HIP_CHECK(hipMalloc(&d_Id, D*D*sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_Id, Id_full.data(), D*D*sizeof(Scalar), hipMemcpyHostToDevice));

        for (auto& site : grid.sites) {
            Scalar vs = vertex_scalar(handle, site, psi[site.id],
                                      grid, messages, d_Id, ws_.cv1, ws_.cv2);
            float absv = std::abs(vs);
            if (absv < 1e-12f) continue;
            // T_v → T_v * sign(vs) / sqrt(|vs|)  (Julia: s * inv(sqrt(vn)))
            float factor = (vs > 0.f ? 1.f : -1.f) / std::sqrt(absv);

            int total = D;
            for (int d=0; d<MAX_DEGREE; d++)
                if (site.bond_id[d] >= 0)
                    total *= grid.bonds[site.bond_id[d]].chi;

            std::vector<Scalar> h(total);
            HIP_CHECK(hipMemcpy(h.data(), psi[site.id],
                total*sizeof(Scalar), hipMemcpyDeviceToHost));
            for (auto& v : h) v *= factor;
            HIP_CHECK(hipMemcpy(psi[site.id], h.data(),
                total*sizeof(Scalar), hipMemcpyHostToDevice));
        }

        HIP_CHECK(hipFree(d_Id));
    }

    /* ── print_all ───────────────────────────────────────────────────
     * Prints Z expectation at every site — useful for debugging.
     * Shows the spatial spreading of the operator across the grid. */
    void print_all(
        const Grid& grid,
        const std::vector<Scalar*>& psi,
        const std::vector<Scalar*>& messages,
        int layer)
    {
        Scalar den = norm_sqr(grid, psi, messages);  // D^n_sites
        printf("Layer %d  Tr(I)=%.0f\n", layer, den);
        for (int r=0; r<grid.nrows; r++) {
            printf("  row %d: ", r);
            for (int c=0; c<grid.ncols; c++) {
                int sid = grid.site_id(r,c);
                Scalar num = bp_partitionfunction(handle, grid, psi, messages,
                    Operators::Z(), sid, ws.cv1, ws.cv2);
                printf("Z(%d,%d)=%+.4f  ", r, c, num/den);
            }
            printf("\n");
        }
    }

    /* ── compute_pauli_profile ───────────────────────────────────────────
     * Non-identity Pauli weight at each site, normalized by total weight.
     *
     *   n(v) = [w_X(v) + w_Y(v) + w_Z(v)] / Σ_{v,d} w_d(v)
     *
     * where  w_d(v) = Σ_{L,R,...} ψ_v[L,d,R,...]² × env(norm_bp_msg).
     *
     * Shows the operator footprint spreading from center outward.
     * At t=0: n(center)=1, n(elsewhere)=0.
     * As t grows: spreading disc tracing the operator lightcone.
     *
     * Calls bp_converge_norm internally — free if compute_norm already ran
     * this layer (messages are at fixed point, exits in 1 sweep).
     */
    void compute_pauli_profile(
        const Grid& grid,
        const std::vector<Scalar*>& psi,
        float* out_nid)   // [n_sites] normalized non-identity weight
    {
        bp_converge_norm(grid, psi, 100);

        // Upload the 4 Pauli selector matrices (D×D diagonal, picks one component)
        const std::vector<Scalar> op_data[D] = {
            Operators::I(), Operators::X(), Operators::Y(), Operators::Z()
        };
        float* d_ops[D];
        for (int d = 0; d < D; d++) {
            HIP_CHECK(hipMalloc(&d_ops[d], D*D*sizeof(Scalar)));
            HIP_CHECK(hipMemcpy(d_ops[d], op_data[d].data(),
                D*D*sizeof(Scalar), hipMemcpyHostToDevice));
        }

        hipStream_t rbstream; rocblas_get_stream(handle, &rbstream);

        std::vector<float> w((size_t)grid.n_sites * D, 0.f);
        double total_w = 0.0;

        for (auto& site : grid.sites) {
            int total = D;
            for (int bd = 0; bd < MAX_DEGREE; bd++)
                if (site.bond_id[bd] >= 0) total *= grid.bonds[site.bond_id[bd]].chi;

            int sq_blocks = (total + 255) / 256;
            square_elements<<<sq_blocks, 256, 0, rbstream>>>(
                psi[site.id], d_psi_sq, total);
            HIP_CHECK(hipStreamSynchronize(rbstream));

            for (int d = 0; d < D; d++) {
                Scalar vs = vertex_scalar(handle, site, d_psi_sq,
                                          grid, norm_bp_msg, d_ops[d],
                                          ws.cv1, ws.cv2);
                float wv = std::abs((float)vs);
                w[(size_t)site.id * D + d] = wv;
                total_w += wv;
            }
        }

        float inv = (total_w > 1e-30) ? 1.f / (float)total_w : 0.f;
        for (int s = 0; s < (int)grid.n_sites; s++) {
            float nid = 0.f;
            for (int d = 1; d < D; d++) nid += w[(size_t)s * D + d];  // X + Y + Z
            out_nid[s] = nid * inv;
        }

        for (int d = 0; d < D; d++) HIP_CHECK(hipFree(d_ops[d]));
    }

    /* ── compute_norm ────────────────────────────────────────────────────
     * Bethe approximation of Tr(O†O) = ||ψ||²_Frobenius in operator space.
     *
     * Formula:  ||ψ||²_Bethe = (∏_v vs_v[Id_full]) / (∏_e ze_e)
     *
     * vs_v[Id_full] = vertex_scalar with the full D×D identity at site v.
     *   This sums all D Pauli components squared: Σ_d ψ_v[L,d,R]² × env.
     *   (Operators::Z()/I() are selectors that pick ONE component; here we
     *   want ALL D components — same operator used by rescale().)
     *
     * ze_e = Frobenius inner product of both directional messages:
     *   Σ_{ij} M_{a→b}[i,j] × M_{b→a}[i,j]  (= Σ_k ma[k]*mb[k] flat)
     *   This is the same edge correction used by bp_partitionfunction.
     *
     * Accumulate numerator/denominator in log-space to avoid overflow at
     * large chi (product of 16 floats for a 4×4 grid overflows easily).
     * After rescale() the tensors are O(1), so the ratio stays near 1.
     */
    /* ── compute_norm ────────────────────────────────────────────────────
     * Bethe approximation of Tr(O†O)/D^N — should be 1.0 under exact
     * (zero-truncation) evolution, drifting below 1 as gate_err accumulates.
     *
     * Uses a SEPARATE norm BP (bp_converge_norm) whose messages are
     * self-consistent for the quadratic form Σ_{d,L,R} ψ[L,i,R,d]²·env,
     * distinct from the C(t) bilinear-form messages. Running with C(t)
     * messages gives C(t) back, not the norm — hence the separate BP.
     *
     * Vertex formula (squared ψ):
     *   f_v = Σ_{d,L,R,...} ψ_v[L,d,R,...]² × Π_b μ_b[bond_index]
     * Edge correction (same as C(t)):
     *   ze_e = Σ_i μ_{a→b}[i] × μ_{b→a}[i]
     * norm = (Π_v f_v) / (Π_e ze_e)
     */
    float compute_norm(
        const Grid& grid,
        const std::vector<Scalar*>& psi,
        const std::vector<Scalar*>& /*messages*/)  // ignored — runs its own BP
    {
        // Run norm BP to convergence (separate from C(t) BP)
        bp_converge_norm(grid, psi, 100);

        const auto& msg = norm_bp_msg;

        // Upload Id_full operator once (sums all D components linearly on ψ²)
        std::vector<Scalar> Id_full(D*D, 0.f);
        for (int d = 0; d < D; d++) Id_full[d + D*d] = 1.f;
        float* d_Id;
        HIP_CHECK(hipMalloc(&d_Id, D*D*sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(d_Id, Id_full.data(), D*D*sizeof(Scalar), hipMemcpyHostToDevice));

        // d_psi_sq is a permanent member (pre-allocated chi_max^4*D in constructor).

        hipStream_t rbstream; rocblas_get_stream(handle, &rbstream);

        // Numerator: Π_v f_v  where  f_v = Σ_{d,L,R,...} ψ_v[L,d,R,...]² × env(norm_msg)
        double log_num  = 0.0;
        double sign_num = 1.0;
        for (auto& site : grid.sites) {
            std::vector<int> shape;
            for (int d = 0; d < MAX_DEGREE; d++)
                if (site.bond_id[d] >= 0)
                    shape.push_back(grid.bonds[site.bond_id[d]].chi);
            shape.push_back(D);
            int total = 1; for (auto s : shape) total *= s;

            int sq_blocks = (total + 255) / 256;
            square_elements<<<sq_blocks, 256, 0, rbstream>>>(psi[site.id], d_psi_sq, total);
            HIP_CHECK(hipStreamSynchronize(rbstream));

            Scalar vs = vertex_scalar(handle, site, d_psi_sq,
                                      grid, msg, d_Id,
                                      ws.cv1, ws.cv2);
            if (!std::isfinite(vs)) {
                HIP_CHECK(hipFree(d_Id)); HIP_CHECK(hipFree(d_psi_sq));
                return std::numeric_limits<float>::quiet_NaN();
            }
            if (vs == 0.f) { sign_num = 0.0; break; }
            sign_num *= (vs > 0.f ? 1.0 : -1.0);
            log_num  += std::log(std::abs((double)vs));
        }

        HIP_CHECK(hipFree(d_Id));

        if (sign_num == 0.0) return 0.f;

        // Denominator: Π_e ze_e = Σ_i μ_{a→b}[i] × μ_{b→a}[i]
        double log_den  = 0.0;
        double sign_den = 1.0;
        for (auto& bond : grid.bonds) {
            int chi = bond.chi;
            // norm_bp_msg are chi-vectors stored as chi×chi with zeros off-diagonal
            // — read only the diagonal elements (stride chi+1)
            std::vector<Scalar> ma_full(chi*chi), mb_full(chi*chi);
            HIP_CHECK(hipMemcpy(ma_full.data(), msg[bond.msg_from_a()],
                chi*chi*sizeof(Scalar), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(mb_full.data(), msg[bond.msg_from_b()],
                chi*chi*sizeof(Scalar), hipMemcpyDeviceToHost));
            double ze = 0.0;
            for (int i = 0; i < chi; i++)
                ze += (double)ma_full[i*(chi+1)] * (double)mb_full[i*(chi+1)];
            if (std::abs(ze) > 1e-30) {
                sign_den *= (ze > 0 ? 1.0 : -1.0);
                log_den  += std::log(std::abs(ze));
            }
        }

        return (float)(sign_num * sign_den * std::exp(log_num - log_den));
    }

    /* ── compute_osee ────────────────────────────────────────────────────
     * Operator Space Entanglement Entropy across a horizontal bipartition.
     *
     * Uses Vidal-gauge singular values already in state.messages (set by
     * Trotter SVD — no extra GPU work). For each bond crossing the cut
     * between rows (cut_row) and (cut_row+1), reads the diagonal of the
     * message matrix, normalises to get Schmidt probabilities p_i = λ²_i/Σλ²,
     * and accumulates S = -Σ p_i log p_i. Sums over all cut bonds.
     *
     * cut_row = -1 (default): horizontal cut at nrows/2 - 1 (center).
     * Also returns n_cut_bonds and log(chi_max) so the caller can normalise:
     *   OSEE / (n_cut_bonds * log(chi_max))  →  0..1 saturation metric.
     */
    float compute_osee(
        const Grid& grid,
        const std::vector<Scalar*>& messages,
        int cut_row = -1,
        int* n_cut_bonds_out = nullptr,
        float* osee_max_out  = nullptr)
    {
        if (cut_row < 0) cut_row = grid.nrows / 2 - 1;

        float osee      = 0.f;
        int   n_bonds   = 0;
        int   chi_max_cut = 1;

        for (auto& bond : grid.bonds) {
            int ra = bond.site_a / grid.ncols;
            int rb = bond.site_b / grid.ncols;
            bool crosses = (ra == cut_row && rb == cut_row + 1) ||
                           (rb == cut_row && ra == cut_row + 1);
            if (!crosses) continue;

            int chi = bond.chi;
            if (chi == 0) continue;
            chi_max_cut = std::max(chi_max_cut, chi);
            n_bonds++;

            // D2H the full chi×chi message — diagonal holds Vidal singular values
            int msg_idx = bond.msg_from_a();
            std::vector<Scalar> h_msg((size_t)chi * chi);
            HIP_CHECK(hipMemcpy(h_msg.data(), messages[msg_idx],
                (size_t)chi * chi * sizeof(Scalar), hipMemcpyDeviceToHost));

            // Squared singular values: diagonal element i is at h_msg[i*(chi+1)]
            float norm2 = 0.f;
            for (int i = 0; i < chi; i++) {
                float v = h_msg[(size_t)i * (chi + 1)];
                norm2 += v * v;
            }
            if (norm2 < 1e-12f) continue;

            float inv_norm2 = 1.f / norm2;
            for (int i = 0; i < chi; i++) {
                float v = h_msg[(size_t)i * (chi + 1)];
                float p = v * v * inv_norm2;
                if (p > 1e-15f) osee -= p * logf(p);
            }
        }

        if (n_cut_bonds_out) *n_cut_bonds_out = n_bonds;
        if (osee_max_out)    *osee_max_out    = n_bonds * logf((float)chi_max_cut);
        return osee;
    }
};
