#pragma once
/*
 * gate.hpp — CppSim gate computation (D=4 Pauli basis)
 *
 * Matches Julia's simple_update exactly.
 *
 * KEY INSIGHT — TENSOR LAYOUT AND QR:
 *
 *   Canonical tensor shape: [chi_UP, chi_DOWN, chi_LEFT, chi_RIGHT, D]
 *   (missing bonds skipped, D always last)
 *
 *   QR requires layout [env_dims..., chi_cut, D] — cut bond second-to-last.
 *   When cut_dir has bond_local != degree-1, we permute before QR and
 *   inverse-permute after reconstruction (before deabsorb_env).
 *
 *   All permutations are host-side element shuffles on small tensors.
 *
 * ENV POLICY (FROZEN):
 *   frozen_chi and frozen_msg are captured once at layer start.
 *   They never change during the Rxx layer — correct physics.
 *   Actual bond.chi grows between color classes; absorb_env reads
 *   the actual tensor size but skips env contraction for bonds that
 *   grew (chi_tens != chi_msg).
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "permute_gpu.hpp"
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <cmath>
#include <cfloat>
#include <vector>
#include <cstdio>
#include <cassert>
#include <algorithm>

/* ── symmetrize_copy ─────────────────────────────────────────────────────────
 * Copies src[n×n] → dst[n×n] with dst[i,j] = 0.5*(src[i,j]+src[j,i]).
 * Replaces: D2H M (chi²) + CPU symmetrize loop + H2D sqE. */
__global__ void symmetrize_copy(const Scalar* __restrict__ src, Scalar* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n*n) return;
    int i = idx % n, j = idx / n;
    dst[idx] = 0.5f * (src[i + (size_t)n*j] + src[j + (size_t)n*i]);
}

/* ── form_sqrt_matrices ──────────────────────────────────────────────────────
 * Given eigenvectors V[chi×chi] (col-major) and eigenvalues W[chi], computes:
 *   sqM[i,j]  = Σ_k V[i,k] * V[j,k] * sqrt(|W[k]|)   (if |W[k]| >= cut)
 *   isqM[i,j] = Σ_k V[i,k] * V[j,k] / sqrt(|W[k]|)
 * Replaces: D2H hW + D2H hV + CPU scale + hipMalloc×3 + H2D×3 + 2×GEMM + hipFree×3. */
__global__ void form_sqrt_matrices(const Scalar* __restrict__ V,
                                    const Scalar* __restrict__ W,
                                    Scalar* sqM, Scalar* isqM,
                                    int chi, Scalar cut) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chi*chi) return;
    int i = idx % chi, j = idx / chi;
    Scalar s = 0.f, si = 0.f;
    for (int k = 0; k < chi; k++) {
        Scalar w = W[k];
        Scalar sq = 0.f, sqi = 0.f;
        if (fabsf(w) >= cut) { sq = sqrtf(fabsf(w)); sqi = 1.f / sq; }
        Scalar vij = V[i + chi*k] * V[j + chi*k];
        s  += vij * sq;
        si += vij * sqi;
    }
    sqM[idx]  = s;
    isqM[idx] = si;
}

/* ── cast_f32_to_f64 / cast_f64_to_f32 ──────────────────────────────────────
 * Element-wise float32↔float64 conversion for mixed-precision SVD. */
__global__ void cast_f32_to_f64(const float* __restrict__ src, double* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (double)src[i];
}
__global__ void cast_f64_to_f32(const double* __restrict__ src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (float)src[i];
}

/* ── extract_diagonal ────────────────────────────────────────────────────────
 * Extracts diagonal of chi×chi matrix M into chi-vector d: d[i] = M[i + chi*i]. */
__global__ void extract_diagonal(const Scalar* __restrict__ M, Scalar* d, int chi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < chi) d[i] = M[i + chi * i];
}

/* ── find_max_diag ───────────────────────────────────────────────────────────
 * Finds the maximum value in a chi-vector d, writes result to out[0].
 * Single-block reduction — chi is small (≤ chi_max). */
__global__ void find_max_diag(const Scalar* __restrict__ d, Scalar* out, int chi) {
    __shared__ Scalar smax[256];
    int tid = threadIdx.x;
    Scalar v = (tid < chi) ? d[tid] : 0.f;
    // Reduce within block
    for (int i = blockDim.x/2; i > 0; i >>= 1) {
        if (tid < i) v = (v > __shfl_down(v, i)) ? v : __shfl_down(v, i);
        // Use shared memory for small block sizes
    }
    smax[tid] = v;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) smax[tid] = (smax[tid] > smax[tid+s]) ? smax[tid] : smax[tid+s];
        __syncthreads();
    }
    if (tid == 0) out[0] = smax[0];
}

/* ── diag_sqrt_matrices ──────────────────────────────────────────────────────
 * Fast path for diagonal messages: sqM = diag(sqrt(d_k)), isqM = diag(1/sqrt(d_k)).
 * Bypasses ssyevd entirely — O(chi) instead of O(chi³).
 * Safe because BP diagonal path (normalize_and_delta) always produces diagonal
 * messages: off-diagonal entries are explicitly zeroed every sweep. */
__global__ void diag_sqrt_matrices(const Scalar* __restrict__ diag, // chi diagonal entries
                                    Scalar* sqM, Scalar* isqM,
                                    int chi, Scalar cut) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chi * chi) return;
    int row = idx % chi;
    int col = idx / chi;
    if (row == col) {
        Scalar w  = diag[row];
        Scalar sq = 0.f, sqi = 0.f;
        if (fabsf(w) >= cut) { sq = sqrtf(fabsf(w)); sqi = 1.f / sq; }
        sqM[idx]  = sq;
        isqM[idx] = sqi;
    } else {
        sqM[idx]  = 0.f;
        isqM[idx] = 0.f;
    }
}

/* ── extract_upper_triangle ─────────────────────────────────────────────────
 * Extracts the k×n upper triangle from Q[m×n] (col-major, stride m) into
 * R[k×n] (col-major, stride k), zeroing the lower triangle.
 * Replaces: D2H Q (up to 655 MB) → CPU loop → H2D R.
 * Must run on the same stream as the surrounding rocsolver calls. */
__global__ void extract_upper_triangle(const Scalar* Q, Scalar* R,
                                        int m, int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * n) return;
    int i = idx % k;
    int j = idx / k;
    R[idx] = (i <= j) ? Q[i + (size_t)m * j] : 0.f;
}

/* ── add_diagonal ────────────────────────────────────────────────────────────
 * Adds val to each diagonal element of n×n col-major matrix A.
 * Replaces: D2H gram → CPU loop → H2D gram (2×n²×float per gate). */
__global__ void add_diagonal(Scalar* A, int n, Scalar val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    A[i + (size_t)n * i] += val;
}

/* ── fix_qr_sign ─────────────────────────────────────────────────────────────
 * Enforce positive diagonal of R after Householder QR.
 * If R[i,i] < 0: flip sign of Q[:,i] (col i, stride m) and R[i,:] (row i).
 * This makes the QR factorization unique (positive-diagonal convention),
 * ensuring independent runs on nearly-identical states produce consistent
 * gauge choices — required for bilinear_form_bp cross inner products. */
__global__ void fix_qr_sign_Q(Scalar* Q, const Scalar* R, int m, int k) {
    // One thread per (row, col) in Q[m×k]
    int col = blockIdx.x;   // k columns
    int row = blockIdx.y * blockDim.x + threadIdx.x;  // m rows
    if (col >= k || row >= m) return;
    // R is k×n col-major; R[i,i] = R[i + k*i]
    if (R[col + (size_t)k * col] < 0.f)
        Q[row + (size_t)m * col] = -Q[row + (size_t)m * col];
}
__global__ void fix_qr_sign_R(Scalar* R, int k, int n) {
    // One thread per (row, col) in R[k×n] — only upper triangle matters
    int col = blockIdx.x;   // n columns
    int row = blockIdx.y * blockDim.x + threadIdx.x;  // k rows
    if (col >= n || row >= k || row > col) return;
    if (R[row + (size_t)k * row] < 0.f)
        R[row + (size_t)k * col] = -R[row + (size_t)k * col];
}

/* ── build_M_from_U ──────────────────────────────────────────────────────────
 * Scales U columns by sqrt(S[c]*inv_snorm) and reorders into M for GEMM.
 * M[r + k*(c + new_chi*d)] = U[(r*D_+d) + k_D*c] * sqrt(S[c]*inv_snorm)
 * Replaces: D2H U (up to 6.5 MB) + CPU scale+reorder + H2D M1. */
__global__ void build_M_from_U(const Scalar* __restrict__ U,
                                const Scalar* __restrict__ S,
                                Scalar* M, int k, int k_D,
                                int new_chi, int D_, Scalar inv_snorm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * new_chi * D_) return;
    int r   = idx % k;
    int col = idx / k;      // col = c + new_chi*d
    int c   = col % new_chi;
    int d   = col / new_chi;
    Scalar sqS = sqrtf(fmaxf(S[c] * inv_snorm, 0.f));
    M[idx] = U[(r * D_ + d) + (size_t)k_D * c] * sqS;
}

/* ── build_M_from_Vt ─────────────────────────────────────────────────────────
 * Scales Vt rows by sqrt(S[c]*inv_snorm), transposes, and reorders into M.
 * Vt is [kfull × k2D] col-major: Vt[c + kfull*(r*D_+d)]
 * M[r + k2*(c + new_chi*d)] = Vt[c + kfull*(r*D_+d)] * sqrt(S[c]*inv_snorm)
 * Replaces: D2H Vt (up to 6.5 MB) + CPU scale+reorder + H2D M2. */
__global__ void build_M_from_Vt(const Scalar* __restrict__ Vt,
                                 const Scalar* __restrict__ S,
                                 Scalar* M, int k2, int kfull,
                                 int new_chi, int D_, Scalar inv_snorm) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k2 * new_chi * D_) return;
    int r   = idx % k2;
    int col = idx / k2;     // col = c + new_chi*d
    int c   = col % new_chi;
    int d   = col / new_chi;
    Scalar sqS = sqrtf(fmaxf(S[c] * inv_snorm, 0.f));
    M[idx] = Vt[c + (size_t)kfull * (r * D_ + d)] * sqS;
}

/* ── apply_gate_to_RR4 ───────────────────────────────────────────────────────
 * Contracts the D²×D² gate G with RR4[k1D×k2D] → oR4[k1D×k2D] (col-major).
 * oR4[r1*D+d1p, r2*D+d2p] = Σ_{d1,d2} G[d1p*D+d2p, d1*D+d2] * RR4[r1*D+d1, r2*D+d2]
 * G is cached in shared memory (D⁴ = 256 floats = 1 KB). */
__global__ void apply_gate_to_RR4(const Scalar* __restrict__ G,
                                   const Scalar* __restrict__ RR4,
                                   Scalar* oR4, int k1, int k2, int D_) {
    extern __shared__ Scalar sG[];
    int D2 = D_ * D_;
    for (int i = threadIdx.x; i < D2*D2; i += blockDim.x) sG[i] = G[i];
    __syncthreads();

    int k1D = k1 * D_;
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k1D * k2 * D_) return;
    int row = idx % k1D, col = idx / k1D;
    int r1 = row / D_, d1p = row % D_;
    int r2 = col / D_, d2p = col % D_;
    int g_row = d1p * D_ + d2p;
    Scalar s = 0.f;
    for (int d1 = 0; d1 < D_; d1++)
        for (int d2 = 0; d2 < D_; d2++)
            s += sG[g_row + D2*(d1*D_+d2)]
               * RR4[(r1*D_+d1) + (size_t)k1D*(r2*D_+d2)];
    oR4[idx] = s;
}

/* ── reshape_R_to_A ──────────────────────────────────────────────────────────
 * Reshapes R[k × chi_cut*D_] (col-major) into A[k*D_ × chi_cut] (col-major).
 * A[(r*D_+d) + k*D_*b] = R[r + k*(b + chi_cut*d)]
 * Replaces: D2H R1/R2 → CPU triple loop → H2D A1/A2 (~19 MB/layer). */
__global__ void reshape_R_to_A(const Scalar* R, Scalar* A,
                                int k, int chi_cut, int D_) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int kD  = k * D_;
    if (idx >= kD * chi_cut) return;
    int row = idx % kD;
    int b   = idx / kD;
    int r   = row / D_;
    int d   = row % D_;
    A[idx] = R[r + (size_t)k * (b + (size_t)chi_cut * d)];
}

/* ── LayerMetrics ────────────────────────────────────────────────────────────
 * Per-layer SVD quality metrics accumulated across all gates in one Trotter layer.
 * All fields are max/sum over gates; reset to zero before each layer. */
struct LayerMetrics {
    float gate_err   = 0.f; // max truncation weight Σ_{i>χ} S_i² / Σ_all S_i²  (per gate, take max)
    float Se_over_logchi = 0.f; // max S_bottleneck/log(χ_max): bond entanglement saturation
    float gap        = 1.f; // min singular value gap σ_{χ+1}/σ_χ (min over gates; 1 = no truncation)
    int   n_gates    = 0;   // number of gates that actually truncated (new_chi < kfull)
};

// QR strategy — selectable via --qr= on the command line.
enum class QrMode {
    Adaptive,    // Cholesky for large tall-thin matrices, Householder otherwise (default)
    Cholesky,    // always attempt Cholesky (may fall back if SPOTRF fails)
    Householder, // always Householder (backward stable, avoids float32 conditioning issues)
};

struct GateOps {
    rocblas_handle handle;
    Scalar* d_G = nullptr;  // device copy of current gate matrix [D²×D²]
    QrMode qr_mode = QrMode::Householder;

    GateOps(rocblas_handle h) : handle(h) {
        HIP_CHECK(hipMalloc(&d_G, D*D*D*D*sizeof(Scalar)));
    }
    ~GateOps() {
        if (d_G) { HIP_CHECK(hipFree(d_G)); d_G = nullptr; }
    }

    // Threshold for pseudo_sqrt: channels with λ < PSEUDO_SQRT_CUT are dropped.
    // Reverted to original — 1e-4 was too aggressive and zeroed valid channels
    // in the Vidal-normalized messages (Σλ²=1 at chi=400 has many small entries).
    static constexpr Scalar PSEUDO_SQRT_CUT = 10.f * FLT_EPSILON;
    static constexpr Scalar CHOL_DIAG_REG   = 1e-8f;
    static constexpr int    CHOL_QR_MIN_N   = 150;
    static constexpr float  CHOL_QR_ASPECT  = 150.f;

    /* ── GEMM ─────────────────────────────────────────────────────────── */
    void sgemm(rocblas_operation opA, rocblas_operation opB,
               int m, int n, int k,
               const Scalar* A, int lda,
               const Scalar* B, int ldb,
               Scalar* C, int ldc,
               Scalar alpha=1.f, Scalar beta=0.f)
    {
        RB_CHECK(rocblas_sgemm(handle,opA,opB,m,n,k,
                               &alpha,A,lda,B,ldb,&beta,C,ldc));
    }

    /* ── mode_n_contract ─────────────────────────────────────────────── */
    void mode_n_contract(const Scalar* in, const int* shape, int ndim,
                         const Scalar* E, int chi_in, int chi_out,
                         int mode, Scalar* out)
    {
        int L=1, R=1;
        for(int i=0;i<mode;i++)      L *= shape[i];
        for(int i=mode+1;i<ndim;i++) R *= shape[i];
        float alpha=1.f, beta=0.f;
        RB_CHECK(rocblas_sgemm_strided_batched(handle,
            rocblas_operation_none,
            rocblas_operation_transpose,
            L, chi_out, chi_in,
            &alpha,
            in,  L, (rocblas_stride)(L*chi_in),
            E,   chi_out, (rocblas_stride)(0),
            &beta,
            out, L, (rocblas_stride)(L*chi_out),
            R));
    }

    /* ── pseudo_sqrt ─────────────────────────────────────────────────── */
    // Fast diagonal path: BP diagonal messages are always diagonal (normalize_and_delta
    // zeros off-diagonal elements every sweep). Extract diagonal entries and compute
    // sqrt/inv_sqrt directly — O(chi) instead of O(chi³) ssyevd.
    // cut_rel: threshold relative to max diagonal entry. Entries below cut_rel*max
    // are zeroed in isqE (avoiding 1/sqrt(tiny) amplification). Default 1e-4 matches
    // the typical SVD cutoff — entries below this were already truncated by the gate.
    void pseudo_sqrt(const Scalar* M, Scalar* sqE, Scalar* isqE,
                     int chi, Workspace& ws,
                     Scalar cut=PSEUDO_SQRT_CUT)
    {
        hipStream_t rbstream; rocblas_get_stream(handle, &rbstream);
        // Extract diagonal entries, then compute sqrt/inv_sqrt with threshold cut.
        // Channels with λ < cut get isqE=0 — prevents 1/sqrt(tiny) amplification.
        { int blocks=(chi+255)/256;
          extract_diagonal<<<blocks,256,0,rbstream>>>(M, ws.syevd_W, chi); }
        { int blocks=(chi*chi+255)/256;
          diag_sqrt_matrices<<<blocks,256,0,rbstream>>>(
              ws.syevd_W, sqE, isqE, chi, cut); }
    }

    /* ── absorb_env ──────────────────────────────────────────────────── */
    // Uses ACTUAL bond.chi for tensor shape (the full tensor including cut bond).
    // Skips env contraction for bonds where chi grew (chi_tens != chi_msg).
    void absorb_env(const Scalar* psi, const Site& site, const Grid& grid,
                    Workspace& ws, int cut_dir,
                    const std::vector<int>& frozen_chi,
                    Scalar* dst, Scalar* tmp)
    {
        std::vector<int> shape;
        for(int d=0;d<MAX_DEGREE;d++)
            if(site.bond_id[d]>=0)
                shape.push_back(grid.bonds[site.bond_id[d]].chi);
        shape.push_back(D);
        int ndim=(int)shape.size(), total=1;
        for(auto s:shape) total*=s;
        HIP_CHECK(hipMemcpy(dst,psi,total*sizeof(Scalar),hipMemcpyDeviceToDevice));
        Scalar* cur=dst; Scalar* nxt=tmp;
        for(int d=0;d<MAX_DEGREE;d++){
            if(site.bond_id[d]<0||d==cut_dir) continue;
            int li       = site.bond_local[d];
            int chi_msg  = frozen_chi[grid.bonds[site.bond_id[d]].id];
            int chi_tens = grid.bonds[site.bond_id[d]].chi;
            if(chi_tens == chi_msg){
                mode_n_contract(cur,shape.data(),ndim,ws.sqrt_env[li],
                                chi_tens,chi_tens,li,nxt);
                std::swap(cur,nxt);
            }
        }
        if(cur!=dst)
            HIP_CHECK(hipMemcpy(dst,cur,total*sizeof(Scalar),hipMemcpyDeviceToDevice));
    }

    /* ── permute_cut_to_last ─────────────────────────────────────────── */
    // Permute tensor in-place so the cut bond is at position (degree-1),
    // just before D. Required for QR to work as a flat [m×n] matrix.
    // Returns the new shape. No-op if cut is already last bond.
    // scratch must point to a GPU buffer of at least `total` Scalars.
    std::vector<int> permute_cut_to_last(Scalar* cv, Scalar* scratch,
                                          const Site& sv, int cut_dir,
                                          const Grid& grid)
    {
        // Build current canonical shape
        std::vector<int> shape;
        for(int d=0;d<MAX_DEGREE;d++)
            if(sv.bond_id[d]>=0)
                shape.push_back(grid.bonds[sv.bond_id[d]].chi);
        shape.push_back(D);
        int ndim=(int)shape.size();
        int degree = sv.degree;
        int cut_local = sv.bond_local[cut_dir];

        if(cut_local == degree-1) return shape;  // already in right place

        // perm[new_pos] = old_pos: move cut_local to position degree-1
        std::vector<int> perm;
        for(int i=0;i<ndim;i++) perm.push_back(i);
        perm.erase(perm.begin()+cut_local);
        perm.insert(perm.begin()+(degree-1), cut_local);

        int total=1; for(auto s:shape) total*=s;

        // GPU permute: cv → scratch (no PCIe), then copy back
        gpu_tensor_permute(cv, scratch, shape.data(), perm.data(), ndim, total);
        HIP_CHECK(hipMemcpy(cv, scratch, (size_t)total*sizeof(Scalar),
                            hipMemcpyDeviceToDevice));

        std::vector<int> new_shape(ndim);
        for(int i=0;i<ndim;i++) new_shape[i]=shape[perm[i]];
        return new_shape;
    }

    /* ── inverse_permute_cut ─────────────────────────────────────────── */
    // Inverse of permute_cut_to_last: move cut bond from position (degree-1)
    // back to its canonical position cut_local.
    // new_chi_val is the size of the cut bond after SVD.
    // scratch must point to a GPU buffer of at least `total` Scalars.
    void inverse_permute_cut(Scalar* cv, Scalar* scratch,
                              const Site& sv, int cut_dir, int new_chi_val,
                              const Grid& grid)
    {
        int degree = sv.degree;
        int cut_local = sv.bond_local[cut_dir];
        if(cut_local == degree-1) return;  // no permutation was done

        // Current (permuted) shape: all non-cut bonds in dir order,
        // then new_chi_val at position degree-1, then D
        std::vector<int> perm_shape;
        for(int d=0;d<MAX_DEGREE;d++){
            if(sv.bond_id[d]<0||d==cut_dir) continue;
            perm_shape.push_back(grid.bonds[sv.bond_id[d]].chi);
        }
        perm_shape.insert(perm_shape.begin()+(degree-1), new_chi_val);
        perm_shape.push_back(D);
        int ndim=(int)perm_shape.size();

        // Target canonical shape: cut at cut_local
        std::vector<int> canon_shape;
        for(int d=0;d<MAX_DEGREE;d++){
            if(sv.bond_id[d]<0) continue;
            canon_shape.push_back(d==cut_dir ? new_chi_val : grid.bonds[sv.bond_id[d]].chi);
        }
        canon_shape.push_back(D);

        int total=1; for(auto s:perm_shape) total*=s;

        // The forward permutation perm[new_pos]=old_pos moved cut from cut_local to degree-1.
        // Invert it: inv_perm[old_pos] = new_pos, then apply as forward permute on perm_shape.
        // This maps permuted → canonical layout.
        std::vector<int> fwd_perm;
        for(int i=0;i<ndim;i++) fwd_perm.push_back(i);
        fwd_perm.erase(fwd_perm.begin()+cut_local);
        fwd_perm.insert(fwd_perm.begin()+(degree-1), cut_local);
        std::vector<int> inv_perm(ndim);
        for(int i=0;i<ndim;i++) inv_perm[fwd_perm[i]]=i;
        // inv_perm[new_canon_pos] = old_perm_pos — use as perm arg to gpu_tensor_permute

        // GPU permute: cv (permuted layout) → scratch (canonical layout), then copy back
        gpu_tensor_permute(cv, scratch, perm_shape.data(), inv_perm.data(), ndim, total);
        HIP_CHECK(hipMemcpy(cv, scratch, (size_t)total*sizeof(Scalar),
                            hipMemcpyDeviceToDevice));
    }

    /* ── deabsorb_env ────────────────────────────────────────────────── */
    // Called after inverse_permute_cut, so cv is in canonical layout.
    void deabsorb_env(Scalar* psi, const Site& site, const Grid& grid,
                      Workspace& ws, int cut_dir, int new_chi,
                      const std::vector<int>& frozen_chi, Scalar* tmp)
    {
        std::vector<int> shape;
        for(int d=0;d<MAX_DEGREE;d++){
            if(site.bond_id[d]<0) continue;
            shape.push_back(d==cut_dir ? new_chi : grid.bonds[site.bond_id[d]].chi);
        }
        shape.push_back(D);
        int ndim=(int)shape.size(), total=1;
        for(auto s:shape) total*=s;
        Scalar* cur=psi; Scalar* nxt=tmp;
        for(int d=0;d<MAX_DEGREE;d++){
            if(site.bond_id[d]<0||d==cut_dir) continue;
            int li       = site.bond_local[d];
            int chi_msg  = frozen_chi[grid.bonds[site.bond_id[d]].id];
            int chi_tens = grid.bonds[site.bond_id[d]].chi;
            if(chi_tens == chi_msg){
                mode_n_contract(cur,shape.data(),ndim,ws.inv_sqrt_env[li],
                                chi_tens,chi_tens,li,nxt);
                std::swap(cur,nxt);
            }
        }
        if(cur!=psi)
            HIP_CHECK(hipMemcpy(psi,cur,total*sizeof(Scalar),hipMemcpyDeviceToDevice));
    }

    /* ── apply_unary_gate ────────────────────────────────────────────── */
    void apply_unary_gate(Scalar* psi, const Site& site, const Grid& grid,
                          const Scalar* G, Scalar* tmp)
    {
        std::vector<int> shape;
        for(int d=0;d<MAX_DEGREE;d++)
            if(site.bond_id[d]>=0)
                shape.push_back(grid.bonds[site.bond_id[d]].chi);
        shape.push_back(D);
        int ndim=(int)shape.size(), total=1;
        for(auto s:shape) total*=s;
        Scalar* dG;
        HIP_CHECK(hipMalloc(&dG,D*D*sizeof(Scalar)));
        HIP_CHECK(hipMemcpy(dG,G,D*D*sizeof(Scalar),hipMemcpyHostToDevice));
        mode_n_contract(psi,shape.data(),ndim,dG,D,D,ndim-1,tmp);
        HIP_CHECK(hipMemcpy(psi,tmp,total*sizeof(Scalar),hipMemcpyDeviceToDevice));
        HIP_CHECK(hipFree(dG));
    }

    /* ── householder_qr ──────────────────────────────────────────────── */
    void householder_qr(const Scalar* A, int m, int n,
                        Scalar* Q, Scalar* R, int& k_out)
    {
        hipStream_t rbstream; rocblas_get_stream(handle, &rbstream);
        int k=std::min(m,n); k_out=k;
        HIP_CHECK(hipMemcpy(Q,A,(size_t)m*n*sizeof(Scalar),hipMemcpyDeviceToDevice));
        Scalar* tau; HIP_CHECK(hipMalloc(&tau,k*sizeof(Scalar)));
        RB_CHECK(rocsolver_sgeqrf(handle,m,n,Q,m,tau));
        HIP_CHECK(hipStreamSynchronize(rbstream));
        {
            int threads = 256;
            int blocks  = ((size_t)k*n + threads-1) / threads;
            extract_upper_triangle<<<blocks, threads, 0, rbstream>>>(Q, R, m, k, n);
        }
        RB_CHECK(rocsolver_sorgqr(handle,m,k,k,Q,m,tau));
        HIP_CHECK(hipStreamSynchronize(rbstream));
        HIP_CHECK(hipFree(tau));
        // Enforce positive diagonal of R → unique gauge for cross inner products
        {
            int threads = 256;
            dim3 bQ(k, (m + threads - 1) / threads);
            fix_qr_sign_Q<<<bQ, threads, 0, rbstream>>>(Q, R, m, k);
            dim3 bR(n, (k + threads - 1) / threads);
            fix_qr_sign_R<<<bR, threads, 0, rbstream>>>(R, k, n);
            HIP_CHECK(hipStreamSynchronize(rbstream));
        }
    }

    /* ── cholesky_qr ─────────────────────────────────────────────────── */
    bool cholesky_qr(const Scalar* A, int m, int n,
                     Scalar* Q, Scalar* R,
                     Scalar* gram, int* potrf_info, int& k_out)
    {
        hipStream_t rbstream; rocblas_get_stream(handle, &rbstream);
        k_out=n;
        sgemm(rocblas_operation_transpose,rocblas_operation_none,
              n,n,m,A,m,A,m,gram,n);
        { int blocks=(n+255)/256; add_diagonal<<<blocks,256,0,rbstream>>>(gram,n,CHOL_DIAG_REG); }
        RB_CHECK(rocsolver_spotrf(handle,rocblas_fill_upper,n,gram,n,potrf_info));
        HIP_CHECK(hipStreamSynchronize(rbstream));
        int hinfo; HIP_CHECK(hipMemcpy(&hinfo,potrf_info,sizeof(int),hipMemcpyDeviceToHost));
        if(hinfo!=0){k_out=0;return false;}
        { int blocks=(n*n+255)/256; extract_upper_triangle<<<blocks,256,0,rbstream>>>(gram,R,n,n,n); }
        // STRSM: Q = A × R^{-1} on-device — replaces CPU back-sub + H2D Rinv + SGEMM.
        // Solve Q × R = A in-place: init Q ← A, then rocblas_strsm overwrites Q.
        // rocblas_fill_upper: only reads upper triangle of R, lower ignored.
        HIP_CHECK(hipMemcpy(Q, A, (size_t)m*n*sizeof(Scalar), hipMemcpyDeviceToDevice));
        float alpha = 1.f;
        RB_CHECK(rocblas_strsm(handle,
            rocblas_side_right,
            rocblas_fill_upper,
            rocblas_operation_none,
            rocblas_diagonal_non_unit,
            m, n, &alpha,
            R, n,
            Q, m));
        return true;
    }

    /* ── adaptive_qr ─────────────────────────────────────────────────── */
    int adaptive_qr(const Scalar* A, int m, int n,
                    Scalar* Q, Scalar* R,
                    Scalar* gram, int* potrf_info,
                    int& chol_attempted, int& chol_fallback, int& chol_skipped)
    {
        int k=0;
        bool try_chol = false;
        switch (qr_mode) {
            case QrMode::Cholesky:   try_chol = true; break;
            case QrMode::Householder: try_chol = false; break;
            case QrMode::Adaptive:
            default:
                try_chol = (n > CHOL_QR_MIN_N && (float)m/n > CHOL_QR_ASPECT);
        }
        if (try_chol) {
            chol_attempted++;
            if(cholesky_qr(A,m,n,Q,R,gram,potrf_info,k)&&k==n) return k;
            chol_fallback++;   // SPOTRF failed or rank deficient → Householder
        } else {
            chol_skipped++;    // skipped by mode or below threshold
        }
        householder_qr(A,m,n,Q,R,k);
        return k;
    }

    /* ================================================================
     * apply_binary_gate
     * ================================================================ */
    template<typename GridT = Grid>
    void apply_binary_gate(
        Scalar* psi_v1, Scalar* psi_v2,
        const Gate& gate, GridT& grid,
        std::vector<Scalar*>& out_messages,
        const std::vector<Scalar*>& env_messages,
        const std::vector<int>& frozen_chi,
        const Scalar* G,
        Workspace& ws,
        int chi_max, Scalar cutoff=1e-10f,
        LayerMetrics* lm=nullptr,
        std::vector<Scalar*>* out_messages_v=nullptr)  // chi-vector output (optional)
    {
        hipStream_t rbstream; rocblas_get_stream(handle, &rbstream);
        const Site& sv1=grid.sites[gate.site_a];
        const Site& sv2=grid.sites[gate.site_b];
        int cd1=gate.dir_a_to_b, cd2=gate.dir_b_to_a;
        Bond& bond=grid.bonds[gate.bond_id];

        // ── Step 1+2: pseudo_sqrt then absorb — per site, not interleaved.
        // sv1 and sv2 share ws.sqrt_env/inv_sqrt_env indexed by bond_local.
        // Computing both sites' pseudo_sqrts before either absorb would let
        // sv2's loop overwrite sv1's slots when bond_local indices coincide.
        for(int d=0;d<MAX_DEGREE;d++){
            if(sv1.bond_id[d]<0||d==cd1) continue;
            int li=sv1.bond_local[d];
            const Bond& eb=grid.bonds[sv1.bond_id[d]];
            pseudo_sqrt(env_messages[eb.msg_incoming(sv1.id)],
                        ws.sqrt_env[li],ws.inv_sqrt_env[li],frozen_chi[eb.id],ws);
        }
        absorb_env(psi_v1,sv1,grid,ws,cd1,frozen_chi,ws.cv1,ws.Q1);

        for(int d=0;d<MAX_DEGREE;d++){
            if(sv2.bond_id[d]<0||d==cd2) continue;
            int li=sv2.bond_local[d];
            const Bond& eb=grid.bonds[sv2.bond_id[d]];
            pseudo_sqrt(env_messages[eb.msg_incoming(sv2.id)],
                        ws.sqrt_env[li],ws.inv_sqrt_env[li],frozen_chi[eb.id],ws);
        }
        absorb_env(psi_v2,sv2,grid,ws,cd2,frozen_chi,ws.cv2,ws.Q2);

        // ── Step 2b: permute cut bond to last-bond position ───────────
        // QR needs layout [..., chi_cut, D]. Permute if cut_local != degree-1.
        std::vector<int> shape1 = permute_cut_to_last(ws.cv1, ws.Q1, sv1, cd1, grid);
        std::vector<int> shape2 = permute_cut_to_last(ws.cv2, ws.Q2, sv2, cd2, grid);

        // ── Step 3: QR dimensions from permuted shapes ────────────────
        // shape = [...env_dims..., chi_cut, D]
        // m = product of all except last two; n = chi_cut * D
        int chi_cut = frozen_chi[bond.id];
        int n1 = chi_cut * D;
        int m1 = 1;
        for(int i=0;i<(int)shape1.size()-2;i++) m1*=shape1[i];
        int k1 = std::min(m1,n1);

        int n2 = chi_cut * D;
        int m2 = 1;
        for(int i=0;i<(int)shape2.size()-2;i++) m2*=shape2[i];
        int k2 = std::min(m2,n2);

        // ── Step 4: QR decomposition ──────────────────────────────────
        int k1_out = adaptive_qr(ws.cv1,m1,n1,ws.Q1,ws.R1,ws.gram,ws.potrf_info,
                                 ws.chol_attempted,ws.chol_fallback,ws.chol_skipped);
        int k2_out = adaptive_qr(ws.cv2,m2,n2,ws.Q2,ws.R2,ws.gram,ws.potrf_info,
                                 ws.chol_attempted,ws.chol_fallback,ws.chol_skipped);
        k1=k1_out; k2=k2_out;

        // ── Step 5: Build RR4[k1D × k2D] via GPU GEMM ────────────────
        // Tensor after permute_cut_to_last: [...env..., chi_cut, D] (col-major).
        // QR left=env, right=[chi_cut,D], n1=chi_cut*D columns.
        // R1 col-major [k1 × n1]: R1[r1 + k1*col] where col = b + chi_cut*d
        // A1 col-major [k1D × chi_cut]: A1[(r1*D+d) + k1D*b] = R1[r1 + k1*(b + chi_cut*d)]
        // RR4 = A1 * A2^T
        int k1D=k1*D, k2D=k2*D;
        { int b1=(k1D*chi_cut+255)/256; reshape_R_to_A<<<b1,256,0,rbstream>>>(ws.R1,ws.A1,k1,chi_cut,D); }
        { int b2=(k2D*chi_cut+255)/256; reshape_R_to_A<<<b2,256,0,rbstream>>>(ws.R2,ws.A2,k2,chi_cut,D); }
        sgemm(rocblas_operation_none,rocblas_operation_transpose,
              k1D,k2D,chi_cut,ws.A1,k1D,ws.A2,k2D,ws.RR4,k1D);

        // ── Step 6: Apply gate to RR4 ─────────────────────────────────
        HIP_CHECK(hipMemcpyAsync(d_G,G,D*D*D*D*sizeof(Scalar),hipMemcpyHostToDevice,rbstream));
        { int tot=(k1D*k2D+255)/256;
          apply_gate_to_RR4<<<tot,256,D*D*D*D*sizeof(Scalar),rbstream>>>(
              d_G,ws.RR4,ws.oR4,k1,k2,D); }

        // ── Step 7: SVD of oR4 [k1D × k2D] in float64 (mixed precision) ──
        // float32 SVD loses small singular values at large chi (Σλ²=1 with 400
        // channels → λ_min can be ~1e-6, below float32 relative precision at
        // scale). float64 gives 15 decimal digits — accurate singular values
        // → accurate Vidal messages → BP finds correct fixed point.
        int kfull=std::min(k1D,k2D);
        int oR4_n = k1D * k2D;
        { int blocks=(oR4_n+255)/256;
          cast_f32_to_f64<<<blocks,256,0,rbstream>>>(ws.oR4, ws.oR4_d, oR4_n); }
        { int blocks=(k1D*kfull+255)/256;  // U_d needs k1D×kfull
          // zero U_d and Vt_d (dgesvd may not write unused entries)
        }
        HIP_CHECK(hipMemset(ws.svd_info, 0, sizeof(int)));
        RB_CHECK(rocsolver_dgesvd(handle,
            rocblas_svect_singular, rocblas_svect_singular,
            k1D, k2D, ws.oR4_d, k1D, ws.S_d, ws.U_d, k1D, ws.Vt_d, kfull,
            ws.svd_work_d, rocblas_outofplace, ws.svd_info));
        // Cast S_d (float64) → S (float32) for downstream use
        { int blocks=(kfull+255)/256;
          cast_f64_to_f32<<<blocks,256,0,rbstream>>>(ws.S_d, ws.S, kfull); }
        // Cast U_d → U, Vt_d → Vt for reconstruction GEMMs (float32 sufficient)
        { int blocks=(k1D*kfull+255)/256;
          cast_f64_to_f32<<<blocks,256,0,rbstream>>>(ws.U_d, ws.U, k1D*kfull); }
        { int blocks=(kfull*k2D+255)/256;
          cast_f64_to_f32<<<blocks,256,0,rbstream>>>(ws.Vt_d, ws.Vt, kfull*k2D); }
        HIP_CHECK(hipStreamSynchronize(rbstream));

        // D2H singular values to determine new_chi (data-dependent truncation rank)
        std::vector<Scalar> hS(kfull);
        HIP_CHECK(hipMemcpy(hS.data(),ws.S,(size_t)kfull*sizeof(Scalar),hipMemcpyDeviceToHost));
        Scalar smax=hS.empty()?1.f:hS[0];
        int new_chi=0;
        for(int i=0;i<kfull;i++){
            if(hS[i]<cutoff*smax||i>=chi_max) break;
            new_chi++;
        }
        if(new_chi==0) new_chi=1;

        // ── SVD quality metrics ───────────────────────────────────────
        if(lm) {
            // Truncation error: discarded weight fraction (normalized by total)
            float total_w = 0.f, kept_w = 0.f;
            for(int i=0;i<kfull;i++) total_w += hS[i]*hS[i];
            for(int i=0;i<new_chi;i++) kept_w  += hS[i]*hS[i];
            float err = (total_w > 0.f) ? (total_w - kept_w) / total_w : 0.f;
            lm->gate_err = std::max(lm->gate_err, err);

            // Bond entanglement entropy S_e = -Σ p_i log p_i, normalized by log(chi_max)
            float Se = 0.f;
            if(kept_w > 0.f){
                for(int i=0;i<new_chi;i++){
                    float p = (hS[i]*hS[i]) / kept_w;
                    if(p > 1e-12f) Se -= p * std::log(p);
                }
            }
            float logchi = (chi_max > 1) ? std::log((float)chi_max) : 1.f;
            lm->Se_over_logchi = std::max(lm->Se_over_logchi, Se / logchi);

            // Singular value gap: σ_{new_chi} / σ_{new_chi-1} (ratio at cut edge)
            if(new_chi < kfull && new_chi > 0 && hS[new_chi-1] > 0.f)
                lm->gap = std::min(lm->gap, hS[new_chi] / hS[new_chi-1]);

            lm->n_gates++;
        }

        // Vidal gauge normalization: enforce Σ λ²=1 over kept singular values.
        // Prevents float32 overflow at large chi; physics preserved because
        // autocorrelation_with_bp computes num/den and the scale cancels.
        Scalar snorm2=0.f;
        for(int i=0;i<new_chi;i++) snorm2+=hS[i]*hS[i];
        Scalar snorm=(snorm2>0.f)?std::sqrt(snorm2):1.f;
        Scalar inv_snorm=1.f/snorm;

        // ── Step 8: Reconstruct tensors via GPU kernels + GEMM ────────
        // build_M_from_U:  ws.U[k1D×kfull]  → ws.A1[k1×new_chi*D]
        // build_M_from_Vt: ws.Vt[kfull×k2D] → ws.A2[k2×new_chi*D]
        // M[r + k*(c + new_chi*d)] = src * sqrt(S[c]*inv_snorm)
        // Then GEMM: cv = Q[m×k] × M[k×new_chi*D]
        { int tot=(k1*new_chi*D+255)/256;
          build_M_from_U<<<tot,256,0,rbstream>>>(
              ws.U,ws.S,ws.A1,k1,k1D,new_chi,D,inv_snorm); }
        { int tot=(k2*new_chi*D+255)/256;
          build_M_from_Vt<<<tot,256,0,rbstream>>>(
              ws.Vt,ws.S,ws.A2,k2,kfull,new_chi,D,inv_snorm); }
        sgemm(rocblas_operation_none,rocblas_operation_none,
              m1,new_chi*D,k1,ws.Q1,m1,ws.A1,k1,ws.cv1,m1);
        sgemm(rocblas_operation_none,rocblas_operation_none,
              m2,new_chi*D,k2,ws.Q2,m2,ws.A2,k2,ws.cv2,m2);

        // ── Step 9: inverse permute to canonical layout ───────────────
        // cv is in permuted layout [env_permuted..., new_chi, D].
        // Restore canonical [UP,DOWN,LEFT,RIGHT order..., D] so that
        // deabsorb_env (which uses canonical direction order) works correctly.
        inverse_permute_cut(ws.cv1, ws.Q1, sv1, cd1, new_chi, grid);
        inverse_permute_cut(ws.cv2, ws.Q2, sv2, cd2, new_chi, grid);

        // ── Step 9b: deabsorb inv_sqrt envs ──────────────────────────
        // Recompute pseudo_sqrt per site immediately before its deabsorb so
        // ws.inv_sqrt_env slots hold the correct site's values.
        for(int d=0;d<MAX_DEGREE;d++){
            if(sv1.bond_id[d]<0||d==cd1) continue;
            int li=sv1.bond_local[d];
            const Bond& eb=grid.bonds[sv1.bond_id[d]];
            pseudo_sqrt(env_messages[eb.msg_incoming(sv1.id)],
                        ws.sqrt_env[li],ws.inv_sqrt_env[li],frozen_chi[eb.id],ws);
        }
        deabsorb_env(ws.cv1,sv1,grid,ws,cd1,new_chi,frozen_chi,ws.Q1);

        for(int d=0;d<MAX_DEGREE;d++){
            if(sv2.bond_id[d]<0||d==cd2) continue;
            int li=sv2.bond_local[d];
            const Bond& eb=grid.bonds[sv2.bond_id[d]];
            pseudo_sqrt(env_messages[eb.msg_incoming(sv2.id)],
                        ws.sqrt_env[li],ws.inv_sqrt_env[li],frozen_chi[eb.id],ws);
        }
        deabsorb_env(ws.cv2,sv2,grid,ws,cd2,new_chi,frozen_chi,ws.Q2);

        // ── Step 10: copy back to psi ─────────────────────────────────
        // Shape is now canonical [chi_UP,...,new_chi_at_cut,...,D]
        int tot1=new_chi*D;
        for(int d=0;d<MAX_DEGREE;d++)
            if(sv1.bond_id[d]>=0&&d!=cd1)
                tot1*=grid.bonds[sv1.bond_id[d]].chi;
        int tot2=new_chi*D;
        for(int d=0;d<MAX_DEGREE;d++)
            if(sv2.bond_id[d]>=0&&d!=cd2)
                tot2*=grid.bonds[sv2.bond_id[d]].chi;
        HIP_CHECK(hipMemcpy(psi_v1,ws.cv1,(size_t)tot1*sizeof(Scalar),hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy(psi_v2,ws.cv2,(size_t)tot2*sizeof(Scalar),hipMemcpyDeviceToDevice));

        // ── Step 11: update bond.chi and messages ────────────────────
        // Store Vidal-normalized singular values (λ_i / snorm so Σ λ²=1).
        bond.chi=new_chi;
        std::vector<Scalar> msg((size_t)new_chi*new_chi,0.f);
        std::vector<Scalar> msg_v(new_chi);
        for(int i=0;i<new_chi;i++){
            Scalar lam = std::max(hS[i]/snorm,0.f);
            msg[i+i*new_chi] = lam;
            msg_v[i]         = lam;
        }
        HIP_CHECK(hipMemcpy(out_messages[bond.msg_from_a()],msg.data(),
                            (size_t)new_chi*new_chi*sizeof(Scalar),hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(out_messages[bond.msg_from_b()],msg.data(),
                            (size_t)new_chi*new_chi*sizeof(Scalar),hipMemcpyHostToDevice));
        // Also write chi-vector if provided
        if(out_messages_v){
            HIP_CHECK(hipMemcpy((*out_messages_v)[bond.msg_from_a()],msg_v.data(),
                                (size_t)new_chi*sizeof(Scalar),hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy((*out_messages_v)[bond.msg_from_b()],msg_v.data(),
                                (size_t)new_chi*sizeof(Scalar),hipMemcpyHostToDevice));
        }
    }

    /* ── convenience overload ────────────────────────────────────────── */
    void apply_binary_gate(Scalar* psi_v1, Scalar* psi_v2,
                           const Gate& gate, Grid& grid,
                           std::vector<Scalar*>& messages,
                           const Scalar* G, Workspace& ws,
                           int chi_max, Scalar cutoff=1e-10f)
    {
        std::vector<int> cur_chi(grid.n_bonds);
        for(auto& b:grid.bonds) cur_chi[b.id]=b.chi;
        apply_binary_gate(psi_v1,psi_v2,gate,grid,
                          messages,messages,cur_chi,
                          G,ws,chi_max,cutoff);
    }
};
