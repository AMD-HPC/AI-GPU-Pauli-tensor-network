#pragma once
/*
 * permute_gpu.hpp — GPU tensor permutation kernel (HIP)
 *
 * Replaces the CPU D2H + element-shuffle + H2D in permute_cut_to_last and
 * inverse_permute_cut. One GPU thread per element; no PCIe traffic.
 *
 * Layout: col-major (Fortran order), ndim ≤ 6 (MAX_DEGREE=4 bonds + D, plus
 * we keep one spare). Permutation descriptor fits in 6 ints uploaded as a
 * small struct — no global memory round-trips for metadata.
 *
 * Usage:
 *   gpu_tensor_permute(src, dst, shape_in, perm, ndim, total)
 *
 * where perm[new_dim] = old_dim (same convention as the CPU code).
 * src and dst must NOT alias. Call with a temporary GPU buffer when permuting
 * in-place; copy back after.
 */

#include <hip/hip_runtime.h>
#include "types.hpp"   // Scalar, HIP_CHECK

#define PERMUTE_MAX_DIM 6

struct PermuteMeta {
    int shape_in [PERMUTE_MAX_DIM];   // input dimensions (for index decomposition)
    int stride_out[PERMUTE_MAX_DIM];  // output strides (col-major of permuted shape)
    int perm[PERMUTE_MAX_DIM];        // perm[new_dim] = old_dim
    int ndim;
};

/* ── kernel ────────────────────────────────────────────────────────────── */
// Each thread handles one element. Decompose flat input index into per-dim
// indices, permute them, recompose into flat output index.
__global__ void tensor_permute_kernel(const Scalar* __restrict__ src,
                                       Scalar*       __restrict__ dst,
                                       PermuteMeta   meta,
                                       int           total)
{
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    if(flat >= total) return;

    // Decompose flat index in input (col-major) layout
    int idx[PERMUTE_MAX_DIM];
    int rem = flat;
#pragma unroll
    for(int i = 0; i < PERMUTE_MAX_DIM; i++){
        if(i < meta.ndim){
            idx[i] = rem % meta.shape_in[i];
            rem    /= meta.shape_in[i];
        }
    }

    // Build output flat index: out_flat = Σ_i idx[perm[i]] * stride_out[i]
    int out_flat = 0;
#pragma unroll
    for(int i = 0; i < PERMUTE_MAX_DIM; i++){
        if(i < meta.ndim)
            out_flat += idx[meta.perm[i]] * meta.stride_out[i];
    }

    dst[out_flat] = src[flat];
}

/* ── host-side launcher ─────────────────────────────────────────────── */
// Permutes `total` elements from src → dst on the GPU.
// perm[new_dim] = old_dim; both arrays live on GPU.
// Uses `tmp_buf` as scratch when src==dst (in-place).
inline void gpu_tensor_permute(const Scalar* src,
                                Scalar*       dst,
                                const int*    shape_in,   // host array, ndim entries
                                const int*    perm,       // host array, ndim entries
                                int           ndim,
                                int           total)
{
    PermuteMeta meta;
    meta.ndim = ndim;

    // Build input shape and output strides (col-major of permuted shape)
    meta.stride_out[0] = 1;
    for(int i = 0; i < ndim; i++){
        meta.shape_in[i] = shape_in[i];
        if(i > 0)
            meta.stride_out[i] = meta.stride_out[i-1] * shape_in[perm[i-1]];
        meta.perm[i] = perm[i];
    }
    // Zero-fill unused slots so the kernel's unrolled loop is safe
    for(int i = ndim; i < PERMUTE_MAX_DIM; i++){
        meta.shape_in[i]  = 1;
        meta.stride_out[i] = 0;
        meta.perm[i] = i;
    }

    const int BLOCK = 256;
    int grid = (total + BLOCK - 1) / BLOCK;
    hipLaunchKernelGGL(tensor_permute_kernel, dim3(grid), dim3(BLOCK), 0, 0,
                       src, dst, meta, total);
}

/* ── in-place variant ───────────────────────────────────────────────── */
// Uses a pre-allocated GPU scratch buffer of at least `total` Scalars.
inline void gpu_tensor_permute_inplace(Scalar* cv,
                                        Scalar* scratch,
                                        const int* shape_in,
                                        const int* perm,
                                        int        ndim,
                                        int        total)
{
    gpu_tensor_permute(cv, scratch, shape_in, perm, ndim, total);
    HIP_CHECK(hipMemcpy(cv, scratch, (size_t)total * sizeof(Scalar),
                        hipMemcpyDeviceToDevice));
}
