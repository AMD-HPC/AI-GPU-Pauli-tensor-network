#pragma once
/*
 * overlap_gpu.hpp — GPU-to-GPU inner product ⟨ψ_nl|ψ_n⟩ via P2P
 *
 * Uses HIP peer-to-peer memory access to compute the overlap between
 * a noiseless state on GPU 0 and a noisy state on GPU 1 entirely on GPU.
 *
 * Protocol:
 *   1. Enable P2P access (call enable_p2p() once at startup)
 *   2. For each noisy state: P2P copy psi_n → GPU 0 scratch buffers
 *   3. Run autocorrelation_with_bp on GPU 0 using:
 *      - noiseless messages (environment)
 *      - noisy psi (operator to measure)
 *   4. Free scratch buffers
 *
 * The overlap ⟨ψ_nl(t)|ψ_n(t)⟩ in the Heisenberg picture equals
 * Tr(ψ_nl† · ψ_n) approximated by the Bethe formula using the
 * noiseless BP messages as the environment.
 *
 * IMPORTANT: This uses the noiseless messages as an approximation
 * for the environment — equivalent to computing C(t) of the noisy
 * operator in the noiseless background. For the exact bilinear form
 * BP would need to run on the double-layer network.
 */

#include "types.hpp"
#include "grid.hpp"
#include "measures.hpp"
#include "heavyhex.hpp"
#include <vector>
#include <hip/hip_runtime.h>

/* ── Enable P2P access between all GPU pairs ────────────────────────────── */
inline void enable_p2p(int n_gpus) {
    for (int i = 0; i < n_gpus; i++) {
        HIP_CHECK(hipSetDevice(i));
        for (int j = 0; j < n_gpus; j++) {
            if (i == j) continue;
            int can = 0;
            HIP_CHECK(hipDeviceCanAccessPeer(&can, i, j));
            if (can) {
                // Tolerate "already enabled" — benign on repeat calls.
                hipError_t e = hipDeviceEnablePeerAccess(j, 0);
                if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
                    HIP_CHECK(e);
            }
        }
    }
    printf("[P2P] Peer access enabled between %d GPUs\n", n_gpus);
}

/* ── P2P copy: noisy psi (GPU 1) → scratch buffers on GPU 0 ─────────────── */
inline std::vector<Scalar*> p2p_copy_psi(
    const std::vector<Scalar*>& psi_src,  // on GPU src_dev
    const FlatHeavyHexGrid& grid,
    int chi_max,
    int src_dev,
    int dst_dev)
{
    HIP_CHECK(hipSetDevice(dst_dev));
    std::vector<Scalar*> psi_dst(grid.n_sites, nullptr);
    for (auto& site : grid.sites) {
        size_t sz = (size_t)site.tensor_size_max(chi_max) * sizeof(Scalar);
        HIP_CHECK(hipMalloc(&psi_dst[site.id], sz));
        // P2P copy: src_dev → dst_dev without going through host
        HIP_CHECK(hipMemcpyPeer(psi_dst[site.id], dst_dev,
                                psi_src[site.id],  src_dev, sz));
    }
    return psi_dst;
}

inline void free_psi(std::vector<Scalar*>& psi, int dev) {
    HIP_CHECK(hipSetDevice(dev));
    for (auto p : psi) if (p) HIP_CHECK(hipFree(p));
    psi.clear();
}

/* ── Compute overlap ⟨ψ_nl|ψ_n(γ)⟩ for all gammas ──────────────────────── */
/*
 * Uses the noiseless state messages (already converged by BP on GPU 0)
 * as the environment, and the noisy psi (copied via P2P to GPU 0) as
 * the operator.
 *
 * Returns overlaps[n_gammas].
 */
inline void compute_overlaps_p2p(
    Measures& meas,                          // on GPU 0
    const Grid& grid_nl,                     // noiseless grid (GPU 0)
    const std::vector<Scalar*>& psi_nl,      // noiseless psi (GPU 0)
    const std::vector<Scalar*>& msg_nl,      // noiseless messages (GPU 0)
    int center_site,
    const NoisyHeavyHexGrid* grid_n[],       // noisy grids (GPU 1)
    const HeavyHexState* state_n[],          // noisy states (GPU 1)
    int chi_n,
    int n_gammas,
    int gpu_nl,                              // GPU 0
    int gpu_n,                               // GPU 1
    float* overlaps_out,                     // [n_gammas] output
    float* Ct_nl_out = nullptr)              // optional: noiseless C(t)
{
    // Compute noiseless C(t) using noiseless psi + messages
    HIP_CHECK(hipSetDevice(gpu_nl));
    double tb=0, tp=0; int sw=0; float dl=0.f;
    if (Ct_nl_out) {
        *Ct_nl_out = meas.autocorrelation_with_bp(
            grid_nl, psi_nl, msg_nl,
            center_site, 100, &tb, &tp, &sw, &dl);
    }

    // For each gamma: P2P copy noisy psi + messages to GPU 0, compute overlap
    for (int g = 0; g < n_gammas; g++) {
        // P2P copy noisy psi: GPU 1 → GPU 0
        auto psi_n_gpu0 = p2p_copy_psi(
            state_n[g]->psi, *grid_n[g], chi_n, gpu_n, gpu_nl);

        // P2P copy noisy messages: GPU 1 → GPU 0
        // (messages must match the grid's bond chi values)
        HIP_CHECK(hipSetDevice(gpu_nl));
        int nm = 2 * grid_n[g]->n_bonds;
        std::vector<Scalar*> msg_n_gpu0(nm, nullptr);
        for (auto& bond : grid_n[g]->bonds) {
            for (int d = 0; d < 2; d++) {
                int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                int chi = bond.chi;
                size_t sz = (size_t)chi*chi*sizeof(Scalar);
                HIP_CHECK(hipMalloc(&msg_n_gpu0[idx], sz));
                HIP_CHECK(hipMemcpyPeer(msg_n_gpu0[idx], gpu_nl,
                                        state_n[g]->messages[idx], gpu_n, sz));
            }
        }

        // Compute ⟨ψ_nl|ψ_n⟩: use noisy grid + noisy messages + noisy psi
        // This gives C(t) of the noisy operator in the noisy environment —
        // which is the correct approximation for the MPF overlap.
        overlaps_out[g] = meas.autocorrelation_with_bp(
            static_cast<const Grid&>(*grid_n[g]), psi_n_gpu0, msg_n_gpu0,
            center_site, 100, &tb, &tp, &sw, &dl);

        // Free scratch on GPU 0
        free_psi(psi_n_gpu0, gpu_nl);
        for (auto p : msg_n_gpu0) if (p) HIP_CHECK(hipFree(p));
    }
}
