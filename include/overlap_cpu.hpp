#pragma once
/*
 * overlap_cpu.hpp — CPU inner product ⟨ψ_nl|ψ_n⟩ for heavy-hex MPF
 *
 * Computes the overlap between two Heisenberg-picture operator states
 * using the Bethe approximation on CPU. Used for the MPF observable:
 *
 *   overlap(ψ_nl, ψ_n) = Tr(ψ_nl · ψ_n) / Tr(ψ_nl · ψ_nl)
 *                       ≈ Bethe_inner(ψ_nl, ψ_n, messages)
 *
 * For diagonal BP messages (CppSim default), the Bethe approximation gives:
 *
 *   inner(ψ_nl, ψ_n) = [∏_v vertex_scalar(v)] / [∏_e edge_scalar(e)]
 *
 * where:
 *   vertex_scalar(v) = Σ_d ψ_nl_v[...d...] * ψ_n_v[...d...] * env_v
 *   edge_scalar(e)   = Tr(msg_e)
 *
 * Both states must be D2H to CPU host vectors before calling this.
 *
 * This is a simplified implementation for testing — GPU version to follow.
 */

#include "types.hpp"
#include "grid.hpp"
#include "heavyhex.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <cblas.h>   // CPU BLAS for GEMV/DOT

/* ── D2H helpers ─────────────────────────────────────────────────────────── */

// Download one site tensor from GPU to CPU host vector
inline std::vector<float> download_psi_site(const float* d_psi, int size) {
    std::vector<float> h(size);
    hipMemcpy(h.data(), d_psi, size*sizeof(float), hipMemcpyDeviceToHost);
    return h;
}

// Download all site tensors from GPU to CPU
inline std::vector<std::vector<float>> download_psi(
    const std::vector<Scalar*>& psi,
    const FlatHeavyHexGrid& grid, int chi_max)
{
    std::vector<std::vector<float>> h_psi(grid.n_sites);
    for (auto& site : grid.sites) {
        int sz = site.tensor_size_max(chi_max);
        h_psi[site.id].resize(sz);
        hipMemcpy(h_psi[site.id].data(), psi[site.id],
                  sz*sizeof(float), hipMemcpyDeviceToHost);
    }
    return h_psi;
}

// Download diagonal messages (chi-vector per bond direction) from GPU
// Messages are chi×chi diagonal on GPU — download as chi-vectors
inline std::vector<std::vector<float>> download_messages_diag(
    const std::vector<Scalar*>& messages,
    const FlatHeavyHexGrid& grid, int chi_max)
{
    int nm = 2 * grid.n_bonds;
    std::vector<std::vector<float>> h_msg(nm);
    for (auto& bond : grid.bonds) {
        for (int d = 0; d < 2; d++) {
            int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
            int chi = bond.chi;
            // Download full chi×chi then extract diagonal
            std::vector<float> full(chi*chi);
            hipMemcpy(full.data(), messages[idx],
                      chi*chi*sizeof(float), hipMemcpyDeviceToHost);
            h_msg[idx].resize(chi);
            for (int i = 0; i < chi; i++) h_msg[idx][i] = full[i + chi*i];
        }
    }
    return h_msg;
}

/* ── CPU Bethe inner product ─────────────────────────────────────────────── */

/*
 * Compute ⟨ψ_nl|ψ_n⟩ via the Bethe approximation using BP messages
 * from the noiseless state.
 *
 * For each site v:
 *   vertex_scalar(v) = Σ_{bond_indices, d} ψ_nl_v[...] * ψ_n_v[...] * env_v[...]
 *
 * For diagonal messages, env_v = ∏_{b ∈ bonds(v)} diag(msg_b).
 * The contraction reduces to:
 *   vertex_scalar(v) = Σ_d [Σ_{b0} msg_b0[i0] * ψ_nl_v[i0,...,d] * ψ_n_v[i0,...,d]]
 *
 * This is a simplified version that works for product-like states.
 * For full accuracy we'd need to run BP on the double-layer network.
 *
 * Returns: log of the numerator and denominator separately for numerical stability.
 */
inline float cpu_bethe_inner(
    const std::vector<std::vector<float>>& h_psi_nl,
    const std::vector<std::vector<float>>& h_psi_n,
    const std::vector<std::vector<float>>& h_msg,
    const FlatHeavyHexGrid& grid,
    int chi_max)
{
    double log_num = 0.0;  // log of numerator product
    double log_den = 0.0;  // log of denominator product

    // Vertex contributions
    for (auto& site : grid.sites) {
        const auto& psi_nl = h_psi_nl[site.id];
        const auto& psi_n  = h_psi_n[site.id];

        // Build env: product of diagonal messages on all bonds
        // env is a tensor of same shape as psi (without D)
        auto shape = site.tensor_shape(grid.bonds);
        int  n_env = 1;
        for (int i = 0; i < (int)shape.size()-1; i++) n_env *= shape[i];

        std::vector<float> env(n_env, 1.0f);

        // Multiply in each bond's diagonal message
        for (int d = 0; d < MAX_DEGREE; d++) {
            if (site.bond_id[d] < 0) continue;
            int li  = site.bond_local[d];
            int bid = site.bond_id[d];
            // Determine direction: incoming message
            const Bond& bond = grid.bonds[bid];
            int msg_idx = bond.msg_incoming(site.id);
            const auto& msg = h_msg[msg_idx];
            int chi = (int)msg.size();

            // env[...] *= msg[index along dimension li]
            // Stride: product of all dims before li
            int stride_before = 1;
            for (int dd = 0; dd < li; dd++) stride_before *= shape[dd];
            int stride_after  = n_env / (stride_before * chi);

            for (int a = 0; a < stride_before; a++)
                for (int i = 0; i < chi; i++)
                    for (int b = 0; b < stride_after; b++) {
                        int env_idx = a * chi * stride_after + i * stride_after + b;
                        if (i < (int)msg.size()) env[env_idx] *= msg[i];
                    }
        }

        // Contract: Σ_{env_idx, d} env[env_idx] * psi_nl[env_idx*D+d] * psi_n[env_idx*D+d]
        double v_scalar = 0.0;
        for (int ei = 0; ei < n_env; ei++)
            for (int d = 0; d < D; d++)
                v_scalar += env[ei] * psi_nl[ei*D+d] * psi_n[ei*D+d];

        double av = std::fabs(v_scalar);
        if (av > 0) log_num += std::log(av);
    }

    // Edge corrections: Σ_e log(Tr(msg_e))
    for (auto& bond : grid.bonds) {
        for (int d = 0; d < 2; d++) {
            int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
            const auto& msg = h_msg[idx];
            double tr = 0.0;
            for (float v : msg) tr += v;
            if (tr > 0) log_den += std::log(tr);
        }
    }

    // Return exp(log_num - log_den) — the Bethe approximation
    return (float)std::exp(log_num - log_den);
}

/* ── Main overlap function ───────────────────────────────────────────────── */

/*
 * Compute MPF overlaps for all 3 gamma values.
 * Downloads states from GPU 0 (noiseless) and GPU 1 (noisy) to CPU.
 * Returns overlaps[3] = { ⟨nl|n1.0⟩, ⟨nl|n1.4⟩, ⟨nl|n1.8⟩ }
 */
inline void compute_mpf_overlaps(
    const FlatHeavyHexGrid& grid_nl,
    const std::vector<Scalar*>& psi_nl,
    const std::vector<Scalar*>& msg_nl,
    int chi_nl,
    const NoisyHeavyHexGrid* grid_n[],
    const std::vector<Scalar*>* psi_n[],
    const std::vector<Scalar*>* msg_n[],
    int chi_n,
    int n_gammas,
    float* overlaps_out)   // [n_gammas]
{
    // Download noiseless state from GPU 0
    hipSetDevice(0);
    auto h_psi_nl = download_psi(psi_nl, grid_nl, chi_nl);
    auto h_msg_nl = download_messages_diag(msg_nl, grid_nl, chi_nl);

    // For each gamma: download noisy state from GPU 1 and compute overlap
    hipSetDevice(1);
    for (int g = 0; g < n_gammas; g++) {
        auto h_psi_n = download_psi(*psi_n[g], *grid_n[g], chi_n);
        // Use noiseless messages for the Bethe approximation
        // (approximation: use nl environment for both states)
        overlaps_out[g] = cpu_bethe_inner(
            h_psi_nl, h_psi_n, h_msg_nl, grid_nl, chi_nl);
    }
}
