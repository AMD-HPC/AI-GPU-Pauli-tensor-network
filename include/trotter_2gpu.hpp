#pragma once
/*
 * trotter_2gpu.hpp — 2-GPU Trotter step with color-driven boundary sync.
 *
 * Splits a rectangular grid horizontally at a cut row. GPU 0 owns the top
 * half, GPU 1 owns the bottom half. The two halves share a boundary bond
 * color; before that color fires, boundary site tensors are exchanged via
 * P2P hipMemcpyPeer. Each GPU then computes the boundary bond gate normally,
 * keeping only its owned site result.
 *
 * do_trotter (trotter.hpp) is UNCHANGED. This file adds:
 *   TrotterColorCtx  — per-GPU pre-allocated color-loop context
 *   do_trotter_one_color() — apply one color on one GPU (extracted body)
 *   do_trotter_2gpu()      — orchestrate 2-GPU step with P2P sync
 *
 * Usage:
 *   // Setup (once at startup, one per GPU):
 *   TrotterColorCtx ctx0(grid0, ws0, chi_max);  // GPU 0
 *   TrotterColorCtx ctx1(grid1, ws1, chi_max);  // GPU 1
 *
 *   // Per step:
 *   do_trotter_2gpu(ctx0, ctx1, psi0, msg0, psi1, msg1,
 *                   bond_ptm, nl_ptm, site_ptm,
 *                   gate_ops0, gate_ops1,
 *                   chi_max, cutoff, cut_color,
 *                   cut_site_top, cut_site_bot,  // boundary site ids (0-based)
 *                   gpu0_id, gpu1_id);
 */

#include "grid.hpp"
#include "gate.hpp"
#include "workspace.hpp"
#include "types.hpp"
#include "trotter.hpp"
#include <vector>
#include <hip/hip_runtime.h>

#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do { \
    hipError_t e=(cmd); \
    if(e!=hipSuccess){fprintf(stderr,"HIP %s at %s:%d\n", \
        hipGetErrorString(e),__FILE__,__LINE__);exit(1);} \
} while(0)
#endif

// ── Per-GPU color-loop context ────────────────────────────────────────────────
// Pre-allocates frozen/new message buffers for one GPU's grid.
// Lifetime: same as the GPU state (create once at startup, free at end).
struct TrotterColorCtx {
    int                  n_bonds;
    int                  nm;
    std::vector<Scalar*> frozen_msg;
    std::vector<Scalar*> new_msg;
    std::vector<int>     frozen_chi;

    TrotterColorCtx(const Grid& grid, int chi_max) {
        n_bonds = grid.n_bonds;
        nm = 2 * n_bonds;
        frozen_chi.resize(n_bonds);
        size_t alloc = (size_t)chi_max * chi_max * sizeof(Scalar);
        frozen_msg.assign(nm, nullptr);
        new_msg.assign(nm, nullptr);
        for (auto& bond : grid.bonds) for (int d = 0; d < 2; d++) {
            int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
            HIP_CHECK(hipMalloc(&frozen_msg[idx], alloc));
            HIP_CHECK(hipMalloc(&new_msg[idx],    alloc));
        }
    }

    ~TrotterColorCtx() {
        for (auto p : frozen_msg) if (p) (void)hipFree(p);
        for (auto p : new_msg)    if (p) (void)hipFree(p);
    }

    // No copy
    TrotterColorCtx(const TrotterColorCtx&) = delete;
    TrotterColorCtx& operator=(const TrotterColorCtx&) = delete;
};

// ── do_trotter_one_color ──────────────────────────────────────────────────────
// Apply one color class on one GPU. Uses pre-allocated ctx buffers.
// psi and messages must be on the GPU where ctx was created.
inline void do_trotter_one_color(
    TrotterColorCtx&        ctx,
    int                     color_id,
    std::vector<Scalar*>&   psi,
    std::vector<Scalar*>&   messages,
    Grid&                   grid,
    PtmFn                   bond_ptm,
    const float*            nl_ptm,
    GateOps&                gate_ops,
    Workspace&              ws,
    int                     chi_max,
    float                   cutoff,
    LayerMetrics*           lm = nullptr)
{
    // Snapshot messages for this color
    for (auto& b : grid.bonds) ctx.frozen_chi[b.id] = b.chi;
    for (auto& bond : grid.bonds) {
        int chi = bond.chi;
        size_t sz = (size_t)chi * chi * sizeof(Scalar);
        for (int d = 0; d < 2; d++) {
            int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
            HIP_CHECK(hipMemcpy(ctx.frozen_msg[idx], messages[idx],
                sz, hipMemcpyDeviceToDevice));
            HIP_CHECK(hipMemcpy(ctx.new_msg[idx], messages[idx],
                sz, hipMemcpyDeviceToDevice));
        }
    }

    // Apply bond gates for this color
    for (auto& gate : grid.colors[color_id]) {
        if (grid.bonds[gate.bond_id].chi_alloc == 1) continue;
        const float* G = bond_ptm ? bond_ptm(gate.bond_id)
                                   : nl_ptm + gate.bond_id * 256;
        gate_ops.apply_binary_gate(
            psi[gate.site_a], psi[gate.site_b],
            gate, grid,
            ctx.new_msg, ctx.frozen_msg, ctx.frozen_chi,
            G, ws,
            chi_max, cutoff, lm, nullptr, nullptr);
    }

    // Commit updated messages
    for (auto& gate : grid.colors[color_id]) {
        const Bond& bond = grid.bonds[gate.bond_id];
        if (bond.chi_alloc == 1) continue;
        int chi = bond.chi;
        size_t sz = (size_t)chi * chi * sizeof(Scalar);
        for (int d = 0; d < 2; d++) {
            int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
            HIP_CHECK(hipMemcpy(messages[idx], ctx.new_msg[idx],
                sz, hipMemcpyDeviceToDevice));
        }
    }
}

// ── do_trotter_2gpu ───────────────────────────────────────────────────────────
// 2-GPU Trotter step. Colors before cut_color run in parallel on both GPUs.
// At cut_color: bidirectional P2P exchange of boundary sites, then both GPUs
// compute the cut color (each keeps only its owned site).
// Colors after cut_color run in parallel again.
inline void do_trotter_2gpu(
    TrotterColorCtx&        ctx0,           // GPU 0 context
    TrotterColorCtx&        ctx1,           // GPU 1 context
    std::vector<Scalar*>&   psi0,           // site tensors on GPU 0
    std::vector<Scalar*>&   messages0,      // bond messages on GPU 0
    std::vector<Scalar*>&   psi1,           // site tensors on GPU 1
    std::vector<Scalar*>&   messages1,      // bond messages on GPU 1
    Grid&                   grid0,          // top-half grid (GPU 0)
    Grid&                   grid1,          // bottom-half grid (GPU 1)
    Workspace&              ws0,            // workspace on GPU 0
    Workspace&              ws1,            // workspace on GPU 1
    PtmFn                   bond_ptm,
    const float*            nl_ptm,
    const float*            site_ptm,
    GateOps&                gate_ops0,
    GateOps&                gate_ops1,
    int                     chi_max,
    float                   cutoff,
    int                     cut_color,      // color index containing boundary bond
    int                     cut_bond_id,    // bond id of the boundary bond
    int                     cut_site_top,   // site id of boundary site on GPU 0 (0-based)
    int                     cut_site_bot,   // site id of boundary site on GPU 1 (0-based)
    int                     gpu0_id,
    int                     gpu1_id,
    LayerMetrics*           lm0 = nullptr,
    LayerMetrics*           lm1 = nullptr)
{
    const int n_colors = (int)grid0.colors.size();

    // ── Colors before cut: fully parallel ─────────────────────────────────────
    for (int c = 0; c < cut_color; c++) {
        // GPU 0 and GPU 1 run independently (no dependency)
        HIP_CHECK(hipSetDevice(gpu0_id));
        do_trotter_one_color(ctx0, c, psi0, messages0, grid0,
                             bond_ptm, nl_ptm, gate_ops0, ws0, chi_max, cutoff, lm0);
        HIP_CHECK(hipSetDevice(gpu1_id));
        do_trotter_one_color(ctx1, c, psi1, messages1, grid1,
                             bond_ptm, nl_ptm, gate_ops1, ws1, chi_max, cutoff, lm1);
    }

    // ── Bidirectional P2P boundary site exchange ───────────────────────────────
    // Only needed if the boundary bond is inside the lightcone (chi_alloc > 1).
    // If chi_alloc == 1, the gate is skipped anyway — no sync needed.
    // The lightcone allocation IS the on/off switch — free correctness.
    if (grid0.bonds[cut_bond_id].chi_alloc > 1) {
        // Site tensor size: D * product of chi over all incident bonds
        auto site_tensor_size = [](const Grid& g, int site_id) -> size_t {
            size_t sz = (size_t)D;
            for (int k = 0; k < MAX_DEGREE; k++) {
                int bid = g.sites[site_id].bond_id[k];
                if (bid >= 0) sz *= g.bonds[bid].chi_alloc;
            }
            return sz;
        };
        size_t sz_top = site_tensor_size(grid0, cut_site_top);
        size_t sz_bot = site_tensor_size(grid1, cut_site_bot);

        // GPU 0 → GPU 1: send top boundary site (ghost for V_odd on GPU 1)
        HIP_CHECK(hipSetDevice(gpu0_id));
        HIP_CHECK(hipMemcpyPeer(psi1[cut_site_bot], gpu1_id,
                                psi0[cut_site_top], gpu0_id,
                                sz_top * sizeof(Scalar)));
        // GPU 1 → GPU 0: send bottom boundary site (ghost for V_even on GPU 0)
        HIP_CHECK(hipSetDevice(gpu1_id));
        HIP_CHECK(hipMemcpyPeer(psi0[cut_site_top], gpu0_id,
                                psi1[cut_site_bot], gpu1_id,
                                sz_bot * sizeof(Scalar)));
        HIP_CHECK(hipDeviceSynchronize());
    }

    // ── Cut color: both GPUs compute, each keeps own site ─────────────────────
    HIP_CHECK(hipSetDevice(gpu0_id));
    do_trotter_one_color(ctx0, cut_color, psi0, messages0, grid0,
                         bond_ptm, nl_ptm, gate_ops0, ws0, chi_max, cutoff, lm0);
    HIP_CHECK(hipSetDevice(gpu1_id));
    do_trotter_one_color(ctx1, cut_color, psi1, messages1, grid1,
                         bond_ptm, nl_ptm, gate_ops1, ws1, chi_max, cutoff, lm1);

    // ── Colors after cut: fully parallel ──────────────────────────────────────
    for (int c = cut_color + 1; c < n_colors; c++) {
        HIP_CHECK(hipSetDevice(gpu0_id));
        do_trotter_one_color(ctx0, c, psi0, messages0, grid0,
                             bond_ptm, nl_ptm, gate_ops0, ws0, chi_max, cutoff, lm0);
        HIP_CHECK(hipSetDevice(gpu1_id));
        do_trotter_one_color(ctx1, c, psi1, messages1, grid1,
                             bond_ptm, nl_ptm, gate_ops1, ws1, chi_max, cutoff, lm1);
    }

    // ── Site kick (once per site, end of step) ─────────────────────────────────
    if (site_ptm) {
        auto do_kick = [&](int gpu, Grid& grid, std::vector<Scalar*>& psi,
                           GateOps& gops) {
            HIP_CHECK(hipSetDevice(gpu));
            size_t tmp_sz = 0;
            for (auto& site : grid.sites) {
                int vol = D;
                for (int d = 0; d < MAX_DEGREE; d++)
                    if (site.bond_id[d] >= 0)
                        vol *= grid.bonds[site.bond_id[d]].chi;
                tmp_sz = std::max(tmp_sz, (size_t)vol);
            }
            Scalar* tmp; HIP_CHECK(hipMalloc(&tmp, tmp_sz * sizeof(Scalar)));
            for (auto& site : grid.sites)
                gops.apply_unary_gate(psi[site.id], site, grid, site_ptm, tmp);
            HIP_CHECK(hipFree(tmp));
        };
        do_kick(gpu0_id, grid0, psi0, gate_ops0);
        do_kick(gpu1_id, grid1, psi1, gate_ops1);
    }
}
