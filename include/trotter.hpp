#pragma once
/*
 * trotter.hpp — Topology-agnostic single Trotter step.
 *
 * Works for any Grid subclass: heavy-hex, rectangular 2D, 1D chain,
 * honeycomb, or any graph whose bonds are colored into independent sets.
 *
 * Algorithm (same for all topologies):
 *   For each color class c = 0 .. n_colors-1:
 *     1. Snapshot current messages → frozen_msg
 *     2. Apply bond PTMs to all gates in color c (independent → parallel)
 *     3. Commit updated messages for color c
 *   Apply site kick PTM once per site.
 *
 * Parameters:
 *   state     — HeavyHexState (psi tensors + messages on GPU)
 *   grid      — any Grid subclass (colors, bonds, sites all in base class)
 *   bond_ptm  — function pointer: bond_id → const float* (16×16 PTM, 256 floats)
 *               Use nullptr for noiseless ideal gates (must provide nl_ptm_buf instead)
 *   nl_ptm    — contiguous buffer [n_bonds × 256] for noiseless PTMs (if bond_ptm==nullptr)
 *   site_ptm  — site kick PTM [16 floats], applied once per site after bond gates
 *               Use nullptr to skip the kick
 *   ws, gate_ops, chi_max, cutoff — standard simulation parameters
 *
 * Usage:
 *   // Noiseless (function pointer)
 *   do_trotter(state, grid, [](int b){return IDEAL[b];}, nullptr,
 *              SITE_PTM, ws, gate_ops, chi_max, cutoff);
 *
 *   // Noiseless (contiguous buffer)
 *   do_trotter(state, grid, nullptr, nl_ptm_buf.data(),
 *              SITE_PTM, ws, gate_ops, chi_max, cutoff);
 *
 *   // Noisy (function pointer, same kick as noiseless)
 *   do_trotter(state, grid, [](int b){return NOISY[b];}, nullptr,
 *              SITE_PTM, ws, gate_ops, chi_max, cutoff);
 */

#include "grid.hpp"
#include "gate.hpp"
#include "workspace.hpp"
#include "types.hpp"
#include <vector>
#include <hip/hip_runtime.h>

#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do { \
    hipError_t e=(cmd); \
    if(e!=hipSuccess){fprintf(stderr,"HIP %s at %s:%d\n", \
        hipGetErrorString(e),__FILE__,__LINE__);exit(1);} \
} while(0)
#endif

// Function pointer type: bond_id → 16×16 PTM (256 floats, col-major)
using PtmFn = const float*(*)(int);

// ── do_trotter ────────────────────────────────────────────────────────────────
inline void do_trotter(
    std::vector<Scalar*>&       psi,       // site tensors [n_sites], on GPU
    std::vector<Scalar*>&       messages,  // bond messages [2*n_bonds], on GPU
    Grid&                       grid,      // topology (colors, bonds, sites)
    PtmFn                       bond_ptm,  // f(bond_id)→PTM  (or nullptr)
    const float*                nl_ptm,    // contiguous PTM buffer (or nullptr)
    const float*                site_ptm,  // kick PTM [16] (or nullptr=no kick)
    Workspace&                  ws,
    GateOps&                    gate_ops,
    int                         chi_max,
    float                       cutoff)
{
    const int n_colors = (int)grid.colors.size();
    const int n_bonds  = grid.n_bonds;
    const int nm       = 2 * n_bonds;

    // Use pre-allocated Trotter buffers if available — eliminates the
    // hipMalloc/hipFree that previously fired every step and caused VRAM
    // spikes exactly when chi first saturated at chi_max, corrupting results.
    // Call ws.alloc_trotter_buffers(n_bonds) once at startup to activate.
    // Falls back to dynamic allocation if not pre-allocated.
    bool using_preallocated = (!ws.trotter_frozen_msg.empty() &&
                               (int)ws.trotter_frozen_msg.size() == nm);
    std::vector<Scalar*> frozen_msg_dyn, new_msg_dyn;
    std::vector<Scalar*>& frozen_msg = using_preallocated
                                       ? ws.trotter_frozen_msg : frozen_msg_dyn;
    std::vector<Scalar*>& new_msg    = using_preallocated
                                       ? ws.trotter_new_msg    : new_msg_dyn;
    std::vector<int> frozen_chi(n_bonds);

    if (!using_preallocated) {
        size_t alloc = (size_t)chi_max * chi_max * sizeof(Scalar);
        frozen_msg_dyn.assign(nm, nullptr);
        new_msg_dyn.assign(nm, nullptr);
        for (auto& bond : grid.bonds) for (int d = 0; d < 2; d++) {
            int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
            HIP_CHECK(hipMalloc(&frozen_msg_dyn[idx], alloc));
            HIP_CHECK(hipMalloc(&new_msg_dyn[idx],    alloc));
        }
    }

    // ── Color loop ────────────────────────────────────────────────────────────
    for (int c = 0; c < n_colors; c++) {

        // Snapshot messages
        for (auto& b : grid.bonds) frozen_chi[b.id] = b.chi;
        for (auto& bond : grid.bonds) {
            int chi = bond.chi;
            size_t sz = (size_t)chi * chi * sizeof(Scalar);
            for (int d = 0; d < 2; d++) {
                int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                HIP_CHECK(hipMemcpy(frozen_msg[idx], messages[idx],
                    sz, hipMemcpyDeviceToDevice));
                HIP_CHECK(hipMemcpy(new_msg[idx], messages[idx],
                    sz, hipMemcpyDeviceToDevice));
            }
        }

        // Apply bond gates for this color class.
        // Skip bonds where chi_alloc=1: both endpoints outside lightcone,
        // tensors are identity and need no update. Saves gate operations.
        for (auto& gate : grid.colors[c]) {
            if (grid.bonds[gate.bond_id].chi_alloc == 1) continue;
            const float* G = bond_ptm ? bond_ptm(gate.bond_id)
                                       : nl_ptm + gate.bond_id * 256;
            gate_ops.apply_binary_gate(
                psi[gate.site_a], psi[gate.site_b],
                gate, grid,
                new_msg, frozen_msg, frozen_chi,
                G, ws, chi_max, cutoff);
        }

        // Commit updated messages — skip outside-lightcone bonds
        for (auto& gate : grid.colors[c]) {
            const Bond& bond = grid.bonds[gate.bond_id];
            int chi = bond.chi;
            if (bond.chi_alloc == 1) continue;
            size_t sz = (size_t)chi * chi * sizeof(Scalar);
            for (int d = 0; d < 2; d++) {
                int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                HIP_CHECK(hipMemcpy(messages[idx], new_msg[idx],
                    sz, hipMemcpyDeviceToDevice));
            }
        }
    }

    // Only free if we dynamically allocated
    if (!using_preallocated) {
        for (auto p : frozen_msg_dyn) if (p) HIP_CHECK(hipFree(p));
        for (auto p : new_msg_dyn)    if (p) HIP_CHECK(hipFree(p));
    }

    // ── Site kick ─────────────────────────────────────────────────────────────
    if (site_ptm) {
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
            gate_ops.apply_unary_gate(psi[site.id], site, grid, site_ptm, tmp);
        HIP_CHECK(hipFree(tmp));
    }
}

// ── Convenience overload for HeavyHexState ────────────────────────────────────
// HeavyHexState has psi and messages as members — pass them directly.
#ifdef HEAVYHEX_PHYSICS_HPP   // only if heavyhex_physics.hpp included first
inline void do_trotter(
    HeavyHexState&  state,
    const Grid&     grid,
    PtmFn           bond_ptm,
    const float*    nl_ptm,
    const float*    site_ptm,
    Workspace&      ws,
    GateOps&        gate_ops,
    int             chi_max,
    float           cutoff)
{
    do_trotter(state.psi, state.messages, grid,
               bond_ptm, nl_ptm, site_ptm,
               ws, gate_ops, chi_max, cutoff);
}
#endif
