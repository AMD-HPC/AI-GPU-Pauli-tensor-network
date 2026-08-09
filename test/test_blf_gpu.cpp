/*
 * test_blf_gpu.cpp — Validate BLF_GPU against CPU bilinear_form_bp at cycle 1.
 *
 * Runs one noiseless + one noisy gamma=1.0 step, then computes:
 *   A[0,0] = <psi_nl | psi_nl>  (expect ~1.0)
 *   A[1,1] = <psi_n1 | psi_n1>  (expect ~1.0)
 *   B[0]   = <psi_nl | psi_n1>  (Julia target: ~0.9809)
 *
 * Build: make PHYS_DIM=4 test_blf_gpu
 * Run:   ./test_blf_gpu [chi_max] [cutoff]
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "measures.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"
#include "heavyhex_ideal_kick_eps010_data.hpp"
#include "heavyhex_noisy_eps010_gamma1_data.hpp"
#include "bilinear_form_bp.hpp"

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

static const float* ptm_gamma1(int b) { return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA1_GAMMA1_0[b]; }

struct SimNoiseless {
    rocblas_handle handle; GateOps gate_ops;
    SimNoiseless(rocblas_handle h) : handle(h), gate_ops(h) {}

    void step(HeavyHexState& state, FlatHeavyHexGrid& grid, Workspace& ws,
              int chi_max, float cutoff) {
        int nm = 2 * grid.n_bonds;
        std::vector<int>     frozen_chi(grid.n_bonds);
        std::vector<Scalar*> frozen_msg(nm, nullptr), new_msg(nm, nullptr);
        size_t alloc = (size_t)chi_max * chi_max * sizeof(Scalar);
        for (auto& bond : grid.bonds) for (int d = 0; d < 2; d++) {
            int idx = (d == 0) ? bond.msg_from_a() : bond.msg_from_b();
            HIP_CHECK(hipMalloc(&frozen_msg[idx], alloc));
            HIP_CHECK(hipMalloc(&new_msg[idx],    alloc));
        }
        for (int c = 0; c < 3; c++) {
            for (auto& b : grid.bonds) frozen_chi[b.id] = b.chi;
            for (auto& bond : grid.bonds) {
                int chi = bond.chi; size_t sz = (size_t)chi * chi * sizeof(Scalar);
                for (int d = 0; d < 2; d++) {
                    int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(frozen_msg[idx], state.messages[idx], sz, hipMemcpyDeviceToDevice));
                    HIP_CHECK(hipMemcpy(new_msg[idx],    state.messages[idx], sz, hipMemcpyDeviceToDevice));
                }
            }
            for (auto& gate : grid.colors[c]) {
                const float* G = HEAVY_HEX_3x3_IDEAL_KICK_EPS010[gate.bond_id];
                gate_ops.apply_binary_gate(state.psi[gate.site_a], state.psi[gate.site_b],
                    gate, grid, new_msg, frozen_msg, frozen_chi, G, ws, chi_max, cutoff);
            }
            for (auto& gate : grid.colors[c]) {
                const Bond& bond = grid.bonds[gate.bond_id];
                int chi = bond.chi; size_t sz = (size_t)chi * chi * sizeof(Scalar);
                for (int d = 0; d < 2; d++) {
                    int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(state.messages[idx], new_msg[idx], sz, hipMemcpyDeviceToDevice));
                }
            }
        }
        for (auto p : frozen_msg) if (p) HIP_CHECK(hipFree(p));
        for (auto p : new_msg)    if (p) HIP_CHECK(hipFree(p));
        // kick
        size_t tmp_sz = 0;
        for (auto& site : grid.sites) {
            int vol = D;
            for (int d = 0; d < MAX_DEGREE; d++) if (site.bond_id[d] >= 0) vol *= grid.bonds[site.bond_id[d]].chi;
            tmp_sz = std::max(tmp_sz, (size_t)vol);
        }
        Scalar* tmp; HIP_CHECK(hipMalloc(&tmp, tmp_sz * sizeof(Scalar)));
        for (auto& site : grid.sites)
            gate_ops.apply_unary_gate(state.psi[site.id], site, grid,
                                      HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM, tmp);
        HIP_CHECK(hipFree(tmp));
    }
};

struct SimNoisy {
    rocblas_handle handle; GateOps gate_ops;
    SimNoisy(rocblas_handle h) : handle(h), gate_ops(h) {}

    void step(HeavyHexState& state, FlatHeavyHexGrid& grid, Workspace& ws,
              int chi_max, float cutoff) {
        int nm = 2 * grid.n_bonds;
        std::vector<int>     frozen_chi(grid.n_bonds);
        std::vector<Scalar*> frozen_msg(nm, nullptr), new_msg(nm, nullptr);
        size_t alloc = (size_t)chi_max * chi_max * sizeof(Scalar);
        for (auto& bond : grid.bonds) for (int d = 0; d < 2; d++) {
            int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
            HIP_CHECK(hipMalloc(&frozen_msg[idx], alloc));
            HIP_CHECK(hipMalloc(&new_msg[idx],    alloc));
        }
        for (int c = 0; c < 3; c++) {
            for (auto& b : grid.bonds) frozen_chi[b.id] = b.chi;
            for (auto& bond : grid.bonds) {
                int chi = bond.chi; size_t sz = (size_t)chi * chi * sizeof(Scalar);
                for (int d = 0; d < 2; d++) {
                    int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(frozen_msg[idx], state.messages[idx], sz, hipMemcpyDeviceToDevice));
                    HIP_CHECK(hipMemcpy(new_msg[idx],    state.messages[idx], sz, hipMemcpyDeviceToDevice));
                }
            }
            for (auto& gate : grid.colors[c]) {
                const float* G = ptm_gamma1(gate.bond_id);
                gate_ops.apply_binary_gate(state.psi[gate.site_a], state.psi[gate.site_b],
                    gate, grid, new_msg, frozen_msg, frozen_chi, G, ws, chi_max, cutoff);
            }
            for (auto& gate : grid.colors[c]) {
                const Bond& bond = grid.bonds[gate.bond_id];
                int chi = bond.chi; size_t sz = (size_t)chi * chi * sizeof(Scalar);
                for (int d = 0; d < 2; d++) {
                    int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(state.messages[idx], new_msg[idx], sz, hipMemcpyDeviceToDevice));
                }
            }
        }
        for (auto p : frozen_msg) if (p) HIP_CHECK(hipFree(p));
        for (auto p : new_msg)    if (p) HIP_CHECK(hipFree(p));
        // kick
        size_t tmp_sz = 0;
        for (auto& site : grid.sites) {
            int vol = D;
            for (int d = 0; d < MAX_DEGREE; d++) if (site.bond_id[d] >= 0) vol *= grid.bonds[site.bond_id[d]].chi;
            tmp_sz = std::max(tmp_sz, (size_t)vol);
        }
        Scalar* tmp; HIP_CHECK(hipMalloc(&tmp, tmp_sz * sizeof(Scalar)));
        for (auto& site : grid.sites)
            gate_ops.apply_unary_gate(state.psi[site.id], site, grid,
                                      HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM, tmp);
        HIP_CHECK(hipFree(tmp));
    }
};

int main(int argc, char** argv) {
    int   chi_max = (argc > 1) ? atoi(argv[1]) : 4;
    float cutoff  = (argc > 2) ? atof(argv[2]) : 1e-4f;
    int   center  = 2;

    printf("=== BLF_GPU cycle-1 test  chi=%d cutoff=%.0e ===\n", chi_max, (double)cutoff);

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // ── Noiseless state ──────────────────────────────────────────────
    FlatHeavyHexGrid grid_nl(chi_max);
    Workspace        ws_nl(chi_max, 3);
    HeavyHexState    state_nl(grid_nl);
    Measures         meas(handle, ws_nl);
    SimNoiseless     sim_nl(handle);

    state_nl.init_heisenberg(grid_nl, center);
    sim_nl.step(state_nl, grid_nl, ws_nl, chi_max, cutoff);

    float C_nl = meas.autocorrelation_with_bp(
        static_cast<Grid&>(grid_nl), state_nl.psi, state_nl.messages, center, 100);
    printf("C_nl(1) = %.6f\n", C_nl);

    // ── Noisy gamma=1.0 state ────────────────────────────────────────
    FlatHeavyHexGrid grid_n1(chi_max);
    Workspace        ws_n1(chi_max, 3);
    HeavyHexState    state_n1(grid_n1);
    SimNoisy         sim_n1(handle);

    state_n1.init_heisenberg(grid_n1, center);
    sim_n1.step(state_n1, grid_n1, ws_n1, chi_max, cutoff);

    float C_n1 = meas.autocorrelation_with_bp(
        static_cast<Grid&>(grid_n1), state_n1.psi, state_n1.messages, center, 100);
    printf("C_n1(1) = %.6f\n", C_n1);

    // ── CPU inner products via bilinear_form_bp_cpu ──────────────────
    int device = 0;

    StateSnapshot snap_nl, snap_n1;
    snap_nl.download(static_cast<Grid&>(grid_nl), state_nl.psi, state_nl.messages, device);
    snap_n1.download(static_cast<Grid&>(grid_n1), state_n1.psi, state_n1.messages, device);

    // A[0,0] = <nl|nl>
    float A00 = bilinear_form_bp_cpu(snap_nl, snap_nl, true, 25, 1e-5f, true);
    printf("A[0,0] = <nl|nl> = %.6f  (expect ~1.0)\n", A00);

    // A[1,1] = <n1|n1>
    float A11 = bilinear_form_bp_cpu(snap_n1, snap_n1, true, 25, 1e-5f, true);
    printf("A[1,1] = <n1|n1> = %.6f  (expect ~1.0)\n", A11);

    // B[0] = <nl|n1>
    float B0 = bilinear_form_bp_cpu(snap_nl, snap_n1, true, 25, 1e-5f, true);
    printf("B[0]   = <nl|n1> = %.6f  (Julia target: ~0.9809)\n", B0);

    rocblas_destroy_handle(handle);
    return 0;
}
