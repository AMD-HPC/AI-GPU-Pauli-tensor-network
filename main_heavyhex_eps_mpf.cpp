/*
 * main_heavyhex_eps_mpf.cpp — CppSim epsilon-MPF simulation
 *
 * Matches Julia's epsilon-MPF formulation:
 *   - Noiseless: IDEAL_KICK at epsilon=0.1, phi=1.41371669
 *   - Noisy:     NOISY at epsilon=0.1, 0.15, 0.2, gamma=1.0
 *
 * Computes C(t) = Tr(Z_center(t) rho) per cycle for noiseless + 3 noisy states.
 * Compare output with Julia Z(t) from results/MI100/results_all.csv
 *
 * Julia ground truth (chi=70, Float64, uncapped):
 *   noiseless: [-0.909878, +0.920168, -0.873560, +0.880257, ...]
 *   noisy 0.1: [-0.869887, +0.825904, -0.742851, +0.714086, ...]
 *
 * Usage:
 *   ./main_heavyhex_eps_mpf [chi_max] [n_steps] [cutoff]
 * Defaults:
 *   chi_max=70  n_steps=10  cutoff=1e-4
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "measures.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"

// Epsilon-specific HPPs
// Noiseless: eps=0.10, K=1 (one full step per cycle)
#include "heavyhex_ideal_kick_eps010_data.hpp"
// Noisy: eps=0.10, 0.15, 0.20, gamma=1.0
#include "heavyhex_noisy_eps010_data.hpp"
#include "heavyhex_noisy_eps015_data.hpp"
#include "heavyhex_noisy_eps020_data.hpp"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

static constexpr int BP_SWEEPS = 100;

static double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

// ── Noiseless simulator using IDEAL_KICK_EPS010 ───────────────────────────────
struct HeavyHexSimEps010 {
    rocblas_handle handle;
    GateOps gate_ops;
    HeavyHexSimEps010(rocblas_handle h) : handle(h), gate_ops(h) {}

    int trotter_layer_bonds_only(HeavyHexState& state, FlatHeavyHexGrid& grid,
                                 Workspace& ws, int chi_max, Scalar cutoff=1e-4f)
    {
        int nm = 2 * grid.n_bonds;
        std::vector<int>     frozen_chi(grid.n_bonds);
        std::vector<Scalar*> frozen_msg(nm, nullptr), new_msg(nm, nullptr);
        size_t alloc = (size_t)chi_max * chi_max * sizeof(Scalar);
        for (auto& bond : grid.bonds) for (int d=0; d<2; d++) {
            int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
            HIP_CHECK(hipMalloc(&frozen_msg[idx], alloc));
            HIP_CHECK(hipMalloc(&new_msg[idx],    alloc));
        }
        for (int c = 0; c < 3; c++) {
            for (auto& b : grid.bonds) frozen_chi[b.id] = b.chi;
            for (auto& bond : grid.bonds) {
                int chi = bond.chi; size_t sz = (size_t)chi*chi*sizeof(Scalar);
                for (int d=0; d<2; d++) {
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
                int chi = bond.chi; size_t sz = (size_t)chi*chi*sizeof(Scalar);
                for (int d=0; d<2; d++) {
                    int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(state.messages[idx], new_msg[idx], sz, hipMemcpyDeviceToDevice));
                }
            }
        }
        for (auto p : frozen_msg) if(p) HIP_CHECK(hipFree(p));
        for (auto p : new_msg)    if(p) HIP_CHECK(hipFree(p));
        int max_chi = 1;
        for (auto& bond : grid.bonds) max_chi = std::max(max_chi, bond.chi);
        return max_chi;
    }

    void apply_kick(HeavyHexState& state, FlatHeavyHexGrid& grid)
    {
        size_t tmp_sz = 0;
        for (auto& site : grid.sites) {
            int vol = D;
            for (int d=0; d<MAX_DEGREE; d++) if(site.bond_id[d]>=0) vol *= grid.bonds[site.bond_id[d]].chi;
            tmp_sz = std::max(tmp_sz, (size_t)vol);
        }
        Scalar* tmp; HIP_CHECK(hipMalloc(&tmp, tmp_sz*sizeof(Scalar)));
        for (auto& site : grid.sites)
            gate_ops.apply_unary_gate(state.psi[site.id], site, grid,
                                      HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM, tmp);
        HIP_CHECK(hipFree(tmp));
    }

    int trotter_layer(HeavyHexState& state, FlatHeavyHexGrid& grid,
                      Workspace& ws, int chi_max, Scalar cutoff=1e-4f)
    {
        int r = trotter_layer_bonds_only(state, grid, ws, chi_max, cutoff);
        apply_kick(state, grid);
        return r;
    }
};

// ── Noisy simulator parametric on epsilon HPP ─────────────────────────────────
// We use a function pointer to select the PTM per epsilon
using PtmFunc = const float*(*)(int);

struct NoisySimEps {
    rocblas_handle handle;
    GateOps gate_ops;
    PtmFunc ptm_fn;
    const float* site_ptm;
    NoisySimEps(rocblas_handle h, PtmFunc fn, const float* sptm)
        : handle(h), gate_ops(h), ptm_fn(fn), site_ptm(sptm) {}

    int trotter_layer_bonds_only(HeavyHexState& state, FlatHeavyHexGrid& grid,
                                 Workspace& ws, int chi_max, Scalar cutoff=1e-4f)
    {
        int nm = 2 * grid.n_bonds;
        std::vector<int>     frozen_chi(grid.n_bonds);
        std::vector<Scalar*> frozen_msg(nm, nullptr), new_msg(nm, nullptr);
        size_t alloc = (size_t)chi_max * chi_max * sizeof(Scalar);
        for (auto& bond : grid.bonds) for (int d=0; d<2; d++) {
            int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
            HIP_CHECK(hipMalloc(&frozen_msg[idx], alloc));
            HIP_CHECK(hipMalloc(&new_msg[idx],    alloc));
        }
        for (int c = 0; c < 3; c++) {
            for (auto& b : grid.bonds) frozen_chi[b.id] = b.chi;
            for (auto& bond : grid.bonds) {
                int chi = bond.chi; size_t sz = (size_t)chi*chi*sizeof(Scalar);
                for (int d=0; d<2; d++) {
                    int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(frozen_msg[idx], state.messages[idx], sz, hipMemcpyDeviceToDevice));
                    HIP_CHECK(hipMemcpy(new_msg[idx],    state.messages[idx], sz, hipMemcpyDeviceToDevice));
                }
            }
            for (auto& gate : grid.colors[c]) {
                const float* G = ptm_fn(gate.bond_id);
                gate_ops.apply_binary_gate(state.psi[gate.site_a], state.psi[gate.site_b],
                    gate, grid, new_msg, frozen_msg, frozen_chi, G, ws, chi_max, cutoff);
            }
            for (auto& gate : grid.colors[c]) {
                const Bond& bond = grid.bonds[gate.bond_id];
                int chi = bond.chi; size_t sz = (size_t)chi*chi*sizeof(Scalar);
                for (int d=0; d<2; d++) {
                    int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(state.messages[idx], new_msg[idx], sz, hipMemcpyDeviceToDevice));
                }
            }
        }
        for (auto p : frozen_msg) if(p) HIP_CHECK(hipFree(p));
        for (auto p : new_msg)    if(p) HIP_CHECK(hipFree(p));
        int max_chi = 1;
        for (auto& bond : grid.bonds) max_chi = std::max(max_chi, bond.chi);
        return max_chi;
    }

    void apply_kick(HeavyHexState& state, FlatHeavyHexGrid& grid)
    {
        size_t tmp_sz = 0;
        for (auto& site : grid.sites) {
            int vol = D;
            for (int d=0; d<MAX_DEGREE; d++) if(site.bond_id[d]>=0) vol *= grid.bonds[site.bond_id[d]].chi;
            tmp_sz = std::max(tmp_sz, (size_t)vol);
        }
        Scalar* tmp; HIP_CHECK(hipMalloc(&tmp, tmp_sz*sizeof(Scalar)));
        for (auto& site : grid.sites)
            gate_ops.apply_unary_gate(state.psi[site.id], site, grid, site_ptm, tmp);
        HIP_CHECK(hipFree(tmp));
    }

    int trotter_layer(HeavyHexState& state, FlatHeavyHexGrid& grid,
                      Workspace& ws, int chi_max, Scalar cutoff=1e-4f)
    {
        int r = trotter_layer_bonds_only(state, grid, ws, chi_max, cutoff);
        apply_kick(state, grid);
        return r;
    }
};

// PTM accessor functions for each epsilon
static const float* ptm_eps010(int bond_id) { return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA1_0[bond_id]; }
static const float* ptm_eps015(int bond_id) { return HEAVY_HEX_3x3_NOISY_EPS015_GAMMA1_0[bond_id]; }
static const float* ptm_eps020(int bond_id) { return HEAVY_HEX_3x3_NOISY_EPS020_GAMMA1_0[bond_id]; }

int main(int argc, char** argv) {
    int   chi_max = (argc>1) ? atoi(argv[1]) : 70;
    int   n_steps = (argc>2) ? atoi(argv[2]) : 10;
    float cutoff  = (argc>3) ? atof(argv[3]) : 1e-4f;
    int   K       = (argc>4) ? atoi(argv[4]) : 1;  // sub-steps per cycle (k=1: one PTM, k=2: two half-steps)
    int   center  = 2;

    printf("═══════════════════════════════════════════════\n");
    printf("  CppSim — Epsilon-MPF  (epsilon=0.1/0.15/0.2)\n");
    printf("  phi=%.8f  chi_max=%d  n_steps=%d  K=%d\n",
           HEAVY_HEX_3x3_IDEAL_KICK_EPS010_PHI, chi_max, n_steps, K);
    printf("  center=site%d\n", center);
    printf("═══════════════════════════════════════════════\n\n");

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // ── Noiseless ─────────────────────────────────────────────────────────────
    FlatHeavyHexGrid grid_nl(chi_max);
    Workspace        ws_nl(chi_max, 3);
    HeavyHexState    state_nl(grid_nl);
    Measures         meas(handle, ws_nl);
    HeavyHexSimEps010 sim_nl(handle);
    state_nl.init_heisenberg(grid_nl, center);

    // ── Noisy states ──────────────────────────────────────────────────────────
    static constexpr int N_EPS = 3;
    static constexpr float EPSILONS[N_EPS] = {0.1f, 0.15f, 0.2f};
    PtmFunc ptm_fns[N_EPS] = {ptm_eps010, ptm_eps015, ptm_eps020};
    // Kick PTM from noiseless HPP (phi=1.41371669, independent of epsilon)
    const float* site_ptms[N_EPS] = {
        HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM,
        HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM,
        HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM
    };

    FlatHeavyHexGrid* grids_n[N_EPS];
    HeavyHexState*    states_n[N_EPS];
    Workspace*        ws_n[N_EPS];
    NoisySimEps*      sims_n[N_EPS];
    for (int e = 0; e < N_EPS; e++) {
        grids_n[e]  = new FlatHeavyHexGrid(chi_max);
        states_n[e] = new HeavyHexState(*grids_n[e]);
        ws_n[e]     = new Workspace(chi_max, 3);
        sims_n[e]   = new NoisySimEps(handle, ptm_fns[e], site_ptms[e]);
        states_n[e]->init_heisenberg(*grids_n[e], center);
    }

    // ── Julia ground truth (chi=70, Float64, uncapped) ────────────────────────
    static const float julia_nl[10]   = {-0.909878f,+0.920168f,-0.873560f,+0.880257f,-0.887555f,
                                          +0.878006f,-0.878092f,+0.862867f,-0.868786f,+0.870531f};
    static const float julia_n[3][10] = {
        {-0.869887f,+0.825904f,-0.742851f,+0.714086f,-0.667969f,  // ε=0.1
         +0.630084f,-0.597785f,+0.552941f,-0.524737f,+0.490561f},
        {-0.813765f,+0.834621f,-0.734775f,+0.674611f,-0.627394f,  // ε=0.15
         +0.576392f,-0.537014f,+0.486853f,-0.460132f,+0.428229f},
        {-0.738583f,+0.782922f,-0.647174f,+0.587380f,-0.530884f,  // ε=0.2
         +0.463703f,-0.430853f,+0.376932f,-0.346968f,+0.315956f}
    };

    // ── Header ────────────────────────────────────────────────────────────────
    printf("%-5s  %-10s  %-10s  %-8s  %-10s  %-8s  %-10s  %-8s  %-10s  %-8s  %-10s  %-8s  %-6s  %-6s  t(s)\n",
           "Step","C_nl","Julia_nl","|Δnl|",
           "C_n(0.1)","|Δ0.1|","C_n(0.15)","|Δ0.15|","C_n(0.2)","|Δ0.2|",
           "C_mpf","|Δmpf|","chi_nl","chi_n");

    for (int step = 1; step <= n_steps; step++) {
        double t0 = now_ms();

        // Apply K bond-gate sub-steps then one kick per physical cycle
        for (int sub = 0; sub < K; sub++)
            sim_nl.trotter_layer_bonds_only(state_nl, grid_nl, ws_nl, chi_max, cutoff);
        sim_nl.apply_kick(state_nl, grid_nl);
        for (int e = 0; e < N_EPS; e++) {
            for (int sub = 0; sub < K; sub++)
                sims_n[e]->trotter_layer_bonds_only(*states_n[e], *grids_n[e], *ws_n[e], chi_max, cutoff);
            sims_n[e]->apply_kick(*states_n[e], *grids_n[e]);
        }

        float C_nl = meas.autocorrelation_with_bp(
            static_cast<Grid&>(grid_nl), state_nl.psi, state_nl.messages,
            center, BP_SWEEPS);

        float C_n[N_EPS];
        for (int e = 0; e < N_EPS; e++)
            C_n[e] = meas.autocorrelation_with_bp(
                static_cast<Grid&>(*grids_n[e]), states_n[e]->psi,
                states_n[e]->messages, center, BP_SWEEPS);

        int chi_nl_max = 1, chi_n_max = 1;
        for (auto& b : grid_nl.bonds)     chi_nl_max = std::max(chi_nl_max, b.chi);
        for (auto& b : grids_n[0]->bonds) chi_n_max  = std::max(chi_n_max,  b.chi);

        // ── MPF extrapolation: fit C_n(ε,t) = c0 + c1·ε² + c2·ε⁴ ──────────
        // Three points (ε=0.1, 0.15, 0.2) → solve 3×3 Vandermonde in ε²
        float eps2[N_EPS], eps4[N_EPS];
        for (int e = 0; e < N_EPS; e++) {
            eps2[e] = EPSILONS[e] * EPSILONS[e];
            eps4[e] = eps2[e] * eps2[e];
        }
        // Build 3×3 system A·x = b where x = [c0, c1, c2]
        float A[3][3] = {
            {1.f, eps2[0], eps4[0]},
            {1.f, eps2[1], eps4[1]},
            {1.f, eps2[2], eps4[2]}
        };
        float b3[3] = {C_n[0], C_n[1], C_n[2]};
        float x[3] = {0.f, 0.f, 0.f};
        // Gaussian elimination
        for (int col = 0; col < 3; col++) {
            int pivot = col;
            for (int row = col+1; row < 3; row++)
                if (fabsf(A[row][col]) > fabsf(A[pivot][col])) pivot = row;
            for (int k = 0; k < 3; k++) { float tmp=A[col][k]; A[col][k]=A[pivot][k]; A[pivot][k]=tmp; }
            float tmp=b3[col]; b3[col]=b3[pivot]; b3[pivot]=tmp;
            if (fabsf(A[col][col]) > 1e-10f)
                for (int row = col+1; row < 3; row++) {
                    float f = A[row][col] / A[col][col];
                    for (int k = col; k < 3; k++) A[row][k] -= f * A[col][k];
                    b3[row] -= f * b3[col];
                }
        }
        for (int row = 2; row >= 0; row--) {
            x[row] = b3[row];
            for (int k = row+1; k < 3; k++) x[row] -= A[row][k] * x[k];
            if (fabsf(A[row][row]) > 1e-10f) x[row] /= A[row][row];
        }
        float C_mpf = x[0];  // c0 = extrapolated value at ε=0

        float jul_nl = (step<=10) ? julia_nl[step-1] : 0.f;
        printf("%-5d  %+10.6f  %+10.6f  %8.2e",
               step, C_nl, jul_nl, fabsf(C_nl-jul_nl));
        for (int e = 0; e < N_EPS; e++) {
            float jul = (step<=10) ? julia_n[e][step-1] : 0.f;
            printf("  %+10.6f  %8.2e", C_n[e], fabsf(C_n[e]-jul));
        }
        printf("  %+10.6f  %8.2e  %-6d  %-6d  %.2f\n",
               C_mpf, fabsf(C_mpf-C_nl), chi_nl_max, chi_n_max, (now_ms()-t0)/1000.0);
        fflush(stdout);
    }

    // Cleanup
    rocblas_destroy_handle(handle);
    for (int e = 0; e < N_EPS; e++) {
        delete sims_n[e]; delete ws_n[e];
        delete states_n[e]; delete grids_n[e];
    }
    printf("\nDone.\n");
    return 0;
}
