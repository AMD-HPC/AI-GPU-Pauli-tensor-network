/*
 * main_heavyhex_gamma_mpf_ip.cpp — Gamma-MPF inner product dump
 *
 * Trotter + BP + inner products → dump to stdout for Python MPF solve.
 *
 * Output per step (parseable):
 *   STEP <n>  C_nl <val>  C_n <v1> <v2> <v3>
 *   IP_A <a00> <a01> <a02> <a11> <a12> <a22>   (upper triangle, row-major)
 *   IP_B <b0> <b1> <b2>
 *
 * Build: make PHYS_DIM=4 main_heavyhex_gamma_mpf_ip
 * Run:   ./main_heavyhex_gamma_mpf_ip [chi_max] [n_steps] [cutoff]
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "measures.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"
#include "bilinear_form_bp.hpp"

#include "heavyhex_ideal_kick_eps010_data.hpp"
#include "heavyhex_noisy_eps010_gamma1_data.hpp"
#include "heavyhex_noisy_eps010_gamma2_data.hpp"
#include "heavyhex_noisy_eps010_gamma3_data.hpp"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

static constexpr int   BP_SWEEPS = 100;
static constexpr int   N_GAMMA   = 3;

static double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

using PtmFunc = const float*(*)(int);
static const float* ptm_gamma1(int b) { return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA1_GAMMA1_0[b]; }
static const float* ptm_gamma2(int b) { return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA2_GAMMA2_0[b]; }
static const float* ptm_gamma3(int b) { return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA3_GAMMA3_0[b]; }

// ── Simulators ────────────────────────────────────────────────────────────────

struct SimNoiseless {
    rocblas_handle handle; GateOps gate_ops;
    SimNoiseless(rocblas_handle h) : handle(h), gate_ops(h) {}
    void step(HeavyHexState& state, FlatHeavyHexGrid& grid,
              Workspace& ws, int chi_max, float cutoff) {
        int nm = 2*grid.n_bonds;
        std::vector<int> frozen_chi(grid.n_bonds);
        std::vector<Scalar*> frozen_msg(nm,nullptr), new_msg(nm,nullptr);
        size_t alloc = (size_t)chi_max*chi_max*sizeof(Scalar);
        for (auto& bond:grid.bonds) for (int d=0;d<2;d++) {
            int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
            HIP_CHECK(hipMalloc(&frozen_msg[idx],alloc));
            HIP_CHECK(hipMalloc(&new_msg[idx],alloc));
        }
        for (int c=0;c<3;c++) {
            for (auto& b:grid.bonds) frozen_chi[b.id]=b.chi;
            for (auto& bond:grid.bonds) {
                int chi=bond.chi; size_t sz=(size_t)chi*chi*sizeof(Scalar);
                for (int d=0;d<2;d++) {
                    int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(frozen_msg[idx],state.messages[idx],sz,hipMemcpyDeviceToDevice));
                    HIP_CHECK(hipMemcpy(new_msg[idx],state.messages[idx],sz,hipMemcpyDeviceToDevice));
                }
            }
            for (auto& gate:grid.colors[c]) {
                const float* G=HEAVY_HEX_3x3_IDEAL_KICK_EPS010[gate.bond_id];
                gate_ops.apply_binary_gate(state.psi[gate.site_a],state.psi[gate.site_b],
                    gate,grid,new_msg,frozen_msg,frozen_chi,G,ws,chi_max,cutoff);
            }
            for (auto& gate:grid.colors[c]) {
                const Bond& bond=grid.bonds[gate.bond_id];
                int chi=bond.chi; size_t sz=(size_t)chi*chi*sizeof(Scalar);
                for (int d=0;d<2;d++) {
                    int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(state.messages[idx],new_msg[idx],sz,hipMemcpyDeviceToDevice));
                }
            }
        }
        for (auto p:frozen_msg) if(p) HIP_CHECK(hipFree(p));
        for (auto p:new_msg)    if(p) HIP_CHECK(hipFree(p));
        size_t tmp_sz=0;
        for (auto& site:grid.sites) {
            int vol=D;
            for (int d=0;d<MAX_DEGREE;d++) if(site.bond_id[d]>=0) vol*=grid.bonds[site.bond_id[d]].chi;
            tmp_sz=std::max(tmp_sz,(size_t)vol);
        }
        Scalar* tmp; HIP_CHECK(hipMalloc(&tmp,tmp_sz*sizeof(Scalar)));
        for (auto& site:grid.sites)
            gate_ops.apply_unary_gate(state.psi[site.id],site,grid,
                HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM,tmp);
        HIP_CHECK(hipFree(tmp));
    }
};

struct SimNoisy {
    rocblas_handle handle; GateOps gate_ops; PtmFunc ptm_fn;
    SimNoisy(rocblas_handle h, PtmFunc fn) : handle(h), gate_ops(h), ptm_fn(fn) {}
    void step(HeavyHexState& state, FlatHeavyHexGrid& grid,
              Workspace& ws, int chi_max, float cutoff) {
        int nm = 2*grid.n_bonds;
        std::vector<int> frozen_chi(grid.n_bonds);
        std::vector<Scalar*> frozen_msg(nm,nullptr), new_msg(nm,nullptr);
        size_t alloc = (size_t)chi_max*chi_max*sizeof(Scalar);
        for (auto& bond:grid.bonds) for (int d=0;d<2;d++) {
            int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
            HIP_CHECK(hipMalloc(&frozen_msg[idx],alloc));
            HIP_CHECK(hipMalloc(&new_msg[idx],alloc));
        }
        for (int c=0;c<3;c++) {
            for (auto& b:grid.bonds) frozen_chi[b.id]=b.chi;
            for (auto& bond:grid.bonds) {
                int chi=bond.chi; size_t sz=(size_t)chi*chi*sizeof(Scalar);
                for (int d=0;d<2;d++) {
                    int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(frozen_msg[idx],state.messages[idx],sz,hipMemcpyDeviceToDevice));
                    HIP_CHECK(hipMemcpy(new_msg[idx],state.messages[idx],sz,hipMemcpyDeviceToDevice));
                }
            }
            for (auto& gate:grid.colors[c]) {
                const float* G=ptm_fn(gate.bond_id);
                gate_ops.apply_binary_gate(state.psi[gate.site_a],state.psi[gate.site_b],
                    gate,grid,new_msg,frozen_msg,frozen_chi,G,ws,chi_max,cutoff);
            }
            for (auto& gate:grid.colors[c]) {
                const Bond& bond=grid.bonds[gate.bond_id];
                int chi=bond.chi; size_t sz=(size_t)chi*chi*sizeof(Scalar);
                for (int d=0;d<2;d++) {
                    int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(state.messages[idx],new_msg[idx],sz,hipMemcpyDeviceToDevice));
                }
            }
        }
        for (auto p:frozen_msg) if(p) HIP_CHECK(hipFree(p));
        for (auto p:new_msg)    if(p) HIP_CHECK(hipFree(p));
        size_t tmp_sz=0;
        for (auto& site:grid.sites) {
            int vol=D;
            for (int d=0;d<MAX_DEGREE;d++) if(site.bond_id[d]>=0) vol*=grid.bonds[site.bond_id[d]].chi;
            tmp_sz=std::max(tmp_sz,(size_t)vol);
        }
        Scalar* tmp; HIP_CHECK(hipMalloc(&tmp,tmp_sz*sizeof(Scalar)));
        for (auto& site:grid.sites)
            gate_ops.apply_unary_gate(state.psi[site.id],site,grid,
                HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM,tmp);
        HIP_CHECK(hipFree(tmp));
    }
};

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int   chi_max = (argc>1)?atoi(argv[1]):4;
    int   n_steps = (argc>2)?atoi(argv[2]):10;
    float cutoff  = (argc>3)?atof(argv[3]):1e-4f;
    int   center  = 2;
    int   device  = 0;

    printf("# CppSim gamma-MPF inner product dump\n");
    printf("# epsilon=%.4f  chi_max=%d  n_steps=%d  cutoff=%.0e\n",
           HEAVY_HEX_3x3_IDEAL_KICK_EPS010_EPSILON, chi_max, n_steps, (double)cutoff);
    printf("# gammas: 1.0 2.0 3.0\n");
    printf("# STEP C_nl C_n1 C_n2 C_n3  IP_A a00 a01 a02 a11 a12 a22  IP_B b0 b1 b2  chi_nl chi_n  t_trotter(s) t_ip(s)\n");
    fflush(stdout);

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    FlatHeavyHexGrid  grid_nl(chi_max);
    Workspace         ws_nl(chi_max, 3);
    HeavyHexState     state_nl(grid_nl);
    Measures          meas(handle, ws_nl);
    SimNoiseless      sim_nl(handle);
    state_nl.init_heisenberg(grid_nl, center);

    PtmFunc ptm_fns[N_GAMMA] = {ptm_gamma1, ptm_gamma2, ptm_gamma3};
    FlatHeavyHexGrid* grids_n[N_GAMMA];
    HeavyHexState*    states_n[N_GAMMA];
    Workspace*        ws_n[N_GAMMA];
    SimNoisy*         sims_n[N_GAMMA];
    for (int g=0; g<N_GAMMA; g++) {
        grids_n[g]  = new FlatHeavyHexGrid(chi_max);
        states_n[g] = new HeavyHexState(*grids_n[g]);
        ws_n[g]     = new Workspace(chi_max, 3);
        sims_n[g]   = new SimNoisy(handle, ptm_fns[g]);
        states_n[g]->init_heisenberg(*grids_n[g], center);
    }

    for (int step = 1; step <= n_steps; step++) {
        // Trotter
        double t_trotter0 = now_ms();
        sim_nl.step(state_nl, grid_nl, ws_nl, chi_max, cutoff);
        for (int g=0; g<N_GAMMA; g++)
            sims_n[g]->step(*states_n[g], *grids_n[g], *ws_n[g], chi_max, cutoff);
        double t_trotter = (now_ms() - t_trotter0) / 1000.0;

        // BP → C(t)
        float C_nl = meas.autocorrelation_with_bp(
            static_cast<Grid&>(grid_nl), state_nl.psi, state_nl.messages, center, BP_SWEEPS);
        float C_n[N_GAMMA];
        for (int g=0; g<N_GAMMA; g++)
            C_n[g] = meas.autocorrelation_with_bp(
                static_cast<Grid&>(*grids_n[g]), states_n[g]->psi,
                states_n[g]->messages, center, BP_SWEEPS);

        // Chi stats
        int chi_nl = 1, chi_n = 1;
        for (auto& b : grid_nl.bonds)      chi_nl = std::max(chi_nl, b.chi);
        for (auto& b : grids_n[0]->bonds)  chi_n  = std::max(chi_n,  b.chi);

        // Download snapshots (once per step)
        double t_ip0 = now_ms();
        StateSnapshot snap_nl, snap_n[N_GAMMA];
        snap_nl.download(static_cast<Grid&>(grid_nl),
                         state_nl.psi, state_nl.messages, device);
        for (int g=0; g<N_GAMMA; g++)
            snap_n[g].download(static_cast<Grid&>(*grids_n[g]),
                               states_n[g]->psi, states_n[g]->messages, device);

        // A[i,j] — upper triangle only (symmetric)
        float A[N_GAMMA][N_GAMMA] = {};
        for (int i=0; i<N_GAMMA; i++) {
            A[i][i] = bilinear_form_bp_cpu(snap_n[i], snap_n[i], true, 25, 1e-5f);
            for (int j=i+1; j<N_GAMMA; j++) {
                A[i][j] = bilinear_form_bp_cpu(snap_n[i], snap_n[j], true, 25, 1e-5f);
                A[j][i] = A[i][j];
            }
        }

        // B[i]
        float B[N_GAMMA] = {};
        for (int g=0; g<N_GAMMA; g++)
            B[g] = bilinear_form_bp_cpu(snap_nl, snap_n[g], true, 25, 1e-5f);

        double t_ip = (now_ms() - t_ip0) / 1000.0;

        printf("STEP %d  C_nl %+.6f  C_n %+.6f %+.6f %+.6f"
               "  IP_A %.6f %.6f %.6f %.6f %.6f %.6f"
               "  IP_B %.6f %.6f %.6f"
               "  chi %d %d  t_trotter %.2f  t_ip %.2f\n",
               step, C_nl, C_n[0], C_n[1], C_n[2],
               A[0][0], A[0][1], A[0][2], A[1][1], A[1][2], A[2][2],
               B[0], B[1], B[2],
               chi_nl, chi_n, t_trotter, t_ip);
        fflush(stdout);
    }

    rocblas_destroy_handle(handle);
    for (int g=0; g<N_GAMMA; g++) {
        delete sims_n[g]; delete ws_n[g];
        delete states_n[g]; delete grids_n[g];
    }
    return 0;
}
