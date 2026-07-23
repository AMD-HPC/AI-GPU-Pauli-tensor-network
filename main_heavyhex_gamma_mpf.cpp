/*
 * main_heavyhex_gamma_mpf.cpp — CppSim Gamma-MPF simulation
 *
 * Fixed epsilon=0.1, varying gamma=1.0/2.0/3.0 (noise amplification)
 * Noiseless reference: IDEAL_KICK at epsilon=0.1
 * Noisy states: NOISY at epsilon=0.1, gamma=1.0/2.0/3.0
 *
 * Computes C(t) for noiseless + 3 noisy states per cycle.
 * MPF polynomial fit: C_n(gamma,t) = c0 + c1*gamma + c2*gamma^2
 * Extrapolate to gamma=0 → C_mpf(t)
 *
 * Usage:
 *   ./main_heavyhex_gamma_mpf [chi_max] [n_steps] [cutoff]
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

// Noiseless reference: IDEAL_KICK at epsilon=0.1
#include "heavyhex_ideal_kick_eps010_data.hpp"
// Noisy: NOISY PTM at epsilon=0.1, gamma=1.0/2.0/3.0
#include "heavyhex_noisy_eps010_gamma1_data.hpp"
#include "heavyhex_noisy_eps010_gamma2_data.hpp"
#include "heavyhex_noisy_eps010_gamma3_data.hpp"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

static constexpr int BP_SWEEPS = 100;
static constexpr int N_GAMMA   = 3;
static constexpr float GAMMAS[N_GAMMA] = {1.0f, 2.0f, 3.0f};

static double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

// Julia ground truth (chi=70, Float64, uncapped, gamma-MPF)
// From gpu_run_simulation_tnqs_noisy_hardware_gamma.jl
static const float julia_nl[10]    = {-0.909878f,+0.920168f,-0.873560f,+0.880257f,-0.887555f,
                                        +0.878006f,-0.878092f,+0.862867f,-0.868786f,+0.870531f};

using PtmFunc = const float*(*)(int);

// PTM accessor functions per gamma
static const float* ptm_gamma1(int b) { return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA1_GAMMA1_0[b]; }
static const float* ptm_gamma2(int b) { return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA2_GAMMA2_0[b]; }
static const float* ptm_gamma3(int b) { return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA3_GAMMA3_0[b]; }

// ── Noiseless simulator (IDEAL_KICK_EPS010) ───────────────────────────────────
struct HeavyHexSimNoiseless {
    rocblas_handle handle; GateOps gate_ops;
    HeavyHexSimNoiseless(rocblas_handle h) : handle(h), gate_ops(h) {}

    int trotter_layer_bonds_only(HeavyHexState& state, FlatHeavyHexGrid& grid,
                                 Workspace& ws, int chi_max, Scalar cutoff=1e-4f) {
        int nm = 2*grid.n_bonds;
        std::vector<int> frozen_chi(grid.n_bonds);
        std::vector<Scalar*> frozen_msg(nm,nullptr), new_msg(nm,nullptr);
        size_t alloc = (size_t)chi_max*chi_max*sizeof(Scalar);
        for (auto& bond : grid.bonds) for (int d=0; d<2; d++) {
            int idx = (d==0)?bond.msg_from_a():bond.msg_from_b();
            HIP_CHECK(hipMalloc(&frozen_msg[idx],alloc));
            HIP_CHECK(hipMalloc(&new_msg[idx],alloc));
        }
        for (int c=0; c<3; c++) {
            for (auto& b : grid.bonds) frozen_chi[b.id]=b.chi;
            for (auto& bond : grid.bonds) {
                int chi=bond.chi; size_t sz=(size_t)chi*chi*sizeof(Scalar);
                for (int d=0; d<2; d++) {
                    int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(frozen_msg[idx],state.messages[idx],sz,hipMemcpyDeviceToDevice));
                    HIP_CHECK(hipMemcpy(new_msg[idx],state.messages[idx],sz,hipMemcpyDeviceToDevice));
                }
            }
            for (auto& gate : grid.colors[c]) {
                const float* G = HEAVY_HEX_3x3_IDEAL_KICK_EPS010[gate.bond_id];
                gate_ops.apply_binary_gate(state.psi[gate.site_a],state.psi[gate.site_b],
                    gate,grid,new_msg,frozen_msg,frozen_chi,G,ws,chi_max,cutoff);
            }
            for (auto& gate : grid.colors[c]) {
                const Bond& bond=grid.bonds[gate.bond_id];
                int chi=bond.chi; size_t sz=(size_t)chi*chi*sizeof(Scalar);
                for (int d=0; d<2; d++) {
                    int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(state.messages[idx],new_msg[idx],sz,hipMemcpyDeviceToDevice));
                }
            }
        }
        for (auto p:frozen_msg) if(p) HIP_CHECK(hipFree(p));
        for (auto p:new_msg)    if(p) HIP_CHECK(hipFree(p));
        int max_chi=1;
        for (auto& bond:grid.bonds) max_chi=std::max(max_chi,bond.chi);
        return max_chi;
    }

    void apply_kick(HeavyHexState& state, FlatHeavyHexGrid& grid) {
        size_t tmp_sz=0;
        for (auto& site:grid.sites) {
            int vol=D;
            for (int d=0; d<MAX_DEGREE; d++) if(site.bond_id[d]>=0) vol*=grid.bonds[site.bond_id[d]].chi;
            tmp_sz=std::max(tmp_sz,(size_t)vol);
        }
        Scalar* tmp; HIP_CHECK(hipMalloc(&tmp,tmp_sz*sizeof(Scalar)));
        for (auto& site:grid.sites)
            gate_ops.apply_unary_gate(state.psi[site.id],site,grid,
                                      HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM,tmp);
        HIP_CHECK(hipFree(tmp));
    }

    int trotter_layer(HeavyHexState& state, FlatHeavyHexGrid& grid,
                      Workspace& ws, int chi_max, Scalar cutoff=1e-4f) {
        int r=trotter_layer_bonds_only(state,grid,ws,chi_max,cutoff);
        apply_kick(state,grid); return r;
    }
};

// ── Noisy simulator parametric on gamma ──────────────────────────────────────
struct NoisySimGamma {
    rocblas_handle handle; GateOps gate_ops; PtmFunc ptm_fn;
    NoisySimGamma(rocblas_handle h, PtmFunc fn) : handle(h), gate_ops(h), ptm_fn(fn) {}

    int trotter_layer_bonds_only(HeavyHexState& state, FlatHeavyHexGrid& grid,
                                 Workspace& ws, int chi_max, Scalar cutoff=1e-4f) {
        int nm=2*grid.n_bonds;
        std::vector<int> frozen_chi(grid.n_bonds);
        std::vector<Scalar*> frozen_msg(nm,nullptr), new_msg(nm,nullptr);
        size_t alloc=(size_t)chi_max*chi_max*sizeof(Scalar);
        for (auto& bond:grid.bonds) for (int d=0; d<2; d++) {
            int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
            HIP_CHECK(hipMalloc(&frozen_msg[idx],alloc));
            HIP_CHECK(hipMalloc(&new_msg[idx],alloc));
        }
        for (int c=0; c<3; c++) {
            for (auto& b:grid.bonds) frozen_chi[b.id]=b.chi;
            for (auto& bond:grid.bonds) {
                int chi=bond.chi; size_t sz=(size_t)chi*chi*sizeof(Scalar);
                for (int d=0; d<2; d++) {
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
                for (int d=0; d<2; d++) {
                    int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(state.messages[idx],new_msg[idx],sz,hipMemcpyDeviceToDevice));
                }
            }
        }
        for (auto p:frozen_msg) if(p) HIP_CHECK(hipFree(p));
        for (auto p:new_msg)    if(p) HIP_CHECK(hipFree(p));
        int max_chi=1;
        for (auto& bond:grid.bonds) max_chi=std::max(max_chi,bond.chi);
        return max_chi;
    }

    void apply_kick(HeavyHexState& state, FlatHeavyHexGrid& grid) {
        size_t tmp_sz=0;
        for (auto& site:grid.sites) {
            int vol=D;
            for (int d=0; d<MAX_DEGREE; d++) if(site.bond_id[d]>=0) vol*=grid.bonds[site.bond_id[d]].chi;
            tmp_sz=std::max(tmp_sz,(size_t)vol);
        }
        Scalar* tmp; HIP_CHECK(hipMalloc(&tmp,tmp_sz*sizeof(Scalar)));
        for (auto& site:grid.sites)
            gate_ops.apply_unary_gate(state.psi[site.id],site,grid,
                                      HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM,tmp);
        HIP_CHECK(hipFree(tmp));
    }

    int trotter_layer(HeavyHexState& state, FlatHeavyHexGrid& grid,
                      Workspace& ws, int chi_max, Scalar cutoff=1e-4f) {
        int r=trotter_layer_bonds_only(state,grid,ws,chi_max,cutoff);
        apply_kick(state,grid); return r;
    }
};

int main(int argc, char** argv) {
    int   chi_max = (argc>1)?atoi(argv[1]):70;
    int   n_steps = (argc>2)?atoi(argv[2]):10;
    float cutoff  = (argc>3)?atof(argv[3]):1e-4f;
    int   center  = 2;

    printf("═══════════════════════════════════════════════\n");
    printf("  CppSim — Gamma-MPF  (gamma=1.0/2.0/3.0)\n");
    printf("  epsilon=%.4f  phi=%.8f  chi_max=%d  n_steps=%d\n",
           HEAVY_HEX_3x3_IDEAL_KICK_EPS010_EPSILON,
           HEAVY_HEX_3x3_IDEAL_KICK_EPS010_PHI, chi_max, n_steps);
    printf("  center=site%d\n", center);
    printf("═══════════════════════════════════════════════\n\n");

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Noiseless
    FlatHeavyHexGrid  grid_nl(chi_max);
    Workspace         ws_nl(chi_max,3);
    HeavyHexState     state_nl(grid_nl);
    Measures          meas(handle,ws_nl);
    HeavyHexSimNoiseless sim_nl(handle);
    state_nl.init_heisenberg(grid_nl,center);

    // Noisy states
    PtmFunc ptm_fns[N_GAMMA] = {ptm_gamma1, ptm_gamma2, ptm_gamma3};
    FlatHeavyHexGrid* grids_n[N_GAMMA];
    HeavyHexState*    states_n[N_GAMMA];
    Workspace*        ws_n[N_GAMMA];
    NoisySimGamma*    sims_n[N_GAMMA];
    for (int g=0; g<N_GAMMA; g++) {
        grids_n[g]  = new FlatHeavyHexGrid(chi_max);
        states_n[g] = new HeavyHexState(*grids_n[g]);
        ws_n[g]     = new Workspace(chi_max,3);
        sims_n[g]   = new NoisySimGamma(handle,ptm_fns[g]);
        states_n[g]->init_heisenberg(*grids_n[g],center);
    }

    printf("%-5s  %-10s  %-10s  %-8s  %-10s  %-8s  %-10s  %-8s  %-10s  %-8s  %-10s  %-8s  %-6s  %-6s  %4s  %6s  t(s)\n",
           "Step","C_nl","Julia_nl","|Δnl|",
           "C_n(γ=1)","|(Δ1)|","C_n(γ=2)","|(Δ2)|","C_n(γ=3)","|(Δ3)|",
           "C_mpf","|Δmpf|","chi_nl","chi_n","#sw","bp_ms");

    for (int step=1; step<=n_steps; step++) {
        double t0=now_ms();

        sim_nl.trotter_layer(state_nl,grid_nl,ws_nl,chi_max,cutoff);
        for (int g=0; g<N_GAMMA; g++)
            sims_n[g]->trotter_layer(*states_n[g],*grids_n[g],*ws_n[g],chi_max,cutoff);

        double t_bp_nl=0; int sw_nl=0; float delta_nl=0;
        float C_nl = meas.autocorrelation_with_bp(
            static_cast<Grid&>(grid_nl),state_nl.psi,state_nl.messages,center,BP_SWEEPS,
            &t_bp_nl,nullptr,&sw_nl,&delta_nl);
        float C_n[N_GAMMA];
        double t_bp_n[N_GAMMA]={0,0,0}; int sw_n[N_GAMMA]={0,0,0}; float delta_n[N_GAMMA]={0,0,0};
        for (int g=0; g<N_GAMMA; g++)
            C_n[g] = meas.autocorrelation_with_bp(
                static_cast<Grid&>(*grids_n[g]),states_n[g]->psi,
                states_n[g]->messages,center,BP_SWEEPS,
                &t_bp_n[g],nullptr,&sw_n[g],&delta_n[g]);

        int chi_nl_max=1, chi_n_max=1;
        for (auto& b:grid_nl.bonds)    chi_nl_max=std::max(chi_nl_max,b.chi);
        for (auto& b:grids_n[0]->bonds) chi_n_max=std::max(chi_n_max,b.chi);

        // MPF: C_n(gamma,t) = c0 + c1*gamma + c2*gamma^2 → extrapolate to gamma=0
        float gam2[N_GAMMA];
        for (int g=0; g<N_GAMMA; g++) gam2[g]=GAMMAS[g]*GAMMAS[g];
        float A[3][3]={
            {1.f,GAMMAS[0],gam2[0]},
            {1.f,GAMMAS[1],gam2[1]},
            {1.f,GAMMAS[2],gam2[2]}};
        float b3[3]={C_n[0],C_n[1],C_n[2]}, x[3]={0.f,0.f,0.f};
        for (int col=0; col<3; col++) {
            int pivot=col;
            for (int row=col+1; row<3; row++)
                if (fabsf(A[row][col])>fabsf(A[pivot][col])) pivot=row;
            std::swap(A[col],A[pivot]); std::swap(b3[col],b3[pivot]);
            if (fabsf(A[col][col])>1e-10f)
                for (int row=col+1; row<3; row++) {
                    float f=A[row][col]/A[col][col];
                    for (int k=col; k<3; k++) A[row][k]-=f*A[col][k];
                    b3[row]-=f*b3[col];
                }
        }
        for (int row=2; row>=0; row--) {
            x[row]=b3[row];
            for (int k=row+1; k<3; k++) x[row]-=A[row][k]*x[k];
            if (fabsf(A[row][row])>1e-10f) x[row]/=A[row][row];
        }
        float C_mpf=x[0];

        float jul_nl=(step<=10)?julia_nl[step-1]:0.f;
        printf("%-5d  %+10.6f  %+10.6f  %8.2e",step,C_nl,jul_nl,fabsf(C_nl-jul_nl));
        for (int g=0; g<N_GAMMA; g++)
            printf("  %+10.6f  %8.2e",C_n[g],fabsf(C_n[g]-C_nl));
        printf("  %+10.6f  %8.2e  %-6d  %-6d  %4d  %6.1f  %.2f\n",
               C_mpf,fabsf(C_mpf-C_nl),chi_nl_max,chi_n_max,
               sw_nl,t_bp_nl,(now_ms()-t0)/1000.0);
        fflush(stdout);
    }

    rocblas_destroy_handle(handle);
    for (int g=0; g<N_GAMMA; g++) {
        delete sims_n[g]; delete ws_n[g];
        delete states_n[g]; delete grids_n[g];
    }
    printf("\nDone.\n");
    return 0;
}
