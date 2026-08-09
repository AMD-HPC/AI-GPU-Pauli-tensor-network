/*
 * main_heavyhex_gamma_mpf_DG_2gpu.cpp — Gamma-MPF with DG gauge inner products
 *
 * GPU 0: ALL Trotter (noiseless + 3 noisy) + BP → C(t)
 * GPU 1: DG inner products (A matrix + B vector) via DG_StateGauge + DG_Pair
 *
 * Pipeline (same overlap as QR 2gpu):
 *   GPU 0: Trotter step N+1
 *   GPU 1: DG precompute (4 streams) + pair contractions (9 streams) for step N
 *   P2P:   ψ, messages copied GPU0→GPU1 at precompute time
 *   CPU:   collect 9 floats → print
 *
 * DG gauge: eigendecompose M_a per bond (rocsolver_ssyevd), absorb sqrt(D)*A^T
 * into psi_a and A^T into psi_b, dot product per site — no BP sweeps.
 * 4 eigendecompositions shared across 9 pairs (2.25× fewer syevd calls than QR).
 *
 * Build: make PHYS_DIM=4 main_heavyhex_gamma_mpf_DG_2gpu
 *        make PHYS_DIM=4 DFLAGS=-DERGODIC2 main_heavyhex_gamma_mpf_DG_2gpu
 * Run:   ./main_heavyhex_gamma_mpf_DG_2gpu [chi_max] [n_steps] [cutoff]
 *
 * Output format: same as main_heavyhex_gamma_mpf_2gpu
 *   STEP N  C_nl ...  C_n ...  IP_A ...  IP_B ...  chi ...
 *           t_cycle t_trotter_pure t_ip_wall t_precompute
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "measures.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"
#include "diag_gauge_ip.hpp"

#include "heavyhex_ideal_kick_eps010_data.hpp"

#ifdef ERGODIC2
#include "heavyhex_ideal_kick_eps010_gamma2_data.hpp"
#include "heavyhex_ideal_kick_eps015_gamma2_data.hpp"
#include "heavyhex_ideal_kick_eps020_gamma2_data.hpp"
static const float* ptm_n1(int b){return HEAVY_HEX_3x3_IDEAL_KICK_EPS010_GAMMA2[b];}
static const float* ptm_n2(int b){return HEAVY_HEX_3x3_IDEAL_KICK_EPS015_GAMMA2[b];}
static const float* ptm_n3(int b){return HEAVY_HEX_3x3_IDEAL_KICK_EPS020_GAMMA2[b];}
static const char* MODE_STR = "ergodic  eps=0.10/0.15/0.20  gamma=2.0";
#elif defined(ERGODIC)
#include "heavyhex_ideal_kick_eps010_gamma1_data.hpp"
#include "heavyhex_ideal_kick_eps015_gamma1_data.hpp"
#include "heavyhex_ideal_kick_eps020_gamma1_data.hpp"
static const float* ptm_n1(int b){return HEAVY_HEX_3x3_IDEAL_KICK_EPS010_GAMMA1[b];}
static const float* ptm_n2(int b){return HEAVY_HEX_3x3_IDEAL_KICK_EPS015_GAMMA1[b];}
static const float* ptm_n3(int b){return HEAVY_HEX_3x3_IDEAL_KICK_EPS020_GAMMA1[b];}
static const char* MODE_STR = "ergodic  eps=0.10/0.15/0.20  gamma=1.0";
#else
#include "heavyhex_noisy_eps010_gamma1_data.hpp"
#include "heavyhex_noisy_eps010_gamma2_data.hpp"
#include "heavyhex_noisy_eps010_gamma3_data.hpp"
static const float* ptm_n1(int b){return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA1_GAMMA1_0[b];}
static const float* ptm_n2(int b){return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA2_GAMMA2_0[b];}
static const float* ptm_n3(int b){return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA3_GAMMA3_0[b];}
static const char* MODE_STR = "gamma-MPF  gamma=1.0/2.0/3.0  eps=0.10";
#endif

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <atomic>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

static constexpr int BP_SWEEPS = 100;
static constexpr int N_GAMMA   = 3;

static double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

using PtmFunc = const float*(*)(int);

static void enable_p2p(int n_gpus) {
    for (int i=0;i<n_gpus;i++)
        for (int j=0;j<n_gpus;j++)
            if(i!=j) hipDeviceEnablePeerAccess(j,0);
}

// ── SimNoiseless / SimNoisy (same as 2gpu binary) ─────────────────────────────
struct SimNoiseless {
    rocblas_handle handle; GateOps gate_ops;
    SimNoiseless(rocblas_handle h):handle(h),gate_ops(h){}
    void step(HeavyHexState& state,FlatHeavyHexGrid& grid,
              Workspace& ws,int chi_max,float cutoff){
        int nm=2*grid.n_bonds;
        std::vector<int> frozen_chi(grid.n_bonds);
        std::vector<Scalar*> frozen_msg(nm,nullptr),new_msg(nm,nullptr);
        size_t alloc=(size_t)chi_max*chi_max*sizeof(Scalar);
        for(auto& bond:grid.bonds) for(int d=0;d<2;d++){
            int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
            HIP_CHECK(hipMalloc(&frozen_msg[idx],alloc));
            HIP_CHECK(hipMalloc(&new_msg[idx],alloc));
        }
        for(int c=0;c<3;c++){
            for(auto& b:grid.bonds) frozen_chi[b.id]=b.chi;
            for(auto& bond:grid.bonds){
                int chi=bond.chi; size_t sz=(size_t)chi*chi*sizeof(Scalar);
                for(int d=0;d<2;d++){
                    int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(frozen_msg[idx],state.messages[idx],sz,hipMemcpyDeviceToDevice));
                    HIP_CHECK(hipMemcpy(new_msg[idx],state.messages[idx],sz,hipMemcpyDeviceToDevice));
                }
            }
            for(auto& gate:grid.colors[c]){
                const float* G=HEAVY_HEX_3x3_IDEAL_KICK_EPS010[gate.bond_id];
                gate_ops.apply_binary_gate(state.psi[gate.site_a],state.psi[gate.site_b],
                    gate,grid,new_msg,frozen_msg,frozen_chi,G,ws,chi_max,cutoff);
            }
            for(auto& gate:grid.colors[c]){
                const Bond& bond=grid.bonds[gate.bond_id];
                int chi=bond.chi; size_t sz=(size_t)chi*chi*sizeof(Scalar);
                for(int d=0;d<2;d++){
                    int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(state.messages[idx],new_msg[idx],sz,hipMemcpyDeviceToDevice));
                }
            }
        }
        for(auto p:frozen_msg) if(p) HIP_CHECK(hipFree(p));
        for(auto p:new_msg)    if(p) HIP_CHECK(hipFree(p));
        size_t tsz=0;
        for(auto& s:grid.sites){int vol=D;for(int d=0;d<MAX_DEGREE;d++)if(s.bond_id[d]>=0)vol*=grid.bonds[s.bond_id[d]].chi;tsz=std::max(tsz,(size_t)vol);}
        Scalar* tmp; HIP_CHECK(hipMalloc(&tmp,tsz*sizeof(Scalar)));
        for(auto& s:grid.sites)
            gate_ops.apply_unary_gate(state.psi[s.id],s,grid,HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM,tmp);
        HIP_CHECK(hipFree(tmp));
    }
};

struct SimNoisy {
    rocblas_handle handle; GateOps gate_ops; PtmFunc ptm_fn;
    SimNoisy(rocblas_handle h,PtmFunc fn):handle(h),gate_ops(h),ptm_fn(fn){}
    void step(HeavyHexState& state,FlatHeavyHexGrid& grid,
              Workspace& ws,int chi_max,float cutoff){
        int nm=2*grid.n_bonds;
        std::vector<int> frozen_chi(grid.n_bonds);
        std::vector<Scalar*> frozen_msg(nm,nullptr),new_msg(nm,nullptr);
        size_t alloc=(size_t)chi_max*chi_max*sizeof(Scalar);
        for(auto& bond:grid.bonds) for(int d=0;d<2;d++){
            int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
            HIP_CHECK(hipMalloc(&frozen_msg[idx],alloc));
            HIP_CHECK(hipMalloc(&new_msg[idx],alloc));
        }
        for(int c=0;c<3;c++){
            for(auto& b:grid.bonds) frozen_chi[b.id]=b.chi;
            for(auto& bond:grid.bonds){
                int chi=bond.chi; size_t sz=(size_t)chi*chi*sizeof(Scalar);
                for(int d=0;d<2;d++){
                    int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(frozen_msg[idx],state.messages[idx],sz,hipMemcpyDeviceToDevice));
                    HIP_CHECK(hipMemcpy(new_msg[idx],state.messages[idx],sz,hipMemcpyDeviceToDevice));
                }
            }
            for(auto& gate:grid.colors[c]){
                const float* G=ptm_fn(gate.bond_id);
                gate_ops.apply_binary_gate(state.psi[gate.site_a],state.psi[gate.site_b],
                    gate,grid,new_msg,frozen_msg,frozen_chi,G,ws,chi_max,cutoff);
            }
            for(auto& gate:grid.colors[c]){
                const Bond& bond=grid.bonds[gate.bond_id];
                int chi=bond.chi; size_t sz=(size_t)chi*chi*sizeof(Scalar);
                for(int d=0;d<2;d++){
                    int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(state.messages[idx],new_msg[idx],sz,hipMemcpyDeviceToDevice));
                }
            }
        }
        for(auto p:frozen_msg) if(p) HIP_CHECK(hipFree(p));
        for(auto p:new_msg)    if(p) HIP_CHECK(hipFree(p));
        size_t tsz=0;
        for(auto& s:grid.sites){int vol=D;for(int d=0;d<MAX_DEGREE;d++)if(s.bond_id[d]>=0)vol*=grid.bonds[s.bond_id[d]].chi;tsz=std::max(tsz,(size_t)vol);}
        Scalar* tmp; HIP_CHECK(hipMalloc(&tmp,tsz*sizeof(Scalar)));
        for(auto& s:grid.sites)
            gate_ops.apply_unary_gate(state.psi[s.id],s,grid,HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM,tmp);
        HIP_CHECK(hipFree(tmp));
    }
};

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc,char** argv){
    int   chi_max = (argc>1)?atoi(argv[1]):70;
    int   n_steps = (argc>2)?atoi(argv[2]):10;
    float cutoff  = (argc>3)?atof(argv[3]):1e-3f;
    float eig_cut = 1e-2f;
    int   center  = 2;
    for(int i=1;i<argc;i++){
        std::string a(argv[i]);
        if(a.substr(0,10)=="--eig-cut=") eig_cut=atof(argv[i]+10);
        if(a.substr(0,9)=="--center=") center=atoi(argv[i]+9);
    }

    int n_gpus=0; HIP_CHECK(hipGetDeviceCount(&n_gpus));
    if(n_gpus<2){fprintf(stderr,"Need 2 GPUs\n");return 1;}
    enable_p2p(n_gpus);

    printf("# main_heavyhex_gamma_mpf_DG_2gpu\n");
    printf("# GPU 0: Trotter  GPU 1: DG inner products\n");
    printf("# mode: %s  chi=%d  n_steps=%d  cutoff=%.0e  eig_cut=%.0e\n",
           MODE_STR,chi_max,n_steps,(double)cutoff,(double)eig_cut);
    printf("# STEP C_nl C_n1 C_n2 C_n3"
           "  IP_A a00 a01 a02 a11 a12 a22"
           "  IP_B b0 b1 b2"
           "  chi_nl chi_n"
           "  t_cycle t_trotter t_dg_precompute t_dg_pairs\n");
    fflush(stdout);

    // ── GPU 0 ─────────────────────────────────────────────────────────────────
    HIP_CHECK(hipSetDevice(0));
    rocblas_handle handle0; rocblas_create_handle(&handle0);

    FlatHeavyHexGrid grid_nl(chi_max);
    Workspace        ws_nl(chi_max,3);
    HeavyHexState    state_nl(grid_nl);
    Measures         meas(handle0,ws_nl);
    SimNoiseless     sim_nl(handle0);
    state_nl.init_heisenberg(grid_nl,center);

    PtmFunc ptm_fns[N_GAMMA]={ptm_n1,ptm_n2,ptm_n3};
    FlatHeavyHexGrid* grids_n[N_GAMMA];
    HeavyHexState*    states_n[N_GAMMA];
    Workspace*        ws_n[N_GAMMA];
    SimNoisy*         sims_n[N_GAMMA];
    for(int g=0;g<N_GAMMA;g++){
        grids_n[g]  = new FlatHeavyHexGrid(chi_max);
        states_n[g] = new HeavyHexState(*grids_n[g]);
        ws_n[g]     = new Workspace(chi_max,3);
        sims_n[g]   = new SimNoisy(handle0,ptm_fns[g]);
        states_n[g]->init_heisenberg(*grids_n[g],center);
    }

    // ── GPU 1: DG objects ─────────────────────────────────────────────────────
    // 4 DG_StateGauge (one per state) — eigendecompose once, share W across pairs
    // 9 DG_Pair (one per pair) — GEMM + sdot only
    // Each DG_StateGauge has its own rocblas stream for parallel precompute
    HIP_CHECK(hipSetDevice(1));
    rocblas_handle hdg_nl, hdg_n[N_GAMMA];
    rocblas_create_handle(&hdg_nl);
    for(int i=0;i<N_GAMMA;i++) rocblas_create_handle(&hdg_n[i]);

    DG_StateGauge* dg_gauge_nl = new DG_StateGauge(grid_nl,hdg_nl,chi_max,eig_cut);
    DG_StateGauge* dg_gauge_n[N_GAMMA];
    for(int i=0;i<N_GAMMA;i++)
        dg_gauge_n[i] = new DG_StateGauge(*grids_n[i],hdg_n[i],chi_max,eig_cut);

    // 9 pairs: each gets its own handle (future: own stream for parallelism)
    rocblas_handle hdgp_B[N_GAMMA], hdgp_A[N_GAMMA][N_GAMMA];
    DG_Pair* dgp_B[N_GAMMA];
    DG_Pair* dgp_A[N_GAMMA][N_GAMMA]={};
    for(int i=0;i<N_GAMMA;i++){
        rocblas_create_handle(&hdgp_B[i]);
        dgp_B[i] = new DG_Pair(grid_nl,hdgp_B[i],chi_max);
        for(int j=i;j<N_GAMMA;j++){
            rocblas_create_handle(&hdgp_A[i][j]);
            dgp_A[i][j] = new DG_Pair(*grids_n[i],hdgp_A[i][j],chi_max);
        }
    }

    // Shared result arrays
    float A_ip[N_GAMMA][N_GAMMA]={};
    float B_ip[N_GAMMA]={};
    std::atomic<double> t_dg_precompute_last{0.0};
    std::atomic<double> t_dg_pairs_last{0.0};

    // ── Cycle loop ────────────────────────────────────────────────────────────
    for(int step=1;step<=n_steps;step++){
        double t_cycle0 = now_ms();
        std::atomic<double> t_trotter_pure{0.0};

        // ── Thread 0: GPU 0 Trotter ──────────────────────────────────────────
        std::thread thr_trotter([&](){
            HIP_CHECK(hipSetDevice(0));
            double t0=now_ms();
            sim_nl.step(state_nl,grid_nl,ws_nl,chi_max,cutoff);
            for(int g=0;g<N_GAMMA;g++)
                sims_n[g]->step(*states_n[g],*grids_n[g],*ws_n[g],chi_max,cutoff);
            HIP_CHECK(hipDeviceSynchronize());
            t_trotter_pure=(now_ms()-t0)/1000.0;
        });

        // ── Thread 1: GPU 1 DG inner products for previous step ──────────────
        std::thread thr_dg([&](){
            if(step==1){return;}
            HIP_CHECK(hipSetDevice(1));

            // Phase 1: 4 precomputes (can be parallelized with streams in future)
            // Each reads psi+messages from GPU 0 via P2P
            double tp0=now_ms();
            dg_gauge_nl->precompute(state_nl.psi,state_nl.messages,0);
            for(int i=0;i<N_GAMMA;i++)
                dg_gauge_n[i]->precompute(states_n[i]->psi,states_n[i]->messages,0);
            t_dg_precompute_last=(now_ms()-tp0)/1000.0;

            // Phase 2: 9 pair contractions (can be parallelized with streams)
            double tq0=now_ms();
            for(int i=0;i<N_GAMMA;i++){
                dgp_B[i]->compute_pair(*dg_gauge_nl,states_n[i]->psi,0);
                B_ip[i]=dgp_B[i]->bethe(*dg_gauge_nl);
                for(int j=i;j<N_GAMMA;j++){
                    dgp_A[i][j]->compute_pair(*dg_gauge_n[i],states_n[j]->psi,0);
                    A_ip[i][j]=dgp_A[i][j]->bethe(*dg_gauge_n[i]);
                }
            }
            t_dg_pairs_last=(now_ms()-tq0)/1000.0;
        });

        thr_trotter.join();
        thr_dg.join();
        double t_cycle=(now_ms()-t_cycle0)/1000.0;

        // ── C(t) on GPU 0 ─────────────────────────────────────────────────────
        HIP_CHECK(hipSetDevice(0));
        float C_nl=meas.autocorrelation_with_bp(
            static_cast<Grid&>(grid_nl),state_nl.psi,state_nl.messages,center,BP_SWEEPS);
        float C_n[N_GAMMA];
        for(int g=0;g<N_GAMMA;g++)
            C_n[g]=meas.autocorrelation_with_bp(
                static_cast<Grid&>(*grids_n[g]),states_n[g]->psi,
                states_n[g]->messages,center,BP_SWEEPS);

        int chi_nl=1,chi_n=1;
        for(auto& b:grid_nl.bonds)     chi_nl=std::max(chi_nl,b.chi);
        for(auto& b:grids_n[0]->bonds) chi_n =std::max(chi_n, b.chi);

        // ── Print ─────────────────────────────────────────────────────────────
        if(step==1){
            printf("STEP %d  C_nl %+.6f  C_n %+.6f %+.6f %+.6f"
                   "  IP_A nan nan nan nan nan nan"
                   "  IP_B nan nan nan"
                   "  chi %d %d  t_cycle %.2f  t_trotter %.2f"
                   "  t_dg_precompute nan  t_dg_pairs nan\n",
                   step,C_nl,C_n[0],C_n[1],C_n[2],
                   chi_nl,chi_n,t_cycle,(double)t_trotter_pure);
        } else {
            printf("STEP %d  C_nl %+.6f  C_n %+.6f %+.6f %+.6f"
                   "  IP_A %.6f %.6f %.6f %.6f %.6f %.6f"
                   "  IP_B %.6f %.6f %.6f"
                   "  chi %d %d  t_cycle %.2f  t_trotter %.2f"
                   "  t_dg_precompute %.2f  t_dg_pairs %.2f\n",
                   step,C_nl,C_n[0],C_n[1],C_n[2],
                   A_ip[0][0],A_ip[0][1],A_ip[0][2],
                   A_ip[1][1],A_ip[1][2],A_ip[2][2],
                   B_ip[0],B_ip[1],B_ip[2],
                   chi_nl,chi_n,t_cycle,(double)t_trotter_pure,
                   (double)t_dg_precompute_last,(double)t_dg_pairs_last);
        }
        fflush(stdout);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    HIP_CHECK(hipSetDevice(0));
    rocblas_destroy_handle(handle0);
    for(int g=0;g<N_GAMMA;g++){
        delete sims_n[g]; delete ws_n[g];
        delete states_n[g]; delete grids_n[g];
    }

    HIP_CHECK(hipSetDevice(1));
    delete dg_gauge_nl; rocblas_destroy_handle(hdg_nl);
    for(int i=0;i<N_GAMMA;i++){
        delete dg_gauge_n[i]; rocblas_destroy_handle(hdg_n[i]);
        delete dgp_B[i];      rocblas_destroy_handle(hdgp_B[i]);
        for(int j=i;j<N_GAMMA;j++){
            delete dgp_A[i][j]; rocblas_destroy_handle(hdgp_A[i][j]);
        }
    }
    return 0;
}
