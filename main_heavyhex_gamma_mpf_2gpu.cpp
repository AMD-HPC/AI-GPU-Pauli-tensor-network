/*
 * main_heavyhex_gamma_mpf_2gpu.cpp — Gamma-MPF with 2-GPU overlap
 *
 * GPU 0: ALL Trotter (noiseless + 3 noisy) + BP → C(t)
 * GPU 1: ALL inner products (A matrix + B vector) via BLF_GPU
 *
 * Per cycle overlap:
 *   GPU 0: Trotter step N+1
 *   GPU 1: inner products for step N  (A and B)
 *   P2P:   ψ_nl, ψ_n1,ψ_n2,ψ_n3 copied GPU0→GPU1 after each step
 *   CPU:   read 9 floats → print to log
 *
 * Output format (same as main_heavyhex_gamma_mpf_ip.cpp):
 *   STEP N  C_nl ...  C_n ...  IP_A ...  IP_B ...  chi ... t_trotter t_ip
 *
 * Build: make PHYS_DIM=4 main_heavyhex_gamma_mpf_2gpu
 * Run:   ./main_heavyhex_gamma_mpf_2gpu [chi_max] [n_steps] [cutoff]
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "measures.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"
#include "bilinear_form_bp_gpu.hpp"
#include "bilinear_form_bp_gpu3.hpp"
#include "overlap_gpu.hpp"

#include "heavyhex_ideal_kick_eps010_data.hpp"

#ifdef ERGODIC2
// Ergodic amplification: fixed γ=2, varying ε=0.10/0.15/0.20
#include "heavyhex_ideal_kick_eps010_gamma2_data.hpp"
#include "heavyhex_ideal_kick_eps015_gamma2_data.hpp"
#include "heavyhex_ideal_kick_eps020_gamma2_data.hpp"
static const float* ptm_n1(int b){return HEAVY_HEX_3x3_IDEAL_KICK_EPS010_GAMMA2[b];}
static const float* ptm_n2(int b){return HEAVY_HEX_3x3_IDEAL_KICK_EPS015_GAMMA2[b];}
static const float* ptm_n3(int b){return HEAVY_HEX_3x3_IDEAL_KICK_EPS020_GAMMA2[b];}
static const char* MODE_STR = "ergodic  eps=0.10/0.15/0.20  gamma=2.0";
#elif defined(ERGODIC)
// Ergodic amplification: fixed γ=1, varying ε=0.10/0.15/0.20
#include "heavyhex_ideal_kick_eps010_gamma1_data.hpp"
#include "heavyhex_ideal_kick_eps015_gamma1_data.hpp"
#include "heavyhex_ideal_kick_eps020_gamma1_data.hpp"
static const float* ptm_n1(int b){return HEAVY_HEX_3x3_IDEAL_KICK_EPS010_GAMMA1[b];}
static const float* ptm_n2(int b){return HEAVY_HEX_3x3_IDEAL_KICK_EPS015_GAMMA1[b];}
static const float* ptm_n3(int b){return HEAVY_HEX_3x3_IDEAL_KICK_EPS020_GAMMA1[b];}
static const char* MODE_STR = "ergodic  eps=0.10/0.15/0.20  gamma=1.0";
#else
// Gamma-MPF: fixed ε=0.10, varying γ=1.0/2.0/3.0
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

static constexpr int   BP_SWEEPS = 100;
static constexpr int   N_GAMMA   = 3;

static double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

using PtmFunc = const float*(*)(int);

// ── Simulator structs (GPU 0) ─────────────────────────────────────────────────

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
    bool  verbose_ip = true;   // print per-inner-product timing

    // ── Check 2 GPUs + P2P ───────────────────────────────────────────────────
    int n_gpus = 0;
    HIP_CHECK(hipGetDeviceCount(&n_gpus));
    if (n_gpus < 2) {
        fprintf(stderr, "ERROR: need 2 GPUs, found %d\n", n_gpus);
        return 1;
    }
    enable_p2p(n_gpus);

    printf("# CppSim gamma-MPF 2-GPU inner product dump\n");
    printf("# GPU 0: all Trotter  GPU 1: all inner products\n");
    printf("# epsilon=%.4f  chi_max=%d  n_steps=%d  cutoff=%.0e\n",
           HEAVY_HEX_3x3_IDEAL_KICK_EPS010_EPSILON, chi_max, n_steps, (double)cutoff);
    printf("# mode: %s\n", MODE_STR);
    printf("# STEP C_nl C_n1 C_n2 C_n3  IP_A a00 a01 a02 a11 a12 a22  IP_B b0 b1 b2"
           "  chi_nl chi_n  t_cycle(s) t_trotter_pure(s) t_ip_wall(s) t_upload(s)\n");
    fflush(stdout);

    // ── GPU 0: all states ─────────────────────────────────────────────────────
    HIP_CHECK(hipSetDevice(0));
    rocblas_handle handle0;
    rocblas_create_handle(&handle0);

    FlatHeavyHexGrid  grid_nl(chi_max);
    Workspace         ws_nl(chi_max, 3);
    HeavyHexState     state_nl(grid_nl);
    Measures          meas(handle0, ws_nl);
    SimNoiseless      sim_nl(handle0);
    state_nl.init_heisenberg(grid_nl, center);

    PtmFunc ptm_fns[N_GAMMA] = {ptm_n1, ptm_n2, ptm_n3};
    FlatHeavyHexGrid* grids_n[N_GAMMA];
    HeavyHexState*    states_n[N_GAMMA];
    Workspace*        ws_n[N_GAMMA];
    SimNoisy*         sims_n[N_GAMMA];
    for (int g=0; g<N_GAMMA; g++) {
        grids_n[g]  = new FlatHeavyHexGrid(chi_max);
        states_n[g] = new HeavyHexState(*grids_n[g]);
        ws_n[g]     = new Workspace(chi_max, 3);
        sims_n[g]   = new SimNoisy(handle0, ptm_fns[g]);
        states_n[g]->init_heisenberg(*grids_n[g], center);
    }

    // ── GPU 1: BLF_GPU objects (one per pair) ────────────────────────────────
    // A[i,j]: 3 diagonal + 3 off-diagonal = 6 BLF_GPU objects
    // B[i]:   3 cross (nl vs ni)
    // All live on GPU 1
    HIP_CHECK(hipSetDevice(1));
    rocblas_handle handle1;
    rocblas_create_handle(&handle1);

    // BLF_GPU3 objects: one per inner product pair, each with its own handle+stream
    // This enables concurrent execution of independent inner products on GPU 1.
    BLF_GPU3* blf_A[N_GAMMA][N_GAMMA] = {};
    BLF_GPU3* blf_B[N_GAMMA]          = {};
    for (int i=0; i<N_GAMMA; i++) {
        for (int j=i; j<N_GAMMA; j++) {
            rocblas_handle hA; rocblas_create_handle(&hA);
            blf_A[i][j] = new BLF_GPU3(hA);
        }
        rocblas_handle hB; rocblas_create_handle(&hB);
        blf_B[i] = new BLF_GPU3(hB);
    }

    // ── Async inner product results (shared between threads) ──────────────────
    std::atomic<bool> ip_ready{false};
    float A_ip[N_GAMMA][N_GAMMA] = {};
    float B_ip[N_GAMMA]          = {};
    double t_ip_last = 0.0;
    std::atomic<double> t_ip_wall_last{0.0};

    // ── Cycle loop ────────────────────────────────────────────────────────────
    for (int step = 1; step <= n_steps; step++) {

        // ── Thread 0: GPU 0 Trotter ──────────────────────────────────────────
        double t_trotter0 = now_ms();
        std::atomic<double> t_trotter_pure{0.0};
        std::thread thr_trotter([&]() {
            HIP_CHECK(hipSetDevice(0));
            double t0 = now_ms();
            sim_nl.step(state_nl, grid_nl, ws_nl, chi_max, cutoff);
            for (int g=0; g<N_GAMMA; g++)
                sims_n[g]->step(*states_n[g], *grids_n[g], *ws_n[g], chi_max, cutoff);
            HIP_CHECK(hipDeviceSynchronize());
            t_trotter_pure = (now_ms() - t0) / 1000.0;
        });

        // ── Thread 1: GPU 1 inner products — all 9 pairs in parallel streams ──
        std::thread thr_ip([&]() {
            if (step == 1) return;
            HIP_CHECK(hipSetDevice(1));

            // Launch all 9 inner products concurrently — each BLF_GPU3 has
            // its own stream, so they run in parallel on GPU 1.
            struct PairResult { float val; double t_ms; };
            std::vector<std::thread> ip_threads;
            PairResult res_A[N_GAMMA][N_GAMMA] = {};
            PairResult res_B[N_GAMMA]          = {};

            for (int i=0; i<N_GAMMA; i++) {
                for (int j=i; j<N_GAMMA; j++) {
                    ip_threads.push_back(std::thread([&, i, j]() {
                        HIP_CHECK(hipSetDevice(1));
                        res_A[i][j].val = blf_A[i][j]->compute(25,1e-5f,false,&res_A[i][j].t_ms);
                    }));
                }
                ip_threads.push_back(std::thread([&, i]() {
                    HIP_CHECK(hipSetDevice(1));
                    res_B[i].val = blf_B[i]->compute(25,1e-5f,false,&res_B[i].t_ms);
                }));
            }
            double t_ip_wall0 = now_ms();
            for (auto& t : ip_threads) t.join();
            double t_ip_wallclock = (now_ms() - t_ip_wall0) / 1000.0;

            // Collect results
            double t_A=0, t_B=0;
            for (int i=0; i<N_GAMMA; i++) {
                A_ip[i][i] = res_A[i][i].val; t_A += res_A[i][i].t_ms;
                for (int j=i+1; j<N_GAMMA; j++) {
                    A_ip[i][j] = A_ip[j][i] = res_A[i][j].val; t_A += res_A[i][j].t_ms;
                }
                B_ip[i] = res_B[i].val; t_B += res_B[i].t_ms;
            }
            t_ip_wall_last = t_ip_wallclock;
            if (verbose_ip)
                printf("  [IP] wall=%.2fs  serial=%.2fs\n",
                       t_ip_wallclock, (t_A+t_B)/1000);
            ip_ready = true;
        });

        thr_trotter.join();
        thr_ip.join();
        double t_trotter = (now_ms() - t_trotter0) / 1000.0;

        // ── BP on GPU 0: C(t) ────────────────────────────────────────────────
        HIP_CHECK(hipSetDevice(0));
        float C_nl = meas.autocorrelation_with_bp(
            static_cast<Grid&>(grid_nl), state_nl.psi, state_nl.messages,
            center, BP_SWEEPS);
        float C_n[N_GAMMA];
        for (int g=0; g<N_GAMMA; g++)
            C_n[g] = meas.autocorrelation_with_bp(
                static_cast<Grid&>(*grids_n[g]), states_n[g]->psi,
                states_n[g]->messages, center, BP_SWEEPS);

        int chi_nl = 1, chi_n = 1;
        for (auto& b : grid_nl.bonds)      chi_nl = std::max(chi_nl, b.chi);
        for (auto& b : grids_n[0]->bonds)  chi_n  = std::max(chi_n,  b.chi);

        // ── Upload current state to GPU 1 (for next step's inner products) ───
        double t_ip0 = now_ms();
        HIP_CHECK(hipSetDevice(1));
        for (int i=0; i<N_GAMMA; i++) {
            for (int j=i; j<N_GAMMA; j++)
                blf_A[i][j]->upload(*grids_n[i], states_n[i]->psi, 0,
                                    *grids_n[j], states_n[j]->psi, 0);
            blf_B[i]->upload(grid_nl,       state_nl.psi,         0,
                             *grids_n[i],   states_n[i]->psi,     0);
        }
        t_ip_last = (now_ms() - t_ip0) / 1000.0;

        // ── Print ─────────────────────────────────────────────────────────────
        // Step 1: no inner products yet — print zeros
        if (step == 1) {
            printf("STEP %d  C_nl %+.6f  C_n %+.6f %+.6f %+.6f"
                   "  IP_A nan nan nan nan nan nan"
                   "  IP_B nan nan nan"
                   "  chi %d %d  t_cycle %.2f  t_trotter_pure %.2f  t_ip_wall nan  t_upload nan\n",
                   step, C_nl, C_n[0], C_n[1], C_n[2],
                   chi_nl, chi_n, t_trotter, (double)t_trotter_pure);
        } else {
            printf("STEP %d  C_nl %+.6f  C_n %+.6f %+.6f %+.6f"
                   "  IP_A %.6f %.6f %.6f %.6f %.6f %.6f"
                   "  IP_B %.6f %.6f %.6f"
                   "  chi %d %d  t_cycle %.2f  t_trotter_pure %.2f  t_ip_wall %.2f  t_upload %.2f\n",
                   step, C_nl, C_n[0], C_n[1], C_n[2],
                   A_ip[0][0], A_ip[0][1], A_ip[0][2],
                   A_ip[1][1], A_ip[1][2], A_ip[2][2],
                   B_ip[0], B_ip[1], B_ip[2],
                   chi_nl, chi_n, t_trotter, (double)t_trotter_pure,
                   (double)t_ip_wall_last, t_ip_last);
        }
        fflush(stdout);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    HIP_CHECK(hipSetDevice(0));
    rocblas_destroy_handle(handle0);
    for (int g=0; g<N_GAMMA; g++) {
        delete sims_n[g]; delete ws_n[g];
        delete states_n[g]; delete grids_n[g];
    }

    HIP_CHECK(hipSetDevice(1));
    rocblas_destroy_handle(handle1);
    for (int i=0; i<N_GAMMA; i++) {
        for (int j=i; j<N_GAMMA; j++) delete blf_A[i][j];
        delete blf_B[i];
    }

    return 0;
}
