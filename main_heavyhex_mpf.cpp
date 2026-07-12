/*
 * main_heavyhex_mpf.cpp — Two-GPU MPF heavy-hex simulation
 *
 * GPU 0: noiseless  — trotter → BP → C_nl(t)
 * GPU 1: 3x noisy   — trotter → BP  [gamma=1.0, 1.4, 1.8, chi capped]
 * P2P:  noisy psi copied GPU1→GPU0 → overlap ⟨ψ_nl|ψ_n(γ)⟩ on GPU 0
 *
 * True parallel execution via std::thread:
 *   Thread 0 owns GPU 0 (noiseless trotter + BP)
 *   Thread 1 owns GPU 1 (3x noisy trotter + BP)
 *   Main thread: sync → P2P overlaps → print
 *
 * Output per cycle:
 *   cycle  C_nl  OV(1.0)  OV(1.4)  OV(1.8)  chi_nl  chi_n  t_nl  t_n  t_tot
 *
 * Usage:
 *   ./main_heavyhex_mpf [chi_nl] [chi_n] [n_steps] [dt] [J] [cutoff]
 *                       [--center=N] [--qr=MODE]
 * Defaults:
 *   chi_nl=430  chi_n=200  n_steps=10  dt=0.1  J=1.0  cutoff=1e-4  center=2
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "measures.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"
#include "overlap_gpu.hpp"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <algorithm>
#include <thread>
#include <atomic>
#include <hip/hip_runtime.h>

static constexpr float GAMMAS[3]  = {1.0f, 1.4f, 1.8f};
static constexpr int   N_GAMMAS   = 3;
static constexpr int   BP_SWEEPS  = 100;

static double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    // ── Parse arguments ───────────────────────────────────────────
    int    chi_nl  = (argc > 1) ? atoi(argv[1]) : 430;
    int    chi_n   = (argc > 2) ? atoi(argv[2]) : 200;
    int    n_steps = (argc > 3) ? atoi(argv[3]) : 10;
    Scalar dt      = (argc > 4) ? atof(argv[4]) : 0.1f;
    Scalar J_coup  = (argc > 5) ? atof(argv[5]) : 1.0f;
    Scalar cutoff  = (argc > 6) ? (Scalar)atof(argv[6]) : 1e-4f;

    QrMode qr_mode = QrMode::Householder;
    int    center  = 2;

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if      (arg == "--qr=adaptive")    qr_mode = QrMode::Adaptive;
        else if (arg == "--qr=cholesky")    qr_mode = QrMode::Cholesky;
        else if (arg == "--qr=householder") qr_mode = QrMode::Householder;
        else if (arg.substr(0,9) == "--center=") center = atoi(argv[i]+9);
    }

    // ── Check GPUs + enable P2P ───────────────────────────────────
    int n_gpus = 0;
    hipGetDeviceCount(&n_gpus);
    if (n_gpus < 2) {
        fprintf(stderr, "ERROR: need at least 2 GPUs, found %d\n", n_gpus);
        return 1;
    }
    enable_p2p(n_gpus);

    printf("═══════════════════════════════════════════════\n");
    printf("  CppSim — Heavy-Hex MPF (2-GPU + P2P)\n");
    printf("  BUILD: %s %s\n", __DATE__, __TIME__);
    printf("  Graph: heavy_hex_3x3  68 sites  76 bonds  D=%d\n", D);
    printf("  GPU 0 (noiseless): chi_max=%d\n", chi_nl);
    printf("  GPU 1 (noisy x3):  chi_max=%d  gamma=%.1f/%.1f/%.1f\n",
           chi_n, GAMMAS[0], GAMMAS[1], GAMMAS[2]);
    printf("  n_steps=%d  dt=%.3f  J=%.3f  cutoff=%.2e  center=site%d\n",
           n_steps, dt, J_coup, cutoff, center);
    printf("  QR: %s  |  Parallel: std::thread + P2P overlap\n",
           qr_mode==QrMode::Householder ? "householder" :
           qr_mode==QrMode::Cholesky    ? "cholesky" : "adaptive");
    printf("═══════════════════════════════════════════════\n\n");

    // ── Setup GPU 0: noiseless ────────────────────────────────────
    hipSetDevice(0);
    rocblas_handle handle0;
    rocblas_create_handle(&handle0);

    FlatHeavyHexGrid   grid_nl(chi_nl);
    Workspace          ws_nl(chi_nl, 3);
    HeavyHexState      state_nl(grid_nl);
    Measures           meas_nl(handle0, ws_nl);
    HeavyHexSimulator  sim_nl(handle0, J_coup, dt);
    sim_nl.gate_ops.qr_mode = qr_mode;
    state_nl.init_heisenberg(grid_nl, center);

    printf("GPU 0 memory: state=%.1f MB  msg=%.1f MB  ws=%.1f MB\n",
           grid_nl.total_tensor_bytes()/1e6,
           grid_nl.total_message_bytes()/1e6,
           ws_nl.total_bytes/1e6);

    // ── Setup GPU 1: 3 noisy states ───────────────────────────────
    hipSetDevice(1);
    rocblas_handle handle1;
    rocblas_create_handle(&handle1);

    NoisyHeavyHexGrid*     grid_n[N_GAMMAS];
    HeavyHexState*         state_n[N_GAMMAS];
    Workspace*             ws_n[N_GAMMAS];
    Measures*              meas_n[N_GAMMAS];
    NoisyHeavyHexSimulator* sim_n[N_GAMMAS];

    for (int g = 0; g < N_GAMMAS; g++) {
        grid_n[g]  = new NoisyHeavyHexGrid(chi_n, GAMMAS[g]);
        state_n[g] = new HeavyHexState(*grid_n[g]);
        ws_n[g]    = new Workspace(chi_n, 3);
        meas_n[g]  = new Measures(handle1, *ws_n[g]);
        sim_n[g]   = new NoisyHeavyHexSimulator(handle1, J_coup, dt);
        sim_n[g]->gate_ops.qr_mode = qr_mode;
        state_n[g]->init_heisenberg(*grid_n[g], center);
    }

    printf("GPU 1 memory: %.1f MB per state × 3 = %.1f MB total\n\n",
           (grid_n[0]->total_tensor_bytes() + grid_n[0]->total_message_bytes() +
            ws_n[0]->total_bytes)/1e6,
           3*(grid_n[0]->total_tensor_bytes() + grid_n[0]->total_message_bytes() +
              ws_n[0]->total_bytes)/1e6);

    // ── Header ────────────────────────────────────────────────────
    printf("%-6s  %-12s %-12s %-12s %-12s  %-7s %-7s  %-8s %-8s %-8s\n",
           "Cycle","C_nl","OV(1.0)","OV(1.4)","OV(1.8)",
           "chi_nl","chi_n","t_nl(s)","t_n(s)","t_tot(s)");
    printf("%-6s  %-12s %-12s %-12s %-12s  %-7s %-7s  %-8s %-8s %-8s\n",
           "------","------------","------------","------------","------------",
           "-------","-------","--------","--------","--------");
    printf("%-6d  %-12.6f %-12.6f %-12.6f %-12.6f  %-7d %-7d  %-8s %-8s %-8s\n",
           0,1.0f,1.0f,1.0f,1.0f,1,1,"-","-","-");

    // ── Trotter evolution ─────────────────────────────────────────
    double t_nl_s = 0, t_n_s = 0;

    for (int layer = 1; layer <= n_steps; layer++) {
        double t0 = now_ms();

        // ── Parallel trotter: thread 0 = GPU 0, thread 1 = GPU 1 ─
        std::atomic<double> t_nl_done{0}, t_n_done{0};

        std::thread thr0([&]() {
            hipSetDevice(0);
            LayerMetrics lm = {};
            sim_nl.trotter_layer(state_nl, grid_nl, ws_nl, chi_nl, cutoff, &lm);
            hipDeviceSynchronize();
            t_nl_done = now_ms() - t0;
        });

        std::thread thr1([&]() {
            hipSetDevice(1);
            for (int g = 0; g < N_GAMMAS; g++)
                sim_n[g]->trotter_layer(*state_n[g], *grid_n[g], *ws_n[g], chi_n, cutoff);
            hipDeviceSynchronize();
            t_n_done = now_ms() - t0;
        });

        thr0.join();
        thr1.join();

        t_nl_s = t_nl_done / 1000.0;
        t_n_s  = t_n_done  / 1000.0;

        // ── BP on GPU 0: noiseless C(t) + messages for overlap ───
        hipSetDevice(0);
        double tb=0,tp=0; int sw=0; float dl=0.f;
        Scalar C_nl = meas_nl.autocorrelation_with_bp(
            static_cast<Grid&>(grid_nl), state_nl.psi, state_nl.messages,
            center, BP_SWEEPS, &tb, &tp, &sw, &dl);

        int chi_nl_max = 1;
        for (auto& b : grid_nl.bonds) chi_nl_max = std::max(chi_nl_max, b.chi);

        // ── BP on GPU 1: noisy C(t) (for diagnostics) ────────────
        hipSetDevice(1);
        Scalar C_n[N_GAMMAS];
        for (int g = 0; g < N_GAMMAS; g++) {
            C_n[g] = meas_n[g]->autocorrelation_with_bp(
                static_cast<Grid&>(*grid_n[g]), state_n[g]->psi,
                state_n[g]->messages, center, BP_SWEEPS,
                &tb, &tp, &sw, &dl);
        }

        int chi_n_max = 1;
        for (auto& b : grid_n[0]->bonds) chi_n_max = std::max(chi_n_max, b.chi);

        // ── P2P overlaps: ⟨ψ_nl|ψ_n(γ)⟩ on GPU 0 ────────────────
        hipSetDevice(0);
        float overlaps[N_GAMMAS];
        const NoisyHeavyHexGrid* cgrid_n[N_GAMMAS];
        const HeavyHexState*     cstate_n[N_GAMMAS];
        for (int g = 0; g < N_GAMMAS; g++) {
            cgrid_n[g]  = grid_n[g];
            cstate_n[g] = state_n[g];
        }
        compute_overlaps_p2p(meas_nl,
            static_cast<Grid&>(grid_nl), state_nl.psi, state_nl.messages,
            center, cgrid_n, cstate_n, chi_n, N_GAMMAS, 0, 1,
            overlaps, nullptr);

        double t_tot = (now_ms() - t0) / 1000.0;

        printf("%-6d  %-12.6f %-12.6f %-12.6f %-12.6f  %-7d %-7d  %-8.3f %-8.3f %-8.3f\n",
               layer, C_nl, overlaps[0], overlaps[1], overlaps[2],
               chi_nl_max, chi_n_max,
               t_nl_s, t_n_s, t_tot);
        fflush(stdout);
    }

    // ── Cleanup ───────────────────────────────────────────────────
    hipSetDevice(0);
    rocblas_destroy_handle(handle0);

    hipSetDevice(1);
    for (int g = 0; g < N_GAMMAS; g++) {
        delete sim_n[g]; delete meas_n[g];
        delete ws_n[g];  delete state_n[g];
        delete grid_n[g];
    }
    rocblas_destroy_handle(handle1);

    printf("\nDone.\n");
    return 0;
}
