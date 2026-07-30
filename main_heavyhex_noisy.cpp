/*
 * main_heavyhex_noisy.cpp — Noisy heavy-hex Heisenberg simulation
 *
 * Simulates the isotropic Heisenberg model H = J*Σ(XX+YY+ZZ) on the
 * heavy_hex_3x3 graph WITH hardware noise (PTM channels per bond).
 *
 * Gamma parameter scales noise strength:
 *   gamma=1.0: hardware noise as measured
 *   gamma=1.4: amplified noise (MPF extrapolation point)
 *   gamma=1.8: further amplified noise (MPF extrapolation point)
 *
 * Usage:
 *   ./main_heavyhex_noisy [chi_max] [n_steps] [dt] [J] [cutoff] [--gamma=X] [--qr=MODE] [--spatial=file]
 *
 * Defaults:
 *   chi_max=200  n_steps=10  dt=0.1  J=1.0  cutoff=1e-4  gamma=1.0
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "measures.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <algorithm>
#include <cstring>

static constexpr int   BP_MAX_SWEEPS = 100;

static double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    // ── Parse arguments ───────────────────────────────────────────
    int    chi_max = (argc > 1) ? atoi(argv[1]) : 200;
    int    n_steps = (argc > 2) ? atoi(argv[2]) : 10;
    Scalar dt      = (argc > 3) ? atof(argv[3]) : 0.1f;
    Scalar J_coup  = (argc > 4) ? atof(argv[4]) : 1.0f;
    Scalar cutoff  = (argc > 5) ? (Scalar)atof(argv[5]) : 1e-4f;

    QrMode qr_mode   = QrMode::Householder;
    float  gamma     = 1.0f;
    int    center    = 2;
    bool   measure_spatial = false;
    bool   use_exact = false;
    const char* spatial_file = nullptr;

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if      (arg == "--qr=adaptive")    qr_mode = QrMode::Adaptive;
        else if (arg == "--qr=cholesky")    qr_mode = QrMode::Cholesky;
        else if (arg == "--qr=householder") qr_mode = QrMode::Householder;
        else if (arg.substr(0,8) == "--gamma=")  gamma  = atof(argv[i]+8);
        else if (arg.substr(0,9) == "--center=") center = atoi(argv[i]+9);
        else if (arg == "--exact")          use_exact = true;
        else if (arg == "--measure=spatial") measure_spatial = true;
        else if (arg.substr(0,10) == "--spatial=") {
            measure_spatial = true;
            spatial_file = argv[i]+10;
        }
    }

    printf("═══════════════════════════════════════════════\n");
    printf("  CppSim — Heavy-Hex Noisy Heisenberg\n");
    printf("  BUILD: %s %s\n", __DATE__, __TIME__);
    printf("  Graph: heavy_hex_3x3  68 sites  76 bonds  D=%d\n", D);
    printf("  chi_max=%d  n_steps=%d  dt=%.3f  J=%.3f  cutoff=%.2e\n",
           chi_max, n_steps, dt, J_coup, cutoff);
    printf("  gamma=%.1f  center=site%d  noise=%s\n", gamma, center,
           use_exact ? "exact-interleaved" : "approx-NOISE_POWER");
    printf("  QR: %s\n",
           qr_mode == QrMode::Cholesky    ? "cholesky" :
           qr_mode == QrMode::Householder ? "householder" : "adaptive");
    printf("═══════════════════════════════════════════════\n\n");

    // ── Setup ─────────────────────────────────────────────────────
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    NoisyHeavyHexGrid    grid(chi_max, gamma, use_exact);
    Workspace            ws(chi_max, 3);
    HeavyHexState        state(grid);
    Measures             meas(handle, ws);
    NoisyHeavyHexSimulator sim(handle, J_coup, dt);
    sim.gate_ops.qr_mode = qr_mode;

    printf("Center site: id=%d (degree=%d)\n\n", center,
           (int)grid.sites[center].degree);
    state.init_heisenberg(grid, center);

    printf("Memory:\n");
    printf("  State tensors: %.1f MB\n",   grid.total_tensor_bytes()/1e6);
    printf("  Messages:      %.1f MB\n",   grid.total_message_bytes()/1e6);
    printf("  Workspace:     %.1f MB\n\n", ws.total_bytes/1e6);

    // ── Spatial setup ─────────────────────────────────────────────
    std::vector<float> spatial(grid.n_sites, 0.f);
    FILE* spatial_fp = nullptr;
    if (measure_spatial) {
        spatial_fp = spatial_file ? fopen(spatial_file, "w") : stdout;
        if (!spatial_fp) { fprintf(stderr, "Cannot open %s\n", spatial_file); return 1; }
        fprintf(spatial_fp, "# heavy_hex_3x3_noisy  n_sites=%d  center=%d  gamma=%.1f\n",
                grid.n_sites, center, gamma);
        fprintf(spatial_fp, "# columns: site_id weight\n");
    }

    // ── Header ────────────────────────────────────────────────────
    printf("%-6s %-8s %-8s %-12s %-10s  %-12s %-10s %-10s %-7s %-10s  %-10s %-8s\n",
           "Layer","chi_min","chi_max","C(t)","total(s)",
           "trotter(ms)","bp(ms)","Cfn(ms)","sweeps","delta",
           "gate_err","Se/logX");
    printf("%-6s %-8s %-8s %-12s %-10s  %-12s %-10s %-10s %-7s %-10s  %-10s %-8s\n",
           "------","--------","--------","------------","----------",
           "------------","----------","----------","-------","----------",
           "----------","--------");
    printf("%-6d %-8d %-8d %-12.6f %-10s  %-12s %-10s %-10s %-7s %-10s  %-10s %-8s\n",
           0,1,1,1.0f,"-","-","-","-","-","-","-","-");

    // ── Trotter evolution ─────────────────────────────────────────
    for (int layer = 1; layer <= n_steps; layer++) {
        double t0 = now_ms();

        LayerMetrics lm = {};
        int max_chi = sim.trotter_layer(state, grid, ws, chi_max, cutoff, &lm);

        double t_trotter = now_ms() - t0;

        int min_chi = max_chi;
        for (auto& bond : grid.bonds)
            min_chi = std::min(min_chi, bond.chi);

        double t_bp=0, t_pf=0;
        int bp_sweeps=0; float bp_delta=0.f;
        Scalar C = meas.autocorrelation_with_bp(
            static_cast<Grid&>(grid), state.psi, state.messages,
            center, BP_MAX_SWEEPS, &t_bp, &t_pf, &bp_sweeps, &bp_delta);

        double dt_total = now_ms() - t0;
        float gap_print = (lm.n_gates > 0 && lm.gap < 1.f) ? lm.gap : 1.f;

        printf("%-6d %-8d %-8d %-12.6f %-10.3f  %-12.1f %-10.1f %-10.1f %-7d %-10.2e  %-10.2e %-8.4f\n",
               layer, min_chi, max_chi, C, dt_total/1000.0,
               t_trotter, t_bp, t_pf, bp_sweeps, bp_delta,
               lm.gate_err, lm.Se_over_logchi);
        fflush(stdout);

        if (measure_spatial) {
            meas.compute_pauli_profile(static_cast<Grid&>(grid),
                                       state.psi, spatial.data());
            fprintf(spatial_fp, "# cycle %d  C=%.6f  chi=%d  gamma=%.1f\n",
                    layer, C, max_chi, gamma);
            for (int s = 0; s < grid.n_sites; s++)
                fprintf(spatial_fp, "%d %.6f\n", s, spatial[s]);
            fflush(spatial_fp);
        }
    }

    printf("\nCholesky-QR: attempted=%d  fallback=%d  skipped=%d\n",
           ws.chol_attempted, ws.chol_fallback, ws.chol_skipped);
    printf("\nDone.\n");
    if (spatial_fp && spatial_fp != stdout) fclose(spatial_fp);
    rocblas_destroy_handle(handle);
    return 0;
}
