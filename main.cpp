/*
 * main.cpp — 4×4 Ising Heisenberg picture simulation
 *
 * Matches Julia's 2dIsing_dynamics_Heisenbergpicture_multi.jl
 *
 * Usage:
 *   ./main [nx] [ny] [chi_max] [n_steps] [dt] [h] [J] [cutoff] [--bp=diag|full] [--qr=adaptive|cholesky|householder]
 *
 * Defaults:
 *   nx=4, ny=4, chi_max=50, n_steps=20, dt=0.1, h=-1.0, J=-1.0, cutoff=1e-11
 *   --bp=diag        (diagonal messages, default)
 *   --bp=full        (full chi×chi messages, Julia-equivalent)
 *   --qr=householder (default: backward stable, uniform gauge, preserves lattice symmetry)
 *   --qr=adaptive    (Cholesky for large tall-thin matrices, Householder otherwise)
 *   --qr=cholesky    (always attempt Cholesky; falls back if SPOTRF fails)
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "physics.hpp"
#include "measures.hpp"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <algorithm>
#include <vector>

static constexpr int   BP_MAX_SWEEPS = 100;  // maximum BP sweeps; exits early on convergence
static constexpr float NORM_STOP     = 2.0f; // norm > threshold → Bethe broken, mark inf and skip spatial

static double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(
        high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    // ── Parse arguments ───────────────────────────────────────────
    int    nx       = (argc > 1) ? atoi(argv[1]) : 4;
    int    ny       = (argc > 2) ? atoi(argv[2]) : 4;
    int    chi_max  = (argc > 3) ? atoi(argv[3]) : 50;
    int    n_steps  = (argc > 4) ? atoi(argv[4]) : 20;
    Scalar dt       = (argc > 5) ? atof(argv[5]) : 0.1f;
    Scalar h_field  = (argc > 6) ? atof(argv[6]) : -1.0f;
    Scalar J_coup   = (argc > 7) ? atof(argv[7]) : -1.0f;
    Scalar cutoff   = (argc > 8) ? (Scalar)atof(argv[8]) : 1e-11f;

    // Optional --bp=diag|damped|full and --alpha=<float> flags
    bool  full_bp   = false;
    bool  damped_bp = false;
    float bp_alpha  = 0.5f;   // default damping factor when --bp=damped

    // Optional --qr=adaptive|cholesky|householder flag
    QrMode qr_mode = QrMode::Householder;

    // Optional --measure=norm|spatial|osee|all flags
    bool  measure_norm    = false;
    bool  measure_spatial = false;
    bool  measure_osee    = false;

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--bp=full")              { full_bp = true;  damped_bp = false; }
        else if (arg == "--bp=damped")       { damped_bp = true; full_bp = false;  }
        else if (arg == "--bp=diag")         { full_bp = false; damped_bp = false; }
        else if (arg.substr(0,8) == "--alpha=") { bp_alpha = (float)atof(arg.substr(8).c_str()); }
        else if (arg == "--qr=adaptive")    { qr_mode = QrMode::Adaptive;    }
        else if (arg == "--qr=cholesky")    { qr_mode = QrMode::Cholesky;    }
        else if (arg == "--qr=householder") { qr_mode = QrMode::Householder; }
        else if (arg == "--measure=norm")    { measure_norm    = true; }
        else if (arg == "--measure=spatial") { measure_spatial = true; }
        else if (arg == "--measure=osee")    { measure_osee    = true; }
        else if (arg == "--measure=all")     { measure_norm = measure_spatial = measure_osee = true; }
    }

    printf("═══════════════════════════════════════════════\n");
    printf("  CppSim — Ising Heisenberg Picture\n");
    printf("  BUILD: %s %s\n", __DATE__, __TIME__);
    printf("  Bethe edge corr, BP %s%s\n",
           full_bp   ? "full chi×chi (d=0/d=3)" :
           damped_bp ? "damped diagonal" : "diagonal (d=0/d=3)",
           damped_bp ? (std::string(" α=") + std::to_string(bp_alpha)).c_str() : "");
    printf("  QR: %s\n",
           qr_mode == QrMode::Cholesky    ? "cholesky (forced)" :
           qr_mode == QrMode::Householder ? "householder (forced)" : "adaptive");
    printf("  Grid: %dx%d  chi_max=%d  D=%d\n", nx, ny, chi_max, D);
    printf("  Steps: %d  dt=%.3f  h=%.3f  J=%.3f  cutoff=%.2e\n",
           n_steps, dt, h_field, J_coup, cutoff);
    printf("═══════════════════════════════════════════════\n\n");

    // ── Setup ─────────────────────────────────────────────────────
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    Grid       grid(nx, ny, chi_max);
    Workspace  ws(chi_max);
    IsingState state(grid);
    Measures   meas(handle, ws);
    meas.full_bp   = full_bp;
    meas.damped_bp = damped_bp;
    meas.bp_alpha  = bp_alpha;
    IsingSimulator sim(handle, h_field, J_coup, dt);
    sim.gate_ops.qr_mode = qr_mode;

    // Center site — geometric center, works for both even and odd grids.
    // Even: (nx-1)/2 == nx/2-1 (same as Julia convention). Odd: gives true midpoint.
    int cr = (nx-1)/2, cc = (ny-1)/2;
    int center = grid.site_id(cr, cc);
    printf("Center site: (%d,%d) id=%d\n\n", cr, cc, center);

    // Initialize state: Z at center, I elsewhere
    state.init_heisenberg(grid, center);

    printf("Memory:\n");
    printf("  State tensors + messages: %.1f MB\n",
           state.total_bytes(grid)/1e6);
    printf("  Workspace: %.1f MB\n", ws.total_bytes/1e6);
    printf("\n");

    // ── Print header ──────────────────────────────────────────────
    printf("%-6s %-8s %-8s %-12s %-10s  %-12s %-10s %-10s %-7s %-10s  %-10s %-8s %-8s",
           "Layer", "chi_min", "chi_max", "C(t)", "total(s)",
           "trotter(ms)", "bp(ms)", "Cfn(ms)", "sweeps", "delta",
           "gate_err", "Se/logX", "gap");
    if (measure_norm)    printf("  %-12s", "norm");
    if (measure_spatial) printf("  %-12s", "n_off(r,t)");
    if (measure_osee)    printf("  %-14s", "OSEE/Smax");
    printf("\n");
    printf("%-6s %-8s %-8s %-12s %-10s  %-12s %-10s %-10s %-7s %-10s  %-10s %-8s %-8s",
           "------", "--------", "--------", "------------", "----------",
           "------------", "----------", "----------", "-------", "----------",
           "----------", "--------", "--------");
    if (measure_norm)    printf("  %-12s", "------------");
    if (measure_spatial) printf("  %-12s", "------------");
    if (measure_osee)    printf("  %-14s", "--------------");
    printf("\n");

    // ── t=0 measurement ───────────────────────────────────────────
    printf("%-6d %-8d %-8d %-12.6f %-10s  %-12s %-10s %-10s %-7s %-10s  %-10s %-8s %-8s",
           0, 1, 1, 1.0f, "-", "-", "-", "-", "-", "-", "-", "-", "-");
    if (measure_norm)    printf("  %-12s", "-");
    if (measure_spatial) printf("  %-12s", "-");
    if (measure_osee)    printf("  %-14s", "-");
    printf("\n");

    // ── Trotter evolution ─────────────────────────────────────────
    bool norm_exploded = false;  // set when norm > NORM_STOP; gates norm+spatial from that layer on
    for (int layer=1; layer<=n_steps; layer++) {
        double t0 = now_ms();

        LayerMetrics lm;
        int max_chi = sim.trotter_layer(state, grid, ws, chi_max, cutoff, &meas, center, &lm);

        double t_trotter = now_ms() - t0;

        int min_chi = max_chi;
        for (auto& bond : grid.bonds)
            min_chi = std::min(min_chi, bond.chi);

        // BP on a copy — Vidal gauge state.messages left intact for next layer
        double t_bp=0, t_pf=0;
        int bp_sweeps=0; float bp_delta=0.f;
        std::vector<float> spatial(grid.n_sites, 0.f);
        Scalar C = meas.autocorrelation_with_bp(grid, state.psi,
                                                 state.messages, center, BP_MAX_SWEEPS,
                                                 &t_bp, &t_pf, &bp_sweeps, &bp_delta);

        double dt_total = now_ms() - t0;

        // gap == 1.f means no truncation occurred (all singular values kept)
        float gap_print = (lm.n_gates > 0 && lm.gap < 1.f) ? lm.gap : 1.f;
        printf("%-6d %-8d %-8d %-12.6f %-10.3f  %-12.1f %-10.1f %-10.1f %-7d %-10.2e  %-10.2e %-8.4f %-8.4f",
               layer, min_chi, max_chi, C, dt_total/1000.0,
               t_trotter, t_bp, t_pf, bp_sweeps, bp_delta,
               lm.gate_err, lm.Se_over_logchi, gap_print);

        // norm: gated by norm_exploded (print inf once Bethe has broken down)
        // spatial: always computed — bp re-converges after explosion and spatial remains informative
        float norm_val = 0.f;
        if (measure_norm || measure_spatial) {
            norm_val = meas.compute_norm(grid, state.psi, state.messages);
            if (norm_val > NORM_STOP) norm_exploded = true;
        }
        if (measure_norm) {
            if (norm_exploded && norm_val <= NORM_STOP) printf("  %-12s",   "inf");
            else                                        printf("  %-12.6f", norm_val);
        }
        if (measure_spatial) {
            meas.compute_pauli_profile(grid, state.psi, spatial.data());
            float n_off = 0.f;
            for (int s = 0; s < (int)grid.n_sites; s++)
                if (s != center) n_off = std::max(n_off, spatial[s]);
            printf("  %-12.6f", n_off);
        }
        if (measure_osee) {
            int   n_cut  = 0;
            float s_max  = 0.f;
            float osee   = meas.compute_osee(grid, state.messages, -1, &n_cut, &s_max);
            float s_norm = (s_max > 0.f) ? osee / s_max : 0.f;
            printf("  %-6.4f/%-6.4f", osee, s_norm);
        }
        printf("\n");

        // Full spatial grid on subsequent lines for post-processing / animation
        if (measure_spatial) {
            printf("  n(r,t) [non-identity Pauli weight]:");
            for (int r = 0; r < grid.nrows; r++) {
                printf("\n    ");
                for (int c = 0; c < grid.ncols; c++)
                    printf("%.4f  ", spatial[grid.site_id(r,c)]);
            }
            printf("\n");
        }
    }

    printf("\nCholesky-QR: attempted=%d  fallback=%d  skipped=%d\n",
           ws.chol_attempted, ws.chol_fallback, ws.chol_skipped);
    printf("\nDone.\n");
    rocblas_destroy_handle(handle);
    return 0;
}
