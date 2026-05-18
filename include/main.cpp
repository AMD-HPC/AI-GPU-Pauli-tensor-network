/*
 * main.cpp — 4×4 Ising Heisenberg picture simulation
 *
 * Matches Julia's 2dIsing_dynamics_Heisenbergpicture_multi.jl
 *
 * Usage:
 *   ./main [nx] [ny] [chi_max] [n_steps] [dt] [h] [J] [cutoff]
 *
 * Defaults:
 *   nx=4, ny=4, chi_max=50, n_steps=20, dt=0.1, h=-1.0, J=-1.0, cutoff=1e-11
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

static constexpr int BP_MAX_SWEEPS = 100; // maximum BP sweeps; exits early on convergence

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

    printf("═══════════════════════════════════════════════\n");
    printf("  CppSim — Ising Heisenberg Picture\n");
    printf("  BUILD: %s %s\n", __DATE__, __TIME__);
    printf("  Bethe edge corr, BP observable-specific (d=0/d=3)\n");
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
    IsingSimulator sim(handle, h_field, J_coup, dt);

    // Center site
    // Julia uses 1-indexed (div(nx,2), div(ny,2)); convert to 0-indexed by subtracting 1.
    int center = grid.site_id(nx/2-1, ny/2-1);
    printf("Center site: (%d,%d) id=%d\n\n", nx/2-1, ny/2-1, center);

    // Initialize state: Z at center, I elsewhere
    state.init_heisenberg(grid, center);

    printf("Memory:\n");
    printf("  State tensors + messages: %.1f MB\n",
           state.total_bytes(grid)/1e6);
    printf("  Workspace: %.1f MB\n", ws.total_bytes/1e6);
    printf("\n");

    // ── Print header ──────────────────────────────────────────────
    printf("%-6s %-8s %-8s %-12s %-10s  %-12s %-10s %-10s\n",
           "Layer", "chi_min", "chi_max", "C(t)", "total(s)",
           "trotter(ms)", "bp(ms)", "Cfn(ms)");
    printf("%-6s %-8s %-8s %-12s %-10s  %-12s %-10s %-10s\n",
           "------", "--------", "--------", "------------", "----------",
           "------------", "----------", "----------");

    // ── t=0 measurement ───────────────────────────────────────────
    printf("%-6d %-8d %-8d %-12.6f %-10s  %-12s %-10s %-10s\n",
           0, 1, 1, 1.0f, "-", "-", "-", "-");

    // ── Trotter evolution ─────────────────────────────────────────
    for (int layer=1; layer<=n_steps; layer++) {
        double t0 = now_ms();

        int max_chi = sim.trotter_layer(state, grid, ws, chi_max, cutoff, &meas, center);

        double t_trotter = now_ms() - t0;

        int min_chi = max_chi;
        for (auto& bond : grid.bonds)
            min_chi = std::min(min_chi, bond.chi);

        // BP on a copy — Vidal gauge state.messages left intact for next layer
        double t_bp=0, t_pf=0;
        Scalar C = meas.autocorrelation_with_bp(grid, state.psi,
                                                 state.messages, center, BP_MAX_SWEEPS,
                                                 &t_bp, &t_pf);

        double dt_total = now_ms() - t0;

        printf("%-6d %-8d %-8d %-12.6f %-10.3f  %-12.1f %-10.1f %-10.1f\n",
               layer, min_chi, max_chi, C, dt_total/1000.0,
               t_trotter, t_bp, t_pf);
    }

    printf("\nDone.\n");
    rocblas_destroy_handle(handle);
    return 0;
}
