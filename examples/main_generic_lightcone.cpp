/*
 * main_generic_lightcone.cpp — Topology-agnostic lightcone measurement.
 *
 * Runs noiseless Trotter evolution on any pre-built grid and outputs
 * the Pauli weight profile n(v,t) per site per cycle — same format as
 * main_lightcone.cpp but for any topology.
 *
 * Graph and PTMs are selected at compile time via -DGRAPH=<name>:
 *   -DGRAPH=heavy_hex_3x3   (default, 68 sites, 3 colors)
 *   -DGRAPH=rect_7x7        (49 sites, 2 colors)
 *   -DGRAPH=rect_8x8        (64 sites, 2 colors)
 *
 * To add a new graph:
 *   1. Create data/graphs/<name>.json
 *   2. python3 scripts/build_ptm.py --graph data/graphs/<name>.json \
 *              --out include/<name>_ideal_kick_eps010_data.hpp
 *   3. make PHYS_DIM=4 DFLAGS=-DGRAPH=<name> main_generic_lightcone
 *
 * Build:
 *   make PHYS_DIM=4 main_generic_lightcone              # heavy_hex_3x3
 *   make PHYS_DIM=4 DFLAGS=-DGRAPH=rect_7x7 main_generic_lightcone
 *
 * Run:
 *   ./main_generic_lightcone [chi_max] [n_steps] [cutoff] [--center=N]
 *
 * Output: STEP + PAULI lines (same as main_lightcone)
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "measures.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"
#include "trotter.hpp"

// ── Graph selection at compile time ──────────────────────────────────────────
#if defined(GRAPH_rect_7x7)
  #include "rect_7x7.hpp"
  #include "rect_7x7_ideal_kick_eps010_data.hpp"
  #define GRID_TAG      "rect_7x7"
  #define GRID_CLASS    FlatRect7x7Grid
  static PtmFn  s_bond_ptm = [](int b) -> const float* { return RECT_7x7_IDEAL_KICK_EPS010[b]; };
  static const float* s_site_ptm = RECT_7x7_IDEAL_KICK_EPS010_SITE_PTM;

#else
  // Default: heavy_hex_3x3
  #include "heavyhex_ideal_kick_eps010_data.hpp"
  #define GRID_TAG      "heavy_hex_3x3"
  #define GRID_CLASS    FlatHeavyHexGrid
  static PtmFn  s_bond_ptm = [](int b) -> const float* { return HEAVY_HEX_3x3_IDEAL_KICK_EPS010[b]; };
  static const float* s_site_ptm = HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM;
#endif

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

static constexpr int BP_SWEEPS = 100;

static double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    int   chi_max = 70;
    int   n_steps = 20;
    float cutoff  = 1e-3f;
    int   center  = -1;  // -1 = use default from grid

    for (int i=1;i<argc;i++) {
        std::string a(argv[i]);
        if (a.substr(0,9)=="--center=") center = atoi(argv[i]+9);
        else if (i==1) chi_max = atoi(argv[i]);
        else if (i==2) n_steps = atoi(argv[i]);
        else if (i==3) cutoff  = atof(argv[i]);
    }

    GRID_CLASS grid(chi_max);

    // Auto-select center: highest-degree site (most interior)
    if (center < 0) {
        int max_deg = 0;
        for (auto& site : grid.sites) {
            int deg=0;
            for (int d=0;d<MAX_DEGREE;d++) if(site.bond_id[d]>=0) deg++;
            if (deg > max_deg) { max_deg=deg; center=site.id; }
        }
    }

    printf("# main_generic_lightcone — %s\n", GRID_TAG);
    printf("# sites=%d  bonds=%d  colors=%d  chi_max=%d  n_steps=%d  cutoff=%.0e  center=%d\n",
           grid.n_sites, grid.n_bonds, (int)grid.colors.size(),
           chi_max, n_steps, (double)cutoff, center);
    printf("# STEP C_nl chi t(s)\n");
    printf("# PAULI <step> <w_0> ... <w_{n_sites-1}>\n");
    fflush(stdout);

    rocblas_handle handle; rocblas_create_handle(&handle);
    // Workspace sized by max_degree of grid sites.
    // Note: degree-4 (rect) needs chi^3 workspace — use small chi_max for large grids.
    int max_deg = 1;
    for (auto& site : grid.sites) {
        int deg=0; for(int d=0;d<MAX_DEGREE;d++) if(site.bond_id[d]>=0) deg++;
        max_deg = std::max(max_deg, deg);
    }
    // Safety: warn if workspace will be large
    {
        long long m = 1;
        for(int i=0;i<max_deg-1;i++) m*=chi_max;
        if(m > 100000)
            fprintf(stderr,"[WARNING] Workspace m_max=%lld — consider smaller chi_max for degree-%d grid\n",m,max_deg);
    }
    Workspace   ws(chi_max, max_deg);
    HeavyHexState state(grid);
    Measures    meas(handle, ws);
    GateOps     gate_ops(handle);

    state.init_heisenberg(grid, center);

    for (int step=1; step<=n_steps; step++) {
        double t0 = now_ms();

        do_trotter(state.psi, state.messages, static_cast<Grid&>(grid),
                   s_bond_ptm, nullptr, s_site_ptm,
                   ws, gate_ops, chi_max, cutoff);

        float C_nl = meas.autocorrelation_with_bp(
            static_cast<Grid&>(grid), state.psi, state.messages,
            center, BP_SWEEPS);

        int chi_actual = 1;
        for (auto& b : grid.bonds) chi_actual = std::max(chi_actual, b.chi);

        double t = (now_ms()-t0)/1000.0;
        printf("STEP %d  C_nl %+.6f  chi %d  t %.2f\n", step, C_nl, chi_actual, t);

        std::vector<float> nid(grid.n_sites);
        meas.compute_pauli_profile(static_cast<Grid&>(grid), state.psi, nid.data());
        printf("PAULI %d", step);
        for (int v=0; v<grid.n_sites; v++) printf(" %.4f", nid[v]);
        printf("\n");
        fflush(stdout);
    }

    rocblas_destroy_handle(handle);
    return 0;
}
