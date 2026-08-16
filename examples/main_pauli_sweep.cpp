/*
 * main_pauli_sweep.cpp — Pauli weight sweep for frontier calibration.
 *
 * Runs 4 states (noiseless + noisy γ=1,2,3) on a SINGLE GPU and outputs
 * PAULI_NL + PAULI_N1/N2/N3 per cycle. No inner products, no 2-GPU overhead.
 *
 * Graph and PTMs selected at compile time:
 *   default:           heavy_hex_3x3
 *   -DGRAPH_rect_7x7:  rect_7x7
 *
 * Build:
 *   make PHYS_DIM=4 main_pauli_sweep                    # honeycomb
 *   make PHYS_DIM=4 DFLAGS=-DGRAPH_rect_7x7 main_pauli_sweep_rect7x7
 *
 * Run:
 *   ./main_pauli_sweep [chi_max] [n_steps] [cutoff] [--center=N] [--v_hop=F]
 *
 *   --v_hop=F       : shared lightcone (all states same frontier, required for inner products)
 *   --v_hops=a,b,c,d: per-state frontiers NL,N1,N2,N3 (wavemap-only, max savings)
 *
 * Use scripts/measure_vhop.py to extract the correct v_hop values from a
 * cheap pre-flight run (small chi). Always use threshold-consistent values:
 * measure and allocate at the same epsilon_th.
 *
 * Output per cycle:
 *   STEP N  C_nl <val>  C_n1 <val>  C_n2 <val>  C_n3 <val>  chi <chi>  t <s>
 *   PAULI_NL N  <w_0> ... <w_{n_sites-1}>
 *   PAULI_N1 N  <w_0> ... <w_{n_sites-1}>
 *   PAULI_N2 N  <w_0> ... <w_{n_sites-1}>
 *   PAULI_N3 N  <w_0> ... <w_{n_sites-1}>
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

// ── Graph + PTM selection ─────────────────────────────────────────────────────
#ifdef GRAPH_heavyhex_dep
  // Heavy-hex with UNIFORM synthetic depolarizing noise (p=0.01 per bond)
  // Isolates topology effect from hardware noise heterogeneity
  #include "heavyhex_ideal_kick_eps010_data.hpp"
  #include "heavyhex_dep_eps010_data.hpp"
  #define GRID_TAG   "heavy_hex_3x3_dep"
  #define GRID_CLASS FlatHeavyHexGrid
  #define N_COLORS   3
  static PtmFn s_ptm_nl = [](int b)->const float*{ return HEAVY_HEX_3x3_IDEAL_KICK_EPS010[b]; };
  static PtmFn s_ptm_n1 = [](int b)->const float*{ return HEAVY_HEX_3x3_DEP_IDEAL_KICK_EPS010_GAMMA1_0[b]; };
  static PtmFn s_ptm_n2 = [](int b)->const float*{ return HEAVY_HEX_3x3_DEP_IDEAL_KICK_EPS010_GAMMA2_0[b]; };
  static PtmFn s_ptm_n3 = [](int b)->const float*{ return HEAVY_HEX_3x3_DEP_IDEAL_KICK_EPS010_GAMMA3_0[b]; };
  static const float* s_site_ptm = HEAVY_HEX_3x3_DEP_IDEAL_KICK_EPS010_SITE_PTM;
#elif defined(GRAPH_rect_7x7)
  #include "rect_7x7.hpp"
  #include "rect_7x7_ideal_kick_eps010_data.hpp"
  #include "rect_7x7_noisy_eps010_gamma1_data.hpp"
  #include "rect_7x7_noisy_eps010_gamma2_data.hpp"
  #include "rect_7x7_noisy_eps010_gamma3_data.hpp"
  #define GRID_TAG   "rect_7x7"
  #define GRID_CLASS FlatRect7x7Grid
  #define N_COLORS   4
  static PtmFn s_ptm_nl = [](int b)->const float*{ return RECT_7x7_IDEAL_KICK_EPS010[b]; };
  static PtmFn s_ptm_n1 = [](int b)->const float*{ return RECT_7x7_IDEAL_KICK_EPS010_GAMMA1_GAMMA1_0[b]; };
  static PtmFn s_ptm_n2 = [](int b)->const float*{ return RECT_7x7_IDEAL_KICK_EPS010_GAMMA2_GAMMA2_0[b]; };
  static PtmFn s_ptm_n3 = [](int b)->const float*{ return RECT_7x7_IDEAL_KICK_EPS010_GAMMA3_GAMMA3_0[b]; };
  static const float* s_site_ptm = RECT_7x7_IDEAL_KICK_EPS010_SITE_PTM;
#else
  // Default: heavy-hex with hardware noise PTMs
  #include "heavyhex_ideal_kick_eps010_data.hpp"
  #include "heavyhex_noisy_eps010_gamma1_data.hpp"
  #include "heavyhex_noisy_eps010_gamma2_data.hpp"
  #include "heavyhex_noisy_eps010_gamma3_data.hpp"
  #define GRID_TAG   "heavy_hex_3x3"
  #define GRID_CLASS FlatHeavyHexGrid
  #define N_COLORS   3
  static PtmFn s_ptm_nl = [](int b)->const float*{ return HEAVY_HEX_3x3_IDEAL_KICK_EPS010[b]; };
  static PtmFn s_ptm_n1 = [](int b)->const float*{ return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA1_GAMMA1_0[b]; };
  static PtmFn s_ptm_n2 = [](int b)->const float*{ return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA2_GAMMA2_0[b]; };
  static PtmFn s_ptm_n3 = [](int b)->const float*{ return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA3_GAMMA3_0[b]; };
  static const float* s_site_ptm = HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM;
#endif

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

static constexpr int BP_SWEEPS = 100;

static double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    int   chi_max = (argc>1)?atoi(argv[1]):70;
    int   n_steps = (argc>2)?atoi(argv[2]):20;
    float cutoff  = (argc>3)?atof(argv[3]):1e-3f;
    int   center  = -1;     // auto
    float v_hop   = -1.0f;  // shared frontier (for inner products, or single state)
    float v_hops[4] = {-1.f,-1.f,-1.f,-1.f};  // per-state: NL,N1,N2,N3
    bool  per_state = false;

    for (int i=1; i<argc; i++) {
        std::string a(argv[i]);
        if (a.substr(0,9) =="--center=") center = atoi(argv[i]+9);
        if (a.substr(0,8) =="--v_hop=") v_hop  = atof(argv[i]+8);
        if (a.substr(0,9)=="--v_hops=") {
            // Parse comma-separated list: NL,N1,N2,N3
            std::string vals = a.substr(9);
            int idx = 0;
            size_t pos = 0;
            while (idx < 4 && pos != std::string::npos) {
                size_t next = vals.find(',', pos);
                v_hops[idx++] = std::stof(vals.substr(pos, next-pos));
                pos = (next == std::string::npos) ? next : next+1;
            }
            per_state = true;
        }
    }
    // If single --v_hop given, broadcast to all states
    if (v_hop > 0.0f && !per_state)
        for (int g=0; g<4; g++) v_hops[g] = v_hop;

    // Each state needs its own grid — bond.chi is mutable and updated by Trotter.
    GRID_CLASS* grids[4];
    for (int g=0; g<4; g++) grids[g] = new GRID_CLASS(chi_max);
    GRID_CLASS& grid = *grids[0];  // reference for metadata

    // Auto center: highest-degree site
    if (center < 0) {
        int max_deg=0;
        for (auto& s:grid.sites){
            int d=0; for(int k=0;k<MAX_DEGREE;k++) if(s.bond_id[k]>=0) d++;
            if(d>max_deg){max_deg=d;center=s.id;}
        }
    }

    // Lightcone allocation: apply per-state v_hops before state construction.
    // --v_hop=F:         shared frontier (same for all states, required for inner products)
    // --v_hops=a,b,c,d: per-state frontiers NL,N1,N2,N3 (wavemap-only, max savings)
    // No flag:           full chi_max everywhere (default, backward compatible)
    static const char* state_names[4] = {"NL","N1","N2","N3"};
    bool any_lc = false;
    for (int g=0; g<4; g++) if (v_hops[g] > 0.0f) { any_lc = true; break; }
    if (any_lc) {
        float full_mb = grids[0]->lightcone_memory_mb(-1, n_steps, -1.0f);
        printf("# Lightcone allocation (full=%.1f MB):\n", full_mb);
        for (int g=0; g<4; g++) {
            if (v_hops[g] <= 0.0f) continue;
            float lc_mb = grids[g]->lightcone_memory_mb(center, n_steps, v_hops[g]);
            printf("#   %s: v_hop=%.4f  %.1f MB  (%.0f%% saving)\n",
                   state_names[g], v_hops[g], lc_mb,
                   100.0f*(1.0f - lc_mb/full_mb));
            grids[g]->apply_lightcone(center, n_steps, v_hops[g]);
        }
    }

    // Max degree for workspace
    int max_deg=0;
    for (auto& s:grid.sites){
        int d=0; for(int k=0;k<MAX_DEGREE;k++) if(s.bond_id[k]>=0) d++;
        max_deg=std::max(max_deg,d);
    }

    printf("# main_pauli_sweep — %s\n", GRID_TAG);
    printf("# chi=%d  n_steps=%d  cutoff=%.0e  center=%d\n",
           chi_max, n_steps, (double)cutoff, center);
    printf("# STEP C_nl C_n1 C_n2 C_n3  chi  t(s)\n");
    printf("# PAULI_NL/N1/N2/N3 <step> <w_0> ... <w_{n_sites-1}>\n");
    fflush(stdout);

    rocblas_handle handle; rocblas_create_handle(&handle);
    // Each state needs its own Workspace — shared workspace gets corrupted
    // after Trotter+BP of one state bleeds into the next.
    Workspace ws0(chi_max, max_deg);
    Workspace ws1(chi_max, max_deg);
    Workspace ws2(chi_max, max_deg);
    Workspace ws3(chi_max, max_deg);
    Workspace* wss[4] = {&ws0, &ws1, &ws2, &ws3};
    GateOps gate_ops(handle);

    // 4 independent states — each with its own grid
    HeavyHexState* states[4];
    for (int g=0; g<4; g++) {
        states[g] = new HeavyHexState(static_cast<Grid&>(*grids[g]));
        states[g]->init_heisenberg(static_cast<Grid&>(*grids[g]), center);
    }

    PtmFn ptms[4] = {s_ptm_nl, s_ptm_n1, s_ptm_n2, s_ptm_n3};
    const char* labels[4] = {"NL","N1","N2","N3"};

    for (int step=1; step<=n_steps; step++) {
        double t0 = now_ms();

        // Per state: Trotter → autocorrelation → PAULI (avoid workspace interference)
        float C[4];
        std::vector<float> nid(grid.n_sites);
        for (int g=0; g<4; g++) {
            do_trotter(states[g]->psi, states[g]->messages,
                       static_cast<Grid&>(*grids[g]),
                       ptms[g], nullptr, s_site_ptm,
                       *wss[g], gate_ops, chi_max, cutoff);
            Measures meas_g(handle, *wss[g]);
            C[g] = meas_g.autocorrelation_with_bp(
                static_cast<Grid&>(*grids[g]), states[g]->psi,
                states[g]->messages, center, BP_SWEEPS);
        }

        int chi_actual=1;
        for (auto& b:grids[0]->bonds) chi_actual=std::max(chi_actual,b.chi);

        double t=(now_ms()-t0)/1000.0;
        printf("STEP %d  C_nl %+.6f  C_n1 %+.6f  C_n2 %+.6f  C_n3 %+.6f  chi %d  t %.2f\n",
               step, C[0], C[1], C[2], C[3], chi_actual, t);

        // Pauli weights all 4 states
        for (int g=0; g<4; g++) {
            Measures meas_g(handle, *wss[g]);
            meas_g.compute_pauli_profile(static_cast<Grid&>(*grids[g]),
                                         states[g]->psi, nid.data());
            printf("PAULI_%s %d", labels[g], step);
            for (int v=0; v<grid.n_sites; v++) printf(" %.7f", nid[v]);
            printf("\n");
        }
        fflush(stdout);
    }

    for (int g=0; g<4; g++) { delete states[g]; delete grids[g]; }
    rocblas_destroy_handle(handle);
    return 0;
}
