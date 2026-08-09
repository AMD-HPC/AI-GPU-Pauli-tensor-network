/*
 * main_lightcone.cpp — Noiseless lightcone measurement
 *
 * Single state evolution + Pauli weight n(v,t) per site per cycle.
 * No noise, no inner products, no 2-GPU.
 *
 * Build: make PHYS_DIM=4 main_lightcone
 * Run:   ./main_lightcone [chi_max] [n_steps] [cutoff] [--center=N]
 *
 * Output:
 *   # header
 *   STEP <n>  C_nl <val>  chi <chi>  t <s>
 *   PAULI <n> <w_0> <w_1> ... <w_{n_sites-1}>
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
#ifdef RECT_7x7
#include "rect_7x7_ideal_kick_eps010_data.hpp"
#define GRID_TAG "rect_7x7"
static PtmFn s_bond_ptm = [](int b) -> const float* { return RECT_7x7_IDEAL_KICK_EPS010[b]; };
static const float* s_site_ptm = RECT_7x7_IDEAL_KICK_EPS010_SITE_PTM;
#else
#include "heavyhex_ideal_kick_eps010_data.hpp"
#define GRID_TAG "heavy_hex_3x3"
static PtmFn s_bond_ptm = [](int b) -> const float* { return HEAVY_HEX_3x3_IDEAL_KICK_EPS010[b]; };
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
    int   chi_max = (argc>1)?atoi(argv[1]):200;
    int   n_steps = (argc>2)?atoi(argv[2]):20;
    float cutoff  = (argc>3)?atof(argv[3]):2e-3f;
    int   center  = 2;  // site 13 in FlatHeavyHexGrid ordering

    for (int i=1; i<argc; i++) {
        std::string a(argv[i]);
        if (a.substr(0,9) == "--center=") center = atoi(argv[i]+9);
    }

    printf("# main_lightcone — noiseless Pauli weight\n");
    printf("# epsilon=%.4f  chi_max=%d  n_steps=%d  cutoff=%.0e  center=%d\n",
           HEAVY_HEX_3x3_IDEAL_KICK_EPS010_EPSILON, chi_max, n_steps, (double)cutoff, center);
    printf("# STEP C_nl chi t(s)\n");
    printf("# PAULI <step> <w_0> ... <w_%d>\n", 67);
    fflush(stdout);

    rocblas_handle handle; rocblas_create_handle(&handle);

    FlatHeavyHexGrid grid(chi_max);
    Workspace        ws(chi_max, 3);
    HeavyHexState    state(grid);
    Measures         meas(handle, ws);
    GateOps          gate_ops(handle);

    state.init_heisenberg(grid, center);

    for (int step = 1; step <= n_steps; step++) {
        double t0 = now_ms();

        // ── Trotter step (topology-agnostic) ─────────────────────────────────
        do_trotter(state.psi, state.messages, static_cast<Grid&>(grid),
                   s_bond_ptm, nullptr, s_site_ptm,
                   ws, gate_ops, chi_max, cutoff);

        // ── C(t) ─────────────────────────────────────────────────────────────
        float C_nl = meas.autocorrelation_with_bp(
            static_cast<Grid&>(grid), state.psi, state.messages, center, BP_SWEEPS);

        // ── chi ──────────────────────────────────────────────────────────────
        int chi_max_actual = 1;
        for (auto& b:grid.bonds) chi_max_actual=std::max(chi_max_actual,b.chi);

        double t = (now_ms()-t0)/1000.0;
        printf("STEP %d  C_nl %+.6f  chi %d  t %.2f\n", step, C_nl, chi_max_actual, t);

        // ── Pauli weights n(v,t) ─────────────────────────────────────────────
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
