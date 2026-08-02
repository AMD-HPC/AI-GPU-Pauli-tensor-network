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
#include "heavyhex_ideal_kick_eps010_data.hpp"

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

        // ── Trotter step ──────────────────────────────────────────────────────
        {
            int nm = 2*grid.n_bonds;
            std::vector<int>     frozen_chi(grid.n_bonds);
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
                    int chi2=bond.chi; size_t sz=(size_t)chi2*chi2*sizeof(Scalar);
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
                    int chi2=bond.chi; size_t sz=(size_t)chi2*chi2*sizeof(Scalar);
                    for (int d=0;d<2;d++) {
                        int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
                        HIP_CHECK(hipMemcpy(state.messages[idx],new_msg[idx],sz,hipMemcpyDeviceToDevice));
                    }
                }
            }
            for (auto p:frozen_msg) if(p)HIP_CHECK(hipFree(p));
            for (auto p:new_msg)    if(p)HIP_CHECK(hipFree(p));
            // kick
            size_t tsz=0;
            for (auto& site:grid.sites){int vol=D;for(int d=0;d<MAX_DEGREE;d++)if(site.bond_id[d]>=0)vol*=grid.bonds[site.bond_id[d]].chi;tsz=std::max(tsz,(size_t)vol);}
            Scalar* tmp; HIP_CHECK(hipMalloc(&tmp,tsz*sizeof(Scalar)));
            for (auto& site:grid.sites)
                gate_ops.apply_unary_gate(state.psi[site.id],site,grid,
                    HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM,tmp);
            HIP_CHECK(hipFree(tmp));
        }

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
