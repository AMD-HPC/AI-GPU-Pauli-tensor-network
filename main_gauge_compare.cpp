/*
 * main_gauge_compare.cpp — QR gauge vs SQRT gauge inner product comparison
 *
 * GPU 0: Trotter (noiseless + 3 noisy states) + BP → C(t)
 * GPU 1: IP_QR via double-layer BP (BLF_GPU2)
 *         IP_SQ via asymmetric gauge GPU kernel (SQRT_GAUGE_IP)
 *
 * For each pair (nl,n1), (nl,n2), (nl,n3) and diagonal (n1,n1) etc:
 *   IP_QR[i,j] = BLF_GPU2(psi_i, psi_j)        — double-layer BP, QR gauge
 *   IP_SQ[i,j] = SQRT_GAUGE_IP(psi_i, msg_i, psi_j) — asymmetric gauge, no BP
 *
 * Question: do IP_QR and IP_SQ agree? Is the bias constant across cycles?
 *
 * Build: make PHYS_DIM=4 main_gauge_compare
 * Run:   ./main_gauge_compare [chi_max] [n_steps] [cutoff] [center]
 *
 * Output per step:
 *   STEP N  C_nl C_n1 C_n2 C_n3
 *   QR  N   A00 A01 A02 A11 A12 A22  B0 B1 B2
 *   SQ  N   A00 A01 A02 A11 A12 A22  B0 B1 B2
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
#include "heavyhex_noisy_eps010_gamma1_data.hpp"
#include "heavyhex_noisy_eps010_gamma2_data.hpp"
#include "heavyhex_noisy_eps010_gamma3_data.hpp"
// Ergodic amplification: fixed gamma=2, varying eps=0.10/0.15/0.20
#include "heavyhex_ideal_kick_eps010_gamma2_data.hpp"
#include "heavyhex_ideal_kick_eps015_gamma2_data.hpp"
#include "heavyhex_ideal_kick_eps020_gamma2_data.hpp"
#include "bilinear_form_blf_gpu2.hpp"
#include "sqrt_gauge_ip.hpp"
#include "diag_gauge_ip.hpp"
#include "sqrt_bp_ip.hpp"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

static constexpr int BP_SWEEPS = 100;
static constexpr int N_GAMMA   = 3;

static double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

using PtmFunc = const float*(*)(int);
// Gamma-MPF: fixed eps=0.10, varying gamma=1/2/3
static const float* ptm_g1(int b){return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA1_GAMMA1_0[b];}
static const float* ptm_g2(int b){return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA2_GAMMA2_0[b];}
static const float* ptm_g3(int b){return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA3_GAMMA3_0[b];}
// Ergodic: fixed gamma=2, varying eps=0.10/0.15/0.20
static const float* ptm_e1(int b){return HEAVY_HEX_3x3_IDEAL_KICK_EPS010_GAMMA2[b];}
static const float* ptm_e2(int b){return HEAVY_HEX_3x3_IDEAL_KICK_EPS015_GAMMA2[b];}
static const float* ptm_e3(int b){return HEAVY_HEX_3x3_IDEAL_KICK_EPS020_GAMMA2[b];}

// Site PTM for ergodic: use eps=0.10 gamma=2 site PTM
static const float* ergodic_site_ptm = HEAVY_HEX_3x3_IDEAL_KICK_EPS010_GAMMA2_SITE_PTM;

// ── enable P2P ────────────────────────────────────────────────────────────────
static void enable_p2p(int n_gpus) {
    for (int i = 0; i < n_gpus; i++)
        for (int j = 0; j < n_gpus; j++)
            if (i != j) hipDeviceEnablePeerAccess(j, 0);
}

// ── Trotter step ──────────────────────────────────────────────────────────────
static void do_trotter(HeavyHexState& state, FlatHeavyHexGrid& grid,
                       PtmFunc ptm_fn, const float* nl_ptm,
                       Workspace& ws, GateOps& gate_ops,
                       int chi_max, float cutoff,
                       const float* site_ptm=HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM)
{
    int nm = 2*grid.n_bonds;
    size_t alloc = (size_t)chi_max*chi_max*sizeof(Scalar);
    std::vector<Scalar*> frozen_msg(nm,nullptr), new_msg(nm,nullptr);
    std::vector<int> frozen_chi(grid.n_bonds);
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
            const float* G = ptm_fn ? ptm_fn(gate.bond_id) : nl_ptm + gate.bond_id*256;
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
    // kick
    size_t tsz=0;
    for (auto& s:grid.sites) {
        int vol=D;
        for (int d=0;d<MAX_DEGREE;d++) if(s.bond_id[d]>=0) vol*=grid.bonds[s.bond_id[d]].chi;
        tsz=std::max(tsz,(size_t)vol);
    }
    Scalar* tmp; HIP_CHECK(hipMalloc(&tmp,tsz*sizeof(Scalar)));
    for (auto& s:grid.sites)
        gate_ops.apply_unary_gate(state.psi[s.id],s,grid,site_ptm,tmp);
    HIP_CHECK(hipFree(tmp));
}

int main(int argc, char** argv) {
    int    chi_max  = (argc>1)?atoi(argv[1]):70;
    int    n_steps  = (argc>2)?atoi(argv[2]):10;
    float  cutoff   = (argc>3)?atof(argv[3]):1e-3f;
    int    center   = (argc>4)?atoi(argv[4]):13;
    double eig_cut   = 1e-2;
    int    sb_sweeps = 0;
    bool   ergodic    = false;
    bool   qr_dg_only = false;  // skip SQ and SB
    bool   dg_only    = false;  // skip QR, SQ, SB — DG only (avoids BLF_GPU2 CPU fallback)
    for (int i=1;i<argc;i++) {
        std::string a(argv[i]);
        if (a.substr(0,10)=="--eig-cut=")  eig_cut  = atof(argv[i]+10);
        if (a.substr(0,12)=="--sb-sweeps=") sb_sweeps = atoi(argv[i]+12);
        if (a=="--ergodic")    ergodic    = true;
        if (a=="--qr-dg-only") qr_dg_only = true;
        if (a=="--dg-only")  { dg_only    = true; qr_dg_only = true; }
    }

    int n_gpus=0;
    HIP_CHECK(hipGetDeviceCount(&n_gpus));
    if (n_gpus < 2) { fprintf(stderr,"Need 2 GPUs\n"); return 1; }
    enable_p2p(n_gpus);

    printf("# main_gauge_compare — QR vs SQRT gauge inner product\n");
    printf("# chi=%d n_steps=%d cutoff=%.0e center=%d eig_cut=%.0e sb_sweeps=%d mode=%s\n",
           chi_max, n_steps, (double)cutoff, center, eig_cut, sb_sweeps,
           ergodic ? "ergodic(eps=0.10/0.15/0.20 gamma=2)" : "gamma-MPF(gamma=1/2/3 eps=0.10)");
    printf("# STEP  C_nl C_n1 C_n2 C_n3\n");
    printf("# QR/SQ N  A00 A01 A02 A11 A12 A22  B0 B1 B2\n");
    fflush(stdout);

    // ── GPU 0: Trotter + autocorrelation ──────────────────────────────────────
    HIP_CHECK(hipSetDevice(0));
    rocblas_handle h0; rocblas_create_handle(&h0);

    FlatHeavyHexGrid grid_nl(chi_max);
    Workspace        ws_nl(chi_max, 3);
    HeavyHexState    state_nl(grid_nl);
    Measures         meas(h0, ws_nl);
    GateOps          gate_ops0(h0);
    state_nl.init_heisenberg(grid_nl, center);

    PtmFunc ptm_fns[N_GAMMA];
    if (ergodic) { ptm_fns[0]=ptm_e1; ptm_fns[1]=ptm_e2; ptm_fns[2]=ptm_e3; }
    else         { ptm_fns[0]=ptm_g1; ptm_fns[1]=ptm_g2; ptm_fns[2]=ptm_g3; }
    FlatHeavyHexGrid* grids_n[N_GAMMA];
    HeavyHexState*    states_n[N_GAMMA];
    Workspace*        ws_n[N_GAMMA];
    for (int g=0;g<N_GAMMA;g++) {
        grids_n[g]  = new FlatHeavyHexGrid(chi_max);
        ws_n[g]     = new Workspace(chi_max,3);
        states_n[g] = new HeavyHexState(*grids_n[g]);
        states_n[g]->init_heisenberg(*grids_n[g], center);
    }

    // Pack noiseless PTMs contiguously for do_trotter
    int nb = grid_nl.n_bonds;
    std::vector<float> nl_ptm_buf(nb*256);
    for (int b=0;b<nb;b++)
        memcpy(nl_ptm_buf.data()+b*256, HEAVY_HEX_3x3_IDEAL_KICK_EPS010[b], 256*sizeof(float));

    // ── GPU 1: IP objects ─────────────────────────────────────────────────────
    HIP_CHECK(hipSetDevice(1));
    rocblas_handle h1; rocblas_create_handle(&h1);

    // QR gauge: BLF_GPU2 — one per pair (nl,ni) and (ni,nj)
    // B[i] = IP(nl, ni),  A[i][j] = IP(ni, nj)  i<=j
    BLF_GPU2* blf_B[N_GAMMA];
    BLF_GPU2* blf_A[N_GAMMA][N_GAMMA];
    for (int i=0;i<N_GAMMA;i++) {
        blf_B[i] = new BLF_GPU2(grid_nl, *grids_n[i], h1);
        for (int j=i;j<N_GAMMA;j++)
            blf_A[i][j] = new BLF_GPU2(*grids_n[i], *grids_n[j], h1);
    }

    // SQRT gauge: SQRT_GAUGE_IP — same pairs
    SQRT_GAUGE_IP* sq_B[N_GAMMA];
    SQRT_GAUGE_IP* sq_A[N_GAMMA][N_GAMMA];
    for (int i=0;i<N_GAMMA;i++) {
        sq_B[i] = new SQRT_GAUGE_IP(grid_nl, chi_max);
        for (int j=i;j<N_GAMMA;j++)
            sq_A[i][j] = new SQRT_GAUGE_IP(*grids_n[i], chi_max);
    }

    // DG gauge: 4 eigendecompositions (1 nl + 3 noisy), shared across 9 pairs
    rocblas_handle hdg; rocblas_create_handle(&hdg);
    DG_StateGauge* dg_gauge_nl = new DG_StateGauge(grid_nl, hdg, chi_max, eig_cut);
    DG_StateGauge* dg_gauge_n[N_GAMMA];
    for (int i=0;i<N_GAMMA;i++)
        dg_gauge_n[i] = new DG_StateGauge(*grids_n[i], hdg, chi_max, eig_cut);

    // 9 DG_Pair objects (no eigendecomposition — just GEMM + sdot)
    DG_Pair* dgp_B[N_GAMMA];     // B[i] = IP(nl, ni) — uses gauge_nl
    DG_Pair* dgp_A[N_GAMMA][N_GAMMA]={};  // A[i][j] = IP(ni, nj) — uses gauge_n[i]
    for (int i=0;i<N_GAMMA;i++) {
        dgp_B[i] = new DG_Pair(grid_nl, hdg, chi_max);
        for (int j=i;j<N_GAMMA;j++)
            dgp_A[i][j] = new DG_Pair(*grids_n[i], hdg, chi_max);
    }

    // SB gauge: symmetric SQRT gauge, diagonal messages, no eigendecomposition
    SQRT_BP_IP* sb_B[N_GAMMA];
    SQRT_BP_IP* sb_A[N_GAMMA][N_GAMMA];
    for (int i=0;i<N_GAMMA;i++) {
        sb_B[i] = new SQRT_BP_IP(grid_nl, *grids_n[i], chi_max, eig_cut);
        for (int j=i;j<N_GAMMA;j++)
            sb_A[i][j] = new SQRT_BP_IP(*grids_n[i], *grids_n[j], chi_max, eig_cut);
    }

    // ── Cycle loop ────────────────────────────────────────────────────────────
    for (int step=1; step<=n_steps; step++) {
        double t0 = now_ms();

        // ── GPU 0: Trotter all 4 states ───────────────────────────────────────
        HIP_CHECK(hipSetDevice(0));
        const float* noisy_site_ptm = ergodic
            ? ergodic_site_ptm
            : HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM;
        do_trotter(state_nl, grid_nl, nullptr, nl_ptm_buf.data(),
                   ws_nl, gate_ops0, chi_max, cutoff);
        for (int g=0;g<N_GAMMA;g++)
            do_trotter(*states_n[g], *grids_n[g], ptm_fns[g], nullptr,
                       *ws_n[g], gate_ops0, chi_max, cutoff, noisy_site_ptm);

        // Compute C(t) — uses temp copy of messages internally, does NOT modify state.messages
        float C_nl = meas.autocorrelation_with_bp(
            static_cast<Grid&>(grid_nl), state_nl.psi, state_nl.messages,
            center, BP_SWEEPS);
        float C_n[N_GAMMA];
        for (int g=0;g<N_GAMMA;g++)
            C_n[g] = meas.autocorrelation_with_bp(
                static_cast<Grid&>(*grids_n[g]), states_n[g]->psi,
                states_n[g]->messages, center, BP_SWEEPS);

        // DG gauge uses state.messages directly (post-Trotter, pre-BP).
        // BP-converged messages make all noisy states look identical → A_hat≈1.
        // Trotter messages retain state-specific structure → good discrimination.
        // No copy needed — DG upload reads from GPU 0, doesn't modify messages.

        printf("STEP %d  C_nl %+.6f  C_n1 %+.6f  C_n2 %+.6f  C_n3 %+.6f\n",
               step, C_nl, C_n[0], C_n[1], C_n[2]);

        // ── GPU 1: QR and SQRT inner products ─────────────────────────────────
        HIP_CHECK(hipSetDevice(1));

        // Upload to QR objects (skipped in --dg-only mode)
        if (!dg_only) for (int i=0;i<N_GAMMA;i++) {
            blf_B[i]->upload(state_nl.psi,      0, states_n[i]->psi, 0,
                             &state_nl.messages, &states_n[i]->messages);
            for (int j=i;j<N_GAMMA;j++)
                blf_A[i][j]->upload(states_n[i]->psi, 0, states_n[j]->psi, 0,
                                    &states_n[i]->messages, &states_n[j]->messages);
        }
        // Upload to SQRT objects (skipped in --qr-dg-only mode)
        if (!qr_dg_only) for (int i=0;i<N_GAMMA;i++) {
            sq_B[i]->upload(state_nl.psi, state_nl.messages, states_n[i]->psi);
            for (int j=i;j<N_GAMMA;j++)
                sq_A[i][j]->upload(states_n[i]->psi, states_n[i]->messages, states_n[j]->psi);
        }

        // DG gauge: 4 precomputes (shared across pairs)
        HIP_CHECK(hipSetDevice(0));
        dg_gauge_nl->precompute(state_nl.psi, state_nl.messages, 0);
        for (int i=0;i<N_GAMMA;i++)
            dg_gauge_n[i]->precompute(states_n[i]->psi, states_n[i]->messages, 0);

        // 9 pair contractions using precomputed gauges
        for (int i=0;i<N_GAMMA;i++) {
            dgp_B[i]->compute_pair(*dg_gauge_nl, states_n[i]->psi, 0);
            for (int j=i;j<N_GAMMA;j++)
                dgp_A[i][j]->compute_pair(*dg_gauge_n[i], states_n[j]->psi, 0);
        }
        HIP_CHECK(hipSetDevice(1));

        // Upload to SB objects (skipped in --qr-dg-only mode)
        if (!qr_dg_only) { HIP_CHECK(hipSetDevice(1));
        for (int i=0;i<N_GAMMA;i++) {
            sb_B[i]->upload(state_nl.psi,     0, state_nl.messages,
                            states_n[i]->psi,  0, states_n[i]->messages);
            for (int j=i;j<N_GAMMA;j++)
                sb_A[i][j]->upload(states_n[i]->psi, 0, states_n[i]->messages,
                                   states_n[j]->psi, 0, states_n[j]->messages);
        }}

        // Compute QR (skipped in --dg-only mode)
        float qr_B[N_GAMMA]={}, qr_A[N_GAMMA][N_GAMMA]={};
        if (!dg_only) for (int i=0;i<N_GAMMA;i++) {
            qr_B[i] = blf_B[i]->compute(25, 1e-5f);
            for (int j=i;j<N_GAMMA;j++)
                qr_A[i][j] = blf_A[i][j]->compute(25, 1e-5f);
        }

        // Compute SQRT (skipped in --qr-dg-only mode)
        float sq_B_val[N_GAMMA]={}, sq_A_val[N_GAMMA][N_GAMMA]={};
        if (!qr_dg_only) for (int i=0;i<N_GAMMA;i++) {
            sq_B_val[i] = sq_B[i]->compute();
            for (int j=i;j<N_GAMMA;j++)
                sq_A_val[i][j] = sq_A[i][j]->compute();
        }

        // DG gauge: Bethe PF (uses precomputed h_log_av from compute_pair)
        float dg_B_val[N_GAMMA]={}, dg_A_val[N_GAMMA][N_GAMMA]={};
        for (int i=0;i<N_GAMMA;i++) {
            dg_B_val[i] = dgp_B[i]->bethe(*dg_gauge_nl);
            for (int j=i;j<N_GAMMA;j++)
                dg_A_val[i][j] = dgp_A[i][j]->bethe(*dg_gauge_n[i]);
        }

        // Compute SB (skipped in --qr-dg-only mode)
        float sb_B_val[N_GAMMA]={}, sb_A_val[N_GAMMA][N_GAMMA]={};
        if (!qr_dg_only) for (int i=0;i<N_GAMMA;i++) {
            sb_B_val[i] = sb_B[i]->compute(sb_sweeps);
            for (int j=i;j<N_GAMMA;j++)
                sb_A_val[i][j] = sb_A[i][j]->compute(sb_sweeps);
        }

        double t1 = now_ms();

        // QR gauge — format matches analyze_ip_mpf.py parser
        printf("STEP %d  C_nl %+.6f  C_n %+.6f %+.6f %+.6f"
               "  IP_A %+.6f %+.6f %+.6f %+.6f %+.6f %+.6f"
               "  IP_B %+.6f %+.6f %+.6f  gauge QR  t %.2fs\n",
               step, C_nl, C_n[0], C_n[1], C_n[2],
               qr_A[0][0], qr_A[0][1], qr_A[0][2],
               qr_A[1][1], qr_A[1][2], qr_A[2][2],
               qr_B[0], qr_B[1], qr_B[2],
               (t1-t0)/1000.0);
        // SQ gauge — same format
        printf("STEP %d  C_nl %+.6f  C_n %+.6f %+.6f %+.6f"
               "  IP_A %+.6f %+.6f %+.6f %+.6f %+.6f %+.6f"
               "  IP_B %+.6f %+.6f %+.6f  gauge SQ\n",
               step, C_nl, C_n[0], C_n[1], C_n[2],
               sq_A_val[0][0], sq_A_val[0][1], sq_A_val[0][2],
               sq_A_val[1][1], sq_A_val[1][2], sq_A_val[2][2],
               sq_B_val[0], sq_B_val[1], sq_B_val[2]);
        // SB gauge — SQRT transform + BP identity warm-start
        printf("STEP %d  C_nl %+.6f  C_n %+.6f %+.6f %+.6f"
               "  IP_A %+.6f %+.6f %+.6f %+.6f %+.6f %+.6f"
               "  IP_B %+.6f %+.6f %+.6f  gauge SB\n",
               step, C_nl, C_n[0], C_n[1], C_n[2],
               sb_A_val[0][0], sb_A_val[0][1], sb_A_val[0][2],
               sb_A_val[1][1], sb_A_val[1][2], sb_A_val[2][2],
               sb_B_val[0], sb_B_val[1], sb_B_val[2]);
        // DG gauge — eigendecomposed, float64 CPU, no BP
        printf("STEP %d  C_nl %+.6f  C_n %+.6f %+.6f %+.6f"
               "  IP_A %+.6f %+.6f %+.6f %+.6f %+.6f %+.6f"
               "  IP_B %+.6f %+.6f %+.6f  gauge DG\n",
               step, C_nl, C_n[0], C_n[1], C_n[2],
               dg_A_val[0][0], dg_A_val[0][1], dg_A_val[0][2],
               dg_A_val[1][1], dg_A_val[1][2], dg_A_val[2][2],
               dg_B_val[0], dg_B_val[1], dg_B_val[2]);
        fflush(stdout);
    }

    // Cleanup
    HIP_CHECK(hipSetDevice(1));
    for (int i=0;i<N_GAMMA;i++) {
        delete blf_B[i]; delete sq_B[i]; delete sb_B[i];
        delete dg_gauge_n[i]; delete dgp_B[i];
        for (int j=i;j<N_GAMMA;j++) {
            delete blf_A[i][j]; delete sq_A[i][j]; delete sb_A[i][j];
            delete dgp_A[i][j];
        }
    }
    delete dg_gauge_nl;
    rocblas_destroy_handle(hdg);
    rocblas_destroy_handle(h1);
    HIP_CHECK(hipSetDevice(0));
    for (int g=0;g<N_GAMMA;g++) {
        delete states_n[g]; delete grids_n[g]; delete ws_n[g];
    }
    rocblas_destroy_handle(h0);
    return 0;
}
