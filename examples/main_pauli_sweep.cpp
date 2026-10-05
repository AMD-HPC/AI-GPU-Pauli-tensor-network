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
 *   --state=X       : run only state X (NL, N1, N2, N3, or comma-separated subset)
 *                     Default: all 4 states. Combine logs with cat for MPF scripts.
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
#include <sstream>
#include <vector>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

static constexpr int BP_SWEEPS = 100;

// ── Hamiltonian tomography: perturbed bond PTM ────────────────────────────────
static int    g_perturb_bond    = -1;
static float  g_perturbed_ptm[256];
static PtmFn  g_original_ptm_nl = nullptr;

static const float* perturbed_ptm_fn(int b) {
    if (b == g_perturb_bond) return g_perturbed_ptm;
    return g_original_ptm_nl(b);
}
// ─────────────────────────────────────────────────────────────────────────────

static double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    int   chi_max = (argc>1)?atoi(argv[1]):70;
    int   n_steps = (argc>2)?atoi(argv[2]):20;
    float cutoff  = (argc>3)?atof(argv[3]):1e-3f;
    int   center  = -1;     // auto (single center, legacy)
    std::vector<int> centers;  // multi-center seismic mode
    int   perturb_bond = -1;   // --perturb-bond=B,J: bond index to perturb
    float perturb_J    = 1.0f; // perturbed coupling (default: uniform J=1)
    float v_hop   = -1.0f;  // shared frontier (for inner products, or single state)
    float v_hops[4] = {-1.f,-1.f,-1.f,-1.f};  // per-state: NL,N1,N2,N3
    bool  per_state = false;
    bool  active[4] = {true,true,true,true};  // which states to run
    int   diagnose_step = -1;  // --diagnose=N: print message diagnostics at step N
    // Bonds to diagnose (site pairs, 0-based) — hardcoded for heavy-hex guilty bonds
    // identified from n(v,t) analysis at step 9 chi=400 δ=1e-4
    std::vector<std::pair<int,int>> diag_bonds = {
        {5,36},{5,38},{7,40},{7,41},{19,47},{19,56},{19,58},{23,52},{23,62}
    };

    QrMode qr_mode = QrMode::Householder;
    float  chol_reg = 1e-8f;  // --chol-reg=X
    for (int i=1; i<argc; i++) {
        std::string a(argv[i]);
        if (a.substr(0,11)=="--diagnose=") diagnose_step = atoi(argv[i]+11);
        if (a.substr(0,15)=="--perturb-bond=") {
            // --perturb-bond=12,1.2  (bond index, new J value)
            std::string val = a.substr(15);
            auto comma = val.find(',');
            perturb_bond = atoi(val.substr(0,comma).c_str());
            perturb_J    = (comma!=std::string::npos) ? atof(val.substr(comma+1).c_str()) : 1.2f;
        }
        if (a.substr(0,9) =="--center=") {
            // Parse comma-separated list: --center=13 or --center=13,24,10
            std::string val = a.substr(9);
            std::stringstream ss(val);
            std::string tok;
            while (std::getline(ss, tok, ',')) centers.push_back(atoi(tok.c_str()));
            center = centers[0];  // legacy single center = first
        }
        if (a.substr(0,8) =="--v_hop=") v_hop  = atof(argv[i]+8);
        if (a=="--qr=cholesky")    qr_mode = QrMode::Cholesky;
        if (a=="--qr=householder") qr_mode = QrMode::Householder;
        if (a=="--qr=adaptive")    qr_mode = QrMode::Adaptive;
        if (a.substr(0,11)=="--chol-reg=") chol_reg = atof(argv[i]+11);
        if (a.substr(0,8) =="--state=") {
            // --state=NL or --state=N1 or --state=NL,N1,N2 etc.
            // Sets which states to run; all others are skipped.
            std::string sel = a.substr(8);
            for (int g=0; g<4; g++) active[g] = false;
            static const char* snames[4] = {"NL","N1","N2","N3"};
            for (int g=0; g<4; g++)
                if (sel.find(snames[g]) != std::string::npos) active[g] = true;
        }
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

    // Each active state needs its own grid — bond.chi is mutable and updated by Trotter.
    GRID_CLASS* grids[4] = {nullptr,nullptr,nullptr,nullptr};
    for (int g=0; g<4; g++)
        if (active[g]) grids[g] = new GRID_CLASS(chi_max);
    // Reference grid for metadata (use first active)
    int ref_g = 0; while (ref_g < 4 && !active[ref_g]) ref_g++;
    GRID_CLASS& grid = *grids[ref_g];

    // Auto center: highest-degree site (use first active grid)
    if (center < 0) {
        int max_deg=0;
        for (auto& s:grids[ref_g]->sites){
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
        float full_mb = grids[ref_g]->lightcone_memory_mb(-1, n_steps, -1.0f);
        printf("# Lightcone allocation (full=%.1f MB):\n", full_mb);
        for (int g=0; g<4; g++) {
            if (!active[g] || v_hops[g] <= 0.0f) continue;
            float lc_mb = grids[g]->lightcone_memory_mb(center, n_steps, v_hops[g]);
            printf("#   %s: v_hop=%.4f  %.1f MB  (%.0f%% saving)\n",
                   state_names[g], v_hops[g], lc_mb,
                   100.0f*(1.0f - lc_mb/full_mb));
            grids[g]->apply_lightcone(center, n_steps, v_hops[g]);
        }
    }

    // Max degree for workspace (use first active grid)
    int max_deg=0;
    for (auto& s:grids[ref_g]->sites){
        int d=0; for(int k=0;k<MAX_DEGREE;k++) if(s.bond_id[k]>=0) d++;
        max_deg=std::max(max_deg,d);
    }

    printf("# main_pauli_sweep — %s\n", GRID_TAG);
    // Print all centers
    printf("# centers=");
    for (size_t ci=0; ci<centers.size(); ci++) printf("%s%d", ci?",":"", centers[ci]);
    printf("\n");
    printf("# chi=%d  n_steps=%d  cutoff=%.0e  center=%d  Scalar=%s (%zu bytes/elem)  qr=%s  chol_reg=%.0e\n",
           chi_max, n_steps, (double)cutoff, center,
           (sizeof(Scalar)==8?"double":"float"), sizeof(Scalar),
           qr_mode==QrMode::Cholesky?"cholesky":qr_mode==QrMode::Adaptive?"adaptive":"householder",
           (double)chol_reg);
    printf("# STEP C_nl C_n1 C_n2 C_n3  chi  t(s)\n");
    printf("# PAULI_NL/N1/N2/N3 <step> <w_0> ... <w_{n_sites-1}>\n");
    fflush(stdout);

    rocblas_handle handle; rocblas_create_handle(&handle);
    // Each active state needs its own Workspace.
    Workspace* ws_storage[4] = {nullptr,nullptr,nullptr,nullptr};
    Workspace* wss[4] = {nullptr,nullptr,nullptr,nullptr};
    for (int g=0; g<4; g++)
        if (active[g]) { ws_storage[g] = new Workspace(chi_max, max_deg); wss[g] = ws_storage[g]; }
    GateOps gate_ops(handle);
    gate_ops.qr_mode = qr_mode;
    gate_ops.CHOL_DIAG_REG = (Scalar)chol_reg;

    // Active states only — each with its own grid
    HeavyHexState* states[4] = {nullptr,nullptr,nullptr,nullptr};
    for (int g=0; g<4; g++) {
        if (!active[g]) continue;
        states[g] = new HeavyHexState(static_cast<Grid&>(*grids[g]));
        if (centers.size() > 1)
            states[g]->init_heisenberg(static_cast<Grid&>(*grids[g]), centers);
        else
            states[g]->init_heisenberg(static_cast<Grid&>(*grids[g]), center);
    }

    // ── Runtime bond perturbation (Hamiltonian tomography game) ──────────────
    // --perturb-bond=B,J: override bond B's NL PTM with coupling J*eps rotation
    // Physically correct: recomputes the Heisenberg gate at new coupling J.
    // Only affects s_ptm_nl (noiseless) — use --state=NL for clean tomography.
    //
    // Heisenberg gate PTM for exp(-i*theta*(XX+YY+ZZ)), theta=J*eps:
    // In Pauli basis the gate acts as a rotation. The 16x16 PTM has the structure:
    //   M[0,0] = 1  (identity preserved)
    //   Certain off-diagonal blocks rotate by 2*theta
    // We compute it from the base PTM by replacing the rotation angle.
    if (perturb_bond >= 0) {
        // Compute the perturbed PTM analytically for exp(-i*J_new*eps*(XX+YY+ZZ))
        // The Heisenberg gate in the 2-site Pauli basis {II,IX,...,ZZ} (16 elements)
        // is a block rotation. Key elements (theta = J*eps):
        //   cos(2*theta) and sin(2*theta) appear in XX/YY/ZZ blocks
        // We start from the base PTM (J=1) and replace theta:
        // The base PTM encodes theta_old = 1.0*eps. We need theta_new = J_new*eps.
        // The PTM structure for the Heisenberg gate (from build_ptm.py):
        // Key non-trivial elements involve cos(4*theta) and sin(4*theta).
        // Rather than recomputing from scratch, we read the base PTM
        // and rescale the off-diagonal elements using the ratio of angles:
        //   For uniform J=1 grid all PTMs are identical — we copy element-by-element
        //   and replace cos/sin factors analytically.
        const float eps     = 0.10f;
        const float th_old  = 1.0f * eps;   // J=1, eps=0.10
        const float th_new  = perturb_J * eps;
        // The Heisenberg PTM key trig values (from build_ptm.py analysis):
        // c2o = cos(2*th_old), s2o = sin(2*th_old) etc.
        const float c2n = cosf(2.0f*th_new), s2n = sinf(2.0f*th_new);
        const float c2o = cosf(2.0f*th_old), s2o = sinf(2.0f*th_old);
        const float c4n = cosf(4.0f*th_new), s4n = sinf(4.0f*th_new);
        const float c4o = cosf(4.0f*th_old), s4o = sinf(4.0f*th_old);

        // SANITY CHECK: set perturbed PTM to identity (II block only).
        // This removes all gate action on bond perturb_bond.
        // If sites connected by this bond show zero propagation, the hook works.
        // Replace with physically correct PTM once hook is verified.
        const float* base_ptm = s_ptm_nl(perturb_bond);
        for (int k=0; k<256; k++) g_perturbed_ptm[k] = 0.0f;
        for (int i=0; i<16; i++) g_perturbed_ptm[i*16+i] = 1.0f; // identity
        g_perturb_bond    = perturb_bond;
        g_original_ptm_nl = s_ptm_nl;
        s_ptm_nl          = perturbed_ptm_fn;

        // Verification: print first few changed elements
        int n_changed = 0;
        for (int k=0; k<256; k++) if (fabsf(g_perturbed_ptm[k]-base_ptm[k])>1e-6f) n_changed++;
        printf("# perturb-bond=%d  J_new=%.4f  th_new=%.4f  PTM_elements_changed=%d\n",
               perturb_bond, perturb_J, th_new, n_changed);
    }

    PtmFn ptms[4] = {s_ptm_nl, s_ptm_n1, s_ptm_n2, s_ptm_n3};
    const char* labels[4] = {"NL","N1","N2","N3"};

    // Pre-allocate Trotter message buffers — eliminates dynamic hipMalloc/hipFree
    // every step that spikes VRAM exactly when chi first saturates at chi_max.
    for (int g=0; g<4; g++)
        if (active[g]) wss[g]->alloc_trotter_buffers(grids[g]->n_bonds);

    // Pre-allocate Measures objects — avoid hipMalloc/hipFree every step.
    Measures* meas[4] = {nullptr,nullptr,nullptr,nullptr};
    for (int g=0; g<4; g++)
        if (active[g]) meas[g] = new Measures(handle, *wss[g]);

    for (int step=1; step<=n_steps; step++) {
        double t0 = now_ms();

        // Per active state: Trotter → autocorrelation → PAULI
        float C[4] = {0.f,0.f,0.f,0.f};
        int   bp_sweeps[4] = {0,0,0,0};
        float bp_delta[4]  = {0.f,0.f,0.f,0.f};
        LayerMetrics lm_arr[4];
        std::vector<float> nid(grid.n_sites);
        for (int g=0; g<4; g++) {
            if (!active[g]) continue;
            lm_arr[g] = LayerMetrics{};
            do_trotter(states[g]->psi, states[g]->messages,
                       static_cast<Grid&>(*grids[g]),
                       ptms[g], nullptr, s_site_ptm,
                       *wss[g], gate_ops, chi_max, cutoff, &lm_arr[g],
                       &states[g]->messages_v);
            C[g] = meas[g]->autocorrelation_with_bp(
                static_cast<Grid&>(*grids[g]), states[g]->psi,
                states[g]->messages, center, BP_SWEEPS,
                nullptr, nullptr, &bp_sweeps[g], &bp_delta[g],
                nullptr, &states[g]->messages_v);
        }

        int chi_actual=1;
        for (auto& b:grids[ref_g]->bonds) chi_actual=std::max(chi_actual,b.chi);

        double t=(now_ms()-t0)/1000.0;
        printf("STEP %d  C_nl %+.6f  C_n1 %+.6f  C_n2 %+.6f  C_n3 %+.6f"
               "  chi %d  bp %d/%.2e  gate_err %.2e  t %.2f\n",
               step, C[0], C[1], C[2], C[3], chi_actual,
               bp_sweeps[ref_g], (double)bp_delta[ref_g],
               (double)lm_arr[ref_g].gate_err, t);

        // Pauli weights — active states only
        for (int g=0; g<4; g++) {
            if (!active[g]) continue;
            meas[g]->compute_pauli_profile(static_cast<Grid&>(*grids[g]),
                                            states[g]->psi, nid.data());
            printf("PAULI_%s %d", labels[g], step);
            for (int v=0; v<grid.n_sites; v++) printf(" %.7f", nid[v]);
            printf("\n");
        }
        // ── Message diagnostics ──────────────────────────────────────────
        if (step == diagnose_step && active[ref_g]) {
            printf("# DIAGNOSE step=%d state=NL\n", step);
            printf("# Bond(a,b)  chi  diag_norm  offdiag_norm  diag_frac  min_diag  max_diag\n");
            for (auto& [sa, sb] : diag_bonds) {
                // Find bond id for this site pair
                int bond_id = -1;
                for (auto& b : grids[ref_g]->bonds)
                    if ((b.site_a==sa&&b.site_b==sb)||(b.site_a==sb&&b.site_b==sa))
                        { bond_id = b.id; break; }
                if (bond_id < 0) continue;
                auto& bond = grids[ref_g]->bonds[bond_id];
                int chi = bond.chi;
                if (chi == 0) continue;
                size_t sz = (size_t)chi*chi;
                std::vector<float> msg(sz);
                // D2H message from_a (one direction)
                int idx = bond.msg_from_a();
                HIP_CHECK(hipMemcpy(msg.data(), states[ref_g]->messages[idx],
                    sz*sizeof(float), hipMemcpyDeviceToHost));
                // Compute diagonal vs off-diagonal norms
                double diag_sq = 0, offdiag_sq = 0;
                float mn = 1e30f, mx = 0.f;
                for (int r=0; r<chi; r++)
                    for (int c=0; c<chi; c++) {
                        float v = msg[r + chi*c];
                        if (r==c) { diag_sq += v*v; mn=std::min(mn,v); mx=std::max(mx,v); }
                        else        offdiag_sq += v*v;
                    }
                double total = diag_sq + offdiag_sq;
                double frac  = (total>0) ? diag_sq/total : 1.0;
                printf("# (%2d,%2d)  chi=%3d  diag=%.3e  offdiag=%.3e  frac=%.6f"
                       "  min=%.3e  max=%.3e\n",
                       sa, sb, chi,
                       std::sqrt(diag_sq), std::sqrt(offdiag_sq), frac, mn, mx);
            }
            fflush(stdout);
        }

        fflush(stdout);
    }

    for (int g=0; g<4; g++) {
        delete meas[g]; delete states[g]; delete grids[g]; delete ws_storage[g];
    }
    rocblas_destroy_handle(handle);
    return 0;
}
