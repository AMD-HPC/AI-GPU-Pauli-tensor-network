/*
 * test_inner_op_state.cpp — Unit tests for inner_op_state_cpu()
 *
 * Validates C(t) = inner(ψ(t), ρ; alg="bp") against Julia golden reference.
 *
 * Julia verified values at cycle 1:
 *   inner(ψ, Z⊗I...⊗I)   = -0.803318  ← autocorrelation_with_bp (already works)
 *   inner(ψ, all-zero ρ)  = -0.961822
 *   inner(ψ, device ρ)    = -0.678563  ← GOLDEN TARGET
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "measures.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"
#include "inner_op_state.hpp"
#include "bilinear_form_bp.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

static int n_pass = 0, n_fail = 0;

static void check(const char* name, float got, float expected, float tol=0.05f) {
    bool ok = std::abs(got - expected) < tol;
    printf("  %-50s %s  got=%.6f  expected=%.6f\n",
           name, ok?"PASS":"FAIL", got, expected);
    if(ok) n_pass++; else n_fail++;
}

int main() {
    printf("═══════════════════════════════════════════════\n");
    printf("  test_inner_op_state\n");
    printf("  BUILD: %s %s\n", __DATE__, __TIME__);
    printf("═══════════════════════════════════════════════\n\n");

    rocblas_handle handle; rocblas_create_handle(&handle);

    int chi_max=100, center=2;
    FlatHeavyHexGrid  grid(chi_max);
    Workspace         ws(chi_max, 3);
    HeavyHexState     state(grid);
    Measures          meas(handle, ws);
    HeavyHexSimulator sim(handle, 1.0f, HEAVY_HEX_3x3_IDEAL_KICK_DT);

    const int* device_spins = HEAVY_HEX_3x3_IDEAL_KICK_INITIAL_STATE;
    std::vector<int> all_plus(grid.n_sites, +1);

    // ── t=0 ──────────────────────────────────────────────────────────
    printf("── t=0 (no evolution) ───────────────────────────\n");
    state.init_heisenberg(grid, center);
    check("t=0, device rho → C=+1",
        inner_op_state_cpu(static_cast<Grid&>(grid), state.psi, state.messages, device_spins, center, meas),
        +1.0f, 1e-4f);
    check("t=0, all+1 rho → C=+1",
        inner_op_state_cpu(static_cast<Grid&>(grid), state.psi, state.messages, all_plus.data(), center, meas),
        +1.0f, 1e-4f);

    // ── t=1 ──────────────────────────────────────────────────────────
    printf("\n── t=1 (one Trotter step) ───────────────────────\n");
    state.init_heisenberg(grid, center);
    sim.trotter_layer(state, grid, ws, chi_max, 1e-4f);
    Scalar C_auto = meas.autocorrelation_with_bp(
        static_cast<Grid&>(grid), state.psi, state.messages, center, 100);
    printf("  autocorrelation (Z⊗I...⊗I) = %.6f\n", C_auto);
    check("autocorrelation ≈ Julia -0.803318", C_auto, -0.803318f, 0.01f);

    // ── Hardcode Julia's converged messages to verify formula ────────
    // From test_inner_minimal.jl output (all-zero rho, after 25 sweeps):
    // Julia bond (3,)→(33,) = CppSim bond id=5 (site_a=2,site_b=33)
    // Julia vertex numbering is 1-based, CppSim is 0-based
    // Julia msg (3,)→(33,): [1.073, 0.030, 0.165, -0.268]
    // Julia msg (3,)→(34,): [0.995, -0.156, 0.108, 0.053]  → bond (2,33)
    // Julia msg (3,)→(35,): [1.088, -0.143, -0.054, 0.108]  → bond (2,34)
    // Julia msg (2,)→(33,): [0.724]  → chi=1, bond to site 32 or neighbor
    // Julia msg (4,)→(34,): [2.544, -1.306, 0.004, -0.242]  → bond (3,33)
    // All other messages: [1.0] (chi=1 bonds)
    //
    // For CppSim: bond ids for center=2:
    //   bond[0]: id=4 a=2 b=32 chi=4  (Julia: (3,)→(31,) but 0-based: 2→31)
    //   bond[1]: id=5 a=2 b=33 chi=4
    //   bond[2]: id=6 a=2 b=34 chi=4
    // Julia msg directions are from site (v) to neighbor, so:
    //   msg_fwd[4] = msg 2→32 = Julia (3,)→(31,) ... not printed separately
    //   Actually from Julia: msg (3,)→(33,)=[1.073,...] maps to bond5, site_a=2→b=33
    //   msg (3,)→(34,) maps to bond6, msg (3,)→(35,) maps to...
    //   Note: Julia sites 31,32,33,34,35 are 0-based: 30,31,32,33,34
    // Julia (3,)=(CppSim 2), (33,)=(CppSim 32), (34,)=(CppSim 33), (35,)=(CppSim 34)
    printf("\n  -- Hardcoded Julia messages test --\n");
    {
        // Copy state messages but override chi=4 bonds with Julia values
        int nm=2*grid.n_bonds;
        std::vector<float*> julia_msgs(nm, nullptr);
        for (auto& bond:grid.bonds) {
            int chi=bond.chi, chi2=chi*chi;
            for(int d=0;d<2;d++){
                int idx=(d==0)?bond.msg_from_a():bond.msg_from_b();
                HIP_CHECK(hipMalloc(&julia_msgs[idx], chi2*sizeof(float)));
                HIP_CHECK(hipMemcpy(julia_msgs[idx], state.messages[idx],
                    chi2*sizeof(float), hipMemcpyDeviceToDevice));
            }
        }
        // Override with Julia's chi-vector messages as diagonal matrices
        // Bond 5: site_a=2, site_b=33 → Julia (3,)→(34,) and (34,)→(3,)
        // Julia msg (3,)→(33,)=[1.073,0.030,0.165,-0.268] → bond4 (2→32) or bond5?
        // Bond 4: a=2,b=32; Bond 5: a=2,b=33; Bond 6: a=2,b=34
        // Julia 1-based: (33,)=site32=0based, (34,)=site33, (35,)=site34
        // So: Julia msg (3,)→(33,) = CppSim bond4 (2→32), msg_fwd direction
        // Julia msg (3,)→(34,) = CppSim bond5 (2→33)
        // Julia msg (3,)→(35,) = CppSim bond6 (2→34)
        auto set_chi_msg=[&](int bid, int site_from, std::vector<float> vec){
            int chi=vec.size(), chi2=chi*chi;
            std::vector<float> mat(chi2,0.f);
            for(int i=0;i<chi;i++) mat[i*chi+i]=vec[i];
            int idx=(site_from==grid.bonds[bid].site_a)?
                     grid.bonds[bid].msg_from_a():grid.bonds[bid].msg_from_b();
            HIP_CHECK(hipMemcpy(julia_msgs[idx], mat.data(),
                chi2*sizeof(float), hipMemcpyHostToDevice));
        };
        // Julia (3,)→(33,) = bond4 site2→site32
        set_chi_msg(4, 2, {1.0727f, 0.0305f, 0.1645f, -0.2677f});
        // Julia (3,)→(34,) = bond5 site2→site33
        set_chi_msg(5, 2, {0.9954f, -0.1564f, 0.1083f, 0.0527f});
        // Julia (3,)→(35,) = bond6 site2→site34
        set_chi_msg(6, 2, {1.0877f, -0.1425f, -0.0536f, 0.1084f});
        // Julia (4,)→(34,) = bond from site3→site33 ... need bond id
        // site3 connects to site33 via bond id=9 (from EDGES list: (3,35)→(2,34) 0-based)
        // Actually let's just use what we have and see if center a_v changes

        float C_julia_msgs = inner_op_state_cpu(
            static_cast<Grid&>(grid), state.psi, julia_msgs, all_plus.data(), center, meas);
        printf("  inner_op_state with Julia msgs (all+1) = %.6f  (Julia: -0.961822)\n", C_julia_msgs);

        for(auto p:julia_msgs) if(p) HIP_CHECK(hipFree(p));
    }

    // Print ALL 68 site phi values for Python validation
    printf("PHI_DUMP_BEGIN\n");
    {
        const float rho_p[4]={1.f,0.f,0.f,+1.f};
        const float rho_m[4]={1.f,0.f,0.f,-1.f};
        const float rho_Z[4]={0.f,0.f,0.f, 1.f};
        const int* dev = HEAVY_HEX_3x3_IDEAL_KICK_INITIAL_STATE;
        for (auto& site : grid.sites) {
            // Shape: (chi_bond0,...,chi_bondk, D)
            std::vector<int> sh;
            for (int d=0; d<MAX_DEGREE; d++)
                if (site.bond_id[d]>=0) sh.push_back(grid.bonds[site.bond_id[d]].chi);
            sh.push_back(D);
            int tot=1; for(auto s:sh) tot*=s;
            std::vector<float> t(tot);
            hipMemcpy(t.data(), state.psi[site.id], tot*sizeof(float), hipMemcpyDeviceToHost);
            const float* rho = (site.id==center) ? rho_Z : (dev[site.id]>=0 ? rho_p : rho_m);
            int bvol=tot/D;
            printf("site %d spin %d bvol %d chi", site.id, dev[site.id], bvol);
            for(int d=0;d<MAX_DEGREE;d++) if(site.bond_id[d]>=0) printf(" %d",grid.bonds[site.bond_id[d]].chi);
            printf(" phi");
            for(int b=0;b<bvol;b++){float v=0;for(int d=0;d<D;d++) v+=t[b*D+d]*rho[d]; printf(" %.8f",v);}
            printf("\n");
        }
    }
    printf("PHI_DUMP_END\n");

    // Dump autocorrelation messages (chi×chi) for gauge transform validation
    printf("MSG_DUMP_BEGIN\n");
    for (auto& bond : grid.bonds) {
        int chi = bond.chi, chi2 = chi*chi;
        std::vector<float> ma(chi2), mb(chi2);
        hipMemcpy(ma.data(), state.messages[bond.msg_from_a()], chi2*sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(mb.data(), state.messages[bond.msg_from_b()], chi2*sizeof(float), hipMemcpyDeviceToHost);
        printf("bond %d a %d b %d chi %d a2b", bond.id, bond.site_a, bond.site_b, chi);
        for(int k=0;k<chi2;k++) printf(" %.8f", ma[k]);
        printf(" b2a");
        for(int k=0;k<chi2;k++) printf(" %.8f", mb[k]);
        printf("\n");
    }
    printf("MSG_DUMP_END\n");

    // Dump FULL psi tensors (with physical index) for gauge transform
    printf("PSI_DUMP_BEGIN\n");
    for (auto& site : grid.sites) {
        std::vector<int> sh;
        for (int d=0; d<MAX_DEGREE; d++)
            if (site.bond_id[d]>=0) sh.push_back(grid.bonds[site.bond_id[d]].chi);
        sh.push_back(D);
        int tot=1; for(auto s:sh) tot*=s;
        std::vector<float> t(tot);
        hipMemcpy(t.data(), state.psi[site.id], tot*sizeof(float), hipMemcpyDeviceToHost);
        printf("site %d spin %d chi", site.id, HEAVY_HEX_3x3_IDEAL_KICK_INITIAL_STATE[site.id]);
        for(int d=0;d<MAX_DEGREE;d++) if(site.bond_id[d]>=0) printf(" %d",grid.bonds[site.bond_id[d]].chi);
        printf(" bond");
        for(int d=0;d<MAX_DEGREE;d++) if(site.bond_id[d]>=0) printf(" %d",site.bond_id[d]);
        printf(" psi");
        for(int k=0;k<tot;k++) printf(" %.8f", t[k]);
        printf("\n");
    }
    printf("PSI_DUMP_END\n");

    // Print phi_32, phi_33, phi_34 for Python validation
    {
        const float rho_p[4]={1.f,0.f,0.f,+1.f};
        const float rho_m[4]={1.f,0.f,0.f,-1.f};
        const int* dev = HEAVY_HEX_3x3_IDEAL_KICK_INITIAL_STATE;
        for (int sid : {32, 33, 34}) {
            const Site& s = grid.sites[sid];
            // site 32,33,34 have shape [chi_to_center, chi_other, D] or [chi_to_center, D]
            std::vector<int> sh;
            for (int d=0; d<MAX_DEGREE; d++)
                if (s.bond_id[d]>=0) sh.push_back(grid.bonds[s.bond_id[d]].chi);
            sh.push_back(D);
            int tot=1; for(auto x:sh) tot*=x;
            std::vector<float> t(tot);
            hipMemcpy(t.data(), state.psi[sid], tot*sizeof(float), hipMemcpyDeviceToHost);
            const float* rho = (dev[sid]>=0) ? rho_p : rho_m;
            int bvol=1; for(int i=0;i<(int)sh.size()-1;i++) bvol*=sh[i];
            printf("phi_%d (rho=[1,0,0,%+d]): ", sid, dev[sid]);
            for(int b=0;b<bvol;b++){
                float v=0; for(int d=0;d<D;d++) v+=t[b*D+d]*rho[d];
                printf("%.6f ", v);
            }
            printf("\n");
        }
    }

    float C_allplus = inner_op_state_cpu(
        static_cast<Grid&>(grid), state.psi, state.messages, all_plus.data(), center, meas);
    float C_device = inner_op_state_cpu(
        static_cast<Grid&>(grid), state.psi, state.messages, device_spins, center, meas);

    printf("  inner_op_state (all+1 rho)  = %.6f  (Julia all-zero: -0.961822)\n", C_allplus);
    printf("  inner_op_state (device rho) = %.6f  (Julia golden:   -0.678563)\n", C_device);

    check("t=1, all+1 rho ≈ Julia -0.9618", C_allplus, -0.961822f, 0.05f);
    check("t=1, device rho ≈ Julia -0.6786 ← GOLDEN", C_device, -0.678563f, 0.05f);

    // ── Dump noisy psi for Python verification ────────────────────────
    {
        // Noisy state: NoisyKickSimulator — NOISY PTM (gate+noise composed) + kick
        FlatHeavyHexGrid  grid_n(chi_max);
        Workspace          ws_n(chi_max, 3);
        HeavyHexState      state_n(grid_n);
        NoisyKickSimulator sim_n(handle, 1.0f);
        state_n.init_heisenberg(grid_n, center);
        sim_n.trotter_layer(state_n, grid_n, ws_n, chi_max, 1e-4f);
        meas.autocorrelation_with_bp(
            static_cast<Grid&>(grid_n), state_n.psi, state_n.messages, center, 100);

        printf("PSI_N_DUMP_BEGIN\n");
        const int* dev = HEAVY_HEX_3x3_IDEAL_KICK_INITIAL_STATE;
        for (auto& s : grid_n.sites) {
            std::vector<int> sh;
            for (int d=0;d<MAX_DEGREE;d++)
                if (s.bond_id[d]>=0) sh.push_back(grid_n.bonds[s.bond_id[d]].chi);
            sh.push_back(D);
            int tot=1; for(auto x:sh) tot*=x;
            std::vector<float> t(tot);
            hipMemcpy(t.data(), state_n.psi[s.id], tot*sizeof(float), hipMemcpyDeviceToHost);
            printf("site %d spin %d chi", s.id, dev[s.id]);
            for(int d=0;d<MAX_DEGREE;d++) if(s.bond_id[d]>=0) printf(" %d",grid_n.bonds[s.bond_id[d]].chi);
            printf(" bond");
            for(int d=0;d<MAX_DEGREE;d++) if(s.bond_id[d]>=0) printf(" %d",s.bond_id[d]);
            printf(" psi");
            for(auto x:t) printf(" %.8f",x);
            printf("\n");
        }
        printf("PSI_N_DUMP_END\n");
    }

    // ── Test bilinear_form_bp ──────────────────────────────────────────
    printf("\n── bilinear_form_bp tests ───────────────────────────────────\n");

    // Self-overlap ⟨ψ_nl|ψ_nl⟩
    float C_self = bilinear_form_bp(
        static_cast<Grid&>(grid), state.psi,
        static_cast<Grid&>(grid), state.psi);
    printf("  bilinear_form_bp(ψ_nl,ψ_nl) = %.6f  (Julia: 0.99991)\n", C_self);

    // Noisy state for cross-overlap ⟨ψ_nl|ψ_noisy⟩
    // NoisyKickSimulator: single NOISY PTM per bond + kick wave — same structure as noiseless
    {
        FlatHeavyHexGrid   grid2(chi_max);
        Workspace          ws2(chi_max, 3);
        HeavyHexState      state_n(grid2);
        NoisyKickSimulator sim_n(handle, 1.0f);
        state_n.init_heisenberg(grid2, center);
        sim_n.trotter_layer(state_n, grid2, ws2, chi_max, 1e-4f);
        // Run BP on noisy state
        meas.autocorrelation_with_bp(
            static_cast<Grid&>(grid2), state_n.psi, state_n.messages, center, 100);

        // Self-overlaps (no gauge transform needed — same state, sign cancels)
        float C_self_n = bilinear_form_bp(
            static_cast<Grid&>(grid2), state_n.psi,
            static_cast<Grid&>(grid2), state_n.psi);
        printf("  bilinear_form_bp(ψ_nl,ψ_nl) = %.6f  (Julia: 0.99991)\n", C_self);
        printf("  bilinear_form_bp(ψ_n,ψ_n)   = %.6f  (Julia: 0.96230)\n", C_self_n);

        // Cross-overlap with sqrt-gauge transform using converged BP messages
        float C_cross = bilinear_form_bp(
            static_cast<Grid&>(grid),  state.psi,
            static_cast<Grid&>(grid2), state_n.psi,
            &state.messages,           // msg_a: noiseless BP messages
            &state_n.messages,         // msg_b: noisy BP messages
            25, 1e-5f, true);
        float cosine = C_cross / std::sqrt(C_self * C_self_n);

        printf("  bilinear_form_bp(ψ_nl,ψ_n)  = %.6f  (Julia: 0.98092)\n", C_cross);
        printf("  cosine similarity            = %.6f  (Julia: ~0.9810)\n", cosine);

        check("cosine(ψ_nl,ψ_n) ≈ Julia 0.98092", cosine, 0.98092f, 0.05f);
    }

    printf("\n═══════════════════════════════════════════════\n");
    printf("  Results: %d passed  %d failed\n", n_pass, n_fail);
    printf("═══════════════════════════════════════════════\n");

    rocblas_destroy_handle(handle);
    return n_fail > 0 ? 1 : 0;
}
