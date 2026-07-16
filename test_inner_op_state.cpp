/*
 * test_inner_op_state.cpp — Unit tests for inner_op_state()
 *
 * Tests the Vidal-gauge inner product C(t) = Tr(O(t) · ρ).
 *
 * Test 1: t=0, device ρ    → C(0) = +1 (center site spin=+1)
 * Test 2: t=0, all +1 ρ   → C(0) = +1 (all |0>)
 * Test 3: t=0, all -1 ρ   → C(0) = +1 (Z·|1><1| = -(-Z) = still picks Z component)
 * Test 4: t=0, flip site 32 only → compare with test 2
 *
 * At t=0: ψ = Z at center, I elsewhere, chi=1, messages=1
 * In this trivial case inner_op_state should give +1 regardless of ρ
 * because I⊗...⊗Z⊗...⊗I contracted with any product state = <Z>_center
 * and center site spin=+1.
 *
 * At t=1: ψ has spread to neighbors — ρ correction matters
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
#include <cmath>
#include <vector>

static int n_pass = 0;
static int n_fail = 0;

static void check(const char* name, float got, float expected, float tol=1e-4f) {
    bool ok = std::abs(got - expected) < tol;
    printf("  %-40s %s  got=%.6f  expected=%.6f\n",
           name, ok ? "PASS" : "FAIL", got, expected);
    if (ok) n_pass++; else n_fail++;
}

int main() {
    printf("═══════════════════════════════════════════════\n");
    printf("  test_inner_op_state\n");
    printf("  BUILD: %s %s\n", __DATE__, __TIME__);
    printf("═══════════════════════════════════════════════\n\n");

    rocblas_handle handle;
    rocblas_create_handle(&handle);

    int chi_max = 100;
    FlatHeavyHexGrid grid(chi_max);
    Workspace        ws(chi_max, 3);
    HeavyHexState    state(grid);
    Measures         meas(handle, ws);
    HeavyHexSimulator sim(handle, 1.0f,
                          HEAVY_HEX_3x3_IDEAL_KICK_DT);

    int center = 2;  // Julia site (3,) → 0-based site 2

    // ── Device initial state from HPP ─────────────────────────────
    const int* device_spins = HEAVY_HEX_3x3_IDEAL_KICK_INITIAL_STATE;

    // ── All +1 state ──────────────────────────────────────────────
    std::vector<int> all_plus(grid.n_sites, +1);

    // ── All -1 state ──────────────────────────────────────────────
    std::vector<int> all_minus(grid.n_sites, -1);

    // ── Flip only site 32 (neighbor of center=2) ──────────────────
    std::vector<int> flip32(device_spins, device_spins + grid.n_sites);
    flip32[32] = -flip32[32];

    // ─────────────────────────────────────────────────────────────
    // Test group 1: t=0 (no evolution)
    // ψ = Z at center, I elsewhere, chi=1, messages=identity
    // ─────────────────────────────────────────────────────────────
    printf("── Group 1: t=0 (trivial state) ────────────────\n");
    state.init_heisenberg(grid, center);

    {
        Scalar C = meas.inner_op_state(
            static_cast<Grid&>(grid), state.psi, state.messages,
            device_spins, center);
        check("t=0, device rho → C=+1", C, +1.0f);
    }
    {
        Scalar C = meas.inner_op_state(
            static_cast<Grid&>(grid), state.psi, state.messages,
            all_plus.data(), center);
        check("t=0, all+1 rho → C=+1", C, +1.0f);
    }
    {
        Scalar C = meas.inner_op_state(
            static_cast<Grid&>(grid), state.psi, state.messages,
            all_minus.data(), center);
        // Z at center, center spin=+1 → still +1 (non-center I doesn't matter at t=0)
        check("t=0, all-1 rho → C=+1", C, +1.0f);
    }

    // ─────────────────────────────────────────────────────────────
    // Test group 2: t=1 (one Trotter step)
    // ψ has spread to neighbors of center — ρ correction matters
    // ─────────────────────────────────────────────────────────────
    printf("\n── Group 2: t=1 (one Trotter step) ─────────────\n");
    state.init_heisenberg(grid, center);

    // Run BP to converge messages before Trotter step
    sim.trotter_layer(state, grid, ws, chi_max, 1e-4f);

    // Run BP to converge messages for measurement
    {
        // Print physical index (d=0,1,2,3) of center and neighbor sites
        // Shape: center=site2 has degree 3 → [chi,chi,chi,D]; neighbors degree 2 → [chi,chi,D]
        // At chi=4: center tensor has 4*4*4*4=256 elements, neighbor has 4*4*4=64
        auto print_site = [&](int sid, const char* label) {
            // Get degree
            int deg = 0;
            for (int d=0; d<3; d++) if (grid.sites[sid].bond_id[d]>=0) deg++;
            int chi = grid.bonds[grid.sites[sid].bond_id[0]].chi;
            int total = 1;
            for (int d=0; d<deg; d++) total *= chi;
            total *= D;
            std::vector<Scalar> t(total, 0.f);
            HIP_CHECK(hipMemcpy(t.data(), state.psi[sid], total*sizeof(Scalar), hipMemcpyDeviceToHost));
            // Print the D=4 physical components at bond index [0,0,...,0]
            printf("  %s (site %d, deg=%d, chi=%d): psi[0,0,...,d] = [%.4f, %.4f, %.4f, %.4f]\n",
                   label, sid, deg, chi, t[0], t[1], t[2], t[3]);
        };
        print_site(2,  "center  (site 2) ");
        print_site(32, "neighbor(site 32)");
        print_site(33, "neighbor(site 33)");
        print_site(34, "neighbor(site 34)");

        // Reuse autocorrelation_with_bp to run BP, then call inner_op_state
        // with the same converged messages
        Scalar C_old = meas.autocorrelation_with_bp(
            static_cast<Grid&>(grid), state.psi, state.messages,
            center, 100);
        printf("  autocorrelation (old, all-zero rho) = %.6f\n", C_old);

        Scalar C_all_plus = meas.inner_op_state(
            static_cast<Grid&>(grid), state.psi, state.messages,
            all_plus.data(), center);
        printf("  inner_op_state (all+1 rho)          = %.6f\n", C_all_plus);

        // all+1 ρ = [1,0,0,+1] picks I+Z at each site — different from
        // autocorrelation which uses [1,0,0,0] picking only I component.
        // These are different observables — just print for information.
        printf("  Note: all+1 rho picks I+Z; autocorrelation picks I only\n");

        Scalar C_device = meas.inner_op_state(
            static_cast<Grid&>(grid), state.psi, state.messages,
            device_spins, center);
        printf("  inner_op_state (device rho)         = %.6f\n", C_device);
        printf("  Julia golden reference (cycle 1)    = -0.678563\n");

        // Test: start from evolved messages instead of identity
        Scalar C_device_warm = meas.inner_op_state(
            static_cast<Grid&>(grid), state.psi, state.messages,  // use evolved messages
            device_spins, center, 100);
        printf("  inner_op_state (device rho, warm)   = %.6f\n", C_device_warm);

        // Key tests verified against Julia:
        // inner(ψ, Z⊗I...⊗I) = -0.803318 → matches autocorrelation ✓
        // inner(ψ, all-zero ρ) = -0.961822
        // inner(ψ, device ρ)  = -0.678563 ← Julia golden
        check("t=1, device rho ≈ Julia golden (-0.6786)", C_device,  -0.678563f, 0.05f);
        check("t=1, all+1 rho ≈ Julia all-zero (-0.9618)", C_all_plus, -0.961822f, 0.05f);

        Scalar C_flip32 = meas.inner_op_state(
            static_cast<Grid&>(grid), state.psi, state.messages,
            flip32.data(), center);
        printf("  inner_op_state (flip site 32)       = %.6f\n", C_flip32);

        // Flipping site 32 (neighbor, spin=-1→+1) should change C(t)
        bool differs = std::abs(C_device - C_flip32) > 1e-6f;
        printf("  %-40s %s  |delta|=%.6f\n",
               "flip32 changes C(t)",
               differs ? "PASS" : "FAIL",
               std::abs(C_device - C_flip32));
        if (differs) n_pass++; else n_fail++;
    }

    printf("\n═══════════════════════════════════════════════\n");
    printf("  Results: %d passed  %d failed\n", n_pass, n_fail);
    printf("═══════════════════════════════════════════════\n");

    rocblas_destroy_handle(handle);
    return n_fail > 0 ? 1 : 0;
}
