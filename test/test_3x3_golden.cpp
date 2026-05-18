/*
 * tests/test_3x3_golden.cpp
 *
 * Golden-reference diagnostic for 3×3 Ising Heisenberg simulation.
 * Dumps checkpoint values matching golden_reference_3x3.jl output.
 *
 * All output lines use KEY=... prefix for grep-based comparison.
 *
 * Build:
 *   hipcc -O0 -g -std=c++17 -DPHYS_DIM=4 -I../include \
 *     -o tests/test_3x3_golden tests/test_3x3_golden.cpp \
 *     -lrocblas -lrocsolver -lm
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "physics.hpp"
#include "measures.hpp"
#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>
#include <string>

static_assert(D == 4, "test_3x3_golden requires PHYS_DIM=4");

static rocblas_handle g_handle = nullptr;

// ── helpers ──────────────────────────────────────────────────────────────────

static std::vector<float> dl(const float* d, int n) {
    std::vector<float> h(n);
    HIP_CHECK(hipMemcpy(h.data(), d, n*sizeof(float), hipMemcpyDeviceToHost));
    return h;
}

static void print_tensor(const char* tag, const float* d, int n) {
    auto h = dl(d, n);
    printf("%s", tag);
    for (auto x : h) printf(" %.8g", x);
    printf("\n");
}

static void print_messages(const char* label,
                            const Grid& grid,
                            const std::vector<float*>& messages)
{
    for (auto& bond : grid.bonds) {
        int chi = bond.chi;
        auto ha = dl(messages[bond.msg_from_a()], chi*chi);
        auto hb = dl(messages[bond.msg_from_b()], chi*chi);
        // print diagonal only (non-diagonal should be ~0 for Vidal gauge)
        printf("MSG label=%s bond=%d->%d chi=%d diag_a=", label, bond.site_a, bond.site_b, chi);
        for (int i = 0; i < chi; i++) printf(" %.8g", ha[i + i*chi]);
        printf("  diag_b=");
        for (int i = 0; i < chi; i++) printf(" %.8g", hb[i + i*chi]);
        printf("\n");
    }
}

static void print_all_tensors(const char* label,
                               const Grid& grid,
                               const std::vector<float*>& psi)
{
    for (auto& site : grid.sites) {
        int sz = site.tensor_size(grid.bonds);
        printf("PSI v=%d label=%s", site.id, label);
        auto h = dl(psi[site.id], sz);
        for (auto x : h) printf(" %.8g", x);
        printf("\n");
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    printf("=== test_3x3_golden (D=%d) BUILD %s %s ===\n", D, __DATE__, __TIME__);
    printf("=== Bethe edge corr, BP observable-specific (d=0 or d=3) ===\n");
    rocblas_create_handle(&g_handle);

    // ── parameters ───────────────────────────────────────────────────────────
    int nx = 3, ny = 3, chi_max = 4;
    float h_field = -1.0f, J_coup = -1.0f, dt = 0.1f;
    float cutoff = 1e-6f;

    printf("PARAMS nx=%d ny=%d h=%.4f J=%.4f dt=%.4f maxdim=%d cutoff=%.2e\n",
           nx, ny, h_field, J_coup, dt, chi_max, cutoff);

    // ── grid setup ────────────────────────────────────────────────────────────
    Grid grid(nx, ny, chi_max);
    Workspace ws(chi_max);
    IsingState state(grid);

    // Center of 3×3: Julia uses first(center(g)) which is vertex (2,2) in 1-based
    // = site (1,1) in 0-based (middle of 3×3)
    int center = grid.site_id(1, 1);
    printf("CENTER_SITE id=%d (row=1,col=1)\n", center);

    printf("ALL_SITES");
    for (auto& s : grid.sites) printf(" %d", s.id);
    printf("\n");

    printf("=== EDGE COLORS ===\n");
    for (int c = 0; c < 4; c++) {
        printf("EDGE_COLOR c=%d", c);
        for (auto& gate : grid.colors[c])
            printf(" %d->%d", gate.site_a, gate.site_b);
        printf("\n");
    }

    // ── initialize state ──────────────────────────────────────────────────────
    state.init_heisenberg(grid, center);

    printf("=== INITIAL TENSORS (t=0) ===\n");
    print_all_tensors("t0", grid, state.psi);

    printf("=== INITIAL MESSAGES ===\n");
    print_messages("t0", grid, state.messages);

    // ── gate operators ────────────────────────────────────────────────────────
    auto Gz  = IsingGates::Rz(h_field * dt);
    auto Gxx = IsingGates::Rxx(2.f * J_coup * dt);

    printf("GATE_RZ theta=%.6f", h_field * dt);
    for (auto x : Gz) printf(" %.8g", x);
    printf("\n");

    printf("GATE_RXX theta=%.6f", 2.f * J_coup * dt);
    for (auto x : Gxx) printf(" %.8g", x);
    printf("\n");

    // ── Phase 1: Rz layer 1 ───────────────────────────────────────────────────
    printf("--- Phase: Rz layer 1 ---\n");
    GateOps ops(g_handle);
    for (auto& site : grid.sites)
        ops.apply_unary_gate(state.psi[site.id], site, grid, Gz.data(), ws.cv1);
    print_all_tensors("after_rz1", grid, state.psi);
    print_messages("after_rz1", grid, state.messages);

    // ── Phase 2: Rxx by color class ──────────────────────────────────────────
    // Frozen snapshots: re-taken at start of each color class
    int nm = 2 * grid.n_bonds;
    std::vector<int>     frozen_chi(grid.n_bonds);
    std::vector<float*>  frozen_msg(nm, nullptr), new_msg(nm, nullptr);
    size_t alloc = (size_t)chi_max * chi_max * sizeof(float);
    for (auto& bond : grid.bonds) {
        for (int d = 0; d < 2; d++) {
            int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
            HIP_CHECK(hipMalloc(&frozen_msg[idx], alloc));
            HIP_CHECK(hipMalloc(&new_msg[idx],    alloc));
        }
    }

    for (int c = 0; c < 4; c++) {
        // Re-snapshot
        for (auto& b : grid.bonds) frozen_chi[b.id] = b.chi;
        for (auto& bond : grid.bonds) {
            int chi = bond.chi;
            size_t sz = (size_t)chi * chi * sizeof(float);
            for (int d = 0; d < 2; d++) {
                int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                HIP_CHECK(hipMemcpy(frozen_msg[idx], state.messages[idx],
                    sz, hipMemcpyDeviceToDevice));
                HIP_CHECK(hipMemcpy(new_msg[idx], state.messages[idx],
                    sz, hipMemcpyDeviceToDevice));
            }
        }

        printf("--- Phase: Rxx color=%d ---\n", c);
        printf("SNAPSHOT color=%d", c);
        for (auto& b : grid.bonds) printf(" bond%d_chi=%d", b.id, b.chi);
        printf("\n");

        for (auto& gate : grid.colors[c]) {
            int sa = gate.site_a, sb = gate.site_b;
            // Capture pre-gate sizes (before apply_binary_gate changes bond.chi)
            int sz_a = grid.sites[sa].tensor_size(grid.bonds);
            int sz_b = grid.sites[sb].tensor_size(grid.bonds);
            int old_chi = frozen_chi[gate.bond_id];

            printf("--- PRE_GATE color=%d site_a=%d site_b=%d ---\n", c, sa, sb);
            print_tensor(("PRE_GATE_PSI site=" + std::to_string(sa) + " ").c_str(),
                          state.psi[sa], sz_a);
            print_tensor(("PRE_GATE_PSI site=" + std::to_string(sb) + " ").c_str(),
                          state.psi[sb], sz_b);

            // Dump env messages for these two sites
            for (auto& bond : grid.bonds) {
                if ((bond.site_a == sa && bond.site_b == sb) ||
                    (bond.site_a == sb && bond.site_b == sa)) {
                    auto fa = dl(frozen_msg[bond.msg_from_a()],
                                 frozen_chi[bond.id] * frozen_chi[bond.id]);
                    auto fb = dl(frozen_msg[bond.msg_from_b()],
                                 frozen_chi[bond.id] * frozen_chi[bond.id]);
                    printf("ENV_MSG_DIRECT bond=%d->%d fchi=%d",
                           bond.site_a, bond.site_b, frozen_chi[bond.id]);
                    printf(" a=");
                    for (auto x : fa) printf("%.8g ", x);
                    printf(" b=");
                    for (auto x : fb) printf("%.8g ", x);
                    printf("\n");
                }
            }

            ops.apply_binary_gate(
                state.psi[sa], state.psi[sb],
                gate, grid,
                new_msg, frozen_msg, frozen_chi,
                Gxx.data(), ws, chi_max, cutoff);

            // Commit this gate's messages
            const Bond& b = grid.bonds[gate.bond_id];
            int new_chi = b.chi;
            size_t msz = (size_t)new_chi * new_chi * sizeof(float);
            for (int d = 0; d < 2; d++) {
                int idx = (d==0) ? b.msg_from_a() : b.msg_from_b();
                HIP_CHECK(hipMemcpy(state.messages[idx], new_msg[idx],
                    msz, hipMemcpyDeviceToDevice));
            }

            printf("--- POST_GATE color=%d site_a=%d site_b=%d new_chi=%d ---\n",
                   c, sa, sb, new_chi);
            sz_a = grid.sites[sa].tensor_size(grid.bonds);
            sz_b = grid.sites[sb].tensor_size(grid.bonds);
            print_tensor(("POST_GATE_PSI site=" + std::to_string(sa) + " ").c_str(),
                          state.psi[sa], sz_a);
            print_tensor(("POST_GATE_PSI site=" + std::to_string(sb) + " ").c_str(),
                          state.psi[sb], sz_b);

            // Print the new message (singular values)
            auto ma = dl(new_msg[b.msg_from_a()], new_chi * new_chi);
            auto mb = dl(new_msg[b.msg_from_b()], new_chi * new_chi);
            printf("POST_GATE_MSG bond=%d->%d new_chi=%d diag_a=",
                   sa, sb, new_chi);
            for (int i = 0; i < new_chi; i++) printf("%.8g ", ma[i + i*new_chi]);
            printf(" diag_b=");
            for (int i = 0; i < new_chi; i++) printf("%.8g ", mb[i + i*new_chi]);
            printf("\n");
        }

        print_all_tensors(("after_rxx_color" + std::to_string(c)).c_str(), grid, state.psi);
        print_messages(("after_rxx_color" + std::to_string(c)).c_str(), grid, state.messages);
    }

    for (auto p : frozen_msg) if (p) HIP_CHECK(hipFree(p));
    for (auto p : new_msg)    if (p) HIP_CHECK(hipFree(p));

    // ── Phase 3: Rz layer 2 ───────────────────────────────────────────────────
    printf("--- Phase: Rz layer 2 ---\n");
    for (auto& site : grid.sites)
        ops.apply_unary_gate(state.psi[site.id], site, grid, Gz.data(), ws.cv1);
    print_all_tensors("after_rz2", grid, state.psi);
    print_messages("after_rz2", grid, state.messages);

    Measures meas(g_handle, ws);

    // ── measure without any rescale (baseline, should match pre-rescale value)
    printf("=== MEASUREMENT (no rescale) ===\n");
    Scalar C_raw = meas.autocorrelation(grid, state.psi, state.messages, center);
    printf("AUTOCORR_NORESCALE %.8g\n", C_raw);

    // ── BP (match Julia's apply_gates bp_update_kwargs=100 iter) ─────────────
    printf("=== BP (100 sweeps) ===\n");
    meas.bp_converge(grid, state.psi, state.messages, 100, center);
    Scalar C_bp = meas.autocorrelation(grid, state.psi, state.messages, center);
    printf("AUTOCORR_AFTER_BP %.8g\n", C_bp);

    // ── rescale (match Julia's rescale(ψ_bpc)) ────────────────────────────────
    printf("=== RESCALE ===\n");
    meas.rescale(grid, state.psi, state.messages, ws);
    print_all_tensors("after_rescale", grid, state.psi);

    // ── BP after rescale (match Julia's update() after rescale) ──────────────
    printf("=== BP after rescale (100 sweeps) ===\n");
    meas.bp_converge(grid, state.psi, state.messages, 100, center);

    // ── final measurement ─────────────────────────────────────────────────────
    printf("=== MEASUREMENT (after rescale+BP) ===\n");
    Scalar C = meas.autocorrelation(grid, state.psi, state.messages, center);
    printf("AUTOCORR_LAYER1 %.8g\n", C);
    printf("Julia_target    0.92261845\n");

    rocblas_destroy_handle(g_handle);
    printf("=== DONE ===\n");
    return 0;
}
