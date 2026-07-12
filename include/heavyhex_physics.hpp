#pragma once
/*
 * heavyhex_physics.hpp — Gates and simulator for the heavy-hex workload
 *
 * ═══════════════════════════════════════════════════════════════════
 * HEAVY-HEX HAMILTONIAN
 * ═══════════════════════════════════════════════════════════════════
 *
 * H = Σ_{<ij>} J_ij (XX + YY + ZZ)   (isotropic Heisenberg / XXX model)
 *
 * Two-site gate: U = exp(-iθ(X⊗X + Y⊗Y + Z⊗Z))   θ = J * dt
 *
 * BUILD: make PHYS_DIM=4
 */

#include "types.hpp"
#include "heavyhex.hpp"
#include "gate.hpp"
#include "measures.hpp"
#include "workspace.hpp"
#include <cmath>
#include <vector>
#include <cstdio>
#include <algorithm>

static_assert(D == 4, "Heavy-hex simulation requires PHYS_DIM=4 (Pauli basis)");

/* =========================================================
 * HeavyHexGates — adjoint action superoperators in Pauli basis
 * ========================================================= */
struct HeavyHexGates {

    /* ── XxYyZz(θ) superoperator [16×16 col-major] ──────────────────
     *
     * U = exp(-iθ(X⊗X + Y⊗Y + Z⊗Z))
     *
     * Basis: {II,IX,IY,IZ,XI,XX,XY,XZ,YI,YX,YY,YZ,ZI,ZX,ZY,ZZ}
     * Index: α*4 + β  (α = site1 Pauli: 0=I,1=X,2=Y,3=Z)
     *
     * Let:
     *   a = cos²(2θ)
     *   b = sin²(2θ)
     *   g = sin(2θ)cos(2θ) = sin(4θ)/2
     *
     * Four invariant elements (diagonal = 1):
     *   II(0), XX(5), YY(10), ZZ(15)
     *
     * Three 4×4 mixing blocks (verified against Julia's numerical PTM):
     *
     *   Block A: indices {IX=1, XI=4, YZ=11, ZY=14}
     *   Block B: indices {IY=2, YI=8, XZ=7,  ZX=13}
     *   Block C: indices {IZ=3, ZI=12, XY=6, YX=9}
     *
     * Each block has the same structure for its 4 elements {p,q,r,s}:
     *   col p: [ a,  b, -g,  g]
     *   col q: [ b,  a,  g, -g]
     *   col r: [ g, -g,  a,  b]
     *   col s: [-g,  g,  b,  a]
     * (rows ordered as p,q,r,s within the block)
     *
     * Signs verified at theta=0.1: a=0.9605, b=0.0395, g=0.1947
     */
    static std::vector<Scalar> XxYyZz(Scalar theta) {
        float c2 = std::cos(2.f * theta);
        float s2 = std::sin(2.f * theta);
        float a  = c2 * c2;    // cos²(2θ)
        float b  = s2 * s2;    // sin²(2θ)
        float g  = s2 * c2;    // sin(4θ)/2

        std::vector<Scalar> M(16*16, 0.f);
        // M[row + 16*col]: col=input Pauli index, row=output Pauli index
        auto set = [&](int row, int col, Scalar v) { M[row + 16*col] = v; };

        // Invariant
        set( 0,  0, 1.f);
        set( 5,  5, 1.f);
        set(10, 10, 1.f);
        set(15, 15, 1.f);

        // Block A: {IX=1, XI=4, YZ=11, ZY=14}  p=1,q=4,r=11,s=14
        set( 1,  1,  a);  set( 4,  1,  b);  set(11,  1, -g);  set(14,  1,  g);
        set( 1,  4,  b);  set( 4,  4,  a);  set(11,  4,  g);  set(14,  4, -g);
        set( 1, 11,  g);  set( 4, 11, -g);  set(11, 11,  a);  set(14, 11,  b);
        set( 1, 14, -g);  set( 4, 14,  g);  set(11, 14,  b);  set(14, 14,  a);

        // Block B: {IY=2, YI=8, XZ=7, ZX=13}  p=2,q=8,r=7,s=13
        set( 2,  2,  a);  set( 8,  2,  b);  set( 7,  2,  g);  set(13,  2, -g);
        set( 2,  8,  b);  set( 8,  8,  a);  set( 7,  8, -g);  set(13,  8,  g);
        set( 2,  7, -g);  set( 8,  7,  g);  set( 7,  7,  a);  set(13,  7,  b);
        set( 2, 13,  g);  set( 8, 13, -g);  set( 7, 13,  b);  set(13, 13,  a);

        // Block C: {IZ=3, ZI=12, XY=6, YX=9}  p=3,q=12,r=6,s=9
        set( 3,  3,  a);  set(12,  3,  b);  set( 6,  3, -g);  set( 9,  3,  g);
        set( 3, 12,  b);  set(12, 12,  a);  set( 6, 12,  g);  set( 9, 12, -g);
        set( 3,  6,  g);  set(12,  6, -g);  set( 6,  6,  a);  set( 9,  6,  b);
        set( 3,  9, -g);  set(12,  9,  g);  set( 6,  9,  b);  set( 9,  9,  a);

        return M;
    }
};

/* =========================================================
 * HeavyHexState — GPU state for heavy-hex simulation
 * ========================================================= */
struct HeavyHexState {
    std::vector<Scalar*> psi;
    std::vector<Scalar*> messages;
    int chi_max;
    int n_sites, n_bonds;

    explicit HeavyHexState(const FlatHeavyHexGrid& g)
        : chi_max(g.chi_max), n_sites(g.n_sites), n_bonds(g.n_bonds)
    {
        psi.resize(n_sites, nullptr);
        messages.resize(2*n_bonds, nullptr);

        for (auto& site : g.sites) {
            size_t sz = (size_t)site.tensor_size_max(chi_max)*sizeof(Scalar);
            HIP_CHECK(hipMalloc(&psi[site.id], sz));
            HIP_CHECK(hipMemset(psi[site.id], 0, sz));
        }
        for (auto& bond : g.bonds) {
            size_t sz = (size_t)chi_max*chi_max*sizeof(Scalar);
            HIP_CHECK(hipMalloc(&messages[bond.msg_from_a()], sz));
            HIP_CHECK(hipMemset(messages[bond.msg_from_a()], 0, sz));
            HIP_CHECK(hipMalloc(&messages[bond.msg_from_b()], sz));
            HIP_CHECK(hipMemset(messages[bond.msg_from_b()], 0, sz));
        }
    }

    ~HeavyHexState() {
        for (auto p : psi)      if (p) HIP_CHECK(hipFree(p));
        for (auto p : messages) if (p) HIP_CHECK(hipFree(p));
    }

    HeavyHexState(const HeavyHexState&) = delete;
    HeavyHexState& operator=(const HeavyHexState&) = delete;

    void init_heisenberg(FlatHeavyHexGrid& g, int center_site) {
        std::vector<Scalar> h_I = {1.f, 0.f, 0.f, 0.f};  // I in Pauli basis
        std::vector<Scalar> h_Z = {0.f, 0.f, 0.f, 1.f};  // Z in Pauli basis

        for (auto& site : g.sites) {
            Scalar* data = (site.id == center_site) ? h_Z.data() : h_I.data();
            HIP_CHECK(hipMemcpy(psi[site.id], data,
                D*sizeof(Scalar), hipMemcpyHostToDevice));
        }

        Scalar one = 1.f;
        for (auto& bond : g.bonds) {
            HIP_CHECK(hipMemcpy(messages[bond.msg_from_a()], &one,
                sizeof(Scalar), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(messages[bond.msg_from_b()], &one,
                sizeof(Scalar), hipMemcpyHostToDevice));
            bond.chi = 1;
        }
        printf("[HeavyHexState] D=%d center=%d all chi=1\n", D, center_site);
    }
};

/* =========================================================
 * HeavyHexSimulator — Trotter layer for heavy-hex
 *
 * One Trotter step applies the XxYyZz gate to all bonds,
 * processed in 3 color classes (no overlapping bonds within
 * a color class). Matches Julia's layer structure exactly.
 * ========================================================= */
struct HeavyHexSimulator {
    rocblas_handle handle;
    GateOps gate_ops;
    Scalar J, dt;

    HeavyHexSimulator(rocblas_handle h_, Scalar J_, Scalar dt_)
        : handle(h_), gate_ops(h_), J(J_), dt(dt_) {}

    int trotter_layer(HeavyHexState& state, FlatHeavyHexGrid& grid,
                      Workspace& ws, int chi_max, Scalar cutoff=1e-10f,
                      LayerMetrics* lm=nullptr)
    {
        auto G = HeavyHexGates::XxYyZz(J * dt);

        int nm = 2 * grid.n_bonds;
        std::vector<int>     frozen_chi(grid.n_bonds);
        std::vector<Scalar*> frozen_msg(nm, nullptr);
        std::vector<Scalar*> new_msg(nm, nullptr);

        size_t alloc = (size_t)chi_max * chi_max * sizeof(Scalar);
        for (auto& bond : grid.bonds) {
            for (int d = 0; d < 2; d++) {
                int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                HIP_CHECK(hipMalloc(&frozen_msg[idx], alloc));
                HIP_CHECK(hipMalloc(&new_msg[idx], alloc));
            }
        }

        for (int c = 0; c < 3; c++) {
            // Snapshot current state into frozen buffers
            for (auto& b : grid.bonds) frozen_chi[b.id] = b.chi;
            for (auto& bond : grid.bonds) {
                int chi = bond.chi;
                size_t sz = (size_t)chi * chi * sizeof(Scalar);
                for (int d = 0; d < 2; d++) {
                    int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(frozen_msg[idx], state.messages[idx],
                        sz, hipMemcpyDeviceToDevice));
                    HIP_CHECK(hipMemcpy(new_msg[idx], state.messages[idx],
                        sz, hipMemcpyDeviceToDevice));
                }
            }

            for (auto& gate : grid.colors[c]) {
                gate_ops.apply_binary_gate(
                    state.psi[gate.site_a], state.psi[gate.site_b],
                    gate, grid,
                    new_msg, frozen_msg, frozen_chi,
                    G.data(), ws, chi_max, cutoff, lm);
            }

            // Commit updated messages for this color class
            for (auto& gate : grid.colors[c]) {
                const Bond& bond = grid.bonds[gate.bond_id];
                int chi = bond.chi;
                size_t sz = (size_t)chi * chi * sizeof(Scalar);
                for (int d = 0; d < 2; d++) {
                    int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(state.messages[idx], new_msg[idx],
                        sz, hipMemcpyDeviceToDevice));
                }
            }
        }

        for (auto p : frozen_msg) if (p) HIP_CHECK(hipFree(p));
        for (auto p : new_msg)    if (p) HIP_CHECK(hipFree(p));

        int max_chi = 1;
        for (auto& bond : grid.bonds)
            max_chi = std::max(max_chi, bond.chi);
        return max_chi;
    }
};

/* =========================================================
 * NoisyHeavyHexSimulator — noisy Trotter layer for heavy-hex
 *
 * Each Trotter step applies:
 *   1. Hardware gate (XxYyZz PTM) per bond — same as HeavyHexSimulator
 *   2. Noise PTM per bond — from NoisyHeavyHexGrid::noise_ptm(bond_id)
 *
 * Noise is applied sequentially after hardware gates within each
 * color class, using the same frozen/new message protocol.
 * ========================================================= */
struct NoisyHeavyHexSimulator {
    rocblas_handle handle;
    GateOps gate_ops;
    Scalar J, dt;

    NoisyHeavyHexSimulator(rocblas_handle h_, Scalar J_, Scalar dt_)
        : handle(h_), gate_ops(h_), J(J_), dt(dt_) {}

    int trotter_layer(HeavyHexState& state, NoisyHeavyHexGrid& grid,
                      Workspace& ws, int chi_max, Scalar cutoff=1e-10f,
                      LayerMetrics* lm=nullptr)
    {
        auto G_hw = HeavyHexGates::XxYyZz(J * dt);

        int nm = 2 * grid.n_bonds;
        std::vector<int>     frozen_chi(grid.n_bonds);
        std::vector<Scalar*> frozen_msg(nm, nullptr);
        std::vector<Scalar*> new_msg(nm, nullptr);

        size_t alloc = (size_t)chi_max * chi_max * sizeof(Scalar);
        for (auto& bond : grid.bonds) {
            for (int d = 0; d < 2; d++) {
                int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                HIP_CHECK(hipMalloc(&frozen_msg[idx], alloc));
                HIP_CHECK(hipMalloc(&new_msg[idx], alloc));
            }
        }

        for (int c = 0; c < 3; c++) {
            // Snapshot current state
            for (auto& b : grid.bonds) frozen_chi[b.id] = b.chi;
            for (auto& bond : grid.bonds) {
                int chi = bond.chi;
                size_t sz = (size_t)chi * chi * sizeof(Scalar);
                for (int d = 0; d < 2; d++) {
                    int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(frozen_msg[idx], state.messages[idx],
                        sz, hipMemcpyDeviceToDevice));
                    HIP_CHECK(hipMemcpy(new_msg[idx], state.messages[idx],
                        sz, hipMemcpyDeviceToDevice));
                }
            }

            // Step 1: hardware gates
            for (auto& gate : grid.colors[c]) {
                gate_ops.apply_binary_gate(
                    state.psi[gate.site_a], state.psi[gate.site_b],
                    gate, grid, new_msg, frozen_msg, frozen_chi,
                    G_hw.data(), ws, chi_max, cutoff, lm);
            }

            // Step 2: noise gates (per-bond PTM, sequential)
            // Re-snapshot after hardware gates so noise sees updated messages
            for (auto& b : grid.bonds) frozen_chi[b.id] = b.chi;
            for (auto& bond : grid.bonds) {
                int chi = bond.chi;
                size_t sz = (size_t)chi * chi * sizeof(Scalar);
                for (int d = 0; d < 2; d++) {
                    int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(frozen_msg[idx], new_msg[idx],
                        sz, hipMemcpyDeviceToDevice));
                }
            }
            for (auto& gate : grid.colors[c]) {
                const float* G_noise = grid.noise_ptm(gate.bond_id);
                gate_ops.apply_binary_gate(
                    state.psi[gate.site_a], state.psi[gate.site_b],
                    gate, grid, new_msg, frozen_msg, frozen_chi,
                    G_noise, ws, chi_max, cutoff, nullptr);  // no metrics for noise
            }

            // Commit updated messages
            for (auto& gate : grid.colors[c]) {
                const Bond& bond = grid.bonds[gate.bond_id];
                int chi = bond.chi;
                size_t sz = (size_t)chi * chi * sizeof(Scalar);
                for (int d = 0; d < 2; d++) {
                    int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(state.messages[idx], new_msg[idx],
                        sz, hipMemcpyDeviceToDevice));
                }
            }
        }

        for (auto p : frozen_msg) if (p) HIP_CHECK(hipFree(p));
        for (auto p : new_msg)    if (p) HIP_CHECK(hipFree(p));

        int max_chi = 1;
        for (auto& bond : grid.bonds)
            max_chi = std::max(max_chi, bond.chi);
        return max_chi;
    }
};
