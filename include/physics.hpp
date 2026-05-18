#pragma once
/*
 * physics.hpp
 *
 * ═══════════════════════════════════════════════════════════════════
 * ISING MODEL — HEISENBERG PICTURE WITH PAULI BASIS (D=4)
 * ═══════════════════════════════════════════════════════════════════
 *
 * CRITICAL INSIGHT: D=4, NOT D=2
 *
 * Julia uses siteinds("Pauli", g) which gives D=4.
 * Each site tensor stores the operator in the PAULI BASIS:
 *   ψ_v = [a_I, a_X, a_Y, a_Z]
 * where O_v = a_I·I + a_X·X + a_Y·Y + a_Z·Z
 *
 * Initialization:
 *   Identity site: [1, 0, 0, 0]  (O = I)
 *   Z site:        [0, 0, 0, 1]  (O = Z)
 *
 * Adjoint action of gates in Pauli basis:
 *
 * For unitary U, the adjoint action on operator O is:
 *   O → U† O U
 *
 * In the Pauli basis, this is a 4×4 real matrix (the "superoperator"):
 *   [a_I']   [U† I U in Pauli basis] [a_I]
 *   [a_X'] = [U† X U in Pauli basis] [a_X]
 *   [a_Y']   [U† Y U in Pauli basis] [a_Y]
 *   [a_Z']   [U† Z U in Pauli basis] [a_Z]
 *
 * Rz(θ) adjoint action on Paulis:
 *   Rz† I Rz = I    → [1, 0, 0, 0]
 *   Rz† X Rz = cos(θ)X + sin(θ)Y  → [0, cos(θ), sin(θ), 0]
 *   Rz† Y Rz = -sin(θ)X + cos(θ)Y → [0,-sin(θ), cos(θ), 0]
 *   Rz† Z Rz = Z    → [0, 0, 0, 1]
 *
 * So the 4×4 superoperator for Rz(θ) (col-major, basis {I,X,Y,Z}):
 *   [[1,  0,      0,     0],
 *    [0,  cos(θ),-sin(θ),0],
 *    [0,  sin(θ), cos(θ),0],
 *    [0,  0,      0,     1]]
 *
 * KEY PROPERTY: Rz† Z Rz = Z → Z is invariant under Rz.
 * This means C(t) stays at 1 if only Rz is applied (J=0).
 *
 * Rxx(θ) = exp(-iθ/2 · X⊗X) adjoint action:
 *   More complex — mixes Pauli operators on two sites.
 *   The two-site superoperator is 16×16 in the Pauli⊗Pauli basis.
 *
 * ═══════════════════════════════════════════════════════════════════
 * COMPILE WITH: make PHYS_DIM=4
 * ═══════════════════════════════════════════════════════════════════
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "measures.hpp"
#include <cmath>
#include <vector>
#include <cstdio>

// Verify D=4 at compile time
static_assert(D == 4, "Heisenberg picture Ising simulation requires PHYS_DIM=4 (Pauli basis)");

/* =========================================================
 * Pauli basis index convention:
 *   0 = I (identity)
 *   1 = X (bit flip)
 *   2 = Y (bit+phase flip)
 *   3 = Z (phase flip)
 * ========================================================= */
enum Pauli { PI=0, PX=1, PY=2, PZ=3 };

/* =========================================================
 * IsingGates — adjoint action superoperators in Pauli basis
 *
 * These are 4×4 (unary) or 16×16 (binary) real matrices.
 * They act on the Pauli-basis components of the operator.
 * ========================================================= */
struct IsingGates {

    /* ── Rz(θ) superoperator [4×4 col-major] ────────────────────────
     *
     * Rz(θ) = exp(-iθ/2 · Z)
     * Adjoint action Rz†·P·Rz for each Pauli P:
     *   I → I
     *   X → cos(θ)·X + sin(θ)·Y
     *   Y → -sin(θ)·X + cos(θ)·Y
     *   Z → Z
     *
     * Matrix (cols = images of basis vectors {I,X,Y,Z}):
     * Col I: [1, 0,    0,   0]
     * Col X: [0, c,    s,   0]   c=cos(θ), s=sin(θ)
     * Col Y: [0,-s,    c,   0]
     * Col Z: [0, 0,    0,   1]
     *
     * Note: full angle θ (not θ/2) because this is the ADJOINT action
     * U†·P·U where U=Rz(θ/2) contains θ/2, giving net rotation θ. */
    static std::vector<Scalar> Rz(Scalar theta) {
        Scalar c = std::cos(theta);
        Scalar s = std::sin(theta);
        // 4×4 col-major: element [row,col] at row + 4*col
        return {
            1.f, 0.f, 0.f, 0.f,   // col 0: image of I
            0.f,   c,   s, 0.f,   // col 1: image of X
            0.f,  -s,   c, 0.f,   // col 2: image of Y
            0.f, 0.f, 0.f, 1.f    // col 3: image of Z
        };
    }

    /* ── Rxx(θ) superoperator [16×16 col-major] ─────────────────────
     *
     * Rxx(θ) = exp(-iθ/2 · X⊗X)
     * Two-site adjoint action on Pauli⊗Pauli basis.
     *
     * Basis ordering: {II, IX, IY, IZ, XI, XX, XY, XZ,
     *                  YI, YX, YY, YZ, ZI, ZX, ZY, ZZ}
     * Index: α·4 + β where α=Pauli on site1, β=Pauli on site2
     *
     * Adjoint action Rxx†·(P⊗Q)·Rxx for each basis element:
     *
     * Commutator analysis: [X⊗X, P⊗Q] determines the mixing.
     * XX commutes with: II, XX, IX, XI, YY, ZZ, XY, YX, XZ, ZX
     * XX anticommutes with: IY, IZ, YI, ZI, YZ, ZY
     *   (these get rotated by θ mixing with their partner)
     *
     * Key results (Heisenberg adjoint):
     *   II → II
     *   XX → XX
     *   IZ → cos(θ)·IZ - sin(θ)·XY   (mixes IZ↔XY)
     *   ZI → cos(θ)·ZI - sin(θ)·YX   (mixes ZI↔YX)
     *   ZZ → cos(θ)·ZZ + ...
     *   etc.
     *
     * Full 16×16 superoperator computed from:
     *   M[αβ, α'β'] = Tr((Pα⊗Pβ)† · Rxx† · (Pα'⊗Pβ') · Rxx) / 4
     */
    static std::vector<Scalar> Rxx(Scalar theta) {
        Scalar c = std::cos(theta);
        Scalar s = std::sin(theta);

        // 16×16 matrix, initialized to zero
        std::vector<Scalar> M(16*16, 0.f);

        // Helper: set M[row, col] = val (col-major: index = row + 16*col)
        auto set = [&](int r, int c_, Scalar v) { M[r + 16*c_] = v; };

        // Basis index: site1_pauli * 4 + site2_pauli
        // Commuting operators (invariant under Rxx — diagonal = 1):
        // These commute with X⊗X: {II,IX,XI,XX} (both sites ∈{I,X})
        // and {YY,YZ,ZY,ZZ} (both sites ∈{Y,Z}, each anticommutes twice → commutes)
        set(0,  0,  1.f);   // II → II
        set(1,  1,  1.f);   // IX → IX
        set(4,  4,  1.f);   // XI → XI
        set(5,  5,  1.f);   // XX → XX
        set(10, 10, 1.f);   // YY → YY
        set(11, 11, 1.f);   // YZ → YZ  (commutes with XX)
        set(14, 14, 1.f);   // ZY → ZY  (commutes with XX)
        set(15, 15, 1.f);   // ZZ → ZZ  (commutes with XX)

        // Rotating pairs — operators where exactly one site anticommutes with X.
        // [A,XX] determines rotation direction. Sign matters.
        //
        // [IY, XX] = -2i·XZ  →  IY → cos·IY - sin·XZ,  XZ → sin·IY + cos·XZ
        int IY=2, XZ=7;
        set(IY, IY,  c);  set(IY, XZ,  s);
        set(XZ, IY, -s);  set(XZ, XZ,  c);

        // [IZ, XX] = +2i·XY  →  IZ → cos·IZ + sin·XY,  XY → -sin·IZ + cos·XY
        int IZ=3, XY=6;
        set(IZ, IZ,  c);  set(IZ, XY, -s);
        set(XY, IZ,  s);  set(XY, XY,  c);

        // [YI, XX] = -2i·ZX  →  YI → cos·YI - sin·ZX,  ZX → sin·YI + cos·ZX
        int YI=8, ZX=13;
        set(YI, YI,  c);  set(YI, ZX,  s);
        set(ZX, YI, -s);  set(ZX, ZX,  c);

        // [ZI, XX] = +2i·YX  →  ZI → cos·ZI + sin·YX,  YX → -sin·ZI + cos·YX
        int ZI=12, YX=9;
        set(ZI, ZI,  c);  set(ZI, YX, -s);
        set(YX, ZI,  s);  set(YX, YX,  c);

        return M;
    }
};

/* =========================================================
 * IsingState — state tensors and BP messages on GPU
 * ========================================================= */
struct IsingState {
    std::vector<Scalar*> psi;
    std::vector<Scalar*> messages;
    int chi_max;
    int n_sites, n_bonds;

    IsingState(const Grid& g)
        : chi_max(g.chi_max), n_sites(g.n_sites), n_bonds(g.n_bonds)
    {
        psi.resize(n_sites, nullptr);
        messages.resize(2*n_bonds, nullptr); // 2 directed messages per bond

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

    ~IsingState() {
        for (auto p : psi)      if (p) HIP_CHECK(hipFree(p));
        for (auto p : messages) if (p) HIP_CHECK(hipFree(p));
    }

    IsingState(const IsingState&) = delete;
    IsingState& operator=(const IsingState&) = delete;

    /* ── Initialize Heisenberg picture state ─────────────────────────
     * Pauli basis (D=4): [a_I, a_X, a_Y, a_Z]
     * Identity site: [1, 0, 0, 0]   (O = I)
     * Z site:        [0, 0, 0, 1]   (O = Z)
     * BP messages: 1×1 = scalar 1.0
     */
    void init_heisenberg(Grid& g, int center_site) {
        std::vector<Scalar> h_I = {1.f, 0.f, 0.f, 0.f};  // I in Pauli basis
        std::vector<Scalar> h_Z = {0.f, 0.f, 0.f, 1.f};  // Z in Pauli basis

        for (auto& site : g.sites) {
            Scalar* data = (site.id == center_site) ?
                           h_Z.data() : h_I.data();
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

        printf("[IsingState] Pauli basis D=%d, center=%d, all chi=1\n",
               D, center_site);
    }

    size_t total_bytes(const Grid& g) const {
        size_t total = 0;
        for (auto& s : g.sites)
            total += (size_t)s.tensor_size_max(chi_max)*sizeof(Scalar);
        total += (size_t)n_bonds * 2 * chi_max*chi_max*sizeof(Scalar); // 2 directed msgs/bond
        return total;
    }
};

/* =========================================================
 * IsingSimulator — Trotter layer driver
 * ========================================================= */
struct IsingSimulator {
    rocblas_handle handle;
    GateOps gate_ops;
    Scalar h, J, dt;

    IsingSimulator(rocblas_handle h_, Scalar h_f, Scalar J_c, Scalar dt_)
        : handle(h_), gate_ops(h_), h(h_f), J(J_c), dt(dt_) {}

    void apply_rz_layer(IsingState& state, Grid& grid, Workspace& ws) {
        // Symmetric Trotter: Rz with half angle at each end
        auto G = IsingGates::Rz(h * dt);
        for (auto& site : grid.sites)
            gate_ops.apply_unary_gate(
                state.psi[site.id], site, grid, G.data(), ws.cv1);
    }

    void apply_rxx_layer(IsingState& state, Grid& grid, Workspace& ws,
                         int chi_max, Scalar cutoff=1e-10f,
                         Measures* meas=nullptr, int bp_sweeps_between=0,
                         int center_site=-1, LayerMetrics* lm=nullptr) {
        auto G = IsingGates::Rxx(2.f * J * dt);
        int nm = 2 * grid.n_bonds;

        // Allocate frozen/new message buffers (chi_max^2 each).
        std::vector<int> frozen_chi(grid.n_bonds);
        std::vector<Scalar*> frozen_msg(nm, nullptr);
        std::vector<Scalar*> new_msg(nm, nullptr);

        size_t alloc = (size_t)chi_max * chi_max * sizeof(Scalar);
        for (auto& bond : grid.bonds) {
            for (int d=0; d<2; d++) {
                int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                HIP_CHECK(hipMalloc(&frozen_msg[idx], alloc));
                HIP_CHECK(hipMalloc(&new_msg[idx], alloc));
            }
        }

        // Apply gates one color class at a time.
        // Re-freeze chi and messages at the START of each color class so that
        // bond dimensions grown by a previous color class are visible as env.
        // Within a color class gates share no bonds, so the frozen snapshot is
        // consistent for all gates in that class.
        for (int c=0; c<4; c++) {
            // Re-snapshot current state into frozen buffers.
            for (auto& b : grid.bonds) frozen_chi[b.id] = b.chi;
            for (auto& bond : grid.bonds) {
                int chi = bond.chi;
                size_t sz = (size_t)chi * chi * sizeof(Scalar);
                for (int d=0; d<2; d++) {
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
                    new_msg,    // output messages
                    frozen_msg, // env (frozen for this color class)
                    frozen_chi, // frozen bond dims for this color class
                    G.data(), ws, chi_max, cutoff, lm);
            }

            // Commit only the bonds that were updated by this color class.
            for (auto& gate : grid.colors[c]) {
                const Bond& bond = grid.bonds[gate.bond_id];
                int chi = bond.chi;
                size_t sz = (size_t)chi * chi * sizeof(Scalar);
                for (int d=0; d<2; d++) {
                    int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                    HIP_CHECK(hipMemcpy(state.messages[idx], new_msg[idx],
                        sz, hipMemcpyDeviceToDevice));
                }
            }

            (void)meas; (void)bp_sweeps_between; (void)center_site;
        }

        for (auto p : frozen_msg) if (p) HIP_CHECK(hipFree(p));
        for (auto p : new_msg)    if (p) HIP_CHECK(hipFree(p));
    }

    int trotter_layer(IsingState& state, Grid& grid, Workspace& ws,
                      int chi_max, Scalar cutoff=1e-10f,
                      Measures* meas=nullptr, int center_site=-1,
                      LayerMetrics* lm=nullptr) {
        printf("  [Trotter] Rz(%.4f) -> Rxx(%.4f) -> Rz(%.4f)\n",
               h*dt, 2.f*J*dt, h*dt);
        apply_rz_layer(state, grid, ws);
        // 100 BP sweeps between color classes — matches Julia's apply_gates
        apply_rxx_layer(state, grid, ws, chi_max, cutoff, meas, 100, center_site, lm);
        apply_rz_layer(state, grid, ws);

        int max_chi = 1;
        for (auto& bond : grid.bonds)
            max_chi = std::max(max_chi, bond.chi);
        return max_chi;
    }
};
