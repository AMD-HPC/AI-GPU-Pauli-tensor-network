#pragma once
/*
 * bilinear_form_bp.hpp — Two-state overlap ⟨ψ_a|ψ_b⟩ via BilinearForm BP
 *
 * Pure CPU. Matches Julia's inner(ψ_a, ψ_b; alg="bp").
 *
 * Handles DIFFERENT chi between states (noiseless vs noisy evolve differently).
 * T_v bond dim = chi_a × chi_b per bond (chi from each state's grid).
 *
 * Kronecker warm-start: pass msg_a_gpu / msg_b_gpu (converged BP messages)
 * to initialize double-layer messages as kron(diag(msg_a), diag(msg_b)).
 * This puts BP in the right basin of attraction for cross inner products
 * where independent QR runs produce states with arbitrary relative signs.
 *
 * Algorithm:
 *   1. Download ψ_a, ψ_b
 *   2. Build T_v[ia0*chi_b0+ib0,...] = Σ_d ψa[ia...,d] × ψb[ib...,d]
 *   3. Init messages: kron(diag(msg_a), diag(msg_b)) if provided, else identity
 *   4. BP sweeps with max-abs normalization
 *   5. Bethe: C = Π_v a_v / Π_e ze_e
 */

#include "grid.hpp"
#include "types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <hip/hip_runtime.h>

// ── StateSnapshot — CPU copy of psi + messages from any GPU device ───────────
// Download once (with correct hipSetDevice), then pass to bilinear_form_bp_cpu.
struct StateSnapshot {
    const Grid*                        grid;
    std::vector<std::vector<float>>    psi;   // [n_sites][vol]
    std::vector<std::vector<float>>    msgs;  // [2*n_bonds][chi*chi]

    void download(const Grid& g,
                  const std::vector<float*>& psi_gpu,
                  const std::vector<float*>& msg_gpu,
                  int device)
    {
        grid = &g;
        int n_sites = g.n_sites, n_bonds = g.n_bonds;
        psi.resize(n_sites);
        msgs.resize(2 * n_bonds);

        HIP_CHECK(hipSetDevice(device));
        for (auto& site : g.sites) {
            int vol = D;
            for (int d = 0; d < MAX_DEGREE; d++)
                if (site.bond_id[d] >= 0) vol *= g.bonds[site.bond_id[d]].chi;
            psi[site.id].resize(vol);
            HIP_CHECK(hipMemcpy(psi[site.id].data(), psi_gpu[site.id],
                vol * sizeof(float), hipMemcpyDeviceToHost));
        }
        for (auto& bond : g.bonds) {
            int sz = bond.chi * bond.chi;
            for (int d = 0; d < 2; d++) {
                int idx = (d == 0) ? bond.msg_from_a() : bond.msg_from_b();
                msgs[idx].resize(sz);
                HIP_CHECK(hipMemcpy(msgs[idx].data(), msg_gpu[idx],
                    sz * sizeof(float), hipMemcpyDeviceToHost));
            }
        }
    }
};

// Forward declaration of the core BP engine (works on CPU data).
float bilinear_form_bp_cpu(
    const StateSnapshot& sa,
    const StateSnapshot& sb,
    bool use_msg_warmstart = true,
    int  n_sweeps = 25,
    float tol = 1e-5f,
    bool verbose = false);

inline float bilinear_form_bp(
    const Grid&                grid_a,
    const std::vector<float*>& psi_a_gpu,
    const Grid&                grid_b,
    const std::vector<float*>& psi_b_gpu,
    // Optional: converged BP messages for sqrt-gauge transform.
    // Pass nullptr to skip (self-overlaps don't need it).
    const std::vector<float*>* msg_a_gpu = nullptr,  // 2*n_bonds: [msg_from_a, msg_from_b]
    const std::vector<float*>* msg_b_gpu = nullptr,
    int                        n_sweeps = 25,
    float                      tol = 1e-5f,
    bool                       verbose = false)
{
    int n_sites = grid_a.n_sites;
    int n_bonds = grid_a.n_bonds;

    // ── 1. Download ψ_a and ψ_b ─────────────────────────────────────
    std::vector<std::vector<float>> psi_a(n_sites), psi_b(n_sites);
    std::vector<std::vector<int>>   sha(n_sites), shb(n_sites);
    double log_scale_a = 0.0, log_scale_b = 0.0;

    // Download msg diagonals if sqrt-gauge requested
    // msg layout: msg[bond.msg_from_a()] = message from site_a → site_b (chi×chi)
    //             msg[bond.msg_from_b()] = message from site_b → site_a (chi×chi)
    std::vector<std::vector<float>> ma_cpu, mb_cpu;
    if (msg_a_gpu) {
        int nm = 2 * n_bonds;
        ma_cpu.resize(nm);
        for (auto& bond : grid_a.bonds) {
            int chi = bond.chi;
            int sz  = chi * chi;
            for (int d = 0; d < 2; d++) {
                int idx = (d == 0) ? bond.msg_from_a() : bond.msg_from_b();
                ma_cpu[idx].resize(sz);
                HIP_CHECK(hipMemcpy(ma_cpu[idx].data(), (*msg_a_gpu)[idx],
                    sz * sizeof(float), hipMemcpyDeviceToHost));
            }
        }
    }
    if (msg_b_gpu) {
        int nm = 2 * n_bonds;
        mb_cpu.resize(nm);
        for (auto& bond : grid_b.bonds) {
            int chi = bond.chi;
            int sz  = chi * chi;
            for (int d = 0; d < 2; d++) {
                int idx = (d == 0) ? bond.msg_from_a() : bond.msg_from_b();
                mb_cpu[idx].resize(sz);
                HIP_CHECK(hipMemcpy(mb_cpu[idx].data(), (*msg_b_gpu)[idx],
                    sz * sizeof(float), hipMemcpyDeviceToHost));
            }
        }
    }

    // Helper: absorb sqrt(diag(msg_incoming)) into tensor along each bond axis.
    // msg_incoming[bond k] = message sent FROM the other endpoint TO this site.
    // Multiplies psi[..., i_k, ...] by sqrt(|msg[i_k, i_k]|) for each bond axis k.
    auto absorb_sqrt_msgs = [&](
        std::vector<float>& psi,
        const std::vector<int>& shape,          // [chi_0, chi_1, ..., D]
        const Site& site,
        const Grid& grid,
        const std::vector<std::vector<float>>& msg_cpu)
    {
        int nb = (int)shape.size() - 1;
        for (int k = 0; k < nb; k++) {
            int chi_k = shape[k];
            int bid   = site.bond_id[k];
            const Bond& b = grid.bonds[bid];
            // Message incoming to site = sent from the other side
            int msg_idx = (site.id == b.site_a) ? b.msg_from_b() : b.msg_from_a();
            const auto& m = msg_cpu[msg_idx];  // chi×chi col-major
            // vol_before = product of dims 0..k-1; vol_after = product of dims k+1..end
            int vol_before = 1; for (int j = 0; j < k; j++) vol_before *= shape[j];
            int vol_after  = 1; for (int j = k+1; j < (int)shape.size(); j++) vol_after *= shape[j];
            for (int b2 = 0; b2 < vol_before; b2++)
                for (int i = 0; i < chi_k; i++) {
                    float w = std::sqrt(std::abs(m[i*(chi_k+1)]));  // diag: m[i*chi+i]
                    for (int a = 0; a < vol_after; a++)
                        psi[b2*chi_k*vol_after + i*vol_after + a] *= w;
                }
        }
    };

    for (auto& site : grid_a.sites) {
        std::vector<int> s;
        for (int d = 0; d < MAX_DEGREE; d++)
            if (site.bond_id[d] >= 0) s.push_back(grid_a.bonds[site.bond_id[d]].chi);
        s.push_back(D);
        sha[site.id] = s;
        int tot = 1; for (auto x : s) tot *= x;
        psi_a[site.id].resize(tot);
        HIP_CHECK(hipMemcpy(psi_a[site.id].data(), psi_a_gpu[site.id],
            tot * sizeof(float), hipMemcpyDeviceToHost));
        // absorb_sqrt_msgs disabled — QR positive-diagonal fix handles sign
        float nm = 0.f; for (auto v : psi_a[site.id]) nm = std::max(nm, std::abs(v));
        if (nm > 1e-30f) { for (auto& v : psi_a[site.id]) v /= nm; log_scale_a += std::log((double)nm); }
    }

    for (auto& site : grid_b.sites) {
        std::vector<int> s;
        for (int d = 0; d < MAX_DEGREE; d++)
            if (site.bond_id[d] >= 0) s.push_back(grid_b.bonds[site.bond_id[d]].chi);
        s.push_back(D);
        shb[site.id] = s;
        int tot = 1; for (auto x : s) tot *= x;
        psi_b[site.id].resize(tot);
        HIP_CHECK(hipMemcpy(psi_b[site.id].data(), psi_b_gpu[site.id],
            tot * sizeof(float), hipMemcpyDeviceToHost));
        // absorb_sqrt_msgs disabled
        float nm = 0.f; for (auto v : psi_b[site.id]) nm = std::max(nm, std::abs(v));
        if (nm > 1e-30f) { for (auto& v : psi_b[site.id]) v /= nm; log_scale_b += std::log((double)nm); }
    }

    // Dense bond list per site
    std::vector<std::vector<int>> bonds_of(n_sites);
    for (auto& site : grid_a.sites)
        for (int d = 0; d < MAX_DEGREE; d++)
            if (site.bond_id[d] >= 0)
                bonds_of[site.id].push_back(site.bond_id[d]);

    // ── 2. Build T_v ─────────────────────────────────────────────────
    std::vector<std::vector<float>> T(n_sites);
    std::vector<std::vector<int>>   T_shape(n_sites);

    for (auto& site : grid_a.sites) {
        const auto& sa = sha[site.id]; const auto& sb = shb[site.id];
        int nb = (int)sa.size() - 1; int Dp = sa.back();
        std::vector<int> chia(nb), chib(nb);
        int vol_a = 1, vol_b = 1;
        for (int k = 0; k < nb; k++) {
            chia[k] = sa[k]; vol_a *= chia[k];
            chib[k] = sb[k]; vol_b *= chib[k];
        }
        const float* pa = psi_a[site.id].data();
        const float* pb = psi_b[site.id].data();

        std::vector<float> T_big(vol_a * vol_b, 0.f);
        for (int ia = 0; ia < vol_a; ia++)
            for (int ib = 0; ib < vol_b; ib++)
                for (int d = 0; d < Dp; d++)
                    T_big[ia * vol_b + ib] += pa[ia * Dp + d] * pb[ib * Dp + d];

        std::vector<int> T_sh;
        for (int k = 0; k < nb; k++) T_sh.push_back(chia[k] * chib[k]);
        int total_T = 1; for (auto s : T_sh) total_T *= s;
        std::vector<float> T_out(total_T, 0.f);

        for (int ia_f = 0; ia_f < vol_a; ia_f++)
        for (int ib_f = 0; ib_f < vol_b; ib_f++) {
            float val = T_big[ia_f * vol_b + ib_f];
            std::vector<int> ia(nb), ib(nb);
            int ta = ia_f, tb = ib_f;
            for (int k = nb-1; k >= 0; k--) { ia[k] = ta % chia[k]; ta /= chia[k]; ib[k] = tb % chib[k]; tb /= chib[k]; }
            int idx = 0;
            for (int k = 0; k < nb; k++) idx = idx * T_sh[k] + ia[k] * chib[k] + ib[k];
            T_out[idx] = val;
        }
        T[site.id] = T_out; T_shape[site.id] = T_sh;
    }

    // ── 3. Initialize messages ────────────────────────────────────────
    // If msgs provided: warm-start with kron(diag(msg_a), diag(msg_b)).
    // This encodes the actual bond content of both states, breaking sign
    // degeneracy that causes BP to converge to the wrong fixed point.
    // M_fwd[bond][ia*cb+ib] = diag_a[ia] × diag_b[ib]  (outer product of diagonals)
    std::vector<std::vector<float>> M_fwd(n_bonds), M_bwd(n_bonds);
    for (auto& bond : grid_a.bonds) {
        int ca = bond.chi, cb = grid_b.bonds[bond.id].chi;
        int cab = ca * cb;
        M_fwd[bond.id].assign(cab, 0.f); M_bwd[bond.id].assign(cab, 0.f);

        if (msg_a_gpu && msg_b_gpu) {
            // Extract diagonals of msg_a and msg_b for this bond
            // msg_from_a = message flowing a→b (fwd), msg_from_b = b→a (bwd)
            auto diag = [](const std::vector<float>& m, int chi) {
                std::vector<float> d(chi);
                for (int i = 0; i < chi; i++) d[i] = std::abs(m[i*(chi+1)]);
                return d;
            };
            auto da_fwd = diag(ma_cpu[bond.msg_from_a()], ca);
            auto da_bwd = diag(ma_cpu[bond.msg_from_b()], ca);
            auto db_fwd = diag(mb_cpu[bond.msg_from_a()], cb);
            auto db_bwd = diag(mb_cpu[bond.msg_from_b()], cb);

            // kron outer product: M[ia*cb+ib] = da[ia] * db[ib]
            for (int ia = 0; ia < ca; ia++)
                for (int ib = 0; ib < cb; ib++) {
                    M_fwd[bond.id][ia*cb+ib] = da_fwd[ia] * db_fwd[ib];
                    M_bwd[bond.id][ia*cb+ib] = da_bwd[ia] * db_bwd[ib];
                }

            // Normalize
            float sf = 0.f; for (auto v : M_fwd[bond.id]) sf = std::max(sf, v);
            float sb = 0.f; for (auto v : M_bwd[bond.id]) sb = std::max(sb, v);
            if (sf > 1e-20f) for (auto& v : M_fwd[bond.id]) v /= sf;
            if (sb > 1e-20f) for (auto& v : M_bwd[bond.id]) v /= sb;
        } else {
            // Default: identity warm-start (works for self-overlaps)
            int mn = std::min(ca, cb);
            float iv = (mn > 0) ? 1.f / mn : 1.f;
            for (int i = 0; i < mn; i++) {
                M_fwd[bond.id][i*cb+i] = iv;
                M_bwd[bond.id][i*cb+i] = iv;
            }
        }
    }

    auto get_inc = [&](int bid, int sid) -> const float* {
        const Bond& b = grid_a.bonds[bid];
        return (sid == b.site_a) ? M_bwd[bid].data() : M_fwd[bid].data();
    };
    auto get_out = [&](int bid, int sid) -> std::vector<float>& {
        const Bond& b = grid_a.bonds[bid];
        return (sid == b.site_a) ? M_fwd[bid] : M_bwd[bid];
    };

    // ── 4. BP sweeps ─────────────────────────────────────────────────
    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        float avg_delta = 0.f; int n_msgs = 0;
        for (auto& bond : grid_a.bonds) {
            for (int dir = 0; dir < 2; dir++) {
                int src = (dir == 0) ? bond.site_a : bond.site_b;
                const auto& sh = T_shape[src];
                int ndim = (int)sh.size();
                const auto& db = bonds_of[src];
                int skip = -1;
                for (int k = 0; k < (int)db.size(); k++) if (db[k] == bond.id) { skip = k; break; }
                if (skip < 0) continue;
                int chi_out = sh[skip];
                int total = 1; for (auto s : sh) total *= s;
                std::vector<float> nm(chi_out, 0.f);
                for (int flat = 0; flat < total; flat++) {
                    std::vector<int> idx(ndim);
                    int tmp = flat;
                    for (int d = ndim-1; d >= 0; d--) { idx[d] = tmp % sh[d]; tmp /= sh[d]; }
                    float prod = T[src][flat];
                    for (int k = 0; k < ndim; k++) {
                        if (k == skip) continue;
                        prod *= get_inc(db[k], src)[idx[k]];
                    }
                    nm[idx[skip]] += prod;
                }
                float s = 0.f; for (auto v : nm) s += v;
                if (std::abs(s) > 1e-20f) for (auto& v : nm) v /= s;
                auto& om = get_out(bond.id, src);
                for (int k = 0; k < chi_out; k++) { avg_delta += std::abs(nm[k] - om[k]); n_msgs++; }
                om = nm;
            }
        }
        if (n_msgs > 0 && avg_delta / n_msgs < tol) break;
    }

    // ── 5. Bethe PF ──────────────────────────────────────────────────
    double log_num = 0.0, sign_num = 1.0;
    for (auto& site : grid_a.sites) {
        const auto& sh = T_shape[site.id];
        int ndim = (int)sh.size(), total = 1;
        for (auto s : sh) total *= s;
        const auto& db = bonds_of[site.id];
        double a_v = 0.0;
        for (int flat = 0; flat < total; flat++) {
            std::vector<int> idx(ndim);
            int tmp = flat;
            for (int d = ndim-1; d >= 0; d--) { idx[d] = tmp % sh[d]; tmp /= sh[d]; }
            float prod = T[site.id][flat];
            for (int k = 0; k < ndim; k++) prod *= get_inc(db[k], site.id)[idx[k]];
            a_v += (double)prod;
        }
        if (a_v == 0.0) { sign_num = 0.0; break; }
        sign_num *= (a_v > 0.0 ? 1.0 : -1.0);
        log_num  += std::log(std::abs(a_v));
    }

    double log_den = 0.0, sign_den = 1.0;
    for (auto& bond : grid_a.bonds) {
        int cab = (int)M_fwd[bond.id].size();
        double ze = 0.0;
        for (int k = 0; k < cab; k++) ze += (double)M_fwd[bond.id][k] * M_bwd[bond.id][k];
        if (std::abs(ze) > 1e-30) {
            sign_den *= (ze > 0 ? 1.0 : -1.0);
            log_den  += std::log(std::abs(ze));
        }
    }

    if (sign_num == 0.0) return 0.f;
    double result = sign_num * sign_den * std::exp(log_num - log_den + log_scale_a + log_scale_b);
    if (verbose)
        printf("  [BLF] sign_num=%+.0f sign_den=%+.0f log_num=%.4f log_den=%.4f"
               " log_sa=%.4f log_sb=%.4f => %.6f\n",
               sign_num, sign_den, log_num, log_den,
               log_scale_a, log_scale_b, (float)result);
    return (float)result;
}

// ── bilinear_form_sqrt_gauge ──────────────────────────────────────────────────
//
// Computes ⟨ψ_a|ψ_b⟩ using asymmetric sqrt gauge on diagonal BP messages.
//
// For each bond e with diagonal message m_ref[i] (from a shared reference):
//   L[i] = sign(m_ref[i]) × sqrt(|m_ref[i]|)   absorbed into ψ_a bond axis
//   R[i] = sqrt(|m_ref[i]|)                      absorbed into ψ_b bond axis
//
// This makes the gauge consistent across all pairs without complex arithmetic.
// Reference messages: diagonal of sa.msgs (ψ_a's own converged BP messages).
// For cross terms: use the state with larger chi as reference.
//
// After gauge transform, runs standard bilinear_form_bp_cpu with identity init.
//
inline float bilinear_form_sqrt_gauge(
    const StateSnapshot& sa,   // reference state (use its messages as gauge)
    const StateSnapshot& sb,
    int   n_sweeps = 25,
    float tol      = 1e-5f,
    bool  verbose  = false)
{
    const Grid& grid_a = *sa.grid;
    const Grid& grid_b = *sb.grid;
    int n_sites = grid_a.n_sites;

    // Build gauge-transformed psi copies
    StateSnapshot sa_g, sb_g;
    sa_g.grid = &grid_a;  sa_g.psi = sa.psi;  sa_g.msgs = sa.msgs;
    sb_g.grid = &grid_b;  sb_g.psi = sb.psi;  sb_g.msgs = {};  // no msgs for sb

    // For each site, absorb L (signed sqrt) into ψ_a and R (unsigned sqrt) into ψ_b
    // along each bond axis, using the diagonal of sa.msgs as reference.
    auto apply_gauge = [&](
        std::vector<float>& psi,
        const std::vector<int>& shape,   // [chi_0,...,chi_{nb-1}, D]
        const Site& site,
        const Grid& grid,
        bool signed_side)  // true=L (signed sqrt), false=R (unsigned sqrt)
    {
        int nb = (int)shape.size() - 1;
        for (int k = 0; k < nb; k++) {
            int chi_k = shape[k];
            int bid   = site.bond_id[k];
            if (bid < 0) continue;
            const Bond& b = grid_a.bonds[bid];  // always reference grid_a for messages
            // Incoming message to this site from the other endpoint
            int msg_idx = (site.id == b.site_a) ? b.msg_from_b() : b.msg_from_a();
            if (msg_idx >= (int)sa.msgs.size() || sa.msgs[msg_idx].empty()) continue;
            const auto& m = sa.msgs[msg_idx];  // chi_ref × chi_ref, diagonal

            // chi_ref may differ from chi_k (e.g. for ψ_b with different chi)
            int chi_ref = (int)std::sqrt((float)m.size() + 0.5f);  // m is chi×chi

            int vol_before = 1; for (int j = 0; j < k; j++) vol_before *= shape[j];
            int vol_after  = 1; for (int j = k+1; j < (int)shape.size(); j++) vol_after *= shape[j];

            for (int b2 = 0; b2 < vol_before; b2++)
                for (int i = 0; i < chi_k; i++) {
                    // Diagonal element: m[i,i] = m[i*(chi_ref+1)]
                    float mi = (i < chi_ref) ? m[i*(chi_ref+1)] : 0.f;
                    float w;
                    if (signed_side)
                        w = (mi >= 0.f ? 1.f : -1.f) * std::sqrt(std::abs(mi));
                    else
                        w = std::sqrt(std::abs(mi));
                    for (int a = 0; a < vol_after; a++)
                        psi[b2*chi_k*vol_after + i*vol_after + a] *= w;
                }
        }
    };

    // Apply L to ψ_a, R to ψ_b
    for (auto& site : grid_a.sites) {
        std::vector<int> s;
        for (int d = 0; d < MAX_DEGREE; d++)
            if (site.bond_id[d] >= 0) s.push_back(grid_a.bonds[site.bond_id[d]].chi);
        s.push_back(D);
        apply_gauge(sa_g.psi[site.id], s, site, grid_a, true);   // L = signed sqrt
    }
    for (auto& site : grid_b.sites) {
        std::vector<int> s;
        for (int d = 0; d < MAX_DEGREE; d++)
            if (site.bond_id[d] >= 0) s.push_back(grid_b.bonds[site.bond_id[d]].chi);
        s.push_back(D);
        apply_gauge(sb_g.psi[site.id], s, site, grid_b, false);  // R = unsigned sqrt
    }

    // Run standard BP on gauge-transformed tensors (no warmstart needed — gauge is consistent)
    return bilinear_form_bp_cpu(sa_g, sb_g, false, n_sweeps, tol, verbose);
}

// ── bilinear_form_bp_cpu — works on pre-downloaded StateSnapshot data ─────────
// Use this for cross-GPU inner products: download each state on its own device,
// then call this function which is pure CPU (no hipMemcpy inside).
inline float bilinear_form_bp_cpu(
    const StateSnapshot& sa,
    const StateSnapshot& sb,
    bool  use_msg_warmstart,
    int   n_sweeps,
    float tol,
    bool  verbose)
{
    const Grid& grid_a = *sa.grid;
    const Grid& grid_b = *sb.grid;
    int n_sites = grid_a.n_sites;
    int n_bonds = grid_a.n_bonds;

    std::vector<std::vector<float>> psi_a(n_sites), psi_b(n_sites);
    std::vector<std::vector<int>>   sha(n_sites), shb(n_sites);
    double log_scale_a = 0.0, log_scale_b = 0.0;

    for (auto& site : grid_a.sites) {
        std::vector<int> s;
        for (int d = 0; d < MAX_DEGREE; d++)
            if (site.bond_id[d] >= 0) s.push_back(grid_a.bonds[site.bond_id[d]].chi);
        s.push_back(D);
        sha[site.id] = s;
        psi_a[site.id] = sa.psi[site.id];  // CPU copy — no memcpy needed
        float nm = 0.f; for (auto v : psi_a[site.id]) nm = std::max(nm, std::abs(v));
        if (nm > 1e-30f) { for (auto& v : psi_a[site.id]) v /= nm; log_scale_a += std::log((double)nm); }
    }
    for (auto& site : grid_b.sites) {
        std::vector<int> s;
        for (int d = 0; d < MAX_DEGREE; d++)
            if (site.bond_id[d] >= 0) s.push_back(grid_b.bonds[site.bond_id[d]].chi);
        s.push_back(D);
        shb[site.id] = s;
        psi_b[site.id] = sb.psi[site.id];
        float nm = 0.f; for (auto v : psi_b[site.id]) nm = std::max(nm, std::abs(v));
        if (nm > 1e-30f) { for (auto& v : psi_b[site.id]) v /= nm; log_scale_b += std::log((double)nm); }
    }

    // Dense bond list per site
    std::vector<std::vector<int>> bonds_of(n_sites);
    for (auto& site : grid_a.sites)
        for (int d = 0; d < MAX_DEGREE; d++)
            if (site.bond_id[d] >= 0)
                bonds_of[site.id].push_back(site.bond_id[d]);

    // Build T_v
    std::vector<std::vector<float>> T(n_sites);
    std::vector<std::vector<int>>   T_shape(n_sites);
    for (auto& site : grid_a.sites) {
        const auto& s_a = sha[site.id]; const auto& s_b = shb[site.id];
        int nb = (int)s_a.size() - 1; int Dp = s_a.back();
        std::vector<int> chia(nb), chib(nb);
        int vol_a = 1, vol_b = 1;
        for (int k = 0; k < nb; k++) { chia[k]=s_a[k]; vol_a*=chia[k]; chib[k]=s_b[k]; vol_b*=chib[k]; }
        const float* pa = psi_a[site.id].data();
        const float* pb = psi_b[site.id].data();
        std::vector<float> T_big(vol_a*vol_b, 0.f);
        for (int ia=0;ia<vol_a;ia++) for (int ib=0;ib<vol_b;ib++) for (int d=0;d<Dp;d++)
            T_big[ia*vol_b+ib] += pa[ia*Dp+d]*pb[ib*Dp+d];
        std::vector<int> T_sh; for (int k=0;k<nb;k++) T_sh.push_back(chia[k]*chib[k]);
        int total_T=1; for (auto s:T_sh) total_T*=s;
        std::vector<float> T_out(total_T,0.f);
        for (int ia_f=0;ia_f<vol_a;ia_f++) for (int ib_f=0;ib_f<vol_b;ib_f++) {
            float val=T_big[ia_f*vol_b+ib_f];
            std::vector<int> ia(nb),ib(nb); int ta=ia_f,tb=ib_f;
            for (int k=nb-1;k>=0;k--){ia[k]=ta%chia[k];ta/=chia[k];ib[k]=tb%chib[k];tb/=chib[k];}
            int idx=0; for (int k=0;k<nb;k++) idx=idx*T_sh[k]+ia[k]*chib[k]+ib[k];
            T_out[idx]=val;
        }
        T[site.id]=T_out; T_shape[site.id]=T_sh;
    }

    // Init messages
    std::vector<std::vector<float>> M_fwd(n_bonds), M_bwd(n_bonds);
    for (auto& bond : grid_a.bonds) {
        int ca=bond.chi, cb=grid_b.bonds[bond.id].chi, cab=ca*cb;
        M_fwd[bond.id].assign(cab,0.f); M_bwd[bond.id].assign(cab,0.f);
        if (use_msg_warmstart && !sa.msgs.empty() && !sb.msgs.empty()) {
            auto diag=[](const std::vector<float>& m,int chi){
                std::vector<float> d(chi); for(int i=0;i<chi;i++) d[i]=std::abs(m[i*(chi+1)]); return d;};
            auto da_fwd=diag(sa.msgs[bond.msg_from_a()],ca);
            auto da_bwd=diag(sa.msgs[bond.msg_from_b()],ca);
            auto db_fwd=diag(sb.msgs[bond.msg_from_a()],cb);
            auto db_bwd=diag(sb.msgs[bond.msg_from_b()],cb);
            for (int ia=0;ia<ca;ia++) for (int ib=0;ib<cb;ib++){
                M_fwd[bond.id][ia*cb+ib]=da_fwd[ia]*db_fwd[ib];
                M_bwd[bond.id][ia*cb+ib]=da_bwd[ia]*db_bwd[ib];
            }
            float sf=0.f; for(auto v:M_fwd[bond.id]) sf=std::max(sf,v);
            float sb2=0.f; for(auto v:M_bwd[bond.id]) sb2=std::max(sb2,v);
            if(sf>1e-20f) for(auto& v:M_fwd[bond.id]) v/=sf;
            if(sb2>1e-20f) for(auto& v:M_bwd[bond.id]) v/=sb2;
        } else {
            int mn=std::min(ca,cb); float iv=(mn>0)?1.f/mn:1.f;
            for(int i=0;i<mn;i++){M_fwd[bond.id][i*cb+i]=iv; M_bwd[bond.id][i*cb+i]=iv;}
        }
    }

    auto get_inc=[&](int bid,int sid)->const float*{
        const Bond& b=grid_a.bonds[bid]; return (sid==b.site_a)?M_bwd[bid].data():M_fwd[bid].data();};
    auto get_out=[&](int bid,int sid)->std::vector<float>&{
        const Bond& b=grid_a.bonds[bid]; return (sid==b.site_a)?M_fwd[bid]:M_bwd[bid];};

    // BP sweeps
    for (int sweep=0;sweep<n_sweeps;sweep++){
        float avg_delta=0.f; int n_msgs=0;
        for (auto& bond:grid_a.bonds) for (int dir=0;dir<2;dir++){
            int src=(dir==0)?bond.site_a:bond.site_b;
            const auto& sh=T_shape[src]; int ndim=(int)sh.size();
            const auto& db=bonds_of[src];
            int skip=-1; for(int k=0;k<(int)db.size();k++) if(db[k]==bond.id){skip=k;break;}
            if(skip<0) continue;
            int chi_out=sh[skip], total=1; for(auto s:sh) total*=s;
            std::vector<float> nm(chi_out,0.f);
            for(int flat=0;flat<total;flat++){
                std::vector<int> idx(ndim); int tmp=flat;
                for(int d=ndim-1;d>=0;d--){idx[d]=tmp%sh[d];tmp/=sh[d];}
                float prod=T[src][flat];
                for(int k=0;k<ndim;k++){if(k==skip)continue; prod*=get_inc(db[k],src)[idx[k]];}
                nm[idx[skip]]+=prod;
            }
            float s=0.f; for(auto v:nm) s+=v;
            if(std::abs(s)>1e-20f) for(auto& v:nm) v/=s;
            auto& om=get_out(bond.id,src);
            for(int k=0;k<chi_out;k++){avg_delta+=std::abs(nm[k]-om[k]);n_msgs++;}
            om=nm;
        }
        if(n_msgs>0&&avg_delta/n_msgs<tol) break;
    }

    // Bethe PF
    double log_num=0.0,sign_num=1.0;
    for(auto& site:grid_a.sites){
        const auto& sh=T_shape[site.id]; int ndim=(int)sh.size(),total=1;
        for(auto s:sh) total*=s;
        const auto& db=bonds_of[site.id]; double a_v=0.0;
        for(int flat=0;flat<total;flat++){
            std::vector<int> idx(ndim); int tmp=flat;
            for(int d=ndim-1;d>=0;d--){idx[d]=tmp%sh[d];tmp/=sh[d];}
            float prod=T[site.id][flat];
            for(int k=0;k<ndim;k++) prod*=get_inc(db[k],site.id)[idx[k]];
            a_v+=(double)prod;
        }
        if(a_v==0.0){sign_num=0.0;break;}
        sign_num*=(a_v>0.0?1.0:-1.0); log_num+=std::log(std::abs(a_v));
    }
    double log_den=0.0,sign_den=1.0;
    for(auto& bond:grid_a.bonds){
        int cab=(int)M_fwd[bond.id].size(); double ze=0.0;
        for(int k=0;k<cab;k++) ze+=(double)M_fwd[bond.id][k]*M_bwd[bond.id][k];
        if(std::abs(ze)>1e-30){sign_den*=(ze>0?1.0:-1.0); log_den+=std::log(std::abs(ze));}
    }
    if(sign_num==0.0) return 0.f;
    double result=sign_num*sign_den*std::exp(log_num-log_den+log_scale_a+log_scale_b);
    if(verbose) printf("  [BLF_CPU] sign_num=%+.0f sign_den=%+.0f => %.6f\n",
                       sign_num,sign_den,(float)result);
    return (float)result;
}

// ── bilinear_form_factored ────────────────────────────────────────────────────
//
// Computes ⟨ψ_a|ψ_b⟩ using FACTORED product-form messages.
//
// Each bond message M_e[ia, ib] = mu_a[ia] × mu_b[ib]  (rank-1 outer product).
// This avoids building chi_a×chi_b tensors explicitly.
//
// BP update for message from site v toward bond k:
//   For each (ia_k, ib_k):
//     contract ψ_a[...,d] with Π_{j≠k} mu_a_j[ia_j]  → f_a[ia_k, d]
//     contract ψ_b[...,d] with Π_{j≠k} mu_b_j[ib_j]  → f_b[ib_k, d]
//     nm_a[ia_k] = Σ_{ib_k, d} f_a[ia_k,d] × f_b[ib_k,d] × mu_b_k[ib_k]
//     nm_b[ib_k] = Σ_{ia_k, d} f_a[ia_k,d] × f_b[ib_k,d] × mu_a_k[ia_k]
//
// Cost: O(chi_a^deg × chi_b × D + chi_b^deg × chi_a × D) per site per sweep.
// For B[i] with chi_a=chi_n=150, chi_b=chi_nl=1000, deg=3:
//   150³×1000×4 + 1000³×150×4 ≈ way too big still for chi_nl=1000.
//
// PRACTICAL USE: A[i,j] with chi_a=chi_b=chi_n (both noisy, same chi) →
//   O(chi_n^deg × chi_n × D) per site.
// For B[i]: swap order → use ψ_nl as "a" (environment), ψ_n as "b" (operator).
//   Contract ψ_nl first (absorb mu_nl messages), then sum over ψ_n.
//   Cost: O(chi_nl^deg × chi_n × D) per site — matches autocorrelation_with_bp.
//
inline float bilinear_form_factored(
    const StateSnapshot& sa,   // "environment" state (larger chi → put here)
    const StateSnapshot& sb,   // "operator" state   (smaller chi)
    int   n_sweeps = 25,
    float tol      = 1e-5f,
    bool  verbose  = false)
{
    const Grid& grid_a = *sa.grid;
    const Grid& grid_b = *sb.grid;
    int n_sites = grid_a.n_sites;
    int n_bonds = grid_a.n_bonds;

    // Download shapes
    std::vector<std::vector<int>> sha(n_sites), shb(n_sites);
    for (auto& site : grid_a.sites) {
        for (int d = 0; d < MAX_DEGREE; d++)
            if (site.bond_id[d] >= 0) sha[site.id].push_back(grid_a.bonds[site.bond_id[d]].chi);
        sha[site.id].push_back(D);
    }
    for (auto& site : grid_b.sites) {
        for (int d = 0; d < MAX_DEGREE; d++)
            if (site.bond_id[d] >= 0) shb[site.id].push_back(grid_b.bonds[site.bond_id[d]].chi);
        shb[site.id].push_back(D);
    }

    // Normalize psi copies (log scale)
    std::vector<std::vector<float>> psi_a(n_sites), psi_b(n_sites);
    double log_scale_a = 0.0, log_scale_b = 0.0;
    for (auto& site : grid_a.sites) {
        psi_a[site.id] = sa.psi[site.id];
        float nm = 0.f; for (auto v : psi_a[site.id]) nm = std::max(nm, std::abs(v));
        if (nm > 1e-30f) { for (auto& v : psi_a[site.id]) v /= nm; log_scale_a += std::log((double)nm); }
    }
    for (auto& site : grid_b.sites) {
        psi_b[site.id] = sb.psi[site.id];
        float nm = 0.f; for (auto v : psi_b[site.id]) nm = std::max(nm, std::abs(v));
        if (nm > 1e-30f) { for (auto& v : psi_b[site.id]) v /= nm; log_scale_b += std::log((double)nm); }
    }

    // Bond list per site
    std::vector<std::vector<int>> bonds_of(n_sites);
    for (auto& site : grid_a.sites)
        for (int d = 0; d < MAX_DEGREE; d++)
            if (site.bond_id[d] >= 0) bonds_of[site.id].push_back(site.bond_id[d]);

    // Factored messages: mu_a[bond][dir][chi_a], mu_b[bond][dir][chi_b]
    // dir=0: fwd (site_a→site_b), dir=1: bwd (site_b→site_a)
    std::vector<std::array<std::vector<float>,2>> mu_a(n_bonds), mu_b(n_bonds);
    for (auto& bond : grid_a.bonds) {
        int ca = bond.chi, cb = grid_b.bonds[bond.id].chi;
        for (int d = 0; d < 2; d++) {
            mu_a[bond.id][d].assign(ca, 1.f / ca);
            mu_b[bond.id][d].assign(cb, 1.f / cb);
        }
        // Warm-start from converged single-layer messages (diagonal)
        if (!sa.msgs.empty()) {
            for (int d = 0; d < 2; d++) {
                int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                for (int i = 0; i < ca; i++)
                    mu_a[bond.id][d][i] = std::abs(sa.msgs[idx][i*(ca+1)]);
                float s = 0.f; for (auto v : mu_a[bond.id][d]) s += v;
                if (s > 1e-20f) for (auto& v : mu_a[bond.id][d]) v /= s;
            }
        }
        if (!sb.msgs.empty()) {
            for (int d = 0; d < 2; d++) {
                int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                for (int i = 0; i < cb; i++)
                    mu_b[bond.id][d][i] = std::abs(sb.msgs[idx][i*(cb+1)]);
                float s = 0.f; for (auto v : mu_b[bond.id][d]) s += v;
                if (s > 1e-20f) for (auto& v : mu_b[bond.id][d]) v /= s;
            }
        }
    }

    // Helper: get incoming mu_a for bond bid at site sid
    auto get_mu_a = [&](int bid, int sid) -> const std::vector<float>& {
        const Bond& b = grid_a.bonds[bid];
        return (sid == b.site_a) ? mu_a[bid][1] : mu_a[bid][0];  // incoming = other dir
    };
    auto get_mu_b = [&](int bid, int sid) -> const std::vector<float>& {
        const Bond& b = grid_b.bonds[bid];
        return (sid == b.site_a) ? mu_b[bid][1] : mu_b[bid][0];
    };
    auto get_out_a = [&](int bid, int sid) -> std::vector<float>& {
        const Bond& b = grid_a.bonds[bid];
        return (sid == b.site_a) ? mu_a[bid][0] : mu_a[bid][1];
    };
    auto get_out_b = [&](int bid, int sid) -> std::vector<float>& {
        const Bond& b = grid_b.bonds[bid];
        return (sid == b.site_a) ? mu_b[bid][0] : mu_b[bid][1];
    };

    // BP sweeps with factored messages
    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        float avg_delta = 0.f; int n_msgs = 0;

        for (auto& bond : grid_a.bonds) {
            for (int dir = 0; dir < 2; dir++) {
                int src = (dir==0) ? bond.site_a : bond.site_b;
                const auto& sa_sh = sha[src];   // [chi_0,...,D]
                const auto& sb_sh = shb[src];
                int nb = (int)sa_sh.size() - 1;
                int Dp = sa_sh.back();
                const auto& db = bonds_of[src];

                int skip = -1;
                for (int k = 0; k < (int)db.size(); k++)
                    if (db[k] == bond.id) { skip = k; break; }
                if (skip < 0) continue;

                int ca_skip = sa_sh[skip];
                int cb_skip = sb_sh[skip];

                // vol_a / vol_b: product of all bond chis except skip
                int vol_a = 1, vol_b = 1;
                for (int k = 0; k < nb; k++) {
                    if (k != skip) { vol_a *= sa_sh[k]; vol_b *= sb_sh[k]; }
                }

                // Step 1: contract ψ_a with Π_{j≠skip} mu_a_j → f_a[ia_skip, d]
                // ψ_a shape: [chi_0,...,chi_{nb-1}, D] row-major
                // f_a[ia_skip, d] = Σ_{ia_j, j≠skip} ψ_a[ia_0,...,d] × Π_{j≠skip} mu_a_j[ia_j]
                std::vector<float> f_a(ca_skip * Dp, 0.f);
                {
                    const float* pa = psi_a[src].data();
                    // Iterate over all flat indices of ψ_a
                    int total_a = vol_a * ca_skip * Dp;
                    // Strides for psi_a
                    std::vector<int> strides_a(nb+1);
                    strides_a[nb] = 1;
                    for (int k = nb-1; k >= 0; k--) strides_a[k] = strides_a[k+1]*sa_sh[k];
                    int total_flat = 1; for (auto s : sa_sh) total_flat *= s;
                    for (int flat = 0; flat < total_flat; flat++) {
                        float w = 1.f;
                        int tmp = flat, ia_skip_val = 0, d_val = 0;
                        for (int k = nb; k >= 0; k--) {
                            int idx_k = tmp % sa_sh[k]; tmp /= sa_sh[k];
                            if (k == nb) { d_val = idx_k; }
                            else if (k == skip) { ia_skip_val = idx_k; }
                            else { w *= get_mu_a(db[k], src)[idx_k]; }
                        }
                        f_a[ia_skip_val * Dp + d_val] += pa[flat] * w;
                    }
                }

                // Step 2: contract ψ_b with Π_{j≠skip} mu_b_j → f_b[ib_skip, d]
                std::vector<float> f_b(cb_skip * Dp, 0.f);
                {
                    const float* pb = psi_b[src].data();
                    std::vector<int> strides_b(nb+1);
                    strides_b[nb] = 1;
                    for (int k = nb-1; k >= 0; k--) strides_b[k] = strides_b[k+1]*sb_sh[k];
                    int total_flat = 1; for (auto s : sb_sh) total_flat *= s;
                    for (int flat = 0; flat < total_flat; flat++) {
                        float w = 1.f;
                        int tmp = flat, ib_skip_val = 0, d_val = 0;
                        for (int k = nb; k >= 0; k--) {
                            int idx_k = tmp % sb_sh[k]; tmp /= sb_sh[k];
                            if (k == nb) { d_val = idx_k; }
                            else if (k == skip) { ib_skip_val = idx_k; }
                            else { w *= get_mu_b(db[k], src)[idx_k]; }
                        }
                        f_b[ib_skip_val * Dp + d_val] += pb[flat] * w;
                    }
                }

                // Step 3: compute new mu_a and mu_b from f_a, f_b
                // nm_a[ia] = Σ_{ib,d} f_a[ia,d] × f_b[ib,d] × mu_b_skip[ib]
                // nm_b[ib] = Σ_{ia,d} f_a[ia,d] × f_b[ib,d] × mu_a_skip[ia]
                const auto& mu_b_in = get_mu_b(bond.id, src);
                const auto& mu_a_in = get_mu_a(bond.id, src);

                std::vector<float> nm_a(ca_skip, 0.f), nm_b(cb_skip, 0.f);
                // f_dot[d] = Σ_ib f_b[ib,d] × mu_b_in[ib]
                std::vector<float> f_dot(Dp, 0.f);
                for (int ib = 0; ib < cb_skip; ib++)
                    for (int d = 0; d < Dp; d++)
                        f_dot[d] += f_b[ib*Dp+d] * mu_b_in[ib];
                for (int ia = 0; ia < ca_skip; ia++)
                    for (int d = 0; d < Dp; d++)
                        nm_a[ia] += f_a[ia*Dp+d] * f_dot[d];

                // g_dot[d] = Σ_ia f_a[ia,d] × mu_a_in[ia]
                std::vector<float> g_dot(Dp, 0.f);
                for (int ia = 0; ia < ca_skip; ia++)
                    for (int d = 0; d < Dp; d++)
                        g_dot[d] += f_a[ia*Dp+d] * mu_a_in[ia];
                for (int ib = 0; ib < cb_skip; ib++)
                    for (int d = 0; d < Dp; d++)
                        nm_b[ib] += f_b[ib*Dp+d] * g_dot[d];

                // Normalize
                float sa2 = 0.f; for (auto v : nm_a) sa2 += v;
                float sb2 = 0.f; for (auto v : nm_b) sb2 += v;
                if (std::abs(sa2) > 1e-20f) for (auto& v : nm_a) v /= sa2;
                if (std::abs(sb2) > 1e-20f) for (auto& v : nm_b) v /= sb2;

                // Track convergence
                auto& oa = get_out_a(bond.id, src);
                auto& ob = get_out_b(bond.id, src);
                for (int i = 0; i < ca_skip; i++) { avg_delta += std::abs(nm_a[i]-oa[i]); n_msgs++; }
                for (int i = 0; i < cb_skip; i++) { avg_delta += std::abs(nm_b[i]-ob[i]); n_msgs++; }
                oa = nm_a; ob = nm_b;
            }
        }
        if (n_msgs > 0 && avg_delta/n_msgs < tol) break;
    }

    // Bethe PF with factored messages
    // vertex scalar: a_v = Σ_{ia...,ib...,d} ψ_a[ia...,d] × ψ_b[ib...,d]
    //                      × Π_k mu_a_k[ia_k] × mu_b_k[ib_k]
    // = Σ_d (Σ_{ia...} ψ_a[ia...,d] Π_k mu_a_k[ia_k]) × (Σ_{ib...} ψ_b[ib...,d] Π_k mu_b_k[ib_k])
    double log_num = 0.0, sign_num = 1.0;
    for (auto& site : grid_a.sites) {
        const auto& sa_sh = sha[site.id];
        const auto& sb_sh = shb[site.id];
        int nb = (int)sa_sh.size() - 1, Dp = sa_sh.back();
        const auto& db = bonds_of[site.id];

        // fa[d] = Σ_{ia...} ψ_a[ia...,d] × Π_k mu_a_k[ia_k]
        std::vector<double> fa(Dp, 0.0), fb(Dp, 0.0);
        {
            int total = 1; for (auto s : sa_sh) total *= s;
            const float* pa = psi_a[site.id].data();
            for (int flat = 0; flat < total; flat++) {
                double w = 1.0; int tmp = flat, d_val = 0;
                for (int k = nb; k >= 0; k--) {
                    int idx_k = tmp % sa_sh[k]; tmp /= sa_sh[k];
                    if (k == nb) d_val = idx_k;
                    else w *= get_mu_a(db[k], site.id)[idx_k];
                }
                fa[d_val] += pa[flat] * w;
            }
        }
        {
            int total = 1; for (auto s : sb_sh) total *= s;
            const float* pb = psi_b[site.id].data();
            for (int flat = 0; flat < total; flat++) {
                double w = 1.0; int tmp = flat, d_val = 0;
                for (int k = nb; k >= 0; k--) {
                    int idx_k = tmp % sb_sh[k]; tmp /= sb_sh[k];
                    if (k == nb) d_val = idx_k;
                    else w *= get_mu_b(db[k], site.id)[idx_k];
                }
                fb[d_val] += pb[flat] * w;
            }
        }
        double a_v = 0.0;
        for (int d = 0; d < Dp; d++) a_v += fa[d] * fb[d];
        if (a_v == 0.0) { sign_num = 0.0; break; }
        sign_num *= (a_v > 0.0 ? 1.0 : -1.0);
        log_num  += std::log(std::abs(a_v));
    }

    // Edge scalar: ze = (Σ_ia mu_a_fwd[ia]) × (Σ_ib mu_b_fwd[ib])
    //                 × (Σ_ia mu_a_bwd[ia]) × (Σ_ib mu_b_bwd[ib])
    // Since messages are normalized to sum=1 after normalization,
    // ze = dot(mu_a_fwd, mu_a_bwd) × dot(mu_b_fwd, mu_b_bwd) approximately.
    // More precisely: Ze = Σ_{ia,ib} M_fwd[ia,ib] × M_bwd[ia,ib]
    //                    = Σ_{ia,ib} mu_a_fwd[ia]×mu_b_fwd[ib] × mu_a_bwd[ia]×mu_b_bwd[ib]
    //                    = (Σ_ia mu_a_fwd[ia]×mu_a_bwd[ia]) × (Σ_ib mu_b_fwd[ib]×mu_b_bwd[ib])
    double log_den = 0.0, sign_den = 1.0;
    for (auto& bond : grid_a.bonds) {
        const auto& ma_f = mu_a[bond.id][0]; const auto& ma_b = mu_a[bond.id][1];
        const auto& mb_f = mu_b[bond.id][0]; const auto& mb_b = mu_b[bond.id][1];
        double za = 0.0; for (int i = 0; i < (int)ma_f.size(); i++) za += ma_f[i]*ma_b[i];
        double zb = 0.0; for (int i = 0; i < (int)mb_f.size(); i++) zb += mb_f[i]*mb_b[i];
        double ze = za * zb;
        if (std::abs(ze) > 1e-30) {
            sign_den *= (ze > 0.0 ? 1.0 : -1.0);
            log_den  += std::log(std::abs(ze));
        }
    }

    if (sign_num == 0.0) return 0.f;
    double result = sign_num * sign_den *
                    std::exp(log_num - log_den + log_scale_a + log_scale_b);
    if (verbose)
        printf("  [BLF_F] sign_num=%+.0f sign_den=%+.0f => %.6f\n",
               sign_num, sign_den, (float)result);
    return (float)result;
}
