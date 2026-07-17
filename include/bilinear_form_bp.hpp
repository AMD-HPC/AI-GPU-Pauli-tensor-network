#pragma once
/*
 * bilinear_form_bp.hpp — Two-state overlap ⟨ψ_a|ψ_b⟩ via BilinearForm BP
 *
 * Pure CPU. Matches Julia's inner(ψ_a, ψ_b; alg="bp") exactly.
 *
 * Algorithm:
 *   1. Build T_v[ia0,ib0,...] = Σ_d ψa[ia...,d] × ψb[ib...,d]
 *      (double-layer site tensor, bond dim = chi_a × chi_b per bond)
 *   2. Initialize messages M_e[ia,ib] = identity / sum (Julia: delta(linds))
 *   3. BP sweeps: M_{v→e}[ia,ib] ∝ Σ_{other} T_v × Π_{e'≠e} M_{e'→v}
 *      Normalize by sum (Julia convention)
 *   4. Bethe PF: C = Π_v a_v / Π_e ze_e
 *      a_v = Σ T_v × Π_e M_{e→v}
 *      ze_e = Σ_{ia,ib} M_fwd[ia,ib] × M_bwd[ia,ib]
 *
 * Both states must be on the SAME grid (same bond structure).
 * CPU only — downloads psi from GPU once.
 */

#include "grid.hpp"
#include "types.hpp"
#include <vector>
#include <cmath>
#include <hip/hip_runtime.h>

inline float bilinear_form_bp(
    const Grid&                grid,
    const std::vector<float*>& psi_a_gpu,  // state a site tensors
    const std::vector<float*>& psi_b_gpu,  // state b site tensors
    int                        n_sweeps = 25,
    float                      tol = 1e-5f)
{
    // ── 1. Download ψ_a and ψ_b ─────────────────────────────────────
    std::vector<std::vector<float>> psi_a(grid.n_sites), psi_b(grid.n_sites);
    std::vector<std::vector<int>>   psi_shape(grid.n_sites);

    for (auto& site : grid.sites) {
        std::vector<int> sh;
        for (int d = 0; d < MAX_DEGREE; d++)
            if (site.bond_id[d] >= 0)
                sh.push_back(grid.bonds[site.bond_id[d]].chi);
        sh.push_back(D);
        psi_shape[site.id] = sh;
        int tot = 1; for (auto s : sh) tot *= s;
        psi_a[site.id].resize(tot);
        psi_b[site.id].resize(tot);
        HIP_CHECK(hipMemcpy(psi_a[site.id].data(), psi_a_gpu[site.id],
            tot*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(psi_b[site.id].data(), psi_b_gpu[site.id],
            tot*sizeof(float), hipMemcpyDeviceToHost));
    }

    // Dense bond list per site (tensor order)
    std::vector<std::vector<int>> bonds_of(grid.n_sites);
    for (auto& site : grid.sites)
        for (int d = 0; d < MAX_DEGREE; d++)
            if (site.bond_id[d] >= 0)
                bonds_of[site.id].push_back(site.bond_id[d]);

    // ── 2. Build T_v[ia0,ib0,...] = Σ_d ψa[ia...,d] × ψb[ib...,d] ──
    // T_v has interleaved pairs: axis 2k = ia_k (from ψa), axis 2k+1 = ib_k (from ψb)
    // Both ψa and ψb have SAME bond structure (same grid), so chi_a = chi_b = chi
    std::vector<std::vector<float>> T(grid.n_sites);
    std::vector<std::vector<int>>   T_shape(grid.n_sites);

    for (auto& site : grid.sites) {
        const auto& sh = psi_shape[site.id];
        int n_bonds = (int)sh.size() - 1;
        int D_phys = sh.back();
        int bond_vol_a = 1; for (int k = 0; k < n_bonds; k++) bond_vol_a *= sh[k];
        int bond_vol_b = bond_vol_a;  // same grid

        // T[ia_flat, ib_flat] = Σ_d ψa[ia_flat,d] × ψb[ib_flat,d]
        // = outer product over bonds, contracted over d
        const float* pa = psi_a[site.id].data();
        const float* pb = psi_b[site.id].data();

        std::vector<float> T_big(bond_vol_a * bond_vol_b, 0.f);
        for (int ia = 0; ia < bond_vol_a; ia++)
            for (int ib = 0; ib < bond_vol_b; ib++)
                for (int d = 0; d < D_phys; d++)
                    T_big[ia * bond_vol_b + ib] += pa[ia*D_phys+d] * pb[ib*D_phys+d];

        // Reshape to (chi0,chi0, chi1,chi1, ...) — interleaved pairs
        // Then we can use the same contraction logic as inner_op_state
        // For now store as (chi0*chi0, chi1*chi1,...) = flattened pairs
        // Shape: (chi_0^2, chi_1^2, ...) per bond pair
        std::vector<int> T_sh;
        std::vector<int> chi_vals;
        for (int k = 0; k < n_bonds; k++) {
            int chi = sh[k];
            T_sh.push_back(chi * chi);  // ia*chi + ib flattened
            chi_vals.push_back(chi);
        }

        // Rearrange T_big from (ia_flat, ib_flat) to interleaved flat pairs
        // ia_flat = ia0*chi1*chi2*... + ia1*chi2*... + ...
        // T[ia0,ia1,...,ib0,ib1,...] → T[ia0*chi+ib0, ia1*chi+ib1, ...]
        int total_T = 1; for (auto s : T_sh) total_T *= s;
        std::vector<float> T_interleaved(total_T, 0.f);

        for (int ia_flat = 0; ia_flat < bond_vol_a; ia_flat++) {
            for (int ib_flat = 0; ib_flat < bond_vol_b; ib_flat++) {
                float val = T_big[ia_flat * bond_vol_b + ib_flat];

                // Decode ia_flat to (ia0, ia1, ...)
                std::vector<int> ia(n_bonds), ib(n_bonds);
                int tmp_a = ia_flat, tmp_b = ib_flat;
                for (int k = n_bonds-1; k >= 0; k--) {
                    int chi = chi_vals[k];
                    ia[k] = tmp_a % chi; tmp_a /= chi;
                    ib[k] = tmp_b % chi; tmp_b /= chi;
                }

                // Compute flat index in T_interleaved
                int idx = 0;
                for (int k = 0; k < n_bonds; k++) {
                    int chi = chi_vals[k];
                    idx = idx * (chi*chi) + ia[k]*chi + ib[k];
                }
                T_interleaved[idx] = val;
            }
        }

        T[site.id] = T_interleaved;
        T_shape[site.id] = T_sh;
    }

    // ── 3. Initialize BilinearForm messages ───────────────────────────
    // M_fwd[bond_id] = μ_{a→b}[ia,ib] — chi×chi matrix (stored flat, col-major)
    // M_bwd[bond_id] = μ_{b→a}[ia,ib]
    // Init: identity / sum = (1/chi) × I
    std::vector<std::vector<float>> M_fwd(grid.n_bonds), M_bwd(grid.n_bonds);
    for (auto& bond : grid.bonds) {
        int chi = bond.chi; int chi2 = chi*chi;
        M_fwd[bond.id].assign(chi2, 0.f);
        M_bwd[bond.id].assign(chi2, 0.f);
        float init_val = 1.f / chi;
        for (int i = 0; i < chi; i++) {
            M_fwd[bond.id][i*chi+i] = init_val;
            M_bwd[bond.id][i*chi+i] = init_val;
        }
    }

    auto get_inc = [&](int bid, int sid) -> const float* {
        const Bond& b = grid.bonds[bid];
        return (sid == b.site_a) ? M_bwd[bid].data() : M_fwd[bid].data();
    };
    auto get_out = [&](int bid, int sid) -> std::vector<float>& {
        const Bond& b = grid.bonds[bid];
        return (sid == b.site_a) ? M_fwd[bid] : M_bwd[bid];
    };

    // ── 4. BP sweeps ──────────────────────────────────────────────────
    for (int sweep = 0; sweep < n_sweeps; sweep++) {
        float avg_delta = 0.f; int n_msgs = 0;

        for (auto& bond : grid.bonds) {
            for (int dir = 0; dir < 2; dir++) {
                int src = (dir==0) ? bond.site_a : bond.site_b;
                const auto& sh = T_shape[src];
                int ndim = (int)sh.size();
                const auto& db = bonds_of[src];

                // Find skip index
                int skip = -1;
                for (int k = 0; k < (int)db.size(); k++)
                    if (db[k] == bond.id) { skip = k; break; }
                if (skip < 0) continue;

                int chi_out = sh[skip];  // chi^2 for this bond pair
                int total = 1; for (auto s : sh) total *= s;

                // Compute new message: contract T_v with all incoming except skip
                std::vector<float> nm(chi_out, 0.f);
                for (int flat = 0; flat < total; flat++) {
                    // Decode multi-index
                    std::vector<int> idx(ndim);
                    int tmp = flat;
                    for (int d = ndim-1; d >= 0; d--) {
                        idx[d] = tmp % sh[d];
                        tmp /= sh[d];
                    }
                    float prod = T[src][flat];
                    for (int k = 0; k < ndim; k++) {
                        if (k == skip) continue;
                        int bid_k = db[k];
                        const float* M = get_inc(bid_k, src);
                        int chi_k = (int)std::sqrt((float)sh[k]);  // chi^2 → chi
                        // M is chi×chi matrix, idx[k] = ia*chi + ib
                        prod *= M[idx[k]];  // M[ia*chi+ib]
                    }
                    nm[idx[skip]] += prod;
                }

                // Normalize by sum
                float s = 0.f; for (auto v : nm) s += v;
                if (std::abs(s) > 1e-20f) for (auto& v : nm) v /= s;

                auto& om = get_out(bond.id, src);
                for (int k = 0; k < chi_out; k++) {
                    avg_delta += std::abs(nm[k] - om[k]);
                    n_msgs++;
                }
                om = nm;
            }
        }
        if (n_msgs > 0 && avg_delta/n_msgs < tol) break;
    }

    // ── 5. Bethe partition function ───────────────────────────────────
    double log_num = 0.0, sign_num = 1.0;
    for (auto& site : grid.sites) {
        const auto& sh = T_shape[site.id];
        int ndim = (int)sh.size();
        int total = 1; for (auto s : sh) total *= s;
        const auto& db = bonds_of[site.id];

        double a_v = 0.0;
        for (int flat = 0; flat < total; flat++) {
            std::vector<int> idx(ndim);
            int tmp = flat;
            for (int d = ndim-1; d >= 0; d--) { idx[d]=tmp%sh[d]; tmp/=sh[d]; }
            float prod = T[site.id][flat];
            for (int k = 0; k < ndim; k++)
                prod *= get_inc(db[k], site.id)[idx[k]];
            a_v += (double)prod;
        }
        if (a_v == 0.0) { sign_num = 0.0; break; }
        sign_num *= (a_v > 0.0 ? 1.0 : -1.0);
        log_num  += std::log(std::abs(a_v));
    }

    double log_den = 0.0, sign_den = 1.0;
    for (auto& bond : grid.bonds) {
        int chi = bond.chi, chi2 = chi*chi;
        double ze = 0.0;
        for (int k = 0; k < chi2; k++)
            ze += (double)M_fwd[bond.id][k] * M_bwd[bond.id][k];
        if (std::abs(ze) > 1e-30) {
            sign_den *= (ze > 0 ? 1.0 : -1.0);
            log_den  += std::log(std::abs(ze));
        }
    }

    if (sign_num == 0.0) return 0.f;
    return (float)(sign_num * sign_den * std::exp(log_num - log_den));
}
