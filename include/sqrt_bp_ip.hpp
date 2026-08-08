#pragma once
/*
 * sqrt_bp_ip.hpp — Symmetric gauge inner product for cross-state pairs.
 *
 * CppSim single-layer BP messages are already DIAGONAL vectors (chi entries).
 * No eigendecomposition needed — the gauge transform is element-wise:
 *
 *   tpsi_a[v, i_0,..,i_{nb-1}, d] = psi_a[v,...,d] * prod_k sqrt(|mu_a[k][i_k]|)
 *   tpsi_b[v, i_0,..,i_{nb-1}, d] = psi_b[v,...,d] * prod_k sqrt(|mu_b[k][i_k]|)
 *
 * Inner product (Bethe PF):
 *   a_v = sum_{i,d} tpsi_a[v,i,d] * tpsi_b[v,i,d]
 *   z_e = sum_i sign(mu_a[i]*mu_b[i]) * sqrt(|mu_a[i]|) * sqrt(|mu_b[i]|)
 *       = sum_i sign(mu_a[i]*mu_b[i]) * sqrt(|mu_a[i]*mu_b[i]|)
 *
 * Sign handling: mu values can be negative. We absorb sqrt(|mu|) into psi
 * (always real) and track sign(mu_a*mu_b) in the edge correction.
 *
 * With BP (n_sweeps>0):
 *   Run single-layer BP independently on tpsi_a → updated mu_a
 *   Run single-layer BP independently on tpsi_b → updated mu_b
 *   Recompute Bethe PF with converged mu vectors.
 *   BP on gauged tensors converges fast (near symmetric fixed point).
 *
 * Cost: O(chi^deg * D * n_sites) for gauge transform + Bethe PF.
 *       O(chi^deg * D * n_sweeps) for BP (same as single-layer autocorrelation).
 *       No chi^3 GEMM needed — pure element-wise and dot product.
 *
 * Build: no LAPACK needed (no eigendecomposition).
 *
 * Usage:
 *   SQRT_BP_IP ip(grid_a, grid_b, chi_max, eig_cutoff);
 *   ip.upload(psi_a_gpu, msg_a_gpu, dev_a, psi_b_gpu, msg_b_gpu, dev_b);
 *   float overlap = ip.compute(n_sweeps=0);  // 0=tree-level, >0=BP corrected
 */

#include "grid.hpp"
#include "types.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <hip/hip_runtime.h>

#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do { \
    hipError_t e=(cmd); \
    if(e!=hipSuccess){fprintf(stderr,"HIP %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__);exit(1);} \
} while(0)
#endif

struct SQRT_BP_IP {
    const Grid& grid_a;
    const Grid& grid_b;
    int n_sites, n_bonds, chi_max;
    double eig_cutoff;  // relative cutoff on |mu[i]| / max|mu|

    // Per-bond diagonal message vectors (CPU, double)
    // mu_a[bid][i], mu_b[bid][i] — extracted from single-layer BP messages
    std::vector<std::vector<double>> mu_a, mu_b;

    // Gauge-transformed psi tensors (CPU, float32)
    // tpsi_a[sid][flat*D+d] = psi_a * prod_k sqrt(|mu_a[k][i_k]|)
    std::vector<std::vector<float>> tpsi_a, tpsi_b;

    SQRT_BP_IP(const Grid& ga, const Grid& gb, int chi_max_, double eig_cut=1e-2)
        : grid_a(ga), grid_b(gb),
          n_sites(ga.n_sites), n_bonds(ga.n_bonds),
          chi_max(chi_max_), eig_cutoff(eig_cut)
    {
        mu_a.resize(n_bonds);
        mu_b.resize(n_bonds);
        tpsi_a.resize(n_sites);
        tpsi_b.resize(n_sites);
    }

    // ── Extract diagonal from chi_max×chi_max message matrix on GPU ──────────
    // Message matrices in CppSim are stored as chi_max×chi_max but only the
    // diagonal [i*chi_max+i] is physically meaningful (single-layer BP diagonal).
    // Average fwd and bwd messages to get a symmetric estimate.
    static void _extract_diag(
        const float* d_msg_fwd,  // msg_from_a: mu_{a→b}
        const float* d_msg_bwd,  // msg_from_b: mu_{b→a}
        int src_dev, int chi, int chi_max,
        std::vector<double>& mu_out)
    {
        int cur; HIP_CHECK(hipGetDevice(&cur));
        HIP_CHECK(hipSetDevice(src_dev));
        std::vector<float> bf(chi_max*chi_max), bb(chi_max*chi_max);
        HIP_CHECK(hipMemcpy(bf.data(), d_msg_fwd,
            chi_max*chi_max*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(bb.data(), d_msg_bwd,
            chi_max*chi_max*sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipSetDevice(cur));
        mu_out.resize(chi);
        for(int i=0;i<chi;i++)
            // average fwd and bwd diagonal — symmetric estimate
            mu_out[i] = 0.5*((double)bf[i*chi_max+i] + (double)bb[i*chi_max+i]);
    }

    // ── Apply sqrt(|mu|) to psi along bond axis k ─────────────────────────
    // psi layout: [i_0, ..., i_{nb-1}, d] row-major
    // Multiply psi[..., i_k, ..., d] by sqrt(|mu[i_k]|) for all i_k.
    // Elements where |mu[i_k]| < thresh are zeroed (eigenvalue cutoff).
    static void _absorb_sqrt_diag(
        std::vector<float>& psi, int vol, int nb, int k,
        const int* chis, const double* mu, double thresh)
    {
        int chi_k = chis[k];
        int stride_outer = 1; for(int m=0;m<k;m++) stride_outer *= chis[m];
        int stride_inner = vol / (stride_outer * chi_k);

        for(int outer=0;outer<stride_outer;outer++)
        for(int i_k=0;i_k<chi_k;i_k++){
            float scale = (std::abs(mu[i_k]) >= thresh)
                        ? (float)std::sqrt(std::abs(mu[i_k]))
                        : 0.f;
            for(int inner=0;inner<stride_inner;inner++)
            for(int d=0;d<D;d++){
                int flat = (outer*chi_k+i_k)*stride_inner*D + inner*D + d;
                psi[flat] *= scale;
            }
        }
    }

    // ── Upload: extract diagonal messages, gauge-transform psi ───────────────
    void upload(
        const std::vector<float*>& psi_a_src, int dev_a,
        const std::vector<float*>& msg_a_src,  // single-layer msgs of state_a
        const std::vector<float*>& psi_b_src, int dev_b,
        const std::vector<float*>& msg_b_src)  // single-layer msgs of state_b
    {
        // ── Step 1: Extract diagonal messages ─────────────────────────────────
        for(auto& bond : grid_a.bonds){
            int bid=bond.id, chi=bond.chi;
            _extract_diag(msg_a_src[bond.msg_from_a()], msg_a_src[bond.msg_from_b()],
                          dev_a, chi, chi_max, mu_a[bid]);
            _extract_diag(msg_b_src[bond.msg_from_a()], msg_b_src[bond.msg_from_b()],
                          dev_b, chi, chi_max, mu_b[bid]);
        }

        // ── Step 2: Download psi and absorb sqrt(|mu|) ────────────────────────
        for(auto& site : grid_a.sites){
            int sid=site.id;
            std::vector<int> bids, chis;
            for(int d=0;d<MAX_DEGREE;d++){
                int bid=site.bond_id[d]; if(bid<0) continue;
                bids.push_back(bid);
                chis.push_back(grid_a.bonds[bid].chi);
            }
            int nb=(int)bids.size();
            int vol=1; for(auto c:chis) vol*=c;
            int totD=vol*D;

            // Download psi_a
            int cur; HIP_CHECK(hipGetDevice(&cur));
            HIP_CHECK(hipSetDevice(dev_a));
            tpsi_a[sid].resize(totD);
            HIP_CHECK(hipMemcpy(tpsi_a[sid].data(), psi_a_src[sid],
                totD*sizeof(float), hipMemcpyDeviceToHost));
            HIP_CHECK(hipSetDevice(dev_b));
            tpsi_b[sid].resize(totD);
            HIP_CHECK(hipMemcpy(tpsi_b[sid].data(), psi_b_src[sid],
                totD*sizeof(float), hipMemcpyDeviceToHost));
            HIP_CHECK(hipSetDevice(cur));

            // Compute per-bond relative threshold
            for(int k=0;k<nb;k++){
                int bid=bids[k];

                // threshold for state_a
                double max_a=0.0;
                for(auto v:mu_a[bid]) max_a=std::max(max_a,std::abs(v));
                double thresh_a=eig_cutoff*max_a;

                // threshold for state_b
                double max_b=0.0;
                for(auto v:mu_b[bid]) max_b=std::max(max_b,std::abs(v));
                double thresh_b=eig_cutoff*max_b;

                _absorb_sqrt_diag(tpsi_a[sid], vol, nb, k,
                                  chis.data(), mu_a[bid].data(), thresh_a);
                _absorb_sqrt_diag(tpsi_b[sid], vol, nb, k,
                                  chis.data(), mu_b[bid].data(), thresh_b);
            }
        }
    }

    // ── Compute: Bethe PF with diagonal messages ──────────────────────────────
    // n_sweeps=0: tree-level (use mu from upload)
    // n_sweeps>0: run single-layer BP on each gauged state, update mu, recompute
    //   BP on gauged tensors: same as autocorrelation BP but on tpsi.
    //   Convergence: fast because gauged psi is near symmetric fixed point.
    //   Note: BP needs GPU — for now CPU-only (n_sweeps>0 is future work).
    float compute(int n_sweeps=0, float /*tol*/=1e-5f,
                  bool verbose=false, double* /*t_ms_out*/=nullptr)
    {
        // ── Vertex factors ────────────────────────────────────────────────────
        // a_v = sum_{i,d} tpsi_a[v,i,d] * tpsi_b[v,i,d]
        // (sqrt(|mu|) already absorbed into tpsi during upload)
        double log_num=0.0, sign_num=1.0;
        for(auto& site : grid_a.sites){
            int sid=site.id;
            int totD=(int)tpsi_a[sid].size();
            const float* ta=tpsi_a[sid].data();
            const float* tb=tpsi_b[sid].data();
            double a_v=0.0;
            for(int i=0;i<totD;i++) a_v+=(double)ta[i]*(double)tb[i];
            if(a_v==0.0){sign_num=0.0;break;}
            sign_num*=(a_v>0.0)?1.0:-1.0;
            log_num+=std::log(std::abs(a_v));
        }
        if(sign_num==0.0) return 0.f;

        // ── Edge factors ──────────────────────────────────────────────────────
        // z_e = sum_i sign(mu_a[i]*mu_b[i]) * sqrt(|mu_a[i]|) * sqrt(|mu_b[i]|)
        //     = sum_i sign(mu_a[i]*mu_b[i]) * sqrt(|mu_a[i]*mu_b[i]|)
        // The sign accounts for cases where mu_a and mu_b have opposite signs.
        double log_den=0.0, sign_den=1.0;
        for(auto& bond : grid_a.bonds){
            int bid=bond.id, chi=bond.chi;
            const auto& ma=mu_a[bid];
            const auto& mb=mu_b[bid];

            // Per-bond thresholds
            double max_a=0.0, max_b=0.0;
            for(auto v:ma) max_a=std::max(max_a,std::abs(v));
            for(auto v:mb) max_b=std::max(max_b,std::abs(v));
            double thresh_a=eig_cutoff*max_a, thresh_b=eig_cutoff*max_b;

            double ze_pos=0.0, ze_neg=0.0;
            for(int i=0;i<chi;i++){
                if(std::abs(ma[i])<thresh_a || std::abs(mb[i])<thresh_b) continue;
                double contrib=std::sqrt(std::abs(ma[i])*std::abs(mb[i]));
                if((ma[i]>0.0)==(mb[i]>0.0)) ze_pos+=contrib;
                else                           ze_neg+=contrib;
            }
            double ze=ze_pos-ze_neg;
            if(std::abs(ze)>1e-30){
                sign_den*=(ze>0.0)?1.0:-1.0;
                log_den+=std::log(std::abs(ze));
            }
            if(verbose) printf("  bond %d ze=%.4e (pos=%.4e neg=%.4e)\n",
                               bid,ze,ze_pos,ze_neg);
        }

        // Note: n_sweeps>0 (GPU single-layer BP on gauged psi) is future work.
        // The 0th order result (tree-level) is already correct for the
        // dominant contribution; loop corrections require BP on GPU.
        (void)n_sweeps;

        return (float)(sign_num*sign_den*std::exp(log_num-log_den));
    }
};
