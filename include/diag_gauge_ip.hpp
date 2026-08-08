#pragma once
/*
 * diag_gauge_ip.hpp — Diagonal gauge inner product via QR→diag gauge change.
 *
 * Algorithm:
 *   1. For each bond e, download message M_e (chi×chi, float32 GPU)
 *   2. Eigendecompose in float64: M_e = A_e D_e A_e^T  (DSYEV)
 *   3. For each site v, absorb A_e^T into psi_a and psi_b:
 *        tpsi[..., i_k, ..., d] += A_e[j, i_k] * psi[..., j, ..., d]
 *      This rotates each bond index into the eigenbasis of its message.
 *   4. Inner product (no BP sweeps):
 *        a_v = sum_{i,d} tpsi_a[v,i,d] * D[i] * tpsi_b[v,i,d]
 *        z_e = sum_i D_a[i] * D_b[i]
 *        IP  = prod_v a_v / prod_e z_e
 *
 * Cost:
 *   Eigendecompose: O(chi^3 * n_bonds)  CPU, float64, once per cycle
 *   Gauge transform: O(chi^2 * D * n_sites)  CPU (GPU later)
 *   Inner product:   O(chi^deg * D * n_sites) CPU (GPU later)
 *
 * Note: eigendecomposition uses LAPACK dsyev via the system LAPACK.
 * Link with -llapack -lblas (or -lopenblas).
 *
 * CPU-only for now — GPU port straightforward:
 *   Eigendecompose stays CPU (microseconds per bond)
 *   Gauge transform: one DGEMM per bond per site → rocBLAS
 *   Inner product: one sdot per site → rocBLAS
 */

#include "grid.hpp"
#include "types.hpp"
#include <vector>
#include <cmath>
#include <cstring>
#include <cassert>
#include <hip/hip_runtime.h>

#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do { \
    hipError_t e=(cmd); \
    if(e!=hipSuccess){fprintf(stderr,"HIP %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__);exit(1);} \
} while(0)
#endif

// LAPACK dsyev declaration
extern "C" {
void dsyev_(const char* jobz, const char* uplo, int* n,
            double* A, int* lda, double* w,
            double* work, int* lwork, int* info);
}

// ── DIAG_GAUGE_IP ─────────────────────────────────────────────────────────────
struct DIAG_GAUGE_IP {
    const Grid& grid;
    int n_sites, n_bonds, chi_max;

    // Per-bond: eigenvectors A_e [chi×chi, col-major] and eigenvalues D_e [chi]
    // Stored for state_a and state_b separately (each has its own messages)
    struct BondGauge {
        std::vector<double> A;   // chi×chi eigenvector matrix (col-major)
        std::vector<double> D;   // chi eigenvalues
        int chi = 0;
    };
    std::vector<BondGauge> gauge_a, gauge_b;

    // Per-site: gauge-transformed psi tensors (float64, CPU)
    // tpsi_a[site_id] = psi_a after absorbing A^T from all incident bonds
    std::vector<std::vector<double>> tpsi_a, tpsi_b;

    double eig_cutoff = 1e-4;  // truncate eigenvalues below this threshold

    explicit DIAG_GAUGE_IP(const Grid& g, int chi_max_, double eig_cut = 1e-4)
        : grid(g), n_sites(g.n_sites), n_bonds(g.n_bonds), chi_max(chi_max_),
          eig_cutoff(eig_cut)
    {
        gauge_a.resize(n_bonds);
        gauge_b.resize(n_bonds);
        tpsi_a.resize(n_sites);
        tpsi_b.resize(n_sites);
    }

    // Eigendecompose one message matrix (chi×chi float32 on GPU → float64 CPU)
    // M symmetric positive semi-definite (BP messages).
    // Returns A (eigenvectors, col-major), D (eigenvalues).
    void _eigen(const float* d_msg, int chi, std::vector<double>& A, std::vector<double>& D) {
        // Download
        std::vector<float> buf(chi_max * chi_max);
        HIP_CHECK(hipMemcpy(buf.data(), d_msg,
            chi_max*chi_max*sizeof(float), hipMemcpyDeviceToHost));

        // Extract chi×chi submatrix, upcast to double, symmetrize
        A.assign(chi * chi, 0.0);
        for (int i = 0; i < chi; i++)
            for (int j = 0; j < chi; j++)
                A[j*chi + i] = 0.5 * ((double)buf[i*chi_max + j] +
                                       (double)buf[j*chi_max + i]);  // col-major: A[i,j] = A[j*chi+i]

        D.resize(chi);

        // DSYEV: compute all eigenvalues and eigenvectors
        int n = chi, lda = chi, info = 0, lwork = -1;
        double wkopt;
        dsyev_("V", "U", &n, A.data(), &lda, D.data(), &wkopt, &lwork, &info);
        lwork = (int)wkopt;
        std::vector<double> work(lwork);
        dsyev_("V", "U", &n, A.data(), &lda, D.data(), work.data(), &lwork, &info);
        if (info != 0) {
            fprintf(stderr, "[diag_gauge] dsyev failed info=%d chi=%d\n", info, chi);
            // Fallback: identity
            A.assign(chi*chi, 0.0);
            for (int i = 0; i < chi; i++) { A[i*chi+i] = 1.0; D[i] = 1.0/chi; }
        }
    }

    // Absorb sqrt(|D|) * A^T into psi for all bonds of a site.
    // Eigenvalues below eig_cutoff are truncated — their eigenvector directions
    // are zeroed out, projecting the message into its significant subspace.
    // For bond k: tpsi[..., j_k, ..., d] = sqrt(|D_k[j_k]|) * sum_i A_k[i,j_k] * psi[..., i_k, ..., d]
    //             (skipped when |D_k[j_k]| < eig_cutoff)
    static void _absorb_sqrt_gauge(
        const double* psi_in, int vol, int deg,
        const int* chis,
        const double* const* As,
        const double* const* Ds,
        double eig_cutoff,
        double* psi_out)
    {
        int totD = vol * D;
        std::vector<double> buf1(totD), buf2(totD);
        memcpy(buf1.data(), psi_in, totD * sizeof(double));

        int stride_outer = 1;
        int vol_remain   = vol;
        for (int k = 0; k < deg; k++) {
            int chi_k        = chis[k];
            vol_remain      /= chi_k;
            int stride_inner = vol_remain;

            const double* Ak = As[k];
            const double* Dk = Ds[k];
            memset(buf2.data(), 0, totD * sizeof(double));

            double max_abs = 0.0;
            for (int j = 0; j < chi_k; j++) max_abs = std::max(max_abs, std::abs(Dk[j]));
            double thresh = eig_cutoff * max_abs;

            for (int outer = 0; outer < stride_outer; outer++)
            for (int i_k   = 0; i_k   < chi_k; i_k++)
            for (int inner = 0; inner < stride_inner; inner++)
            for (int j_k   = 0; j_k   < chi_k; j_k++) {
                if (std::abs(Dk[j_k]) < thresh) continue;  // relative cutoff
                double w = std::sqrt(std::abs(Dk[j_k])) * Ak[i_k + j_k*chi_k];
                int in_flat  = (outer*chi_k + i_k)*stride_inner*D + inner*D;
                int out_flat = (outer*chi_k + j_k)*stride_inner*D + inner*D;
                for (int d = 0; d < D; d++)
                    buf2[out_flat + d] += w * buf1[in_flat + d];
            }
            std::swap(buf1, buf2);
            stride_outer *= chi_k;
        }
        memcpy(psi_out, buf1.data(), totD * sizeof(double));
    }

    // Upload: eigendecompose state_a messages only → apply same rotation to both.
    // This is the Julia approach: use A_a as the shared basis, apply sqrt(|D_a|)*A_a^T
    // to psi_a and A_a^T only (without sqrt) to psi_b.
    //
    // After transform:
    //   tpsi_a[i] = sum_j sqrt(|D_a[i]|) * A_a[j,i] * psi_a[j]
    //   tpsi_b[i] = sum_j              1  * A_a[j,i] * psi_b[j]
    //
    // Inner product:
    //   a_v = sum_{i,d} tpsi_a[v,i,d] * tpsi_b[v,i,d]
    //       = sum_{i,d} sqrt(|D_a[i]|) * (A_a^T psi_a)[i,d] * (A_a^T psi_b)[i,d]
    //
    // Edge correction:
    //   z_e = sum_i |D_a[i]|   (eigenvalues of M_a, normalizer for the transform)
    //
    // This preserves the discriminating structure between states because A_a
    // is state-a specific — it does NOT symmetrize the two states.
    void upload(
        const std::vector<float*>& psi_a_src,
        const std::vector<float*>& msg_a_src,
        const std::vector<float*>& psi_b_src,
        const std::vector<float*>& msg_b_src)
    {
        // ── Eigendecompose state_a messages only ──────────────────────────────
        for (auto& bond : grid.bonds) {
            int bid = bond.id;
            int chi = bond.chi;
            gauge_a[bid].chi = chi;
            gauge_b[bid].chi = chi;
            // Only eigendecompose M_a — use A_a as shared basis
            _eigen(msg_a_src[bond.msg_from_a()], chi, gauge_a[bid].A, gauge_a[bid].D);
            // gauge_b reuses A_a but with unit eigenvalues (no sqrt scaling for psi_b)
            gauge_b[bid].A = gauge_a[bid].A;
            gauge_b[bid].D.assign(chi, 1.0);  // no sqrt scaling for psi_b
        }

        // ── Download psi and apply gauge transform per site ───────────────────
        for (auto& site : grid.sites) {
            int sid = site.id;

            // Collect bond info
            std::vector<int> bids, chis_a, chis_b;
            for (int d = 0; d < MAX_DEGREE; d++) {
                int bid = site.bond_id[d];
                if (bid < 0) continue;
                bids.push_back(bid);
                chis_a.push_back(grid.bonds[bid].chi);
                chis_b.push_back(grid.bonds[bid].chi);
            }
            int nb   = (int)bids.size();
            int vol  = 1; for (auto c : chis_a) vol *= c;
            int totD = vol * D;

            // Download psi_a, upcast to double
            std::vector<float> fbuf(totD);
            HIP_CHECK(hipMemcpy(fbuf.data(), psi_a_src[sid],
                totD*sizeof(float), hipMemcpyDeviceToHost));
            std::vector<double> psi_a_d(totD);
            for (int i = 0; i < totD; i++) psi_a_d[i] = (double)fbuf[i];

            // Download psi_b, upcast to double
            HIP_CHECK(hipMemcpy(fbuf.data(), psi_b_src[sid],
                totD*sizeof(float), hipMemcpyDeviceToHost));
            std::vector<double> psi_b_d(totD);
            for (int i = 0; i < totD; i++) psi_b_d[i] = (double)fbuf[i];

            // Eigenvector matrices and eigenvalues for incoming bonds
            std::vector<const double*> As_a(nb), Ds_a(nb);
            std::vector<const double*> As_b(nb), Ds_b(nb);
            for (int k = 0; k < nb; k++) {
                As_a[k] = gauge_a[bids[k]].A.data();
                Ds_a[k] = gauge_a[bids[k]].D.data();
                As_b[k] = gauge_b[bids[k]].A.data();
                Ds_b[k] = gauge_b[bids[k]].D.data();
            }

            // Apply full SQRT gauge: absorb sqrt(|D|) * A^T into psi
            tpsi_a[sid].resize(totD);
            tpsi_b[sid].resize(totD);
            _absorb_sqrt_gauge(psi_a_d.data(), vol, nb,
                               chis_a.data(), As_a.data(), Ds_a.data(),
                               eig_cutoff, tpsi_a[sid].data());
            _absorb_sqrt_gauge(psi_b_d.data(), vol, nb,
                               chis_b.data(), As_b.data(), Ds_b.data(),
                               eig_cutoff, tpsi_b[sid].data());
        }
    }

    // Compute inner product in SQRT gauge.
    // After absorbing sqrt(|D|)*A^T into both psi_a and psi_b, messages are
    // identity on a tree. Inner product is a pure site dot product.
    // Edge correction: z_e = sum_i sqrt(|D_a[i]|) * sqrt(|D_b[i]|)
    // (accounts for the fact that D_a != D_b for cross-state pairs).
    float compute() {
        // ── Vertex factors: pure dot product ──────────────────────────────────
        double log_num = 0.0, sign_num = 1.0;

        for (auto& site : grid.sites) {
            int sid = site.id;
            int vol = 1;
            for (int d = 0; d < MAX_DEGREE; d++)
                if (site.bond_id[d] >= 0) vol *= grid.bonds[site.bond_id[d]].chi;
            int totD = vol * D;

            const double* ta = tpsi_a[sid].data();
            const double* tb = tpsi_b[sid].data();

            // Pure dot product — sqrt(D) already absorbed
            double a_v = 0.0;
            for (int i = 0; i < totD; i++) a_v += ta[i] * tb[i];

            if (a_v == 0.0) { sign_num = 0.0; break; }
            sign_num *= (a_v > 0.0) ? 1.0 : -1.0;
            log_num  += std::log(std::abs(a_v));
        }

        // ── Edge correction ────────────────────────────────────────────────────
        // We applied sqrt(|D_a|)*A_a^T to psi_a and A_a^T to psi_b.
        // The normalizer is the trace of M_a in the eigenbasis = sum_i |D_a[i]|.
        // Sign of D_a[i] is tracked separately.
        double log_den = 0.0, sign_den = 1.0;
        for (auto& bond : grid.bonds) {
            int bid = bond.id;
            int chi = bond.chi;
            double max_a = 0.0;
            for (int i = 0; i < chi; i++)
                max_a = std::max(max_a, std::abs(gauge_a[bid].D[i]));
            double thresh = eig_cutoff * max_a;
            double ze_pos = 0.0, ze_neg = 0.0;
            for (int i = 0; i < chi; i++) {
                double da = gauge_a[bid].D[i];
                if (std::abs(da) < thresh) continue;
                if (da > 0.0) ze_pos += da;
                else          ze_neg -= da;  // add |da|
            }
            double ze = ze_pos - ze_neg;
            if (std::abs(ze) > 1e-30) {
                sign_den *= (ze > 0.0) ? 1.0 : -1.0;
                log_den  += std::log(std::abs(ze));
            }
        }

        if (sign_num == 0.0) return 0.f;
        return (float)(sign_num * sign_den * std::exp(log_num - log_den));
    }
};
