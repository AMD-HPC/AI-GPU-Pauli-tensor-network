#pragma once
/*
 * sqrt_gauge_ip.hpp — Asymmetric gauge inner product via GPU dot product.
 *
 * Algorithm (single-layer message absorption):
 *   For each site v with bonds k=0..nb-1:
 *     tilde_psi_a[v, i_0,..,i_{nb-1}, d] =
 *         psi_a[v, i_0,..,i_{nb-1}, d]
 *         * prod_k  msg_incoming_a[k][ i_k ]
 *   a_v = sum_{i,d} tilde_psi_a[v,i,d] * psi_b[v,i,d]
 *
 * Edge correction:
 *   z_e = sum_i msg_a_fwd[e][i] * msg_a_bwd[e][i]   (diagonal dot)
 *   IP_SQ = prod_v a_v / prod_e z_e
 *
 * Messages are stored as chi_max x chi_max matrices on device.
 * Only the diagonal is used: msg[i*chi_max + i].
 *
 * GPU kernel: one block per site, threads over the flat bond volume.
 *   - reads psi_a and psi_b from device
 *   - reads diagonal of each incoming message
 *   - accumulates dot product in shared memory reduction
 *   - writes one float per site to d_site_vals
 *
 * Then Bethe PF is computed on CPU (68 values — negligible cost).
 */

#include "grid.hpp"
#include "types.hpp"
#include <vector>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do { \
    hipError_t e=(cmd); \
    if(e!=hipSuccess){fprintf(stderr,"HIP %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__);exit(1);} \
} while(0)
#endif

// ── GPU kernel ────────────────────────────────────────────────────────────────
// One block per site. Threads stride over flat bond-index × D.
// msg_diag_ptrs[site_id * MAX_DEGREE + k] = pointer to diagonal of incoming
//   message k for site site_id (null if bond k does not exist).
// chi_per_bond[site_id * MAX_DEGREE + k] = chi for that bond.
// vol_per_site[site_id] = product of chi values for that site (bond volume).
// psi_a_ptrs[site_id], psi_b_ptrs[site_id] = device pointers to site tensors.
// d_site_vals[site_id] = output: sum_{i,d} tilde_psi_a * psi_b.

__global__ void sqrt_gauge_site_dot(
    const float* const* __restrict__ psi_a_ptrs,
    const float* const* __restrict__ psi_b_ptrs,
    const float* const* __restrict__ msg_diag_ptrs,  // [n_sites * MAX_DEGREE]
    const int*   __restrict__ chi_per_bond,            // [n_sites * MAX_DEGREE]
    const int*   __restrict__ nb_per_site,             // [n_sites]
    const int*   __restrict__ vol_per_site,            // [n_sites]
    float*       __restrict__ d_site_vals,             // [n_sites]
    int n_sites)
{
    int sid = blockIdx.x;
    if (sid >= n_sites) return;

    int nb   = nb_per_site[sid];
    int vol  = vol_per_site[sid];  // bond volume (product of chi)
    int totD = vol * D;            // total elements including Pauli index

    const float* psi_a = psi_a_ptrs[sid];
    const float* psi_b = psi_b_ptrs[sid];

    // Each thread accumulates its portion of the dot product
    float acc = 0.f;
    for (int flat = threadIdx.x; flat < totD; flat += blockDim.x) {
        int d    = flat % D;
        int bidx = flat / D;   // flat bond index

        // Decode multi-index and compute message product
        float mp  = 1.f;
        int   tmp = bidx;
        // Bond order: last bond varies fastest (row-major psi layout)
        for (int k = nb - 1; k >= 0; k--) {
            int chi_k = chi_per_bond[sid * MAX_DEGREE + k];
            int i_k   = tmp % chi_k;
            tmp       /= chi_k;
            const float* diag = msg_diag_ptrs[sid * MAX_DEGREE + k];
            if (diag) mp *= diag[i_k];
        }
        (void)d;  // d does not affect the message product
        acc += psi_a[flat] * mp * psi_b[flat];
    }

    // Warp reduction
    for (int offset = warpSize/2; offset > 0; offset >>= 1)
        acc += __shfl_down(acc, offset);

    // Block reduction via shared memory
    extern __shared__ float smem[];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    if (lane == 0) smem[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        acc = (threadIdx.x < (blockDim.x / warpSize)) ? smem[lane] : 0.f;
        for (int offset = warpSize/2; offset > 0; offset >>= 1)
            acc += __shfl_down(acc, offset);
        if (threadIdx.x == 0) d_site_vals[sid] = acc;
    }
}

// ── SQRT_GAUGE_IP ─────────────────────────────────────────────────────────────
struct SQRT_GAUGE_IP {
    const Grid& grid;
    int n_sites, n_bonds, chi_max;

    // Local device copies of psi (owned, on current device)
    std::vector<float*> d_psi_a;   // [n_sites]
    std::vector<float*> d_psi_b;   // [n_sites]

    // Device arrays for kernel launch (on current device)
    float**  d_psi_a_ptrs    = nullptr;
    float**  d_psi_b_ptrs    = nullptr;
    float**  d_msg_diag_ptrs = nullptr;  // [n_sites * MAX_DEGREE]
    int*     d_chi_per_bond  = nullptr;  // [n_sites * MAX_DEGREE]
    int*     d_nb_per_site   = nullptr;  // [n_sites]
    int*     d_vol_per_site  = nullptr;  // [n_sites]
    float*   d_site_vals     = nullptr;  // [n_sites]

    // Per-bond diagonal buffers on current device
    std::vector<float*> d_msg_diag_fwd;  // [n_bonds]
    std::vector<float*> d_msg_diag_bwd;  // [n_bonds]

    // Host metadata
    std::vector<int> h_nb, h_vol;
    std::vector<int> h_chi_per_bond;

    explicit SQRT_GAUGE_IP(const Grid& g, int chi_max_)
        : grid(g), n_sites(g.n_sites), n_bonds(g.n_bonds), chi_max(chi_max_)
    {
        d_psi_a.resize(n_sites, nullptr);
        d_psi_b.resize(n_sites, nullptr);
        d_msg_diag_fwd.resize(n_bonds, nullptr);
        d_msg_diag_bwd.resize(n_bonds, nullptr);
        h_nb.resize(n_sites, 0);
        h_vol.resize(n_sites, 1);
        h_chi_per_bond.resize(n_sites * MAX_DEGREE, 1);

        for (int b = 0; b < n_bonds; b++) {
            HIP_CHECK(hipMalloc(&d_msg_diag_fwd[b], chi_max * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_msg_diag_bwd[b], chi_max * sizeof(float)));
        }
        HIP_CHECK(hipMalloc(&d_psi_a_ptrs,    n_sites * sizeof(float*)));
        HIP_CHECK(hipMalloc(&d_psi_b_ptrs,    n_sites * sizeof(float*)));
        HIP_CHECK(hipMalloc(&d_msg_diag_ptrs, n_sites * MAX_DEGREE * sizeof(float*)));
        HIP_CHECK(hipMalloc(&d_chi_per_bond,  n_sites * MAX_DEGREE * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_nb_per_site,   n_sites * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_vol_per_site,  n_sites * sizeof(int)));
        HIP_CHECK(hipMalloc(&d_site_vals,     n_sites * sizeof(float)));
    }

    ~SQRT_GAUGE_IP() {
        for (auto p : d_psi_a)        if (p) HIP_CHECK(hipFree(p));
        for (auto p : d_psi_b)        if (p) HIP_CHECK(hipFree(p));
        for (auto p : d_msg_diag_fwd) if (p) HIP_CHECK(hipFree(p));
        for (auto p : d_msg_diag_bwd) if (p) HIP_CHECK(hipFree(p));
        if (d_psi_a_ptrs)    HIP_CHECK(hipFree(d_psi_a_ptrs));
        if (d_psi_b_ptrs)    HIP_CHECK(hipFree(d_psi_b_ptrs));
        if (d_msg_diag_ptrs) HIP_CHECK(hipFree(d_msg_diag_ptrs));
        if (d_chi_per_bond)  HIP_CHECK(hipFree(d_chi_per_bond));
        if (d_nb_per_site)   HIP_CHECK(hipFree(d_nb_per_site));
        if (d_vol_per_site)  HIP_CHECK(hipFree(d_vol_per_site));
        if (d_site_vals)     HIP_CHECK(hipFree(d_site_vals));
    }

    // Upload: copy psi and messages from source device (GPU 0) to current device.
    // msg_a: single-layer BP messages of state_a — only diagonal is used.
    void upload(
        const std::vector<float*>& psi_a_src,  // GPU 0 psi tensors for state_a
        const std::vector<float*>& msg_a_src,  // GPU 0 messages for state_a
        const std::vector<float*>& psi_b_src)  // GPU 0 psi tensors for state_b
    {
        // ── Copy psi_a and psi_b to local device ──────────────────────────────
        for (auto& site : grid.sites) {
            int sid  = site.id;
            int vol  = D;
            for (int d = 0; d < MAX_DEGREE; d++)
                if (site.bond_id[d] >= 0) vol *= grid.bonds[site.bond_id[d]].chi;

            if (d_psi_a[sid]) HIP_CHECK(hipFree(d_psi_a[sid]));
            if (d_psi_b[sid]) HIP_CHECK(hipFree(d_psi_b[sid]));
            HIP_CHECK(hipMalloc(&d_psi_a[sid], vol * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_psi_b[sid], vol * sizeof(float)));

            // D2H from source, H2D to current device
            std::vector<float> buf(vol);
            HIP_CHECK(hipMemcpy(buf.data(), psi_a_src[sid],
                vol*sizeof(float), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(d_psi_a[sid], buf.data(),
                vol*sizeof(float), hipMemcpyHostToDevice));

            HIP_CHECK(hipMemcpy(buf.data(), psi_b_src[sid],
                vol*sizeof(float), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(d_psi_b[sid], buf.data(),
                vol*sizeof(float), hipMemcpyHostToDevice));
        }

        // ── Extract message diagonals (D2H from GPU 0, H2D to current) ───────
        std::vector<float> mbuf(chi_max * chi_max);
        for (auto& bond : grid.bonds) {
            int bid = bond.id;
            int chi = bond.chi;

            // fwd = msg_from_a
            HIP_CHECK(hipMemcpy(mbuf.data(), msg_a_src[bond.msg_from_a()],
                chi_max*chi_max*sizeof(float), hipMemcpyDeviceToHost));
            std::vector<float> dfwd(chi);
            for (int i = 0; i < chi; i++) dfwd[i] = mbuf[i * chi_max + i];
            HIP_CHECK(hipMemcpy(d_msg_diag_fwd[bid], dfwd.data(),
                chi*sizeof(float), hipMemcpyHostToDevice));

            // bwd = msg_from_b
            HIP_CHECK(hipMemcpy(mbuf.data(), msg_a_src[bond.msg_from_b()],
                chi_max*chi_max*sizeof(float), hipMemcpyDeviceToHost));
            std::vector<float> dbwd(chi);
            for (int i = 0; i < chi; i++) dbwd[i] = mbuf[i * chi_max + i];
            HIP_CHECK(hipMemcpy(d_msg_diag_bwd[bid], dbwd.data(),
                chi*sizeof(float), hipMemcpyHostToDevice));
        }

        // ── Build per-site metadata ───────────────────────────────────────────
        std::vector<float*> h_msg_diag_ptrs(n_sites * MAX_DEGREE, nullptr);
        for (auto& site : grid.sites) {
            int sid   = site.id;
            h_nb[sid] = 0; h_vol[sid] = 1;
            for (int d = 0; d < MAX_DEGREE; d++) {
                int bid = site.bond_id[d];
                if (bid < 0) {
                    h_chi_per_bond[sid * MAX_DEGREE + d] = 1;
                    h_msg_diag_ptrs[sid * MAX_DEGREE + d] = nullptr;
                    continue;
                }
                int chi = grid.bonds[bid].chi;
                int k   = h_nb[sid]++;
                h_chi_per_bond[sid * MAX_DEGREE + k] = chi;
                h_vol[sid] *= chi;
                // incoming to sid: if sid==site_a, incoming is bwd; else fwd
                h_msg_diag_ptrs[sid * MAX_DEGREE + k] =
                    (sid == grid.bonds[bid].site_a)
                    ? d_msg_diag_bwd[bid]
                    : d_msg_diag_fwd[bid];
            }
        }

        // Upload local psi pointer arrays
        HIP_CHECK(hipMemcpy(d_psi_a_ptrs, d_psi_a.data(),
            n_sites*sizeof(float*), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_psi_b_ptrs, d_psi_b.data(),
            n_sites*sizeof(float*), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_msg_diag_ptrs, h_msg_diag_ptrs.data(),
            n_sites * MAX_DEGREE * sizeof(float*), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_chi_per_bond, h_chi_per_bond.data(),
            n_sites * MAX_DEGREE * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_nb_per_site, h_nb.data(),
            n_sites * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vol_per_site, h_vol.data(),
            n_sites * sizeof(int), hipMemcpyHostToDevice));
    }

    // Compute IP_SQ = prod_v a_v / prod_e z_e
    // Diagonals already extracted during upload().
    float compute() {
        // ── Site dot products on GPU ───────────────────────────────────────
        int block = 256;
        int smem  = (block / 32) * sizeof(float);
        sqrt_gauge_site_dot<<<n_sites, block, smem>>>(
            d_psi_a_ptrs, d_psi_b_ptrs,
            (const float* const*)d_msg_diag_ptrs,
            d_chi_per_bond, d_nb_per_site, d_vol_per_site,
            d_site_vals, n_sites);
        HIP_CHECK(hipDeviceSynchronize());

        // Download site values
        std::vector<float> site_vals(n_sites);
        HIP_CHECK(hipMemcpy(site_vals.data(), d_site_vals,
            n_sites*sizeof(float), hipMemcpyDeviceToHost));

        // ── Bethe PF: numerator ───────────────────────────────────────────
        double log_num = 0.0, sign_num = 1.0;
        for (int sid = 0; sid < n_sites; sid++) {
            double a_v = (double)site_vals[sid];
            if (a_v == 0.0) { sign_num = 0.0; break; }
            sign_num *= (a_v > 0.0) ? 1.0 : -1.0;
            log_num  += std::log(std::abs(a_v));
        }

        // ── Bethe PF: edge correction ─────────────────────────────────────
        // Asymmetric gauge: full msg absorbed into psi_a, psi_b ungauged.
        // z_e = sum_i msg_fwd[i] * msg_bwd[i]  (same as single-layer Bethe)
        // This is the correct denominator for the asymmetric absorption:
        // the psi_b side sees identity messages, so the edge factor reduces
        // to the single-layer normalizer of psi_a's messages.
        double log_den = 0.0, sign_den = 1.0;
        for (auto& bond : grid.bonds) {
            int bid = bond.id;
            int chi = bond.chi;
            std::vector<float> df(chi), db(chi);
            HIP_CHECK(hipMemcpy(df.data(), d_msg_diag_fwd[bid],
                chi*sizeof(float), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(db.data(), d_msg_diag_bwd[bid],
                chi*sizeof(float), hipMemcpyDeviceToHost));
            // z_e = fwd . bwd (both sides of psi_a's gauge)
            double ze = 0.0;
            for (int i = 0; i < chi; i++) ze += (double)df[i] * (double)db[i];
            if (std::abs(ze) > 1e-30) {
                sign_den *= (ze > 0.0) ? 1.0 : -1.0;
                log_den  += std::log(std::abs(ze));
            }
        }

        if (sign_num == 0.0) return 0.f;
        return (float)(sign_num * sign_den * std::exp(log_num - log_den));
    }
};
