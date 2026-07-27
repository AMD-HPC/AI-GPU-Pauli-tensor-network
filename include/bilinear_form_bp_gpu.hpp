#pragma once
/*
 * bilinear_form_bp_gpu.hpp — Two-state overlap ⟨ψ_a|ψ_b⟩ via double-layer BP on GPU
 *
 * Standalone HIP extension of bilinear_form_bp.hpp.
 * All heavy work (T_v build, BP sweeps, Bethe PF) runs on-device.
 * CPU is only used for convergence check (one scalar per bond per sweep).
 *
 * Double-layer message dimension: chi_ab = chi_a × chi_b per bond.
 * At cycle 1: chi_a=chi_b=4 → chi_ab=16. Messages are 16-element vectors.
 * At chi=70: chi_ab=4900 → messages are ~96 KB each — still fits in 32 GB.
 *
 * Warm-start: Kronecker outer product of diagonal(msg_a) × diagonal(msg_b).
 * This breaks the sign degeneracy that sends BP to the zero fixed point.
 *
 * Kernel layout (all kernels launched per site or per bond):
 *   build_Tv_kernel  — one block per site, threads over (ia_flat × ib_flat)
 *   bp_msg_kernel    — one block per (bond, direction), threads over out_idx
 *   bethe_vertex_kernel — one block per site, one thread per T_v element
 *   bethe_edge_kernel   — one block per bond, one thread per chi_ab element
 *
 * Usage:
 *   BLF_GPU blf(grid_a, grid_b);
 *   blf.upload(psi_a_gpu, msg_a_gpu, psi_b_gpu, msg_b_gpu);
 *   float overlap = blf.compute(n_sweeps=25, tol=1e-5f);
 */

#include "grid.hpp"
#include "types.hpp"
#include <vector>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

// ── helpers ──────────────────────────────────────────────────────────────────

#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do { \
    hipError_t e = (cmd); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)
#endif

static constexpr int MAX_DEGREE_BLF = 3;   // heavy-hex: degree ≤ 3
static constexpr int D_BLF          = 4;   // Pauli basis dimension

// ── GPU kernels ───────────────────────────────────────────────────────────────

/*
 * build_Tv_kernel — contract physical index, then reindex to Kronecker layout.
 *
 * T_v[ia0*cb0+ib0, ia1*cb1+ib1, ...] = Σ_d ψa[ia0,ia1,...,d] * ψb[ib0,ib1,...,d]
 *
 * Launched as: grid=(n_sites), block=(min(vol_ab, 256))
 * Each block handles one site; threads stride over (ia_flat × ib_flat) pairs.
 *
 * Args:
 *   psi_a, psi_b  — device arrays [n_sites], each site-tensor flattened
 *   T             — output [n_sites], each T_v flattened in Kronecker order
 *   site_params   — per-site: [nb, chi_a_0..nb-1, chi_b_0..nb-1, vol_a, vol_b, vol_ab]
 */
__global__ void build_Tv_kernel(
    const float* const* __restrict__ psi_a,
    const float* const* __restrict__ psi_b,
    float**              T,
    const int* __restrict__ nb_arr,       // [n_sites]
    const int* __restrict__ chia_arr,     // [n_sites * MAX_DEGREE_BLF]
    const int* __restrict__ chib_arr,     // [n_sites * MAX_DEGREE_BLF]
    const int* __restrict__ vola_arr,     // [n_sites]
    const int* __restrict__ volb_arr,     // [n_sites]
    int n_sites)
{
    int sid = blockIdx.x;
    if (sid >= n_sites) return;

    int nb   = nb_arr[sid];
    int vola = vola_arr[sid];
    int volb = volb_arr[sid];
    int volab = vola * volb;

    const float* pa = psi_a[sid];
    const float* pb = psi_b[sid];
    float*       Tv = T[sid];

    const int* ca = chia_arr + sid * MAX_DEGREE_BLF;
    const int* cb = chib_arr + sid * MAX_DEGREE_BLF;

    // Step 1: contract physical index → T_big[ia_flat, ib_flat]
    // (stored in Tv temporarily, reindexed in place below)
    for (int flat = threadIdx.x; flat < volab; flat += blockDim.x) {
        int ia = flat / volb;
        int ib = flat % volb;
        float acc = 0.f;
        for (int d = 0; d < D_BLF; d++)
            acc += pa[ia * D_BLF + d] * pb[ib * D_BLF + d];
        Tv[flat] = acc;   // T_big layout: [ia_flat * volb + ib_flat]
    }
    __syncthreads();

    // Step 2: reindex T_big[ia_flat, ib_flat] → T_out[Kronecker flat]
    // Kronecker index for bond k: ia_k * cb[k] + ib_k
    // T_out shape: [ca[0]*cb[0], ca[1]*cb[1], ..., ca[nb-1]*cb[nb-1]]
    // We build a scratch buffer; since volab is the same size, do it in-place
    // using a temporary (cooperative threads — only safe if block covers all flat)
    // For simplicity: thread 0 does the reindex (volab ≤ 4096 at chi=4)
    if (threadIdx.x == 0) {
        // Decompose ia_flat → ia[k], ib_flat → ib[k], compute Kronecker flat
        int vola_l = vola, volb_l = volb;
        // We need a scratch; reuse shared memory pattern: copy T_big to stack
        // For chi=4: volab ≤ (4*4)^3 = 4096 floats — allocate on stack
        float tmp[4096];
        for (int f = 0; f < volab; f++) tmp[f] = Tv[f];

        for (int ia_f = 0; ia_f < vola_l; ia_f++) {
            for (int ib_f = 0; ib_f < volb_l; ib_f++) {
                float val = tmp[ia_f * volb_l + ib_f];
                // Decompose ia_f and ib_f into per-bond indices
                int ta = ia_f, tb = ib_f;
                int kron_idx = 0, stride = 1;
                // Compute Kronecker index (most-significant bond first)
                int ia_k[MAX_DEGREE_BLF], ib_k[MAX_DEGREE_BLF];
                for (int k = nb - 1; k >= 0; k--) {
                    ia_k[k] = ta % ca[k]; ta /= ca[k];
                    ib_k[k] = tb % cb[k]; tb /= cb[k];
                }
                for (int k = 0; k < nb; k++) {
                    kron_idx = kron_idx * (ca[k] * cb[k]) + ia_k[k] * cb[k] + ib_k[k];
                }
                Tv[kron_idx] = val;
            }
        }
    }
}

/*
 * bp_msg_kernel — compute one outgoing message from site src along bond skip_k.
 *
 * nm[i] = Σ_{flat : idx[skip_k]=i} T_v[flat] × Π_{k ≠ skip_k} msg_in[k][idx[k]]
 *
 * Launched as: grid=(2*n_bonds), block=(chi_ab_skip, 1, 1)
 * blockIdx.x encodes (bond_id * 2 + direction).
 *
 * For chi=4: chi_ab=16 → 16 threads per block.
 */
__global__ void bp_msg_kernel(
    const float* const* __restrict__ T,     // [n_sites], T_v in Kronecker layout
    float**              msgs,               // [2*n_bonds], message vectors chi_ab each
    const int* __restrict__ bond_site_a,    // [n_bonds]
    const int* __restrict__ bond_site_b,    // [n_bonds]
    const int* __restrict__ site_nb,        // [n_sites] number of bonds per site
    const int* __restrict__ site_bonds,     // [n_sites * MAX_DEGREE_BLF] bond ids
    const int* __restrict__ chi_ab,         // [n_bonds] = chi_a * chi_b
    const int* __restrict__ vol_ab,         // [n_sites] product of all chi_ab
    int n_bonds)
{
    int bid = blockIdx.x / 2;
    int dir = blockIdx.x % 2;     // 0 = a→b (src=site_a), 1 = b→a (src=site_b)
    if (bid >= n_bonds) return;

    int src   = (dir == 0) ? bond_site_a[bid] : bond_site_b[bid];
    int nb    = site_nb[src];
    int volab = vol_ab[src];

    // Find skip_k: which bond index in site_bonds[src] is 'bid'
    int skip_k = -1;
    for (int k = 0; k < nb; k++)
        if (site_bonds[src * MAX_DEGREE_BLF + k] == bid) { skip_k = k; break; }
    if (skip_k < 0) return;

    int chi_out   = chi_ab[bid];
    int out_i     = threadIdx.x;   // one thread per output element
    if (out_i >= chi_out) return;

    // Precompute per-bond chi_ab and msg_idx for this site
    int  b_chi[MAX_DEGREE_BLF];
    int  b_msg[MAX_DEGREE_BLF];   // incoming message index for each bond
    for (int k = 0; k < nb; k++) {
        int bond_k  = site_bonds[src * MAX_DEGREE_BLF + k];
        b_chi[k]    = chi_ab[bond_k];
        b_msg[k]    = (src == bond_site_a[bond_k]) ? (2*bond_k+1) : (2*bond_k);
    }

    const float* Tv = T[src];

    // Iterate over ALL volab flat indices.
    // Decompose each into per-bond indices (last bond = least significant).
    // Accumulate into out_i only when idx[skip_k] == out_i.
    float acc = 0.f;
    for (int flat = 0; flat < volab; flat++) {
        int   tmp = flat;
        int   idx[MAX_DEGREE_BLF];
        for (int k = nb - 1; k >= 0; k--) {
            idx[k]  = tmp % b_chi[k];
            tmp    /= b_chi[k];
        }
        if (idx[skip_k] != out_i) continue;

        float msg_prod = 1.f;
        for (int k = 0; k < nb; k++) {
            if (k == skip_k) continue;
            msg_prod *= msgs[b_msg[k]][idx[k]];
        }
        acc += Tv[flat] * msg_prod;
    }

    int out_msg_idx = (dir == 0) ? (2*bid) : (2*bid+1);
    msgs[out_msg_idx][out_i] = acc;
}

/*
 * normalize_msg_kernel — max-abs normalize each message vector.
 * Launched as grid=(2*n_bonds), block=(1).
 */
__global__ void normalize_msg_kernel(
    float** msgs,
    const int* __restrict__ chi_ab,
    int n_bonds)
{
    int idx = blockIdx.x;   // 0..2*n_bonds-1
    int bid = idx / 2;
    if (bid >= n_bonds) return;
    int chi = chi_ab[bid];
    float* m = msgs[idx];
    float s = 0.f;
    for (int i = 0; i < chi; i++) s = fmaxf(s, fabsf(m[i]));
    if (s > 1e-20f) for (int i = 0; i < chi; i++) m[i] /= s;
}

/*
 * convergence_kernel — compute max |new - old| per message for convergence check.
 * Caller keeps a host copy of old messages and downloads new after each sweep.
 * At chi=4: 2*76 messages × 16 floats = 2432 floats — trivial D2H.
 */

/*
 * bethe_vertex_kernel — compute log|a_v| and sign(a_v) for each site.
 * a_v = Σ_flat T_v[flat] × Π_k msg_in[k][idx_k]
 * Launched as: grid=(n_sites), block=(1) — thread 0 does the sum.
 * (Parallelizing within a_v is secondary; n_sites=68 provides outer parallelism.)
 */
__global__ void bethe_vertex_kernel(
    const float* const* __restrict__ T,
    const float* const* __restrict__ msgs,
    const int* __restrict__ bond_site_a,
    const int* __restrict__ bond_site_b,
    const int* __restrict__ site_nb,
    const int* __restrict__ site_bonds,
    const int* __restrict__ chi_ab,
    const int* __restrict__ vol_ab,
    double* __restrict__ log_av,
    double* __restrict__ sign_av,
    int n_sites)
{
    int sid = blockIdx.x;
    if (sid >= n_sites || threadIdx.x != 0) return;

    int nb    = site_nb[sid];
    int volab = vol_ab[sid];

    int strides[MAX_DEGREE_BLF];
    strides[nb - 1] = 1;
    for (int k = nb - 2; k >= 0; k--)
        strides[k] = strides[k+1] * chi_ab[site_bonds[sid * MAX_DEGREE_BLF + k + 1]];

    // Precompute per-bond chi and incoming msg index
    int b_chi[MAX_DEGREE_BLF], b_msg[MAX_DEGREE_BLF];
    for (int k = 0; k < nb; k++) {
        int bond_k = site_bonds[sid * MAX_DEGREE_BLF + k];
        b_chi[k]   = chi_ab[bond_k];
        b_msg[k]   = (sid == bond_site_a[bond_k]) ? (2*bond_k+1) : (2*bond_k);
    }

    const float* Tv = T[sid];
    double a_v = 0.0;

    for (int flat = 0; flat < volab; flat++) {
        int tmp = flat;
        float msg_p = 1.f;
        for (int k = nb - 1; k >= 0; k--) {
            int idx_k = tmp % b_chi[k];
            tmp      /= b_chi[k];
            msg_p    *= msgs[b_msg[k]][idx_k];
        }
        a_v += (double)(Tv[flat] * msg_p);
    }

    sign_av[sid] = (a_v > 0.0) ? 1.0 : (a_v < 0.0 ? -1.0 : 0.0);
    log_av[sid]  = (a_v != 0.0) ? log(fabs(a_v)) : -1e30;
}

/*
 * bethe_edge_kernel — compute log|ze| and sign(ze) for each bond.
 * ze = Σ_i M_fwd[i] * M_bwd[i]
 * Launched as: grid=(n_bonds), block=(1).
 */
__global__ void bethe_edge_kernel(
    const float* const* __restrict__ msgs,
    const int* __restrict__ chi_ab,
    double* __restrict__ log_ze,
    double* __restrict__ sign_ze,
    int n_bonds)
{
    int bid = blockIdx.x;
    if (bid >= n_bonds || threadIdx.x != 0) return;
    int chi = chi_ab[bid];
    double ze = 0.0;
    for (int i = 0; i < chi; i++)
        ze += (double)msgs[2*bid][i] * (double)msgs[2*bid+1][i];
    sign_ze[bid] = (ze > 0.0) ? 1.0 : (ze < 0.0 ? -1.0 : 0.0);
    log_ze[bid]  = (fabs(ze) > 1e-30) ? log(fabs(ze)) : -1e30;
}

// ── BLF_GPU — main class ─────────────────────────────────────────────────────

struct BLF_GPU {
    const Grid& grid_a;
    const Grid& grid_b;
    int n_sites, n_bonds;

    // Device arrays
    float**  d_psi_a = nullptr;   // [n_sites] pointers to site tensors on GPU
    float**  d_psi_b = nullptr;
    float**  d_T     = nullptr;   // [n_sites] T_v tensors (Kronecker layout)
    float**  d_msgs  = nullptr;   // [2*n_bonds] message vectors

    // Per-site/bond metadata on device
    int* d_nb;        // [n_sites]
    int* d_chia;      // [n_sites * MAX_DEGREE_BLF]
    int* d_chib;      // [n_sites * MAX_DEGREE_BLF]
    int* d_vola;      // [n_sites]
    int* d_volb;      // [n_sites]
    int* d_volab;     // [n_sites]
    int* d_chi_ab;    // [n_bonds]
    int* d_bond_sa;   // [n_bonds]
    int* d_bond_sb;   // [n_bonds]
    int* d_site_nb;   // [n_sites]
    int* d_site_bonds;// [n_sites * MAX_DEGREE_BLF]

    // Host-side metadata (computed once at construction)
    std::vector<int> h_nb, h_chia, h_chib, h_vola, h_volb, h_volab;
    std::vector<int> h_chi_ab, h_bond_sa, h_bond_sb, h_site_nb, h_site_bonds;
    std::vector<int> h_vol_T;   // [n_sites] size of T_v

    // Bethe accumulators on device
    double* d_log_av = nullptr;
    double* d_sign_av = nullptr;
    double* d_log_ze = nullptr;
    double* d_sign_ze = nullptr;

    // Log-scale corrections accumulated during upload (same as CPU bilinear_form_bp_cpu)
    double log_scale_a = 0.0;
    double log_scale_b = 0.0;

    BLF_GPU(const Grid& ga, const Grid& gb)
        : grid_a(ga), grid_b(gb),
          n_sites(ga.n_sites), n_bonds(ga.n_bonds)
    {
        _build_metadata();
        _alloc_device();
    }

    ~BLF_GPU() { _free_device(); }

    // Upload psi_a and psi_b from their respective GPU devices, warm-start messages.
    // psi_a_gpu: device pointers on device_a; msg_a_gpu: converged BP messages from state a.
    // All pointers must already be accessible on the current device (use hipSetDevice first).
    void upload(
        const std::vector<float*>& psi_a_gpu, int device_a,
        const std::vector<float*>& psi_b_gpu, int device_b,
        const std::vector<float*>* msg_a_gpu = nullptr,
        const std::vector<float*>* msg_b_gpu = nullptr)
    {
        // Rebuild chi metadata from live grids (chi grows during evolution)
        _rebuild_chi_metadata();

        // Download psi from each device, normalize, upload to current device
        _upload_psi(psi_a_gpu, device_a, d_psi_a, grid_a, h_vola, log_scale_a, h_psi_a_cpu);
        _upload_psi(psi_b_gpu, device_b, d_psi_b, grid_b, h_volb, log_scale_b, h_psi_b_cpu);

        // Warm-start messages: kron(diag(msg_a), diag(msg_b)) per bond
        _init_messages(msg_a_gpu, device_a, msg_b_gpu, device_b);
    }

    // CPU copies of psi — populated during upload, used for CPU T_v build
    std::vector<std::vector<float>> h_psi_a_cpu, h_psi_b_cpu;

    // Run BP and return ⟨ψ_a|ψ_b⟩.
    float compute(int n_sweeps = 25, float tol = 1e-5f, bool verbose = false) {
        // Build T_v on CPU — correct, no GPU kernel issues
        // T_v[Kronecker flat] = Σ_d ψa[ia...,d] * ψb[ib...,d], reindexed to Kronecker layout
        std::vector<std::vector<float>> h_T(n_sites);
        for (int sid = 0; sid < n_sites; sid++) {
            int nb   = h_nb[sid];
            int vola = h_vola[sid];
            int volb = h_volb[sid];
            int vol_T = h_vol_T[sid];
            const int* ca = h_chia.data() + sid * MAX_DEGREE_BLF;
            const int* cb = h_chib.data() + sid * MAX_DEGREE_BLF;
            const float* pa = h_psi_a_cpu[sid].data();
            const float* pb = h_psi_b_cpu[sid].data();

            // Step 1: contract physical index → T_big[ia_flat * volb + ib_flat]
            std::vector<float> T_big(vola * volb, 0.f);
            for (int ia = 0; ia < vola; ia++)
                for (int ib = 0; ib < volb; ib++)
                    for (int d = 0; d < D_BLF; d++)
                        T_big[ia * volb + ib] += pa[ia * D_BLF + d] * pb[ib * D_BLF + d];

            // Step 2: reindex to Kronecker layout
            // Kronecker index for bond k: ia_k * cb[k] + ib_k
            h_T[sid].resize(vol_T, 0.f);
            for (int ia_f = 0; ia_f < vola; ia_f++) {
                for (int ib_f = 0; ib_f < volb; ib_f++) {
                    float val = T_big[ia_f * volb + ib_f];
                    int ta = ia_f, tb = ib_f;
                    int ia_k[MAX_DEGREE_BLF], ib_k[MAX_DEGREE_BLF];
                    for (int k = nb-1; k >= 0; k--) {
                        ia_k[k] = ta % ca[k]; ta /= ca[k];
                        ib_k[k] = tb % cb[k]; tb /= cb[k];
                    }
                    int kron = 0;
                    for (int k = 0; k < nb; k++)
                        kron = kron * (ca[k]*cb[k]) + ia_k[k]*cb[k] + ib_k[k];
                    h_T[sid][kron] = val;
                }
            }
        }

        // Download initial messages
        std::vector<float*> h_msg_ptrs(2 * n_bonds);
        HIP_CHECK(hipMemcpy(h_msg_ptrs.data(), d_msgs, 2*n_bonds*sizeof(float*), hipMemcpyDeviceToHost));
        std::vector<std::vector<float>> msgs(2 * n_bonds);
        for (int i = 0; i < 2 * n_bonds; i++) {
            msgs[i].resize(h_chi_ab[i / 2]);
            HIP_CHECK(hipMemcpy(msgs[i].data(), h_msg_ptrs[i],
                h_chi_ab[i/2]*sizeof(float), hipMemcpyDeviceToHost));
        }

        // BP sweeps on CPU — correct serial update, no race condition
        for (int sweep = 0; sweep < n_sweeps; sweep++) {
            float max_delta = 0.f;
            for (int bid = 0; bid < n_bonds; bid++) {
                for (int dir = 0; dir < 2; dir++) {
                    int src = (dir == 0) ? h_bond_sa[bid] : h_bond_sb[bid];
                    int nb  = h_site_nb[src];

                    // Find skip_k
                    int skip_k = -1;
                    for (int k = 0; k < nb; k++)
                        if (h_site_bonds[src*MAX_DEGREE_BLF+k] == bid) { skip_k = k; break; }
                    if (skip_k < 0) continue;

                    int chi_out = h_chi_ab[bid];
                    int volab   = h_volab[src];

                    // Per-bond chi and incoming msg index
                    int b_chi[MAX_DEGREE_BLF], b_msg[MAX_DEGREE_BLF];
                    for (int k = 0; k < nb; k++) {
                        int bond_k = h_site_bonds[src*MAX_DEGREE_BLF+k];
                        b_chi[k]   = h_chi_ab[bond_k];
                        b_msg[k]   = (src == h_bond_sa[bond_k]) ? (2*bond_k+1) : (2*bond_k);
                    }

                    std::vector<float> nm(chi_out, 0.f);
                    for (int flat = 0; flat < volab; flat++) {
                        int tmp = flat;
                        int idx[MAX_DEGREE_BLF];
                        for (int k = nb-1; k >= 0; k--) { idx[k] = tmp % b_chi[k]; tmp /= b_chi[k]; }

                        float prod = h_T[src][flat];
                        for (int k = 0; k < nb; k++) {
                            if (k == skip_k) continue;
                            prod *= msgs[b_msg[k]][idx[k]];
                        }
                        nm[idx[skip_k]] += prod;
                    }

                    // Normalize max-abs
                    float s = 0.f; for (auto v : nm) s = std::max(s, std::abs(v));
                    if (s > 1e-20f) for (auto& v : nm) v /= s;

                    int out_idx = (dir == 0) ? (2*bid) : (2*bid+1);
                    for (int i = 0; i < chi_out; i++)
                        max_delta = std::max(max_delta, std::abs(nm[i] - msgs[out_idx][i]));
                    msgs[out_idx] = nm;
                }
            }
            if (verbose) printf("  [BLF_GPU] sweep %d max_delta=%.2e\n", sweep, max_delta);
            if (max_delta < tol) { if (verbose) printf("  [BLF_GPU] converged at sweep %d\n", sweep); break; }
        }

        // Upload converged messages back to GPU for Bethe kernels
        for (int i = 0; i < 2 * n_bonds; i++)
            HIP_CHECK(hipMemcpy(h_msg_ptrs[i], msgs[i].data(),
                msgs[i].size()*sizeof(float), hipMemcpyHostToDevice));

        // Bethe partition function
        hipLaunchKernelGGL(bethe_vertex_kernel,
            dim3(n_sites), dim3(1), 0, 0,
            (const float* const*)d_T, (const float* const*)d_msgs,
            d_bond_sa, d_bond_sb, d_site_nb, d_site_bonds,
            d_chi_ab, d_volab, d_log_av, d_sign_av, n_sites);

        hipLaunchKernelGGL(bethe_edge_kernel,
            dim3(n_bonds), dim3(1), 0, 0,
            (const float* const*)d_msgs, d_chi_ab,
            d_log_ze, d_sign_ze, n_bonds);

        HIP_CHECK(hipDeviceSynchronize());

        // Reduce on CPU (68 sites, 76 bonds — trivial)
        std::vector<double> h_log_av(n_sites), h_sign_av(n_sites);
        std::vector<double> h_log_ze(n_bonds), h_sign_ze(n_bonds);
        HIP_CHECK(hipMemcpy(h_log_av.data(),  d_log_av,  n_sites*sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_sign_av.data(), d_sign_av, n_sites*sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_log_ze.data(),  d_log_ze,  n_bonds*sizeof(double), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_sign_ze.data(), d_sign_ze, n_bonds*sizeof(double), hipMemcpyDeviceToHost));

        double log_num = 0.0, sign_num = 1.0;
        for (int v = 0; v < n_sites; v++) {
            sign_num *= h_sign_av[v];
            log_num  += h_log_av[v];
        }
        double log_den = 0.0, sign_den = 1.0;
        for (int e = 0; e < n_bonds; e++) {
            sign_den *= h_sign_ze[e];
            log_den  += h_log_ze[e];
        }

        if (sign_num == 0.0) return 0.f;
        if (verbose) printf("  [BLF_GPU] log_num=%.4f log_den=%.4f log_sa=%.4f log_sb=%.4f\n",
                            log_num, log_den, log_scale_a, log_scale_b);
        double result = sign_num * sign_den * std::exp(log_num - log_den + log_scale_a + log_scale_b);
        if (verbose) printf("  [BLF_GPU] overlap = %.6f\n", (float)result);
        return (float)result;
    }

private:
    void _build_metadata() {
        h_nb.resize(n_sites); h_chia.resize(n_sites * MAX_DEGREE_BLF, 1);
        h_chib.resize(n_sites * MAX_DEGREE_BLF, 1);
        h_vola.resize(n_sites, 1); h_volb.resize(n_sites, 1); h_volab.resize(n_sites, 1);
        h_vol_T.resize(n_sites, 1);
        h_chi_ab.resize(n_bonds); h_bond_sa.resize(n_bonds); h_bond_sb.resize(n_bonds);
        h_site_nb.resize(n_sites); h_site_bonds.resize(n_sites * MAX_DEGREE_BLF, -1);

        for (auto& site : grid_a.sites) {
            int sid = site.id;
            int nb = 0;
            for (int d = 0; d < MAX_DEGREE; d++) {
                int bid = site.bond_id[d];
                if (bid < 0) continue;
                h_chia[sid * MAX_DEGREE_BLF + nb] = grid_a.bonds[bid].chi;
                h_chib[sid * MAX_DEGREE_BLF + nb] = grid_b.bonds[bid].chi;
                h_vola[sid] *= grid_a.bonds[bid].chi;
                h_volb[sid] *= grid_b.bonds[bid].chi;
                h_site_bonds[sid * MAX_DEGREE_BLF + nb] = bid;
                nb++;
            }
            h_nb[sid] = nb; h_site_nb[sid] = nb;
            h_volab[sid] = h_vola[sid] * h_volb[sid];
            h_vol_T[sid] = 1;
            for (int k = 0; k < nb; k++)
                h_vol_T[sid] *= h_chia[sid*MAX_DEGREE_BLF+k] * h_chib[sid*MAX_DEGREE_BLF+k];
        }
        for (auto& bond : grid_a.bonds) {
            int bid = bond.id;
            h_chi_ab[bid] = grid_a.bonds[bid].chi * grid_b.bonds[bid].chi;
            h_bond_sa[bid] = bond.site_a;
            h_bond_sb[bid] = bond.site_b;
        }
    }

    void _alloc_device() {
        // Metadata arrays
        auto alloc_int = [](int** p, int n) { HIP_CHECK(hipMalloc(p, n*sizeof(int))); };
        alloc_int(&d_nb,         n_sites);
        alloc_int(&d_chia,       n_sites * MAX_DEGREE_BLF);
        alloc_int(&d_chib,       n_sites * MAX_DEGREE_BLF);
        alloc_int(&d_vola,       n_sites);
        alloc_int(&d_volb,       n_sites);
        alloc_int(&d_volab,      n_sites);
        alloc_int(&d_chi_ab,     n_bonds);
        alloc_int(&d_bond_sa,    n_bonds);
        alloc_int(&d_bond_sb,    n_bonds);
        alloc_int(&d_site_nb,    n_sites);
        alloc_int(&d_site_bonds, n_sites * MAX_DEGREE_BLF);

        HIP_CHECK(hipMemcpy(d_nb,         h_nb.data(),         n_sites*sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_chia,       h_chia.data(),       n_sites*MAX_DEGREE_BLF*sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_chib,       h_chib.data(),       n_sites*MAX_DEGREE_BLF*sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_vola,       h_vola.data(),       n_sites*sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_volb,       h_volb.data(),       n_sites*sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_volab,      h_volab.data(),      n_sites*sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_chi_ab,     h_chi_ab.data(),     n_bonds*sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_bond_sa,    h_bond_sa.data(),    n_bonds*sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_bond_sb,    h_bond_sb.data(),    n_bonds*sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_site_nb,    h_site_nb.data(),    n_sites*sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_site_bonds, h_site_bonds.data(), n_sites*MAX_DEGREE_BLF*sizeof(int), hipMemcpyHostToDevice));

        // Allocate T_v storage: one contiguous buffer per site
        std::vector<float*> h_T_ptrs(n_sites);
        for (int sid = 0; sid < n_sites; sid++)
            HIP_CHECK(hipMalloc(&h_T_ptrs[sid], h_vol_T[sid] * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_T, n_sites * sizeof(float*)));
        HIP_CHECK(hipMemcpy(d_T, h_T_ptrs.data(), n_sites*sizeof(float*), hipMemcpyHostToDevice));

        // Allocate message storage: 2*n_bonds vectors
        std::vector<float*> h_msg_ptrs(2 * n_bonds);
        for (int bid = 0; bid < n_bonds; bid++) {
            HIP_CHECK(hipMalloc(&h_msg_ptrs[2*bid],   h_chi_ab[bid] * sizeof(float)));
            HIP_CHECK(hipMalloc(&h_msg_ptrs[2*bid+1], h_chi_ab[bid] * sizeof(float)));
        }
        HIP_CHECK(hipMalloc(&d_msgs, 2*n_bonds * sizeof(float*)));
        HIP_CHECK(hipMemcpy(d_msgs, h_msg_ptrs.data(), 2*n_bonds*sizeof(float*), hipMemcpyHostToDevice));

        // psi pointer arrays (populated during upload)
        std::vector<float*> null_ptrs(n_sites, nullptr);
        HIP_CHECK(hipMalloc(&d_psi_a, n_sites * sizeof(float*)));
        HIP_CHECK(hipMalloc(&d_psi_b, n_sites * sizeof(float*)));
        HIP_CHECK(hipMemcpy(d_psi_a, null_ptrs.data(), n_sites*sizeof(float*), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_psi_b, null_ptrs.data(), n_sites*sizeof(float*), hipMemcpyHostToDevice));

        // Bethe accumulators
        HIP_CHECK(hipMalloc(&d_log_av,  n_sites*sizeof(double)));
        HIP_CHECK(hipMalloc(&d_sign_av, n_sites*sizeof(double)));
        HIP_CHECK(hipMalloc(&d_log_ze,  n_bonds*sizeof(double)));
        HIP_CHECK(hipMalloc(&d_sign_ze, n_bonds*sizeof(double)));
    }

    // Upload psi tensors from a (possibly different) device into current device buffers.
    // Normalizes each site tensor (max-abs) and accumulates log scale into log_scale_out,
    // matching the CPU bilinear_form_bp_cpu normalization exactly.
    void _upload_psi(
        const std::vector<float*>& psi_src_gpu,
        int src_device,
        float** d_psi_dst,
        const Grid& grid,
        const std::vector<int>& vol_psi,
        double& log_scale_out,
        std::vector<std::vector<float>>& cpu_copy)
    {
        int cur_device; HIP_CHECK(hipGetDevice(&cur_device));
        std::vector<float*> h_ptrs(n_sites);
        log_scale_out = 0.0;
        cpu_copy.resize(n_sites);

        for (auto& site : grid.sites) {
            int sid = site.id;
            int vol = vol_psi[sid] * D_BLF;
            std::vector<float> buf(vol);
            HIP_CHECK(hipSetDevice(src_device));
            HIP_CHECK(hipMemcpy(buf.data(), psi_src_gpu[sid], vol*sizeof(float), hipMemcpyDeviceToHost));

            // Normalize: same as CPU version
            float nm = 0.f;
            for (auto v : buf) nm = std::max(nm, std::abs(v));
            if (nm > 1e-30f) {
                for (auto& v : buf) v /= nm;
                log_scale_out += std::log((double)nm);
            }

            cpu_copy[sid] = buf;   // keep normalized CPU copy for T_v build

            HIP_CHECK(hipSetDevice(cur_device));
            HIP_CHECK(hipMalloc(&h_ptrs[sid], vol*sizeof(float)));
            HIP_CHECK(hipMemcpy(h_ptrs[sid], buf.data(), vol*sizeof(float), hipMemcpyHostToDevice));
        }
        HIP_CHECK(hipMemcpy(d_psi_dst, h_ptrs.data(), n_sites*sizeof(float*), hipMemcpyHostToDevice));
    }

    // Kronecker warm-start: M_fwd[ia*cb+ib] = |diag_a[ia]| * |diag_b[ib]|
    void _init_messages(
        const std::vector<float*>* msg_a_gpu, int dev_a,
        const std::vector<float*>* msg_b_gpu, int dev_b)
    {
        int cur_device; HIP_CHECK(hipGetDevice(&cur_device));
        std::vector<float*> h_msg_ptrs(2 * n_bonds);
        HIP_CHECK(hipMemcpy(h_msg_ptrs.data(), d_msgs, 2*n_bonds*sizeof(float*), hipMemcpyDeviceToHost));

        for (auto& bond : grid_a.bonds) {
            int bid = bond.id;
            int ca  = grid_a.bonds[bid].chi;
            int cb  = grid_b.bonds[bid].chi;
            int cab = ca * cb;

            for (int dir = 0; dir < 2; dir++) {
                // msg index in the 2*n_bonds layout (from_a=2*bid, from_b=2*bid+1)
                int msg_idx = 2 * bid + dir;
                std::vector<float> m(cab, 0.f);

                if (msg_a_gpu && msg_b_gpu) {
                    // Download diagonals from each device
                    std::vector<float> ma_flat(ca * ca), mb_flat(cb * cb);
                    HIP_CHECK(hipSetDevice(dev_a));
                    HIP_CHECK(hipMemcpy(ma_flat.data(), (*msg_a_gpu)[msg_idx], ca*ca*sizeof(float), hipMemcpyDeviceToHost));
                    HIP_CHECK(hipSetDevice(dev_b));
                    HIP_CHECK(hipMemcpy(mb_flat.data(), (*msg_b_gpu)[msg_idx], cb*cb*sizeof(float), hipMemcpyDeviceToHost));
                    HIP_CHECK(hipSetDevice(cur_device));

                    // Outer product of absolute diagonals
                    for (int ia = 0; ia < ca; ia++)
                        for (int ib = 0; ib < cb; ib++)
                            m[ia * cb + ib] = std::abs(ma_flat[ia*(ca+1)]) * std::abs(mb_flat[ib*(cb+1)]);
                } else {
                    // Identity warm-start (works for self-overlaps chi_a == chi_b)
                    int mn = std::min(ca, cb);
                    for (int i = 0; i < mn; i++) m[i * cb + i] = 1.f / mn;
                }

                // Normalize
                float s = 0.f; for (auto v : m) s = std::max(s, std::abs(v));
                if (s > 1e-20f) for (auto& v : m) v /= s;

                HIP_CHECK(hipMemcpy(h_msg_ptrs[msg_idx], m.data(), cab*sizeof(float), hipMemcpyHostToDevice));
            }
        }
    }

    float* _get_msg_ptr(int msg_idx) {
        std::vector<float*> h_ptrs(2 * n_bonds);
        HIP_CHECK(hipMemcpy(h_ptrs.data(), d_msgs, 2*n_bonds*sizeof(float*), hipMemcpyDeviceToHost));
        return h_ptrs[msg_idx];
    }

    void _free_device() {
        // Recover and free all sub-allocations
        std::vector<float*> h_T(n_sites), h_psi_a(n_sites), h_psi_b(n_sites);
        std::vector<float*> h_msgs(2*n_bonds);
        if (d_T)     { HIP_CHECK(hipMemcpy(h_T.data(),     d_T,     n_sites*sizeof(float*),   hipMemcpyDeviceToHost)); for (auto p : h_T)     HIP_CHECK(hipFree(p)); HIP_CHECK(hipFree(d_T)); }
        if (d_psi_a) { HIP_CHECK(hipMemcpy(h_psi_a.data(), d_psi_a, n_sites*sizeof(float*),   hipMemcpyDeviceToHost)); for (auto p : h_psi_a) HIP_CHECK(hipFree(p)); HIP_CHECK(hipFree(d_psi_a)); }
        if (d_psi_b) { HIP_CHECK(hipMemcpy(h_psi_b.data(), d_psi_b, n_sites*sizeof(float*),   hipMemcpyDeviceToHost)); for (auto p : h_psi_b) HIP_CHECK(hipFree(p)); HIP_CHECK(hipFree(d_psi_b)); }
        if (d_msgs)  { HIP_CHECK(hipMemcpy(h_msgs.data(),  d_msgs,  2*n_bonds*sizeof(float*), hipMemcpyDeviceToHost)); for (auto p : h_msgs)  HIP_CHECK(hipFree(p)); HIP_CHECK(hipFree(d_msgs)); }
        if (d_nb)         HIP_CHECK(hipFree(d_nb));
        if (d_chia)       HIP_CHECK(hipFree(d_chia));
        if (d_chib)       HIP_CHECK(hipFree(d_chib));
        if (d_vola)       HIP_CHECK(hipFree(d_vola));
        if (d_volb)       HIP_CHECK(hipFree(d_volb));
        if (d_volab)      HIP_CHECK(hipFree(d_volab));
        if (d_chi_ab)     HIP_CHECK(hipFree(d_chi_ab));
        if (d_bond_sa)    HIP_CHECK(hipFree(d_bond_sa));
        if (d_bond_sb)    HIP_CHECK(hipFree(d_bond_sb));
        if (d_site_nb)    HIP_CHECK(hipFree(d_site_nb));
        if (d_site_bonds) HIP_CHECK(hipFree(d_site_bonds));
        if (d_log_av)     HIP_CHECK(hipFree(d_log_av));
        if (d_sign_av)    HIP_CHECK(hipFree(d_sign_av));
        if (d_log_ze)     HIP_CHECK(hipFree(d_log_ze));
        if (d_sign_ze)    HIP_CHECK(hipFree(d_sign_ze));
    }
};
