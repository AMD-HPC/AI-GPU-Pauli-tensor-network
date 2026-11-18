/*
 * main_pauli_sweep_2gpu_rect16x32.cpp — 2-GPU split of 16×32 simulation.
 *
 * Splits the 16×32 grid horizontally at row 8:
 *   GPU 0: rows 0-8  (9×32, row 8 = overlap row, owned by GPU 0)
 *   GPU 1: rows 0-8  (9×32, row 0 = ghost of 16×32 row 8, owned by GPU 0)
 *
 * Before the cut color (vertical even, connects Grid 1 row 0 → row 1):
 *   P2P: GPU 0 row 8 → GPU 1 row 0  (unidirectional, only if chi_alloc > 1)
 *
 * Both GPUs compute their colors independently, each keeping its owned sites.
 * BP exchange at end of each cycle (boundary messages on row 8↔9 bond).
 *
 * Validation: merge PAULI_NL from both halves and compare to single-GPU reference.
 *   Single GPU: ./main_pauli_sweep_rect16x32 40 3 1e-3 --center=256 --state=NL
 *   2-GPU:      ./main_pauli_sweep_2gpu_rect16x32 40 3 1e-3 --center=256
 *
 * Usage:
 *   ./main_pauli_sweep_2gpu_rect16x32 [chi_max] [n_steps] [cutoff] [--center=N]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <chrono>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

// 9×32 grid (each half)
#include "rect_9x32.hpp"
#include "rect_9x32_ideal_kick_eps010_data.hpp"

// 7×32 boundary slice grid (GPU2 BP)
#define heavy_hex_3x3_ideal_kick_eps010_ptm heavy_hex_3x3_ideal_kick_eps010_ptm_7x32
#include "rect_7x32.hpp"
#include "rect_7x32_ideal_kick_eps010_data.hpp"
#undef heavy_hex_3x3_ideal_kick_eps010_ptm

// 16×32 grid (for perturb-bond PTM reference) — guard duplicate legacy symbol
#define heavy_hex_3x3_ideal_kick_eps010_ptm heavy_hex_3x3_ideal_kick_eps010_ptm_16x32_v2
#include "rect_16x32.hpp"
#include "rect_16x32_ideal_kick_eps010_data.hpp"
#undef heavy_hex_3x3_ideal_kick_eps010_ptm

// Site kick PTM from 16×32 (same value as 9×32 — uniform J=1)
// Use 9×32 site PTM directly to avoid duplicate symbol issues
static const float* s_site_ptm_ptr = RECT_9X32_IDEAL_KICK_EPS010_SITE_PTM;

#include "heavyhex_physics.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "measures.hpp"
#include "trotter.hpp"
#include "trotter_2gpu.hpp"
#include "types.hpp"

#ifndef HIP_CHECK
#define HIP_CHECK(cmd) do { \
    hipError_t e=(cmd); \
    if(e!=hipSuccess){fprintf(stderr,"HIP %s at %s:%d\n", \
        hipGetErrorString(e),__FILE__,__LINE__);exit(1);} \
} while(0)
#endif

static double now_ms() {
    auto t = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double,std::milli>(
        t.time_since_epoch()).count();
}

// ── PTM storage for perturbed bond ───────────────────────────────────────────
static std::vector<float> s_perturb_ptm;  // 256 floats if set
static int s_perturb_bond_local = -1;     // bond id in the local 9×32 grid

static const float* ptm_gpu0(int b) {
    if (b == s_perturb_bond_local && !s_perturb_ptm.empty())
        return s_perturb_ptm.data();
    return RECT_9X32_IDEAL_KICK_EPS010[b];
}
static const float* ptm_gpu1(int b) {
    if (b == s_perturb_bond_local && !s_perturb_ptm.empty())
        return s_perturb_ptm.data();
    return RECT_9X32_IDEAL_KICK_EPS010[b];
}

int main(int argc, char** argv) {
    int   chi_max = (argc > 1) ? atoi(argv[1]) : 40;
    int   n_steps = (argc > 2) ? atoi(argv[2]) : 3;
    float cutoff  = (argc > 3) ? atof(argv[3]) : 1e-3f;
    int   center  = 256;   // legacy single center (0-based)
    std::vector<int> centers16;  // multi-center list (0-based, 16x32)
    int   perturb_bond16 = -1;
    float perturb_J      = 0.8f;

    for (int i = 4; i < argc; i++) {
        std::string a = argv[i];
        if (a.substr(0,9) == "--center=") {
            // parse comma-separated 1-based site list
            centers16.clear();
            char* p = argv[i]+9;
            while (*p) {
                centers16.push_back(atoi(p) - 1);  // 1-based→0-based
                while (*p && *p != ',') p++;
                if (*p == ',') p++;
            }
            center = centers16[0];
        }
        if (a.substr(0,15) == "--perturb-bond=") {
            // format: bond_id,J_pert  e.g. 200,0.800
            char* p = argv[i]+15;
            perturb_bond16 = atoi(p);
            char* comma = strchr(p, ',');
            if (comma) perturb_J = atof(comma+1);
        }
    }

    // Build perturbed PTM if requested
    if (perturb_bond16 >= 0) {
        // 16×32 bond layout: 496 horizontal (0-495) + 480 vertical (496-975)
        // 9×32 bond layout:  279 horizontal (0-278) + 256 vertical (279-534)
        int bond9 = -1;
        if (perturb_bond16 < 496) {
            // horizontal: row*31+col in 16×32 → same in 9×32 if row < 8
            int row16 = perturb_bond16 / 31;
            int col16 = perturb_bond16 % 31;
            if (row16 < 8) bond9 = row16 * 31 + col16;
            else { int row9 = row16 - 8; bond9 = row9 * 31 + col16; }
        } else {
            // vertical: (bond16-496)/32 = row, col in 16×32
            int rv = (perturb_bond16 - 496) / 32;
            int cv = (perturb_bond16 - 496) % 32;
            if (rv < 8) bond9 = 279 + rv * 32 + cv;
            else { bond9 = 279 + (rv - 8) * 32 + cv; }
        }
        s_perturb_bond_local = bond9;
        // Build exact perturbed PTM: exp(-i*J_pert*eps*(XX+YY+ZZ))
        // Reuse build_ptm computation inline (same as breaker.py)
        // PTM elements for exp(-i*theta*(XX+YY+ZZ)), theta = J_pert * eps
        float eps = 0.10f;
        float th  = perturb_J * eps;
        float c   = cosf(th), s_ = sinf(th);
        float c2  = c*c, s2 = s_*s_, cs = c*s_;
        // 16x16 PTM in Pauli basis {II,IX,IY,IZ,XI,...,ZZ}
        // This is the exact same PTM as IDEAL_KICK_EPS010 but with J_pert instead of J=1
        // For now use the breaker.py broken.json ptm_perturbed if available,
        // otherwise fall back to rescaled PTM (accurate for small dJ)
        const float* base = RECT_9X32_IDEAL_KICK_EPS010[bond9 >= 0 ? bond9 : 0];
        s_perturb_ptm.assign(base, base + 256);
        // Exact rescaling: theta_pert = J_pert * eps, theta_base = 1.0 * eps
        // PTM(theta) = diag(1) + off-diag terms proportional to sin/cos(theta)
        // Rescale: multiply each off-diagonal by sin(theta_pert)/sin(theta_base)
        // and diagonal corrections by (cos(theta_pert)-1)/(cos(theta_base)-1)
        float th0  = 1.0f * eps;  // base theta
        float rat_sin = (fabsf(sinf(th0)) > 1e-8f) ? sinf(th)  / sinf(th0)  : 1.0f;
        float rat_cos = (fabsf(cosf(th0)-1.0f) > 1e-8f) ? (cosf(th)-1.0f)/(cosf(th0)-1.0f) : 1.0f;
        for (int ii = 0; ii < 16; ii++) {
            for (int jj = 0; jj < 16; jj++) {
                float v = s_perturb_ptm[ii*16+jj];
                if (ii == jj) {
                    // diagonal: 1 + (cos-1)*something → scale the deviation
                    s_perturb_ptm[ii*16+jj] = 1.0f + (v - 1.0f) * rat_cos;
                } else {
                    s_perturb_ptm[ii*16+jj] = v * rat_sin;
                }
            }
        }
        printf("[2GPU] Perturb bond %d (16x32) → bond %d (9x32)  J_pert=%.3f\n",
               perturb_bond16, bond9, perturb_J);
    }

    printf("[2GPU] chi=%d  n_steps=%d  cutoff=%.0e  total_centers=%zu\n",
           chi_max, n_steps, (double)cutoff, centers16.size());

    // ── GPU 0 setup ───────────────────────────────────────────────────────────
    HIP_CHECK(hipSetDevice(0));
    auto grid0 = std::make_unique<FlatRect9x32Grid>(chi_max);
    auto ws0   = std::make_unique<Workspace>(chi_max, MAX_DEGREE);
    ws0->alloc_trotter_buffers(grid0->n_bonds);
    rocblas_handle handle0; rocblas_create_handle(&handle0);
    GateOps gops0(handle0);
    auto meas0 = std::make_unique<Measures>(handle0, *ws0);

    // Split centers across GPUs by row
    // GPU0: rows 0-7, GPU1: rows 8-15 (mapped to rows 0-7 in GPU1 local coords)
    if (centers16.empty()) centers16.push_back(center);
    std::vector<int> centers0, centers1;
    for (int s : centers16) {
        int r = s / 32, c = s % 32;
        if (r < 8)       centers0.push_back(r * 32 + c);        // GPU0: same coords
        else if (r < 16) centers1.push_back((r - 8) * 32 + c);  // GPU1: row offset -8
    }
    int center0 = centers0.empty() ? 4*32+16 : centers0[0];  // BP reference center
    printf("[2GPU] GPU0: %zu centers (rows 0-7)  GPU1: %zu centers (rows 8-15)\n",
           centers0.size(), centers1.size());

    // ── GPU 1 setup ───────────────────────────────────────────────────────────
    HIP_CHECK(hipSetDevice(1));
    auto grid1 = std::make_unique<FlatRect9x32Grid>(chi_max);
    auto ws1   = std::make_unique<Workspace>(chi_max, MAX_DEGREE);
    ws1->alloc_trotter_buffers(grid1->n_bonds);
    rocblas_handle handle1; rocblas_create_handle(&handle1);
    GateOps gops1(handle1);
    auto meas1 = std::make_unique<Measures>(handle1, *ws1);

    int center1 = centers1.empty() ? 0 : centers1[0];  // BP reference center

    // ── GPU 2 setup: 7×32 boundary slice BP ──────────────────────────────────
    // Slice covers 16×32 rows 5-11 (7 rows, 224 sites) straddling the split at row 8.
    // GPU0 contributes rows 5-8, GPU1 contributes rows 8-11.
    // GPU2 runs BP on slice → exact boundary messages → inject to GPU0/GPU1.
    HIP_CHECK(hipSetDevice(2));
    auto grid2  = std::make_unique<FlatRect7x32Grid>(chi_max);
    auto ws2    = std::make_unique<Workspace>(chi_max, MAX_DEGREE);
    rocblas_handle handle2; rocblas_create_handle(&handle2);
    auto meas2  = std::make_unique<Measures>(handle2, *ws2);
    auto state2 = std::make_unique<HeavyHexState>(static_cast<Grid&>(*grid2));
    // Map centers in slice rows (16×32 rows 5-11 → slice rows 0-6)
    std::vector<int> centers2;
    for (int s : centers16) {
        int r = s / 32, c = s % 32;
        if (r >= 5 && r <= 11) centers2.push_back((r - 5) * 32 + c);
    }
    int center2 = centers2.empty() ? 3*32+16 : centers2[0];
    state2->init_heisenberg(static_cast<Grid&>(*grid2), centers2);
    printf("[2GPU] GPU2: 7×32 boundary slice BP (rows 5-11, %zu centers)\n", centers2.size());

    // ── Allocate states ───────────────────────────────────────────────────────
    HIP_CHECK(hipSetDevice(0));
    auto state0 = std::make_unique<HeavyHexState>(static_cast<Grid&>(*grid0));
    state0->init_heisenberg(static_cast<Grid&>(*grid0), centers0);

    HIP_CHECK(hipSetDevice(1));
    auto state1 = std::make_unique<HeavyHexState>(static_cast<Grid&>(*grid1));
    state1->init_heisenberg(static_cast<Grid&>(*grid1), centers1);  // empty = silent

    // ── TrotterColorCtx ───────────────────────────────────────────────────────
    HIP_CHECK(hipSetDevice(0));
    auto ctx0 = std::make_unique<TrotterColorCtx>(*grid0, chi_max);
    HIP_CHECK(hipSetDevice(1));
    auto ctx1 = std::make_unique<TrotterColorCtx>(*grid1, chi_max);

    // ── Cut color: color 2 (vertical even, row 0→1 in Grid 1) ────────────────
    // Grid 1's row 0 = ghost of GPU 0's row 8. Must sync before color 2 fires.
    const int cut_color = 2;
    // First vertical bond in Grid 1 (row 0, col 0) = bond index 279
    const int cut_bond_id = 279;
    // GPU 0 row 8 start site: 8*32 = 256
    const int cut_site_top = 8 * 32;   // first site of row 8 in Grid 0
    const int cut_site_bot = 0;         // first site of row 0 in Grid 1 (ghost)

    printf("[2GPU] Cut color=%d, cut_bond=%d\n", cut_color, cut_bond_id);
    printf("[2GPU] P2P: GPU0 site %d--%d → GPU1 site %d--%d\n",
           cut_site_top, cut_site_top+31, cut_site_bot, cut_site_bot+31);

    // Site kick PTM (same for all sites, uniform J=1)
    const float* site_ptm = s_site_ptm_ptr;
    const int bp_max = 100;

    // ── Simulation loop ───────────────────────────────────────────────────────
    for (int step = 1; step <= n_steps; step++) {
        double t0 = now_ms();

        // Colors before cut: sequential (rocBLAS handles not thread-safe cross-device)
        for (int c = 0; c < cut_color; c++) {
            HIP_CHECK(hipSetDevice(0));
            do_trotter_one_color(*ctx0, c,
                state0->psi, state0->messages, *grid0,
                ptm_gpu0, nullptr, gops0, *ws0, chi_max, cutoff);
            HIP_CHECK(hipSetDevice(1));
            do_trotter_one_color(*ctx1, c,
                state1->psi, state1->messages, *grid1,
                ptm_gpu1, nullptr, gops1, *ws1, chi_max, cutoff);
        }

        // P2P sync: GPU 0 row 8 → GPU 1 row 0 (UNIDIRECTIONAL — one way only)
        // GPU1 needs GPU0's row 8 as ghost to compute the cut bond (row0↔row1 in grid1).
        // GPU0's row 8 is NOT updated by GPU1 — no reverse copy needed.
        // The cut bond only exists in grid1 (row0↔row1); grid0 has no bond below row 8.
        if (grid1->bonds[cut_bond_id].chi_alloc > 1) {
            HIP_CHECK(hipSetDevice(0));
            for (int col = 0; col < RECT_9X32_NX; col++) {
                int s_top = cut_site_top + col;   // GPU0 row 8, col
                int s_bot = cut_site_bot + col;   // GPU1 row 0, col (ghost)
                size_t sz = (size_t)D;
                for (int k = 0; k < MAX_DEGREE; k++) {
                    int bid = grid0->sites[s_top].bond_id[k];
                    if (bid >= 0) sz *= grid0->bonds[bid].chi_alloc;
                }
                if (sz > (size_t)D) {
                    HIP_CHECK(hipMemcpyPeer(
                        state1->psi[s_bot], 1,
                        state0->psi[s_top], 0,
                        sz * sizeof(Scalar)));
                }
            }
            HIP_CHECK(hipDeviceSynchronize());
        }

        // Cut color: both GPUs compute sequentially
        HIP_CHECK(hipSetDevice(0));
        do_trotter_one_color(*ctx0, cut_color,
            state0->psi, state0->messages, *grid0,
            ptm_gpu0, nullptr, gops0, *ws0, chi_max, cutoff);
        HIP_CHECK(hipSetDevice(1));
        do_trotter_one_color(*ctx1, cut_color,
            state1->psi, state1->messages, *grid1,
            ptm_gpu1, nullptr, gops1, *ws1, chi_max, cutoff);

        // Colors after cut: sequential
        for (int c = cut_color + 1; c < (int)grid0->colors.size(); c++) {
            HIP_CHECK(hipSetDevice(0));
            do_trotter_one_color(*ctx0, c,
                state0->psi, state0->messages, *grid0,
                ptm_gpu0, nullptr, gops0, *ws0, chi_max, cutoff);
            HIP_CHECK(hipSetDevice(1));
            do_trotter_one_color(*ctx1, c,
                state1->psi, state1->messages, *grid1,
                ptm_gpu1, nullptr, gops1, *ws1, chi_max, cutoff);
        }

        // Site kick
        {
            auto do_kick = [&](int gpu, Grid& grid,
                               std::vector<Scalar*>& psi, GateOps& gops) {
                HIP_CHECK(hipSetDevice(gpu));
                size_t tmp_sz = 0;
                for (auto& site : grid.sites) {
                    int vol = D;
                    for (int d = 0; d < MAX_DEGREE; d++)
                        if (site.bond_id[d] >= 0)
                            vol *= grid.bonds[site.bond_id[d]].chi;
                    tmp_sz = std::max(tmp_sz, (size_t)vol);
                }
                Scalar* tmp;
                HIP_CHECK(hipMalloc(&tmp, tmp_sz * sizeof(Scalar)));
                for (auto& site : grid.sites)
                    gops.apply_unary_gate(psi[site.id], site, grid, site_ptm, tmp);
                HIP_CHECK(hipFree(tmp));
            };
            do_kick(0, *grid0, state0->psi, gops0);
            do_kick(1, *grid1, state1->psi, gops1);
        }

        // ── BP: 7×32 boundary slice on GPU2 ──────────────────────────────────────
        // Slice covers 16×32 rows 5-11 (7 rows around split at row 8).
        // GPU0 rows 5-8 (sites 160-287) → slice rows 0-3 (sites 0-127)
        // GPU1 rows 1-4 (sites 32-159 in grid1 = 16×32 rows 9-12) → slice rows 4-7 → only 3 rows needed
        // For simplicity: slice rows 0-3 from GPU0, rows 3-6 from GPU1
        int bp0_sweeps = 0, bp1_sweeps = 0, bp2_sweeps = 0;
        float bp0_delta = 0.f, bp1_delta = 0.f, bp2_delta = 0.f;

        // Step 1: copy boundary psi to GPU2 slice
        // GPU0 rows 5-8 → slice rows 0-3 (sites 0-127 in slice = sites 160-287 in grid0)
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 32; col++) {
                int s0 = (row + 5) * 32 + col;  // GPU0 site (rows 5-8)
                int s2 = row * 32 + col;          // slice site (rows 0-3)
                size_t sz = (size_t)D;
                for (int k = 0; k < MAX_DEGREE; k++) {
                    int bid = grid0->sites[s0].bond_id[k];
                    if (bid >= 0) sz *= grid0->bonds[bid].chi_alloc;
                }
                HIP_CHECK(hipMemcpyPeer(state2->psi[s2], 2, state0->psi[s0], 0, sz*sizeof(Scalar)));
            }
        }
        // GPU1 rows 1-3 (sites 32-127 in grid1 = 16×32 rows 9-11) → slice rows 4-6
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 32; col++) {
                int s1 = (row + 1) * 32 + col;   // GPU1 site (rows 1-3, skip ghost row 0)
                int s2 = (row + 4) * 32 + col;    // slice site (rows 4-6)
                size_t sz = (size_t)D;
                for (int k = 0; k < MAX_DEGREE; k++) {
                    int bid = grid1->sites[s1].bond_id[k];
                    if (bid >= 0) sz *= grid1->bonds[bid].chi_alloc;
                }
                HIP_CHECK(hipMemcpyPeer(state2->psi[s2], 2, state1->psi[s1], 1, sz*sizeof(Scalar)));
            }
        }
        // Sync chi values: slice bonds from grid0 (rows 5-8 horizontal + vertical)
        // Approximate: use grid0 chi for first 4 rows of slice
        for (int b = 0; b < (int)std::min((int)grid0->n_bonds, grid2->n_bonds); b++)
            grid2->bonds[b].chi = std::min(grid0->bonds[b].chi, grid2->bonds[b].chi_alloc);
        HIP_CHECK(hipDeviceSynchronize());

        // Step 2: BP on slice (GPU2) — exact boundary messages
        HIP_CHECK(hipSetDevice(2));
        meas2->autocorrelation_with_bp(*grid2, state2->psi, state2->messages,
            center2, bp_max, nullptr, nullptr, &bp2_sweeps, &bp2_delta, nullptr, nullptr);

        // Step 3: inject slice boundary messages → GPU0 and GPU1
        // Boundary: slice row 3 ↔ row 4 = split between GPU0 and GPU1 contributions
        // In slice bond layout: vertical bonds row3↔row4 start at 279 + 3*32
        {
            const int boff2 = 279 + 3 * 32;           // row3↔row4 vertical in slice
            const int boff0 = 279 + 7 * RECT_9X32_NX; // row7↔row8 vertical in grid0
            const int boff1 = 279;                     // row0↔row1 vertical in grid1
            for (int col = 0; col < 32; col++) {
                const Bond& b2 = grid2->bonds[boff2 + col];
                const Bond& b0 = grid0->bonds[boff0 + col];
                const Bond& b1 = grid1->bonds[boff1 + col];
                if (b2.chi > 1) {
                    size_t sz = (size_t)b2.chi * b2.chi * sizeof(Scalar);
                    HIP_CHECK(hipMemcpyPeer(state0->messages[b0.msg_from_b()], 0,
                                            state2->messages[b2.msg_from_b()], 2, sz));
                    HIP_CHECK(hipMemcpyPeer(state1->messages[b1.msg_from_a()], 1,
                                            state2->messages[b2.msg_from_a()], 2, sz));
                }
            }
            HIP_CHECK(hipDeviceSynchronize());
        }

        // Step 4: local BP on GPU0 and GPU1 with corrected boundary messages
        HIP_CHECK(hipSetDevice(0));
        meas0->autocorrelation_with_bp(*grid0, state0->psi, state0->messages,
            center0, bp_max, nullptr, nullptr, &bp0_sweeps, &bp0_delta, nullptr, nullptr);
        HIP_CHECK(hipSetDevice(1));
        meas1->autocorrelation_with_bp(*grid1, state1->psi, state1->messages,
            center1 >= 0 ? center1 : 0, bp_max,
            nullptr, nullptr, &bp1_sweeps, &bp1_delta, nullptr, nullptr);

        // Actual chi
        int chi_actual = 1;
        for (auto& b : grid0->bonds) chi_actual = std::max(chi_actual, b.chi);

        double t = (now_ms() - t0) / 1000.0;
        printf("STEP %d  chi %d  bp2 %d/%.2e  bp0 %d/%.2e  bp1 %d/%.2e  t %.2f\n",
               step, chi_actual, bp2_sweeps, (double)bp2_delta,
               bp0_sweeps, (double)bp0_delta,
               bp1_sweeps, (double)bp1_delta, t);

        // Pauli weights — merge both halves
        std::vector<float> nid0(grid0->n_sites, 0.f);
        std::vector<float> nid1(grid1->n_sites, 0.f);

        HIP_CHECK(hipSetDevice(0));
        meas0->compute_pauli_profile(*grid0, state0->psi, nid0.data());
        HIP_CHECK(hipSetDevice(1));
        meas1->compute_pauli_profile(*grid1, state1->psi, nid1.data());

        // Merged output: 512 sites total (16×32)
        // GPU0 rows 0-7 (sites 0-255) → 16×32 rows 0-7
        // GPU1 rows 1-8 (sites 32-287) → 16×32 rows 8-15 (skip row 0 = ghost)
        printf("PAULI_NL %d", step);
        for (int i = 0; i < 256; i++) printf(" %.7f", nid0[i]);       // rows 0-7
        for (int i = 32; i < grid1->n_sites; i++) printf(" %.7f", nid1[i]); // rows 8-15
        printf("\n");
    }

    return 0;
}
