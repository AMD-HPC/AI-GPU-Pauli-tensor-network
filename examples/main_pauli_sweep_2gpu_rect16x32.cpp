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

// ── PTM functions: uniform J=1, same PTM regardless of grid position ─────────
// Both GPUs use the 9×32 PTM array directly — couplings are uniform so
// bond b in 9×32 has the same PTM as the corresponding bond in 16×32.
static const float* ptm_gpu0(int b) { return RECT_9X32_IDEAL_KICK_EPS010[b]; }
static const float* ptm_gpu1(int b) { return RECT_9X32_IDEAL_KICK_EPS010[b]; }

int main(int argc, char** argv) {
    int   chi_max = (argc > 1) ? atoi(argv[1]) : 40;
    int   n_steps = (argc > 2) ? atoi(argv[2]) : 3;
    float cutoff  = (argc > 3) ? atof(argv[3]) : 1e-3f;
    int   center  = 256;  // 16×32 center (default: row 8, col 0 — boundary area)

    for (int i = 4; i < argc; i++) {
        std::string a = argv[i];
        if (a.substr(0,9) == "--center=") center = atoi(argv[i]+9) - 1; // 1-based→0-based
    }

    printf("[2GPU] 16×32 split: GPU0=rows0-8, GPU1=rows8-15\n");
    printf("[2GPU] chi=%d  n_steps=%d  cutoff=%.0e  center=%d (1-based)\n",
           chi_max, n_steps, (double)cutoff, center+1);

    // ── GPU 0 setup ───────────────────────────────────────────────────────────
    HIP_CHECK(hipSetDevice(0));
    auto grid0 = std::make_unique<FlatRect9x32Grid>(chi_max);
    auto ws0   = std::make_unique<Workspace>(chi_max, MAX_DEGREE);
    ws0->alloc_trotter_buffers(grid0->n_bonds);
    rocblas_handle handle0; rocblas_create_handle(&handle0);
    GateOps gops0(handle0);
    auto meas0 = std::make_unique<Measures>(handle0, *ws0);

    // Map 16×32 center to GPU 0 center (if center is in rows 0-7)
    int center0 = -1;
    {
        int r = center / 32, c = center % 32;
        if (r < 8) center0 = r * 32 + c;  // same position in 9×32
        else center0 = 4 * 32 + 16;        // fallback: middle of grid0
    }

    // ── GPU 1 setup ───────────────────────────────────────────────────────────
    HIP_CHECK(hipSetDevice(1));
    auto grid1 = std::make_unique<FlatRect9x32Grid>(chi_max);
    auto ws1   = std::make_unique<Workspace>(chi_max, MAX_DEGREE);
    ws1->alloc_trotter_buffers(grid1->n_bonds);
    rocblas_handle handle1; rocblas_create_handle(&handle1);
    GateOps gops1(handle1);
    auto meas1 = std::make_unique<Measures>(handle1, *ws1);

    // Map 16×32 center to GPU 1 coordinate system
    // GPU1 rows 0-8 = 16×32 rows 8-15 (row 0 of GPU1 = row 8 of 16×32 = ghost)
    // If center in rows 8-15: map to GPU1 rows 0-7
    // If center in rows 0-7:  GPU1 is silent (center=-1, no wavefront)
    int center1 = -1;
    {
        int r = center / 32, c = center % 32;
        if (r >= 8) center1 = (r - 8) * 32 + c;
        // else: center is in GPU0's domain, GPU1 stays silent
    }

    // ── Allocate states ───────────────────────────────────────────────────────
    HIP_CHECK(hipSetDevice(0));
    auto state0 = std::make_unique<HeavyHexState>(static_cast<Grid&>(*grid0));
    state0->init_heisenberg(static_cast<Grid&>(*grid0), std::vector<int>{center0});

    HIP_CHECK(hipSetDevice(1));
    auto state1 = std::make_unique<HeavyHexState>(static_cast<Grid&>(*grid1));
    if (center1 >= 0)
        state1->init_heisenberg(static_cast<Grid&>(*grid1), std::vector<int>{center1});
    else
        state1->init_heisenberg(static_cast<Grid&>(*grid1), std::vector<int>{});  // silent: all identity

    // ── TrotterColorCtx + streams for parallel execution ─────────────────────
    HIP_CHECK(hipSetDevice(0));
    auto ctx0 = std::make_unique<TrotterColorCtx>(*grid0, chi_max);
    hipStream_t stream0; HIP_CHECK(hipStreamCreate(&stream0));

    HIP_CHECK(hipSetDevice(1));
    auto ctx1 = std::make_unique<TrotterColorCtx>(*grid1, chi_max);
    hipStream_t stream1; HIP_CHECK(hipStreamCreate(&stream1));

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

        // Colors before cut: both GPUs run in parallel threads
        for (int c = 0; c < cut_color; c++) {
            std::thread t0([&,c](){
                hipSetDevice(0);
                do_trotter_one_color(*ctx0, c,
                    state0->psi, state0->messages, *grid0,
                    ptm_gpu0, nullptr, gops0, *ws0, chi_max, cutoff, stream0);
                hipStreamSynchronize(stream0);
            });
            std::thread t1([&,c](){
                hipSetDevice(1);
                do_trotter_one_color(*ctx1, c,
                    state1->psi, state1->messages, *grid1,
                    ptm_gpu1, nullptr, gops1, *ws1, chi_max, cutoff, stream1);
                hipStreamSynchronize(stream1);
            });
            t0.join(); t1.join();
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

        // Cut color: both GPUs compute simultaneously in parallel threads
        {
            std::thread t0([&](){
                hipSetDevice(0);
                do_trotter_one_color(*ctx0, cut_color,
                    state0->psi, state0->messages, *grid0,
                    ptm_gpu0, nullptr, gops0, *ws0, chi_max, cutoff, stream0);
                hipStreamSynchronize(stream0);
            });
            std::thread t1([&](){
                hipSetDevice(1);
                do_trotter_one_color(*ctx1, cut_color,
                    state1->psi, state1->messages, *grid1,
                    ptm_gpu1, nullptr, gops1, *ws1, chi_max, cutoff, stream1);
                hipStreamSynchronize(stream1);
            });
            t0.join(); t1.join();
        }

        // Colors after cut: both GPUs run in parallel threads
        for (int c = cut_color + 1; c < (int)grid0->colors.size(); c++) {
            std::thread t0([&,c](){
                hipSetDevice(0);
                do_trotter_one_color(*ctx0, c,
                    state0->psi, state0->messages, *grid0,
                    ptm_gpu0, nullptr, gops0, *ws0, chi_max, cutoff, stream0);
                hipStreamSynchronize(stream0);
            });
            std::thread t1([&,c](){
                hipSetDevice(1);
                do_trotter_one_color(*ctx1, c,
                    state1->psi, state1->messages, *grid1,
                    ptm_gpu1, nullptr, gops1, *ws1, chi_max, cutoff, stream1);
                hipStreamSynchronize(stream1);
            });
            t0.join(); t1.join();
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

        // ── BP: local convergence in parallel, then boundary exchange ────────────
        int bp0_sweeps = 0, bp1_sweeps = 0;
        float bp0_delta = 0.f, bp1_delta = 0.f;

        // Step 1: local BP on each GPU in parallel threads
        {
            std::thread t0([&](){
                hipSetDevice(0);
                meas0->autocorrelation_with_bp(*grid0, state0->psi, state0->messages,
                    center0, bp_max, nullptr, nullptr, &bp0_sweeps, &bp0_delta,
                    nullptr, nullptr);
            });
            std::thread t1([&](){
                hipSetDevice(1);
                meas1->autocorrelation_with_bp(*grid1, state1->psi, state1->messages,
                    center1 >= 0 ? center1 : 0, bp_max,
                    nullptr, nullptr, &bp1_sweeps, &bp1_delta, nullptr, nullptr);
            });
            t0.join(); t1.join();
        }

        // Step 2: P2P exchange boundary bond messages
        // Boundary bonds: grid0 vertical bonds at row7→row8 (bond ids 279+7*32..279+7*32+31)
        // Matching bonds in grid1: row0→row1 (bond ids 279+0*32..279+31)
        // Exchange: GPU0 msg_from_a (row8→row7 direction) ↔ GPU1 msg_from_b (row1→row0)
        {
            const int n_boundary = RECT_9X32_NX;  // 32 boundary bonds
            const int bond_offset0 = 279 + 7 * RECT_9X32_NX;  // row7→row8 in grid0
            const int bond_offset1 = 279;                       // row0→row1 in grid1

            HIP_CHECK(hipSetDevice(0));
            for (int col = 0; col < n_boundary; col++) {
                const Bond& b0 = grid0->bonds[bond_offset0 + col];
                const Bond& b1 = grid1->bonds[bond_offset1 + col];
                if (b0.chi_alloc == 1) continue;  // outside lightcone, skip
                size_t sz = (size_t)b0.chi * b0.chi * sizeof(Scalar);
                // GPU0 sends outward msg (row8 side) → GPU1 inward msg (row0 side)
                HIP_CHECK(hipMemcpyPeer(
                    state1->messages[b1.msg_from_a()], 1,
                    state0->messages[b0.msg_from_b()], 0, sz));
                // GPU1 sends outward msg (row1 side) → GPU0 inward msg (row8 side)
                HIP_CHECK(hipMemcpyPeer(
                    state0->messages[b0.msg_from_b()], 0,
                    state1->messages[b1.msg_from_a()], 1, sz));
            }
            HIP_CHECK(hipDeviceSynchronize());
        }

        // Step 3: one additional BP sweep on each GPU to propagate boundary info
        {
            std::thread t0([&](){ hipSetDevice(0); meas0->bp_converge(*grid0, state0->psi, state0->messages, 1, center0); });
            std::thread t1([&](){ hipSetDevice(1); meas1->bp_converge(*grid1, state1->psi, state1->messages, 1, center1 >= 0 ? center1 : 0); });
            t0.join(); t1.join();
        }

        // Actual chi
        int chi_actual = 1;
        for (auto& b : grid0->bonds) chi_actual = std::max(chi_actual, b.chi);

        double t = (now_ms() - t0) / 1000.0;
        printf("STEP %d  chi %d  bp0 %d/%.2e  bp1 %d/%.2e  t %.2f\n",
               step, chi_actual, bp0_sweeps, (double)bp0_delta,
               bp1_sweeps, (double)bp1_delta, t);

        // Pauli weights — merge both halves
        std::vector<float> nid0(grid0->n_sites, 0.f);
        std::vector<float> nid1(grid1->n_sites, 0.f);

        HIP_CHECK(hipSetDevice(0));
        meas0->compute_pauli_profile(*grid0, state0->psi, nid0.data());
        HIP_CHECK(hipSetDevice(1));
        meas1->compute_pauli_profile(*grid1, state1->psi, nid1.data());

        // Merged output: GPU0 owns rows 0-7 (sites 0-255), GPU1 owns rows 1-8 (=16×32 rows 9-15)
        // GPU0 row 8 (sites 256-287) = overlap = use GPU0's value
        printf("PAULI_NL %d", step);
        // GPU 0: rows 0-8 (288 sites) → 16×32 sites 0-287
        for (int i = 0; i < grid0->n_sites; i++) printf(" %.7f", nid0[i]);
        // GPU 1: rows 1-8 (256 sites) → 16×32 sites 288-511 (skip row 0 = ghost)
        for (int i = 32; i < grid1->n_sites; i++) printf(" %.7f", nid1[i]);
        printf("\n");
    }

    return 0;
}
