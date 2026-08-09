/*
 * test_rect7x7_trotter.cpp — Diagnose Trotter crash on rect_7x7.
 *
 * Runs one color class at a time with verbose output to pinpoint failure.
 *
 * Build: make PHYS_DIM=4 test_rect7x7_trotter
 * Run:   ./test/test_rect7x7_trotter
 */

#include "types.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"
#include "rect_7x7.hpp"
#include "rect_7x7_ideal_kick_eps010_data.hpp"

#include <cstdio>
#include <vector>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#define HIP_CHECK(cmd) do { \
    hipError_t e=(cmd); \
    if(e!=hipSuccess){fprintf(stderr,"HIP %s at %s:%d\n",hipGetErrorString(e),__FILE__,__LINE__);exit(1);} \
} while(0)

int main() {
    int chi = 20;
    float cutoff = 1e-3f;

    printf("Building grid...\n"); fflush(stdout);
    FlatRect7x7Grid grid(chi);

    int max_deg = 0;
    for (auto& s : grid.sites) {
        int d=0; for(int k=0;k<MAX_DEGREE;k++) if(s.bond_id[k]>=0) d++;
        max_deg = std::max(max_deg, d);
    }
    printf("max_deg=%d\n", max_deg); fflush(stdout);

    Workspace ws(chi, max_deg);
    rocblas_handle h; rocblas_create_handle(&h);

    printf("Allocating state...\n"); fflush(stdout);
    HeavyHexState state(static_cast<Grid&>(grid));
    state.init_heisenberg(static_cast<Grid&>(grid), 24);
    GateOps gate_ops(h);

    printf("Allocating message buffers (nm=%d)...\n", 2*grid.n_bonds); fflush(stdout);
    int nm = 2*grid.n_bonds;
    std::vector<int>     frozen_chi(grid.n_bonds);
    std::vector<Scalar*> frozen_msg(nm, nullptr), new_msg(nm, nullptr);
    size_t alloc = (size_t)chi*chi*sizeof(Scalar);

    for (auto& bond : grid.bonds) for (int d = 0; d < 2; d++) {
        int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
        HIP_CHECK(hipMalloc(&frozen_msg[idx], alloc));
        HIP_CHECK(hipMalloc(&new_msg[idx],    alloc));
    }
    printf("Buffers allocated OK\n"); fflush(stdout);

    // Run one color at a time
    for (int c = 0; c < (int)grid.colors.size(); c++) {
        printf("\nColor %d (%zu gates)...\n", c, grid.colors[c].size()); fflush(stdout);

        // Snapshot
        for (auto& b : grid.bonds) frozen_chi[b.id] = b.chi;
        for (auto& bond : grid.bonds) {
            int chi2 = bond.chi; size_t sz = (size_t)chi2*chi2*sizeof(Scalar);
            for (int d = 0; d < 2; d++) {
                int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                HIP_CHECK(hipMemcpy(frozen_msg[idx], state.messages[idx], sz, hipMemcpyDeviceToDevice));
                HIP_CHECK(hipMemcpy(new_msg[idx],    state.messages[idx], sz, hipMemcpyDeviceToDevice));
            }
        }
        printf("  Snapshot OK\n"); fflush(stdout);

        // Apply gates
        int gi = 0;
        for (auto& gate : grid.colors[c]) {
            printf("  gate %d: bond=%d sites=(%d,%d)\n", gi++, gate.bond_id, gate.site_a, gate.site_b);
            fflush(stdout);
            const float* G = RECT_7x7_IDEAL_KICK_EPS010[gate.bond_id];
            gate_ops.apply_binary_gate(
                state.psi[gate.site_a], state.psi[gate.site_b],
                gate, grid, new_msg, frozen_msg, frozen_chi,
                G, ws, chi, cutoff);
        }
        printf("  All gates OK\n"); fflush(stdout);

        // Commit
        for (auto& gate : grid.colors[c]) {
            const Bond& bond = grid.bonds[gate.bond_id];
            int chi2 = bond.chi; size_t sz = (size_t)chi2*chi2*sizeof(Scalar);
            for (int d = 0; d < 2; d++) {
                int idx = (d==0) ? bond.msg_from_a() : bond.msg_from_b();
                HIP_CHECK(hipMemcpy(state.messages[idx], new_msg[idx], sz, hipMemcpyDeviceToDevice));
            }
        }
        printf("  Commit OK\n"); fflush(stdout);
    }

    printf("\nAll colors done. Freeing...\n"); fflush(stdout);
    for (auto p : frozen_msg) if (p) HIP_CHECK(hipFree(p));
    for (auto p : new_msg)    if (p) HIP_CHECK(hipFree(p));
    rocblas_destroy_handle(h);
    printf("ALL PASS\n");
    return 0;
}
