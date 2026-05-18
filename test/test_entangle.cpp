/*
 * tests/test_entangle.cpp
 *
 * Minimal entanglement test: 2 sites, 1 bond.
 * Site 0 = Z operator [0,0,0,1]
 * Site 1 = I operator [1,0,0,0]
 *
 * After Rxx(θ): ZI → cos(θ)·ZI - sin(θ)·YX
 * This mixes two Pauli components → rank-2 bond → chi grows to 2.
 *
 * We print EVERY intermediate quantity so we can find the bug.
 *
 * Build:
 *   hipcc -O2 -std=c++17 -DPHYS_DIM=4 -I../include \
 *         -o tests/test_entangle tests/test_entangle.cpp -lrocblas -lrocsolver -lm
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "physics.hpp"
#include "measures.hpp"
#include <cstdio>
#include <cmath>
#include <vector>

static rocblas_handle g_handle;

static std::vector<float> dl(const float* d, int n) {
    std::vector<float> h(n);
    HIP_CHECK(hipMemcpy(h.data(), d, n*sizeof(float), hipMemcpyDeviceToHost));
    return h;
}

static void print_vec(const char* name, const float* d, int n) {
    auto h = dl(d, n);
    printf("  %s = [", name);
    for(int i=0;i<n;i++) printf("%.4f%s", h[i], i<n-1?", ":"");
    printf("]\n");
}

static void print_mat(const char* name, const float* d, int rows, int cols) {
    auto h = dl(d, rows*cols);
    printf("  %s [%d×%d]:\n", name, rows, cols);
    for(int i=0;i<rows;i++){
        printf("    [");
        for(int j=0;j<cols;j++) printf("%7.4f", h[i+rows*j]);
        printf(" ]\n");
    }
}

int main() {
    printf("=== test_entangle ===\n");
    printf("D=%d\n\n", D);

    rocblas_create_handle(&g_handle);

    // 1×2 grid: 2 sites, 1 horizontal bond
    // Use chi_max=4 so bond can grow
    int chi_max = 4;
    Grid g(1, 2, chi_max);
    Workspace ws(chi_max);
    GateOps ops(g_handle);

    // Allocate site tensors at chi_max size
    const Site& s0 = g.site(0,0);
    const Site& s1 = g.site(0,1);
    size_t sz0 = (size_t)s0.tensor_size_max(chi_max)*sizeof(float);
    size_t sz1 = (size_t)s1.tensor_size_max(chi_max)*sizeof(float);
    float *psi0, *psi1;
    HIP_CHECK(hipMalloc(&psi0, sz0)); HIP_CHECK(hipMemset(psi0, 0, sz0));
    HIP_CHECK(hipMalloc(&psi1, sz1)); HIP_CHECK(hipMemset(psi1, 0, sz1));

    // 1 bond, allocate message at chi_max size
    float *msg0;
    HIP_CHECK(hipMalloc(&msg0, chi_max*chi_max*sizeof(float)));
    HIP_CHECK(hipMemset(msg0, 0, chi_max*chi_max*sizeof(float)));

    // Initialize: Z at site 0, I at site 1
    std::vector<float> h_Z = {0.f,0.f,0.f,1.f};
    std::vector<float> h_I = {1.f,0.f,0.f,0.f};
    HIP_CHECK(hipMemcpy(psi0, h_Z.data(), D*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(psi1, h_I.data(), D*sizeof(float), hipMemcpyHostToDevice));
    float one=1.f;
    HIP_CHECK(hipMemcpy(msg0, &one, sizeof(float), hipMemcpyHostToDevice));
    g.bonds[0].chi = 1;

    std::vector<float*> msgs = {msg0};

    printf("--- Initial state ---\n");
    print_vec("psi0 (Z)", psi0, D);
    print_vec("psi1 (I)", psi1, D);
    printf("  bond chi = %d\n\n", g.bonds[0].chi);

    // Gate: Rxx(θ) with θ = 2*J*dt = 2*(-1)*0.1 = -0.2
    float theta = -0.2f;
    auto G = IsingGates::Rxx(theta);
    printf("--- Rxx(%.3f) gate ---\n", theta);
    printf("  Expected: ZI → cos(%.3f)·ZI - sin(%.3f)·YX\n",theta,theta);
    printf("  cos=%.4f  -sin=%.4f\n\n", std::cos(theta), -std::sin(theta));

    // ── STEP BY STEP through apply_binary_gate ────────────────────

    // Step 1: pseudo_sqrt of message (msg=1 at chi=1)
    printf("--- Step 1: pseudo_sqrt ---\n");
    // No env bonds on 1×2 grid — site 0 only has RIGHT bond (the cut bond)
    // So there are NO env bonds to compute sqrt for!
    printf("  No env bonds (1×2 grid, only bond is the cut bond)\n\n");

    // Step 2: absorb_env → cv1 = psi0, cv2 = psi1 (no env to absorb)
    printf("--- Step 2: absorb_env ---\n");
    // Manually copy since no env
    HIP_CHECK(hipMemcpy(ws.cv1, psi0, D*sizeof(float), hipMemcpyDeviceToDevice));
    HIP_CHECK(hipMemcpy(ws.cv2, psi1, D*sizeof(float), hipMemcpyDeviceToDevice));
    print_vec("cv1 (=psi0)", ws.cv1, D);
    print_vec("cv2 (=psi1)", ws.cv2, D);
    printf("\n");

    // Step 3: QR of cv1 and cv2
    // m1=1, n1=chi_cut*D=1*4=4  → fat matrix [1×4]
    // k1=min(1,4)=1 → Q1=[1×1], R1=[1×4]
    int m1=1, n1=D, k1=1;
    int m2=1, n2=D, k2=1;
    printf("--- Step 3: CholeskyQR ---\n");
    printf("  cv1: m=%d n=%d k=%d\n", m1,n1,k1);

    bool ok1 = false;
    if(m1 >= 10*n1) ok1 = ops.cholesky_qr(ws.cv1, m1, n1,
                                ws.Q1, ws.R1, ws.Rinv1, ws.gram, ws.potrf_info);
    printf("  using %s for cv1\n", ok1?"CholeskyQR":"Householder");
    if(!ok1) ops.householder_qr(ws.cv1, m1, n1, ws.Q1, ws.R1);
    print_vec("Q1", ws.Q1, m1*k1);
    print_vec("R1", ws.R1, k1*n1);

    bool ok2 = false;
    if(m2 >= 10*n2) ok2 = ops.cholesky_qr(ws.cv2, m2, n2,
                                ws.Q2, ws.R2, ws.Rinv2, ws.gram, ws.potrf_info);
    printf("  using %s for cv2\n", ok2?"CholeskyQR":"Householder");
    if(!ok2) ops.householder_qr(ws.cv2, m2, n2, ws.Q2, ws.R2);
    print_vec("Q2", ws.Q2, m2*k2);
    print_vec("R2", ws.R2, k2*n2);
    printf("\n");

    // Step 4: RR = R1ᵀ · R2
    printf("--- Step 4: RR = R1ᵀ · R2 ---\n");
    // Direct rocBLAS call — bypass gemm helper to verify
    {
        // Print raw memory of R1 and R2 before GEMM
        print_vec("R1 raw memory", ws.R1, k1*n1);
        print_vec("R2 raw memory", ws.R2, k2*n2);
        Scalar alpha=1.f, beta=0.f;
        rocblas_status st = rocblas_sgemm(g_handle,
            rocblas_operation_transpose,
            rocblas_operation_none,
            n1, n2, k1,
            &alpha,
            ws.R1, k1,
            ws.R2, k2,
            &beta,
            ws.RR, n1);
        HIP_CHECK(hipDeviceSynchronize());
        printf("  rocblas_sgemm status = %d (0=success)\n", (int)st);
    }
    print_mat("RR", ws.RR, n1, n2);
    printf("\n");

    // Step 5: apply gate
    printf("--- Step 5: apply Rxx gate to RR ---\n");
    ops.apply_gate_to_RR(ws.RR, n1, n2, G.data(), ws.RR);
    print_mat("oR", ws.RR, n1, n2);
    printf("\n");

    // Step 6: SVD
    printf("--- Step 6: SVD ---\n");
    int rank = ops.truncated_svd(ws.RR, n1, n2,
                                 ws.U, ws.S, ws.Vt,
                                 ws.svd_work, ws.svd_info,
                                 chi_max, 1e-10f);
    printf("  rank = %d\n", rank);
    print_vec("S", ws.S, rank);
    print_mat("U",  ws.U,  n1, rank);
    print_mat("Vt", ws.Vt, rank, n2);
    printf("\n");

    // Step 7: reconstruct
    printf("--- Step 7: reconstruct ---\n");
    ops.gemm(ws.Q1, m1, k1, ws.U,  rank, ws.cv1);
    ops.gemm(ws.Q2, m2, k2, ws.Vt, rank, ws.cv2,
             rocblas_operation_none, rocblas_operation_transpose);
    print_vec("new_psi0", ws.cv1, m1*rank);
    print_vec("new_psi1", ws.cv2, m2*rank);
    printf("  new bond chi = %d\n\n", rank);

    // Expected result:
    // oR should have ZI→cos(θ)·ZI + sin(θ)·... (see Rxx superoperator)
    // With ZI at [3,0] and YX at [2,1]:
    // oR[3,0] = cos(θ) ≈ 0.9800
    // oR[2,1] = -sin(θ) ≈ 0.1987  (from Rxx superoperator)
    printf("--- Expected ---\n");
    printf("  oR[3,0] = cos(%.3f) = %.4f  (ZI component)\n",
           theta, std::cos(theta));
    printf("  oR[2,1] = -sin(%.3f) = %.4f  (YX component)\n",
           theta, std::sin(theta));
    printf("  → SVD should give rank=2, S≈[%.4f, %.4f]\n",
           std::abs(std::cos(theta)), std::abs(std::sin(theta)));

    HIP_CHECK(hipFree(psi0)); HIP_CHECK(hipFree(psi1));
    HIP_CHECK(hipFree(msg0));
    rocblas_destroy_handle(g_handle);
    printf("\n=== test_entangle DONE ===\n");
    return 0;
}
