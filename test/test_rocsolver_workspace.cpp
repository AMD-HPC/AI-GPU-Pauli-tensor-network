/*
 * test_rocsolver_workspace.cpp — Query actual rocsolver workspace requirements
 * at various chi values for the operations used in gate.hpp.
 *
 * Build:
 *   hipcc -O2 -std=c++17 -I./include -DPHYS_DIM=4 \
 *       -o test/test_rocsolver_workspace test/test_rocsolver_workspace.cpp \
 *       -lrocblas -lrocsolver -lm
 * Run:
 *   ./test/test_rocsolver_workspace
 */

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#include <cstdio>
#include <cstdlib>

#define D 4

int main() {
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Query rocBLAS+rocsolver shared workspace size
    size_t default_ws = 0;
    rocblas_get_device_memory_size(handle, &default_ws);
    printf("Default rocBLAS/rocsolver workspace: %.1f MB\n\n", default_ws/1e6);

    printf("%-6s  %-10s  %-10s  %-10s  %-10s  %-10s\n",
           "chi", "m_max", "n_max", "sgeqrf", "sorgqr", "sgesvd");
    printf("%s\n", std::string(70,'-').c_str());

    for (int chi : {50, 100, 150, 200, 250, 300, 350, 400}) {
        int max_deg = 3;  // heavy-hex
        int m_max = chi * chi + 1;  // chi^(max_degree-1) + 1
        int n_max = chi * D;

        // Allocate dummy GPU buffers for size queries
        float *A, *tau, *U, *S, *V;
        int *info;
        hipMalloc(&A,   (size_t)m_max * n_max * sizeof(float));
        hipMalloc(&tau, (size_t)n_max * sizeof(float));
        hipMalloc(&U,   (size_t)m_max * n_max * sizeof(float));
        hipMalloc(&S,   (size_t)n_max * sizeof(float));
        hipMalloc(&V,   (size_t)n_max * n_max * sizeof(float));
        hipMalloc(&info, sizeof(int));

        // Query sgeqrf workspace
        size_t ws_qrf = 0;
        rocblas_start_device_memory_size_query(handle);
        rocsolver_sgeqrf(handle, m_max, n_max, A, m_max, tau);
        rocblas_stop_device_memory_size_query(handle, &ws_qrf);

        // Query sorgqr workspace
        size_t ws_qr = 0;
        rocblas_start_device_memory_size_query(handle);
        rocsolver_sorgqr(handle, m_max, n_max, n_max, A, m_max, tau);
        rocblas_stop_device_memory_size_query(handle, &ws_qr);

        // Query sgesvd workspace
        size_t ws_svd = 0;
        rocblas_start_device_memory_size_query(handle);
        rocsolver_sgesvd(handle, rocblas_svect_singular, rocblas_svect_singular,
                         n_max, n_max, A, n_max, S, U, n_max, V, n_max, tau,
                         rocblas_outofplace, info);
        rocblas_stop_device_memory_size_query(handle, &ws_svd);

        size_t total = std::max({ws_qrf, ws_qr, ws_svd});

        printf("%-6d  %-10d  %-10d  %-8.1fMB  %-8.1fMB  %-8.1fMB  max=%.1fMB\n",
               chi, m_max, n_max,
               ws_qrf/1e6, ws_qr/1e6, ws_svd/1e6, total/1e6);

        hipFree(A); hipFree(tau); hipFree(U); hipFree(S); hipFree(V); hipFree(info);
    }

    rocblas_destroy_handle(handle);
    return 0;
}
