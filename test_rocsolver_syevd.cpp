/*
 * test_rocsolver_syevd.cpp — test rocsolver_ssyevd on a small symmetric matrix
 *
 * Build: hipcc -O2 -o test_rocsolver_syevd test_rocsolver_syevd.cpp -lrocblas -lrocsolver
 * Run:   ./test_rocsolver_syevd
 */
#include <cstdio>
#include <cmath>
#include <vector>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#define HIP_CHECK(cmd) do { \
    hipError_t e=(cmd); \
    if(e!=hipSuccess){fprintf(stderr,"HIP error %s at line %d\n",hipGetErrorString(e),__LINE__);exit(1);} \
} while(0)

int main() {
    rocblas_handle handle;
    rocblas_create_handle(&handle);

    // Test 1: 3×3 identity-like symmetric matrix
    // M = [[2,1,0],[1,2,1],[0,1,2]] → eigenvalues: 2-sqrt(2), 2, 2+sqrt(2)
    int n = 3;
    int lda = 3;

    // Col-major layout
    std::vector<float> h_A = {
        2.f, 1.f, 0.f,   // col 0
        1.f, 2.f, 1.f,   // col 1
        0.f, 1.f, 2.f    // col 2
    };
    std::vector<float> h_D(n, 0.f);
    std::vector<float> h_E(n, 0.f);
    int h_info = 0;

    float* d_A; HIP_CHECK(hipMalloc(&d_A, n*n*sizeof(float)));
    float* d_D; HIP_CHECK(hipMalloc(&d_D, n*sizeof(float)));
    float* d_E; HIP_CHECK(hipMalloc(&d_E, n*sizeof(float)));
    int*   d_info; HIP_CHECK(hipMalloc(&d_info, sizeof(int)));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), n*n*sizeof(float), hipMemcpyHostToDevice));

    rocblas_status st = rocsolver_ssyevd(
        handle,
        rocblas_evect_original,
        rocblas_fill_upper,
        n, d_A, lda, d_D, d_E, d_info);

    if (st != rocblas_status_success) {
        fprintf(stderr, "rocsolver_ssyevd failed: %d\n", (int)st);
        return 1;
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_D.data(), d_D, n*sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost));

    printf("info = %d\n", h_info);
    printf("Eigenvalues: %.6f  %.6f  %.6f\n", h_D[0], h_D[1], h_D[2]);
    printf("Expected:    %.6f  %.6f  %.6f\n",
           2.f - sqrtf(2.f), 2.f, 2.f + sqrtf(2.f));

    float err = 0.f;
    err = std::max(err, std::abs(h_D[0] - (2.f - sqrtf(2.f))));
    err = std::max(err, std::abs(h_D[1] - 2.f));
    err = std::max(err, std::abs(h_D[2] - (2.f + sqrtf(2.f))));
    printf("Max error: %.2e  %s\n", err, err < 1e-5f ? "PASS" : "FAIL");

    // Test 2: 1×1 matrix (chi=1 case that crashed)
    printf("\nTest chi=1:\n");
    float h_A1 = 0.333f;
    float h_D1 = 0.f, h_E1 = 0.f;
    int h_info1 = 0;
    float* d_A1; HIP_CHECK(hipMalloc(&d_A1, sizeof(float)));
    float* d_D1; HIP_CHECK(hipMalloc(&d_D1, sizeof(float)));
    float* d_E1; HIP_CHECK(hipMalloc(&d_E1, sizeof(float)));
    int*  d_i1;  HIP_CHECK(hipMalloc(&d_i1, sizeof(int)));
    HIP_CHECK(hipMemcpy(d_A1, &h_A1, sizeof(float), hipMemcpyHostToDevice));

    st = rocsolver_ssyevd(handle, rocblas_evect_original, rocblas_fill_upper,
                          1, d_A1, 1, d_D1, d_E1, d_i1);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(&h_D1, d_D1, sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&h_info1, d_i1, sizeof(int), hipMemcpyDeviceToHost));
    printf("info=%d  eigenvalue=%.6f  expected=%.6f  %s\n",
           h_info1, h_D1, h_A1, std::abs(h_D1-h_A1)<1e-6f?"PASS":"FAIL");

    rocblas_destroy_handle(handle);
    printf("\nAll tests done.\n");
    return 0;
}
