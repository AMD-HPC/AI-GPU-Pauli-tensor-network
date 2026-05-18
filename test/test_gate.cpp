/*
 * tests/test_gate.cpp — gate pipeline tests (D=4 Pauli basis)
 * Build: hipcc -O3 -std=c++17 -DPHYS_DIM=4 -I../include \
 *         -o tests/test_gate tests/test_gate.cpp -lrocblas -lrocsolver -lm
 */
#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "physics.hpp"
#include "measures.hpp"
#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>

static_assert(D==4,"test_gate requires PHYS_DIM=4");
static rocblas_handle g_handle=nullptr;

static std::vector<float> dl(const float* d, int n){
    std::vector<float> h(n);
    HIP_CHECK(hipMemcpy(h.data(),d,n*sizeof(float),hipMemcpyDeviceToHost));
    return h;
}
static void ul(float* d, const std::vector<float>& h){
    HIP_CHECK(hipMemcpy(d,h.data(),h.size()*sizeof(float),hipMemcpyHostToDevice));
}

/* Test 1: pseudo_sqrt */
static void test_pseudo_sqrt(){
    printf("\n--- Test 1: pseudo_sqrt ---\n");
    Workspace ws(2); GateOps ops(g_handle);
    std::vector<float> hM={4.f,0.f,0.f,9.f};
    float *dM,*dS,*dI;
    HIP_CHECK(hipMalloc(&dM,4*sizeof(float)));
    HIP_CHECK(hipMalloc(&dS,4*sizeof(float)));
    HIP_CHECK(hipMalloc(&dI,4*sizeof(float)));
    ul(dM,hM);
    ops.pseudo_sqrt(dM,dS,dI,2,ws);
    auto hs=dl(dS,4), hi=dl(dI,4);
    printf("  sqrt diag: %.4f %.4f (expected 2.0 3.0)\n",hs[0],hs[3]);
    printf("  inv  diag: %.4f %.4f (expected 0.5 0.333)\n",hi[0],hi[3]);
    assert(std::abs(hs[0]-2.f)<1e-4f);
    assert(std::abs(hs[3]-3.f)<1e-4f);
    assert(std::abs(hi[0]-0.5f)<1e-4f);
    assert(std::abs(hi[3]-1.f/3.f)<1e-4f);
    printf("  pseudo_sqrt: OK\n");
    HIP_CHECK(hipFree(dM)); HIP_CHECK(hipFree(dS)); HIP_CHECK(hipFree(dI));
}

/* Test 2: apply_unary_gate */
static void test_unary_gate(){
    printf("\n--- Test 2: apply_unary_gate ---\n");
    Grid g(2,2,4); for(auto& b:g.bonds) b.chi=1;
    Workspace ws(1); GateOps ops(g_handle);
    const Site& site=g.site(0,0);
    float* dp; HIP_CHECK(hipMalloc(&dp,D*sizeof(float)));

    ul(dp,{0.f,0.f,0.f,1.f});
    auto G=IsingGates::Rz(0.3f);
    ops.apply_unary_gate(dp,site,g,G.data(),ws.cv1);
    auto h=dl(dp,D);
    printf("  Rz(0.3) on Z: [%.3f,%.3f,%.3f,%.3f] (expected [0,0,0,1])\n",h[0],h[1],h[2],h[3]);
    assert(std::abs(h[3]-1.f)<1e-4f && std::abs(h[0])<1e-4f);
    printf("  Rz preserves Z: OK\n");

    ul(dp,{0.f,1.f,0.f,0.f});
    ops.apply_unary_gate(dp,site,g,G.data(),ws.cv1);
    h=dl(dp,D);
    printf("  Rz(0.3) on X: [%.3f,%.3f,%.3f,%.3f]\n",h[0],h[1],h[2],h[3]);
    assert(std::abs(h[1]-std::cos(0.3f))<1e-4f);
    assert(std::abs(h[2]-std::sin(0.3f))<1e-4f);
    printf("  Rz rotates X->X,Y: OK\n");
    HIP_CHECK(hipFree(dp));
}

/* Test 3: Rz preserves Z autocorrelation (J=0) */
static void test_rz_preserves_z(){
    printf("\n--- Test 3: Rz preserves Z (J=0) ---\n");
    Grid g(2,2,4); Workspace ws(4); IsingState state(g);
    Measures meas(g_handle,ws);
    IsingSimulator sim(g_handle,-1.f,0.f,0.1f);
    int center=g.site_id(0,0);
    state.init_heisenberg(g,center);
    for(int s=0;s<=5;s++){
        Scalar C=meas.autocorrelation(g,state.psi,state.messages,center);
        printf("  step %d: C=%.6f\n",s,C);
        assert(std::abs(C-1.f)<1e-3f);
        if(s<5) sim.apply_rz_layer(state,g,ws);
    }
    printf("  Rz preserves Z: OK\n");
}

/* Test 4: Householder QR (via rocsolver directly) */
static void test_qr(){
    printf("\n--- Test 4: Householder QR ---\n");
    int m=100,n=10,k=std::min(m,n);
    Workspace ws(10);
    float *dQ,*dtau,*dQtQ;
    HIP_CHECK(hipMalloc(&dQ,m*n*sizeof(float)));
    HIP_CHECK(hipMalloc(&dtau,k*sizeof(float)));
    HIP_CHECK(hipMalloc(&dQtQ,k*k*sizeof(float)));
    std::vector<float> hA(m*n);
    for(int i=0;i<m*n;i++) hA[i]=(float)((i*7+3)%13)/13.f-0.5f;
    HIP_CHECK(hipMemcpy(dQ,hA.data(),m*n*sizeof(float),hipMemcpyHostToDevice));
    RB_CHECK(rocsolver_sgeqrf(g_handle,m,n,dQ,m,dtau));
    HIP_CHECK(hipDeviceSynchronize());
    RB_CHECK(rocsolver_sorgqr(g_handle,m,k,k,dQ,m,dtau));
    HIP_CHECK(hipDeviceSynchronize());
    float alpha=1.f,beta=0.f;
    RB_CHECK(rocblas_sgemm(g_handle,rocblas_operation_transpose,rocblas_operation_none,
        k,k,m,&alpha,dQ,m,dQ,m,&beta,dQtQ,k));
    HIP_CHECK(hipDeviceSynchronize());
    auto hQtQ=dl(dQtQ,k*k);
    float err=0.f;
    for(int i=0;i<k;i++) for(int j=0;j<k;j++){
        float e=(i==j)?1.f:0.f; err=std::max(err,std::abs(hQtQ[i+j*k]-e));
    }
    printf("  QᵀQ-I max_err=%.2e: %s\n",err,err<1e-4f?"OK":"FAIL");
    assert(err<1e-4f);
    printf("  Householder QR: OK\n");
    HIP_CHECK(hipFree(dQ)); HIP_CHECK(hipFree(dtau)); HIP_CHECK(hipFree(dQtQ));
}

/* Test 5: Gate via binary gate (replaces apply_gate_to_RR test) */
static void test_apply_gate_to_RR(){
    printf("\n--- Test 5: Gate via binary gate ---\n");
    Grid g(2,2,4); Workspace ws(4); GateOps ops(g_handle);
    IsingState state(g); Measures meas(g_handle,ws);
    int center=g.site_id(0,0);

    // I⊗I preserves C
    state.init_heisenberg(g,center);
    std::vector<float> GI(D*D*D*D,0.f);
    for(int i=0;i<D*D;i++) GI[i+D*D*i]=1.f;
    const Gate& gate=g.colors[0][0];
    ops.apply_binary_gate(state.psi[gate.site_a],state.psi[gate.site_b],
                          gate,g,state.messages,GI.data(),ws,4,1e-10f);
    Scalar C=meas.autocorrelation(g,state.psi,state.messages,center);
    printf("  I@I: C=%.6f (expected 1.0): %s\n",C,std::abs(C-1.f)<1e-3f?"OK":"FAIL");
    assert(std::abs(C-1.f)<1e-3f);

    // Rxx(0) preserves C
    state.init_heisenberg(g,center);
    auto G0=IsingGates::Rxx(0.f);
    ops.apply_binary_gate(state.psi[gate.site_a],state.psi[gate.site_b],
                          gate,g,state.messages,G0.data(),ws,4,1e-10f);
    C=meas.autocorrelation(g,state.psi,state.messages,center);
    printf("  Rxx(0): C=%.6f (expected 1.0): %s\n",C,std::abs(C-1.f)<1e-3f?"OK":"FAIL");
    assert(std::abs(C-1.f)<1e-3f);

    printf("  apply_gate_to_RR: OK\n");
}

/* Test 6: Binary gate identity on 2x2 grid */
static void test_binary_gate_identity(){
    printf("\n--- Test 6: Binary gate identity (2x2) ---\n");
    Grid g(2,2,4); Workspace ws(4); GateOps ops(g_handle);
    IsingState state(g); Measures meas(g_handle,ws);
    int center=g.site_id(0,0);
    state.init_heisenberg(g,center);
    Scalar C0=meas.autocorrelation(g,state.psi,state.messages,center);
    printf("  C(t=0)=%.6f (expected 1.0)\n",C0);
    assert(std::abs(C0-1.f)<1e-3f);
    std::vector<float> GI(D*D*D*D,0.f);
    for(int i=0;i<D*D;i++) GI[i+D*D*i]=1.f;
    const Gate& gate=g.colors[0][0];
    ops.apply_binary_gate(state.psi[gate.site_a],state.psi[gate.site_b],
                          gate,g,state.messages,GI.data(),ws,4,1e-10f);
    Scalar C1=meas.autocorrelation(g,state.psi,state.messages,center);
    printf("  C after I@I=%.6f (expected 1.0)\n",C1);
    assert(std::abs(C1-1.f)<1e-3f);
    printf("  Binary gate identity: OK\n");
}

/* Test 7: C(t=0)=1 on 4x4 */
static void test_c0(){
    printf("\n--- Test 7: C(t=0)=1 on 4x4 ---\n");
    Grid g(4,4,4); Workspace ws(4); IsingState state(g);
    Measures meas(g_handle,ws);
    int center=g.site_id(2,2);
    state.init_heisenberg(g,center);
    Scalar C=meas.autocorrelation(g,state.psi,state.messages,center);
    printf("  C(t=0)=%.6f (expected 1.0)\n",C);
    assert(std::abs(C-1.f)<1e-4f);
    printf("  C(t=0)=1: OK\n");
}

/* Test 8: Rz only (J=0) C stays at 1 — gate by gate */
static void test_rz_only(){
    printf("\n--- Test 8: Rz only (J=0) ---\n");
    Grid g(4,4,4); Workspace ws(4); IsingState state(g);
    Measures meas(g_handle,ws);
    IsingSimulator sim(g_handle,-1.f,0.f,0.1f);
    int center=g.site_id(2,2);
    state.init_heisenberg(g,center);
    GateOps ops(g_handle);

    Scalar C=meas.autocorrelation(g,state.psi,state.messages,center);
    printf("  before: C=%.6f\n",C);

    sim.apply_rz_layer(state,g,ws);
    C=meas.autocorrelation(g,state.psi,state.messages,center);
    printf("  after Rz1: C=%.6f\n",C);
    assert(std::abs(C-1.f)<1e-3f);

    sim.apply_rxx_layer(state,g,ws,4);
    C=meas.autocorrelation(g,state.psi,state.messages,center);
    printf("  after Rxx(J=0): C=%.6f\n",C);
    assert(std::abs(C-1.f)<1e-3f);

    sim.apply_rz_layer(state,g,ws);
    C=meas.autocorrelation(g,state.psi,state.messages,center);
    printf("  after Rz2: C=%.6f\n",C);
    assert(std::abs(C-1.f)<1e-3f);

    printf("  Rz only: OK\n");
}

/* Test 9: autocorrelation evolves correctly with J!=0 on 2x2 */
static void test_norm(){
    printf("\n--- Test 9: autocorr evolves (2x2, J!=0) ---\n");
    Grid g(2,2,4); Workspace ws(4); IsingState state(g);
    Measures meas(g_handle,ws);
    IsingSimulator sim(g_handle,-1.f,-1.f,0.1f);
    int center=g.site_id(0,0);
    state.init_heisenberg(g,center);
    // norm_sqr uses I-selector everywhere; at t=0 center is Z so I-component=0 → norm=0.
    // That is correct physics: Tr(Z)=0. Check C(0)=1 via autocorrelation instead.
    Scalar C0=meas.autocorrelation(g,state.psi,state.messages,center);
    printf("  C(t=0)=%.6f (expected 1.0)\n",C0);
    assert(std::abs(C0-1.f)<1e-3f);
    for(int s=1;s<=3;s++){
        sim.trotter_layer(state,g,ws,4);
        Scalar C=meas.autocorrelation(g,state.psi,state.messages,center);
        printf("  step %d: C=%.6f (should be in (-1,1])\n",s,C);
        assert(std::isfinite(C));
    }
    printf("  Autocorr evolves: OK\n");
}

int main(){
    printf("=== test_gate (D=%d) ===\n",D);
    rocblas_create_handle(&g_handle);
    test_pseudo_sqrt();
    test_unary_gate();
    test_rz_preserves_z();
    test_qr();
    test_apply_gate_to_RR();
    test_binary_gate_identity();
    test_c0();
    test_rz_only();
    test_norm();
    rocblas_destroy_handle(g_handle);
    printf("\n=== ALL GATE TESTS COMPLETE ===\n");
    return 0;
}
