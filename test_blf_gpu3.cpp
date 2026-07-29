/*
 * test_blf_gpu3.cpp — Validate BLF_GPU3 against BLF2 CPU reference.
 *
 * Runs one noiseless + one noisy step, then compares:
 *   A[0,0] = <nl|nl>  (expect ~0.172 at chi=4 step 1)
 *   B[0]   = <nl|n1>  (expect ~0.163 at chi=4 step 1)
 *
 * Build: make PHYS_DIM=4 test_blf_gpu3
 * Run:   ./test_blf_gpu3 [chi_max] [n_steps]
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "measures.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"
#include "bilinear_form_bp.hpp"
#include "bilinear_form_bp_cpu2.hpp"
#include "bilinear_form_bp_gpu3.hpp"

#include "heavyhex_ideal_kick_eps010_data.hpp"
#include "heavyhex_noisy_eps010_gamma1_data.hpp"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

static double now_ms(){
    using namespace std::chrono;
    return duration<double,std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

static const float* ptm_gamma1(int b){return HEAVY_HEX_3x3_NOISY_EPS010_GAMMA1_GAMMA1_0[b];}

struct SimNL {
    rocblas_handle h; GateOps g;
    SimNL(rocblas_handle hh):h(hh),g(hh){}
    void step(HeavyHexState& s,FlatHeavyHexGrid& gr,Workspace& ws,int chi,float cut){
        int nm=2*gr.n_bonds;
        std::vector<int> fc(gr.n_bonds);
        std::vector<Scalar*> fm(nm,nullptr),nm2(nm,nullptr);
        size_t al=(size_t)chi*chi*sizeof(Scalar);
        for(auto& b:gr.bonds)for(int d=0;d<2;d++){int idx=(d==0)?b.msg_from_a():b.msg_from_b();HIP_CHECK(hipMalloc(&fm[idx],al));HIP_CHECK(hipMalloc(&nm2[idx],al));}
        for(int c=0;c<3;c++){
            for(auto& b:gr.bonds)fc[b.id]=b.chi;
            for(auto& b:gr.bonds){int chi2=b.chi;size_t sz=(size_t)chi2*chi2*sizeof(Scalar);for(int d=0;d<2;d++){int idx=(d==0)?b.msg_from_a():b.msg_from_b();HIP_CHECK(hipMemcpy(fm[idx],s.messages[idx],sz,hipMemcpyDeviceToDevice));HIP_CHECK(hipMemcpy(nm2[idx],s.messages[idx],sz,hipMemcpyDeviceToDevice));}}
            for(auto& gate:gr.colors[c]){const float* G=HEAVY_HEX_3x3_IDEAL_KICK_EPS010[gate.bond_id];g.apply_binary_gate(s.psi[gate.site_a],s.psi[gate.site_b],gate,gr,nm2,fm,fc,G,ws,chi,cut);}
            for(auto& gate:gr.colors[c]){const Bond& b=gr.bonds[gate.bond_id];int chi2=b.chi;size_t sz=(size_t)chi2*chi2*sizeof(Scalar);for(int d=0;d<2;d++){int idx=(d==0)?b.msg_from_a():b.msg_from_b();HIP_CHECK(hipMemcpy(s.messages[idx],nm2[idx],sz,hipMemcpyDeviceToDevice));}}
        }
        for(auto p:fm)if(p)HIP_CHECK(hipFree(p));for(auto p:nm2)if(p)HIP_CHECK(hipFree(p));
        size_t tsz=0;for(auto& site:gr.sites){int vol=D;for(int d=0;d<MAX_DEGREE;d++)if(site.bond_id[d]>=0)vol*=gr.bonds[site.bond_id[d]].chi;tsz=std::max(tsz,(size_t)vol);}
        Scalar* tmp;HIP_CHECK(hipMalloc(&tmp,tsz*sizeof(Scalar)));
        for(auto& site:gr.sites)g.apply_unary_gate(s.psi[site.id],site,gr,HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM,tmp);
        HIP_CHECK(hipFree(tmp));
    }
};

struct SimN {
    rocblas_handle h; GateOps g;
    SimN(rocblas_handle hh):h(hh),g(hh){}
    void step(HeavyHexState& s,FlatHeavyHexGrid& gr,Workspace& ws,int chi,float cut){
        int nm=2*gr.n_bonds;
        std::vector<int> fc(gr.n_bonds);
        std::vector<Scalar*> fm(nm,nullptr),nm2(nm,nullptr);
        size_t al=(size_t)chi*chi*sizeof(Scalar);
        for(auto& b:gr.bonds)for(int d=0;d<2;d++){int idx=(d==0)?b.msg_from_a():b.msg_from_b();HIP_CHECK(hipMalloc(&fm[idx],al));HIP_CHECK(hipMalloc(&nm2[idx],al));}
        for(int c=0;c<3;c++){
            for(auto& b:gr.bonds)fc[b.id]=b.chi;
            for(auto& b:gr.bonds){int chi2=b.chi;size_t sz=(size_t)chi2*chi2*sizeof(Scalar);for(int d=0;d<2;d++){int idx=(d==0)?b.msg_from_a():b.msg_from_b();HIP_CHECK(hipMemcpy(fm[idx],s.messages[idx],sz,hipMemcpyDeviceToDevice));HIP_CHECK(hipMemcpy(nm2[idx],s.messages[idx],sz,hipMemcpyDeviceToDevice));}}
            for(auto& gate:gr.colors[c]){const float* G=ptm_gamma1(gate.bond_id);g.apply_binary_gate(s.psi[gate.site_a],s.psi[gate.site_b],gate,gr,nm2,fm,fc,G,ws,chi,cut);}
            for(auto& gate:gr.colors[c]){const Bond& b=gr.bonds[gate.bond_id];int chi2=b.chi;size_t sz=(size_t)chi2*chi2*sizeof(Scalar);for(int d=0;d<2;d++){int idx=(d==0)?b.msg_from_a():b.msg_from_b();HIP_CHECK(hipMemcpy(s.messages[idx],nm2[idx],sz,hipMemcpyDeviceToDevice));}}
        }
        for(auto p:fm)if(p)HIP_CHECK(hipFree(p));for(auto p:nm2)if(p)HIP_CHECK(hipFree(p));
        size_t tsz=0;for(auto& site:gr.sites){int vol=D;for(int d=0;d<MAX_DEGREE;d++)if(site.bond_id[d]>=0)vol*=gr.bonds[site.bond_id[d]].chi;tsz=std::max(tsz,(size_t)vol);}
        Scalar* tmp;HIP_CHECK(hipMalloc(&tmp,tsz*sizeof(Scalar)));
        for(auto& site:gr.sites)g.apply_unary_gate(s.psi[site.id],site,gr,HEAVY_HEX_3x3_IDEAL_KICK_EPS010_SITE_PTM,tmp);
        HIP_CHECK(hipFree(tmp));
    }
};

int main(int argc,char** argv){
    int chi_max=argc>1?atoi(argv[1]):4;
    int n_steps=argc>2?atoi(argv[2]):1;
    float cutoff=1e-4f; int center=2, device=0;

    printf("=== BLF_GPU3 test  chi=%d steps=%d ===\n",chi_max,n_steps);

    rocblas_handle handle; rocblas_create_handle(&handle);

    FlatHeavyHexGrid gnl(chi_max); Workspace wnl(chi_max,3);
    HeavyHexState snl(gnl); Measures meas(handle,wnl); SimNL simnl(handle);
    snl.init_heisenberg(gnl,center);
    for(int i=0;i<n_steps;i++) simnl.step(snl,gnl,wnl,chi_max,cutoff);
    float C_nl=meas.autocorrelation_with_bp(static_cast<Grid&>(gnl),snl.psi,snl.messages,center,100);
    printf("C_nl(%d)=%.6f\n",n_steps,C_nl);

    FlatHeavyHexGrid gn1(chi_max); Workspace wn1(chi_max,3);
    HeavyHexState sn1(gn1); SimN simn(handle);
    sn1.init_heisenberg(gn1,center);
    for(int i=0;i<n_steps;i++) simn.step(sn1,gn1,wn1,chi_max,cutoff);
    float C_n1=meas.autocorrelation_with_bp(static_cast<Grid&>(gn1),sn1.psi,sn1.messages,center,100);
    printf("C_n1(%d)=%.6f\n",n_steps,C_n1);

    // CPU reference (BLF2)
    StateSnapshot snap_nl, snap_n1;
    snap_nl.download(static_cast<Grid&>(gnl),snl.psi,snl.messages,device);
    snap_n1.download(static_cast<Grid&>(gn1),sn1.psi,sn1.messages,device);

    printf("\n--- BLF2 CPU reference ---\n");
    double t0=now_ms();
    float A00_ref=BLF2::compute(snap_nl,snap_nl,25,1e-5f);
    float B0_ref =BLF2::compute(snap_nl,snap_n1,25,1e-5f);
    printf("A[0,0]=%.6f  B[0]=%.6f  t=%.2fs\n",A00_ref,B0_ref,(now_ms()-t0)/1000.0);

    // GPU3
    printf("\n--- BLF_GPU3 ---\n");
    BLF_GPU3 blf(handle);
    t0=now_ms();
    blf.upload(gnl,snl.psi,device, gnl,snl.psi,device);
    float A00_gpu=blf.compute(25,1e-5f);
    printf("A[0,0]=%.6f  (ref=%.6f)  diff=%.2e\n",A00_gpu,A00_ref,std::abs(A00_gpu-A00_ref));

    blf.upload(gnl,snl.psi,device, gn1,sn1.psi,device);
    float B0_gpu=blf.compute(25,1e-5f);
    printf("B[0]  =%.6f  (ref=%.6f)  diff=%.2e  t=%.2fs\n",
           B0_gpu,B0_ref,std::abs(B0_gpu-B0_ref),(now_ms()-t0)/1000.0);

    rocblas_destroy_handle(handle);
    return 0;
}
