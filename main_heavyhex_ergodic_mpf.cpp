/*
 * main_heavyhex_ergodic_mpf.cpp — Ergodic amplification MPF
 *
 * Fixed noise γ=1.0, varying ε=0.10/0.15/0.20 (coupling strength/Trotter step)
 * Same structure as gamma-MPF but amplifies ergodicity not noise.
 *
 * Output: STEP C_nl C_n1 C_n2 C_n3 (nl=ε=0.1 noiseless, n1=ε=0.1, n2=ε=0.15, n3=ε=0.20 noisy)
 *
 * Build: make PHYS_DIM=4 main_heavyhex_ergodic_mpf
 * Run:   ./main_heavyhex_ergodic_mpf [chi_max] [n_steps] [cutoff]
 */

#include "types.hpp"
#include "tensor.hpp"
#include "grid.hpp"
#include "workspace.hpp"
#include "gate.hpp"
#include "measures.hpp"
#include "heavyhex.hpp"
#include "heavyhex_physics.hpp"

// Noiseless reference at ε=0.10
#include "heavyhex_ideal_kick_eps010_data.hpp"
// Noisy at γ=1.0, ε=0.10/0.15/0.20
#include "heavyhex_ideal_kick_eps010_gamma1_data.hpp"
#include "heavyhex_ideal_kick_eps015_gamma1_data.hpp"
#include "heavyhex_ideal_kick_eps020_gamma1_data.hpp"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

static constexpr int BP_SWEEPS = 100;
static constexpr int N_EPS     = 3;
static const float   EPSILONS[N_EPS] = {0.10f, 0.15f, 0.20f};

static double now_ms() {
    using namespace std::chrono;
    return duration<double,std::milli>(high_resolution_clock::now().time_since_epoch()).count();
}

// PTM accessors for each ε at γ=1.0
static const float* ptm_eps010(int b) { return HEAVY_HEX_3x3_IDEAL_KICK_EPS010_GAMMA1[b]; }
static const float* ptm_eps015(int b) { return HEAVY_HEX_3x3_IDEAL_KICK_EPS015_GAMMA1[b]; }
static const float* ptm_eps020(int b) { return HEAVY_HEX_3x3_IDEAL_KICK_EPS020_GAMMA1[b]; }

using PtmFunc = const float*(*)(int);

struct Sim {
    rocblas_handle handle; GateOps g; PtmFunc ptm;
    Sim(rocblas_handle h, PtmFunc fn) : handle(h), g(h), ptm(fn) {}
    void step(HeavyHexState& s, FlatHeavyHexGrid& gr, Workspace& ws, int chi, float cut) {
        int nm=2*gr.n_bonds;
        std::vector<int> fc(gr.n_bonds);
        std::vector<Scalar*> fm(nm,nullptr),nm2(nm,nullptr);
        size_t al=(size_t)chi*chi*sizeof(Scalar);
        for(auto& b:gr.bonds)for(int d=0;d<2;d++){int idx=(d==0)?b.msg_from_a():b.msg_from_b();HIP_CHECK(hipMalloc(&fm[idx],al));HIP_CHECK(hipMalloc(&nm2[idx],al));}
        for(int c=0;c<3;c++){
            for(auto& b:gr.bonds)fc[b.id]=b.chi;
            for(auto& b:gr.bonds){int chi2=b.chi;size_t sz=(size_t)chi2*chi2*sizeof(Scalar);for(int d=0;d<2;d++){int idx=(d==0)?b.msg_from_a():b.msg_from_b();HIP_CHECK(hipMemcpy(fm[idx],s.messages[idx],sz,hipMemcpyDeviceToDevice));HIP_CHECK(hipMemcpy(nm2[idx],s.messages[idx],sz,hipMemcpyDeviceToDevice));}}
            for(auto& gate:gr.colors[c]){const float* G=ptm(gate.bond_id);g.apply_binary_gate(s.psi[gate.site_a],s.psi[gate.site_b],gate,gr,nm2,fm,fc,G,ws,chi,cut);}
            for(auto& gate:gr.colors[c]){const Bond& b=gr.bonds[gate.bond_id];int chi2=b.chi;size_t sz=(size_t)chi2*chi2*sizeof(Scalar);for(int d=0;d<2;d++){int idx=(d==0)?b.msg_from_a():b.msg_from_b();HIP_CHECK(hipMemcpy(s.messages[idx],nm2[idx],sz,hipMemcpyDeviceToDevice));}}
        }
        for(auto p:fm)if(p)HIP_CHECK(hipFree(p));for(auto p:nm2)if(p)HIP_CHECK(hipFree(p));
        // kick — use ε=0.10 kick PTM for noiseless reference
        // noisy states use their own ε's kick (encoded in PTM)
        size_t tsz=0;for(auto& site:gr.sites){int vol=D;for(int d=0;d<MAX_DEGREE;d++)if(site.bond_id[d]>=0)vol*=gr.bonds[site.bond_id[d]].chi;tsz=std::max(tsz,(size_t)vol);}
        Scalar* tmp;HIP_CHECK(hipMalloc(&tmp,tsz*sizeof(Scalar)));
        // Use the site PTM from each model's kick
        for(auto& site:gr.sites)g.apply_unary_gate(s.psi[site.id],site,gr,HEAVY_HEX_3x3_IDEAL_KICK_EPS010_GAMMA1_SITE_PTM,tmp);
        HIP_CHECK(hipFree(tmp));
    }
};

int main(int argc, char** argv) {
    int   chi_max = argc>1?atoi(argv[1]):70;
    int   n_steps = argc>2?atoi(argv[2]):10;
    float cutoff  = argc>3?atof(argv[3]):1e-4f;
    int   center  = 2;

    printf("# CppSim ergodic-MPF: fixed γ=1.0, varying ε=0.10/0.15/0.20\n");
    printf("# chi=%d n_steps=%d cutoff=%.0e\n", chi_max, n_steps, (double)cutoff);
    printf("# STEP C_nl(ε=0.10,nl) C_n1(ε=0.10,noisy) C_n2(ε=0.15,noisy) C_n3(ε=0.20,noisy) chi t(s)\n");
    fflush(stdout);

    rocblas_handle handle; rocblas_create_handle(&handle);

    // Noiseless at ε=0.10
    FlatHeavyHexGrid gnl(chi_max); Workspace wnl(chi_max,3);
    HeavyHexState snl(gnl); Measures meas(handle,wnl);
    // Noiseless uses ideal kick eps010
    struct SimNL {
        rocblas_handle handle; GateOps g;
        SimNL(rocblas_handle h):handle(h),g(h){}
        void step(HeavyHexState& s, FlatHeavyHexGrid& gr, Workspace& ws, int chi, float cut){
            int nm=2*gr.n_bonds;std::vector<int> fc(gr.n_bonds);
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
    } sim_nl(handle);
    snl.init_heisenberg(gnl,center);

    // Three noisy states at ε=0.10/0.15/0.20 with γ=1.0
    PtmFunc ptm_fns[N_EPS] = {ptm_eps010, ptm_eps015, ptm_eps020};
    FlatHeavyHexGrid* grids_n[N_EPS];
    HeavyHexState*    states_n[N_EPS];
    Workspace*        ws_n[N_EPS];
    Sim*              sims_n[N_EPS];
    for(int e=0;e<N_EPS;e++){
        grids_n[e]  = new FlatHeavyHexGrid(chi_max);
        states_n[e] = new HeavyHexState(*grids_n[e]);
        ws_n[e]     = new Workspace(chi_max,3);
        sims_n[e]   = new Sim(handle, ptm_fns[e]);
        states_n[e]->init_heisenberg(*grids_n[e],center);
    }

    for(int step=1;step<=n_steps;step++){
        double t0=now_ms();
        sim_nl.step(snl,gnl,wnl,chi_max,cutoff);
        for(int e=0;e<N_EPS;e++) sims_n[e]->step(*states_n[e],*grids_n[e],*ws_n[e],chi_max,cutoff);

        float C_nl = meas.autocorrelation_with_bp(static_cast<Grid&>(gnl),snl.psi,snl.messages,center,BP_SWEEPS);
        float C_n[N_EPS];
        for(int e=0;e<N_EPS;e++)
            C_n[e]=meas.autocorrelation_with_bp(static_cast<Grid&>(*grids_n[e]),states_n[e]->psi,states_n[e]->messages,center,BP_SWEEPS);

        int chi_nl=1,chi_n=1;
        for(auto& b:gnl.bonds) chi_nl=std::max(chi_nl,b.chi);
        for(auto& b:grids_n[0]->bonds) chi_n=std::max(chi_n,b.chi);

        printf("STEP %d  C_nl %+.6f  C_n %+.6f %+.6f %+.6f  chi %d %d  t %.2f\n",
               step, C_nl, C_n[0], C_n[1], C_n[2], chi_nl, chi_n, (now_ms()-t0)/1000.0);
        fflush(stdout);
    }

    rocblas_destroy_handle(handle);
    for(int e=0;e<N_EPS;e++){delete sims_n[e];delete ws_n[e];delete states_n[e];delete grids_n[e];}
    return 0;
}
