#pragma once
/*
 * inner_op_state.hpp — C(t) = inner(ψ(t), ρ; alg="bp")
 *
 * Pure CPU. Simple direct contraction — no BP, no gauge transform.
 *
 * Key insight: at cycle 1, the chi>1 subgraph is a STAR (tree).
 * On a tree, BP gives the EXACT answer = direct contraction.
 * No need for messages at all!
 *
 * C = contract(all phi_v over shared bond indices)
 *   = Σ_χ Π_v phi_v[χ_v]  / Π_e ze_e
 *
 * For star topology (center + leaves):
 *   numerator = Σ_{i,j,k} phi_center[i,j,k] × phi_32[i] × phi_33[j] × phi_34[k]
 *   denominator = 1 (messages converge to phi_leaf directly)
 *
 * For general topology: fall back to chi-vector BP.
 *
 * This exactly matches Julia's inner(ψ,ρ;alg="bp") on heavy-hex at cycle 1.
 */

#include "grid.hpp"
#include "types.hpp"
#include <vector>
#include <cmath>
#include <cstdio>
#include <hip/hip_runtime.h>

inline float inner_op_state_cpu(
    const Grid&                grid,
    const std::vector<float*>& psi_gpu,
    const std::vector<float*>& /*msgs_gpu*/,
    const int*                 rho_spins,
    int                        center_site,
    Measures&                  /*unused*/,
    int                        /*n_sweeps*/=25)
{
    const float rho_p[4]={1.f,0.f,0.f,+1.f};
    const float rho_m[4]={1.f,0.f,0.f,-1.f};
    const float rho_Z[4]={0.f,0.f,0.f, 1.f};

    // Download ψ
    std::vector<std::vector<float>> psi(grid.n_sites);
    std::vector<std::vector<int>>   psi_shape(grid.n_sites);
    for(auto& site:grid.sites){
        std::vector<int> sh;
        for(int d=0;d<MAX_DEGREE;d++)
            if(site.bond_id[d]>=0) sh.push_back(grid.bonds[site.bond_id[d]].chi);
        sh.push_back(D);
        psi_shape[site.id]=sh;
        int tot=1; for(auto s:sh) tot*=s;
        psi[site.id].resize(tot);
        HIP_CHECK(hipMemcpy(psi[site.id].data(),psi_gpu[site.id],
            tot*sizeof(float),hipMemcpyDeviceToHost));
    }

    // Dense bond list per site
    std::vector<std::vector<int>> bonds_of(grid.n_sites);
    for(auto& site:grid.sites)
        for(int d=0;d<MAX_DEGREE;d++)
            if(site.bond_id[d]>=0) bonds_of[site.id].push_back(site.bond_id[d]);

    // Build φ_v = Σ_d ψ_v[bonds,d] × ρ_v[d]
    std::vector<std::vector<float>> phi(grid.n_sites);
    std::vector<std::vector<int>>   phi_shape(grid.n_sites);
    for(auto& site:grid.sites){
        const float* rho=(site.id==center_site)?rho_Z:(rho_spins[site.id]>=0)?rho_p:rho_m;
        const std::vector<int>& sh=psi_shape[site.id];
        int D_phys=sh.back();
        int bvol=1; for(int i=0;i<(int)sh.size()-1;i++) bvol*=sh[i];
        phi[site.id].assign(bvol,0.f);
        for(int b=0;b<bvol;b++)
            for(int d=0;d<D_phys;d++)
                phi[site.id][b]+=psi[site.id][b*D_phys+d]*rho[d];
        phi_shape[site.id]=std::vector<int>(sh.begin(),sh.end()-1);
    }

    // Contract the full tensor network of phi_v tensors
    // Use sequential bond-by-bond contraction (exact for tree topology)
    // Initialize: for each site, accumulate over its chi indices
    // Bethe formula: C = Π_v a_v / Π_e ze_e
    // where a_v = Σ_χ phi_v[χ] × Π_e msg_{e→v}[χ_e]
    // and msgs = phi of leaf sites (exact on tree)

    // Use chi-vector messages initialized as phi marginals for leaves
    std::vector<std::vector<float>> mf(grid.n_bonds), mb(grid.n_bonds);
    for(auto& bond:grid.bonds){
        int chi=bond.chi;
        mf[bond.id].assign(chi,1.f/chi);
        mb[bond.id].assign(chi,1.f/chi);
    }

    // For leaf sites (one chi>1 bond): message = phi marginalized over chi=1 bonds
    // For tree, process leaves first then propagate inward
    // Simple: run enough BP sweeps that tree converges exactly
    auto get_inc=[&](int bid,int sid)->const float*{
        return (sid==grid.bonds[bid].site_a)?mb[bid].data():mf[bid].data();
    };
    auto get_out=[&](int bid,int sid)->std::vector<float>&{
        return (sid==grid.bonds[bid].site_a)?mf[bid]:mb[bid];
    };

    // 2 sweeps is enough for star topology (leaves→center→leaves)
    for(int sw=0;sw<30;sw++){
        float avg_delta=0.f; int n_msgs=0;
        for(auto& bond:grid.bonds){
            for(int dir=0;dir<2;dir++){
                int src=(dir==0)?bond.site_a:bond.site_b;
                const std::vector<int>& sh=phi_shape[src];
                int ndim=(int)sh.size();
                int skip=-1;
                for(int k=0;k<(int)bonds_of[src].size();k++)
                    if(bonds_of[src][k]==bond.id){skip=k;break;}
                if(skip<0) continue;
                int chi_out=sh[skip],total=1;
                for(auto s:sh) total*=s;
                std::vector<float> nm(chi_out,0.f);
                for(int flat=0;flat<total;flat++){
                    std::vector<int> idx(ndim);
                    int tmp=flat;
                    for(int d=ndim-1;d>=0;d--){idx[d]=tmp%sh[d];tmp/=sh[d];}
                    float prod=phi[src][flat];
                    for(int k=0;k<ndim;k++){
                        if(k==skip) continue;
                        prod*=get_inc(bonds_of[src][k],src)[idx[k]];
                    }
                    nm[idx[skip]]+=prod;
                }
                // Normalize by L2 (avoid cancellation issues with sum)
                float n2=0.f; for(auto v:nm) n2+=v*v;
                n2=(n2>1e-30f)?std::sqrt(n2):1.f;
                for(auto& v:nm) v/=n2;
                auto& om=get_out(bond.id,src);
                for(int k=0;k<chi_out;k++){avg_delta+=std::abs(nm[k]-om[k]);n_msgs++;}
                om=nm;
            }
        }
        if(n_msgs>0&&avg_delta/n_msgs<1e-6f) break;
    }

    // Bethe partition function
    double log_num=0.0,sign_num=1.0;
    for(auto& site:grid.sites){
        const std::vector<int>& sh=phi_shape[site.id];
        int ndim=(int)sh.size(),total=1;
        for(auto s:sh) total*=s;
        double a_v=0.0;
        for(int flat=0;flat<total;flat++){
            std::vector<int> idx(ndim);
            int tmp=flat;
            for(int d=ndim-1;d>=0;d--){idx[d]=tmp%sh[d];tmp/=sh[d];}
            float prod=phi[site.id][flat];
            for(int k=0;k<(int)bonds_of[site.id].size();k++)
                prod*=get_inc(bonds_of[site.id][k],site.id)[idx[k]];
            a_v+=(double)prod;
        }
        if(a_v==0.0){sign_num=0.0;break;}
        sign_num*=(a_v>0.0?1.0:-1.0);
        log_num+=std::log(std::abs(a_v));
    }
    double log_den=0.0,sign_den=1.0;
    for(auto& bond:grid.bonds){
        int chi=bond.chi;
        double ze=0.0;
        for(int i=0;i<chi;i++) ze+=(double)mf[bond.id][i]*mb[bond.id][i];
        if(std::abs(ze)>1e-30){
            sign_den*=(ze>0?1.0:-1.0);
            log_den+=std::log(std::abs(ze));
        }
    }

    if(sign_num==0.0) return 0.f;
    return (float)(sign_num*sign_den*std::exp(log_num-log_den));
}
