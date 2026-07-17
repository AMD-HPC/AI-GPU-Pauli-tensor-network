#pragma once
/*
 * bilinear_form_bp.hpp — Two-state overlap ⟨ψ_a|ψ_b⟩ via BilinearForm BP
 *
 * Pure CPU. Matches Julia's inner(ψ_a, ψ_b; alg="bp").
 *
 * Handles DIFFERENT chi between states (noiseless vs noisy evolve differently).
 * T_v bond dim = chi_a × chi_b per bond (chi from each state's grid).
 *
 * Algorithm:
 *   1. Build T_v[ia0*chi_b0+ib0,...] = Σ_d ψa[ia...,d] × ψb[ib...,d]
 *   2. Init messages M_e[ia*chi_b+ib] = identity/sum
 *   3. BP sweeps with sum normalization
 *   4. Bethe: C = Π_v a_v / Π_e ze_e
 */

#include "grid.hpp"
#include "types.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <hip/hip_runtime.h>

inline float bilinear_form_bp(
    const Grid&                grid_a,
    const std::vector<float*>& psi_a_gpu,
    const Grid&                grid_b,
    const std::vector<float*>& psi_b_gpu,
    int                        n_sweeps = 25,
    float                      tol = 1e-5f)
{
    int n_sites = grid_a.n_sites;
    int n_bonds = grid_a.n_bonds;

    // ── 1. Download ψ_a and ψ_b ─────────────────────────────────────
    std::vector<std::vector<float>> psi_a(n_sites), psi_b(n_sites);
    std::vector<std::vector<int>>   sha(n_sites), shb(n_sites);

    for (auto& site : grid_a.sites) {
        std::vector<int> s;
        for (int d=0;d<MAX_DEGREE;d++)
            if (site.bond_id[d]>=0) s.push_back(grid_a.bonds[site.bond_id[d]].chi);
        s.push_back(D);
        sha[site.id]=s;
        int tot=1; for(auto x:s) tot*=x;
        psi_a[site.id].resize(tot);
        HIP_CHECK(hipMemcpy(psi_a[site.id].data(),psi_a_gpu[site.id],
            tot*sizeof(float),hipMemcpyDeviceToHost));
    }
    for (auto& site : grid_b.sites) {
        std::vector<int> s;
        for (int d=0;d<MAX_DEGREE;d++)
            if (site.bond_id[d]>=0) s.push_back(grid_b.bonds[site.bond_id[d]].chi);
        s.push_back(D);
        shb[site.id]=s;
        int tot=1; for(auto x:s) tot*=x;
        psi_b[site.id].resize(tot);
        HIP_CHECK(hipMemcpy(psi_b[site.id].data(),psi_b_gpu[site.id],
            tot*sizeof(float),hipMemcpyDeviceToHost));
    }

    // Dense bond list per site
    std::vector<std::vector<int>> bonds_of(n_sites);
    for (auto& site : grid_a.sites)
        for (int d=0;d<MAX_DEGREE;d++)
            if (site.bond_id[d]>=0)
                bonds_of[site.id].push_back(site.bond_id[d]);

    // ── 2. Build T_v ─────────────────────────────────────────────────
    std::vector<std::vector<float>> T(n_sites);
    std::vector<std::vector<int>>   T_shape(n_sites);  // chi_a*chi_b per bond

    for (auto& site : grid_a.sites) {
        const auto& sa=sha[site.id]; const auto& sb=shb[site.id];
        int nb=(int)sa.size()-1; int Dp=sa.back();
        std::vector<int> chia(nb),chib(nb);
        int vol_a=1,vol_b=1;
        for(int k=0;k<nb;k++){
            chia[k]=sa[k]; vol_a*=chia[k];
            chib[k]=sb[k]; vol_b*=chib[k];
        }
        const float* pa=psi_a[site.id].data();
        const float* pb=psi_b[site.id].data();

        std::vector<float> T_big(vol_a*vol_b,0.f);
        for(int ia=0;ia<vol_a;ia++)
            for(int ib=0;ib<vol_b;ib++)
                for(int d=0;d<Dp;d++)
                    T_big[ia*vol_b+ib]+=pa[ia*Dp+d]*pb[ib*Dp+d];

        std::vector<int> T_sh; for(int k=0;k<nb;k++) T_sh.push_back(chia[k]*chib[k]);
        int total_T=1; for(auto s:T_sh) total_T*=s;
        std::vector<float> T_out(total_T,0.f);

        for(int ia_f=0;ia_f<vol_a;ia_f++)
        for(int ib_f=0;ib_f<vol_b;ib_f++){
            float val=T_big[ia_f*vol_b+ib_f];
            std::vector<int> ia(nb),ib(nb);
            int ta=ia_f,tb=ib_f;
            for(int k=nb-1;k>=0;k--){ia[k]=ta%chia[k];ta/=chia[k];ib[k]=tb%chib[k];tb/=chib[k];}
            int idx=0;
            for(int k=0;k<nb;k++) idx=idx*T_sh[k]+ia[k]*chib[k]+ib[k];
            T_out[idx]=val;
        }
        T[site.id]=T_out; T_shape[site.id]=T_sh;
    }

    // ── 3. Initialize messages ────────────────────────────────────────
    std::vector<std::vector<float>> M_fwd(n_bonds),M_bwd(n_bonds);
    for(auto& bond : grid_a.bonds){
        int ca=bond.chi, cb=grid_b.bonds[bond.id].chi;
        int cab=ca*cb;
        M_fwd[bond.id].assign(cab,0.f); M_bwd[bond.id].assign(cab,0.f);
        int mn=std::min(ca,cb);
        float iv=(mn>0)?1.f/mn:1.f;
        for(int i=0;i<mn;i++){
            M_fwd[bond.id][i*cb+i]=iv;
            M_bwd[bond.id][i*cb+i]=iv;
        }
    }

    auto get_inc=[&](int bid,int sid)->const float*{
        const Bond& b=grid_a.bonds[bid];
        return (sid==b.site_a)?M_bwd[bid].data():M_fwd[bid].data();
    };
    auto get_out=[&](int bid,int sid)->std::vector<float>&{
        const Bond& b=grid_a.bonds[bid];
        return (sid==b.site_a)?M_fwd[bid]:M_bwd[bid];
    };

    // ── 4. BP sweeps ─────────────────────────────────────────────────
    for(int sweep=0;sweep<n_sweeps;sweep++){
        float avg_delta=0.f; int n_msgs=0;
        for(auto& bond : grid_a.bonds){
            for(int dir=0;dir<2;dir++){
                int src=(dir==0)?bond.site_a:bond.site_b;
                const auto& sh=T_shape[src];
                int ndim=(int)sh.size();
                const auto& db=bonds_of[src];
                int skip=-1;
                for(int k=0;k<(int)db.size();k++) if(db[k]==bond.id){skip=k;break;}
                if(skip<0) continue;
                int chi_out=sh[skip];
                int total=1; for(auto s:sh) total*=s;
                std::vector<float> nm(chi_out,0.f);
                for(int flat=0;flat<total;flat++){
                    std::vector<int> idx(ndim);
                    int tmp=flat;
                    for(int d=ndim-1;d>=0;d--){idx[d]=tmp%sh[d];tmp/=sh[d];}
                    float prod=T[src][flat];
                    for(int k=0;k<ndim;k++){
                        if(k==skip) continue;
                        prod*=get_inc(db[k],src)[idx[k]];
                    }
                    nm[idx[skip]]+=prod;
                }
                float s=0.f; for(auto v:nm) s+=v;
                if(std::abs(s)>1e-20f) for(auto& v:nm) v/=s;
                auto& om=get_out(bond.id,src);
                for(int k=0;k<chi_out;k++){avg_delta+=std::abs(nm[k]-om[k]);n_msgs++;}
                om=nm;
            }
        }
        if(n_msgs>0&&avg_delta/n_msgs<tol) break;
    }

    // ── 5. Bethe PF ──────────────────────────────────────────────────
    double log_num=0.0,sign_num=1.0;
    for(auto& site : grid_a.sites){
        const auto& sh=T_shape[site.id];
        int ndim=(int)sh.size(),total=1;
        for(auto s:sh) total*=s;
        const auto& db=bonds_of[site.id];
        double a_v=0.0;
        for(int flat=0;flat<total;flat++){
            std::vector<int> idx(ndim);
            int tmp=flat;
            for(int d=ndim-1;d>=0;d--){idx[d]=tmp%sh[d];tmp/=sh[d];}
            float prod=T[site.id][flat];
            for(int k=0;k<ndim;k++) prod*=get_inc(db[k],site.id)[idx[k]];
            a_v+=(double)prod;
        }
        if(a_v==0.0){sign_num=0.0;break;}
        sign_num*=(a_v>0.0?1.0:-1.0);
        log_num+=std::log(std::abs(a_v));
    }

    double log_den=0.0,sign_den=1.0;
    for(auto& bond : grid_a.bonds){
        int cab=(int)M_fwd[bond.id].size();
        double ze=0.0;
        for(int k=0;k<cab;k++) ze+=(double)M_fwd[bond.id][k]*M_bwd[bond.id][k];
        if(std::abs(ze)>1e-30){
            sign_den*=(ze>0?1.0:-1.0);
            log_den+=std::log(std::abs(ze));
        }
    }

    if(sign_num==0.0) return 0.f;
    return (float)(sign_num*sign_den*std::exp(log_num-log_den));
}
