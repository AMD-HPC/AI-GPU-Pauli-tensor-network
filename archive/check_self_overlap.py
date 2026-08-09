"""
Check: does our exact contraction give ||ψ_nl||² ≈ 0.99991?
If our formula is right for same-state overlap, it should match Julia.
If it gives ~17, the normalization convention differs.
"""
import numpy as np

states={'NL':{},'N':{}}
with open('two_states_dump.txt') as f:
    for line in f:
        p=line.split()
        if p[0]=='OVERLAP': continue
        label=p[0]; v=int(p[1])
        i_nbrs=p.index('nbrs'); i_shape=p.index('shape'); i_data=p.index('data')
        nbrs=[int(x) for x in p[i_nbrs+1:i_shape]]
        shape=[int(x) for x in p[i_shape+1:i_data]]
        data=np.array([float(x) for x in p[i_data+1:]])
        states[label][v]={'nbrs':nbrs,'shape':shape,'data':data}

D=4
# Compute inner(ψ_nl, ψ_nl) using same formula as inner(ψ_nl, ψ_n)
# but with ψb = ψa (NL with NL)

def build_T_self(v, label='NL'):
    s=states[label][v]
    nbrs=s['nbrs']; shape=s['shape']
    t=s['data'].reshape(shape)
    n=len(nbrs); chi_s=shape[:-1]
    # T[ia,ia'] for same state: T[ia,ib] = Σ_d t[ia,d]*t[ib,d]
    T=(t.reshape(-1,D)@t.reshape(-1,D).T).reshape(chi_s+chi_s)
    perm=[]; [perm.extend([k,n+k]) for k in range(n)]
    return np.transpose(T,perm), list(nbrs), chi_s, chi_s

# For chi=1 sites: contribution = Σ_d t[d]² = ||t||²
chi1_self = 1.0
for v in sorted(states['NL'].keys()):
    if not all(c==1 for c in states['NL'][v]['shape'][:-1]):
        continue
    t=states['NL'][v]['data'][:D]
    chi1_self *= float(np.dot(t,t))
print(f"chi=1 self-overlap product: {chi1_self:.6f}")

# chi>1 subgraph self-overlap (ψ_nl with ψ_nl)
# Using same tree structure
chi4_verts=[v for v in sorted(states['NL'].keys())
            if not all(c==1 for c in states['NL'][v]['shape'][:-1])]

def eff_T_self(v, keep_nbrs, label='NL'):
    T,nbrs,chi_as,chi_bs=build_T_self(v,label)
    n=len(nbrs)
    axes_to_sum=[]
    for k in range(n):
        if nbrs[k] not in keep_nbrs or chi_as[k]*chi_bs[k]==1:
            axes_to_sum.extend([2*k,2*k+1])
    cur=T
    for ax in sorted(axes_to_sum,reverse=True):
        cur=cur.sum(axis=ax)
    # Keep paired axes (chi_a, chi_b) for each remaining bond — don't flatten
    remain_nbrs=[nbrs[k] for k in range(n) if nbrs[k] in keep_nbrs and chi_as[k]*chi_bs[k]>1]
    remain_shape=[]
    for k in range(n):
        if nbrs[k] in keep_nbrs and chi_as[k]*chi_bs[k]>1:
            remain_shape.extend([chi_as[k], chi_bs[k]])
    return cur.reshape(remain_shape), remain_nbrs

chi4_bonds_map={31:[2],2:[33,31],33:[3,2],3:[33,34,35],35:[3],34:[3,4],4:[34]}
Teff={}; Teff_nbrs={}
for v in chi4_verts:
    T,nbrs=eff_T_self(v,chi4_bonds_map[v])
    Teff[v]=T; Teff_nbrs[v]=nbrs

def contract_bond(T_parent,nbrs_p,M_leaf,shared_v):
    """Contract M_leaf (chi_a×chi_b) into T_parent over shared_v bond."""
    idx=nbrs_p.index(shared_v)
    # T_parent has axes: (chi_a0,chi_b0, chi_a1,chi_b1,...) — pairs
    ia=2*idx; ib=2*idx+1
    ndim=T_parent.ndim
    if ndim <= ib:
        raise ValueError(f"T_parent ndim={ndim} but trying axes {ia},{ib}")
    axes_rest=[i for i in range(ndim) if i!=ia and i!=ib]
    t=np.transpose(T_parent,axes_rest+[ia,ib])
    rest=t.shape[:-2]; t=t.reshape(-1,t.shape[-2],t.shape[-1])
    if M_leaf.ndim==1:
        M_leaf=np.diag(M_leaf)  # treat 1D as diagonal matrix
    result=(t*M_leaf[None]).sum(axis=(-2,-1)).reshape(rest)
    new_nbrs=[n for i,n in enumerate(nbrs_p) if i!=idx]
    return result,new_nbrs

# Same sequential contraction as before
T31=Teff[31]; T2,n2=contract_bond(Teff[2],Teff_nbrs[2],T31,31)
T4=Teff[4];   T34,n34=contract_bond(Teff[34],Teff_nbrs[34],T4,4)
T35=Teff[35]; T3,n3=contract_bond(Teff[3],Teff_nbrs[3],T35,35)
T3,n3=contract_bond(T3,n3,T34,34)
T2_leaf=T2; T33,n33=contract_bond(Teff[33],Teff_nbrs[33],T2_leaf,2)
T3,n3=contract_bond(T3,n3,T33,33)
chi4_self=float(T3.sum())

print(f"chi>1 self-overlap: {chi4_self:.6f}")
C_self=chi1_self*chi4_self
print(f"Self-overlap total: {C_self:.6f}")
print(f"Julia ||ψ_nl||² = 0.99991")
print(f"Ratio our/Julia: {C_self/0.99991:.4f}")
