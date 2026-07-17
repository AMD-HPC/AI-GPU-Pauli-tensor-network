"""
bethe_with_blf_msgs.py — Bethe formula using Julia's converged BilinearForm messages

Julia confirms: BLF partition function = 0.980920 using converged messages.
We replicate this in Python using the dumped messages.

C = Π_v vertex_scalar_v / Π_e edge_scalar_e
  vertex_scalar_v = Σ_χ T_v[χ] × Π_e msg_{e→v}[χ_e]
  edge_scalar_e   = Σ_{ia,ib} msg_fwd[ia,ib] × msg_bwd[ia,ib]

Target: 0.980920
"""
import numpy as np

# ── Load states ────────────────────────────────────────────────────────────
overlap_target=None; states={'NL':{},'N':{}}
with open('two_states_dump.txt') as f:
    for line in f:
        p=line.split()
        if p[0]=='OVERLAP': overlap_target=float(p[1]); continue
        label=p[0]; v=int(p[1])
        i_nbrs=p.index('nbrs'); i_shape=p.index('shape'); i_data=p.index('data')
        nbrs=[int(x) for x in p[i_nbrs+1:i_shape]]
        shape=[int(x) for x in p[i_shape+1:i_data]]
        data=np.array([float(x) for x in p[i_data+1:]])
        states[label][v]={'nbrs':nbrs,'shape':shape,'data':data}

# ── Load BLF messages ──────────────────────────────────────────────────────
# msg[(src,dst)] = matrix M[ia,ib] shape (chi_a,chi_b)
blf_msgs={}
with open('blf_messages_dump.txt') as f:
    for line in f:
        p=line.split()
        if p[0]=='TARGET': continue
        if p[0]!='MSG': continue
        src=int(p[1]); dst=int(p[2])
        i_size=p.index('size'); i_data=p.index('data')
        sz=[int(x) for x in p[i_size+1:i_data]]
        data=np.array([float(x) for x in p[i_data+1:]])
        blf_msgs[(src,dst)]=data.reshape(sz)

print(f"Loaded {len(blf_msgs)} BLF messages")
print(f"Target: {overlap_target:.6f}")

D=4; verts=sorted(states['NL'].keys())

# ── Build T_v = Σ_d ψa_v[bonds,d] × ψb_v[bonds,d] ────────────────────────
T={}; T_nbrs={}
for v in verts:
    sa=states['NL'][v]; sb=states['N'][v]
    nbrs=sa['nbrs']; sha=sa['shape']; shb=sb['shape']
    ta=sa['data'].reshape(sha); tb=sb['data'].reshape(shb)
    n=len(nbrs); chi_as=sha[:-1]; chi_bs=shb[:-1]
    Tbig=(ta.reshape(-1,D)@tb.reshape(-1,D).T).reshape(list(chi_as)+list(chi_bs))
    perm=[]; [perm.extend([k,n+k]) for k in range(n)]
    T[v]=np.transpose(Tbig,perm)
    T_nbrs[v]=nbrs

def get_msg(v, nbr):
    """Get incoming message to v from nbr."""
    key=(nbr,v)
    if key in blf_msgs: return blf_msgs[key]
    # If not found, use identity (chi=1 bonds)
    return np.array([[1.0]])

# ── Bethe partition function ───────────────────────────────────────────────
log_num=0.; sign_num=1.
a_vals={}
for v in verts:
    nbrs_v=T_nbrs[v]; n=len(nbrs_v)
    T_v=T[v]
    # Contract T_v with all incoming messages (in reverse order to keep axes stable)
    cur=T_v.copy()
    for k in range(n-1,-1,-1):
        nbr=nbrs_v[k]
        M=get_msg(v,nbr)  # (chi_a,chi_b)
        ia_ax=2*k; ib_ax=2*k+1
        ndim=cur.ndim
        axes_rest=[i for i in range(ndim) if i!=ia_ax and i!=ib_ax]
        t=np.transpose(cur,axes_rest+[ia_ax,ib_ax])
        rest=t.shape[:-2]
        cur=(t.reshape(-1,t.shape[-2],t.shape[-1])*M[None]).sum(axis=(-2,-1)).reshape(rest)
    a_v=float(cur.sum()) if cur.ndim>0 else float(cur)
    a_vals[v]=a_v
    if a_v==0.: sign_num=0.; break
    sign_num*=np.sign(a_v); log_num+=np.log(abs(a_v))

# Print key vertex scalars
for v in [3,33,34,35,31,2,4]:
    if v in a_vals: print(f"  a_v[{v}]={a_vals[v]:.6f}")

# Edge scalars: ze_e = Σ_{ia,ib} msg_fwd[ia,ib] × msg_bwd[ia,ib]
log_den=0.; sign_den=1.
ze_vals={}
edges_seen=set()
for (src,dst) in blf_msgs:
    key=tuple(sorted([src,dst]))
    if key in edges_seen: continue
    edges_seen.add(key)
    M_fwd=blf_msgs[(src,dst)]
    M_bwd=blf_msgs.get((dst,src), np.ones_like(M_fwd))
    ze=float(np.sum(M_fwd*M_bwd))
    ze_vals[key]=ze
    if abs(ze)>1e-30:
        sign_den*=np.sign(ze); log_den+=np.log(abs(ze))

print(f"\nAll ze values:")
for k,v in sorted(ze_vals.items()):
    print(f"  ze{k}={v:.6f}")

if sign_num==0.: print("C=0")
else:
    C=sign_num*sign_den*np.exp(log_num-log_den)
    print(f"\nC = {C:.6f}  (target: {overlap_target:.6f})")
    print(f"Ratio: {C/overlap_target:.4f}")
