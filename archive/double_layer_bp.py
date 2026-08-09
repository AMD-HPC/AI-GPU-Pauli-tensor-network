"""
double_layer_bp.py — Two-state overlap ⟨ψ_a|ψ_b⟩ via double-layer BP

Correct formulation:
  T_v[ia0,ib0, ia1,ib1,...] = Σ_d ψa[ia0,ia1,...,d] × ψb[ib0,ib1,...,d]
  Message for bond e: matrix M[ia,ib] of shape chi_a×chi_b
  (NOT a flattened vector — Julia keeps the matrix structure)

BP update:
  new_M_out[ia,ib] = Σ_{(ia_j,ib_j) for j≠skip} T_v × Π_{j≠skip} M_j[ia_j,ib_j]

Init: delta(ia,ib) = identity matrix (Julia's default_message)

Target: 0.980920
"""
import numpy as np

# ── Load dump ─────────────────────────────────────────────────────────────
overlap_target = None
states = {'NL': {}, 'N': {}}
with open('two_states_dump.txt') as f:
    for line in f:
        p = line.split()
        if p[0] == 'OVERLAP':
            overlap_target = float(p[1]); continue
        label = p[0]; v = int(p[1])
        i_nbrs=p.index('nbrs'); i_shape=p.index('shape'); i_data=p.index('data')
        nbrs=[int(x) for x in p[i_nbrs+1:i_shape]]
        shape=[int(x) for x in p[i_shape+1:i_data]]
        data=np.array([float(x) for x in p[i_data+1:]])
        states[label][v]={'nbrs':nbrs,'shape':shape,'data':data}

print("="*60)
print(f"  double_layer_bp.py  target={overlap_target:.6f}")
print("="*60)
D=4
verts=sorted(states['NL'].keys())

# Bond chi_a, chi_b
bonds={}
for v,s in states['NL'].items():
    for k,nbr in enumerate(s['nbrs']):
        key=(min(v,nbr),max(v,nbr))
        if key not in bonds:
            bonds[key]=(s['shape'][k], states['N'][v]['shape'][k])

print(f"Loaded {len(verts)} sites, {len(bonds)} bonds")
print(f"chi>1 bonds: {[(k,v) for k,v in bonds.items() if v[0]>1][:5]}")

# ── Build T_v with alternating (ia,ib) axes ───────────────────────────────
# T_v[ia0,ib0, ia1,ib1,...] = Σ_d ψa[ia0,ia1,...,d] × ψb[ib0,ib1,...,d]
T={}; T_nbrs={}; T_chi_a={}; T_chi_b={}

for v in verts:
    sa=states['NL'][v]; sb=states['N'][v]
    nbrs=sa['nbrs']
    sha=sa['shape']; shb=sb['shape']  # both end in D
    ta=sa['data'].reshape(sha)
    tb=sb['data'].reshape(shb)
    n=len(nbrs)
    chi_as=sha[:-1]; chi_bs=shb[:-1]

    # T[ia0,ia1,...,ib0,ib1,...] = Σ_d ta[ia...d] × tb[ib...d]
    ta_f=ta.reshape(-1,D); tb_f=tb.reshape(-1,D)
    T_big = ta_f @ tb_f.T  # (Π chi_a, Π chi_b)
    T_big=T_big.reshape(list(chi_as)+list(chi_bs))  # (ia0,ia1,...,ib0,ib1,...)

    # Reorder to interleaved: (ia0,ib0, ia1,ib1, ...)
    perm = []
    for k in range(n): perm += [k, n+k]
    T_v = np.transpose(T_big, perm)  # (ia0,ib0, ia1,ib1,...)
    # Shape: (chi_a0,chi_b0, chi_a1,chi_b1,...)
    T[v]=T_v; T_nbrs[v]=nbrs
    T_chi_a[v]=list(chi_as); T_chi_b[v]=list(chi_bs)

print(f"T[3] shape: {T[3].shape}  T[33] shape: {T[33].shape}")

# ── BP messages: chi_a × chi_b matrices ───────────────────────────────────
# msg[(src,dst)] = matrix M[ia,ib] of shape chi_a × chi_b
msg={}
for (v1,v2),(chi_a,chi_b) in bonds.items():
    # Julia init: delta(ia,ib) = identity (or rectangular identity)
    M=np.eye(chi_a,chi_b)  # chi_a×chi_b, ones on diagonal
    s=M.sum(); M=M/s if s>0 else M
    msg[(v1,v2)]=M.copy(); msg[(v2,v1)]=M.copy()  # both directions same init

def get_inc_M(v,nbr):
    """Incoming message to v from nbr as chi_a×chi_b matrix."""
    return msg[(nbr,v)]  # msg from nbr to v

# ── BP sweeps ─────────────────────────────────────────────────────────────
print("\n── BP sweeps ────────────────────────────────────────────────")
for sweep in range(50):
    max_delta=0.
    new_msg={}
    for v in verts:
        nbrs_v=T_nbrs[v]; n=len(nbrs_v)
        chi_as=T_chi_a[v]; chi_bs=T_chi_b[v]
        T_v=T[v]  # shape: (chi_a0,chi_b0, chi_a1,chi_b1, ...)

        for skip_k,skip_nbr in enumerate(nbrs_v):
            chi_a_out=chi_as[skip_k]; chi_b_out=chi_bs[skip_k]

            # Contract T_v with all incoming messages except skip_k
            # T_v has axes (ia0,ib0, ia1,ib1, ...)
            # Contract axis pair (2k, 2k+1) for k≠skip_k with M[ia,ib]
            cur=T_v.copy()
            ndim=cur.ndim

            for k in range(n-1,-1,-1):  # reverse to avoid axis shifting
                if k==skip_k: continue
                M=get_inc_M(v,nbrs_v[k])  # chi_a×chi_b
                ia_ax=2*k; ib_ax=2*k+1
                # Contract: Σ_{ia,ib} cur[...,ia,...,ib,...] × M[ia,ib]
                # First contract ia axis with M's first axis
                cur=np.tensordot(cur, M, axes=([ia_ax],[0]))
                # Result has ib dimension moved to end, ia contracted
                # Move result axis (was ib_ax, now shifted) and the new M axis
                # After tensordot over ia_ax: result shape has ib_ax at position ia_ax,
                # and M's second axis appended at end
                # We want to contract ib_ax too (now at ia_ax position) with M's result axis
                # Actually tensordot gives: result[..., ib_dim, M_second_axis]
                # We need to contract ib with M_second... no, M[ia,ib] is fully contracted now
                # Wait: np.tensordot(cur, M, axes=([ia_ax],[0])) sums over ia,
                # result = Σ_ia cur[...,ia,...] * M[ia,ib_out]
                # Result shape: cur_shape_without_ia + (chi_b_out,)
                # Now contract ib axis (which shifted left by 1 since ia was before it):
                ib_new = ia_ax  # ib was at ia_ax+1, now at ia_ax after removing ia
                cur2=np.tensordot(cur, np.eye(chi_bs[k]), axes=([ib_new],[0]))
                # Wait, we want to marginalize ib too, not keep it
                # Actually M[ia,ib] contracts BOTH ia and ib simultaneously
                # Let me redo: contract axes (ia_ax, ib_ax) together with M

            # Restart - correct approach: einsum over both axes simultaneously
            cur=T_v.copy()
            for k in range(n-1,-1,-1):
                if k==skip_k: continue
                M=get_inc_M(v,nbrs_v[k])  # (chi_a, chi_b)
                ia_ax=2*k; ib_ax=2*k+1
                # Contract cur[...,ia,...,ib,...] with M[ia,ib] -> sum over ia,ib
                # Result: cur with axes ia_ax and ib_ax removed
                # Use einsum-like approach:
                ndim_cur=cur.ndim
                # Move ia_ax and ib_ax to the end
                axes_rest=[i for i in range(ndim_cur) if i!=ia_ax and i!=ib_ax]
                cur=np.transpose(cur, axes_rest+[ia_ax,ib_ax])
                rest_shape=cur.shape[:-2]; ca=cur.shape[-2]; cb=cur.shape[-1]
                cur=cur.reshape(-1,ca,cb)
                # Contract with M: result[batch] = Σ_{ia,ib} cur[batch,ia,ib]*M[ia,ib]
                cur=(cur * M[None,:,:]).sum(axis=(-2,-1))
                cur=cur.reshape(rest_shape)

            # cur now has shape (chi_a_skip, chi_b_skip) = outgoing message
            nm=cur
            # Normalize by sum
            s=nm.sum()
            if abs(s)>1e-15: nm=nm/s

            key=(v,skip_nbr)
            old=msg.get(key,np.zeros_like(nm))
            if old.shape==nm.shape:
                max_delta=max(max_delta,np.abs(nm-old).max())
            new_msg[key]=nm

    msg.update(new_msg)
    if max_delta<1e-6:
        print(f"  Converged at sweep {sweep+1} (delta={max_delta:.2e})"); break
else:
    print(f"  Max sweeps reached (delta={max_delta:.2e})")

# Print a message for comparison
key=(3,33)
if key in msg:
    print(f"\nmsg (3)→(33): {np.round(msg[key].flatten(),4)}")
    print(f"Julia (3,)→(33,): [1.073, 0.030, 0.165, -0.268]")

# ── Bethe partition function ───────────────────────────────────────────────
print("\n── Bethe partition function ─────────────────────────────────")
log_num=0.; sign_num=1.
a_vals={}
for v in verts:
    nbrs_v=T_nbrs[v]; n=len(nbrs_v)
    chi_as=T_chi_a[v]; chi_bs=T_chi_b[v]
    cur=T[v].copy()
    for k in range(n-1,-1,-1):
        M=get_inc_M(v,nbrs_v[k])
        ia_ax=2*k; ib_ax=2*k+1
        ndim_cur=cur.ndim
        axes_rest=[i for i in range(ndim_cur) if i!=ia_ax and i!=ib_ax]
        cur=np.transpose(cur,axes_rest+[ia_ax,ib_ax])
        rest_shape=cur.shape[:-2]; ca=cur.shape[-2]; cb=cur.shape[-1]
        cur=cur.reshape(-1,ca,cb)
        cur=(cur*M[None,:,:]).sum(axis=(-2,-1))
        cur=cur.reshape(rest_shape)
    a_v=float(cur.sum()) if cur.ndim>0 else float(cur)
    a_vals[v]=a_v
    if a_v==0.: sign_num=0.; break
    sign_num*=np.sign(a_v); log_num+=np.log(abs(a_v))

for v in [3,33,34,35]:
    print(f"  a_v[{v}]={a_vals.get(v,0):.6f}")

log_den=0.; sign_den=1.
for (v1,v2),(chi_a,chi_b) in bonds.items():
    Ma=msg.get((v1,v2),np.eye(chi_a,chi_b))
    Mb=msg.get((v2,v1),np.eye(chi_b,chi_a))
    ze=float(np.sum(Ma*Mb))  # Σ_{ia,ib} Ma[ia,ib]*Mb[ia,ib] (shared indices)
    if abs(ze)>1e-30:
        sign_den*=np.sign(ze); log_den+=np.log(abs(ze))

if sign_num==0.: print("C=0")
else:
    C=sign_num*sign_den*np.exp(log_num-log_den)
    print(f"\nC = {C:.6f}  (target: {overlap_target:.6f})")
    print(f"Ratio: {C/overlap_target:.4f}")
