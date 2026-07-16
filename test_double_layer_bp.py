"""
test_double_layer_bp.py — General inner product ⟨φ_a|φ_b⟩ via double-layer BP

Gauge-independent construction. Works in QR gauge.

For inner(ψ, ρ):
  - ψ: operator TN, physical dim D=4, bond dims χ (QR gauge from CppSim)
  - ρ: product state, physical dim D=4, bond dim χ=1

Double-layer bilinear form at each site:
  T_v[χ_ψ_0,...,χ_ψ_k] = Σ_d ψ_v[χ_ψ..., d] × ρ_v[d]
  (ρ has χ=1 so its bonds are trivial → T_v = φ_v, the contracted tensor)

BP on the bilinear form network:
  message μ_{v→e} lives on bond e, dimension χ_ψ_e (since ρ χ=1)
  μ_{v→e}[i] = Σ_{χ except e} T_v[χ] × Π_{e'≠e} μ_{e'→v}[χ_{e'}]

Partition function (gauge-invariant):
  C = Π_v (Σ_χ T_v[χ] Π_e μ_{e→v}[χ_e]) / Π_e (Σ_i μ_{a→b}[i] μ_{b→a}[i])

KEY FIX: proper message initialization and update to avoid degenerate
fixed points. Initialize messages to the marginal of T_v, not uniform.

Target: inner(ψ, all-zero ρ) = -0.961822
"""
import numpy as np

# ── Load full psi tensors ─────────────────────────────────────────────────
psi={}
with open('/tmp/psi_dump.txt') as f:
    for line in f:
        p=line.split()
        sid=int(p[1]);spin=int(p[3])
        chis=[];bonds=[];vals=[];mode=None
        for x in p[4:]:
            if x=='chi':mode='c';continue
            if x=='bond':mode='b';continue
            if x=='psi':mode='p';continue
            if mode=='c':chis.append(int(x))
            if mode=='b':bonds.append(int(x))
            if mode=='p':vals.append(float(x))
        psi[sid]={'spin':spin,'chis':chis,'bonds':bonds,'psi':np.array(vals)}

D=4; CENTER=2
EDGES=[(0,30),(0,31),(1,30),(1,32),(2,32),(2,33),(2,34),(3,33),(3,35),
(4,35),(4,36),(4,37),(5,36),(5,38),(6,38),(6,39),(7,40),(7,41),
(8,31),(8,42),(9,42),(9,43),(9,44),(10,34),(10,43),(10,45),
(11,45),(11,46),(11,47),(12,37),(12,46),(12,48),(13,48),(13,49),(13,50),
(14,39),(14,49),(14,51),(15,51),(15,52),(16,53),(16,54),
(17,44),(17,53),(17,55),(18,55),(18,56),(18,57),(19,47),(19,56),(19,58),
(20,58),(20,59),(20,60),(21,50),(21,59),(21,61),(22,40),(22,61),(22,62),
(23,52),(23,62),(24,54),(24,63),(25,63),(25,64),(26,57),(26,64),(26,65),
(27,65),(27,66),(28,60),(28,66),(28,67),(29,41),(29,67)]

site_bonds=[[] for _ in range(68)]
for bid,(u,v) in enumerate(EDGES):
    site_bonds[u].append((bid,v))
    site_bonds[v].append((bid,u))

print("="*60)
print("  test_double_layer_bp.py")
print("="*60)

def rho_allzero(sid):
    return np.array([0.,0.,0.,1.]) if sid==CENTER else np.array([1.,0.,0.,1.])

# ── Build bilinear form tensors T_v = phi_v ───────────────────────────────
T={}
for sid in range(68):
    s=psi[sid]; chis=s['chis']
    t=s['psi'].reshape(chis+[D])
    T[sid]=np.tensordot(t, rho_allzero(sid), axes=([-1],[0]))  # shape=chis

# ── Message passing ────────────────────────────────────────────────────────
# msg[(bid,src)] = outgoing chi-vector from src along bond bid
# Initialize to ONES (not normalized yet) to avoid killing information
msg={}
for bid,(u,v) in enumerate(EDGES):
    for sid in [u,v]:
        k=[i for i,(b,n) in enumerate(site_bonds[sid]) if b==bid][0]
        chi_k=T[sid].shape[k] if T[sid].ndim>0 else 1
        msg[(bid,sid)]=np.ones(chi_k)

def get_inc(bid,sid):
    u,v=EDGES[bid]; nbr=v if sid==u else u
    return msg[(bid,nbr)]

# BP sweeps — NO per-sweep normalization (only at the end for numerical safety)
# This keeps the exact tree contraction on the star subgraph
print("\n── BP sweeps (unnormalized) ─────────────────────────────────")
for sweep in range(50):
    max_delta=0.
    new_msg={}
    for bid,(ua,vb) in enumerate(EDGES):
        for src in [ua,vb]:
            sh=T[src].shape
            if len(sh)==0:
                new_msg[(bid,src)]=np.array([1.0]); continue
            skip=[i for i,(b,n) in enumerate(site_bonds[src]) if b==bid][0]
            t=T[src].copy()
            for k,(b2,n2) in enumerate(site_bonds[src]):
                if k==skip: continue
                m=get_inc(b2,src)
                shape=[1]*len(sh); shape[k]=len(m)
                t=t*m.reshape(shape)
            axes=[i for i in range(len(sh)) if i!=skip]
            nm=t
            for ax in sorted(axes,reverse=True):
                nm=nm.sum(axis=ax)
            new_msg[(bid,src)]=nm
    # Update and measure delta (normalize only for delta comparison)
    for key,nm in new_msg.items():
        old=msg[key]
        n_old=np.linalg.norm(old); n_new=np.linalg.norm(nm)
        if n_old>1e-30 and n_new>1e-30:
            d=np.abs(nm/n_new - old/n_old).max()
            max_delta=max(max_delta,d)
        msg[key]=nm
    if max_delta<1e-8:
        print(f"Converged sweep {sweep+1} (delta={max_delta:.2e})")
        break

# Print center message (normalized for comparison)
bid_c=site_bonds[CENTER][0][0]
mc=msg[(bid_c,CENTER)]
print(f"\nmsg center→bond{bid_c} (normalized by sum): {np.round(mc/mc.sum(),4)}")
print(f"Julia (3,)→(33,):                          [1.073, 0.030, 0.165, -0.268]")

# ── Partition function ─────────────────────────────────────────────────────
# C = Π_v a_v / Π_e ze_e  (in log space with signs)
log_num=0.;sign_num=1.
a_vals={}
for sid in range(68):
    sh=T[sid].shape
    t=T[sid].copy()
    if len(sh)>0:
        for k,(bid,n) in enumerate(site_bonds[sid]):
            m=get_inc(bid,sid)
            shape=[1]*len(sh);shape[k]=len(m)
            t=t*m.reshape(shape)
    a_v=float(t.sum()) if len(sh)>0 else float(T[sid])
    a_vals[sid]=a_v
    if a_v==0.:sign_num=0.;break
    sign_num*=np.sign(a_v);log_num+=np.log(abs(a_v))

for sid in [CENTER,32,33,34]:
    print(f"  a_v[{sid}]={a_vals.get(sid,0):.6f}")

log_den=0.;sign_den=1.
for bid,(u,v) in enumerate(EDGES):
    ma=msg[(bid,u)];mb=msg[(bid,v)]
    ze=float(np.dot(ma,mb))
    if abs(ze)>1e-30:
        sign_den*=np.sign(ze);log_den+=np.log(abs(ze))

if sign_num==0.:print("\nC=0")
else:
    C=sign_num*sign_den*np.exp(log_num-log_den)
    print(f"\nC = {C:.6f}  (Julia: -0.961822)")
    print(f"Ratio: {C/(-0.961822):.4f}")
