"""
test_chivec_bp.py — chi-vector BP on phi_v (BilinearForm ψ⊗ρ)

Julia's messages for inner(ψ,ρ) are chi-VECTORS (ρ has chi=1).
Julia converged msg (3,)→(33,) = [1.073, 0.030, 0.165, -0.268]

Compute this from CppSim's phi_v (ψ contracted with ρ).
phi_v is the BilinearForm factor: shape (chi_0,...,chi_k) — no physical index.

BP message update (chi-vector):
  msg_{v→e}[i] = Σ_{χ except χ_e=i} phi_v[χ] × Π_{e'≠e} msg_{e'→v}[χ_{e'}]
  normalize by sum (Julia convention)

Then Bethe:
  a_v = Σ_χ phi_v[χ] × Π_e msg_{e→v}[χ_e]
  ze_e = Σ_i msg_a2b[i] × msg_b2a[i]

Target: inner(ψ, all-zero ρ) = -0.961822
"""
import numpy as np

# Load psi (full, to compute phi with all-zero rho)
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
print("  test_chivec_bp.py")
print("="*60)

def rho_allzero(sid):
    return np.array([0.,0.,0.,1.]) if sid==CENTER else np.array([1.,0.,0.,1.])

# Build phi_v (contract psi with rho)
phi={}
for sid in range(68):
    s=psi[sid]; chis=s['chis']
    t = s['psi'].reshape(chis+[D])
    phi[sid] = np.tensordot(t, rho_allzero(sid), axes=([-1],[0]))  # shape chis

print(f"phi_center sum: {phi[CENTER].sum():.6f}")

# chi-vector messages
msg={}  # (bid, src) → chi-vector
for bid,(u,v) in enumerate(EDGES):
    c_u = len(phi[u].shape)  # ndim = number of bonds
    # chi of this bond at u
    for sid in [u,v]:
        k = [i for i,(b,n) in enumerate(site_bonds[sid]) if b==bid][0]
        chi_k = phi[sid].shape[k] if phi[sid].ndim>0 else 1
        msg[(bid,sid)] = np.ones(chi_k)/chi_k

def get_inc(bid, sid):
    u,v = EDGES[bid]
    nbr = v if sid==u else u
    return msg[(bid,nbr)]

# BP sweeps
for sweep in range(50):
    max_delta=0.
    for bid,(ua,vb) in enumerate(EDGES):
        for src in [ua,vb]:
            sh = phi[src].shape
            if len(sh)==0: continue  # scalar site
            skip = [i for i,(b,n) in enumerate(site_bonds[src]) if b==bid][0]
            chi_out = sh[skip]
            # contract phi with incoming messages except skip
            t = phi[src].copy()
            for k,(b2,n2) in enumerate(site_bonds[src]):
                if k==skip: continue
                m = get_inc(b2, src)
                shape=[1]*len(sh); shape[k]=len(m)
                t = t * m.reshape(shape)
            # sum all except skip
            axes=[i for i in range(len(sh)) if i!=skip]
            nm = t
            for ax in sorted(axes,reverse=True):
                nm = nm.sum(axis=ax)
            # normalize by sum
            s = nm.sum()
            if abs(s)>1e-15: nm=nm/s
            old = msg[(bid,src)]
            max_delta=max(max_delta, np.abs(nm-old).max())
            msg[(bid,src)] = nm
    if max_delta<1e-6:
        print(f"Converged sweep {sweep+1}")
        break

# Print center message
bid_c = site_bonds[CENTER][0][0]  # first bond of center
print(f"\nmsg center→bond{bid_c}: {np.round(msg[(bid_c,CENTER)],4)}")
print(f"Julia (3,)→(33,):      [1.073, 0.030, 0.165, -0.268]")

# Bethe
log_num=0.;sign_num=1.
a_vals={}
for sid in range(68):
    sh=phi[sid].shape
    t=phi[sid].copy()
    if len(sh)>0:
        for k,(bid,n) in enumerate(site_bonds[sid]):
            m=get_inc(bid,sid)
            shape=[1]*len(sh);shape[k]=len(m)
            t=t*m.reshape(shape)
    a_v=float(t.sum())
    a_vals[sid]=a_v
    if a_v==0.:sign_num=0.;break
    sign_num*=np.sign(a_v);log_num+=np.log(abs(a_v))

for sid in [CENTER,32,33,34]:
    print(f"  a_v[{sid}]={a_vals.get(sid,0):.6f}")
print(f"  (Julia a_center=-6.618)")

log_den=0.;sign_den=1.
for bid,(u,v) in enumerate(EDGES):
    ma=msg[(bid,u)]; mb=msg[(bid,v)]
    ze=float(np.dot(ma,mb))
    if abs(ze)>1e-30:
        sign_den*=np.sign(ze);log_den+=np.log(abs(ze))

if sign_num==0.:print("C=0")
else:
    C=sign_num*sign_den*np.exp(log_num-log_den)
    print(f"\nC = {C:.6f}  (Julia: -0.961822)")
    print(f"Ratio: {C/(-0.961822):.4f}")
