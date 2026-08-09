"""
test_direct_bethe.py — Bethe formula with CppSim psi + CppSim messages, NO transform

Julia's freenergy:
  C = Π_v vertex_scalar_v / Π_e edge_scalar_e
  vertex_scalar_v = contract(psi_v with ρ, all incoming messages)
  edge_scalar_e = Σ_ij msg_a2b[i,j] × msg_b2a[j,i] = Tr(msg_a2b × msg_b2a)

Use RAW CppSim psi and RAW CppSim messages. No gauge transform.
This is EXACTLY what bp_partitionfunction does.

Target: all-zero ρ → -0.961822
"""
import numpy as np

# Load messages
msgs = {}
with open('/tmp/msg_dump.txt') as f:
    for line in f:
        p=line.split()
        bid=int(p[1]);a=int(p[3]);b=int(p[5]);chi=int(p[7])
        i_a=p.index('a2b');i_b=p.index('b2a')
        a2b=np.array([float(x) for x in p[i_a+1:i_b]]).reshape(chi,chi)
        b2a=np.array([float(x) for x in p[i_b+1:]]).reshape(chi,chi)
        msgs[bid]={'a':a,'b':b,'chi':chi,'a2b':a2b,'b2a':b2a}

# Load psi
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

print("="*60)
print("  test_direct_bethe.py — raw psi + raw messages")
print("="*60)

def rho_allzero(sid):
    return np.array([0.,0.,0.,1.]) if sid==CENTER else np.array([1.,0.,0.,1.])

def incoming(bid, sid):
    m=msgs[bid]
    return m['b2a'] if sid==m['a'] else m['a2b']

# Vertex scalars: contract psi_v with rho and all incoming messages
log_num=0.0; sign_num=1.0
a_vals={}
for sid in range(68):
    s=psi[sid]; chis=s['chis']; bonds=s['bonds']
    t = s['psi'].reshape(chis+[D])
    # contract physical with rho
    t = np.tensordot(t, rho_allzero(sid), axes=([-1],[0]))  # shape = chis
    # contract each bond with incoming message
    for k,bid in enumerate(bonds):
        M = incoming(bid, sid)  # chi×chi
        t = np.tensordot(t, M, axes=([k],[0]))
        t = np.moveaxis(t, -1, k)
    a_v = float(t.sum())
    a_vals[sid]=a_v
    if a_v==0.0: sign_num=0.0; break
    sign_num*=np.sign(a_v); log_num+=np.log(abs(a_v))

for sid in [CENTER,32,33,34]:
    print(f"  a_v[{sid}] = {a_vals.get(sid,0):.6f}")

# Edge scalars: Tr(msg_a2b × msg_b2a)
log_den=0.0; sign_den=1.0
for bid,(u,v) in enumerate(EDGES):
    ze = float(np.trace(msgs[bid]['a2b'] @ msgs[bid]['b2a']))
    if abs(ze)>1e-30:
        sign_den*=np.sign(ze); log_den+=np.log(abs(ze))

if sign_num==0.0:
    print("C = 0")
else:
    C = sign_num*sign_den*np.exp(log_num-log_den)
    print(f"\nC = {C:.6f}  (Julia: -0.961822)")
    print(f"Ratio: {C/(-0.961822):.4f}")
