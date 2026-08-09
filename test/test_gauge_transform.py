"""
test_gauge_transform.py — QR gauge → symmetric gauge, then inner product

CppSim messages are DIAGONAL and symmetric (a2b == b2a).
Bond 4: diag(0.889, 0.346, 0.213, 0.213)

Symmetric gauge transform (Vidal):
  The BP message M_e is the "environment". In symmetric gauge, each site
  tensor absorbs sqrt(M_e) on each bond. The inner product then uses
  identity messages (environment already in tensors).

  ψ_sym_v[χ...,d] = ψ_v[χ...,d] × Π_bonds sqrt(M_e)[χ_e]

  Then: C = Π_v a_v / Π_e ze_e
    a_v  = Σ_χ,d ψ_sym_v[χ,d] × ρ_v[d]   (sum all, messages=identity)
    ze_e = Σ_i sqrt(M)[i] × sqrt(M)[i] = Σ_i M[i] = Tr(M)

Reads /tmp/psi_dump.txt and /tmp/msg_dump.txt.
Target: inner(ψ, all-zero ρ) = -0.961822
"""

import numpy as np

print("="*60)
print("  test_gauge_transform.py")
print("="*60)

# ── Load messages (diagonal) ──────────────────────────────────────────────
msgs = {}
with open('/tmp/msg_dump.txt') as f:
    for line in f:
        p = line.split()
        bid=int(p[1]); a=int(p[3]); b=int(p[5]); chi=int(p[7])
        i_a2b=p.index('a2b'); i_b2a=p.index('b2a')
        a2b=np.array([float(x) for x in p[i_a2b+1:i_b2a]]).reshape(chi,chi)
        b2a=np.array([float(x) for x in p[i_b2a+1:]]).reshape(chi,chi)
        msgs[bid]={'a':a,'b':b,'chi':chi,'a2b':a2b,'b2a':b2a}

# ── Load full psi tensors ─────────────────────────────────────────────────
psi = {}
with open('/tmp/psi_dump.txt') as f:
    for line in f:
        p = line.split()
        sid=int(p[1]); spin=int(p[3])
        chis=[]; bonds=[]; vals=[]; mode=None
        for x in p[4:]:
            if x=='chi': mode='c'; continue
            if x=='bond': mode='b'; continue
            if x=='psi': mode='p'; continue
            if mode=='c': chis.append(int(x))
            if mode=='b': bonds.append(int(x))
            if mode=='p': vals.append(float(x))
        psi[sid]={'spin':spin,'chis':chis,'bonds':bonds,'psi':np.array(vals)}

print(f"Loaded {len(psi)} psi tensors, {len(msgs)} messages")

D=4
CENTER=2
EDGES = [
    (0,30),(0,31),(1,30),(1,32),(2,32),(2,33),(2,34),(3,33),(3,35),
    (4,35),(4,36),(4,37),(5,36),(5,38),(6,38),(6,39),(7,40),(7,41),
    (8,31),(8,42),(9,42),(9,43),(9,44),(10,34),(10,43),(10,45),
    (11,45),(11,46),(11,47),(12,37),(12,46),(12,48),(13,48),(13,49),(13,50),
    (14,39),(14,49),(14,51),(15,51),(15,52),(16,53),(16,54),
    (17,44),(17,53),(17,55),(18,55),(18,56),(18,57),(19,47),(19,56),(19,58),
    (20,58),(20,59),(20,60),(21,50),(21,59),(21,61),(22,40),(22,61),(22,62),
    (23,52),(23,62),(24,54),(24,63),(25,63),(25,64),(26,57),(26,64),(26,65),
    (27,65),(27,66),(28,60),(28,66),(28,67),(29,41),(29,67),
]

def spin_of(sid):
    return psi[sid]['spin']

def rho_vec(sid):
    """All-zero rho: [1,0,0,+1] for spin+1, [1,0,0,-1] for spin-1, Z at center."""
    if sid == CENTER:
        return np.array([0.,0.,0.,1.])  # Z observable
    # all-zero rho means all qubits |0> = [1,0,0,+1] regardless of device spin
    return np.array([1.,0.,0.,1.])

# ── Gauge transform: absorb sqrt(M) into each site ────────────────────────
print("\n── Transform QR → symmetric gauge ──────────────────────────")

def sqrt_diag(M):
    """sqrt of diagonal matrix, preserve sign."""
    d = np.diag(M)
    return np.sqrt(np.abs(d))  # return as vector

psi_sym = {}
for sid in range(68):
    s = psi[sid]
    chis = s['chis']
    bonds = s['bonds']
    t = s['psi'].reshape(chis + [D])

    # Absorb sqrt(M_e) along each bond axis
    for k, bid in enumerate(bonds):
        # incoming message to sid on bond bid
        m = msgs[bid]
        M = m['b2a'] if sid==m['a'] else m['a2b']  # message coming INTO sid
        sq = sqrt_diag(M)  # chi-vector
        # multiply along axis k
        shape = [1]*len(t.shape)
        shape[k] = len(sq)
        t = t * sq.reshape(shape)

    psi_sym[sid] = t

# Check center
phi_c = np.tensordot(psi_sym[CENTER], rho_vec(CENTER), axes=([-1],[0]))
print(f"phi_sym_center sum: {phi_c.sum():.6f}  (Julia: 1.6728)")
print(f"phi_sym_center[0,0,:]: {phi_c[0,0,:] if phi_c.ndim>=2 else phi_c}")

# ── Compute inner product in symmetric gauge ──────────────────────────────
# In symmetric gauge: messages = identity, so
#   a_v = Σ_χ,d psi_sym_v[χ,d] × rho_v[d]  (sum over all bond indices)
#   ze_e = Tr(sqrt(M)·sqrt(M)) = Tr(M) = Σ_i M[i,i]
print("\n── Inner product (symmetric gauge, identity messages) ──────")

log_num=0.0; sign_num=1.0
a_vals = {}
for sid in range(68):
    phi_v = np.tensordot(psi_sym[sid], rho_vec(sid), axes=([-1],[0]))
    a_v = float(phi_v.sum())
    a_vals[sid] = a_v
    if a_v==0.0: sign_num=0.0; break
    sign_num *= np.sign(a_v)
    log_num += np.log(abs(a_v))

for sid in [CENTER,32,33,34]:
    print(f"  a_v[{sid}] = {a_vals.get(sid,0):.6f}")

# ze_e = Tr(M) for each bond
log_den=0.0; sign_den=1.0
for bid,(u,v) in enumerate(EDGES):
    M = msgs[bid]['a2b']
    ze = float(np.trace(M))  # Tr(sqrt(M)·sqrt(M)) = Tr(M)
    if abs(ze)>1e-30:
        sign_den *= np.sign(ze)
        log_den += np.log(abs(ze))

if sign_num==0.0:
    print("C = 0")
else:
    C = sign_num*sign_den*np.exp(log_num-log_den)
    print(f"\nC (symmetric gauge) = {C:.6f}  (Julia: -0.961822)")
    print(f"Ratio: {C/(-0.961822):.4f}")
