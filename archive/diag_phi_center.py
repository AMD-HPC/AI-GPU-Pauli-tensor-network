"""
diag_phi_center.py — Examine phi_center bond structure

Compare CppSim's phi_center marginals against Julia's converged messages.
This reveals whether the bond basis is rotated between the two.

Reads /tmp/psi_dump.txt (run inside docker where the dump lives).
"""
import numpy as np

psi = {}
with open('/tmp/psi_dump.txt') as f:
    for line in f:
        p = line.split()
        sid = int(p[1]); spin = int(p[3])
        chis=[]; bonds=[]; vals=[]; mode=None
        for x in p[4:]:
            if x=='chi': mode='c'; continue
            if x=='bond': mode='b'; continue
            if x=='psi': mode='p'; continue
            if mode=='c': chis.append(int(x))
            if mode=='b': bonds.append(int(x))
            if mode=='p': vals.append(float(x))
        psi[sid] = {'spin':spin, 'chis':chis, 'bonds':bonds, 'psi':np.array(vals)}

D = 4
CENTER = 2

# phi_center with Z selector (all-zero rho at center = Z observable)
t = psi[CENTER]['psi'].reshape(psi[CENTER]['chis'] + [D])
phi_Z = np.tensordot(t, np.array([0.,0.,0.,1.]), axes=([-1],[0]))  # (4,4,4)

print("="*60)
print("  diag_phi_center.py")
print("="*60)
print(f"\nphi_center chis: {psi[CENTER]['chis']}, bonds: {psi[CENTER]['bonds']}")
print(f"phi_center sum: {phi_Z.sum():.6f}  (Julia phi sum with rho: 1.6728)")

print("\n── Marginals (sum over other bonds) ─────────────────────────")
print(f"onto bond4 (sum 5,6): {np.round(phi_Z.sum(axis=(1,2)),4)}")
print(f"onto bond5 (sum 4,6): {np.round(phi_Z.sum(axis=(0,2)),4)}")
print(f"onto bond6 (sum 4,5): {np.round(phi_Z.sum(axis=(0,1)),4)}")

print("\n── Julia converged messages (from test_inner_minimal.jl) ────")
print("  (3,)→(33,): [1.073, 0.030, 0.165, -0.268]")
print("  (3,)→(34,): [0.995, -0.156, 0.108, 0.053]")
print("  (3,)→(35,): [1.088, -0.143, -0.054, 0.108]")

print("\n── phi_center slices ────────────────────────────────────────")
print(f"phi[0,0,:]: {np.round(phi_Z[0,0,:],4)}")
print(f"phi[:,0,0]: {np.round(phi_Z[:,0,0],4)}")
print(f"phi[0,:,0]: {np.round(phi_Z[0,:,0],4)}")
print(f"Julia phi[0,0,:]: [-1.19, 0.11, 0.0006, -0.19]")

# Also try device rho (Z at center, I/±Z elsewhere) — but for center it's Z either way
print("\n── phi with all-zero rho (I+Z at neighbors) ─────────────────")
# For neighbor sites, all-zero rho = [1,0,0,1]. But phi_center only depends on center's rho.
# The neighbor rho affects neighbor phi, which affects messages.

# Compute neighbor phi marginals
for nbr in [32, 33, 34]:
    tn = psi[nbr]['psi'].reshape(psi[nbr]['chis'] + [D])
    # all-zero rho at neighbor = [1,0,0,1]
    phi_n = np.tensordot(tn, np.array([1.,0.,0.,1.]), axes=([-1],[0]))
    print(f"phi_{nbr} shape {phi_n.shape}, sum={phi_n.sum():.4f}")
    if phi_n.ndim == 1:
        print(f"  phi_{nbr}: {np.round(phi_n,4)}")
    else:
        print(f"  phi_{nbr} marginal onto center bond: {np.round(phi_n.sum(axis=tuple(range(1,phi_n.ndim))),4)}")
