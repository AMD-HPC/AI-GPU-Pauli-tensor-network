"""
exact_overlap.py — Exact tree contraction using sequential contraction

Sites and their chi>1 bonds:
  31: nbrs=[1,2] chi=[1,4]  → chi=4 bond to 2
  2:  nbrs=[33,31] chi=[4,4] → chi=4 bonds to 33 and 31
  33: nbrs=[3,2]  chi=[4,4]  → chi=4 bonds to 3 and 2
  3:  nbrs=[33,34,35] chi=[4,4,4] → chi=4 bonds to all 3
  35: nbrs=[11,3] chi=[1,4]  → chi=4 bond to 3
  34: nbrs=[3,4]  chi=[4,4]  → chi=4 bonds to 3 and 4
  4:  nbrs=[36,34] chi=[1,4] → chi=4 bond to 34

T_v shape (after build_T): interleaved pairs (chi_a0,chi_b0, chi_a1,chi_b1,...)
For each chi=1 bond pair: already sum = 1, so we can sum those axes out immediately.

Strategy:
  1. For each site, sum out all chi=1 bond axes to get effective T of only chi=4 bonds
  2. Then do einsum over the chi=4 bond network (tree)
"""
import numpy as np

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

print(f"Target: {overlap_target:.6f}")
D=4

def build_T(v):
    sa=states['NL'][v]; sb=states['N'][v]
    nbrs=sa['nbrs']; sha=sa['shape']; shb=sb['shape']
    ta=sa['data'].reshape(sha); tb=sb['data'].reshape(shb)
    n=len(nbrs); chi_as=sha[:-1]; chi_bs=shb[:-1]
    T=(ta.reshape(-1,D)@tb.reshape(-1,D).T).reshape(list(chi_as)+list(chi_bs))
    perm=[]; [perm.extend([k,n+k]) for k in range(n)]
    return np.transpose(T,perm), list(nbrs), list(chi_as), list(chi_bs)

def effective_T(v, keep_nbrs):
    """Get T_v keeping only bonds to keep_nbrs, summing chi=1 bonds."""
    T,nbrs,chi_as,chi_bs=build_T(v)
    n=len(nbrs)
    # Sum out chi=1 bond pairs (both ia and ib axes)
    # T shape: (chi_a0,chi_b0, chi_a1,chi_b1,...)
    axes_to_sum=[]
    for k in range(n):
        if nbrs[k] not in keep_nbrs or chi_as[k]*chi_bs[k]==1:
            axes_to_sum.extend([2*k,2*k+1])
    cur=T
    for ax in sorted(axes_to_sum,reverse=True):
        cur=cur.sum(axis=ax)
    # Now flatten each remaining (ia,ib) pair
    remain_nbrs=[nbrs[k] for k in range(n) if nbrs[k] in keep_nbrs and chi_as[k]*chi_bs[k]>1]
    remain_chis=[chi_as[k]*chi_bs[k] for k in range(n) if nbrs[k] in keep_nbrs and chi_as[k]*chi_bs[k]>1]
    return cur.reshape(remain_chis), remain_nbrs

# chi=1 sites
log_C=0.; sign_C=1.
chi4_verts=[v for v in sorted(states['NL'].keys())
            if not all(c==1 for c in states['NL'][v]['shape'][:-1])]
for v in sorted(states['NL'].keys()):
    if v in chi4_verts: continue
    ta=states['NL'][v]['data'][:D]; tb=states['N'][v]['data'][:D]
    a_v=float(np.dot(ta,tb))
    if a_v==0: sign_C=0; break
    sign_C*=np.sign(a_v); log_C+=np.log(abs(a_v))
print(f"chi=1 contribution: {np.exp(log_C):.6f}")

# Get effective T for each chi>1 site, keeping only chi>1 bonds
chi4_bonds_map = {31:[2], 2:[33,31], 33:[3,2], 3:[33,34,35], 35:[3], 34:[3,4], 4:[34]}

# Build effective tensors (chi=1 bonds summed out, (ia,ib) pairs flattened)
Teff={}; Teff_nbrs={}
for v in chi4_verts:
    keep=[nbr for nbr in chi4_bonds_map[v]]
    T,nbrs=effective_T(v,keep)
    Teff[v]=T; Teff_nbrs[v]=nbrs
    print(f"Teff[{v}] shape={T.shape} nbrs={nbrs}")

# Sequential contraction: leaf→root on tree
# Leaf: site 31 (one bond A to site 2)
# Leaf: site 4  (one bond F to site 34)
# Leaf: site 35 (one bond E to site 3)

# Step 1: contract 31 into 2
A31 = Teff[31]  # shape (16,)
T2  = Teff[2]   # shape (16,16) — bonds [33,31]
idx31_in2 = Teff_nbrs[2].index(31)
T2_no31 = np.tensordot(T2, A31, axes=([idx31_in2],[0]))  # contracts A
print(f"\nAfter contracting 31 into 2: shape={T2_no31.shape}")

# Step 2: contract 4 into 34
A4  = Teff[4]   # shape (16,)
T34 = Teff[34]  # shape (16,16) — bonds [3,4]
idx4_in34 = Teff_nbrs[34].index(4)
T34_no4 = np.tensordot(T34, A4, axes=([idx4_in34],[0]))
print(f"After contracting 4 into 34: shape={T34_no4.shape}")

# Step 3: contract 35 into 3
A35 = Teff[35]  # shape (16,)
T3  = Teff[3]   # shape (16,16,16) — bonds [33,34,35]
idx35_in3 = Teff_nbrs[3].index(35)
T3_no35 = np.tensordot(T3, A35, axes=([idx35_in3],[0]))
print(f"After contracting 35 into 3: shape={T3_no35.shape}")
nbrs3_left = [n for n in Teff_nbrs[3] if n!=35]

# Step 4: contract 34 (now scalar bond to 3) into 3
idx34_in3 = nbrs3_left.index(34)
T3_no35_no34 = np.tensordot(T3_no35, T34_no4, axes=([idx34_in3],[0]))
print(f"After contracting 34 into 3: shape={T3_no35_no34.shape}")
nbrs3_left2 = [n for n in nbrs3_left if n!=34]

# Step 5: contract 2 (now scalar bond to 33) into 33
T33 = Teff[33]  # shape (16,16) — bonds [3,2]
idx2_in33 = Teff_nbrs[33].index(2)
T33_no2 = np.tensordot(T33, T2_no31, axes=([idx2_in33],[0]))
print(f"After contracting 2 into 33: shape={T33_no2.shape}")
nbrs33_left = [n for n in Teff_nbrs[33] if n!=2]

# Step 6: contract 33 (now scalar bond to 3) into 3
idx33_in3 = nbrs3_left2.index(33)
T3_final = np.tensordot(T3_no35_no34, T33_no2, axes=([idx33_in3],[0]))
print(f"After contracting 33 into 3: shape={T3_final.shape}")

chi4_contrib = float(T3_final.sum())
print(f"\nchi>1 exact contraction = {chi4_contrib:.6f}")

if sign_C==0 or chi4_contrib==0:
    print("C=0")
else:
    C=sign_C*np.sign(chi4_contrib)*np.exp(log_C+np.log(abs(chi4_contrib)))
    print(f"C = {C:.6f}  (target: {overlap_target:.6f})")
    print(f"Ratio: {C/overlap_target:.4f}")
