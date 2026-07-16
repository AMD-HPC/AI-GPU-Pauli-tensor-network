"""
test_two_state_overlap.py — Double-layer BP for ⟨ψ_a|ψ_b⟩ on full heavy-hex

Target (Julia): inner(ψ_nl, ψ_noisy; alg="bp") = 0.98092

Reads two_states_dump.txt (both states in shared site-index basis).

Double-layer construction:
  Site tensor: T_v[i,j,...,I,J,...] = Σ_d ψa_v[i,...,d] × ψb_v[j,...,d]
    where i,j are ψa,ψb bond indices for bond 0, etc.
  Combine (i,j) → single index of dim χa×χb per bond.
  BP message on bond e: matrix of dim (χa_e·χb_e) — but for BP we treat
  it as a vector over the combined index.

Actually the cleanest: for each bond, the double-layer bond index is
the PAIR (i_a, i_b). Message μ_{v→e}[i_a, i_b] is a χa×χb object.

BP update:
  μ_{v→e}[i_a,i_b] = Σ_{other bonds} T_v[...] × Π_{e'≠e} μ_{e'→v}[...]
Bethe:
  a_v = Σ_all T_v × Π_e μ_{e→v}
  ze_e = Σ_{i_a,i_b} μ_{a→b}[i_a,i_b] × μ_{b→a}[i_a,i_b]
"""
import numpy as np

# ── Load two states ────────────────────────────────────────────────────────
overlap_target = None
states = {'NL': {}, 'N': {}}
with open('two_states_dump.txt') as f:
    for line in f:
        p = line.split()
        if p[0] == 'OVERLAP':
            overlap_target = float(p[1])
            continue
        label = p[0]
        v = int(p[1])  # vertex (1-based from Julia)
        i_shape = p.index('shape')
        i_data = p.index('data')
        shape = [int(x) for x in p[i_shape+1:i_data]]
        data = np.array([float(x) for x in p[i_data+1:]])
        states[label][v] = data.reshape(shape)  # shape: (link0,link1,...,pauli)

print("="*60)
print("  test_two_state_overlap.py")
print(f"  Target: {overlap_target:.6f}")
print("="*60)

# Julia vertices are 1-based. Get sorted list.
verts = sorted(states['NL'].keys())
print(f"Loaded {len(verts)} sites")

# ── Build graph structure (need bond connectivity) ────────────────────────
# EDGES in 0-based; Julia verts are 1-based → convert
EDGES_0 = [(0,30),(0,31),(1,30),(1,32),(2,32),(2,33),(2,34),(3,33),(3,35),
(4,35),(4,36),(4,37),(5,36),(5,38),(6,38),(6,39),(7,40),(7,41),
(8,31),(8,42),(9,42),(9,43),(9,44),(10,34),(10,43),(10,45),
(11,45),(11,46),(11,47),(12,37),(12,46),(12,48),(13,48),(13,49),(13,50),
(14,39),(14,49),(14,51),(15,51),(15,52),(16,53),(16,54),
(17,44),(17,53),(17,55),(18,55),(18,56),(18,57),(19,47),(19,56),(19,58),
(20,58),(20,59),(20,60),(21,50),(21,59),(21,61),(22,40),(22,61),(22,62),
(23,52),(23,62),(24,54),(24,63),(25,63),(25,64),(26,57),(26,64),(26,65),
(27,65),(27,66),(28,60),(28,66),(28,67),(29,41),(29,67)]
# Convert to 1-based to match Julia vertex labels
EDGES = [(u+1, v+1) for (u,v) in EDGES_0]

# For each site: ordered list of (bond_id, neighbor)
# BUT: the tensor's link index order must match. Julia orders links by
# how they appear in inds(t). We need to know which link corresponds to
# which bond. This requires matching link dims to neighbor bond dims.
# For now: assume link order = sorted neighbor order (VERIFY THIS).

site_bonds = {v: [] for v in verts}
for bid, (u, w) in enumerate(EDGES):
    site_bonds[u].append((bid, w))
    site_bonds[w].append((bid, u))

# Print shape info to understand link ordering
print("\nSite tensor shapes (link dims, pauli):")
for v in [3, 33, 34, 35]:  # center + neighbors (1-based)
    if v in states['NL']:
        print(f"  site {v}: NL shape {states['NL'][v].shape}, neighbors {[n for _,n in site_bonds[v]]}")

print("\nNOTE: Need to verify link-index → bond mapping before BP.")
print("Julia orders links by inds(t) — may not match sorted neighbors.")
print("Next: match link dimensions to neighbor bond dimensions.")
