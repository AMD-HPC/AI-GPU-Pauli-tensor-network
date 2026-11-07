# Vector Field Detection — Exploratory Notes (2026-11-06)

## Motivation

The standard detection observable L(v,t) = -log(n_pert/n_base) is a scalar
field on sites. It identifies the broken bond by a sign flip at the two
endpoints — but it has limited directional information and is sensitive to
the dominant-site bias discussed in the Gram matrix analysis.

A vector field formulation may provide:
- **Directional information**: the orientation of the broken bond
- **Localization**: the curl is localized at the bond by Stokes' theorem
- **Cleaner G**: a Gram matrix built from curl signatures instead of raw L

---

## From n(v,t) to a field

**Step 1 — Normalize to a spatial PDF at each cycle**:

    p(v,t) = n(v,t) / sum_{v'} n(v',t)

This makes p(v,t) a proper probability distribution over sites at each t,
sum_v p(v,t) = 1. The evolution p(·,1) → p(·,2) → ... is a trajectory
in the space of distributions on the graph.

**Step 2 — Bond current (probability flux)**:

For each bond (u,v) in the graph:

    J_{uv}(t) = p(v,t) - p(u,t)

Directed flow: positive = flow from u to v, negative = flow from v to u.
This is the discrete analog of the probability current in quantum mechanics.

**Step 3 — Divergence per site**:

    div(v,t) = p(v,t+1) - p(v,t) = sum_{u: neighbors of v} J_{uv}(t)

By discrete continuity: divergence = net inflow. A broken bond reduces
inflow to the shadow site (positive divergence, RED) and increases
accumulation at the wall face (negative divergence, BLUE).

**Step 4 — Plaquette circulation (curl)**:

For each elementary 2×2 plaquette (four sites forming a square):

    curl(plaquette, t) = J_{01}(t) + J_{12}(t) + J_{23}(t) + J_{30}(t)

where 0,1,2,3 are the four corners traversed counterclockwise.
On a healthy bond, curl ≈ 0 — the flow is irrotational.
A broken bond is a **vortex source**: adjacent plaquettes show
non-zero curl because information routes around the blockage.

---

## The broken bond signature

A broken bond (u,v) creates:
1. Reduced current J_{uv}^{pert} < J_{uv}^{base} at the bond
2. Vortex in the delta current field: delta_J = J^{base} - J^{pert}
3. Non-zero delta_curl at plaquettes adjacent to (u,v)
4. By Stokes' theorem: any closed path ENCLOSING (u,v) has
   non-zero circulation; any path NOT enclosing it has zero circulation

This directly localizes the broken bond — not just the nearby sites.

**Delta rotation per plaquette**:

    delta_curl(plaquette, t) = curl^{base}(plaquette,t) - curl^{pert}(plaquette,t)

For a single broken bond: exactly the two plaquettes sharing that bond
show delta_curl ≠ 0. For two broken bonds: two pairs of plaquettes.

---

## Why this may be better than L(v,t)

| Property | L(v,t) scalar | delta_curl vector |
|----------|---------------|-----------------|
| Localization | at bond endpoints | at bond itself (Stokes) |
| Direction | sign flip only | vortex orientation |
| Dominant site bias | severe | reduced (curl is differential) |
| Interference (2 bonds) | BLUE-BLUE ambiguity | two separate vortices |
| Gram matrix | dominated by high-weight sites | differential operator suppresses uniform background |

The curl is a **differential operator** — it suppresses common background
(uniform flow) and amplifies local vortices. This is the same mechanism as
whitening but encoded in the physics of the field rather than a statistical
normalization.

---

## Connection to existing tools

- L(v,t) ≈ local contribution to KL divergence between p_base and p_pert
- The divergence field is what BFS frontier detection already uses implicitly
- The curl is new — it encodes the rotational signature of the broken bond
- The simulation computes this gradient internally; we reconstruct it from
  the PAULI_NL output

---

## Temporal structure

The evolution p(·,t) is a 2D PDF trajectory. Two natural comparisons:

1. **Per-cycle KL**: D_KL(p_base(·,t) || p_pert(·,t)) — scalar per cycle,
   peaks when wavefront crosses the broken bond. No geometry.

2. **Wasserstein distance**: measures how much mass must be transported
   across the graph (cost = BFS hop distance) to match the two distributions.
   Encodes both magnitude and location of perturbation.

3. **Temporal correlation**: C(v,τ) = sum_t n(v,t)*n(v,t+τ) — lag structure
   reveals delay caused by broken bond.

---

## Next steps

1. Compute normalized fields p(v,t) from existing logs
2. Compute bond currents J_{uv}(t) for baseline and perturbed
3. Compute delta_curl per plaquette
4. Visualize as vector field on the 7×7 grid
5. Check: do the two plaquettes adjacent to the broken bond
   show unique delta_curl ≠ 0?
6. Build a Gram matrix from delta_curl signatures instead of L(v,t)

Script: scripts/compute_vector_field.py (to be written)
