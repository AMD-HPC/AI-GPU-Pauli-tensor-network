# Distributed BP Architecture for Large Lattices (2026-11-11)

## The vision

A 120×120 2D lattice with 10 Trotter cycles is currently impossible:
- Full tensor network: 14,400 sites × χ^4 × D at χ=50 → ~90GB per state
- 4 states (NL + 3 noisy) → ~360GB → needs multi-GPU with large PCIe transfers

The key insight: **BP messages are the compressed state representation**.
Instead of transferring full site tensors between GPUs, transfer only
boundary BP messages — χ² per bond, tiny by comparison.

---

## The BP space

**Norm BP messages** (diagonal, what we use): each message is a χ-dimensional
positive vector, trace-normalized → lives in probability simplex Δ^{χ-1}.
Dimension per message: χ-1 ≈ χ.

**Full BP messages**: each message is a χ×χ positive semidefinite matrix,
trace-normalized → lives in space of density matrices.
Dimension per message: χ²-1 ≈ χ².

**Total BP state space**: 2×n_bonds × χ² — quadratic in χ, not exponential.
Compare to full tensor network: O(χ^degree × N) — exponential in χ.

**At fixed point**: messages further constrained by self-consistency equations
M_{u→v} = f({M_{w→u}}_{w≠v}) → they live on a self-consistent submanifold,
even lower dimensional than the product of simplices.

**The BP messages ARE the compressed state**. Saving them at cycle i is a
checkpoint. Restarting from them is exact (up to BP approximation).

---

## Checkpoint and restart

For a 120×120 lattice, 10 cycles:
- **Full TN state**: 14,400 × χ^4 × D × 4 states ≈ huge, impractical to save
- **BP message state**: 2 × 34,320 bonds × χ² ≈ 2 × 34,320 × 2,500 × 4B ≈ 686MB at χ=50

**686MB checkpoint** that can restart any simulation from any cycle.
Previously: re-simulate from scratch every time (hours).
With checkpoint: restart from cycle 5, continue to cycle 10 (minutes).

Checkpoint format:
- One χ×χ matrix per directed bond (2 per bond)
- One χ^degree tensor per site (the site tensors)
- Metadata: cycle number, chi, graph, centers

The site tensors are larger but bounded by the lightcone — only active
sites need to be checkpointed.

---

## 3-GPU distributed architecture

**Graph partitioning**: split the 120×120 lattice horizontally at row 60.
- GPU 0: rows 0-59 (7,200 sites, ~17,280 bonds)
- GPU 1: rows 60-119 (7,200 sites, ~17,280 bonds)
- Boundary: 120 bonds connecting row 59 to row 60

**Computation assignment**:
- GPU 0: Trotter gates for left half, local BP messages (all bonds within left half)
- GPU 1: Trotter gates for right half, local BP messages (all bonds within right half)
- GPU 2: global BP convergence — receives boundary messages from GPU 0 and GPU 1,
          runs global BP sweep, sends back updated messages

**Communication cost per BP iteration**:
- 120 boundary bonds × 2 directions × χ² × 4 bytes = 120 × 2 × 2,500 × 4 = **2.4MB**
- At χ=50, NVLink bandwidth ~300GB/s → transfer time ≈ **8 microseconds**
- Completely dominated by compute, not communication

**GPU 2 role**: global BP takes the boundary messages and updates them
consistently with both halves. It only needs:
- The boundary bond messages (2.4MB at χ=50)
- The site tensors at the boundary (120 sites × χ^4 × D ≈ 48MB at χ=50)

**GPU 2 fits a small GPU**: only ~50MB of active state for 120×120 at χ=50.
A 4GB GPU would be sufficient for the global BP coordinator role.

---

## Why this works: locality of BP

The BP self-consistency equation for bond (u,v):
    M_{u→v} = f({M_{w→u}}_{w≠v})

depends only on the **local neighborhood** of u — the other bonds incident
on u. For the global BP coordinator (GPU 2):
- Only boundary bonds send messages to both halves
- Interior messages (within each half) don't need to cross GPUs
- GPU 2 only needs to reconcile the boundary — not the full lattice

**Convergence is global but communication is local**: the global fixed point
is found iteratively by exchanging boundary messages between GPUs. Each
iteration requires only 2.4MB of communication — the number of boundary
bonds times χ².

---

## Scaling to larger lattices

| Lattice | Boundary bonds | Comm per iter (χ=50) | GPU 0/1 mem | GPU 2 mem |
|---------|----------------|----------------------|-------------|-----------|
| 20×20   | 20             | 0.4MB                | ~0.5GB      | ~8MB      |
| 50×50   | 50             | 1.0MB                | ~3GB        | ~20MB     |
| 120×120 | 120            | 2.4MB                | ~18GB       | ~50MB     |
| 200×200 | 200            | 4.0MB                | ~50GB*      | ~83MB     |

*200×200 would need further partitioning (4+ GPUs).

**Key**: communication scales as O(k) with the boundary size k, while
compute scales as O(k²) with the lattice size. Communication is never
the bottleneck — always dominated by compute.

---

## Connection to onion peel

The distributed BP architecture and onion peel are **complementary**:

- **Onion peel**: reduces active sites from volume to surface → smaller tensors per GPU
- **Distributed BP**: splits the lattice across GPUs → smaller lattice per GPU
- **Rank-k BP**: reduces message size from χ² to kχ → smaller communication

Combined:
- Each GPU handles a spatial partition
- Only boundary messages (kχ or k×χ²) are exchanged
- Global BP coordinator is tiny (boundary sites + messages only)
- Onion peel further reduces each partition's memory

This three-way combination enables **arbitrarily large lattices** at fixed
per-GPU memory — the method scales to whatever lattice size the physics demands.

---

## Geographical distribution

The BP messages are geographically distributed — each GPU holds the messages
for its spatial partition. The global coordinator (GPU 2) holds only the
boundary. This maps naturally to:

- **Same node**: NVLink for fast boundary message exchange (~300GB/s)
- **Multi-node**: InfiniBand for larger lattices (slower but still χ²-bounded)
- **Cloud**: even possible at small χ where message size fits in network bandwidth

The distributed structure follows the physics: information propagates
locally (the wavefront), correlations are local (BP messages), and
detection is local (frontier crossing at each site). There is no
global state that needs to be on one GPU.

---

## Implementation path

1. **Checkpoint/restart first** (highest value, lowest complexity):
   - Save BP messages + site tensors to disk at any cycle
   - Restart from checkpoint — no graph partitioning needed
   - Immediate benefit: 10-cycle simulations restartable from any point

2. **2-GPU Trotter split** (next):
   - Extend existing 2-GPU pipeline (GPU 0: Trotter, GPU 1: IP)
   - Add boundary message exchange at each BP sweep
   - Test on 20×20 or 30×30 lattice

3. **3-GPU with global BP coordinator** (full architecture):
   - GPU 2 as global BP, receives boundary from GPU 0 and GPU 1
   - Requires implementing distributed BP convergence
   - Target: 120×120 at χ=50 on 3×32GB GPUs
