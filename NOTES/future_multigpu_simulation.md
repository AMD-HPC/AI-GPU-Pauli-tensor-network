# Future Direction: Multi-GPU Simulation (2026-11-09)

## Vision

Split a large lattice across multiple GPUs — half the graph on GPU 0,
half on GPU 1 — with one GPU dedicated to BP message passing and
inner products.

## Architecture concept

**GPU 0**: sites 0..N/2 — Trotter gates, local BP messages
**GPU 1**: sites N/2..N — Trotter gates, local BP messages
**GPU 2**: cross-boundary BP messages + inner products (DG gauge)

The boundary between GPU 0 and GPU 1 is a cut of the lattice —
bonds crossing the cut require P2P message passing between GPUs.

## Key challenges

1. **BP message passing across GPUs**: messages at the cut bonds need
   to be exchanged via NVLink or PCIe. The cut should minimize
   cross-boundary bonds (minimum bisection width of the graph).

2. **Cut selection**: for a 7×7 grid, a horizontal cut at row 3
   gives 7 boundary bonds — manageable. For 3D, the minimum bisection
   cuts a 4×4 face = 16 bonds.

3. **Synchronization**: BP sweeps need barrier synchronization at each
   iteration. Cross-GPU messages add latency.

4. **DG inner products**: already on a separate GPU in the current 2-GPU
   pipeline. With 3 GPUs, inner products get more resources.

## Potential lattice sizes

With 3×32GB GPUs and graph splitting:
- **14×7** (or similar non-square): 2 GPUs for Trotter, 1 for IP
- **8×8×4** (3D): 2 GPUs for Trotter, 1 for IP
- **10×10** (100 sites): ambitious but possible

## Connection to existing code

The 2-GPU MPF pipeline already does:
- GPU 0: Trotter (4 states)
- GPU 1: inner products (9 pairs, parallel streams)

Multi-GPU Trotter would add graph partitioning and P2P BP messaging.
The DG gauge inner product is already GPU 1 — it would stay there
or move to GPU 2 with 3 GPUs.

## Priority

Lower than 3D tomography. The 3D tomography experiment first establishes
whether the physics is interesting at that scale. If 4×4×4 works well,
the motivation for larger lattices (requiring multi-GPU) is established.
