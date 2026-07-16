# Inner Product for MPF — Status & Plan (2026-07-16)

## THE GOAL (non-negotiable)
MPF requires solving a linear system with:
- **A[i,j] = ⟨ψ_noisy(γᵢ) | ψ_noisy(γⱼ)⟩**  — correlation matrix (N×N)
- **B[i]   = ⟨ψ_noisy(γᵢ) | ψ_noiseless⟩**    — right-hand side (N)

Both are GENUINE two-state inner products between DIFFERENT evolved
operators. There is NO shortcut:
- Autocorrelation ⟨ψ|Z⊗I⟩ does NOT suffice (only a gauge-robust special case)
- overlap_gpu.hpp FAKED it: computes noisy autocorrelation, not true overlap
  (see line 130: autocorrelation_with_bp on noisy state, labeled "overlap")

## WHAT WORKS
- autocorrelation_with_bp = ⟨ψ(t)|Z⊗I...⊗I⟩ = -0.803318 ✓ matches Julia
  Works because I-component contraction = trace = gauge-invariant.
- Gate PTMs match Julia (verified).
- Bethe formula matches Julia's freenergy:
  C = Π_v vertex_scalar_v / Π_e edge_scalar_e
  (verified this is IDENTICAL to Julia src/MessagePassing/*.jl)

## WHAT DOES NOT WORK
- General ⟨φ_a|φ_b⟩ between two evolved operators. Every attempt gave
  ~100x wrong (−0.009 vs −0.962 for ⟨ψ|device_ρ⟩).

## ROOT CAUSE (established, do not re-derive)
Heavy-hex is a HONEYCOMB — has loops (hexagonal cells). NOT a tree at
any cycle. Therefore BP is ALWAYS approximate here, never exact.

BP on a loopy graph is GAUGE-DEPENDENT in its approximation:
- Julia tensors: symmetric/BP-canonical gauge (simple_update absorbs
  sqrt(BP environment) every gate — see src/Apply/simple_update.jl:61-90)
- CppSim tensors: QR gauge (plain QR/SVD, no environment weighting)

Same physical operator, different bond basis. Proof (diag_phi_center.py):
  phi_center marginal onto bond4:
    CppSim: [-0.211, 0.467, -0.0006, -0.0019]
    Julia:  [1.073, 0.030, 0.165, -0.268]
  Genuinely different vectors (not scale/reorder).

The EXACT contraction is gauge-invariant. BP is not exact on loops,
so the two gauges give different BP numbers.

## THE OPEN QUESTION (what to learn next)
Can we build a CORRECT two-state inner product ⟨φ_a|φ_b⟩ in CppSim,
in EITHER gauge (sqrt/symmetric OR QR)?

Key insight not yet tested: the true two-state overlap requires the
DOUBLE-LAYER network:
  - site tensor: φ_a_v ⊗ φ_b_v (physical indices contracted a-to-b)
  - bond dimension: χ_a_e × χ_b_e  (NOT χ_a alone)
  - BP messages: (χ_a × χ_b) matrices
  - This is Julia's BilinearForm(ψ_a, ψ_b) when BOTH have χ>1

We NEVER built this. All our attempts used:
  - single-layer messages (χ only), OR
  - autocorrelation messages (wrong bilinear form), OR
  - chi-vector messages (only valid when one side has χ=1, i.e. ρ)

For ⟨ψ|ρ⟩ ρ has χ=1 → bonds stay χ_ψ → chi-vector messages (Julia's case).
For ⟨ψ_a|ψ_b⟩ both χ>1 → bonds are χ_a×χ_b → matrix messages.

## HYPOTHESIS TO TEST
The double-layer BP with (χ_a × χ_b) messages is gauge-COVARIANT:
the gauge transforms on each layer cancel in the double-layer contraction,
so ⟨φ_a|φ_b⟩ gives the SAME answer regardless of each layer's gauge.

IF TRUE: QR gauge works fine for the two-state overlap. Our failures
were because we never built the true double-layer network — we kept
using single-layer / autocorrelation machinery.

## PLAN (Python first, then C++)
1. Julia: compute ONE concrete ⟨ψ_a|ψ_b⟩ for two evolved states
   (e.g. noiseless vs γ=1.0 at cycle 1) as numerical target.
   Add to test_inner_minimal.jl.
2. Dump both ψ_a and ψ_b tensors from Julia (or CppSim).
3. Python: build double-layer network φ_a_v ⊗ φ_b_v, bonds χ_a×χ_b,
   run BP with matrix messages, compute Bethe PF.
4. Verify it matches Julia's inner(ψ_a, ψ_b; alg="bp").
5. Test BOTH gauges (Julia sqrt tensors AND CppSim QR tensors) —
   confirm the hypothesis that double-layer is gauge-covariant.
6. If confirmed in QR gauge → implement double-layer BP in C++.
   If only works in sqrt gauge → implement environment-weighted
   simple_update in CppSim evolution.

## FILES
Python validation:
- test_double_layer_bp.py — attempt (used chi-vector, WRONG for two-state)
- test_chivec_bp.py — chi-vector BP (only valid for ρ with χ=1)
- test_direct_bethe.py — raw psi+autocorr msgs (wrong messages)
- diag_phi_center.py — proves bond-basis mismatch
- test_gauge_transform.py — sqrt(M) absorption (insufficient alone)
Julia:
- test_inner_minimal.jl — golden ⟨ψ,ρ⟩ step-by-step
- check_tensors.jl — Julia phi_center dump

## KEY NUMBERS (cycle 1)
- Julia inner(ψ, Z⊗I)     = -0.803318  ← CppSim autocorr MATCHES ✓
- Julia inner(ψ, all-0 ρ) = -0.961822  ← CppSim CANNOT match yet
- Julia inner(ψ, device ρ)= -0.678563  ← CppSim CANNOT match yet
- CppSim best attempt: -0.009 (100x off)

## BREAKTHROUGH (2026-07-16 late) — test_gauge_invariance.py

PROVED: the two-state overlap ⟨ψ_a|ψ_b⟩ built as a DOUBLE-LAYER network
is GAUGE-INVARIANT (on tree; exact contraction always).

Test: random 2-site tensors, applied random gauge G·G⁻¹ to ψ_a's bond,
double-layer contraction gave IDENTICAL result (-2.922876) before and
after gauge transform.

CONCLUSION:
- QR gauge WORKS for two-state overlap. No need to change CppSim evolution.
- Our failures: we NEVER built the true double-layer network.
  We used single-layer messages (χ) or autocorrelation messages.
- Correct: double-layer messages have dimension χ_a × χ_b.

For ⟨ψ|ρ⟩ (ρ χ=1): double-layer bond = χ×1 = χ (looked single-layer).
For ⟨ψ_a|ψ_b⟩ (both χ~4): double-layer bond = χ_a×χ_b = 16.

The honeycomb loops make BP approximate, but the approximation is now
gauge-COVARIANT because the double-layer cancels per-layer gauge. So
CppSim QR and Julia sqrt should give the SAME BP result for the overlap.

## PLAN (revised — build double-layer BP)
1. Python: double-layer BP on full heavy-hex with χ_a×χ_b messages
2. Validate ⟨ψ_a|ψ_b⟩ vs Julia inner(ψ_a,ψ_b;alg="bp")
3. Port to C++ (this is the general inner product for A and B matrices)

## HONEST CAVEAT (the real open question)
test_gauge_invariance.py proved gauge-invariance on a TREE (single bond).
BP is EXACT on trees, so gauge-invariance there is trivial/guaranteed.

NOT YET PROVEN: gauge-covariance of double-layer BP on a LOOP.
The honeycomb has loops where BP is approximate. The claim that
double-layer cancels per-layer gauge on loops is UNVERIFIED.

CRITICAL NEXT TEST (before full heavy-hex):
  Build ONE hexagon (6 sites in a loop), two random states, run
  double-layer BP in QR gauge AND symmetric gauge. If results match
  → gauge-covariant on loops → QR gauge works for heavy-hex.
  If they differ → gauge-covariance is tree-only → must match Julia's
  gauge (environment-weighted simple_update).

This single-loop test is the DECISION POINT.

## NEXT SESSION START HERE
Step 1: In test_inner_minimal.jl, add a TRUE two-state overlap:
  ψ_a = evolve 1 cycle noiseless
  ψ_b = evolve 1 cycle with γ=1.0 noise
  print inner(ψ_a, ψ_b; alg="bp")  ← this is a real A/B matrix element
Then build double-layer BP in Python to match it.
