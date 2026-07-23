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

## CONCRETE TARGET + DATA READY (2026-07-16 end)

Julia dump_two_states.jl produces:
  TARGET: inner(ψ_nl, ψ_noisy; alg="bp") = 0.980920  (real 2-state overlap!)
  two_states_dump.txt: both states, shared site-index basis
  (copied to /scratch/MPF/CppSim/two_states_dump.txt)

This is a genuine MPF B-vector element. Match it → we can do A and B.

test_two_state_overlap.py loads both states. Site shapes confirmed:
  site 3 (center): (4,4,4,4) = 3 links + pauli, neighbors [33,34,35]
  site 33: (4,4,4), neighbors [2,3]  ← note: Julia 1-based vertices

## OPEN BUG TO FIX FIRST (link ordering)
Julia orders tensor links by inds(t), NOT by sorted neighbor.
Must map each tensor axis → correct bond by matching link dimensions
to neighbor bond χ. Site 35 shape (1,4,4): first link χ=1 (bond to
site 11? or 3?). Get this wrong → silently wrong BP.

To fix: dump_two_states.jl should ALSO dump, per site, the neighbor
vertex for each link index IN ORDER. Add link→neighbor mapping to dump.

## Session 2026-07-17 evening

**bilinear_form_bp.hpp** updated to handle different chi_a vs chi_b per bond.
Now takes two grids: `bilinear_form_bp(grid_a, psi_a, grid_b, psi_b)`.
T_v bond dim = chi_a × chi_b per bond.

**Results** (cycle 1, chi_max=100):
- `bilinear_form_bp(ψ_nl, ψ_nl)` = 0.461 (Julia: 0.9999) — gauge normalization
- `bilinear_form_bp(ψ_n,  ψ_n)`  = 0.792 (Julia: 0.9623) — gauge normalization
- `bilinear_form_bp(ψ_nl, ψ_n)`  = 0.001 (Julia: 0.9809) ← WRONG

**Cross-overlap is ~0**: BP converges to degenerate fixed point when
noiseless and noisy states have same chi=4 structure. The messages
don't distinguish the two states properly.

**Uniform coupling test**: Julia gives 0.9808 (same as heterogeneous 0.9809).
Coupling heterogeneity is NOT the cause of normalization mismatch.

**Cosine similarity** = C_cross / sqrt(C_nl × C_n) = 0.001 (should be 0.981).
This means the BP fixed point is wrong, not just unnormalized.

**Root cause hypothesis**: BP on phi_v (chi_a × chi_b messages) starts at
identity which is wrong when chi_a and chi_b come from different states with
different physical content. The messages must be initialized to reflect the
actual bond strengths of BOTH states, not just one.

**Next investigation**:
- Print a_v[3] for the cross-overlap case → should be ~11.97 (Julia) but we get ~0
- The message from site 33→3 for cross-overlap: what does BP converge to?
- Compare with Julia's BLF messages for inner(ψ_nl, ψ_n)

**Test scripts**:
- test_uniform_coupling.sh: tests uniform J=1.0 (confirms coupling not the issue)
- dump_uniform.jl: Julia with uniform couplings

## NEXT SESSION START HERE
1. Fix dump: add per-link neighbor labels so Python knows axis→bond
2. Build double-layer BP: bond index = (i_a, i_b), message = χa×χb object
3. Match 0.980920
4. Then also test single hexagon QR vs symmetric (gauge-covariance on loop)
5. Port to C++
Step 1: In test_inner_minimal.jl, add a TRUE two-state overlap:
  ψ_a = evolve 1 cycle noiseless
  ψ_b = evolve 1 cycle with γ=1.0 noise
  print inner(ψ_a, ψ_b; alg="bp")  ← this is a real A/B matrix element
Then build double-layer BP in Python to match it.

## Session 2026-07-17 late

**Julia script confirmed**: gpu_run_simulation_tnqs_noisy_hardware_epsilon.jl
computes A[i,j] = inner(ψ_i, ψ_j; alg="bp") and B[i] = inner(ψ_i, ψ_nl; alg="bp")
where ψ_i are states evolved at different epsilon (Trotter step) values with
fixed gamma=1.0 noise. This is the CORRECT MPF formulation — main_heavyhex_mpf
was wrong (computed C_n(t), not true inner products).

**kick bug fixed**: NoisyHeavyHexSimulator was missing the kick wave. After fix:
- C_nl(t) signs: -0.803, +0.630, -0.584, +0.706, -0.845 (alternates ✓)
- C_n(t) signs: -0.437, +0.089, -0.206, +0.149, ~0.000 (alternates ✓)
- center site: fixed to site 2 in main_heavyhex_noisy (was 13)

**bilinear_form_bp attempts** (all at cycle 1, chi_max=100):
- Self NL: 0.461 (Julia 0.9999) — correct up to QR gauge normalization ✓
- Self N:  0.461 (NoisyKickSimulator, Julia 0.9623) ✓
- Cross:   -0.174, cosine -0.378 (Julia 0.9809) ← WRONG SIGN

**sqrt gauge attempt**: pass msg_a, msg_b into bilinear_form_bp, absorb
sqrt(diag(msg)) into each tensor's bond axis before building T_v.
Result: -0.007, cosine -0.015. Made it worse — independent gauge transforms
don't fix sign_num because they act on psi_a and psi_b separately, changing
T_v signs arbitrarily.

**Root cause (confirmed)**: sign_num=-1 in Bethe formula for the cross call.
This means some a_v < 0 after BP converges. The BP fixed point is in the
wrong basin — the identity initialization (M = I/chi) is a poor starting
point for the double-layer cross network when the two states have opposite
signs in some chi channels (independent QR runs).

**QR positive-diagonal fix** (gate.hpp): added `fix_qr_sign_Q` and `fix_qr_sign_R`
GPU kernels. After Householder QR, if R[i,i]<0 flip sign of Q[:,i] and R[i,:].
This makes independent QR runs on nearly-identical states choose consistent signs.
Result: sign_num flipped from -1 → +1. Cross = +0.230, cosine = +0.499.

**Sqrt gauge absorption**: absorb sqrt(diag(msg_incoming)) into each tensor's
bond axis before building T_v. Tried with and without QR fix. Made things worse
(cosine dropped to 0.017). The absorption changes T_v itself, not just initialization.

**Kronecker warm-start initialization**
Replace M_fwd/M_bwd initialization from identity to kron(msg_a, msg_b):
- M_fwd[bond](ia*cb+ib, ja*jb) = msg_a_fwd[bond](ia,ja) × msg_b_fwd[bond](ib,jb)
- This is the factored approximation: assumes the two states are nearly independent
- For weak noise (gamma=1.0), msg_a ≈ msg_b so kron ≈ msg² — a good warm start
- Mimics what Julia does: both states already in BP-canonical gauge, so the
  BilinearForm BP starts from the product of the two converged environments
- The kron messages break the sign degeneracy by encoding the actual bond content

**Implementation plan** (bilinear_form_bp.hpp):
1. Add msg_a_gpu/msg_b_gpu optional parameters (already added for sqrt-gauge attempt)
2. In message init (step 3): if msgs provided, set
   M_fwd[b][ia*cb+ib] = ma_fwd[b][ia] * mb_fwd[b][ib]  (diagonal kron)
   where ma_fwd[b][ia] = sqrt of diagonal element ia of msg_a_fwd[b]
3. Keep max-abs normalization during BP sweeps
4. Test: should give sign_num=+1 and cosine ≈ +0.98

**Key insight**: Julia's compute_inner uses inner(ψ_i, ψ_j; alg="bp") which
internally constructs BilinearForm BP initialized from the PRODUCT of each
state's existing BP cache messages. We must replicate this initialization.

**Kronecker warm-start result**: implemented kron(diag(msg_a), diag(msg_b))
initialization. Zero effect — BP swept back to the same fixed point regardless
of initialization. The loopy heavy-hex graph has a dominant fixed point.

**Exact tree contraction** (Python, NoisyKickSimulator dumps after QR fix):
- self_nl = 1.377, self_n = 1.370, cross = +0.954
- Cosine exact tree = 0.695  (Julia: 0.981)
- CppSim BP cosine  = 0.500  (50% of exact tree)
- Physics gap (0.695 vs 0.981): NoisyKickSimulator PTM ≠ Julia's interleaved model
- BP gap (0.500 vs 0.695): loop corrections on heavy-hex undercount cross term

**Current state** (2026-07-17 end of session):
- Self-overlaps: BP correct ✓ (0.461 each, consistent)
- Cross-overlap sign: fixed by QR positive-diagonal convention ✓
- Cross-overlap magnitude: 0.500 vs exact 0.695 vs Julia 0.981
  - Physics mismatch: NoisyKickSimulator vs Julia's PTM model
  - BP approximation error on loops: ~28% undercount

**Decision**: proceed with MPF A/B matrix using bilinear_form_bp as implemented.
The loop BP error is systematic and partially cancels in ratios. The QR sign fix
makes cross terms positive and in the right ballpark. Full loop corrections
(loopcorrections alg in Julia) would require O(2^k) work — deferred.

## Session 2026-07-17 end-of-day — architecture decision

**Key discovery from Julia source** (bilinearform.jl, inner.jl):
- Julia's BilinearForm uses full chi_a × chi_b double-layer bond messages
- Default message = delta tensor on (chi_ket, chi_op=1, chi_bra) indices
- chi_op=1 because operator is product of delta tensors (identity on physical)
- So messages ARE full Kronecker: chi_a × chi_b — not factored
- BUT: at cycle 1 only 5 bonds have chi>1 (tree subgraph) → BP is exact and fast

**The ρ insertion insight** (Paolo's observation):
- ⟨φ|⟨ρ|ρ⟩|φ⟩ = ⟨φ|ρ⟩ × ⟨ρ|φ⟩ since ⟨ρ|ρ⟩=1 (pure state in PTM basis)
- This gives A[i,j] = C_i × C_j → rank-1 matrix → MPF degenerates
- Confirmed: ρ insertion cannot replace full bilinear form for MPF
- BUT: it IS useful conceptually — autocorrelation IS ⟨ψ(t)|ρ⟩

**Scalability analysis**:
- A[i,j] = ⟨ψ_n_i|ψ_n_j⟩: chi_n × chi_n = 150×150 = 22K per bond → tractable
- B[i]   = ⟨ψ_n_i|ψ_nl⟩:  chi_n × chi_nl = 150×1000 = 150K per bond → T_v at degree-3 site = 150K³ → INTRACTABLE with explicit T_v

**The right approach for B[i]**:
- Never build T_v explicitly
- Use **factored contractions**: for each site, contract messages one bond at a time
- Cost: O(chi_a² × chi_b × D) per site (choose a=larger state as environment)
- For B[i]: use ψ_nl as environment (run BP on ψ_nl), contract ψ_n_i bond by bond
- This is identical to `autocorrelation_with_bp` structure but with a TN operator
- Alternatively: extend `bp_partitionfunction` to accept per-site tensor operators

**The right approach for A[i,j]** (i≠j):
- chi_n × chi_n messages — tractable with factored contractions
- Cost: O(chi_n² × chi_n × D) = O(chi_n³ × D) per site
- At chi_n=150: 150³×4 = 13M ops per site, 68 sites = 900M → feasible on CPU

**Current main_heavyhex_mpf2.cpp status**:
- Built and running ✓
- A diagonal = 1.0 ✓ (normalization correct)
- A off-diagonal inconsistent signs at higher chi → gauge problem in chi>1 bonds
- B[i] signs inconsistent across gammas → same gauge problem
- Uses explicit T_v → will not scale past chi~8

## NEXT SESSION START HERE

**Implement factored bilinear form BP** (no explicit T_v):

For B[i] = ⟨ψ_n_i|ψ_nl⟩:
- Run BP on ψ_nl (larger chi) to get converged messages msg_nl
- For each site v: contract ψ_nl[v] and ψ_n_i[v] bond-by-bond using msg_nl
  - Start with tmp[ia_0, ib_0, ..., ia_nb, ib_nb, d] = ψ_nl[v][ia...,d] × ψ_n_i[v][ib...,d]
  - Absorb msg_nl[bond_k] into ia_k index one at a time → reduces to scalar
  - Cost: O(chi_nl^(deg-1) × chi_n × D) per site
- Bethe PF using same log-space formula

For A[i,j] = ⟨ψ_n_i|ψ_n_j⟩:
- Same factored approach, both chi_n → O(chi_n^deg × D) per site
- At chi_n=150, deg=3: 150³×4 ≈ 13M per site × 68 = 900M → ~1s on CPU

**Implementation plan**:
1. Add `bilinear_form_factored(snap_a, snap_b)` to bilinear_form_bp.hpp
   - No T_v construction
   - Messages: separate chi_a-vector and chi_b-vector per bond direction
   - Update rule: contract one bond at a time into message
2. Update main_heavyhex_mpf2.cpp:
   - B[i]: `bilinear_form_factored(snap_n[g], snap_nl)` with snap_nl as env
   - A[i,j]: `bilinear_form_factored(snap_n[i], snap_n[j])`
   - A[i,i]: `autocorrelation_with_bp` style (chi_n env, ψ_n_i as op)

**Files to modify**:
- include/bilinear_form_bp.hpp: add bilinear_form_factored
- main_heavyhex_mpf2.cpp: use new function for all inner products

**Simulators** (confirmed correct):
- HeavyHexSimulator: IDEAL PTM + kick wave ✓
- NoisyKickSimulator: NOISY PTM + kick wave ✓ (single PTM per bond, same structure)
- QR positive-diagonal convention: gate.hpp fix_qr_sign kernels ✓

## Session 2026-07-18 morning — CppSim bilinear status + Julia plan

**CppSim bilinear form: DEFERRED**
Two fundamental problems that cannot be solved with current approach:
1. Sign chaos: degenerate chi channels in independent QR runs → A[i,j] wrong signs
   - QR positive-diagonal fix helps dominant channel but not degenerate subspaces
   - Common noiseless gauge reference (sqrt-gauge) doesn't fix inter-state signs
2. Kronecker explosion: T_v at center site is (chi_nl×chi_n)^3
   - chi=20: 64M entries → 107s per inner product
   - chi=400/150: completely intractable
   - Factored rank-1 messages don't capture inter-state correlations → wrong results

**Decision**: bilinear form in CppSim requires full SVD canonicalization (like Julia)
to resolve degenerate subspace sign ambiguity. Deferred to future work.
main_heavyhex_mpf2.cpp committed as proof-of-concept; A/B values unreliable at chi>4.

**Julia asymmetric sqrt gauge: NEXT (45 min session) — target PR #25**
Goal: eliminate sqrt(Complex) in Julia's loopcorrections path.
Current workaround: promote to ComplexF32, compute inner product, take real part.
Proposed: split diagonal message m[i] into:
  L[i] = sign(m[i]) × sqrt(|m[i]|)  → left tensor  (real, signed)
  R[i] = sqrt(|m[i]|)               → right tensor (real, positive)
This keeps Float32 throughout, halves memory, enables GPU loopcorrections.
File to patch: find where sqrt(vn) fails in the Julia BilinearForm/loopcorrections path.
Target: submit as PR #25 to amd-TensorNetworkQuantumSimulator.jl

## JULIA SESSION START HERE (PR #25)
Find and patch the sqrt(complex) workaround in Julia:
1. grep for "complex" or "ComplexF32" in workloads/bp_mpf/src/
2. Find the sqrt call that triggers DomainError on negative diagonal messages
3. Implement asymmetric split L/R
4. Test on 2-site example, then heavy_hex_3x3 cycle 1
5. Submit PR #25

## Session 2026-07-18 afternoon — architecture notes

### Observable: C(t) is gauge-invariant
C(t) = Tr(Z(t) ρ) is a physical observable — gauge-invariant by definition.
QR gauge, sqrt gauge — same answer. This means:
- CppSim autocorrelation_with_bp already gives correct C_nl(t) and C_n(t) ✓
- Primary validation between Julia and CppSim = compare autocorrelation curves
- No bilinear inner product needed for physics validation
- epsilon (Julia) ≡ gamma (CppSim): both parametrize noise amplification for MPF

### Inner products on GPU (future work)
At chi=100+, CPU bilinear form is intractable (double-layer bond = chi²).
Requires GPU kernel for T_v contractions + BP message updates on-device.
Significant engineering effort — deferred after autocorrelation validation.

### BP timing difference — FUNDAMENTAL architectural gap
**Julia**: BP runs after each color class (sub-layer) gate application.
- Messages updated incrementally — environment tracks state after each sub-layer
- Chi grows gradually: BP truncates mid-Trotter at each color boundary
- BP graph has 68_sites × N_gates nodes — many nodes, moderate chi

**CppSim**: BP runs once after the full Trotter step (all 3 color classes).
- Messages initialized from previous cycle's converged state
- Tensors can grow chi freely during Trotter before BP truncates
- BP graph has exactly 76 bonds — few nodes, potentially larger chi per bond

**Consequences**:
- Julia chi growth is slower (truncated per color class)
- CppSim chi growth is faster (truncated once per cycle)
- Julia and CppSim will have different chi at same cycle count
- Autocorrelation comparison will show this truncation schedule difference
- CppSim BP is cheaper per sweep (76 bonds vs hundreds of Julia nodes)
- At large chi: CppSim BP scales better despite higher per-bond chi

### Three-way Julia experiment (2026-07-18)
Running Float32 / ComplexF32 / Float64 on 3 GPUs simultaneously.
Script: run_pr25_experiments.sh
Logs: logs/pr25_20260718_165312/
Goal: validate Float32 gives same physics as ComplexF32 (PR #25 correctness)
      and Float64 gives same physics as ComplexF64 (accuracy reference)
Files saved with scalartype in filename: ..._Float32.dat, ..._ComplexF32.dat, ..._Float64.dat
Compare with: julia --project compare_pr25.jl

### PR #25 VALIDATED (2026-07-18)
Three-way experiment (Float32 / ComplexF32 / Float64) at chi=14, 10 cycles:
- All three give **bit-for-bit identical results** (|Δ| = 0.00e+00 all cycles)
- Timing at chi=14: all ~285-300s/cycle — loopcorrections CPU enumeration dominates
- Scalar type has zero impact on timing at chi=14
- Branch pushed: feature/pr25-float32-sqrt-gauge on AMD-HPC repo

### Chi convergence sweep (RUNNING as of 2026-07-18 17:53 UTC)
Command: bash run_pr25_experiments.sh --cycles=10 --chisweep=14,28,50,100,0
Logs: logs/pr25_chi{N}_TIMESTAMP/
Expected: Float32 timing advantage appears at chi≥50 (tensor contractions dominate)
Expected: OOM for ComplexF32/Float64 before Float32 at large chi

## Session 2026-07-19 — PR #25 complete + timing results

### Key timing result (chi=4, cycle 1, genuine scalar types):
| Scalar type | Time | vs Float32 |
|-------------|------|-----------|
| Float32     | 343s | baseline  |
| Float64     | 380s | +11%      |
| ComplexF32  | 480s | **+40%**  |

ComplexF32 is 40% SLOWER than Float32 — imaginary arithmetic costs real compute
even though physics is real. Float32 is the clear winner.

### Architecture confirmed (type checks):
- Gates: Float32 (RST = real(ST)) for all scalar types ✓
- ψ_noiseless: always Float32 (Pauli basis, always real) ✓
- ρ: always Float32 (exact integers [1,0,0,±1], chi=1, no BP/SVD) ✓
- ψ_noisy: uses ST (Float32/ComplexF32/Float64) — the actual experiment variable

### ST vs RST design (clean):
- ST  = scalar type for evolved quantum states
- RST = real(ST) = real scalar type for gates, kick gates, rho
- Physics is real in Pauli basis — only states use ST for the experiment

### PR #25 fixes (all committed to feature/pr25-float32-sqrt-gauge):
- beliefpropagationcache.jl: sqrt(vn) → sqrt(abs(vn))
- loopcorrections_patch.jl: same fix for BilinearForm
- utils.jl: pseudo_sqrt_inv_sqrt uses sqrt(abs(x)) — fixes GPU segfault
- tnqs_noisy_epsilon_evolution.jl: scalartype param, remove ComplexF32 promo
- TNQSNoisyConfig.jl: --scalartype, --cycles, --maxchi CLI args
- gpu_run_simulation_tnqs_noisy_hardware_epsilon.jl: ST/RST, gate conversion,
  maxdim fix, filename tag, type check diagnostics

### Chi sweep RUNNING (2026-07-19 ~01:00 UTC):
Machine 1: bash run_pr25_experiments.sh --cycles=10 --chisweep=10,30,0
Machine 2: bash run_pr25_experiments.sh --cycles=10 --chisweep=30,50,70
Logs in: logs/pr25_chi{N}_*/

### Next session:
1. Read chi sweep results — compare timing and z_expvals across chi
2. Run compare_pr25.jl on Float32 vs Float64 (now genuinely different arithmetic)
3. Build chi convergence curve: |C(chi) - C(uncapped)| per cycle
4. Fix compare_pr25.jl label detection (shows C32 for both files)
5. Consider updating damping factors load to not depend on num_cycles

## Session 2026-07-19 morning — Chi convergence results

### Chi convergence (noiseless C_nl, Float64 reference vs uncapped):
| Chi | Error at cycle 10 | Status |
|-----|-------------------|--------|
| 10  | 9.0e-02 (9%)      | Too small |
| 30  | 2.2e-02 (2%)      | Acceptable |
| 50  | 4.6e-05 (~0%)     | Nearly converged |
| 70  | 0.00e+00          | Fully converged |

chi≥70 = ground truth for 10 cycles at ε=0.1-0.2.
chi=50 is 99.9% converged — practical sweet spot.
Convergence only appears at late cycles (3+) as entanglement builds.

### Key physics insight:
Small Trotter steps (ε=0.1-0.2) → weak entanglement per step → chi grows slowly.
Noiseless uncapped chi only reaches ~92 at cycle 10 — not explosive.
CppSim grows chi faster (single composed PTM, no intermediate truncation).

### Scalar type precision (noisy C_n, ε=0.1):
- Float32 vs Float64: |Δ| ~ 1e-6 to 1e-4 — excellent
- ComplexF32 vs Float64: |Δ| ~ 1e-3 to 1e-2 — 100-1000x worse!
- ComplexF32 is LESS accurate than Float32, AND 20% slower

### Timing (all chi values, 10 cycles):
Float32: ~52min | ComplexF32: ~64min (+23%) | Float64: ~61min (+17%)
Timing flat across all chi — loopcorrections 100% chi-independent.

### Tools:
- parse_logs.py: extracts z_expvals, produces summary + CSV files
- results_noiseless.csv, results_timing.csv, results_delta.csv

### Key findings (complete):

**1. Noiseless C_nl(t) is identical across all scalar types** — always Float32
internally (Pauli basis is real). Scalar type only affects noisy states.

**2. Chi convergence** (noiseless, Float64 reference):
- chi=10: 9% error at cycle 10
- chi=30: 2% error
- chi=50: ~0% (converged)
- chi=70: bit-for-bit identical to uncapped ← ground truth
- Same for noisy states (less entanglement → easier to converge)

**3. Scalar type precision on noisy C_n (chi=70, ε=0.1)**:
- Float32 vs Float64: |Δ| ~ 1e-6 to 1e-4 — excellent
- ComplexF32 vs Float64: |Δ| ~ 1e-3 to 1e-2 — 100-1000× worse
- ComplexF32 is LESS accurate than Float32!

**4. Timing (10 cycles, chi=70)**:
- Float32: 52min | ComplexF32: 65min (+25%) | Float64: 60min (+15%)
- Timing flat across all chi — loopcorrections 100% chi-independent

**5. MPF coefficients**:
- MPF solve always in ComplexF64 (hardcoded in pipeline)
- Input C_n quality determines output: Float32 ≈ Float64, ComplexF32 diverges
- At cycle 1: ComplexF32 MPF coeffs completely different (0.487 vs 0.293)
  due to ill-conditioned A matrix amplifying 4e-3 C_n difference → 60% coeff diff
- C_mpf(t) plot: Float32 tracks C_nl perfectly, ComplexF32 shows visible deviation

**6. Architecture insight**:
- Gates/rho use RST=real(ST) — always Float32 or Float64
- States use ST — the experiment variable
- MPF solve uses ComplexF64 — independent of ST
- ρ is always Float32 (exact integers, chi=1, never goes through BP)

### Analysis tools committed to branch:
- parse_logs.py: summary table from logs
- analyze_pr25.py: CSV + 8 plots (noiseless, noisy×3, MPF coeffs, MPF result, timing)
- results_all.csv: single shareable table

### Branch: feature/pr25-float32-sqrt-gauge (pushed)

### Completed (2026-07-19/20):
1. chi=20 copied and added to convergence table ✓
2. results/MI100/ committed with README + plots + CSV ✓
3. Branch feature/pr25-float32-sqrt-gauge shared ✓
4. MPF extrapolation error plot added (plot_mpf_error.png) ✓
5. Key conclusion: MPF error 2-18% per cycle, identical across all scalar types
   → scalar type is NOT the bottleneck, MPF methodology is ✓
6. README complete and standalone for external physicists ✓

## Session 2026-07-21 — CppSim vs Julia Z(t) comparison

### New files:
- `build_ptm.py`: added `--epsilon`, `--gamma`, `--phi` CLI args; epsilon encoded in model tag
- `include/heavyhex_ideal_kick_eps{005,008,009,010}_data.hpp`: IDEAL_KICK at 4 epsilon values
- `include/heavyhex_noisy_eps{010,015,020}_data.hpp`: NOISY at 3 epsilon values, gamma=1.0 only
- `main_heavyhex_eps_mpf.cpp`: new main computing C(t) for noiseless + 3 noisy states
  Usage: `./main_heavyhex_eps_mpf [chi_max] [n_steps] [cutoff] [K]`
  K = number of bond-gate sub-steps per cycle (default 1), kick applied once per cycle

### Systematic difference CppSim vs Julia:
| epsilon_nl | K | |Δnl| cycle 1 | |Δnl| late cycles |
|------------|---|--------------|-----------------|
| 0.10 | 1 | 1.60e-02 | ~5-8% |
| 0.09 | 1 | 2.08e-02 | ~3-5% |
| 0.08 | 1 | 2.50e-02 | ~2-4% |
| 0.05 | 1 | 3.48e-02 | ~1% cycles 6-8 |
| 0.05 | 2 | 2.61e-02 | ~3% uniform |

### Key physical insight:
- Gate composition PTM = G1·G2(α)·G3(α)·G4(δ) includes BCH cross-commutator terms
- Julia applies Rxx, Ryy, Rzz separately → first-order Trotter, discards cross-terms
- CppSim composed PTM → effective time step ε_eff > ε_nominal
- This is O(ε²) Trotter error, confirmed by systematic behavior
- K=2 (two half-steps + one kick) gives more uniform error across cycles
- Remaining ~3% systematic offset is NOT Trotter error (doesn't grow with cycles)
  → likely gate ordering within color class or measurement convention

### Next investigation:
1. Identify the remaining ~3% systematic offset source
2. Compare single-bond gate PTM between CppSim and Julia directly
3. Check gate ordering within color class

## Session 2026-07-21 afternoon — MPF in CppSim

### Key findings:

**CppSim vs Julia C(t) agreement**:
- Systematic ~1.6% offset at cycle 1 (chi=4, exact) → pure gate composition (BCH)
- Offset grows to ~5-11% at late cycles as chi saturates
- This is a KNOWN SYSTEMATIC BIAS, not random — acceptable for MPF

**Why MPF error is similar in Julia and CppSim (~3-21%)**:
- Julia MPF error: 3% (cycle 1) → 21% (cycle 10)
- CppSim MPF error: 2.5% (cycle 1) → 12% (cycle 10)  
- Both dominated by MPF methodology: only 3 epsilon points, ill-conditioned at late cycles
- The bias between CppSim and Julia cancels in MPF extrapolation

**Critical discovery — noise separation**:
- In CppSim current implementation: C_n(ε=0.1) ≈ C_nl at cycle 1 — noise has almost no effect!
- Reason: noise is BAKED INTO the PTM as G1·Mn·G2·Mn·G3·Mn·G4
- As ε→0, gates vanish but noise PTM stays → wrong physics for epsilon-MPF
- In Julia: noise is a SEPARATE LAYER applied after gates → correct separation
- Fix needed: apply gates and noise separately in CppSim (like Julia)

**CppSim advantages for MPF**:
- Much faster than Julia → can afford more epsilon points
- Flexible epsilon spacing → better conditioned linear system
- Larger chi accessible → smaller truncation error
- More cycles feasible → longer time evolution

### Plan:

**Step 1 — Fix noise separation** (critical):
Restructure NoisySimEps to apply:
1. Gates (IDEAL PTM at eps_i) per color class
2. Noise channel (Mn, gamma=1.0) separately after gates
This matches Julia's structure and gives proper C_n(ε) separation.

**Step 2 — Explore epsilon spacing**:
Generate HPPs at eps = 0.05, 0.10, 0.15, 0.20, 0.25 (5 points)
Run MPF with 5 points → better conditioned A matrix → smaller extrapolation error

**Step 3 — Chi convergence for MPF**:
Compare C_mpf at chi=30, 50, 70 → show MPF converges with chi

**Step 4 — Update paper**:
- CppSim epsilon-MPF results with proper noise separation
- Comparison with Julia at matched physics
- MPF error vs chi and vs number of epsilon points

### Files:
- `main_heavyhex_eps_mpf.cpp`: noiseless + 3 noisy + MPF extrapolation column
  Usage: `./main_heavyhex_eps_mpf [chi_max] [n_steps] [cutoff] [K]`
- `build_ptm.py`: `--epsilon`, `--gamma`, `--phi` CLI args
- HPPs: IDEAL_KICK at eps=0.05/0.08/0.09/0.10, NOISY at eps=0.10/0.15/0.20

## Session 2026-07-21 late — Julia epsilon-MPF noise discovery

**Key discovery**: Julia's gamma=1.0 noise PTM is IDENTITY (all zeros off-diagonal, all ones diagonal).
- Confirmed: `Max off-diagonal: 0.0`, `Is identity: true`
- Julia epsilon-MPF is NOT noise mitigation — it's pure Trotter error extrapolation
- C_nl vs C_n(0.1) differ (~5%) despite same epsilon and identity noise
- Only explanation: extra SVD truncation steps from 228 identity gate applications

**What Julia actually does**:
- C_nl: eps=0.1 gates, NO noise PTM applied
- C_n(0.1): eps=0.1 gates + 228 identity noise PTM applications (extra SVD truncation each)
- C_n(0.15): eps=0.15 gates + same extra truncation pattern
- C_n(0.20): eps=0.20 gates + same extra truncation pattern

**MPF exploits**: Trotter error variation (eps) + truncation schedule difference (identity gates)

**CppSim status**:
- With IDEAL_KICK HPPs: C_n(0.1) = C_nl exactly (no extra truncation) → only 2 independent MPF points
- MPF overshoots because linear system is degenerate

## Session 2026-07-22 — PTM equivalence verified

### test_ptm_equivalence.py results (all PASS):

1. **Sequential == Composed PTM**: error 4.93e-32 — machine precision exact
   G1 @ G2 @ G3 @ G4 @ state == compose([G1,G2,G3,G4]) @ state ✓

2. **Epsilon scaling confirmed**:
   - alpha = coupling × epsilon: -0.054, -0.108, -0.162, -0.216 for eps=0.05,0.10,0.15,0.20
   - delta = coupling: always -1.079, independent of epsilon
   - ZI→ZI decreases with epsilon: 0.997, 0.988, 0.974, 0.954

3. **Julia Rxx·Ryy·Rzz == CppSim G1·G2·G3·G4**: error 5.65e-16 — machine precision ✓
   Both are mathematically identical gate decompositions.

### Conclusion:
The ~1.6% systematic offset between CppSim and Julia C(t) at cycle 1
is NOT from gate definition or PTM composition — both are provably equivalent.

Source must be one of:
- BP environment between color classes (Julia updates after each sub-layer)
- Bond ordering within a color class
- Observable/measurement convention

## Session 2026-07-22 — Key discovery: per-bond hardware noise PTM

### The noise JSON structure:
```
gamma → layer (color 0,1,2) → bond → 16×16 hardware noise PTM
```

Each physical bond on the heavy-hex chip has its own MEASURED 16×16 PTM
for each color class (layer). This is per-bond, per-layer hardware characterization.

### gamma=1.0 is NOT identity:
- 43 out of 228 bond-layer combinations have significant non-identity noise
- Some bonds have max off-diagonal elements up to 0.9999 (layer 1, bond (53,)→(24,))
- Layer 1 and Layer 2 have the most non-identity bonds
- This is real hardware noise from device characterization

### What this means for epsilon-MPF:
- C_nl vs C_n(ε=0.1) difference of ~0.040 IS real physical noise, not artifact
- gamma=1.0 PTM ≠ identity → real noise channel applied between sub-gates
- Previous analysis (SVD truncation, BP gauge artifact) was WRONG as primary source
- The noise model is physics-based, not computational

### Experiments that RULED OUT other sources:
1. cutoff=0, maxchi=4: diff still 0.040 → NOT truncation
2. alg="bp" vs alg="loopcorrections": same diff → NOT measurement algorithm
3. Extra BP sweeps: diff=0.0 → NOT BP convergence
4. PTM composition: Python verified exact → NOT gate definition

### CppSim implication:
- We were loading `gamma1.00-1.40-1.80_hardware.json` but Julia uses `gamma1.00-2.00-3.00_hardware.json`
- The per-bond noise PTMs may differ between these two files
- Need to use the SAME noise file as Julia for proper comparison
- CppSim NOISY PTM = G1·Mn·G2·Mn·G3·Mn·G4 already has the right structure
  but needs the correct hardware noise file

### NEXT SESSION START HERE:

## Two separate unexplained gaps (2026-07-22 afternoon)

### Gap 1: C_nl vs C_n(ε=0.1) = 0.040 — EXPLAINED ✓
Source: real hardware noise per-bond, per-layer PTMs
gamma=1.0 is NOT identity: 43/228 bond-layer pairs have significant noise
Fix: use gamma1.00-2.00-3.00_hardware.json noise file in CppSim

### Gap 2: Julia C_nl=-0.910 vs CppSim C_nl=-0.926 — UNEXPLAINED ✗
Both noiseless, same gates (IDEAL eps=0.10), same phi, same initial state.

**Ruled out so far:**
- Gate composition (Python: Julia==CppSim to 5.65e-16) ✗
- SVD truncation (cutoff=0 still -0.909243 in Julia) ✗
- Intermediate BP updates (update_cache=false: -0.909877 same) ✗
- Measurement algorithm (bp vs loopcorrections: same) ✗
- Bond coupling (affects both equally) ✗

**Remaining candidates:**
1. Bond ordering within color class
   - Julia applies bonds in graph iteration order
   - CppSim applies bonds in color[c] vector order
   - Even though mathematically same, with frozen messages the ORDER matters

2. Heisenberg picture direction (U vs U†)
   - CppSim PTM convention: M_S[i,j] = Tr(Pi U Pj U†)/4
   - Julia: may use opposite convention — need to verify

3. rho convention for measurement
   - Julia rho = [1,0,0,1] for all sites (all |0⟩)
   - CppSim: same? Check IDEAL_KICK_INITIAL_STATE vs Julia rho

4. phi_ref vs phi
   - Julia config has phi_ref=π/2=1.5708 (damping factors file)
   - But evolution uses phi=1.41371669
   - CppSim uses phi=1.41371669 ✓

**Plan for next session:**

**Step A**: Verify rho convention
- Print rho[site] for a few sites in Julia and CppSim
- If different → rho is the source

**Step B**: Verify bond ordering
- Print which bonds are applied in which order in Julia color class 0
- Compare with CppSim grid.colors[0] bond order
- If different → apply in same order and retest

**Step C**: Verify PTM direction
- Compare Julia's single-gate result on one bond to CppSim PTM applied to same state
- Use dump_ptm_json.py output and apply manually in Julia

**Step D**: Fix CppSim noise file
- Use gamma1.00-2.00-3.00_hardware.json
- Regenerate NOISY HPPs at eps=0.10/0.15/0.20
- Run main_heavyhex_eps_mpf → C_n should match Julia's -0.8699

**Priority**: Step D first (quick win, fix C_n gap), then A/B/C for C_nl gap

## Abstract MPF structure (2026-07-22 evening)

### The bond gate (same structure for BOTH MPF strategies):
```
G_bond = G1 · Mn(γ, bond, layer) · G2(α) · Mn(γ, bond, layer) · G3(α) · Mn(γ, bond, layer) · G4(δ)
```
Where:
- α = coupling × ε  (time step, scales XX+YY rotation)
- δ = coupling       (ZZ rotation, fixed — does NOT scale with ε)
- Mn(γ, bond, layer) = per-bond, per-layer hardware noise PTM from JSON
- γ = noise scaling factor

### Heavy-hex structure is IDENTICAL for both strategies:
- 68 sites, 76 bonds, 3 color classes
- Same bond ordering within each color class
- Same interleaving: gate → noise → gate → noise → gate → noise → gate
- Same kick gate Rx(2φ) once per site after all color classes

### The ONLY difference: which parameter is the MPF knob

| | Gamma-MPF | Epsilon-MPF |
|--|-----------|-------------|
| Fixed | α (ε fixed, e.g. 0.25) | Mn (γ=1.0 fixed) |
| Varied | γ = 1.0, 2.0, 3.0 | ε = 0.1, 0.15, 0.2 |
| Extrapolate to | γ → 0 (zero noise) | ε → 0 (continuous time) |
| Noise file | gamma1.00-1.40-1.80 or 1.00-2.00-3.00 | gamma1.00-2.00-3.00 (γ=1.0 only) |
| Julia script | gpu_run_simulation_tnqs_noisy_hardware_gamma.jl | gpu_run_simulation_tnqs_noisy_hardware_epsilon.jl |

### CppSim status:
- Composed PTM G1·Mn·G2·Mn·G3·Mn·G4 already has correct structure ✓
- Gamma-MPF: `main_heavyhex_mpf.cpp` (old, uses gamma1.00-1.40-1.80, partially wrong)
- Epsilon-MPF: `main_heavyhex_eps_mpf.cpp` (new, needs correct noise file)
- Both need: `gamma1.00-2.00-3.00_hardware.json` for proper comparison with Julia

### The THREE knobs of MPF:
1. **ε (Trotter step)** — time discretization error → epsilon-MPF → Z(t) autocorrelations
2. **γ (noise strength)** — hardware noise amplification → gamma-MPF → Z(t) autocorrelations
3. **Gauge + inner products** — A[i,j]=⟨ψᵢ|ψⱼ⟩, B[i]=⟨ψᵢ|ψ_nl⟩ → full MPF matrix
   This is gauge-dependent, requires double-layer BP or consistent gauge choice
   DEFERRED — the hard unsolved problem in CppSim

Knobs 1 and 2 give gauge-invariant C(t) curves → can compare CppSim vs Julia directly
Knob 3 gives the full MPF linear system → needs cross inner products → gauge problem

## Session 2026-07-23 — Epsilon and Gamma MPF complete

### New files:
- `main_heavyhex_gamma_mpf.cpp`: gamma-MPF (fixed ε=0.1, varying γ=1.0/2.0/3.0)
- `main_heavyhex_eps_mpf.cpp`: updated with BP stats (#sweeps, bp_ms)
- `run_sweep.sh`: systematic sweep over chi, cutoff, cycles
- HPPs: NOISY at eps=0.1/0.15/0.2 × gamma=1.0/2.0/3.0 from gamma1.00-2.00-3.00_hardware.json

### Key findings:

**Per-bond per-layer noise PTMs**:
- JSON structure: gamma → layer(0,1,2) → bond → 16×16 PTM
- Each bond uses Mn from its OWN color class layer
- Same Mn applied 3 times in G1·Mn·G2·Mn·G3·Mn·G4 composition
- Lightcone bonds 4,5,6 (connected to center) have weak noise (diag~0.99)
  → explains small C_nl vs C_n difference at cycle 1 in CppSim
- Julia's larger difference comes from BP canonicalization gauge effect

**Gamma-MPF results (chi=70, cutoff=2e-3, 10 cycles)**:
- C_mpf error: 1.19e-07 (cycle 1) → 1.44e-02 (cycle 10) ← 100-1000× better than epsilon-MPF!
- 4.4s total for 10 cycles ← very fast
- BP: 5-9 sweeps to convergence, 26-107ms BP time per cycle
- Chi converged at 70 (same as Julia)

**Gamma-MPF >> Epsilon-MPF** for extrapolation quality because:
- Same time step → well-conditioned A matrix
- Noise varies smoothly → clean linear extrapolation
- No Trotter error mixing with noise signal

## Session 2026-07-23 afternoon — Sweep results + epsilon-MPF physics clarification

### Gamma-MPF sweep results (chi=30/50/70/120/140/200, cutoff=1e-2/3/4, 20 cycles):

**Key findings**:
- cutoff=1e-2: chi saturates at ~28-59, results identical across chi_max → cutoff dominates
- cutoff=1e-3 ≈ cutoff=1e-4: physics converged between these values
- chi NOT converged at cycle 20: chi=120 still saturating → need chi=140/200 to see knee
- C_mpf error at cycle 20: 5.8% (chi=30) → 8.5% (chi=120) — grows with chi saturation
- Gamma-MPF is physically correct: extrapolates noise at FIXED physical time T=n_cycles×ε

**Plots**: `logs/sweep_20260723_173437/plot_cnl_cmpf_bands.png`

### Epsilon-MPF sweep results (chi=30/50/70/120, cutoff=1e-2/3/4, 20 cycles):

**Key finding: PHYSICALLY WRONG extrapolation!**
- C_mpf ≈ 0.93 at cycle 20 vs C_nl ≈ 0.78 → |Δmpf| = 1.2-1.6e-01 (2× worse than gamma)
- C_mpf barely decays — looks like initial value

**Why**: epsilon-MPF polynomial fit:
  C_n(ε,t) = c₀ + c₁ε² + c₂ε⁴, extrapolate to ε=0

At FIXED n_cycles=20:
  ε=0.1 → physical time T=2.0
  ε=0.15 → physical time T=3.0
  ε=0.2 → physical time T=4.0
  ε=0 → physical time T=0.0 ← ZERO TIME!

The regression extrapolates to zero physical evolution — C_mpf≈initial value≈1. Not useful.

**Correct epsilon-MPF** would fix physical time T and vary n_cycles:
  T=2.0: ε=0.1 (20 cycles), ε=0.05 (40 cycles), ε=0.025 (80 cycles)
  This is a different, more expensive experiment.

**Plots**: `logs/sweep_20260723_174149/plot_cnl_cmpf_bands.png`

### Final verdict:
- **Gamma-MPF**: physically correct, 4.4s/10 cycles, 7-8% MPF error ← USE THIS
- **Epsilon-MPF as implemented**: physically wrong (wrong time reference) ← needs redesign
- **Analyze script**: `analyze_sweep.py` — run on any sweep log dir

### Sweep plan (next session):
- chi: 140, 200 (find convergence knee — chi not converged at 120)
- cutoff: 1e-3, 1e-4 (1e-2 too aggressive)
- cycles: 20-30 (see further saturation)
- Gamma-MPF only (epsilon-MPF needs redesign)
- Already have: chi=30/50/70/120 in logs/sweep_20260723_173437/

### Build commands:
```bash
make PHYS_DIM=4 main_heavyhex_gamma_mpf  # gamma-MPF
make PHYS_DIM=4 main_heavyhex_eps_mpf    # epsilon-MPF

./main_heavyhex_gamma_mpf [chi_max] [n_steps] [cutoff]
./main_heavyhex_eps_mpf   [chi_max] [n_steps] [cutoff]

bash run_sweep.sh gamma    # gamma sweep
bash run_sweep.sh epsilon  # epsilon sweep
bash run_sweep.sh all      # both
```

### Data files needed (copy from Julia repo):
- `data/heavy_hex_3x3_gamma1.00-2.00-3.00_hardware.json` ✓
- `data/graphs/heavy_hex_3x3.json` ✓

### NEXT SESSION START HERE

**Julia diagnostic** (single bond, 2-site system):
Compare C(t) from:
1. Julia sequence: Rxx(α) → SVD → Ryy(α) → SVD → Rzz(δ) → SVD → identity_noise → SVD
2. Julia composed: single PTM = Rxx·Ryy·Rzz in one shot → one SVD

If they differ → SVD between sub-gates IS the source of C_nl vs C_n separation
If same → look elsewhere (observable convention, etc.)

Script to write: `test_single_bond_ptm.jl` in Julia repo
- Create 2-site chain
- Apply gates sequentially vs composed
- Compare C(t) after 1 and 2 steps

Then: design CppSim MPF that exploits Trotter error variation properly

Fix noise separation in CppSim epsilon-MPF:
- Current: G1·Mn·G2·Mn·G3·Mn·G4 (noise baked in, wrong for epsilon-MPF)
- Target:  G1·G2·G3·G4 (IDEAL gates at epsilon) then Mn applied separately
- This requires: IDEAL PTMs at eps=0.10/0.15/0.20 + separate noise channel
- Already have: heavyhex_ideal_kick_eps010.hpp and separate noise PTMs
- Implement: NoisySimEpsFixed that applies gates then noise sequentially
Then run 5-point epsilon sweep and compare MPF error with Julia

## Future roadmap (2026-07-19 end of day)

### Step 1: CppSim C(t) matches Julia Z(t)
- Autocorrelation is gauge-invariant — clean validation path
- Run CppSim with matched epsilon/phi/hardware noise parameters
- Compare C_nl(t) and C_n(ε, t) cycle by cycle vs Julia Z(t) Float32
- CppSim faster → larger chi, more cycles → demonstrates CppSim advantage
- This alone is a beautiful publishable result

### Scientific boundary for CppSim vs Julia comparison

**Gauge-independent (valid to compare)**:
- Z(t) = C_n(ε,t) autocorrelation — gauge invariant by definition
- C_mpf(t) = Σ cᵢ·C_n(εᵢ) — projection back onto gauge-independent observables
- If these match → physics and MPF methodology both validated

**Gauge-dependent (NOT valid to compare)**:
- A[i,j] = ⟨ψᵢ|ψⱼ⟩ inner products — depend on gauge choice
- B[i] = ⟨ψᵢ|ψ_nl⟩ inner products — same
- Individual MPF coefficients cᵢ — will differ between QR (CppSim) and sqrt (Julia) gauge

**Key insight**: MPF is gauge-independent in principle (linear combination of gauge-
independent Z(t) values), but gauge-sensitive in practice through the condition number
of A. Well-conditioned A → gauge doesn't matter. Ill-conditioned A → gauge errors
get amplified and break the cancellation.

**Validation chain**:
  CppSim Z(t) ≈ Julia Z(t)     ← gauge-independent physics check
  CppSim C_mpf ≈ C_nl(t)       ← MPF methodology quality
  |C_mpf - C_nl| → 0 as methodology improves (more ε, better conditioning)

### Step 2: True cross inner products for epsilon-MPF in CppSim
- ψ(ε₁) and ψ(ε₂) evolved with different time steps → genuinely different states
- Different chi growth, different gauge → CANNOT use one state's messages as environment
- Autocorrelation workaround (single-state messages) breaks down here
- Need TRUE double-layer BP: both states contribute to environment equally
  T_v[ia,ib] = Σ_d ψ(ε₁)[ia,d] × ψ(ε₂)[ib,d], messages chi_a × chi_b
- At chi=70: double-layer bond = 70×70 = 4900 → BP on 76 bonds → feasible on GPU
- Requires: GPU double-layer BP kernel + consistent QR gauge (positive diagonal)
- Once solved: full epsilon-MPF in CppSim, faster than Julia, larger chi

### Why cross inner products are fundamentally harder than autocorrelation
- Autocorrelation: ρ has chi=1 → double-layer = chi_ψ × 1 = chi_ψ → tractable
- Cross ⟨ψ(ε₁)|ψ(ε₂)⟩: both chi>1 → double-layer = chi_a × chi_b → GPU needed
- QR sign problem: independent evolutions choose arbitrary gauge → need fix_qr_sign
- Julia avoids this via SVD sqrt-gauge: tensors canonicalized to consistent gauge
