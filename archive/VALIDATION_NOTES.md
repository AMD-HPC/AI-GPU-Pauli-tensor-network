# Heavy-Hex Julia → CppSim Physics Validation Notes

## Overview

This document records the validation process for translating the Julia
`amd-TensorNetworkQuantumSimulator.jl` heavy-hex MPF workload into CppSim.
Each section states what was verified, how to reproduce it, and what remains open.

**Julia repo**: `/scratch/paolod/QC/amd-TensorNetworkQuantumSimulator.jl`
**CppSim repo**: `/scratch/MPF/CppSim`
**Key script**: `workloads/noisy_ptm_simulation/compute_overlaps_gpu.jl`

---

## 1. Physical Model

**Kicked Heisenberg model on heavy_hex_3x3:**
- H = J·Σ(XX+YY+ZZ) on 68 sites, 76 bonds, 3-colorable
- Hardware-native gate decomposition (not exp(-iθ(XX+YY+ZZ)))
- Trotter step: gates1 → gates2(α) → gates3(α) → gates4(δ) per color class
- Kick gate: Rx(2φ) on every site after each full Trotter step
- Parameters: epsilon=0.25, phi=1.41371669, k=1, dt=1/k=1.0
- α = -coupling × epsilon (per bond, from device JSON)
- δ = -coupling (per bond, from device JSON)

**Why kicked?** The kick gate Rx(2φ) makes the model quantum chaotic,
driving faster operator spreading. Without it, chi grows slowly and the
model is too easy to simulate classically.

**Effect on C(t):** C(t) alternates sign every cycle due to the kick gate
(cos(2φ) ≈ -0.951 rotates Z → -Z + small Y mixing).

---

## 2. PTM Gate Validation

### What was verified
The gate PTM for one bond computed in `build_ptm.py` matches Julia's
gate angles exactly.

### Julia convention (IMPORTANT)
Julia uses `Δ = -coupling` (negative sign) and `α = Δ × epsilon`.
CppSim `build_ptm.py` must use the same sign convention.

### PTM convention (IMPORTANT)
Julia's `toitensor_heisenberg` computes M_H (Heisenberg PTM) then stores
`transpose(M_H)` = M_S (Schrödinger PTM). This means Julia applies
`O → U O U†` (Schrödinger convention) to the operator tensors.

**Current status**: Verified coupling angles match. PTM convention under investigation.

### How to reproduce

**Step 1: Check coupling angles match**
```bash
cd /scratch/MPF/CppSim
python3 -c "
from build_ptm import *
import numpy as np, json
with open('data/graphs/heavy_hex_3x3.json') as f:
    d = json.load(f)
coupling_map = {}
for e in d['edges']:
    u=e['edge'][0][0]-1; v=e['edge'][1][0]-1
    coupling_map[(min(u,v),max(u,v))] = e['coupling']
c = coupling_map[(21,50)]
print(f'CppSim alpha={-c*0.25:.17f}')
print(f'Julia  alpha=-0.24299147364462986')
print(f'Match: {abs(-c*0.25 - (-0.24299147364462986)) < 1e-6}')
"
```

**Step 2: Check Julia gate angles**
```bash
cd /scratch/paolod/QC/amd-TensorNetworkQuantumSimulator.jl
julia --project=. -e "
include(\"workloads/noisy_ptm_simulation/src/graphs_utils.jl\")
include(\"workloads/noisy_ptm_simulation/src/tnqs_gates.jl\")
graph_data = load_graph_json(\"data/graphs/heavy_hex_3x3.json\")
gates = setup_hardware_evolution_gates(graph_data, 1, 0.25)
# gates[1] = color 0, element 3 = gates2 group
# Find Rzz for bond (22,51) in group 3
for g in gates[1][3]
    if length(g) == 3 && g[1] == \"Rzz\" && g[2] == [(22,),(51,)]
        println(\"Rzz angle for bond (22,51): \$(g[3])\")
    end
end
" 2>/dev/null
# Expected: -0.24299147364462986
```

**Step 3: Check Rzz angles match for all bonds**
Expected: CppSim alpha = -coupling × 0.25, Julia Rzz = same value.

---

## 3. Heisenberg Operator Initialization

### What was verified
Both Julia and CppSim initialize Z at center site and I elsewhere.

**Julia:**
```julia
const stringtostatemap = Dict("I"=>[1,0,0,0], "X"=>[0,1,0,0], "Y"=>[0,0,1,0], "Z"=>[0,0,0,1])
psi = paulitensornetworkstate(Float32, v -> v == obs_site ? "Z" : "I", g)
# Z at obs_site → tensor [0,0,0,1] in Pauli basis {I,X,Y,Z}
# I elsewhere  → tensor [1,0,0,0]
```

**CppSim** (`include/heavyhex_physics.hpp`):
```cpp
std::vector<Scalar> h_I = {1.f, 0.f, 0.f, 0.f};  // I operator
std::vector<Scalar> h_Z = {0.f, 0.f, 0.f, 1.f};  // Z operator
// Z at center, I elsewhere
```

### How to reproduce
**CppSim site tensor verification** (`test_inner_op_state.cpp`, Group 1):
```bash
cd /scratch/MPF/CppSim
./test_inner_op_state 2>/dev/null | grep "t=0"
# All 3 t=0 tests should PASS with C(0)=1.0
```

**Julia C(0) verification:**
```bash
julia --project=. check_tensors.jl 2>/dev/null | grep "C(0)"
# Expected: C(0)=1.0 for both device rho and all-zero rho
```

### Status: ✓ VERIFIED — both give C(0)=1.0

---

## 4. Inner Product — Three Different Measurements

This is the most important section. Julia and CppSim compute DIFFERENT
inner products depending on what ρ is used. All three are valid but
measure different physics.

### 4a. `inner(ψ(t), Z⊗I...⊗I)` — operator autocorrelation

**Definition**: How much Z remains at the center site after time t.
**CppSim**: `autocorrelation_with_bp` using `d_sel_Z` at center, `d_sel_I` elsewhere.
**Julia**: `inner(ψ(t), psi0)` where `psi0` = Z at center, I elsewhere.

**Verified match at cycle 1:**
- CppSim: -0.803318
- Julia: -0.803318 ✓

**How to reproduce:**
```bash
# CppSim
./test_inner_op_state 2>/dev/null | grep "autocorrelation"
# Julia
julia --project=. check_tensors.jl 2>/dev/null | grep "Z.I"
# Expected: both give -0.803318
```

### 4b. `inner(ψ(t), all-zero ρ)` — all qubits in |0⟩

**Definition**: Expectation of evolved Z in state where all qubits are |0⟩.
**ρ_v = [1,0,0,+1]** for all v (I+Z)/2 = |0⟩⟨0|

**Verified values at cycle 1:**
- Julia: -0.961822
- CppSim `inner_op_state(all+1)`: -0.830471 ← DOES NOT MATCH YET

**Why it differs**: `inner_op_state` BP not yet computing correct bilinear form.

### 4c. `inner(ψ(t), device ρ)` — actual device initial state

**Definition**: Expectation of evolved Z in actual device qubit configuration.
**ρ_v = [1,0,0,±1]** per site from `heavy_hex_3x3.json` initial_state.
- 30 sites: spin=+1 → ρ_v = [1,0,0,+1] = |0⟩⟨0|
- 38 sites: spin=-1 → ρ_v = [1,0,0,-1] = |1⟩⟨1|
- Center site 2: spin=+1

**Verified values at cycle 1:**
- Julia: -0.678563 ← GOLDEN REFERENCE
- CppSim `inner_op_state(device)`: -0.821118 ← DOES NOT MATCH YET

**How to reproduce:**
```bash
# Julia golden
julia --project=. check_tensors.jl 2>/dev/null | grep "device\|all-zero\|Z.I"

# CppSim (currently failing)
./test_inner_op_state 2>/dev/null | grep "inner_op_state\|PASS\|FAIL"
```

### Key insight
The difference between 4a and 4b/4c is REAL physics — the initial state ρ
matters. CppSim previously only computed 4a (autocorrelation). Julia computes
4c (device ρ) as the primary observable.

---

## 5. Initial State ρ from Device JSON

### Structure
```json
"initial_state": {"(1,)": 1, "(2,)": 1, "(3,)": 1, ..., "(57,)": -1, ...}
```
1-based Julia vertex → 0-based CppSim site id: `site_id = vertex - 1`
+1 = qubit in |0⟩, -1 = qubit in |1⟩

### Julia construction
```julia
rho = tensornetworkstate(Float32,
    v -> initial_state[string(v)] == 1 ? [1f0,0f0,0f0,1f0] : [1f0,0f0,0f0,-1f0],
    g, site_indices)
```

### CppSim constant
`HEAVY_HEX_3x3_IDEAL_KICK_INITIAL_STATE[68]` in `heavyhex_ideal_kick_data.hpp`
Generated by `build_ptm.py` from graph JSON.

### How to verify
```bash
python3 -c "
import json, re
with open('/scratch/MPF/CppSim/data/graphs/heavy_hex_3x3.json') as f:
    d = json.load(f)
state = d['initial_state']
plus  = sum(1 for v in state.values() if v==1)
minus = sum(1 for v in state.values() if v==-1)
print(f'+1 (|0>): {plus} sites, -1 (|1>): {minus} sites')
# Check center site 2 (Julia vertex (3,))
for k,v in state.items():
    m = re.findall(r.((\d+),).', k)
    if m and int(m[0])-1 == 2: print(f'Center site 2: {v}')
"
# Expected: +1 (|0>): 30, -1 (|1>): 38, center=+1
```

---

## 6. `inner_op_state` — Current Status and Algorithm

### Critical discovery (2026-07-16 afternoon)

**Root cause of ~0 result:** CppSim's QR gauge concentrates Pauli content in bond
dimension (messages), not site tensor physical index. After one Trotter step:
- phi_center[0,0,0] = psi_center[0,0,0,d] · ρ = [0,-0.087,0,0] · [1,0,0,±1] = 0
  → initial message [1,0,0,0] picks this zero → BP stuck → result ≈ 0

Julia's symmetric gauge has non-zero content at chi-index 0 → works correctly.

**test_inner_minimal.jl** (in Julia repo) validates step-by-step:
- 2-site at t=0: all tests pass C=+1 ✓
- heavy-hex t=1: Julia messages evolve [1,0,0,0]→[1.073,0.030,0.165,-0.268] in 1 sweep ✓
- BilinearForm factors at center: ψ(4,4,4,4) + delta(4,4) + ρ(4,1,1,1)

**What works (partially):**
- bp_partitionfunction with rho ops [1,0,0,±1] + autocorr messages: -0.830/-0.821
  Signs correct, ~15% off from targets -0.962/-0.679

**Current attempt in inner_op_state.hpp:**
- init chi-vector from |diag(autocorr_msg)| / L2-norm
- phi_v BP with L2 normalization
- Result not yet tested after lunch

**Next ideas if current fails:**
- Pre-transform ψ to symmetric gauge (absorb sqrt of messages into site tensors)
  THEN run BilinearForm BP → phi_v has content at all chi-indices
- This is the mathematically correct approach but requires accurate sqrt(M) computation

### Goal
Implement `C(t) = inner(ψ(t), ρ; alg="bp")` matching Julia exactly.

### Julia algorithm (from `src/inner.jl`)
```julia
function ITensors.inner(alg::Algorithm"bp", ψ, ϕ; cache_update_kwargs, ...)
    ψϕ_bpc = BeliefPropagationCache(BilinearForm(ψ, ϕ))  # fresh identity messages
    ψϕ_bpc = update(ψϕ_bpc; cache_update_kwargs...)        # BP sweeps
    return cache_partitionfunction(alg, ψϕ_bpc)            # Bethe contraction
end
```

**Key detail**: `BeliefPropagationCache(BilinearForm(ψ, ϕ))` initializes
messages to `delta(linds)` = plain identity matrix (NOT 1/chi scaled).
Messages have dimension = bond_dim(ψ) × bond_dim(ρ) = chi × 1 = chi.

### CppSim implementation (`include/measures.hpp`)
Current approach: `bp_update_with_rho` — runs BP on ψ using per-site ρ
selectors `[1,0,0,±1]` instead of `[1,0,0,0]`. Fresh identity messages.

**Current results at cycle 1:**
- `all+1 ρ`: -0.830 (target: -0.962) ← FAILING
- `device ρ`: -0.821 (target: -0.679) ← FAILING

### Known issue
`bp_update_with_rho` selects a Pauli component with a vector selector.
Julia's BilinearForm BP contracts ψ_v ⊗ ρ_v as a joint tensor, then
runs standard BP on the product. These are NOT equivalent operations.

### Correct algorithm (TO IMPLEMENT)
1. For each site v: contract `ψ_v[bonds,d]` with `ρ_v[d]` → `φ_v[bonds]`
   (physical index eliminated, bond structure preserved)
2. Initialize fresh identity messages (chi×chi identity)
3. Run standard BP on φ_v tensors using the existing `bp_update` formula
   adapted for D=1 physical dimension
4. Bethe partition function on φ_v

The challenge: `bp_update` uses hardcoded D=4. Need `bp_update_D1` that
works with φ_v (physical dim=1, or equivalently no physical index).

### Test
```bash
cd /scratch/MPF/CppSim
./test_inner_op_state 2>/dev/null | grep -E "PASS|FAIL|Results"
# Currently: 4 pass, 2 fail (device ρ and all+1 ρ don't match Julia)
# Target: 6 pass, 0 fail
```

---

## 7. Kick Gate Validation

### Julia
```julia
kick_gates = setup_kick_gates(g, phi)  # Rx(2φ) on every vertex
push!(reversed_ev_gates, [kick_gates]) # applied LAST in Heisenberg evolution
```

### CppSim
`trotter_layer()` in `heavyhex_physics.hpp` applies kick after all 3 color classes:
```cpp
// Wave 2: kick gate — Rx(2φ) once per site
gate_ops.apply_unary_gate(state.psi[site.id], site, grid,
                          HEAVY_HEX_3x3_IDEAL_KICK_SITE_PTM, tmp);
```
`HEAVY_HEX_3x3_IDEAL_KICK_SITE_PTM` is a 4×4 PTM for Rx(2φ) in `heavyhex_ideal_kick_data.hpp`.

**Why separate from bond PTM**: kick acts on each site ONCE regardless of
degree (2 or 3 bonds). Baking into bond PTM would apply it 2-3 times per site.

### Verification: sign alternation of C(t)
```bash
# CppSim: should show alternating signs
./main_heavyhex 430 4 1e-4 --center=2 2>/dev/null | grep "^[0-9]"
# Expected: C(1)<0, C(2)>0, C(3)<0, C(4)>0

# Julia golden:
# Cycle 1: -0.6786, Cycle 2: +0.8127, Cycle 3: -0.7205, Cycle 4: +0.6808
```

---

## 8. Golden Reference Table

### Julia (Float32, epsilon=0.25, phi=1.41371669, cutoff=1e-4, chi=uncapped)
**Script**: `compute_overlaps_gpu.jl heavy_hex_3x3 10 200`
**Log**: `logs/golden_noiseless_gamma10_eps025_maxdim200.log`

| Cycle | C_nl (device ρ) | C_n(γ=1.0) | overlap⟨ψ_nl\|ψ_n⟩ | χ_nl | χ_n |
|-------|-----------------|------------|---------------------|------|-----|
| 1 | -0.6786 | -0.6665 | 0.9809 | 4 | 4 |
| 2 | +0.8127 | +0.7396 | 0.9312 | 14 | 14 |
| 3 | -0.7205 | -0.6146 | 0.8320 | 28 | 24 |
| 4 | +0.6808 | +0.5475 | 0.7217 | 58 | 39 |
| 5 | -0.6916 | -0.5058 | 0.6208 | 110 | 58 |
| 6 | +0.6207 | +0.4435 | 0.5330 | 202 | 82 |
| 7 | -0.6346 | -0.4044 | 0.4528 | 364 | 108 |
| 8 | +0.5921 | +0.3555 | 0.3832 | 637 | 134 |
| 9 | -0.5998 | -0.3292 | 0.3243 | 1086 | 154 |
| 10 | OOM | — | — | ~2000 | ~175 |

### Julia inner product equivalence (CRITICAL)
Verified at cycle 1 via `check_tensors.jl`:
| ρ used | Julia C(t=1) | CppSim equivalent |
|--------|-------------|-------------------|
| Z⊗I...⊗I | **-0.803318** | `autocorrelation_with_bp` = -0.803318 ✓ |
| all-zero ([1,0,0,+1]) | **-0.961822** | `inner_op_state(all+1)` = -0.830 ✗ |
| device ρ | **-0.678563** | `inner_op_state(device)` = -0.821 ✗ |

---

## 9. Reproduction Scripts

### Full validation sequence
```bash
# Step 1: Build CppSim PTMs
cd /scratch/MPF/CppSim
python3 build_ptm.py data/heavy_hex_3x3_gamma1.00-1.40-1.80_hardware.json \
  data/graphs/heavy_hex_3x3.json IDEAL_KICK include/heavyhex_ideal_kick_data.hpp

# Step 2: Build and run CppSim tests
make PHYS_DIM=4 test_inner_op_state main_heavyhex 2>&1 | tail -3
./test_inner_op_state 2>/dev/null        # must be 6/6 before proceeding

# Step 3: Run CppSim golden (sign alternation check)
./main_heavyhex 430 10 1e-4 --center=2 2>/dev/null | grep "^[0-9]"
# Cycles should alternate sign like Julia

# Step 4: Julia check_tensors (inner product comparison)
cd /scratch/paolod/QC/amd-TensorNetworkQuantumSimulator.jl
julia --project=. check_tensors.jl 2>/dev/null | head -5

# Step 5: Julia golden reference (takes ~30 min)
julia --project=. workloads/noisy_ptm_simulation/compute_overlaps_gpu.jl \
  heavy_hex_3x3 10 200 2>&1 | tee logs/golden_noiseless_gamma10_eps025_maxdim200.log
```

---

## 10. Open Items (Blockers)

### BLOCKER: `inner_op_state` wrong BP
**File**: `include/measures.hpp`
**Test**: `test_inner_op_state.cpp` — currently 4/6 pass
**Fix needed**: Replace `bp_update_with_rho` with proper φ_v BP:
  1. Contract ψ_v with ρ_v → φ_v (no physical index)
  2. Run standard BP on φ_v with D=1
  3. Bethe partition function on φ_v

### Once BLOCKER resolved:
- Wire `inner_op_state` into `main_heavyhex` output
- Run NOISY model, compute C_n(γ) for γ=1.0,1.4,1.8
- Implement A matrix: ⟨ψ_noisy(γᵢ)|ψ_noisy(γⱼ)⟩
- Run full MPF pipeline

### DO NOT PROCEED to noisy/MPF until test_inner_op_state passes 6/6.
