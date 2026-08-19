# Plan: ComputeScalar Abstraction (float32 / float64 compute paths)

## Goal

Single codebase, two compile-time binaries:
```bash
make COMPUTE_SCALAR=float   # fast production path (current)
make COMPUTE_SCALAR=double  # accurate path for large chi
```

Storage always float32 (psi, messages, messages_v).
Computation switches between float32 and float64.

**Why**: at chi=450 with δ=1e-4 the vertex_scalar_diag normalization chain
loses accuracy. Full float64 computation eliminates this without changing
memory layout.

---

## Design Principle — REVISED: Same Type Throughout

```
using Scalar = float;   // -DScalar=float  (default, current)
using Scalar = double;  // -DScalar=double (accurate, 2× memory)
```

**Everything is Scalar**: storage, computation, workspace, messages.
No mixed-precision boundaries. No cast kernels. One type flows end-to-end.

This is simpler than mixed precision:
- One line change (Scalar typedef)
- All kernels, BLAS calls, buffers change together automatically
- No silent precision loss at cast boundaries
- Easy to reason about, easy to test

**Memory tradeoff**:
| Machine | Scalar=float | Scalar=double | With lightcone (50%) |
|---------|-------------|---------------|---------------------|
| MI100 (32GB) | chi=400 | chi=283 | chi=400 double |
| MI355X (309GB) | chi=1200 | chi=850 | chi=1200 double |

**MI355X opportunity**: at double precision with 309GB, chi=565 fits easily.
This is the target machine for production runs once the float64 binary is validated.

---

## CMake / Makefile Flag

```makefile
# Makefile addition
COMPUTE_SCALAR ?= float
ifeq ($(COMPUTE_SCALAR),double)
    DEFINES += -DCOMPUTE_SCALAR_DOUBLE
endif
```

```cpp
// types.hpp addition
#ifdef COMPUTE_SCALAR_DOUBLE
  using CScalar = double;  // computation scalar
#else
  using CScalar = float;   // computation scalar (default)
#endif
// Storage scalar always float32:
using Scalar = float;
```

---

## Functions to Modify (~10 locations)

### gate.hpp

**1. GateOps::sgemm()** (line ~244)
```cpp
// Current: rocblas_sgemm
// Change:  dispatch on CScalar
template<> void sgemm<float>(...) { rocblas_sgemm(...); }
template<> void sgemm<double>(...) { rocblas_dgemm(...); }
```

**2. GateOps::mode_n_contract()** (inline in gate)
```cpp
// Current: rocblas_sgemm_strided_batched
// Change:  dispatch on CScalar
```

**3. GateOps::householder_qr()** (~line 527)
```cpp
// Current: rocsolver_sgeqrf, rocsolver_sorgqr
// Change:  sgeqrf/dgeqrf, sorgqr/dorgqr
// Mixed-precision: upcast Q/cv buffers before call, downcast after
```

**4. GateOps::cholesky_qr()** (~line 556)
```cpp
// Current: rocsolver_spotrf, rocblas_strsm
// Change:  dpotrf, dtrsm
```

**5. GPU kernels via CScalar template** (no change needed)
```cpp
// All Scalar* kernels auto-switch via using Scalar = float/double
// BUT: Scalar stays float32 (storage) — need new CScalar kernels for compute
// Solution: add CScalar template parameter to hot kernels
```

### measures.hpp

**6. mode_n_contract()** (~line 285)
```cpp
// Current: rocblas_sgemm_strided_batched
// Change:  dispatch to sgemm or dgemm based on CScalar
// Existing: mode_n_contract_d() already has dgemm version
// Action:   rename/unify into single template function
```

**7. bp_update()** (~line 1133) — SGEMV and SDOT calls
```cpp
// Current: rocblas_sgemv (×4), rocblas_sdot (×1)
// Change:  rocblas_dgemv, rocblas_ddot when CScalar=double
// Also:    normalize_and_delta vs normalize_and_delta_d dispatch
// Impact:  this is the hot BP loop — doubles the compute cost but
//          eliminates imprecision in vertex_scalar_diag
```

**8. bp_update_with_rho()** (~line 1270)
```cpp
// Same as bp_update() — same SGEMV/SDOT replacements
```

**9. vertex_scalar_diag() — NO CHANGE NEEDED**
```cpp
// Already uses float64 computation (mode_n_scale_d_kernel + dbl_a/dbl_b)
// The normalization chain and final sum are float64
// BUT: the intermediate tensors passed via dbl_a/dbl_b need full-precision
//      input — if bp_update() uses double, messages_v will be more accurate
//      and vertex_scalar_diag will automatically benefit
```

### workspace.hpp

**10. Workspace::Workspace()** — add double compute buffers
```cpp
// When CScalar=double: allocate double-precision versions of:
//   cv1_d, cv2_d: m_max × n_max doubles (for QR in double)
//   Q1_d, Q2_d:  m_max × n_max doubles
//   gram_d:      n_max × n_max doubles
// Memory cost: 2× workspace size
// Alternative: reuse existing oR4_d/U_d/Vt_d for temporary staging
```

---

## What Does NOT Change

| Component | Reason |
|-----------|--------|
| `using Scalar = float` | Storage stays float32 always |
| psi[], messages[], messages_v[] | Float32 storage buffers |
| cast_f32_to_f64/cast_f64_to_f32 | Already exist, used at boundaries |
| upcast_f2d/downcast kernels | Already exist |
| Log-space partition function | Already uses double accumulation |
| SVD (rocsolver_dgesvd) | Already float64 |
| Gate output (messages diag) | Already stored as float32 |
| Trotter driver | Passes pointers, type-agnostic |
| Grid/Bond/Site | Type-agnostic |

---

## Build Targets

```makefile
main_pauli_sweep:           COMPUTE_SCALAR=float  (default, fast)
main_pauli_sweep_f64:       COMPUTE_SCALAR=double (accurate, ~2x slower)
main_pauli_sweep_hh_dep:    COMPUTE_SCALAR=float
main_pauli_sweep_rect7x7:   COMPUTE_SCALAR=float
```

The float64 binary uses the same grid files, PTMs, and scripts.
Only the compute path changes.

---

## Expected Benefit

At chi=450, δ=1e-4:
- float32 compute: vertex_scalar_diag normalization loses accuracy
- float64 compute: normalization chain stays in float64 throughout
- Result: C(t) correct at step 9 with gate_err < δ

---

## Phase 0: sqrt_env chi×chi → chi-vector (prerequisite for float64)

The env absorption/deabsorption currently uses chi×chi SGEMM with sqrt_env/inv_sqrt_env.
These matrices are DIAGONAL (produced by diag_sqrt_matrices which zeros off-diagonal).
Storing and contracting them as chi×chi is wasteful and blocks float64 env computation.

### 0.1 Add chi-vector env buffers to Workspace

```cpp
// workspace.hpp: replace chi×chi sqrt_env with chi-vectors
Scalar* sqrt_env_v    [MAX_DEGREE] = {};  // chi-vector (diagonal of sqrt_env)
Scalar* inv_sqrt_env_v[MAX_DEGREE] = {};  // chi-vector (diagonal of inv_sqrt_env)
// Size: chi_max floats each (vs chi_max² currently)
```

### 0.2 Update pseudo_sqrt to populate chi-vectors

```cpp
// gate.hpp: after diag_sqrt_matrices, also extract diagonals to chi-vectors
extract_diagonal<<<>>>(sqE,  sqrt_env_v[li],     chi);
extract_diagonal<<<>>>(isqE, inv_sqrt_env_v[li],  chi);
```

### 0.3 Replace absorb_env mode_n_contract with mode_n_scale

```cpp
// gate.hpp: absorb_env — replace:
mode_n_contract(cur, shape, ndim, ws.sqrt_env[li], chi, chi, li, nxt);
// with:
mode_n_scale(cur, shape, ndim, ws.sqrt_env_v[li], chi, li, nxt);
// O(chi × volume) instead of O(chi² × volume)
```

### 0.4 Replace deabsorb_env mode_n_contract with mode_n_scale

```cpp
// gate.hpp: deabsorb_env — same replacement with inv_sqrt_env_v
```

**Test after Phase 0**: chi=70 δ=1e-3 baseline unchanged, chi=450 faster per gate step.
This is also the prerequisite for float64 env absorption (mode_n_scale_d is already available).

---

## Implementation Order (Same-Type Design)

### Phase 0: sqrt_env chi×chi → chi-vector (performance, independent of Scalar type)

- [ ] 0.1 Add sqrt_env_v / inv_sqrt_env_v chi-vectors to Workspace (size chi_max per direction)
- [ ] 0.2 Update pseudo_sqrt to populate chi-vectors
- [ ] 0.3 Replace absorb_env: mode_n_contract(chi×chi) → mode_n_scale(chi-vector)
- [ ] 0.4 Replace deabsorb_env: same replacement with inv_sqrt_env_v
- [ ] 0.5 Test: chi=70 δ=1e-3 baseline unchanged (C(t) identical)
- [ ] 0.6 Commit Phase 0

### Phase 1: Scalar typedef + Makefile flag

- [ ] 1.1 types.hpp: keep `using Scalar = float` as default
        Add: `-DScalar=double` flag → `using Scalar = double`
- [ ] 1.2 Makefile: add `main_pauli_sweep_f64` target with `-DScalar=double`
- [ ] 1.3 Test compile: both targets build without errors

### Phase 2: BLAS/solver dispatch by Scalar type

All changes are `#ifdef` or template dispatch — existing Scalar kernels auto-switch.

- [ ] 2.1 `GateOps::sgemm()`: `rocblas_sgemm` → `rocblas_dgemm` when Scalar=double
- [ ] 2.2 `GateOps::mode_n_contract()`: sgemm_strided_batched → dgemm_strided_batched
- [ ] 2.3 `GateOps::householder_qr()`: sgeqrf/sorgqr → dgeqrf/dorgqr
- [ ] 2.4 `GateOps::cholesky_qr()`: spotrf/strsm → dpotrf/dtrsm
- [ ] 2.5 `bp_update()`: sgemv/sdot → dgemv/ddot
- [ ] 2.6 `normalize_and_delta()` kernel: Scalar template (auto-switches)
- [ ] 2.7 Test: chi=70 δ=1e-3 float64 binary matches float32 baseline

### Phase 3: Validation

- [ ] 3.1 chi=200 δ=1e-4 float64: all 10 steps correct, gate_err < δ
- [ ] 3.2 chi=283 δ=1e-4 float64 (same VRAM as chi=400 float32): stable?
- [ ] 3.3 MI355X: chi=565 float64 with lightcone — faithful 10 cycles?

## Full Computation Chain (Scalar=double, same type throughout)

```
psi[double] → absorb_env (mode_n_scale, chi-vector sqrt_env_v[double])
  → cv[double] → QR (dgeqrf, dorgqr) → R1,R2[double]
    → Bond GEMM (dgemm): R1^T R2 → RR4[double]
      → Gate → SVD (dgesvd) → λ_k[double] → messages_v[double]
        → Reconstruction (dgemm) → cv_new[double]
          → deabsorb_env (mode_n_scale, inv_sqrt_env_v[double])
            → psi_new[double]
              → BP (dgemv/ddot): messages updated in double
                → vertex_scalar_diag (mode_n_scale_d → mode_n_scale)
```

No casts. No boundaries. Everything double when -DScalar=double.
Memory doubles but logic is identical — one type flows end-to-end.
