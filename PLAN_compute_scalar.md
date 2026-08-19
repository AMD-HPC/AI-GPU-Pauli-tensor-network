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

## Design Principle

```
Storage (always float32):     psi, messages, messages_v, workspace buffers
Computation (switchable):     QR, GEMM, SVD, BP SGEMV, vertex contractions
```

Mixed-precision boundary: upcast float32→float64 at computation entry,
downcast float64→float32 at storage exit. These cast kernels already exist.

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

## Implementation Order

- [ ] 1. Add `CScalar` typedef to types.hpp, `COMPUTE_SCALAR_DOUBLE` flag to Makefile
- [ ] 2. Abstract bp_update() SGEMV/SDOT dispatch (highest impact)
- [ ] 3. Abstract mode_n_contract() dispatch  
- [ ] 4. Abstract householder_qr() / cholesky_qr() dispatch
- [ ] 5. Add double workspace buffers if needed
- [ ] 6. Add Makefile target `main_pauli_sweep_f64`
- [ ] 7. Test: chi=70 δ=1e-3 both binaries identical
- [ ] 8. Test: chi=450 δ=1e-4 float64 binary correct through step 9
