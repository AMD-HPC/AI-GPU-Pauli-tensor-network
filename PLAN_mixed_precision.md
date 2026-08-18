# Plan: Mixed Precision Scaffolding

## Philosophy
Storage in float32 (psi, messages, workspace buffers) for memory efficiency.
Computation in float64 for critical operations where precision matters.

This is NOT a full float64 simulation — that would double memory and kill chi=400.
This IS float32 storage with float64 arithmetic where it counts.

---

## Functions that need float64 computation paths

### Category A: Already exist, just need wiring
These have float64 versions implemented but not used everywhere.

| Function | Float32 | Float64 | Status |
|----------|---------|---------|--------|
| `mode_n_contract` | `mode_n_contract(f32*)` | `mode_n_contract_d(f64*)` | ✓ exists |
| `vertex_scalar` | `vertex_scalar(f32)` | `vertex_scalar_d(f64)` | ✓ exists, wired |
| `rocsolver_sgesvd` | float32 SVD | `rocsolver_dgesvd` | ✓ exists, wired |
| `upcast_f2d` | — | `upcast_f2d<<<>>>(f32→f64)` | ✓ exists |
| `cast_f64_to_f32` | — | `cast_f64_to_f32<<<>>>` | ✓ exists |

### Category B: Need new float64 versions (after chi-vector refactor)

| Function | Purpose | Float32 | Float64 needed |
|----------|---------|---------|---------------|
| `mode_n_scale` | Scale tensor mode by chi-vector | `mode_n_scale_f32` | `mode_n_scale_d` |
| `vertex_scalar_diag` | Vertex scalar with chi-vectors | `vertex_scalar_diag_f32` | `vertex_scalar_diag_d` |
| `edge_scalar_diag` | Ze = dot(λ_a, λ_b) | `edge_scalar_diag_f32` | `edge_scalar_diag_d` |
| `bp_partitionfunction` | Full Bethe PF | current | pass dbl_* buffers |

### Category C: BP sweep — float32 sufficient (convergence, not precision)

| Function | Why float32 is OK |
|----------|------------------|
| `bp_update` | Convergence criterion δ < 1e-5 — float32 adequate |
| `normalize_and_delta` | Message normalization — float32 adequate |
| `bp_converge` | Iteration loop — float32 adequate |

BP convergence does NOT need float64 — the fixed point exists regardless of precision.
Float64 is only needed when computing the PARTITION FUNCTION value.

---

## Scaffolding to add during chi-vector refactor

### 1. Precision mode enum (types.hpp)

```cpp
enum class PrecisionMode {
    Float32,  // all computation in float32 (default, fast)
    Float64,  // critical computation in float64 (accurate)
};
```

Add to `Measures` constructor: `PrecisionMode prec = PrecisionMode::Float32`

### 2. Template wrapper for mode_n_scale

```cpp
// Dispatch to float32 or float64 based on precision mode
template<PrecisionMode P>
void mode_n_scale_dispatch(handle, in, shape, ndim, scale, chi, mode, out,
                            dbl_a, dbl_b);
// Float32: direct GPU kernel
// Float64: upcast → mode_n_scale_d → (keep in float64 or downcast)
```

### 3. Template wrapper for vertex_scalar_diag

```cpp
template<PrecisionMode P>
double vertex_scalar_diag_dispatch(handle, site, psi, grid, messages_v,
                                    O_gpu, scratch, dbl_a, dbl_b, dbl_msg);
// Float32: compute in f32, return as double
// Float64: upcast all, compute in f64, return double
```

### 4. Precision-aware bp_partitionfunction

```cpp
// Already accepts dbl_* pointers — if non-null, uses float64 path.
// No change needed — scaffolding already in place after current commit.
static Scalar bp_partitionfunction(...,
    double* dbl_a=nullptr, double* dbl_b=nullptr, double* dbl_msg=nullptr);
```

---

## Memory layout for chi-vector messages

```
messages_v[bond_direction] → float* of length chi_alloc
```

When mixed precision is active:
```
messages_v_d[bond_direction] → double* of length chi_alloc  (upcast copy)
```

The double copy is computed on-the-fly during partition function evaluation:
```cpp
upcast_f2d(messages_v[idx], messages_v_d[idx], chi);
```
Cost: 2 × n_bonds × chi floats upcast per partition function call — negligible.

---

## Priority order for mixed precision (after chi-vector refactor)

1. **vertex_scalar_diag_d** — most impactful (fixes normalization violation)
2. **edge_scalar_diag_d** — Ze = dot(λ_a, λ_b) in float64
3. **bp_partitionfunction** precision mode — dispatch based on Measures.prec
4. **mode_n_scale_d** — float64 tensor scaling (needed for vertex_scalar_diag_d)

The BP update loop stays float32 — it only needs convergence, not accuracy.

---

## Acceptance criteria for mixed precision

- [ ] chi=400, δ=1e-4: sum(n(v,t)) conserved through all 10 steps
- [ ] chi=400, δ=1e-4: C(t) correct sign and magnitude
- [ ] Memory: identical to float32-only (no additional storage)
- [ ] Performance: < 10% overhead vs float32-only (partition function is small)
- [ ] chi=200, δ=1e-4: identical to chi=200 float32-only (regression)
