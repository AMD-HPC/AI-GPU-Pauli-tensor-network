# Plan: Chi-Vector Message Refactor

## Goal
Replace chi×chi message matrices with chi-vectors (diagonal entries only).
Messages are always diagonal — the chi×chi format is wasteful and introduces
precision loss through unnecessary SGEMM operations.

## Test command (run after each step)
```bash
make main_pauli_sweep && \
ROCR_VISIBLE_DEVICES=0 ./main_pauli_sweep 70 10 1e-3 --center=13 | grep "^STEP"
```
Expected: C(t) ~±0.8 through all 10 steps, matching baseline.

## Baseline (record before starting)
```bash
ROCR_VISIBLE_DEVICES=0 ./main_pauli_sweep 70 10 1e-3 --center=13 \
    > /tmp/baseline_chi70.log
```

---

## Phase 1: Change message storage (heavyhex_physics.hpp + trotter.hpp)

### [ ] 1.1 — Add chi-vector message buffer to HeavyHexState
- `messages_v` : flat array of chi-vectors, one per bond direction
- Size: `2 × n_bonds × chi_max` floats (vs chi_max² currently)
- Keep `messages` (chi×chi) alongside for now — parallel buffers
- **Build + test**: should compile, no behavior change yet

### [ ] 1.2 — Gate output: write chi-vector alongside chi×chi
- In `gate.hpp` line 766 (Vidal normalization): also write `hS[i]/snorm` to `messages_v`
- `messages_v[bond.msg_from_a()][i] = hS[i]/snorm`
- **Build + test**: chi×chi messages unchanged, chi-vector populated

### [ ] 1.3 — Trotter snapshot: use chi-vector for frozen_msg
- `frozen_msg_v`: chi-vector snapshots at start of color class
- Replace `hipMemcpy` of chi×chi with chi-vector copy
- Gate still reads from `messages` (chi×chi) for env absorption
- **Build + test**: should be identical — frozen_msg_v not yet used

---

## Phase 2: Simplify BP update (measures.hpp)

### [ ] 2.1 — normalize_and_delta writes chi-vector
- `normalize_and_delta` already zeroes off-diagonal (correct)
- Add parallel write to `messages_v[idx][i] = new diagonal value`
- No change to chi×chi path yet
- **Build + test**: chi-vector populated by BP

### [ ] 2.2 — BP update: contract ψ with chi-vector instead of chi×chi
- New function `mode_n_scale`: multiply mode-n of tensor by a chi-vector
  `out[..., i, ...] = in[..., i, ...] × λ[i]`
  This replaces `mode_n_contract(ψ, M[chi×chi])` when M is diagonal
- Replace BP inner loop with `mode_n_scale` using `messages_v`
- **Build + test**: BP uses chi-vector, should give identical C(t)
- **Key test**: chi=400, δ=1e-4 — does sum(n(v,t)) stay conserved?

---

## Phase 3: Simplify vertex_scalar (measures.hpp)

### [ ] 3.1 — New vertex_scalar_diag using chi-vectors
- Replace `mode_n_contract(ψ, M[chi×chi])` with `mode_n_scale(ψ, λ[chi])`
- Uses `messages_v` directly
- Much cheaper: O(chi × tensor_volume) vs O(chi² × tensor_volume)
- **Build + test**: identical C(t) to baseline

### [ ] 3.2 — Wire vertex_scalar_diag into bp_partitionfunction
- Replace `vertex_scalar_d` with `vertex_scalar_diag`
- **Key test**: chi=400, δ=1e-4 — does C(t) stay correct through step 9?
- **Key test**: sum(n(v,t)) conserved?

---

## Phase 4: Simplify edge scalar (measures.hpp)

### [ ] 4.1 — Edge scalar Ze = dot(λ_a, λ_b)
- Replace D2H of chi×chi + chi² multiply with:
  D2H of chi-vector + chi dot product
- Much cheaper: O(chi) vs O(chi²)
- **Build + test**: identical C(t), faster

---

## Phase 5: Remove chi×chi message buffers

### [ ] 5.1 — Remove chi×chi `messages` from HeavyHexState
- Only after all consumers verified to use `messages_v`
- Check all call sites: gate.hpp, measures.hpp, trotter.hpp
- **Build + test**: full regression at chi=70, 200, 400

### [ ] 5.2 — Update memory estimates
- `lightcone_memory_mb`: message contribution shrinks 400× at chi=400
- `alloc_trotter_buffers`: use chi-vector sizes
- **Build + test**: verify memory estimates match actual allocations

---

## Phase 6: Cleanup and commit

### [ ] 6.1 — Remove vertex_scalar_d (superseded by vertex_scalar_diag)
### [ ] 6.2 — Update NOTES and PHILOSOPHY
### [ ] 6.3 — Full regression: chi=70/200/400, δ=1e-3/1e-4
### [ ] 6.4 — Commit and push

---

## Acceptance criteria

At the end of Phase 5:
- [ ] chi=70, δ=1e-3: C(t) matches baseline exactly
- [ ] chi=200, δ=1e-4: C(t) stable through 10 steps, sum(n(v,t)) conserved
- [ ] chi=400, δ=1e-4: C(t) correct sign and magnitude through steps 9-10
- [ ] Message memory: 400× smaller at chi=400
- [ ] BP update: no chi×chi SGEMM

---

## New function needed: mode_n_scale

```cpp
// Scale mode `mode` of tensor `in` by chi-vector `scale`:
// out[i0,...,i_{mode-1}, k, i_{mode+1},...] =
//   in[i0,...,i_{mode-1}, k, i_{mode+1},...] × scale[k]
// O(chi × tensor_volume) — replaces O(chi² × tensor_volume) mode_n_contract
static void mode_n_scale(
    const Scalar* in, const int* shape, int ndim,
    const Scalar* scale, int chi, int mode,
    Scalar* out);
```

This is a GPU kernel, not a GEMM — element-wise multiply along one axis.
Simpler and faster than mode_n_contract for diagonal messages.
