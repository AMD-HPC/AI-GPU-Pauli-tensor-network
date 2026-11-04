# Gram Matrix Color Ordering Experiment — Notes (2026-11-04)

## Experiment design

**Goal**: Does the Trotter color ordering (H-first vs V-first) change which bond pairs
are confusable in the Gram matrix?

**Method** (corrected from earlier attempts):
- 3-scan seismic structure: each scan uses its OWN center subset, not the union
  - Scan 1: centers 1,7,25,43,49 (corners + center)
  - Scan 2: centers 9,13,37,41 (quadrants)
  - Scan 3: centers 4,22,28,46 (mid-edges)
- 3 baselines + 84×3 = 255 simulations per configuration
- Response vector per bond: 735 dimensions (3 scans × 5 steps × 49 sites)
- Perturbation: δJ = -0.20, J = 0.80 (our experimental coupling)
- f32, Householder QR (correct operating regime)
- BP converges at step 1 for each scan (small center set, early cycles)

**Two binaries**:
- `main_pauli_sweep_rect7x7`: H fires before V (colors 0,1 horizontal, 2,3 vertical)
- `main_pauli_sweep_rect7x7_swapped`: V fires before H (colors 0,1 vertical, 2,3 horizontal)
  Source: `include/rect_7x7_swapped.hpp` (H/V color assignments swapped)

## Reproduce

```bash
# Build swapped binary
make PHYS_DIM=4 main_pauli_sweep_rect7x7_swapped

# Run standard (H-first)
python3 scripts/compute_gram.py \
    --logdir logs/gram_standard \
    --binary ./main_pauli_sweep_rect7x7 \
    --out logs/gram_standard/gram.json

# Run swapped (V-first) — after standard finishes
python3 scripts/compute_gram.py \
    --logdir logs/gram_swapped \
    --binary ./main_pauli_sweep_rect7x7_swapped \
    --out logs/gram_swapped/gram.json

# Plot comparison
python3 scripts/plot_gram_color_compare.py \
    --standard logs/gram_standard/gram.json \
    --swapped  logs/gram_swapped/gram.json \
    --out /tmp/gram_color_compare.png
```

## Results

| Metric | Standard (H-first) | Swapped (V-first) |
|--------|-------------------|-------------------|
| Response dim | 735 (3×5×49) | 735 (3×5×49) |
| Median coherence | 0.0206 | 0.0206 |
| Max coherence | 0.6625 | 0.6622 |
| Detectability span | 10.4× | 10.4× |

**Top-10 standard (H-first)**:
Adjacent H-H pairs dominate — bonds 14,15 (sites 17-18-19), bonds 26,27 (sites 31-32-33)

**Top-10 swapped (V-first)**:
Adjacent V-V pairs dominate — bonds 58,65 (sites 17-24-31), bonds 60,67 (sites 19-26-33)

## Key findings

1. **Median coherence is identical (0.0206)** — overall identifiability unchanged by color order

2. **Max coherence nearly identical (0.6625 vs 0.6622)** — same worst-case difficulty

3. **Color ordering rotates which bond TYPE is confusable**:
   - H fires first → adjacent H-H pairs become hard (H wavefront reaches H neighbors before V can disambiguate)
   - V fires first → adjacent V-V pairs become hard (same mechanism, rotated 90°)
   - The first-firing color creates same-direction adjacency confusion

4. **No H-V pairs in top-10** — vertex-sharing H-V pairs are NOT the hardest case
   with our 3-scan seismic plan. This contradicts Muhammad's finding, which was
   an artifact of multi-center BP non-convergence + f64 Cholesky.

5. **Max coherence dropped from 0.93 → 0.66** compared to our earlier (wrong) G —
   the 13-center union with bp=100 artificially inflated coherence.

## Comparison with Muhammad's result

Muhammad's G (single center 1,7,25,43,49, f64 Cholesky, non-converged):
- Max coherence 0.87, top pairs all H-V vertex-sharing
- Result was artifacts of: multi-center BP non-convergence + Cholesky wrong physics

Our correct G (3-scan structure, f32 Householder, BP converged at step 1):
- Max coherence 0.66, top pairs are same-direction adjacent bonds
- Color ordering symmetry: H-first → H-H hard; V-first → V-V hard
- Structure is a symmetry of the problem, not a physical asymmetry

## Physical interpretation

The color that fires first propagates slightly ahead in sub-cycle time.
Adjacent same-direction bonds (e.g., H bond at col 2-3 and H bond at col 3-4)
share site 3 and both fire at the same sub-cycle time — their response vectors
are nearly identical because the wavefront pattern around site 3 is dominated
by the shared endpoint accumulating at the same sub-cycle step.

The color swap rotates this effect by 90° but doesn't change its magnitude.
The overall identifiability of the scan plan is a property of the center geometry,
not the color ordering.
