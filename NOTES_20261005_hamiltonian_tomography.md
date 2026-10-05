# Hamiltonian Tomography via the Operator Lightcone Wavemap
*Notes 2026-10-05*

## What we built

A game-theoretic Hamiltonian tomography experiment using CppSim:
- **C_breaker**: perturbs one bond (coupling or PTM)
- **C_searcher**: identifies the broken bond from wavemap differences

## Key observable

$$\mathcal{L}(v, t) = -\ln\frac{n_{\rm pert}(v,t)}{n_{\rm base}(v,t)}$$

Signed cross-entropy per site per cycle:
- $\mathcal{L} > 0$ → red: wave blocked/reflected (upstream of broken bond)
- $\mathcal{L} < 0$ → blue: wave starved (downstream shadow)
- $\mathcal{L} = 0$ → white: no perturbation

## The identity PTM experiment (bond 12, sites 15↔16)

Setting one bond's PTM to identity blocks all propagation across it.
This is the maximally disruptive perturbation — a perfect switch.

**Cross-entropy field**: red-blue sign flip appears at cycle 2, persists.
Only sites 15 and 16 show signal — localization is exact.

## Ensemble detection

**Individual centers (6 runs):**

| Center | Side | Result |
|--------|------|--------|
| 10 | upstream | ✓ CORRECT: site 15 red (+0.87) |
| 30 | upstream | ✓ CORRECT: site 15 red (+0.37) |
| 25 | downstream | ✗ WRONG → bond 50 (9↔**16**) |
| 32 | downstream | ✗ WRONG → bond 57 (**16**↔23) |
| 6, 34 | too far | silent |

Every WRONG center still identifies one endpoint correctly.
The broken bond is BETWEEN the red and blue triggers.

**Ensemble (sum L fields):**
$$L_{\rm ens}(v) = \sum_k \mathcal{L}_k(v, t^k_{\rm first}(v))$$

Bond score: $S(b) = L_{\rm ens}(v_a) \times L_{\rm ens}(v_b)$ — most negative = broken bond.

Result: Bond 12 score=-0.164, next bond=-0.002 → **82× margin** → ✓ CORRECT.

## Why ensemble works

- Upstream centers: large positive L at upstream endpoint (red)
- Downstream centers: negative L at downstream endpoint (blue)
- Product at broken bond: maximally negative
- Wrong candidates cancel across centers
- Silent centers contribute zero (self-filtering)

## Scripts

```bash
# Run 6 individual centers
bash scripts/run_6centers.sh 12 0.20

# Ensemble search
python3 scripts/searcher.py \
    --baseline  "logs/tomo_6c/base_c10.log,logs/tomo_6c/base_c25.log,..." \
    --perturbed "logs/tomo_6c/pert_c10.log,logs/tomo_6c/pert_c25.log,..." \
    --graph data/graphs/rect_7x7.json \
    --centers 10,25,30,32,6,34 \
    --broken logs/tomo_6c/broken.json

# Animate cross-entropy field
python3 scripts/make_ce_gif.py \
    --baseline logs/tomo_6c/base_c10.log \
    --perturbed logs/tomo_6c/pert_c10.log \
    --graph data/graphs/rect_7x7.json \
    --broken logs/tomo_6c/broken.json \
    --centers 10 --out /tmp/ce_c10.gif
```

## Next experiments

1. **Sensitivity**: δJ=0.05, 0.10 — when does the switch become a dimmer?
2. **Realistic PTM**: physically correct coupling change (not identity)
3. **Multiple bonds**: can ensemble resolve two simultaneous broken bonds?
4. **Heavy-hex**: irregular graph, hardware PTM structure

## Paper direction

**"Hamiltonian Tomography via the Operator Lightcone Wavemap"**

Key result: ensemble cross-entropy at first baseline arrival identifies broken bond
with 82× margin over next candidate, using 6 center runs that individually give
only 2/6 correct answers. Self-intersecting seismic approach achieves same in
one simulation.
