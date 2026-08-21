# Frontier MPF Experiment Plan
# Date: 2026-08-21

## Scientific Goals

1. **Frontier as primary observable**: show N(v,t) tracks the causal frontier
   independently of chi_max/δ; topology literally visible in the shape.
2. **Spatial MPF**: extrapolate N(v,t) matrix with MPF coefficients → noise-mitigated frontier.
3. **Cross-entropy loss**: measure noise distortion of frontier per cycle per topology.
4. **Collapse as detectable statistical shift**: N(v,t) distribution changes
   from smooth causal wavefront → loop-shell spikes at BFS dist 3,5 (heavy-hex)
   or 2,4 (rect_7x7). Self-validating: same observable flags simulation failure.
5. **Float32 vs Float64**: verify MPF coefficients identical at δ=1e-3
   (precision irrelevant in valid regime).

---

## Template per experiment

```
Step 0: Quick lightcone sweep (chi=70, 10 cycles, --state=NL)
         → extract v_hop for this (graph, ε) combination
Step 1: Production run (chi=200, N cycles, --v_hop=measured)
         → C(t), N(v,t) for NL + N1/N2/N3
```

v_hop depends on (graph, ε, J) — NOT on chi_max or δ.
Must re-measure whenever ε changes.

---

## Experiment Table

### Group A: Heavy-hex 3×3, ε=0.10 (current PTMs)

| ID | chi | δ | cycles | states | float | purpose |
|----|-----|---|--------|--------|-------|---------|
| A1 | 70  | 1e-3 | 10 | NL | f32 | lightcone sweep → v_hop |
| A2 | 200 | 1e-3 | 10 | NL+N1+N2+N3 | f32 | main production run |
| A3 | 200 | 1e-3 | 20 | NL+N1+N2+N3 | f32 | long time: see recurrence |
| A4 | 200 | 1e-3 | 10 | NL+N1+N2+N3 | f64 | precision comparison |
| A5 | 380 | 1e-4 | 10 | NL+N1+N2+N3 | f32 | collapse analysis |
| A6 | 380 | 1e-3 | 10 | NL+N1+N2+N3 | f32 | chi=380 valid regime |

### Group B: Rect 7×7, ε=0.10 (existing PTMs)

| ID | chi | δ | cycles | states | float | purpose |
|----|-----|---|--------|--------|-------|---------|
| B1 | 70  | 1e-3 | 10 | NL | f32 | lightcone sweep → v_hop |
| B2 | 200 | 1e-3 | 10 | NL+N1+N2+N3 | f32 | main production run |
| B3 | 200 | 1e-3 | 20 | NL+N1+N2+N3 | f32 | long time: see recurrence |
| B4 | 200 | 1e-3 | 10 | NL+N1+N2+N3 | f64 | precision comparison |
| B5 | 380 | 1e-4 | 10 | NL+N1+N2+N3 | f32 | collapse analysis |

### Group C: Varying ε — DEFERRED after first batch results
Needs new PTMs from build_ptm.py. Priority after A/B analysis.

---

## Execution order (3 GPUs available)

### Phase 1: Lightcone sweeps (fast, ~5 min each)
```bash
GPU 0: A1 (heavy-hex v_hop)
GPU 1: B1 (rect_7x7 v_hop)
# Extract v_hop from both before proceeding
```

### Phase 2: Main production runs (parallel, ~30-60 min each)
```bash
GPU 0: A2 (heavy-hex chi=200 10 cycles f32)
GPU 1: B2 (rect_7x7 chi=200 10 cycles f32)
GPU 2: A4 (heavy-hex chi=200 10 cycles f64)
```

### Phase 3: Long runs + collapse (parallel, ~2-4 hours)
```bash
GPU 0: A3 (heavy-hex 20 cycles)
GPU 1: B3 (rect_7x7 20 cycles)
GPU 2: A5 (heavy-hex chi=380 δ=1e-4 collapse)
```

### Phase 4: Supplementary
```bash
GPU 0: A6 (heavy-hex chi=380 δ=1e-3 valid)
GPU 1: B4 (rect_7x7 f64)
GPU 2: B5 (rect_7x7 collapse)
```

---

## Analysis pipeline per run

1. `analyze_collapse.py` — C(t), ΣN(v,t), causal probe, BFS profiles
2. `plot_nvt_distribution.py` — per-step distribution, evolution, saturation zoom
3. `analyze_frontier_mpf.py` — spatial MPF: N_mpf[v,t] = Σ c_i N_ni[v,t]
4. `analyze_ip_mpf.py` — scalar C(t) MPF (A_ij, B_i from autocorrelation)
5. Cross-entropy between NL and Nγ distributions per cycle

---

## Key scientific comparisons

| Comparison | What it shows |
|---|---|
| A2 vs B2 | Topology dependence of frontier (6-cycle vs 4-cycle loops) |
| A2 vs A3 | Time evolution: recurrence at ~12 cycles, boundary reflection |
| A2 vs A4 | Float vs double: MPF coefficients should be identical |
| A2 vs A5 | Valid vs collapse regime: N(v,t) distribution shift detectable |
| A5 vs A6 | Collapse vs valid at same chi: δ is the control knob |
| A2 vs B2 | Different frontier shapes: topology literally visible |

---

## Commands template (fill v_hop after Phase 1)

```bash
# A1: heavy-hex lightcone sweep
ROCR_VISIBLE_DEVICES=0 ./main_pauli_sweep 70 10 1e-3 \
    --center=13 --state=NL > logs/A1_hh_lightcone.log 2>&1

# B1: rect_7x7 lightcone sweep  
ROCR_VISIBLE_DEVICES=1 ./main_pauli_sweep_rect7x7 70 10 1e-3 \
    --center=24 --state=NL > logs/B1_r7_lightcone.log 2>&1

# A2: heavy-hex production (fill V_HOP after A1)
ROCR_VISIBLE_DEVICES=0 ./main_pauli_sweep 200 10 1e-3 \
    --center=13 --v_hop=V_HOP > logs/A2_hh_200_10c.log 2>&1

# B2: rect_7x7 production (fill V_HOP after B1)
ROCR_VISIBLE_DEVICES=1 ./main_pauli_sweep_rect7x7 200 10 1e-3 \
    --center=24 --v_hop=V_HOP > logs/B2_r7_200_10c.log 2>&1

# A4: heavy-hex f64
ROCR_VISIBLE_DEVICES=2 ./main_pauli_sweep_f64 200 10 1e-3 \
    --center=13 --v_hop=V_HOP > logs/A4_hh_200_10c_f64.log 2>&1
```

---

## Notes

- rect_7x7 center: use site 24 (center of 7×7, 3 hops from all boundaries)
- v_hop for heavy-hex ε=0.10: expected ~0.253 (confirmed stable)
- v_hop for rect_7x7 ε=0.10: measure fresh (different topology/coordination)
- All runs: save full PAULI_NL/N1/N2/N3 rows for spatial analysis
- Log naming: {ID}_{graph}_{chi}_{cycles}c[_{f64}].log
