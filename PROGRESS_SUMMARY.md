# CppSim Heavy-Hex MPF — Progress Summary (July 2026)

**What we built**: A fast C++/HIP GPU implementation of quantum error mitigation via Multi-Product Formula (MPF) on the heavy-hex lattice (68 sites, 76 bonds).

## Two MPF strategies implemented

- **Gamma-MPF**: fixed Trotter step ε=0.1, varying noise γ=1.0/2.0/3.0 → extrapolate to zero noise
- **Epsilon-MPF**: fixed noise, varying ε → extrapolates Trotter error (needs redesign to fix physical time reference)

## Key results

- Gamma-MPF is the correct strategy — 7-8% extrapolation error at 10 cycles, ~4s total
- Chi=140 is sufficient for 20 cycles at ε=0.1 (physics converged)
- Cutoff=1e-3 is sufficient (1e-2 too aggressive, 1e-4 gives no improvement)

## Poor man's MPF

Instead of computing quantum inner products ⟨ψᵢ|ψⱼ⟩ (which requires solving a hard gauge problem), we build the MPF coefficient matrix from Z(t) time correlations:

```
A[i,j] = Σ_t C_n(γᵢ,t) × C_n(γⱼ,t)
B[i]   = Σ_t C_n(γᵢ,t) × C_nl(t)
```

This is gauge-invariant, requires no inner products, and gives results 2-3 orders of magnitude better than simple polynomial extrapolation up to cycle 12. It is also a natural scaffold for the full quantum inner product implementation — same A/B structure, same solver.

## Speed

CppSim completes 20 cycles in seconds vs Julia's ~350s per cycle. Fast enough to explore the full parameter space (chi, cutoff, gamma, cycles) systematically.

## Next

Full quantum inner products (GPU double-layer BP) will be implemented to see if they improve over the poor man's approach at late cycles.
