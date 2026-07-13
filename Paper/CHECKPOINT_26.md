# CHECKPOINT 26 — MI355X appendix numerical corrections and prose hardening

Date: 2026-05-21

## Session summary

Second pass over the MI355X appendix and experiments section. Focus on numerical
correctness, precision of claims, and removing any cross-chip comparisons that
could not be defended under review.

## Edits made

### appendix_mi355x.tex

#### Table 16 (tab:grid_scaling_mi355x) — complete redesign
- Old: absolute Trotter times (ms) for MI100 and MI355X side by side
- New: ratios normalized to each chip's own 3×3 HH baseline (1.00)
- Reason: MI100 and MI355X are different GPU generations; absolute cross-chip
  comparison is not meaningful and invites unfair criticism
- Absolute reference times removed from caption for the same reason
- "no_sat" redefined precisely: χ_min ≠ χ_max (not all bonds reached χ=50)
  — confirmed for MI100 8×8 HH: at layer 20 (last), chi_min=15, chi_max=50
  — root cause: 20 steps < 40 required by light cone formula (n*=d_max/2|J|δt=40)
  — VRAM was not the limiting factor for the run length chosen

#### Table 17 (tab:traps_mi355x) — data correction
- 4×4 Adap trap layer: corrected from "26" → "none"
  - n20 run (the grid-scaling suite run) has max sweeps=25 at layer 19, no Type-1
  - "26" came from the longer unnumbered run AND was misclassified (Type-2, not Type-1:
    sweeps=100 but delta=7.63e-4 >> ε_tol; true Type-1 in that run is layer 29)

#### Narrative paragraphs after Table 16
- "Architecture speedup" paragraph removed (claimed cross-chip absolute speedup)
- Replaced with "Grid scaling" paragraph:
  - Both chips show nearly identical HH scaling ratios → same physical problem
  - Super-linear scaling explained (O(χ³D) per bond)
  - "no_sat" explained as insufficient steps vs light cone, not algorithmic failure
- "Adaptive QR unlocked by HBM3E" rewritten:
  - All claims now within-chip only (MI355X Adap vs MI355X HH)
  - MI100 used as contrast (Adap tracks HH within 10%) without cross-chip speedup claim
  - Growing gap (0.78 → 0.65 with grid size) is the headline trend

### lightconesimulation.tex — Adaptive QR section (§sec:qr)

#### New paragraph: "Numerical precision and the Gram matrix accumulation error"
- Two failure modes of Cholesky-QR at high χ:
  1. GEMM accumulation error: m≈10⁶ terms at χ=100, float32 floor gives ~10⁻¹
     per Gram matrix entry — enough to destroy positive-definiteness
  2. κ(C) = κ(V)² squaring compounds the above
- Float64 fix: drops accumulation floor to ~10⁻¹⁰, better conditioned input
- Concrete ROCm path: rocblas_gemm_ex with F64 accumulators + DPOTRF
- Closing sentence: current float32 Adap already delivers 22-35% on MI355X —
  CholeskyQR is the right direction, fallback is an engineering detail

### experiments_section.tex

#### ε_cut = 10⁻¹¹ explanation added
- Paragraph after the parameter setup sentence explaining why 10⁻¹¹ is
  compatible with float32: SGESVD returns float32 scalars representable to
  ~10⁻³⁸; cutoff is relative (s_i ≥ ε_cut·s_0); all physical SVs above 10⁻⁷;
  χ_max is the active truncation constraint in every experiment

#### α (BP damping) forward reference fixed
- α used in diagnostic table and Se/logχ paragraph before being defined
- Added "BP damping factor" gloss at first use and §ref pointer in table

## Paper status
- All claims defensible under review
- No cross-chip absolute comparisons remain
- Compiled clean (warnings checked)
- Author ready to commit and submit
