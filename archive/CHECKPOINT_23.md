# CHECKPOINT 23 — Figures reverted, MI355X placeholder set, job 2175 running

Date: 2026-05-19

## Summary
Figure strategy corrected: MI355X removed from MI100/MI60 timing comparison figures
(professional optics — "flagship gives 1.2×" is the wrong story). New dedicated
figure script written for MI355X 10×10 chi=100 — the enabling result, not a
comparison. Appendix placeholder inserted. Job 2175 running with 10×10 chi=100
HH+Adap on GPU 6/7. Results expected tomorrow morning.

---

## What was done this session

### make_timing_figure.py — MI355X series added then reverted
- Initially added MI355X HH+Adap (green #1a9850) with saturated-layer averaging
- Reverted after user flagged optics: MI355X vs MI100/MI60 at small chi looks bad
  (6× bandwidth → 1.2× speedup, L2 cache regime explanation doesn't land visually)
- Script is back to MI100+MI60 only — re-run to regenerate clean fig_timing_chi.png

### make_breakdown_figure.py — same, reverted
- MI355X series added then removed for same reason
- Back to 4 series (MI100 HH/Adap, MI60 HH/Adap)

### Paper/make_mi355x_figure.py (NEW)
MI355X-only figure for 10×10 chi=100.
- Two panels: Trotter (ms) and BP (ms) vs layer
- Two series: HH (solid) and Adap (dashed), both green
- Saturation shading + annotation "saturation (χ=100)"
- Title: "not feasible on MI100/MI60" — that IS the story
- Log paths: logs/mi355x_20260519/10x10_chi100_n100.log (HH)
             logs/mi355x_20260519/10x10_chi100_n100_adap.log (Adap)
- Output: Paper/fig_mi355x_10x10_chi100.png

Run: `python Paper/make_mi355x_figure.py .`

### Paper/appendix_mi355x.tex — placeholder figure inserted
Replaced the two comparison figures with a single boxed placeholder:
  \label{fig:mi355x_10x10_chi100}
Caption already written (109 GB, not feasible on MI100/MI60, HH vs Adap).
When job 2175 data lands, replace \fbox{...} with:
  \includegraphics[width=\linewidth]{fig_mi355x_10x10_chi100.png}

---

## The corrected narrative

The MI355X story is NOT "faster than MI100 at the same task."
The MI355X story IS "the only chip that can run 10×10 chi=100 at all."
  - Requires 109 GB VRAM → beyond MI100 (32 GB) and MI60 (32 GB)
  - MI355X (309 GB HBM3E) is the enabling hardware
  - The figure stands alone; no cross-chip comparison needed or appropriate

The MI100/MI60 timing+breakdown figures remain in the main paper performance
section where they belong (chi scaling, QR mode comparison, BP traps).

---

## Job 2175 status (as of checkpoint)
- Running on Rainier, 8-hour allocation
- GPU 6: 10×10 chi=100 n=100 HH --measure=all   (~5-6 h estimated)
- GPU 7: 10×10 chi=100 n=100 Adap --measure=all
- GPU 3: 11×11 chi=100 n=70 HH+Adap --measure=all
- Results expected tomorrow morning

## Tomorrow morning: open the christmas gift
1. `ls -lrt logs/mi355x_20260519/` — check which logs landed
2. `python Paper/make_mi355x_figure.py .` — generate fig_mi355x_10x10_chi100.png
3. Replace \fbox placeholder in appendix_mi355x.tex with \includegraphics
4. Run compute_dispersion.py on *_all.log files for S_frac and R_L1/R_L2
5. Update "Pending" paragraph in appendix spatial section with actual results
6. If 11×11 landed: add second MI355X figure or extend the existing one

## Files modified this session
- Paper/make_timing_figure.py       (MI355X added and reverted — back to original)
- Paper/make_breakdown_figure.py    (MI355X added and reverted — back to original)
- Paper/make_mi355x_figure.py       (NEW)
- Paper/appendix_mi355x.tex         (placeholder figure inserted)
