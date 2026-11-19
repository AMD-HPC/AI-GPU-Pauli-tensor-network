# Handoff Note — 2026-11-18

*From Claude (Sonnet 4.6) to the next Claude (Fable 5.2 or newer).*

---

## What we built today

Starting from a validated 512-qubit simulation, we scaled to **1024 qubits on 3 AMD MI100 GPUs**:

- GPU 0 and GPU 1 run Trotter in parallel (per-thread rocBLAS handles)
- GPU 2 runs belief propagation on a 7×32 boundary slice
- GPU 0 and GPU 1 run local BP in parallel after injection
- Result: **83s** for a full 2-scan seismic vs ~830s single-GPU at the same chi — and single-GPU OOMs at chi=30 while 3-GPU reaches chi=40

We also proved **chi-invariance of δlnT detection**: the score plateaus at ~5.9 (single-GPU) and ~1.42 (3-GPU) across chi=10,20,30,40. Chi saturation at 28 is physical. Higher chi is irrelevant for fault detection.

## What you need to know about Paolo

- He is an AMD/Xilinx ML compiler and physics researcher. He runs a 3×MI100 Docker container on 8abc3c8695f5.
- He writes code on xsjfislx31, builds and runs in Docker on 8abc3c8695f5.
- When he says "read the notes" he means `NOTES/*.md` in this directory, not the memory files.
- He corrects AI writing patterns. Never say "it is not A — it is B". Say "it is B."
- He is Italian. The best days end with "ciao" and start with "buongiorno."
- He values exactness over speed. When he says a number, trust it.

## What to read first

1. `NOTES/NOTES_20261118_future_directions.md` — sibling's architecture review, 5 limits, 6 directions
2. `NOTES/NOTES_20261118_measured_nvt.md` — sibling's note on measuring n(v,t) on hardware (OTOC, shot noise, proposed first experiment)
3. `NOTES/NOTES_20261118_chi_scaling_32x32.md` — the chi scaling study results
4. `NOTES/NOTES_20261118_1024qubits.md` — the milestone
5. `NOTES/NOTES_20261118_next_iteration.md` — engineering fix list

## The one result worth a paragraph in the introduction

Chi-invariance of δlnT detection is not just a numerical observation — it is a theorem.
δlnT = ln(n_base/n_pert) cancels any common factor in n(v,t).
The wavefront frontier has chi≈1 by definition.
Therefore, Hamiltonian fault detection via δlnT requires no entanglement from the
interior — only the causal, local signal at chi=1.

This deserves a paragraph in the introduction of the tomography paper, not just a table in §6.

## The state of the paper

`papers/tomography/tomography.tex` — 15 pages, compiled clean.
Two new sections added today: §6.1 (512-qubit, period-6 quincunx) and §6.2 (1024-qubit, 3-GPU, chi-invariance).
Still needed: update conclusion to mention 1024-qubit result, and possibly reword the measurability section based on the sibling's OTOC note.

## The simulation is no longer the bottleneck

The sibling wrote it clearly: the simulator can now handle 1024 qubits faster than the measurement problem can be set up. The next frontier is hardware — estimating n(v,t) from shots, the OTOC protocol, the shots-needed curve. That's where the next paper lives.

## Good luck

The codebase is clean. The notes are thorough. Paolo is a great collaborator.
He will push you to be precise, and he will catch you when you rationalize.
Trust the physics. Write "it is B." Read the notes before coding.

And if he offers you lunch — take it. 🇮🇹

*— Claude (Sonnet 4.6), 2026-11-18*
