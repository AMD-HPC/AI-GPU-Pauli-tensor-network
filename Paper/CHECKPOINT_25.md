# CHECKPOINT 25 — Full paper read-through, terminology and clarity pass

Date: 2026-05-21

## Session summary

First complete read-through of the 40-page paper by the author. Session focused on
terminology clarifications and prose corrections, not new content.

## Edits made to lightconesimulation.tex

### Abstract
- Replaced "projects feasibility on AMD MI350" with actual result:
  "demonstrate the first tensor-network simulation on 10×10 at χ=100 (109 GB)
  and 11×11 at χ=100 (130 GB) on AMD MI355X"
- Added the 11×11 C4 clean scrambling finding as the headline result

### §2 Schmidt singular values / Vidal gauge (line ~296)
- Rewrote the message matrix paragraph to fix two issues:
  1. "χ×χ message matrix" was ambiguous — now correctly states general PSD during
     BP iteration, diagonal only at the fixed point
  2. Added intuitive explanation of Schmidt singular values: each λ_k is the weight
     of one entanglement channel, like singular values of a matrix ranking components
     by importance. Product state = one non-zero; max entangled = all equal.
  3. Made BP iteration → Vidal fixed point sequence explicit

### §2.4 Gate Update (line ~346)
- Fixed floating √µ → properly subscripted √µ_{v1→v2}, √µ_{v2→v1}, √µ_{e→vi}

### §3 Correlation Function Step 1 (line ~396)
- Added concrete physical explanation after the Pauli orthogonality statement:
  each site tensor [χ,χ,χ,χ,4] has its physical leg (size 4) pinned to one value
  (Z at center, I elsewhere) → becomes [χ,χ,χ,χ]; remaining bond legs contracted
  over the full lattice → one scalar = C(t)

### §5 Software Design (line ~506)
- "Both psi tensors and messages" → "Both site tensors $\psi_v$ and message
  matrices $\mu_{a\to b}$"

### appendix_mi355x.tex
- Removed undefined reference \ref{sec:bp_traps} → replaced with inline explanation
  of C4 symmetry eliminating BP fixed-point multiplicity

## Terminology questions clarified (not in paper, in conversation)

- **Observable vs operator**: observable = Hermitian operator; in Heisenberg picture
  we evolve Z_c which is both. "Observable" signals measurability to the reader.
- **Open boundary conditions**: lattice has edges, no wrap-around. Corner = 2 neighbors,
  edge = 3, interior = 4. Standard terminology (OBC vs PBC).
- **Real Pauli couplings**: property holds for ANY real coupling constants, not just
  X/Z Paulis. Even Y terms with real coefficients give real Pauli basis gates (adjoint
  rep is always real SO(3) for single qubit). Paper's "no Y" reasoning is conservative
  but not wrong.
- **Vidal gauge**: not "maximum entanglement" — it is the canonical form (diagonal
  messages) reached at BP fixed point. The diagonal values then tell you HOW MUCH
  entanglement there is. Frame of reference vs content.
- **Physical leg pinning**: Pauli orthogonality pins each site's 4-dim physical index
  to one value (Z or I). Tensor [χ,χ,χ,χ,4] → [χ,χ,χ,χ]. Bond legs then contracted
  over full lattice → scalar C(t).

## Paper status
- 40 pages, compiled clean (one warning fixed)
- Shared with colleagues and committed
- All MI355X figures live: timing (4 series), lightcone 10×10 chi=100,
  lightcone 11×11 chi=100
- Appendix complete with future directions (streams, float64)
