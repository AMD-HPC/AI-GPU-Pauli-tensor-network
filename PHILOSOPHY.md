# CppSim: Physics to Compute

*A design philosophy and scientific context document.*

---

## What CppSim is

CppSim is a GPU tensor network simulator for real-time quantum dynamics on 2D
lattices. It simulates the kicked Heisenberg (XXX) model in the Heisenberg
picture using Pauli transfer matrices (PTMs) and belief propagation (BP).
It runs on AMD GPUs (MI60, MI100, MI355X) in float32, achieves 9–10× speedup
over reference Julia implementations at saturated bond dimension, and has been
validated across multiple topologies and noise models.

But describing what it does misses the point. The more important question is
*why it was built the way it was*, and what it enables beyond raw performance.

---

## 1. Capabilities

CppSim simulates the kicked Heisenberg model on 2D lattices of arbitrary
topology — rectangular grids, heavy-hex (IBM quantum processor geometry),
and any graph that can be described as a colored bond structure.
Gate application, belief propagation, and workspace management are
topology-agnostic: adding a new lattice requires only a new grid class.
The simulator supports:

- Noiseless (ideal) evolution
- Noisy evolution with per-bond hardware PTMs from device characterization
- Synthetic depolarizing noise for controlled experiments
- Multi-product formula (MPF) error mitigation via noise amplification (γ-MPF)
- Pauli weight lightcone measurement: $n(v,t)$ per site per cycle
- Inner products $\langle\psi_i|\psi_j\rangle$ for full MPF coefficient computation

---

## 2. Computational assumptions

Two assumptions are made explicitly and deliberately.

**A. Pre-composed gates.** The hardware gate decomposition
$G_1 \cdot M_n \cdot G_2 \cdot M_n \cdot G_3 \cdot M_n \cdot G_4$
is pre-composed into a single 16×16 PTM per bond before simulation.
This eliminates intermediate steps and makes the simulation independent
of any specific gate ordering on the quantum processor.
A physical quantum computation does not necessarily decompose gates the
same way; tying the simulation to a specific sub-gate sequence would
introduce false precision. Speed is a secondary benefit.

**B. Messages frozen during Trotter, updated once per full cycle.**
Belief propagation messages are updated once per complete Trotter step,
after all color classes have been applied. Updating after each color class
would make the result order-dependent — the simulation would privilege
whichever color is processed first, introducing a non-physical asymmetry.
Freezing messages removes this ambiguity. It is also consistent with how
observables are recorded: as end-of-cycle snapshots, after the kick gate fires.

---

## 3. Engineering design

CppSim is optimized for float32 operations on modern AMD GPUs and their
linear algebra libraries (rocBLAS, rocsolver).

**Single pre-allocation.** The entire simulation state — site tensors $|\psi\rangle$,
bond messages, gate workspace, and inner product temporaries — is allocated once
at initialization in a custom layout designed for the access patterns of the
Trotter step and BP sweep. No dynamic allocation occurs during simulation.
This eliminates allocation overhead, fragmentation, and unpredictable latency.

**Multi-GPU design.** Communication between GPUs is minimal and planned:
GPUs exchange only scalar observables and MPF coefficients, not tensor state.
The parallelism schedule (GPU0: Trotter; GPU1: inner products/measurements)
is pre-defined, not discovered at runtime, eliminating synchronization overhead.

**Counterintuitive portability.** Simplicity and bare-metal GPU support make
the simulator predictable and portable across GPU generations. Higher-level
frameworks make dynamic decisions about layout and scheduling that may or may
not adapt gracefully to new hardware. CppSim makes those decisions explicitly
at the HIP/C++ level. When rocBLAS improves its SGEMM, the improvement flows
through directly. Moving to a new GPU generation means recompiling.

---

## 4. Physics to Compute

Most simulation tools follow the direction **Compute to Physics**: take an
existing framework and express the physics in whatever idioms it provides.
The physics gets shaped by the tool.

CppSim inverts this: **Physics to Compute**. The physical structure of the
problem — Pauli basis, PTM, QR gauge, Trotter coloring, Lieb-Robinson causality —
directly determines the computational structure.

- The memory layout follows the physics (site tensors indexed by bond dimension)
- The schedule follows the physics (color class ordering, kick at end of cycle)
- The precision follows the physics (real float32, because PTMs are real)
- The gauge follows the physics (QR keeps computation in the real domain,
  respecting the Pauli basis; the symmetric gauge would require sqrt of
  eigenvalues that can be negative, breaking the real domain)
- The multi-GPU split follows the physics (independent states for MPF)

**CppSim is a joint AI–Human contribution** built on this philosophy.
The human brings physical intuition and judgment: which gauge to choose,
what the wavemap means, when an assumption is an assumption.
The AI brings the ability to rapidly translate those choices into working
code and to hold full simulation context across sessions.
Neither alone would have produced CppSim in this form.

---

## 5. Simulation as a measurement instrument

CppSim is fast enough — 10× over reference implementations — that the
simulation parameters themselves become explorable in real time. $\chi$,
cutoff $\delta$, gauge choice, BP frequency: their effect on observables
is measurable, interpretable, and reproducible. This transforms the simulator
from a tool that produces answers into an instrument that explains them.

The central example is the **Pauli lightcone** and its frontier $\mathcal{F}(t)$:
the set of sites that first acquire significant operator weight at cycle $t$.
This is physics — the causal boundary of operator spreading under the Heisenberg
model — but it simultaneously serves as:

**A. Validation.** The frontier velocity $v_{\rm hop}$ is $\chi$-independent.
If it changes with $\chi$ or cutoff, something is wrong. The frontier is the
canary for simulation faithfulness.

**B. Simulation improvement.** Sites outside the lightcone have $\chi=1$ exactly
— no allocation needed. The arrival time $t_{\rm nl}(v) \approx d(v)/v_{\rm hop}$
gives the per-site entry schedule, enabling lightcone-aware memory allocation
that reduces memory requirements by 3–13× depending on topology and cycle count.

**C. Space planning.** The allocation schedule is computable before the expensive
run starts, from a cheap pre-flight run at small $\chi$.

**D. Approximation boundaries.** Belief propagation is exact on trees.
The first approximation error appears when the wavefront traverses a loop of
length equal to the graph girth $g$, at cycle $t_{\rm loop} = g/(2v_{\rm hop})$.
This is predictable from graph theory and $v_{\rm hop}$ — no chi sweep needed.

**E. Performance.** Lightcone-aware allocation reduces memory, enabling
larger $\chi$ within the same GPU budget.

**F. New methodologies.** The entropy-optimal MPF with Lieb-Robinson constraint —
fitting the Pauli weight field across noise levels and extrapolating toward the
noiseless reference — emerged directly from treating the wavefront as a
measurement instrument. It would not exist without the wavemap.

The unifying principle: **the cheapest, most exact part of the simulation
carries the richest information about the expensive part.**

---

## 6. Hardware

Results have been produced on:

- **AMD MI60**: tight memory, validates pre-allocation design and float32 choice
- **AMD MI100**: primary workhorse; production chi sweeps, MPF experiments,
  wavemap measurements; 10× speedup over Julia at saturated $\chi$
- **AMD MI355X**: latest generation; 3–5.6× speedup matches independently
  reported numbers from the L-MPF paper, providing cross-validation

The portability is real: the same C++/HIP code runs across all three
generations without modification. This is a direct consequence of the
"Physics to Compute" design — there is no framework layer to adapt.

---

## 7. Papers

| Paper | Topic | Location |
|-------|-------|----------|
| 2D Ising lightcone (arXiv:2606.30985) | First χ=100 TN simulation on 10×10, 11×11 | `Paper/` |
| Heavy-hex MPF (arXiv:2607.22827) | Heisenberg on heavy-hex, γ-MPF, inner products | `PaperHeavyHex/` |
| Pauli Lightcone (in preparation) | Wavemap, entropy-optimal MPF, two-regime result | `PaperFrontier/` |

---

## 8. What comes next

The control experiment (calibrated uniform synthetic noise on the heavy-hex
topology) has confirmed that the information-starved regime is caused by
per-bond PTM heterogeneity, not sparse lightcone coverage.
Under uniform noise, the same topology becomes information-rich and entropy
MPF recovers 49% more information than the best noisy sample.

Longer simulations (20+ cycles) to observe the wavemap after boundary
reflection.

3D topologies: the lightcone-aware allocation saves >10× in 3D, making
previously infeasible simulations accessible.

Hardware measurement: n(v,t) is measurable on a quantum processor via
randomized benchmarking, which would allow the wavemap to characterize
hardware noise directly from Hamiltonian dynamics — closing the loop between
classical simulation and quantum experiment.

**V2: The reflected lightcone.**
When the wavefront reaches the lattice boundary and reflects, a second frontier
forms moving inward. Two fronts coexist: the original outgoing wave and the
reflected inward wave, with noise acting on both independently.
This has not been investigated. The reflected wave will require:
- Larger chi: the inward wave interferes with the bulk interior where chi has
  already saturated, requiring higher bond dimension to represent the superposition
- More BP sweeps: the loop corrections become more important as the reflected wave
  re-traverses the same bonds from the opposite direction
- A new wavemap definition: the frontier is no longer simply the maximum-distance
  newly-reached sites; there are now two fronts at different distances

The information loss observable becomes richer: does the bulk entropy loss grow
linearly in time after reflection, or does the returning wave partially cancel
the noise-induced decoherence in the interior?
This is the natural V2 of the lightcone paper — same tools, new physics.

**Noise classification via machine learning — the next paper.**
CppSim can generate a labeled training dataset at scale:
sweep over a library of noise models (pure damping, rotation, heterogeneous
per-bond, correlated errors, mixed), compute the wavemap (l_gamma(v),
L_gamma(v)) for each, and label each simulation with the noise type and
per-bond parameters.
A classifier trained on this dataset — a graph neural network operating
on the lattice graph is the natural architecture — learns the map from
wavemap signatures to noise types.
Deployed on hardware: measure n(v,t) via randomized benchmarking, compute
the wavemap, run the classifier.
Noise type identified without any further simulation or analytical inversion.

CppSim becomes a noise characterization laboratory — generating ground truth
that no experiment alone can provide.
This is the path from simulation to tool to instrument to diagnostic:
the wavemap observables defined in the frontier paper are the features;
the classifier is the model; the QPU measurement is the deployment.
