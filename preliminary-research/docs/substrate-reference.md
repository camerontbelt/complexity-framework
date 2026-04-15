# Substrate Reference — a plain-English guide

**Purpose of this document.** We've accumulated a dozen-plus simulations and it's
easy to lose track of what each one is and why we're running it. This is a
one-stop reference. For every substrate we've used, it answers:

- **What is it?** In plain English, what does the simulation actually do.
- **Why was it invented?** The original scientific motivation — often from physics,
  biology, or CS decades before complexity science existed.
- **Why do we care about it?** What the substrate teaches us specifically in the
  context of the C complexity framework.
- **Key parameter and critical point** (if any). The "knob" we sweep, and the
  value where the interesting behavior lives.
- **Profile type** (A/B/C from our taxonomy). How it responds to the framework.

---

## How to read this document

Substrates are grouped by the scientific tradition they come from, not by where
they live in our codebase. Each group has a short intro explaining what the
class of models is trying to capture. The goal is that you can jump in at any
section when you're confused about a specific simulation.

The **profile type** labels (A, B, C) refer to the C-profile taxonomy:
- **Type A** — peak sits right at the critical point, both sides alive. Classic
  symmetry-breaking transitions look like this.
- **Type B** — peak sits *above* the critical point, sub-critical side is dead.
  Absorbing-state transitions (things that can "die" and not come back) look like this.
- **Type C** — system naturally settles into its critical state; C peaks at the
  natural attractor and falls off as you perturb it. Self-Organized Criticality
  (SOC — a system that tunes itself to the critical point without external
  control).

See `c-profile-taxonomy.md` for the full story and the DP (Directed Percolation)
vs CP (Contact Process) caveat.

### Acronym glossary (used throughout)

| Abbrev | Expansion |
|--------|-----------|
| C | Complexity (our composite metric) |
| CA | Cellular Automaton / Automata |
| ECA | Elementary Cellular Automaton (Wolfram's 256 1-D rules) |
| GoL | Game of Life |
| DP | Directed Percolation |
| CP | Contact Process |
| SIR | Susceptible–Infected–Recovered (epidemic model) |
| SOC | Self-Organized Criticality |
| BTW | Bak–Tang–Wiesenfeld (the original sandpile model) |
| RBN | Random Boolean Network |
| NK | Kauffman's N-node K-input network shorthand |
| MI | Mutual Information |
| RG | Renormalization Group |
| FSS | Finite-Size Scaling |
| FWHM | Full Width at Half Maximum |
| AUC | Area Under the Curve |
| OP | Order Parameter |
| w_OP_s, w_OP_t | Spatial / Temporal Opacity weight in C |
| w_H, w_T, w_G | Entropy, temporal-compression, gzip weights in C |
| PD | Prisoner's Dilemma |
| GRU | Gated Recurrent Unit |
| GPT-2 | Generative Pre-trained Transformer version 2 |
| MNIST | Modified National Institute of Standards and Technology (digit dataset) |
| EEG | Electroencephalography |
| BKT | Berezinskii–Kosterlitz–Thouless (XY-model-style transition) |
| BC | Blume-Capel (spin-1 Ising with crystal-field parameter D) |
| D | Crystal-field / single-ion anisotropy parameter in Blume-Capel |
| D_t, T_t | Coordinates of a tricritical point (where 2nd-order meets 1st-order on a critical line) |
| q-ary C | Multi-state generalisation of C (no binarization, entropy normalised by log q) |
| T_c, p_c, λ_c, K_c, μ_c | Critical value of the control parameter (subscript names the parameter) |
| R0 | Basic Reproduction Number (epidemiology) |

---

# GROUP 1 — Statistical mechanics classics

These are the original models from physics. They were built to study how
matter organizes itself (why magnets work, how water boils, how spin-glasses
freeze). They're the *benchmark* for any complexity measure: if you can't
detect the critical point in a 2D Ising model, you're not measuring anything
physically meaningful.

## Ising model (2D)
**What it is.** A grid of tiny magnets ("spins"), each pointing up or down.
Each spin prefers to align with its neighbors, but temperature jiggles them.
At low temperature everything aligns (a magnet). At high temperature everything
is random. In between, at a single critical temperature, you get everything at
once — aligned patches of all sizes, fluctuating at all time-scales.

**Why it was invented.** Lenz and Ising (1920s) wanted the simplest possible
model of ferromagnetism. Onsager solved the 2D case exactly in 1944; the exact
critical temperature T_c = 2 / ln(1 + √2) ≈ 2.269 is one of the most-verified
numbers in physics.

**Why we care.** This is the *canonical* critical system. Every complexity
measure ever published has been tested against it. If C peaks at T_c, we're
measuring something real. If it doesn't, we have a problem.

**Key parameter.** Temperature T. Critical at **T_c = 2.269**.

**Profile type.** Type A. Sharp symmetric peak at T_c.

---

## Potts model (q-state generalization of Ising)
**What it is.** Same as Ising, but each spin can be in one of q states instead
of just 2. For q=2 it *is* Ising. For q=3 or more it can have a first-order
("jumpy") transition instead of the smooth second-order Ising one.

**Why it was invented.** Potts (1952) generalized Ising to model things like
grain growth in metals and foam structure.

**Why we care.** Tests whether C generalizes beyond binary models. A small
correction to historical claims: q=3 is actually *second-order*; the
transition becomes first-order starting at q=5 in 2D. We now have a clean
sweep across q ∈ {2, 3, 5, 10} using the q-ary C pipeline (no binarization),
which pins T_c to within one grid step at every q.

**Key parameter.** Temperature T. For q-state Potts on a 2D square lattice,
**T_c(q) = 1 / ln(1 + √q)**.
  - q=2: T_c ≈ 1.135  (Ising universality)
  - q=3: T_c ≈ 0.995  (second-order)
  - q=5: T_c ≈ 0.852  (first-order)
  - q=10: T_c ≈ 0.701 (strongly first-order)

**Profile type.** Type A for q ≤ 4; at q ≥ 5 the C(T) curve drops more
abruptly on one side of T_c (discontinuous order parameter) but still peaks
at T_c with q-ary C.

---

## Blume-Capel model
**What it is.** A "spin-1 Ising" — each cell can be in *three* states
instead of two: spin up (+1), spin down (−1), or a *vacant* middle state
(0). The rules: neighbouring spins like to agree (same Ising ferromagnetic
coupling), but there's also a "crystal field" cost *D* for every non-zero
spin. So *D* is a knob controlling how expensive it is to have any opinion
at all — large *D* punishes both +1 and −1 equally and pushes cells toward
the middle 0 state.

**Why it was invented.** Blume (1966) and Capel (1966) independently built
it to model how a magnetic material with three spin states (e.g. a uranium
compound) behaves under temperature and external pressure. The *D*
parameter plays the role of a chemical/structural field that can be tuned
experimentally.

**Why we care — the reason we picked it for the blind test.**
Blume-Capel has *two* control parameters (T and D) instead of one, and its
phase diagram contains a famous feature: a **line of critical points**
whose character changes halfway along it.
- For small *D*, as you lower T the system undergoes a smooth
  (second-order) ordering transition — like Ising.
- For large *D*, the same ordering becomes a *jumpy* (first-order)
  transition — like Potts q ≥ 5.
- The two regions meet at a **tricritical point** at
  (D_t, T_t) ≈ (1.99, 0.42) on the 2D square lattice.

This lets us test the detector and the q-ary C pipeline against a *whole
curve* of critical points in a single model, spanning both transition
orders. It is probably the cleanest single experiment one can run to
stress-test a "critical-point finder."

**Key parameters and known values** (2D, J=1, from Fytas et al. arXiv
2401.02720):
  - D = 0.0 : T_c ≈ 1.693  (Ising universality)
  - D = 1.0 : T_c ≈ 1.40
  - D = 1.5 : T_c ≈ 1.15
  - D = 1.9 : T_c ≈ 0.90
  - D = 1.99: T_c ≈ 0.42  (tricritical / first-order side)

**Natively q=3.** The three states {−1, 0, +1} map directly to
{0, 1, 2} for the q-ary C pipeline. No binarization choice needed.

**What we found (G=48, blind test).** Detector consensus hit within one
grid step for D ∈ {0, 1.0, 1.5, 1.9}. Failed at D=1.99 (T_c=0.85 vs true
0.42, error +0.43) — likely finite-size rounding of the first-order
transition on a small lattice. The cliff-ratio hypothesis (that
first-order transitions produce a sharper C(T) cliff than second-order)
was **falsified** by this test: cliff ratio went the opposite direction
(2.99 → 1.27 as D increased), meaning the Potts q-sweep pattern was a
4-point coincidence, not a general feature.

G=128 rerun is in progress to test the finite-size hypothesis.

---

## Kuramoto oscillators
**What it is.** A grid of pendulums, each ticking at its own natural frequency.
Each pendulum nudges its neighbors to sync up with it. At low coupling everyone
ticks independently. At high coupling everyone ticks together. In between there's
a sync transition.

**Why it was invented.** Kuramoto (1975) wanted a solvable model of how
populations of oscillators spontaneously synchronize — fireflies flashing
together, heart cells beating in unison, neurons firing coherently.

**Why we care.** It's continuous (not binary), it's a synchronization transition
rather than a symmetry-breaking one. Tests whether the C metric works on
non-binary, non-Ising-like systems.

**Key parameter.** Coupling strength K. Synchronization onset around K ≈ 2–4
on a 2D lattice (depends on geometry).

**Profile type.** Type A, but with a +17% offset. Not a clean bullseye.

---

# GROUP 2 — Cellular automata

Cellular automata are discrete systems where cells on a grid update based on
their neighbors following fixed rules. Wolfram (1980s) showed that even with
the simplest possible rules, you get four qualitative behaviors (his "classes"),
including Class 4 — complex, computationally universal dynamics. These are the
toy models of "complexity from simplicity."

## Elementary Cellular Automata (ECA, 256 rules)
**What it is.** A 1D line of cells, each 0 or 1. Each cell's next state depends
on itself and its two immediate neighbors. That's 8 possible local configurations
and 2 possible outputs, giving 2⁸ = 256 possible rules total.

**Why it was invented.** Wolfram (1983) cataloged all 256 rules and classified
them. Rule 30 produces a chaotic-looking pattern used in random number
generators. Rule 110 was proven Turing-complete — it can compute anything,
with just 3-cell updates.

**Why we care.** The gold standard for "simple rules, complex behavior."
The 256 rules span the full spectrum from trivial (all cells die) to complex
(Class 4) to chaotic (Class 3). If our framework can reliably identify Class 4
rules as most complex, it's doing its job.

**Key parameter.** Rule number (0–255), discrete. There's no continuous knob.

**Profile type.** No sweep, so not quite applicable — but Class 4 rules score
highest, Class 3 second, Classes 1 and 2 lowest. Consistent with Type A
(complexity lives at a specific "edge-of-chaos" point in rule space).

---

## Game of Life (and Life-like 2D CA)
**What it is.** A 2D grid of cells. A cell survives if it has 2 or 3 live
neighbors; a dead cell becomes alive if it has exactly 3 live neighbors. Simple
rules, endlessly intricate behavior: gliders, oscillators, glider guns,
computers.

**Why it was invented.** Conway (1970) designed it as a puzzle: find the
simplest rules that still produce indefinitely interesting behavior. Eventually
shown to be Turing-complete.

**Why we care.** 2D equivalent of ECA. We run Life-like CA families (where the
birth/survive rules are varied), similar to sweeping a parameter. Tests whether
C identifies Conway's rule (B3/S23) as special.

**Key parameter.** The birth/survive rule pair (B, S). We sweep this rule space
and check what C says.

**Profile type.** Type A. Conway's B3/S23 scores highest.

---

## k=3 totalistic CA
**What it is.** Like ECA but with a radius-2 neighborhood (5 cells instead of
3), and the rule only depends on the *count* of live neighbors, not their
pattern. Fewer rules (32 total), more manageable sweep.

**Why we care.** A cleaner version of the ECA test — a smaller, more
controllable rule space.

**Profile type.** Type A.

---

# GROUP 3 — Absorbing-state transitions

These models have a special "dead" state the system can fall into and never
escape. When you tune the parameter up slowly, activity eventually survives
indefinitely — that's the absorbing-to-active transition. It's the universality
class that covers things like epidemic thresholds (does the disease take off
or die out?), forest fires, and neural avalanches.

## Directed Percolation (DP)
**What it is.** A 2D grid. At each step, each active cell activates its
neighbors with probability p. Below p_c activity dies out; above p_c it spreads
forever. The prototypical absorbing-state transition.

**Why it was invented.** Broadbent and Hammersley (1957) introduced percolation
to model fluid flow through porous rock — does liquid reach the far side?
Directed percolation adds a time direction; it turns out to describe epidemic
spread, turbulence onset, catalytic reactions, and more.

**Why we care.** The DP universality class is to non-equilibrium physics what
Ising is to equilibrium physics. It's the textbook example of an absorbing-state
transition. Our v9 DP variant has **p_c ≈ 0.2873**.

**Profile type.** Type B — in our framework, peak at p = 0.375, **+30% above
p_c**. Sub-critical side is completely dead. See `c-profile-taxonomy.md` §9
for why this offset appears and why it turns out to be a microdynamics artifact,
not a universality-class signature.

---

## Contact Process
**What it is.** Essentially the same physics as DP but with explicit death.
At each step, active cells die with rate 1/(1+λ); each active cell infects each
neighbor with rate λ/(4(1+λ)). Same DP universality class. **λ_c ≈ 1.6489**.

**Why it was invented.** Harris (1974) formalized it as the cleanest
continuous-time absorbing-state model. It's the simplest "epidemic with
recovery" model you can write down.

**Why we care.** Same universality class as DP, different microdynamics. This
is our *test* of whether the C-profile is a pure universality-class signature.
Answer: no. Contact Process peaks *at* λ_c while DP peaks *above* p_c. The
discrepancy is due to the explicit death term in CP vs its absence in DP.

**Profile type.** Surprisingly Type A in our data (peak at λ_c, offset −3%).
The sub-critical side is dead (Type-B-like), but the peak position is Type A.
This is the key puzzle that drove our DP-vs-CP investigation.

---

## SIR epidemic model
**What it is.** Classical epidemiology. Each person is Susceptible, Infected,
or Recovered. Infected people infect susceptible neighbors with rate β;
infected people recover with rate μ. If the basic reproduction number R0 = β/μ
is above 1, the disease takes off; below 1, it fizzles out.

**Why it was invented.** Kermack and McKendrick (1927) — foundational model
of epidemiology. R0 is why we learned about COVID exponentials.

**Why we care.** Real-world grounded example of absorbing-state dynamics.
Critical at R0 = 1.

**Profile type.** Type B with a huge offset (peak at R0 ~ 3, +200% above R0_c).
Same family as DP.

---

## Forest Fire
**What it is.** A 2D grid. Empty cells grow trees at rate p_tree; trees
randomly ignite at rate f_lightning; fires spread to adjacent trees until they
burn out.

**Why it was invented.** Drossel and Schwabl (1992) introduced it as a simple
model of self-organized criticality in ecosystems.

**Why we care.** Two-parameter self-organized critical system. Related to both
absorbing-state and SOC physics — a useful bridge case.

**Profile type.** Mixed. Has elements of Type B (ignition threshold) and
Type C (SOC attractor).

---

# GROUP 4 — Self-Organized Criticality (SOC)

SOC systems don't need a knob. They automatically drive themselves to the
critical point. If you push them away, they relax back. Per Bak, Tang,
Wiesenfeld (1987), this might explain why critical-looking phenomena (avalanche
size distributions, earthquakes, solar flares, neural avalanches) are so common
in nature — the universe tunes itself.

## BTW Sandpile
**What it is.** Grains of sand drop on a 2D grid. When a cell has ≥4 grains,
it topples and distributes one grain to each of its 4 neighbors, possibly
triggering more topples (an avalanche). Avalanche sizes follow a power law.

**Why it was invented.** Bak, Tang, Wiesenfeld (1987) — the original SOC
model. Built specifically to show how a driven system can organize itself to
criticality without any external tuning.

**Why we care.** The textbook Type-C system. If C works on SOC it should peak
at the natural (dissipationless) state and decline as we add dissipation.

**Key parameter.** Dissipation rate ε. Natural SOC at **ε = 0**.

**Profile type.** Type C. Monotonic decline from ε=0. But the signal is weak
in our current framework (C ~ 0.002). Multi-scale may rescue it.

---

## Voter model
**What it is.** A 2D grid of voters, each with opinion 0 or 1. At each step,
each voter copies the opinion of a random neighbor. With no mutations, the
grid eventually becomes uniform (one opinion wins). Adding a small mutation
rate μ keeps both opinions alive.

**Why it was invented.** Clifford and Sudbury (1973) — originally a model of
competing species in ecology; later adopted by statistical physics and
sociology.

**Why we care.** Its "absorbing state" (consensus) is *not* a dead state —
it's reached through interesting coarsening dynamics where domain walls
annihilate. The complex phase is at μ=0, making it look like Type C rather
than Type B despite being an absorbing-state model.

**Key parameter.** Mutation rate μ. Natural absorbing dynamics at **μ = 0**.

**Profile type.** Type C (we initially mis-predicted B). Monotonic decline
from μ=0.

---

# GROUP 5 — Biology-inspired / network models

These models abstract away most of the biology but try to capture one key
structural feature — gene regulation, pattern formation, flocking behavior.

## Random Boolean Networks (Kauffman NK model)
**What it is.** A network of N nodes, each 0 or 1. Each node has K randomly
chosen inputs and a random Boolean function of those inputs. Designed to model
gene regulatory networks — nodes = genes, inputs = regulation edges.

**Why it was invented.** Kauffman (1969) proposed this as a model of cell
differentiation: stable "attractors" of the network correspond to cell types.
He argued biological networks operate at an **edge-of-chaos** point — K=2 —
where the system is neither frozen nor chaotic.

**Why we care.** Not a lattice model. No spatial structure. Tests whether C
works on abstract networks at all. Kauffman's K_c = 2 is a testable prediction.

**Key parameter.** Average connectivity K. Edge-of-chaos at **K_c = 2**.

**Profile type.** Type B in our data. Peak at K ~ 2.5, offset +25%.

---

## Gray-Scott reaction-diffusion
**What it is.** A 2D grid holding concentrations of two chemicals U and V
that react and diffuse. Depending on the parameters (feed rate f, kill rate k)
you get spots, stripes, spirals, labyrinths, traveling waves, or static chaos.

**Why it was invented.** Gray and Scott (1984) studied this as a simplified
model of chemical oscillators. It's the poster child for
reaction-diffusion pattern formation (think Turing patterns — how stripes and
spots form on animal skins).

**Why we care.** Continuous, pattern-forming system. The patterns live at a
specific *spatial scale* (spot size), which is why this substrate *originally
motivated the multi-scale C work*. At the single cell scale, all regimes look
similar. At the right coarsening level, complex patterns jump out.

**Key parameter.** (f, k) pair. Different regions give different behaviors.

**Profile type.** Not a transition per se — a map of behaviors. C identifies
the spot/spiral regions using multi-scale measurements.

---

# GROUP 6 — Agent-based social models

These models abstract humans or animals as simple agents following local rules.
They're not physics in the strict sense, but they exhibit phase-transition-like
behaviors and are good stress-tests for whether C works outside traditional
physics.

## Schelling segregation
**What it is.** A 2D grid of two colors of agents and some empty cells. Each
agent is happy if a fraction ≥ τ of its neighbors share its color; unhappy
agents move to empty cells. As τ rises, the grid spontaneously segregates.

**Why it was invented.** Schelling (1971) showed that mild individual
preferences (τ = 0.3) lead to strong macroscopic segregation. Classic example
of emergent social pattern from individual rules.

**Why we care.** Non-physics complex system. Shows a parameter-driven
structural phase transition — useful range-test for C.

**Key parameter.** Tolerance threshold τ.

**Profile type.** Similar to Type A, with a segregation "critical" threshold
around τ ~ 0.3–0.5.

---

## Boids (flocking)
**What it is.** Simulated particles (birds/fish) following three rules:
separation (don't collide), alignment (match neighbors' direction), cohesion
(move toward group center). Produces flocks, schools, swarms.

**Why it was invented.** Reynolds (1987) — the original flocking simulation
used in computer graphics and later adopted as a canonical emergent-behavior
model.

**Why we care.** Continuous-space, particle-based — very different from
lattice models. Tests whether C works after we discretize particle positions
onto a grid.

**Key parameter.** Alignment strength (and related weights).

**Profile type.** Shows peak complexity at intermediate alignment.

---

## Prisoner's Dilemma (spatial, co-evolutionary)
**What it is.** A 2D grid of agents each playing Prisoner's Dilemma against
their neighbors. Agents copy successful strategies; strategies co-evolve.
Specific variant: the "P6 co-evolutionary dynamics" in our codebase.

**Why it was invented.** Axelrod (1984) studied iterated PD computer tournaments;
Nowak & May (1992) introduced the spatial version. Core question: how can
cooperation evolve in a world of defectors?

**Why we care.** Game theory × spatial dynamics. Shows whether C can track
strategy complexity in an evolutionary system. The last substrate we added to
the paper.

**Profile type.** Varies with payoff matrix.

---

# GROUP 7 — "Real" data (not toy models)

These are our attempts to apply C beyond toy models to actual data from real
systems — neural networks, brain recordings, physical simulations.

## MNIST / GRU / GPT-2 (neural network hidden states)
**What they are.** We treat the hidden-state activations of trained neural
networks as spatiotemporal data. For image models: across pixels and layers.
For language models: across tokens and layers.

**Why we run them.** Shows whether C distinguishes trained networks from
random-init networks, and whether it distinguishes coherent input from noise.
Important for framing C as a "general complexity probe" rather than just a
physics tool.

**Key finding so far.** Signal exists at the single-scale level for some
conditions but doesn't generalize cleanly across temporal striding. Mixed
evidence; needs more work.

---

## EEG (synthetic + real)
**What it is.** Electroencephalography data — voltage recordings from
scalp electrodes. The brain at rest, during sleep, during tasks, during seizures.

**Why we ran it.** Real biological data, claimed in the literature to be
critical. Archived because the signals were noisy enough to be inconclusive.

---

## N-body simulation
**What it is.** Simulated particles attracting each other via gravity-like
forces. Tunable between different dynamical regimes.

**Why we run it.** Continuous-time physical simulation, not a lattice model.
Tests whether C captures structure in a genuinely dynamical system.

---

# GROUP 8 — Recently added or edge cases

## Frustrated Ising
**What it is.** Ising with competing (ferromagnetic and antiferromagnetic)
interactions. Creates spin-glass-like behavior with many metastable states.

**Why we care.** Harder version of Ising. Tests whether C can find structure
in frustrated, glassy systems.

---

# Quick reference table

| Substrate | Group | p_c known | Profile type | Status |
|-----------|-------|-----------|--------------|--------|
| Ising 2D | Stat mech | T_c = 2.269 | A | Gold standard, re-confirmed |
| Potts q=3 | Stat mech | T_c = 0.995 | A | Clean positive, recent add |
| Kuramoto | Stat mech | K ~ 3 | A (with offset) | Works, slightly off |
| ECA | CA | Class 4 rules | A-ish | Original validation |
| Life-like CA | CA | B3/S23 | A-ish | Works |
| k=3 CA | CA | — | A-ish | Works |
| DP | Absorbing | p_c = 0.2873 | **B** | Peak at +30%, now understood |
| Contact Process | Absorbing | λ_c = 1.6489 | **A** | Peaks AT λ_c, puzzle |
| SIR | Absorbing | R0 = 1 | **B** | Peak at +200% |
| Forest Fire | SOC/Abs | Two params | Mixed | Bridging case |
| BTW Sandpile | SOC | ε = 0 | C | Weak signal |
| Voter | Absorbing → SOC | μ = 0 | C | Mis-predicted, turned out Type C |
| RBN | Network | K = 2 | B | Peak at +25% |
| Gray-Scott | Reaction-diffusion | — | — | Motivated multi-scale |
| Schelling | Social | τ ~ 0.3–0.5 | A-ish | Works |
| Boids | Flocking | — | — | Works |
| Prisoner's Dilemma | Game theory | — | — | Last paper add |
| Frustrated Ising | Stat mech | — | — | Edge case |
| MNIST/GPT-2 | Real data | — | — | Mixed evidence |
| EEG | Real data | — | — | Archived |
| N-body | Physics | — | — | Works |

---

# Where each substrate lives in the code

- **Main framework** (`complexity_framework_v9.py`): ECA, k=3 CA, Life, N-body,
  Prisoner's Dilemma, Ising, SIR, DP, Forest Fire, Schelling, RBN, Sandpile.
- **Separate folders**:
  - `experiments/gray-scott/` — Gray-Scott and multiscale experiments
  - `experiments/neural-network/` — MNIST, GRU, GPT-2
  - `experiments/schelling-experiment/` — extended Schelling variants
  - `experiments/lattice-models/` — Boids, generic lattice, finite-size scaling
  - `experiments/ising/` — Ising at multiple grid sizes (finite-size scaling)
  - `experiments/c-profile-fingerprint/` — Contact Process, Voter, Kuramoto,
    Potts q-sweep (v1 binary, v2 majority-cluster, q-ary), SIR q-ary,
    `compute_C_qary.py` (multi-state C), `criticality_detector.py`
    (blind T_c estimator), the 2026-04 blind test batch.

---

# Two complementary C pipelines

**Binary `compute_C`** (the original v9 pipeline) takes any 2-state
spatiotemporal field (T × cells, values 0/1) and returns the full metric suite.
This is the workhorse — used for Ising, DP, CP, Voter, Sandpile, Conway, etc.

**q-ary `compute_C_qary`** (added 2026-04) takes a field with values in
`{0, 1, …, q-1}` directly — no binarization. Components:

| Weight | Binary | q-ary |
|--------|--------|-------|
| `w_H`    | Shannon entropy / log 2 | Shannon entropy / log q |
| `w_OP_s` | spatial MI on 0/1 neighbour pairs | spatial MI on q×q neighbour pairs |
| `w_OP_t` | temporal MI on 0/1 transitions   | temporal MI on q×q transitions |
| `w_T`    | flip-fraction, peaked at 0.5 | state-change-fraction, peaked at 0.5 |
| `w_G`    | unlike-neighbour fraction   | unlike-neighbour fraction (peaked) |

Spatial coarsening for q-ary uses **per-block mode** instead of block-mean-then-
threshold, preserving the categorical structure.

Which to use:
- Intrinsically multi-state systems (Potts, RBN with k>2 colours, SIR with
  S/I/R, categorical agent models): **use q-ary.**
- Intrinsically binary systems (Ising sign, DP occupied/empty, Voter
  consensus): either works; binary is battle-tested.

---

# Where to measure from: the criticality detector

`criticality_detector.py` takes a multi-scale sweep (list of
`{param, C_1, C_2, C_4, C_8}` rows) and estimates the critical point using
three indicators:

1. **Scale-collapse**: parameter where C values across scales agree best
   (weighted by C magnitude to avoid trivial agreement in dead regions).
2. **β-zero**: parameter minimising sum |β(s→2s)| across RG steps.
3. **Coarsest-peak**: parameter maximising C at the coarsest pool.

Consensus = mean of the three; confidence = inverse spread. Validated
to within one T-grid step on Ising (error 0.002), DP (0.004), Potts q-ary
(all q, error ≤ 0.057). Known failure mode: confident wrong answer when
sweep range doesn't bracket the true critical point (CP case).

---

# Changelog

- 2026-04-14 — Initial version of this reference doc. Written during the
  multi-scale diagnostic wait. Covers ~20 substrates accumulated so far.
- 2026-04-14 (late) — Added q-ary C pipeline and criticality detector
  sections. Reflects the Potts q-sweep → q-ary generalisation →
  blind-T_c-detection arc.
