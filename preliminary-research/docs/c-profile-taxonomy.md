# C-Profile Taxonomy: Fingerprinting Complexity Types

## Abbreviations used in this document

C — Complexity (our metric).
CA — Cellular Automaton. ECA — Elementary Cellular Automaton (Wolfram's 256 1-D
rules). GoL — Game of Life.
DP — Directed Percolation. CP — Contact Process. SIR — Susceptible / Infected /
Recovered epidemic model.
SOC — Self-Organized Criticality. BTW — Bak–Tang–Wiesenfeld (the original SOC
sandpile model).
RBN — Random Boolean Network (Kauffman's gene-regulation abstraction).
RG — Renormalization Group. FSS — Finite-Size Scaling. AUC — Area Under the Curve.
OP — Order Parameter. MI — Mutual Information. FWHM — Full Width at Half Maximum.
w_H, w_OP_s, w_OP_t, w_T, w_G — framework weights for Entropy, Spatial Opacity,
Temporal Opacity, Temporal Compression, and Gzip respectively.
T_c, p_c, λ_c, K_c, μ_c — critical value of the control parameter (subscript
names the parameter).
R0 — Basic Reproduction Number (epidemiology).

## 1. Origin

While testing the v9 framework across 7 substrates with the new bit-packed gzip
gate, we found that C peaks at the known critical point for some systems
(Ising, ECA, GoL) but peaks well above it for others (DP, SIR, RBN). Rather than
treating this as a failure, detailed diagnostics revealed that the **shape of
C(parameter)** — the "C profile" — systematically differs between classes of
phase transitions. The profile shape is itself a fingerprint that classifies
what *type* of complexity a system exhibits.


## 2. The Three Profile Types

### Type A — Symmetry-Breaking Complexity
**Signature**: Sharp, roughly symmetric peak centered at or very near the known
critical point. Non-zero C on both sides.

**Physics**: Both phases (ordered and disordered) are *dense* — every cell is in
some state, just differently organised. At the critical point, the system has
maximal structured heterogeneity: long-range correlations, non-trivial spatial
gradients, temporal fluctuations at all scales.

**Why the framework peaks here**: All metric gates fire simultaneously.
- Entropy (w_H): high — neither uniform nor trivially patterned
- Spatial opacity (w_OP_s): high — coexisting ordered/disordered domains
- Temporal opacity (w_OP_t): high — domains fluctuate, merge, split
- tcomp: moderate — near the triple-Gaussian peaks
- gzip: intermediate — compressible but not trivially so

**Observed examples**:
| Substrate | p_c (known) | p_peak (C) | Offset |
|-----------|-------------|------------|--------|
| Ising 2D  | T_c = 2.269 | T ~ 2.22   | -2.2%  |
| ECA       | Class 4     | Rank [1,2,3,4] | Exact |
| Life-like | Class 4     | Rank [1,2,3,4] | Exact |


**Ising wide sweep (T=1.5 to 3.5, T_c=2.269)**:
```
T:  1.5   1.8   2.0   2.1   2.2   2.3   2.4   2.7   3.0   3.5
C:  .000  .000  .000  .530  .798  .091  .000  .000  .000  .002
M:  .986  .956  .908  .584  .164  .066  .033  .011  .008  .005
     ordered ←──────── T_c ────────→ disordered
```
Peak at T=2.2, offset -3% from T_c. Sharp, symmetric. Both sides dense but
unstructured (ordered=all aligned, disordered=all random). Peak where
structured domains coexist.


### Type B — Emergence Complexity
**Signature**: Strongly right-skewed peak. Sub-critical phase is identically
zero (or near-zero). Peak occurs 25-260% above the onset critical point. Rapid
decline into the fully-active/chaotic phase.

**Physics**: One phase is an *absorbing state* — the system is dead/frozen/empty.
At the critical point, activity is sparse, random, and unstructured. Structured
spatiotemporal patterns only appear once enough active material exists to
self-organise. The framework measures where structures are *richest*, not where
activity *begins*.

**Why the framework peaks above p_c**: Opacity is the bottleneck.
- At p_c: density ~ 0.001-0.02. Active sites are rare and randomly scattered.
  Spatial opacity = 0 (no gradients in a near-empty field).
  Temporal opacity = 0 (no autocorrelation in random flickers).
- Above p_c: density rises. Active sites form clusters, fronts, waves.
  Opacity gates open. But push too far and the system becomes uniformly active
  — gzip hits 1.0 (incompressible), killing C.

**Diagnostic proof** (DP at p_c = 0.287):
```
p=0.285:  wH=1.62  wOPs=0.000  wOPt=0.000  wT=0.64  wG=0.97  → C=0.000
p=0.290:  wH=1.01  wOPs=0.000  wOPt=0.000  wT=0.61  wG=1.00  → C=0.000
p=0.310:  wH=1.68  wOPs=0.319  wOPt=0.000  wT=0.86  wG=0.49  → C=0.226
```
Entropy and gzip are fine at p_c. Opacity is zero — the sparse field has no
spatial or temporal structure to measure.

**Observed examples**:
| Substrate | p_c (known) | p_peak (C) | Offset |
|-----------|-------------|------------|--------|
| DP        | p_c = 0.287 | p ~ 0.310  | +8%*   |
| SIR       | R0_c = 1.0  | R0 ~ 3.0   | +200%  |
| RBN       | K_c = 2.0   | K ~ 2.5    | +25%   |

*Fine sweep with 256x256 grid. Coarse sweep showed +30%.


### Type C — Self-Organised Complexity
**Signature**: No parameter to tune — the system naturally evolves to its
critical state. C is highest at the natural attractor (SOC) and declines
monotonically as you push the system away (e.g., by adding dissipation).

**Physics**: The system has an internal feedback loop that drives it to
criticality without external tuning. Perturbations (dissipation, boundary
effects) weaken this feedback and reduce structured complexity.

**Observed examples**:
| Substrate   | Natural state | Perturbation     | Result              |
|-------------|--------------|------------------|---------------------|
| BTW Sandpile| eps=0 (SOC)  | Dissipation eps  | Monotonic decline   |


## 3. Quantitative Profile Metrics

We can characterise any C profile with four shape metrics:

| Metric | Definition | Type A | Type B | Type C |
|--------|-----------|--------|--------|--------|
| **Peak offset** | (p_peak - p_c) / p_c | < 10% | > 20% | 0 (at natural state) |
| **Sub-critical death** | Fraction of sub-critical points with C < 1% of peak | Low (< 0.5) | High (~1.0) | N/A |
| **Asymmetry** | (right_area - left_area) / total_area | ~ 0 (symmetric) | > 0 (right-heavy) | +1.0 (monotonic) |
| **Width at half-max** | Parameter range where C > C_peak/2 | Narrow | Narrow | Broad or N/A |

**Measured values from v9 experiments**:
| Substrate | Offset% | Dead subcrit | Asymmetry | Width | Predicted | Actual |
|-----------|---------|-------------|-----------|-------|-----------|--------|
| Ising     | -2.2%   | 0.00        | *         | 0.035 | A         | A      |
| DP        | +30.5%  | 1.00        | -0.46     | 0.025 | B         | B      |
| SIR       | +260%   | 0.50        | +0.29     | 0.050 | B         | B      |
| RBN       | +25%    | 0.00        | +0.18     | 0.750 | B         | B      |
| Sandpile  | 0%      | N/A         | +1.00     | 0.100 | C         | C      |

*Ising asymmetry from original narrow sweep (T=2.22-2.32) was unreliable.
Wide sweep (T=1.5-3.5) confirms Type A: symmetric peak at T=2.2, C=0.80.
Both sides drop to zero (ordered: M~0.99, disordered: M~0.01).
Updated metrics: offset=-3.0%, asymmetry~-0.3 (slightly left-heavy),
width~0.15, dead_subcrit=0.0.


## 4. Hypotheses

### H1 (C-Profile Fingerprint Hypothesis)
**The shape of C(parameter) is predictable from the universality class of the
underlying phase transition:**
- Symmetry-breaking transitions (Ising universality, Wolfram Class 4)
  produce Type A profiles: symmetric peaks with offset < 10%.
- Absorbing-state transitions (directed percolation universality, epidemic
  threshold, frozen-to-chaotic) produce Type B profiles: right-skewed peaks
  with offset > 20% and dead sub-critical phase.
- Self-organised critical systems produce Type C profiles: monotonic decline
  from the natural SOC state.

Furthermore, the profile type can be **predicted without running the framework**
from one observable property of the system: whether the sub-critical phase is
an absorbing state (Type B), the system is self-organising (Type C), or
neither (Type A).

### H0 (Null Hypothesis)
C profile shapes are not systematically related to the type of phase
transition. Apparent clustering into Types A/B/C is an artifact of:
- Finite-size effects and insufficient resolution
- Implementation-specific choices (grid size, binary encoding, burnin)
- Overfitting to a small sample of 5-7 systems

The profile shape depends on simulation details, not on universal properties
of the transition.


## 5. Experimental Test Plan

### 5a. Predictions for Untested Systems (Blinded)

Before running each experiment, record the predicted profile type based on
known physics. Then run the framework and compare.

| System | Transition type | Predicted profile | Known p_c |
|--------|----------------|-------------------|-----------|
| **XY model** (2D) | Symmetry-breaking (BKT) | **Type A** | T_BKT ~ 0.893 |
| **Potts model** (q=3) | Symmetry-breaking (1st order) | **Type A** | T_c ~ 0.995 |
| **Contact process** | Absorbing-state (DP class) | **Type B** | lambda_c ~ 1.6489 |
| **Voter model** | Absorbing-state (different from DP) | **Type B** | N/A (coarsening) |
| **Kuramoto oscillators** | Synchronisation transition | **Type A?** | K_c = 2/pi |
| **Earthquake model** (OFC) | SOC | **Type C** | Natural state |
| **Neural branching** | Absorbing-state / SOC boundary | **Type B or C?** | branching ratio = 1 |

The most informative tests are systems where the prediction is uncertain
(Kuramoto, neural branching) — these can differentiate between competing
explanations.

### 5b. Controls

1. **Same universality class, different system**: The contact process and DP
   should produce the same profile type (both Type B). If they don't, the
   fingerprint depends on implementation, not universality.

2. **Same system, different encoding**: Run DP with two binary encodings
   (active/inactive vs. recently-changed/unchanged). If the profile type
   changes, the taxonomy is encoding-dependent, not physics-dependent.

3. **Same system, different size**: Run Ising at L=16, 32, 64, 128. The
   offset should shrink toward 0 as L grows (finite-size scaling). For DP,
   the offset should NOT shrink — confirming it's structural, not finite-size.

### 5c. Quantitative Discriminant

Train a simple classifier (e.g., logistic regression or decision boundary)
on the four profile metrics (offset, dead_subcrit, asymmetry, width) from
the 5 existing substrates. Test on new substrates from 5a.

Decision boundaries (provisional, from current data):
- If dead_subcrit > 0.5 AND offset > 15%: **Type B**
- If peak at natural state AND asymmetry > 0.8: **Type C**
- Otherwise: **Type A**


## 6. Implications

### 6a. For the Framework
The framework doesn't just measure "how complex" a system is — it measures
**what kind of complexity** the system exhibits. The C value tells you the
magnitude; the C profile tells you the class. This is a stronger result than
a universal critical-point detector.

### 6b. For Complexity Science
If confirmed across diverse systems, this taxonomy connects information-
theoretic complexity measures to the physics of phase transitions in a new
way. The fact that opacity (spatial and temporal structure) is the bottleneck
for Type B systems — not entropy or compressibility — may provide insight
into why absorbing-state transitions are fundamentally different from
symmetry-breaking transitions at the information-theoretic level.

### 6c. For the Paper
The framework paper can present:
1. Type A results (Ising, ECA, Life) as the primary validation
2. Type B results (DP, SIR, RBN) not as failures but as a *discovery* —
   the profile shape reveals the transition type
3. Type C results (sandpile) as a third category
4. The taxonomy itself as a novel contribution

This reframes "C doesn't peak at p_c for DP" from a negative result into
a positive one: "the C profile correctly classifies DP as an absorbing-state
transition."


## 7. Open Questions

1. **Is the Type A/B boundary sharp?** Are there systems that are intermediate?
   The Kuramoto model (continuous oscillators, synchronisation transition)
   might be one — it's not absorbing-state but it's also not a standard
   Ising-like symmetry-breaking.

2. **Does the offset in Type B have a universal value within a universality
   class?** E.g., do all DP-class systems peak at the same offset ratio?

3. **Can we predict the offset quantitatively?** For Type B, the offset
   depends on where density is high enough for opacity to register. Is there
   a density threshold (empirical or derivable) that predicts where C peaks?

4. **Is Type C actually distinct from Type A?** The sandpile has very weak
   signal (C ~ 0.001). With a better binary encoding, might it look more
   like Type A with a peak at some parameter value?

5. **What about Type D?** Are there complex systems that don't fit any of
   these three categories? Reaction-diffusion systems (Turing patterns)?
   Quasicrystals? Language?


---

## 8. Blind Test Results (fingerprint_test.py, 2026-04)

We ran 4 untested systems with predictions recorded BEFORE any simulation.
Raw scorecard: **2 / 4 matches**. Details:

| Substrate       | Predicted | Classifier output | Peak location            | Notes |
|-----------------|-----------|-------------------|--------------------------|-------|
| Potts q=3       | A         | A                 | T=1.00, offset +0.5%     | Bullseye at T_c=0.9950 |
| Kuramoto        | A         | A                 | K=3.5, offset +16.7%     | Match (classified A, but offset is borderline) |
| Contact Process | **B**     | **A**             | λ=1.60, offset −3.0%     | Peak AT λ_c, but dead_subcrit=0.58 (Type-B-like) |
| Voter model     | **B**     | **A**             | μ=0.001, offset −90%     | Should have been Type **C** — we mis-predicted |

Two substantive mismatches:

1. **Contact Process is anomalous.** Same universality class as DP (directed
   percolation) yet peaks AT λ_c, not above it. The framework classified it as
   Type A, but its sub-critical side is dead (0.58 dead fraction) which is a
   Type B feature. It lives *between* clusters in the shape-metric feature space.

2. **Voter model is Type C, not Type B.** Its absorbing state IS the complex
   coarsening state — maximal C occurs at μ=0 where the system is "dying"
   coherently. Our prediction framework treated it like DP/SIR, but voter
   coarsening is more like SOC: the natural attractor is already complex.

## 9. Revisiting the Taxonomy — the DP/CP Investigation

The Contact Process mismatch is the most informative result we have, because
DP and CP are **in the same universality class** (textbook directed-percolation).
If the C-profile were a universality-class fingerprint, they would look alike.
They don't.

### 9a. The empirical gap

| | DP (v9) | Contact Process |
|---|---|---|
| p_c | 0.2873 | 1.6489 |
| Peak location | p = 0.375 | λ = 1.60 |
| Offset | **+30.5%** | **−3.0%** |
| C_peak | 0.121 | 1.283 |

### 9b. Ruled out: initial conditions

Hypothesis: DP's 2% initial density kills sub-critical runs before the
measurement window, displacing the peak. Tested with init densities
{0.02, 0.10, 0.30, 0.50} — the peak stayed at p=0.375 in **every case**.
BURNIN=50 erases initial conditions. This is not the cause.

### 9c. The real cause: w_OP_t is dead for DP, alive for CP

Weight decomposition across the DP sweep:

```
      p       C     den      wH    wOPs    wOPt      wT      wG
  0.300   0.000   0.236   1.000   0.000   0.000   0.705   0.979    ← just above p_c
  0.325   0.006   0.469   0.440   0.475   0.000   0.615   0.040
  0.350   0.098   0.603   1.438   0.375   0.000   0.746   0.245
  0.375   0.121   0.695   1.699   0.094   0.000   0.997   0.760    ← peak
  0.400   0.001   0.763   1.783   0.001   0.000   0.786   0.942
```

**w_OP_t (temporal opacity) is identically zero across the entire DP sweep.**
The Contact Process hits w_OP_t = 0.97 at λ_c — it's the dominant term.

Because `C = w_H × (w_OP_s + w_OP_t) × w_T × w_G`, if w_OP_t ≡ 0 the only
path to non-zero C is via w_OP_s, and w_OP_s requires spatial heterogeneity
that only appears once activity density is high. That forces the peak to
~p=0.375 regardless of where p_c actually sits.

### 9d. Why w_OP_t differs: microdynamics, not universality

- **Synchronous DP**: `P(active | n active neighbors) = 1 − (1−p)^n`.
  No explicit death. A cell with live neighbors stays lit with near-certainty.
  Per-cell time-series becomes nearly deterministic → no temporal mutual
  information → w_OP_t ≈ 0.
- **Contact Process**: two competing processes per step. Active cells die
  with rate 1/(1+λ) ≈ 0.38 at λ_c. Every cell churns on and off even deep
  in the active phase → w_OP_t ≈ 0.97.

Same universality class, different *microdynamic flavor*. The stochastic
death term in CP is what keeps per-cell time-series informative; pure
synchronous DP lacks it.

### 9e. Revised interpretation of the fingerprint

The C-profile is **not** a clean universality-class signature. It is a
response to the joint structure of:

1. **Transition topology** — symmetry-breaking vs absorbing-state vs SOC
2. **Microdynamic flavor** — does the update rule provide per-cell temporal
   churn? (Stochastic birth/death, yes. Pure deterministic activation-on-input,
   no.)

Two systems with the same transition topology can produce different profiles
if (2) differs. Two systems with the same topology and the same microdynamic
flavor (e.g., CP and Ising both have stochastic local updates at comparable
rates) will produce more similar profiles.

### 9f. Updated Type catalogue

| Type | Topology | Per-cell temporal churn | Example |
|------|----------|-------------------------|---------|
| **A** | Symmetry-breaking | Yes (stochastic spin flips) | Ising, Potts, XY |
| **A′** | Sync transition | Yes (phase drift) | Kuramoto — fits A despite not being symmetry-breaking |
| **B** | Absorbing-state | Yes (stochastic death) | Contact Process — peaks AT p_c |
| **B′** | Absorbing-state | No (deterministic-ish activation) | DP, SIR, RBN — peaks above p_c |
| **C** | Self-organised to boundary | Either | Sandpile, Voter coarsening |

### 9g. Implications

1. **H1 as originally stated is falsified.** "Profile shape is predictable
   from universality class" fails the DP-vs-CP test.
2. **A weaker H1 survives**: "Profile shape is predictable from (transition
   topology × microdynamic flavor)." This is still useful but less
   parsimonious than the original claim.
3. **The peak offset is not a physics observable** — it depends on whether
   the update rule produces temporal opacity at the critical point. It's a
   property of the (system, framework) pair, not of the system alone.
4. **For the paper**: we should not claim "C detects phase transitions."
   A more defensible claim is "C detects the coincidence of spatial and
   temporal structure. Systems where criticality coincides with both
   (Ising, CP, Kuramoto) are detected at p_c; systems where temporal
   structure develops separately from the transition (DP, SIR) are detected
   at an offset determined by the framework's sensitivity profile, not by
   the critical physics."

## 10. Open Questions (revised)

1. **Is the DP/CP discrepancy resolvable?** Could we build a variant of the
   framework whose temporal opacity gate responds to synchronous-DP-style
   dynamics? (E.g., measuring branching ratios instead of cell-level MI.)
   If yes, would DP then peak at p_c?
2. **Can we predict peak offset quantitatively** from the microdynamics?
   Given a rule, can we say "this system will peak at density ρ ≈ X,
   therefore offset Y%"?
3. **The voter model is a puzzle.** Should it go in C or a new D? Its
   absorbing state is the complex state — the *opposite* of Type B where
   the absorbing state is the dead state.
4. **Does the 2/4 blind score mean H1 is wrong, or that our predictor is
   too crude?** If we refine to "predict based on both topology AND
   microdynamic flavor," do we get 4/4 retroactively? (Trivial when you
   use the result to build the theory — the real test needs new substrates.)


---

## 11. Multi-Scale C — the resolution

### 11a. Motivation

The section 9 investigation identified w_OP_t ≡ 0 as the bottleneck for DP:
synchronous DP without explicit death produces deterministic per-cell
time-series, so the cell-level temporal-opacity gate never fires. The
hypothesis: **critical systems have scale-invariant structure**, so the
temporal structure must exist somewhere — just not at the smallest scale.
Measuring C at multiple coarse-grained scales should reveal it.

### 11b. Design

For each substrate, compute C at pool factors {×1, ×2, ×4, ×8}. At pool
factor ×k, partition the grid into k×k blocks, replace each block by its
mean, and re-binarise at the top-25% threshold. Then run the standard C
pipeline on the coarsened history. For non-lattice systems (e.g. RBN) use
*temporal* striding instead, averaging consecutive time-steps.

Scripts: `multiscale_diagnostic.py`, `multiscale_diagnostic_extended.py` in
`preliminary-research/experiments/c-profile-fingerprint/`.

### 11c. Results (2026-04)

**Primary sweep (Ising, DP, CP, Sandpile):**

| Substrate | Cell-level (×1) at p_c | Coarse-level (×8) at p_c | Outcome |
|-----------|------------------------|---------------------------|---------|
| Ising (T_c = 2.269) | C = 0.71, w_OP_t = 0.86 | C = 1.10, w_OP_t = 1.00 | Preserved, stronger |
| DP (p_c = 0.2873) | **C = 0.00, w_OP_t = 0.00** | **C = 0.89, w_OP_t = 0.99** | **Full rescue** |
| CP (λ_c = 1.6489) | C = 0.72, w_OP_t = 0.97 | C = 0.82, w_OP_t = 0.99 | Preserved |
| Sandpile (ε = 0) | C = 0.001, w_OP_t = 0.03 | C = 0.022, w_OP_t = 0.29 | 22× amplified |

**Extended sweep (SIR, RBN, Voter):**

| Substrate | Cell-level peak location | Multi-scale peak location | Outcome |
|-----------|--------------------------|----------------------------|---------|
| SIR (β_c ≈ 0.0125) | β = 0.050, +300% offset | β = 0.030, +140% offset | Partial rescue |
| RBN (K_c = 2.0) | K = 2.5, +25% (temporal) | Signal destroyed at t≥2 | Negative — needs graph coarsening |
| Voter (μ = 0) | C = 0.17 at μ = 0 | **C = 0.99 at μ = 0** | **Full rescue, 6× amplified** |

### 11d. Interpretation

The multi-scale analysis splits the 2D-lattice substrates into three groups:

1. **Already-scale-invariant at cell level**: Ising, CP, Potts, Kuramoto.
   These peak at p_c at every scale. Multi-scale adds robustness, not
   resolution.
2. **Mesoscale-only critical structure**: DP, Voter. Cell-level dynamics are
   too deterministic (DP) or too trivial (Voter copying) to carry information,
   but domain-level dynamics are richly critical. Multi-scale fully resolves
   the apparent offset.
3. **Additional internal state**: SIR. Three-state dynamics (S/I/R) create
   spatial structure the cell-level already partially sees, so coarsening
   helps less than for two-state systems. Multi-scale pulls the peak toward
   β_c but doesn't reach it.

Non-lattice systems (RBN) require a different coarsening operator (graph
partition / community coarsening) — temporal striding destroys information
because state(t+2) is near-deterministic in state(t).

### 11e. Updated H1 — Multi-Scale Version

> **For any 2D-lattice system with scale-invariant critical physics,
> multi-scale C peaks at the critical point** — regardless of whether the
> microdynamics provide cell-level temporal churn.

This is stronger than the section-9 "topology × microdynamics" claim and
weaker than the original "universality-class fingerprint" claim. The honest
scope: **two-dimensional lattice systems with coarsenable binary state and
scale-invariant criticality.** Extensions to graph / continuous / higher-state
systems are open questions.

### 11f. Consequences for the paper

With multi-scale C, we can report:

- **Ising 2D** — clean Type A peak at T_c (multi-scale confirms).
- **Potts q = 3** — clean Type A peak at T_c.
- **DP** — once presented as a failed offset, now a *success* of multi-scale
  C: peak relocates from +30% offset to p_c at coarse scales.
- **Contact Process** — Type A at every scale; bonus evidence that same
  universality class produces consistent multi-scale profiles.
- **Sandpile** — the textbook SOC system; multi-scale amplifies its signal
  22× as expected for a scale-free critical state.
- **Voter** — Type C with multi-scale; the absorbing coarsening state is
  detected cleanly.

SIR and RBN can be documented as **scope statements** rather than failures:
SIR reveals that multi-state substrates are a partial case, RBN reveals that
non-lattice topologies need graph-aware coarsening. Both are useful
information about the framework's reach.


## 12. Open Questions (post-multi-scale)

1. **Potts at high q (first-order regime).** Does multi-scale C distinguish
   first-order (q ≥ 5) from second-order (q ≤ 4) transitions via the RG beta
   function? This would extend the framework from "detects criticality" to
   "discriminates transition order." A direct extension using existing Potts
   code.
2. **RBN with graph coarsening.** Group nodes by module / community,
   re-measure. Does the K = 2 edge-of-chaos appear cleanly when coarsening
   respects the graph topology?
3. **SIR fully multi-scale.** Does measuring on the union of I and S fields
   (rather than I alone) fully rescue the peak?
4. **Frustrated Ising / spin glasses.** Scale-invariant but non-trivially so
   — do they land cleanly in Type A at multi-scale, or do glassy dynamics
   require a different analysis?
5. **Continuous-state systems (Gray-Scott, Kuramoto amplitudes).** Our
   coarsening goes through a 25%-threshold binarisation at each scale. Does
   a continuous multi-scale C (without re-binarisation) preserve information
   better?


## 13. q-ary C and the Criticality Detector (2026-04 extension)

The Potts q-sweep (section 12 item 1) forced us to confront two latent
questions: **(a)** does the framework need a multi-state generalisation, and
**(b)** how do we identify a critical point without prior knowledge of where
it sits? Both were answered affirmatively within a single day of experiments.

### 13a. Motivation — why binary Potts failed at high q

Two Potts binarizations were tried before the q-ary pipeline:

- **v1 "state 0 vs rest"** — degenerate at high q. At q = 10, state 0
  occupies ~10% of cells on average, so coarsened blocks are uniformly zero
  and all spatial structure vanishes at pool factors ×2+. The ×2, ×4 C values
  collapsed to ≈0 for q ≥ 5.
- **v2 "majority-cluster"** — cell = 1 if it matches the most populous state
  at that snapshot. Keeps the binary split near 50/50 regardless of q, but
  introduces a new artifact: the majority state re-labels across snapshots,
  scrambling temporal MI. Produced non-monotonic C vs pool profiles
  (e.g. q=2: C = [0.115, 0.000, 0.032, 0.324]) that look like physics but are
  measurement artifacts.

The fix was not a better binarization. The fix was no binarization.

### 13b. q-ary C (`compute_C_qary.py`)

A compact 100-line generalisation of the binary pipeline:

```
w_H    = H(p_1..p_q) / log q                     # max at uniform
w_OP_s = I(X ; X_neighbour) / H(X)                # q×q joint counts
w_OP_t = I(X_t ; X_{t+1}) / H(X_t)                # q×q transitions
w_T    = 1 − 2|change_frac − 0.5|                 # peaked at 0.5
w_G    = 1 − 2|unlike_frac/(1−1/q) − 0.5|         # peaked at half-max disorder
C      = w_H × ½(w_OP_s + w_OP_t) × w_T × w_G
```

Spatial coarsening replaces "block-mean then threshold" with "per-block
mode" so the categorical structure is preserved under pooling.

**Sanity checks:** uniform-random q-state field → C ≈ 0 (no structure);
frozen field → C ≈ 0 (no variation). Both correct.

### 13c. Potts q-ary results — the decisive test

Re-running the Potts sweep with q-ary C on the raw q-state field:

| q  | Order   | Peak T | C(×1)  | C(×2)  | C(×4)  | C(×8)  |
|----|---------|--------|--------|--------|--------|--------|
| 2  | 2nd     | 1.191  | 0.095  | 0.090  | 0.077  | 0.051  |
| 3  | 2nd     | 1.045  | 0.130  | 0.101  | 0.066  | 0.037  |
| 5  | 1st     | 0.894  | 0.067  | 0.040  | 0.033  | 0.022  |
| 10 | 1st     | 0.675  | 0.068  | 0.052  | 0.035  | 0.024  |

Every q produces a clean, monotone-decaying C vs pool profile — no more ×2
dips, no more ×8 explosions. The v2 binarization artifacts were measurement
noise, not physics.

**The β-sign-pattern hypothesis is falsified by the q-ary data.** The
prediction that 2nd-order Potts would show `+++` (flat/growing) and
1st-order would show a peaked β was an artifact of how binarization
degraded the C(pool) profile for different q.

### 13d. First-order vs second-order discriminator — RETRACTED

**Earlier claim (RETRACTED 2026-04):** The cliff ratio C(peak)/C(peak+1) on
the Potts q-sweep was monotone with transition order (1.06 → 2.17 → 3.05 →
4.00 across q = 2, 3, 5, 10), suggesting a universal discriminator.

**Falsification.** The Blume-Capel blind test (`blume_capel_blind_test.py`,
`blume_capel_blind_test_G128.py`) walks a single model across its phase
diagram by varying the crystal field D ∈ {0, 1.0, 1.5, 1.9, 1.99}, crossing
the tricritical point near D ≈ 1.965 from the second-order side into the
first-order side. Cliff ratios at G=128:

| D    | Expected order    | Cliff ratio | Verdict |
|------|-------------------|-------------|---------|
| 0.00 | 2nd (Ising line)  | 2.58        |         |
| 1.00 | 2nd               | 1.16        |         |
| 1.50 | 2nd               | **3.27**    | highest — but 2nd-order |
| 1.90 | 2nd (near tricrit)| 2.21        |         |
| 1.99 | 1st (past tricrit)| **1.15**    | lowest — supposed to be sharpest |

The cliff ratio goes the *wrong* direction across the tricritical point, and
the highest cliff is at D=1.5 — the most deeply second-order case. The
Potts q-scan pattern was a 4-point coincidence, not a physical
discriminator.

**Candidate replacement signal (not yet validated):** at D=1.99, C_8 is
identically 0.000 at every T even while C_1 peaks at ~0.05. The coarsest
scale sees no structure at all — the correlation length stays below 8 cells
through the transition. This "C_8 flatlines while C_1 peaks" pattern is a
plausible first-order signature, but one data point is not a test. Open
question, not a result.

### 13e. The criticality detector (`criticality_detector.py`)

Three independent indicators applied to a multi-scale sweep:

1. **Scale-collapse**: parameter where C values across scales agree best,
   weighted by C magnitude (suppresses trivial dead-region agreement).
2. **β-zero**: parameter minimising sum of |β(s→2s)| across RG steps.
3. **Coarsest-peak**: parameter maximising C at the coarsest pool.

Consensus = mean. Confidence = 1 − spread/(0.15 × param range).

### 13f. Validation scoreboard

| System       | Known value | Consensus est. | Error   | Verdict |
|--------------|-------------|----------------|---------|---------|
| Ising 2D     | T_c = 2.269 | 2.267          | +0.002  | spot-on |
| DP           | p_c = 0.287 | 0.283          | −0.004  | spot-on |
| CP           | λ_c = 1.649 | 1.500          | −0.149  | *sweep edge — false high-confidence* |
| Potts q=2 q-ary  | 1.135   | 1.191          | +0.057  | within grid step |
| Potts q=3 q-ary  | 0.995   | 1.045          | +0.050  | within grid step |
| Potts q=5 q-ary  | 0.852   | 0.894          | +0.042  | within grid step |
| Potts q=10 q-ary | 0.701   | 0.675          | −0.026  | within grid step |
| SIR          | β_c ≈ 0.014 | 0.037          | +0.023  | offset persists, see 13g |
| Blume-Capel D=0.0 (G=128)    | T_c = 1.693 | 1.819  | +0.126  | within ~1 grid step |
| Blume-Capel D=1.0 (G=128)    | T_c = 1.40  | 1.471  | +0.071  | within grid step |
| Blume-Capel D=1.5 (G=128)    | T_c = 1.15  | 1.157  | +0.007  | spot-on |
| Blume-Capel D=1.9 (G=128)    | T_c = 0.90  | 0.833  | −0.067  | within grid step |
| Blume-Capel D=1.99 (G=128)   | T_t = 0.422| 0.717  | +0.295  | **tricritical failure** |

All errors except CP (sweep-range edge artifact), SIR (see 13g), and
Blume-Capel D=1.99 (tricritical, see 13j) are at or below parameter-grid
spacing — the detector is grid-limited in its success regime.

**Known failure mode**: if the sweep range doesn't bracket the true
critical point, all three indicators agree at the range boundary and
report misleading high confidence (CP case). Mitigation: flag edge hits.

### 13g. SIR remains offset — not a binarization artifact

The q-ary SIR sweep:

| β      | C(×1) | C(×2) | C(×4) | C(×8) |
|--------|-------|-------|-------|-------|
| 0.020  | 0.000 | 0.000 | 0.000 | 0.000 |
| 0.040  | **0.006** | **0.007** | 0.005 | 0.003 |
| 0.060  | 0.004 | 0.004 | 0.003 | 0.002 |

Detector consensus β = 0.037 vs true β_c ≈ 0.014. The v1 binary analysis
peaked at β = 0.050, multi-scale binary at β = 0.030, q-ary at β = 0.037.
**Switching to q-ary alone does not rescue SIR.** The peak amplitude is
also ~20× smaller than Potts — a qualitative structural difference.

Interpretation: SIR is a *transient* absorbing-state system. By the time
the fixed measurement window t ∈ [10, 160] captures the field, the rich
epidemic phase has already burnt through and the lattice is mostly
post-recovery. The offset is a **measurement-window** problem, not a
representation problem. Fix would require an event-triggered window that
locks onto the epidemic front. Added to open questions below.

### 13j. Blume-Capel blind test — documented failure at the tricritical point

Full-sweep results at G=128 (16k cells, 4k burn-in sweeps, 3 seeds per T):

| D    | T_c published | T_c estimate | err     | conf | notes |
|------|---------------|--------------|---------|------|-------|
| 0.00 | 1.693         | 1.819        | +0.126  | 0.55 | 2nd-order |
| 1.00 | 1.40          | 1.471        | +0.071  | 1.00 | 2nd-order |
| 1.50 | 1.15          | 1.157        | +0.007  | 0.55 | 2nd-order, near-perfect |
| 1.90 | 0.90          | 0.833        | −0.067  | 0.55 | 2nd-order, near tricritical |
| 1.99 | 0.422         | 0.717        | +0.295  | 0.40 | past tricritical — **fails** |

Going from G=48 to G=128 reduced the D=1.99 error from +0.43 to +0.30 and
shifted the peak from T≈0.85 to T≈0.79, confirming part of the
finite-size-rounding hypothesis — but the residual error is too large to be
lattice-size alone. At D=1.99 the C-profile has C_8 = 0.000 at every T
while C_1 peaks modestly; the ordering length scale never reaches 8 cells,
so the coarsest indicator is blind and the detector's consensus drifts.

**What's good here.** The detector flagged D=1.99 with its lowest
confidence (0.40) and gave excellent results everywhere on the
second-order line (D ≤ 1.9). That is appropriate self-awareness, not a
cover-up.

**Scope statement for the paper.** The detector is validated on
second-order transitions with bulk correlation length > pool factor × 1
cell. First-order / tricritical transitions where the coarsest scale
flatlines are a known failure mode.

### 13k. Consolidated failure modes

Pulling the individual cases together so the honest scope is visible in one
table. "C signal" is whether the primary C peak sits at the right parameter
value; "detector consensus" is whether the 3-indicator mean agrees.

| System                    | Issue                         | C signal (pool ×1)  | Detector consensus   | Root cause                                             |
|---------------------------|-------------------------------|---------------------|----------------------|--------------------------------------------------------|
| Ising / Potts / DP        | —                             | ✅ at grid-step acc. | ✅ conf > 0.9         | Textbook second-order critical — all indicators align. |
| Contact Process (CP)      | Sweep range too narrow        | ✅ λ≈1.6 shoulder    | ❌ λ=1.5 (edge-pinned)| Supercritical phase looks scale-invariant; detector picks the edge. Fix: widen sweep, edge-hit flag now caps confidence. |
| SIR                       | Transient absorbing state     | ⚠️ 20× weaker peak  | ❌ β=0.037 vs 0.014   | Fixed measurement window samples the post-epidemic lattice, not the active front. Fix: event-triggered window. |
| Blume-Capel D ≤ 1.9       | —                             | ✅                  | ✅                    | Second-order line, no issue. |
| Blume-Capel D = 1.99      | Tricritical / near 1st-order  | ⚠️ peak at wrong T  | ❌ T=0.72 vs 0.42     | C_8 ≡ 0 through the transition — ordering length scale < 8 cells. Detector self-reports low confidence (0.40). |
| Gray-Scott                | Continuous field, quantisation-sensitive | ⚠️ order reversed | N/A (no sweep)   | Binary thresholding pins density (kills w_H), so it measures geometry at fixed density. q-ary preserves intensity, so w_H×w_OP is maximised in chaotic regime. The two measurements answer different questions. |
| Kuramoto                  | Non-scale-invariant transition| ✅ K=2.5 matches binary | ❌ K=3.5, conf=0.00 | Sync is a mean-field phase lock with no correlation-length divergence. C_8 peaks at wrong K (sparse slow-drift regime), β-zero hits noise tail. Single-scale C is fine; multi-scale detector is not. |

**Two orthogonal distinctions this table surfaces.**

1. *Signal vs consensus*: "C peak at right parameter" and "detector consensus
   at right parameter" can disagree. Kuramoto shows the first without the
   second; SIR shows neither; Blume-Capel D=1.99 shows neither with low
   self-reported confidence.

2. *Measurement vs methodology*: CP and SIR are fixable by changing what
   we measure (sweep range, window). Blume-Capel tricritical and Kuramoto
   sync are methodology limits — the multi-scale/consensus apparatus is
   built around scale-invariance and those systems do not have it.

Gray-Scott is a third category: the signal is what you decide to measure.
Binary-at-fixed-density and q-ary-on-intensity are both valid, but they
measure different things. The paper has to commit to one and say why.

### 13h. Updated open questions

1. **SIR event-triggered window.** Lock the measurement window onto
   max dI/dt (peak of epidemic) rather than fixed t ∈ [10, 160]. Does
   q-ary C then peak at β_c?
2. **Detector hardening.** Two easy additions: (a) flag consensus when it
   lies at the edge of the sweep range (CP false-positive fix);
   (b) quadratic interpolation around the peak for sub-grid-step accuracy.
3. **Transition-order discriminator — unresolved.** Cliff ratio is
   falsified (see 13d). Candidate replacement: "C_8 flatlines while C_1
   peaks" observed in Blume-Capel D=1.99, but untested on other first-order
   systems. Needs a multi-model validation before re-claiming any
   discriminator.
4. **q-ary extension to non-lattice substrates.** RBN with k = 3-5 colours
   instead of binary states; does graph coarsening + q-ary fix the
   edge-of-chaos localisation problem?
5. **Continuous-state C** (differential entropy + KSG MI estimators)
   remains open; still the preferred path for Kuramoto amplitudes and
   Gray-Scott concentrations.

### 13i. What this means for the paper

We now have three reportable methodological contributions, not one:

- **Binary multi-scale C** — original tool, peaks at T_c for 2D lattice
  systems with scale-invariant criticality.
- **q-ary C** — substrate-agnostic multi-state extension; preferred for
  Potts, SIR, categorical agent models.
- **Criticality detector** — turns a parameter sweep into a blind T_c
  estimate with confidence metric; validated on 4 systems with known
  answers.

The honest framing of scope: the framework *finds* critical points for
2D-lattice systems with persistent second-order critical phases (validated
on Ising, DP, Potts q=2–10, and the Blume-Capel line for D ≤ 1.9) and
extends cleanly to multi-state fields. It **does not** reliably locate
tricritical / first-order transitions when the ordering length scale
collapses below the coarsest pool (Blume-Capel D=1.99), **does not** rescue
transient absorbing-state systems (SIR) without measurement-window changes,
and **does not** handle non-lattice topologies (RBN) without graph-aware
coarsening. It **does not** currently output a transition-order label —
the cliff-ratio hypothesis was falsified by the Blume-Capel blind test.
All of these are documented scope statements, not hidden failures.

