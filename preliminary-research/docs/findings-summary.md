# Complexity Framework — Findings Summary

**Last updated:** 2026-04-15

This document is the single-page overview of every substrate tested against
the C framework, the verdict for each, and the conclusions we drew. It is
intentionally terse — deeper treatment for each row lives in the cited
source doc.

For the mathematical definition of C, see [`c-metric-math.md`](c-metric-math.md).
For the full taxonomy and per-substrate narrative, see
[`c-profile-taxonomy.md`](c-profile-taxonomy.md). For the multi-scale / RG
angle and the q-ary extension, see [`multiscale-c-summary.md`](multiscale-c-summary.md).
For descriptions of each substrate in plain language, see
[`substrate-reference.md`](substrate-reference.md).

---

## 1. What we set out to do

Build a single, substrate-agnostic complexity metric C that could rank or
classify any dynamical system handed to it. Validate it on systems with
known critical points or known complexity orderings, and use it to
discriminate between trivial / ordered / complex / chaotic regimes.

## 2. What we actually found

A universal scalar "complexity" does not exist. There are **kinds** of
complexity with orthogonal signatures, and each kind needs its own
measurement machinery. The C framework as currently built is a validated
detector for one kind — scale-invariant second-order criticality — and
partially informative on others. See §4 for the kinds we identified.

---

## 3. Experiment scoreboard

Columns:

- **Result**: ✅ works as intended / ⚠️ partial signal / ❌ fails or is
  substantially off.
- **C peak**: does C(θ) peak at the parameter value of interest?
- **Detector**: does the 3-indicator `criticality_detector` consensus land
  at the right parameter?

| Substrate                          | Variant        | C peak           | Detector      | Result | Source |
|------------------------------------|----------------|------------------|---------------|--------|--------|
| Conway's Game of Life              | binary v9      | calibration case | —             | ✅ (anchor) | `ca-multiscale.py` |
| ECA 256-rule scatter               | binary v9      | class 3/4 elevated | —           | ✅ | `eca-full-scatter.py`, `eca-scatter-beta.py`, `eca-agnostic-rank.py` |
| ECA 1D sweep                       | binary v9      | flat across classes | —          | ❌ scale too small — needs 2D | `ca-multiscale.py` |
| 2D Ising                           | binary + q-ary | ✅ T_c = 2.267 vs 2.269 | ✅ conf ≈ 1.0 | ✅ | `multiscale_diagnostic.py`, `ising_grid_convergence.py` |
| Directed Percolation (DP)          | binary + q-ary | ✅ p_c = 0.283 vs 0.287 | ✅ | ✅ | `multiscale_diagnostic.py` |
| Contact Process (CP)               | binary + q-ary | ✅ shoulder at λ≈1.65 | ❌ edge-pinned at λ=1.5 | ⚠️ | `multiscale_diagnostic.py` |
| Potts q=2,3,5,10                   | **q-ary req.** | ✅ all within 1 grid step | ✅ | ✅ | `potts_qary_sweep.py` |
| SIR                                | binary + q-ary | ⚠️ 20× weaker, offset | ❌ β=0.037 vs 0.014 | ⚠️ | `sir_qary_sweep.py` |
| Blume-Capel, D ∈ {0, 1.0, 1.5, 1.9} | q-ary (G=128) | ✅ all within grid step | ✅ conf 0.55–1.0 | ✅ | `blume_capel_blind_test_G128.py` |
| Blume-Capel, D = 1.99 (tricritical) | q-ary (G=128) | ❌ T=0.72 vs 0.42 | ❌ conf 0.40 | ❌ | `blume_capel_blind_test_G128.py` |
| Gray-Scott, 6 Pearson regimes      | binary         | complex > chaotic  | —            | ✅ (under binary) | `gray-scott-multiscale.py` |
| Gray-Scott, 6 Pearson regimes      | q-ary (q=4)    | **chaotic > complex** (reversed) | — | ⚠️ quantisation-sensitive | `gray_scott_qary.py` |
| Kuramoto (2D lattice)              | binary         | peak at K ≈ 2–3.5 | — | ✅ | `fingerprint_test.py` |
| Kuramoto (2D lattice)              | q-ary (q=4 sectors) | ✅ K = 2.5 | ❌ consensus K=3.5, conf=0.00 | ⚠️ | `kuramoto_qary.py` |
| RBN (Kauffman)                     | binary         | ⚠️ K≈2 shoulder, not sharp | ❌ no consensus | ❌ | `fingerprint_test.py` |
| Schelling segregation              | binary         | low peak, broad | — | ⚠️ | `schelling-test.py` |
| Sandpile / abelian avalanches      | binary         | elevated across regime | — | ✅ (SOC) | `lattice-experiments.py` |
| Frustrated lattice                 | binary         | elevated at T≈frustration scale | — | ✅ | `geometrical-frustration.py` |
| Boids                              | binary         | elevated in flocking regime | — | ✅ qualitative | `boids-experiment.py` |
| MNIST CNN (layer activations)      | binary v9      | elevated in middle layers | — | ✅ qualitative | `mnist-experiment.py`, `targeted-experiments.py` |
| GPT-2 hidden states                | binary + temporal multiscale | elevated at deeper layers | — | ✅ qualitative | `gpt2-experiment.py`, `gpt2-multiscale.py` |

### Supporting methodology experiments

| Experiment                        | Finding | Source |
|-----------------------------------|---------|--------|
| Criticality detector calibration  | **Confidence is NOT calibrated.** No monotone relationship between reported confidence and error on 3200 synthetic trials. | `detector_calibration.py` |
| Ising grid convergence            | Error is **not monotone in grid spacing**; bootstrap CIs contain truth at every resolution. Right primary output is the CI, not the point estimate. | `ising_grid_convergence.py`, `detector_bootstrap.py` |
| β-zero mask tuning                | Raising alive-region threshold from 5% → 30% removes noise-tail pathology; Ising consensus error improved from −0.061 to +0.031. | `criticality_detector.py` |
| Edge-hit guard                    | Consensus pinned within 1 grid step of a sweep endpoint now caps reported confidence at 0.10. Catches CP false positive. | `criticality_detector.py` |
| Cliff ratio hypothesis            | **FALSIFIED** by Blume-Capel blind test. Cliff ratio is highest at D=1.5 (2nd-order), lowest at D=1.99 (past tricritical). The Potts q-monotone pattern was a 4-point coincidence. | `blume_capel_blind_test.py`, `blume_capel_blind_test_G128.py` |
| Peak offset (~35%)                | Persistent offset between C peak and analytic T_c across several 2D lattice systems. Documented, not fully explained — probably finite-T slow-mode bias in the composite weights. | `peak-offset-analysis.py`, `offset_analysis.py` |

---

## 4. The kinds of complexity we identified

Failures cluster cleanly into four categories. Each has a distinct
diagnostic signature and would need its own instrument.

| Kind | Signature in our data | Examples | What C as built does |
|------|----------------------|----------|----------------------|
| **Scale-invariant critical** | All indicators align; β ≈ 0; C peaks at every pool scale | Ising, Potts, DP, Blume-Capel D<1.9 | ✅ Detects reliably within one grid step |
| **Entity-level emergent** | C depends strongly on the quantisation AND on whether pool factor matches entity size | Gray-Scott (spots, worms, solitons), arguably MNIST/GPT-2 | ⚠️ Reports *a* number, but ranking is quantisation-dependent |
| **Mean-field / phase-locked** | Single-scale C peaks correctly; multi-scale detector diverges because correlation length is local | Kuramoto sync transition | ⚠️ Peak yes, consensus no |
| **Transient / absorbing** | Fixed measurement window misses the active phase | SIR epidemic, CP near extinction | ❌ Offset unless window locks onto event |

The CP and sweep-range cases are a fifth, milder category — methodological
artefacts that a better sweep fixes. Not a new kind of complexity, just
failure to bracket the right parameter window.

---

## 5. Methodological contributions

Independent of the scope debate, these are real, reusable artefacts:

1. **q-ary C pipeline** (`compute_C_qary.py`) — drops binarisation, uses
   q×q joint-count MI, per-block mode coarsening. Preferred for multi-state
   native substrates (Potts, Blume-Capel, any categorical agent model).
2. **Multi-scale C profile** (`multiscale_diagnostic.py`) — C evaluated at
   pool factors 1, 2, 4, 8. Produces the discrete RG β-function
   β(s → 2s) = C_{2s} − C_s, whose zero-crossing is a candidate fixed-point
   indicator.
3. **Criticality detector** (`criticality_detector.py`) — three independent
   indicators (scale-collapse, β-zero, coarsest-peak) with consensus +
   confidence + edge-hit guard. Validated on 6 systems with known answers.
4. **Bootstrap error bars** (`detector_bootstrap.py`) — the right primary
   output for a detector whose point estimate is not monotone in grid
   spacing. Use the 95% CI, not the mean.
5. **Formal math reference** (`c-metric-math.md`) — single source of truth
   for all three C variants (binary, q-ary, multi-scale).

---

## 6. Honest scope statement for the paper

The framework as it stands is:

- **A validated detector for second-order critical transitions** on 2D
  lattice substrates with diverging correlation length. Within one
  parameter-grid step on 6 canonical systems, including a blind test
  against a model with a published tricritical point.
- **A scope-aware tool with four documented failure categories**, each with
  a diagnostic signature rather than a hidden caveat.
- **A taxonomy of complexity kinds**, not a universal scalar metric. The
  universal-scalar hypothesis was tested and falsified by our own data
  (Gray-Scott binary-vs-q-ary order reversal, Kuramoto detector failure
  on a system whose C does peak at the right parameter).

It is **not** a universal complexity classifier, does not handle
entity-level emergence except incidentally, does not handle mean-field
transitions with its multi-scale machinery, does not handle transient
absorbing dynamics without an event-triggered window, and does not handle
non-lattice topologies without graph-aware coarsening.

Those are load-bearing limitations, not future work notes. The paper that
acknowledges them will age better than the one that doesn't.

---

## 7. Candidate next directions (if picked up later)

Ordered by tractability, not priority:

1. **Decision-tree classifier of complexity kind.** Given an unknown
   system, run 3–4 cheap diagnostics (is there a parameter sweep? does C
   peak across scales? does the field have absorbing states?) and output
   which of the four kinds it belongs to. This is a smaller, honest
   realisation of the original "universal framework" goal.
2. **Event-triggered window for transient systems.** Lock the measurement
   window onto max d(active)/dt. Would test whether SIR / CP-near-extinction
   are rescued.
3. **Entity-aware coarsening for Gray-Scott-type systems.** Segment the
   field into objects first (connected components, watershed), then
   measure C on the entity-level graph. Likely addresses the
   quantisation-sensitivity problem.
4. **Graph-aware coarsening for RBN.** Spectral or modularity-based
   coarsening instead of spatial pooling.
5. **Continuous-state C.** Differential entropy + KSG mutual information
   estimators. Would remove the quantisation choice in Gray-Scott /
   Kuramoto at the cost of heavier machinery.

---

## 8. One-line summary

We set out to build a universal complexity metric. We built a validated
detector for one kind of complexity — scale-invariant criticality — and
discovered in the process that "complexity" is not one thing. The four
kinds we mapped and the failure modes we documented are themselves the
result.
