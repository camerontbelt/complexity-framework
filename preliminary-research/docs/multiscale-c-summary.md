# Multi-Scale C Calculation — Summary

**Motivation:** The existing C metric measures complexity at a single spatial and temporal scale. Real complex systems (Gray-Scott patterns, ECA gliders, flocking) have structure that emerges at a characteristic *entity scale* — not at the finest cell level. This work explored whether measuring C at multiple spatial coarsening levels reveals information the single-scale metric misses.

---

## 1. The Entity-Scale Problem

### What triggered it

The Gray-Scott reaction-diffusion system, when measured at the raw cell level (64×64 grid), gave near-identical C scores for all parameter regimes (trivial, periodic, complex, chaotic). The adaptive thresholding fix helped, but the metric still couldn't cleanly separate chaotic from complex patterns.

Root cause: a chaotic Gray-Scott field and a complex Gray-Scott field look similar at the scale of individual cells. Both have ~25% active density and spatially varying patterns. The difference is that the *complex* field has persistent, organised structures (spots, spirals) at the scale of 4–16 cells, while the chaotic field has no characteristic length scale.

### The proposed fix

Measure C at multiple spatial pooling factors, then compare the profile:

```
pool_factor = 1  →  measure raw cells
pool_factor = 2  →  average 2x2 patches → one super-cell, binarise
pool_factor = 4  →  average 4x4 patches → one super-cell, binarise
pool_factor = 8  →  average 8x8 patches → one super-cell, binarise
```

At coarse scales, the metric "sees" larger structures. If complexity increases with pooling factor (C(2) > C(1)), the system has structure at a scale larger than individual cells. If it peaks then falls (C(4) > C(2) > C(8)), there is a characteristic entity scale around 4×4 cells.

---

## 2. Implementation

### Spatial coarsening

```python
def frames_to_volumes(frames, pool_factor, active_pct=25.0):
    for frame in frames:
        if pool_factor == 1:
            binary = raw_binary_frame   # no pooling
        else:
            super_cells = frame.reshape(n_sup, pool_factor, n_sup, pool_factor)
                                .mean(axis=(1,3))
            threshold   = np.percentile(super_cells, 100 - active_pct)
            binary      = (super_cells > threshold).astype(float)
        volumes.append(binary.reshape(1, 1, -1))
```

The adaptive threshold at coarse scales maintains ~25% active super-cells regardless of scale, keeping the entropy gate from collapsing.

### Temporal striding (GPT-2)

The same principle applied to the temporal axis for GPT-2 hidden states:

```python
def strided_volumes(h, stride):
    indices = np.arange(0, h.shape[0], stride)
    return [h[i][np.newaxis, np.newaxis, :] for i in indices]
```

Stride 1 = every token; stride 4 = every 4th token. Coarser strides reveal longer-range temporal structure.

---

## 3. The Discrete RG Beta Function

The profile `{pool_factor → C_a}` is a vector. To extract a single interpretable number, the **discrete renormalization group (RG) beta function** was defined:

```
beta(s → 2s) = C_a(2s) - C_a(s)
```

| Beta sign | Interpretation |
|-----------|----------------|
| β > 0 everywhere | Complexity increases under coarsening → no characteristic entity scale (trivial or scale-free chaotic) |
| β < 0 somewhere | Complexity peaks then falls → the *entity scale* is the pool factor where beta first goes negative |

This gives a principled, framework-motivated way to extract an "entity scale" from the profile without ad-hoc thresholding. Systems that operate at a characteristic scale (e.g. Gray-Scott spots of radius ~4 cells) should show beta turning negative at that scale.

---

## 4. Results by Substrate

### Gray-Scott (spatial coarsening)

| Regime | Profile shape | Entity scale | C at peak |
|--------|--------------|--------------|-----------|
| Trivial (all dead) | Monotone ↑ (noise artefact) | None (always β > 0) | ~0.5 (artefact) |
| Ordered (periodic) | Peaks at ×2 then ↓ | ×2 | ~0.8 |
| Complex (spots/spirals) | Peaks at ×4 then ↓ | ×4 | ~1.5 |
| Chaotic | Monotone ↑ | None (β > 0 throughout) | ~0.4 at ×8 |

The complex regime clearly separates from chaotic and trivial in the profile shape, even when raw C scores overlap.

### GPT-2 (temporal striding)

| Condition | C at stride 1 | C at stride 4 | AUC |
|-----------|--------------|--------------|-----|
| Trained, coherent text | 0.006 | ~0 | Low |
| Trained, random tokens | 0.004 | ~0 | Low |
| Random model, coherent | 0.011 | 0.008 | Higher |
| Random model, random tokens | 0.011 | 0.007 | Higher |

The trained model's signal exists only at stride 1 (word level) and vanishes at coarser temporal scales. This is consistent with the earlier finding that the trained model processes language at the token level, not through long-range temporal dependencies.

### ECA Multi-scale + AUC

All 256 ECA rules were run through pool factors [1, 2, 4, 8]. The Area Under the Curve (AUC = sum of C_a across all pool factors) was computed as a single scalar:

| Wolfram Class | Typical AUC | Typical C at ×1 |
|--------------|-------------|-----------------|
| Class 1 (trivial) | ~0–1 | ~0 |
| Class 2 (periodic) | ~2–6 | ~0–0.5 |
| Class 3 (chaotic) | ~6–12 | ~1–3 |
| Class 4 (complex) | ~6–10 | ~0–1 |

Classes 3 and 4 both achieve high AUC, but Class 4 rules achieve it with *lower* C at ×1 — their complexity is concentrated at coarser scales (entity/glider level) rather than at the finest cell scale.

### ECA Beta Function (Entity Scale)

Adding entity scale as a third dimension in the 256-rule scatter plot **did not cleanly separate Classes 3 and 4** in 1-D ECA. Both classes showed similar entity scale distributions (mostly ×2). The entity scale approach works better for 2-D systems with physically-sized spatial structures (e.g. Gray-Scott spots) than for 1-D systems where "entities" are propagating signals rather than spatially compact objects.

---

## 5. Key Findings

1. **Multi-scale measurement reveals structure that single-scale misses.** The Gray-Scott regime separation that failed at a single scale became clear once the C profile across scales was examined.

2. **The beta function provides a principled entity scale.** It correctly identifies the pool factor where structured patterns (spots, stripes, spirals) are most visible.

3. **AUC rewards breadth of complexity across scales.** This is a useful summary statistic for ranking systems but penalises "narrow" complexity (e.g. Rule 110, which is highly complex only at scale ×1).

4. **In 1-D, the entity scale does not cleanly separate Wolfram classes.** The method needs 2-D spatial structure (actual objects with spatial extent) to show its discriminating power.

5. **The hierarchical idea is sound in principle.** A chain of C measurements at increasing scales is the natural generalisation of the metric to higher dimensions and continuous substrates. However, it adds significant complexity to the workflow and the results need careful interpretation.

---

## 6. Key Files

| File | Role |
|------|------|
| `experiments/gray-scott/gray-scott-multiscale.py` | Multi-scale C on Gray-Scott, generates profile plots |
| `experiments/gray-scott/gs_beta_analysis.py` | Computes discrete RG beta, entity scale scatter plots |
| `experiments/cellular-automata/ca-multiscale.py` | Multi-scale C on ECA + GoL, computes AUC |
| `experiments/cellular-automata/eca-full-scatter.py` | All 256 ECA rules, 2-D scatter (C×1, AUC) |
| `experiments/cellular-automata/eca-scatter-beta.py` | Augments scatter with entity scale from beta function |
| `experiments/c-profile-fingerprint/compute_C_qary.py` | q-ary C (no binarization, multi-state native) |
| `experiments/c-profile-fingerprint/criticality_detector.py` | Blind T_c estimator from multi-scale sweep |
| `experiments/c-profile-fingerprint/potts_qary_sweep.py` | Potts q ∈ {2,3,5,10} q-ary multi-scale sweep |
| `experiments/c-profile-fingerprint/sir_qary_sweep.py` | SIR (S/I/R natively q=3) q-ary multi-scale sweep |

---

## 7. q-ary C and Blind Critical-Point Detection (2026-04)

Two extensions added during the Potts q-sweep:

**q-ary C** — drops binarization entirely. Each sub-metric generalised to q
states: `w_H = H/log q`, MI weights use q×q joint/transition counts,
w_T/w_G are peaked at half-maximum. Coarsening uses per-block mode instead
of block-mean-then-threshold.

**Criticality detector** — three independent indicators (scale-collapse,
β-zero, coarsest-peak) with consensus and confidence. Validated to within
one grid-step on Ising (T_c 2.267 vs 2.269), DP (p_c 0.283 vs 0.287), and
Potts q-ary (all q ∈ {2,3,5,10}, max error 0.057).

SIR q-ary still offsets (β_est 0.037 vs β_c 0.014) — the offset is not a
binarization artifact but a measurement-window issue: SIR is transient, so
a fixed post-burnin window samples the post-epidemic absorbed state rather
than the active epidemic. See `c-profile-taxonomy.md` §13 for full
treatment.

---

## 8. Quantisation matters — Gray-Scott q-ary disagrees with binary (2026-04)

`experiments/gray-scott/gray_scott_qary.py` re-runs the 6 Pearson/Munafo
regimes with q=4 quantile-bin quantisation of the v-field instead of the
per-frame 75th-percentile threshold used by the original binary pipeline.

**Ranking by peak multi-scale C (q-ary, pools 1/2/4/8, 2 seeds):**

| Regime          | Class    | peak C (q-ary) | peak pool |
|-----------------|----------|----------------|-----------|
| chaotic         | chaotic  | 0.195          | ×4        |
| solitons        | complex  | 0.033          | ×1        |
| worm_complex    | complex  | 0.023          | ×4        |
| static_spots    | ordered  | 0.011          | ×8        |
| self_rep_spots  | complex  | 0.007          | ×1        |
| dead            | trivial  | 0.000          | ×1        |

The binary-multi-scale paper result has "complex" regimes outranking
"chaotic" — that's the edge-of-chaos framing. Under q-ary the ordering is
**reversed**: chaos dominates by 6× and one of the iconic complex regimes
(self-replicating spots) sits below the ordered (static_spots) case.

**Interpretation.** Binary thresholding at the 75th percentile pins frame
density to a constant — every frame has the same fraction of "on" cells,
so `w_H` is eliminated as a signal and C reduces to structure-at-fixed-
density. Under that measurement, labyrinthine/soliton patterns look most
structured. q-ary preserves the intensity distribution; chaos has
maximally mixed intensities *and* local correlations, so `w_H × w_OP`
is maximised there.

This is not "binary was wrong." Binary was measuring pattern geometry at
a chosen density; q-ary is measuring information-theoretic structure over
the raw field. They are different questions and — in Gray-Scott at least
— they produce genuinely different answers. The paper needs to state
which one it is reporting and why.

**Scope statement going forward.** On lattice systems with discrete native
state alphabets (Ising, Potts, Blume-Capel, DP, CP), binary and q-ary C
agree on the qualitative shape of C(θ) up to overall scale. On continuous
fields where the quantisation is a design choice (Gray-Scott, Kuramoto,
neural activations), binary and q-ary C can differ in magnitude *and*
ordering; results must report the quantisation used.

### 8a. Kuramoto q-ary — peak location robust, detector not

`experiments/c-profile-fingerprint/kuramoto_qary.py` re-runs the 2D-lattice
Kuramoto sweep (G=64, K ∈ [0, 10], 3 seeds) with q=4 angular-sector
quantisation of the phase field.

| K    | C×1   | C×2   | C×4   | C×8   |
|------|-------|-------|-------|-------|
| 1.0  | 0.010 | 0.007 | 0.004 | 0.008 |
| 2.0  | 0.019 | 0.013 | 0.007 | 0.005 |
| **2.5**  | **0.021** | 0.015 | 0.008 | 0.004 |
| 3.0  | 0.020 | 0.014 | 0.008 | 0.004 |
| 5.0  | 0.005 | 0.007 | 0.005 | 0.002 |
| 10.0 | 0.002 | 0.003 | 0.004 | 0.002 |

The q-ary peak at K = 2.5 (pool ×1) coincides with the archived binary peak
region (K ≈ 2–3.5). **Peak location is robust to quantisation — unlike
Gray-Scott.** Magnitudes are about 2× lower, consistent with the known
q=2-vs-q-ary offset.

However, the **criticality detector fails** on this substrate:
- scale-collapse → K = 2.5 (correct)
- β-zero → K = 7.0 (noise tail)
- coarsest-peak → K = 1.0 (C×8 maxes at low K, not at the sync transition)
- consensus = 3.5, confidence = 0.00

Interpretation: the Kuramoto sync transition is a *mean-field* phase lock,
not a scale-invariant critical phenomenon. There is no correlation-length
divergence at K_c that a multi-scale indicator can lock onto. C×8 picks up
the *sparse* correlations present at low K (nearly independent oscillators
with slow drift) rather than the ordered sync phase. Two of three detector
indicators mis-fire, and the consensus is garbage. This is distinct from
the Blume-Capel D=1.99 tricritical failure (where C×8 went flat) — here
C×8 is non-zero but peaks at the wrong K.

**Added to scope statement:** the multi-scale detector is validated for
second-order critical transitions on lattices with diverging correlation
length. Sync-type and tricritical transitions are open failure modes.

---

## 9. Key Files (appended)

| File | Role |
|------|------|
| `experiments/neural-network/gpt2-multiscale.py` | Temporal striding on GPT-2 hidden states |
| `experiments/gray-scott/gray_scott_qary.py` | Gray-Scott q-ary multi-scale (2026-04) |
| `experiments/c-profile-fingerprint/kuramoto_qary.py` | Kuramoto q-ary multi-scale (2026-04) |
| `experiments/c-profile-fingerprint/blume_capel_blind_test_G128.py` | Blume-Capel tricritical blind test |
