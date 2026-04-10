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
| `experiments/neural-network/gpt2-multiscale.py` | Temporal striding on GPT-2 hidden states |
