# Agnostic C Calculation — Summary

**Context:** The original complexity framework (v8) used weight functions with Gaussian peaks calibrated empirically on 1-D Elementary Cellular Automata. These weights silenced the metric when applied to any substrate where the raw metrics fall in different ranges (e.g. CNNs, Gray-Scott, GPT-2). This document summarises the investigation into making the metric domain-agnostic.

---

## 1. The Problem with Calibrated Weights

The v8 weight functions encode specific expectations about where "interesting" values sit:

```
wops: peaks at op_up ≈ 0.14, op_down ≈ 0.97
wg:   peaks at gzip_ratio ≈ 0.10
wh:   Gaussian in (mean_H, std_H) calibrated on 1-D CA
```

When CNNs were measured, `op_down` was consistently close to 1.0 (as expected for a high-dimensional system), `gzip_ratio` was ~0.25 (well away from 0.10), and `op_up` was near 0. Result: `C ≈ 0` for every condition, trained or untrained — the calibration had silenced the metric entirely on a new substrate.

**The deeper issue:** Calibrating peaks empirically on one substrate bakes in an assumption that the *specific values* at which complexity is interesting are universal. They are not. What *is* universal is the logical condition: complexity requires metrics to be intermediate, not at their trivial extremes.

---

## 2. The Agnostic Solution

The agnostic weight functions replace substrate-specific Gaussian peaks with **tanh boundary gates** that only require metrics to be non-trivial:

```
gate(x) = tanh(K * x) * tanh(K * (1 - x))
```

This function is:
- Zero at x = 0 (trivially inactive)
- Zero at x = 1 (trivially saturated)
- Positive everywhere in between, peaking near x = 0.5
- No free parameters beyond the sharpness constant K = 10

The five weight functions in agnostic form:

| Weight | Original (v8) | Agnostic |
|--------|---------------|----------|
| `wh` | Gaussian in (mean_H, std_H) with CA-calibrated peak | `gate(mean_H) * (1 + tanh(K_STD * std_H_normalised))` |
| `wops` | Product of two Gaussians (op_up, op_down) | `gate(op_up) + gate(op_down)` — additive OR rather than AND |
| `wopt` | `tanh(K*mi1)*tanh(K*(1-mi1))*tanh(K*decay)` | Same — this was already parameter-free |
| `wt` | Max of three Gaussians at tc ∈ {0.58, 0.73, 0.90} | `gate(tc_mean)` |
| `wg` | Gaussian at gzip_ratio = 0.10 | `gate(gzip_ratio)` |

The key change to `wops` from AND (multiplicative) to OR (additive): the original product collapses to zero whenever either spatial opacity channel is trivial. For CNNs and high-dimensional systems, `op_up` is often near zero while `op_down` carries real signal — the AND form suppressed all spatial opacity signal from those substrates.

### Implementation

All agnostic functions live in `experiments/neural-network/mnist-experiment.py`, and `compute_full_C` uses them by default. Every subsequent experiment script bootstraps `compute_full_C` from that file, so the agnostic calculation is used consistently throughout.

---

## 3. What Changed in Results

| Substrate | C (v8 calibrated) | C_a (agnostic) | Change |
|-----------|-------------------|-----------------|--------|
| ECA Class 4 (Rule 110) | Scored correctly | Scored correctly | — |
| CNN (trained) | ≈ 0 | 0.3–2.5 | Unlocked the metric |
| CNN (random) | ≈ 0 | 0.5–3.0 | Unlocked, still trained < random |
| GPT-2 (coherent input) | ≈ 0 | 0.006 | Small but measurable |
| Gray-Scott (complex) | ≈ 0 | 0.5–1.5 | Clear signal at entity scale |

The agnostic metric does not fix every experiment — the CNN trained < random result persists — but it correctly produces non-zero scores across all substrates, allowing principled comparison.

---

## 4. Trade-offs and Limitations

**What was gained:**
- Works out of the box on any substrate without recalibration
- All weight functions derived from first principles (information-theoretic boundary conditions)
- The `wops` additive form is strictly more informative for high-dimensional substrates

**What was lost:**
- The Gaussian peaks in v8 encode empirical knowledge about *where* complexity tends to sit for ECA specifically. On ECA, the calibrated v8 peaks produce a tighter, more discriminating signal for Wolfram class separation.
- The agnostic metric produces higher raw scores everywhere (gates are broader than Gaussians), making absolute C values harder to interpret across substrates without normalization.

**Remaining open question:** The agnostic form is still a product of five terms. The *multiplicative* structure means any single near-zero term collapses the whole score. A system that is deeply complex along four dimensions but trivially simple along a fifth will score near zero. Whether this is a feature (strict complexity requires all conditions simultaneously) or a bug (it discards partial complexity signals) is unresolved.

---

## 5. Key Files

| File | Role |
|------|------|
| `experiments/neural-network/mnist-experiment.py` | Implements all five agnostic weight functions and `compute_full_C` |
| `docs/neural-network-experiments-summary.md` | Full record of the CNN/GRU/GPT-2 experiments using C_a |
