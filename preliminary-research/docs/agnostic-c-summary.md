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

## 5. Width-Scaling Analysis: Which Metrics Can Go Agnostic?

A systematic analysis of all 256 ECA rules at grid widths W = 50–600 revealed that the six raw metrics split into two categories:

### Width-invariant metrics (candidates for parameter-free treatment)
| Metric | C4 value | Variation across W | Can go agnostic? |
|--------|----------|--------------------|------------------|
| tcomp | 0.579 | ±0.001 (0.02%) | Partially — see §6 |
| gzip | 0.125 | ±0.002 (1.5%) | **Best candidate** — see §6 |
| op_down | 0.971 | ±0.004 (0.4%) | Already near boundary |
| mean_H | 0.984 | negligible | Already uses tanh gates |

### Width-dependent metrics (Gaussian peaks tied to grid size)
| Metric | Scaling law | C4 at W=150 | C4 at W=600 |
|--------|-------------|-------------|-------------|
| op_up | ~W^(-2.1) | 0.139 | 0.002 |
| std_H | decreasing | 0.014 | 0.006 |

**Key finding:** The Gaussian peak at μ_up=0.14 is an artifact of W=150, not a universal constant. At W=600, C4 op_up is nearly zero. However, C3 op_up is width-invariant at ~0.33, so C4/C3 separation on op_up *improves* at larger grids. The fine-tuned Gaussian becomes less necessary as the grid grows — at large W, any monotonically decreasing gate on op_up would work.

---

## 6. Cross-Substrate Analysis: What Is Truly Universal?

Compiled tcomp and gzip at peak C across every substrate with experimental data:

| Substrate | tcomp | gzip |
|-----------|-------|------|
| 1D ECA C4 (W=150) | 0.579 | 0.124 |
| 2D Life-like C4 | 0.90 | — |
| Voter Model (L=64) | 0.557 | 0.141 |
| Potts q=3 (L=64) | 0.848 | 0.140 |
| Potts q=4 (L=64) | 0.822 | 0.145 |
| Potts q=5 (L=64) | 0.869 | 0.125 |
| Ising q=2 (L=64) | 0.893 | 0.107 |
| Contact Process (L=64) | 0.881 | 0.099 |
| Boids/Vicsek (L=32) | 0.829 | 0.135 |

### tcomp: NOT universal — requires triple Gaussian

tcomp clusters into distinct attractors by substrate dimensionality:
- 1D ECA: ~0.58
- 2D lattice models: ~0.86 (SD=0.026)
- 2D Life-like C4: ~0.90

These are exactly the three peaks in the existing triple Gaussian (0.58, 0.73, 0.90). The peaks encode real dynamical attractors, not arbitrary calibration points. They cannot be replaced with a symmetric tanh gate without losing discrimination. The agnostic `gate(tc_mean)` peaking at 0.5 misses all three attractors.

### gzip: NEARLY UNIVERSAL — best candidate for a derived constant

Across all substrates: **mean = 0.127, SD = 0.016** (12.6% coefficient of variation). This is remarkably tight given the diversity of substrates (1D CA, 2D spin models, flocking, contact processes).

The current v8 Gaussian peak at μ=0.10 is slightly miscalibrated — the empirical center is ~0.125. A corrected Gaussian at μ=0.125 or a theoretically derived value near this would be more accurate.

### The 1/8 mystery — RESOLVED

The near-universal gzip ratio of ~0.125 = 1/8 initially appeared to be a potential fundamental constant. Investigation revealed its origin is more prosaic but still informative.

**Root cause: byte encoding overhead.** The framework stores binary (0/1) cell states as `uint8` bytes, wasting 7 bits per cell. Gzip recovers most of this waste, producing ratios near 1/8 for any near-random binary data stored this way.

**Multi-compressor verification:** gzip, bzip2, lzma, and zlib all agree within ±0.02 on the same data. The ratio is a property of the data encoding, not a gzip-specific artifact.

**The real compression ratios (bit-packed):**
| Class | byte-stored gzip | bit-packed gzip | byte × 8 ≈ bit? |
|-------|-----------------|-----------------|------------------|
| C4 (Rule 110) | 0.121 | 0.781 | 0.965 ✓ |
| C4 (Rule 124) | 0.104 | 0.678 | 0.832 ≈ |
| C3 (Rule 30) | 0.162 | 1.004 | 1.295 (incompressible) |
| C1 (Rule 0) | 0.002 | 0.009 | 0.016 (trivial) |

**What the gzip metric actually measures:**
- C4 rules compress to ~78-87% in bit-packed form — they have genuine large-scale pattern redundancy (gliders, repeating structures)
- C3 rules are incompressible even bit-packed (ratio ≈ 1.0) — essentially random
- Random uint8 binary data gives ~0.16 (not 0.125) due to gzip header overhead
- C4 rules scoring *below* the random baseline (0.12 vs 0.16) is the real discriminating signal

**Implication:** The gzip Gaussian peak at μ=0.10 is tuned to the byte-encoded ratio. This is fine as long as the encoding stays consistent, but it means gzip ≈ 0.125 is NOT a derivable physical constant — it's an encoding artifact. The real question would be whether the bit-packed ratio of ~0.80 for complex systems has a theoretical derivation, but that's a much harder question (essentially asking for the Kolmogorov complexity of edge-of-chaos computation).

**Practical consequence:** The gzip Gaussian cannot be made parameter-free by deriving its peak from first principles. However, it remains a useful and consistent discriminator as long as the byte encoding convention is maintained across substrates.

---

## 7. Revised Assessment: Path to Parameter-Freedom

| Parameter | Status | Path forward |
|-----------|--------|--------------|
| Entropy tanh gates | Already parameter-free | Done |
| Temporal opacity tanh gates | Already parameter-free | Done |
| Spatial opacity tanh (non-CA) | Already parameter-free | Done |
| Spatial opacity Gaussian (μ_up=0.14) | Width-dependent artifact | Not needed at large W; use one-sided gate rewarding low op_up |
| Spatial opacity Gaussian (μ_down=0.97) | Near boundary, stable | Low priority — already near tanh saturation |
| Entropy variance bonus (μ_σ=0.012) | Width-dependent | Decreasing importance at larger grids |
| tcomp triple Gaussian | Substrate-dependent attractors | Cannot remove — encodes real physics (three distinct dynamical regimes) |
| gzip Gaussian (μ=0.10) | Byte-encoding artifact (~1/8) | NOT derivable ��� value is an artifact of uint8 storage; keep as empirical calibration |
| Fractal dimension (μ_ex=0.35) | 2D only | Needs more substrates to assess universality |

**Bottom line:** Full parameter-freedom is not achievable. tcomp genuinely requires substrate-aware peaks (three distinct dynamical attractors). The gzip ratio of ~0.125 turned out to be a byte-encoding artifact (1/8), not a fundamental constant — the real bit-packed compressibility of complex systems is ~0.80. Spatial opacity Gaussians become unnecessary at larger grid scales. The honest framing is that the framework has a small number of empirically confirmed parameters that encode real physics (tcomp attractors, gzip byte-ratio peak) plus parameter-free components (entropy gates, temporal opacity gates) that work across all substrates.

---

## 8. Key Files

| File | Role |
|------|------|
| `experiments/neural-network/mnist-experiment.py` | Implements all five agnostic weight functions and `compute_full_C` |
| `experiments/blind-critical-sweep/blind_sweep.py` | Blind Potts sweep (Ising, q=3, q=5) |
| `experiments/blind-critical-sweep/offset_analysis.py` | Post-hoc comparison against literature T_c |
| `experiments/blind-critical-sweep/fss_analysis.py` | Finite-size scaling analysis |
| `experiments/blind-critical-sweep/eca_raw_metrics.csv` | All 256 ECA rules, raw metric values |
| `docs/neural-network-experiments-summary.md` | Full record of the CNN/GRU/GPT-2 experiments using C_a |
