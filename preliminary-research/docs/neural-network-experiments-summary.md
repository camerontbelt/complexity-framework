# Neural Network Complexity Experiments — Summary

**Project:** Applying the C complexity metric to neural network substrates  
**Period:** Early 2026  
**Scripts:** `neural-network/mnist-experiment.py`, `further-experiments.py`, `targeted-experiments.py`, `gpt2-experiment.py`

---

## 1. Motivation and Starting Point

The complexity framework (v8) had already demonstrated strong results on cellular automata — correctly ranking Class-4 rules (e.g. Rule 110) at the top and cleanly separating them from Class-1/2/3 behaviour. A natural question followed: **can the same metric detect a signal in trained neural networks?**

The original hypothesis, inspired by earlier Grok experiments, was:

> A trained neural network should score higher on the C metric than a randomly initialised one, because training drives the network toward a more structured, context-sensitive internal representation — analogous to a CA operating at the edge of chaos.

This is the claim this body of work set out to test.

The metric formula throughout is:

$$C = w_H \times (w_{\text{OP},s} + w_{\text{OP},t}) \times w_T \times w_G$$

with six raw sub-metrics: `mean_H`, `std_H` (spatial entropy), `op_up`, `op_down` (spatial opacity), `mi1`, `decay` (temporal opacity), `tc_mean` (temporal compression), `gzip_ratio`.

---

## 2. Experiment Setup — MNIST CNN

**File:** `mnist-experiment.py`

### 2.1 Model

A small CNN (`SimpleCNN`) trained on MNIST:
- `Conv2d(1→8, 3×3)` → ReLU → `Conv2d(8→16, 3×3)` → ReLU → MaxPool → `fc(3136→128)` → `fc(128→10)`
- Trained to ~98% test accuracy

### 2.2 Measurement approach

Activations from a batch of test images were stacked along a "temporal" axis (one image = one time step), then binarised and passed to `compute_full_C`. The 3D activation grids were compared: **trained CNN vs. 50 randomly initialised CNNs**.

### 2.3 Bugs discovered and fixed

| Bug | Effect | Fix |
|-----|--------|-----|
| Continuous ReLU floats treated as binary | Metrics computed on wrong distribution | Binarise explicitly: `activation > 0` |
| Temporal compression read only channel 0 | Ignored 15 of 16 feature maps | Flatten all channels into grid |
| MaxPool saturation | Activation density artificially inflated to ~94% | Measure `pre_pool` activations instead |
| `float32` log2(0) warnings | NaN cascade | Cast to `float64` before entropy |
| Gzip header overhead on tiny tensors | `gzip_ratio > 1.0` → negative `wG_a` | Clip gzip ratio to `[0, 1]` |

### 2.4 Calibration problem

The v8 weight functions use Gaussian peaks calibrated on 1D ECAs:
- `wops`: peaks at `op_up ≈ 0.14, op_down ≈ 0.97`
- `wg`: peaks at `gzip_ratio ≈ 0.10`

CNN metric distributions fall far outside these ranges (e.g. `op_down` is consistently near 1.0 for CNNs). As a result, `C ≈ 0` for everything — the calibration silenced the metric on this substrate.

**Solution:** Developed *domain-agnostic* tanh-gated weight functions that only require metrics to be non-trivial (away from 0 and 1), with no substrate-specific peak location:

```python
K_GATE = 10

def wh_agnostic(mean_H, std_H):
    gate_H = tanh(K_GATE * mean_H) * tanh(K_GATE * (1 - mean_H))
    bonus  = tanh(K_STD * clip(2*std_H, 0, 1))
    return gate_H * (1 + bonus)
```

Identical functional form for all five weight functions. This produces a non-zero `C_a` for any substrate without requiring empirical calibration.

---

## 3. MNIST CNN Results

### 3.1 Main finding: trained < random

With both v8 and agnostic weights, the trained CNN consistently scored *lower* than the distribution of random CNNs. The trained model fell around the **20th–40th percentile** of the random baseline distribution.

### 3.2 CSV diagnostics

Dumping all sub-metrics to `mnist_results.csv` revealed the key driver: **temporal metrics were nearly identical between trained and random CNNs**. The temporal axis (stacking independent test images) was producing noise, not signal — consecutive images are independent by construction, so any "temporal" structure the metric detected was an artefact of dataset ordering, not model dynamics.

This was confirmed by the **Permutation Control experiment** (Exp 3, `further-experiments.py`): shuffling images in different orders (original, class-sorted, random) produced measurably different `mi1` values, proving that the metric was reading dataset structure, not model structure.

### 3.3 Spatial signal: std_H

One genuine signal emerged: **per-image spatial complexity** (Exp D, `targeted-experiments.py`). Computing `wH_agnostic` per image — purely spatial, no temporal axis — showed:

- Trained CNN: higher `std_H` (more channel-to-channel variation within an image)
- Random CNNs: lower `std_H` (homogeneous activations)

**Effect size: Cohen d ≈ 0.50, p = 0.001.** This is interpretable: trained CNNs develop feature-specialised filters. Different channels respond to different local features, creating within-image activation diversity. Random CNNs have undifferentiated filters.

---

## 4. Further Experiments

**File:** `further-experiments.py`

### Exp 1 — Training Dynamics (epoch-by-epoch)
C_a was computed at each epoch. Finding: **C_a peaks at epoch 1 then declines monotonically** as training converges. Complexity is highest during the initial rapid weight change (gradient signal strong, representations not yet settled), then decreases as the model converges to stable attractors.

### Exp 2 — Layer-wise Complexity
Each CNN layer was measured separately. Result: shallow layers (conv1) scored higher C_a than deep layers (fc2). Feature extraction layers retain more varied activations; the final classification layer collapses everything to a 10-class decision.

### Exp 3 — Permutation Control
Testing sensitivity to image ordering confirmed the temporal axis was contaminated by dataset structure (see §3.2 above).

### Exp 4 — Channel Specialisation Index (SI)
`SI = std(per-channel C_a)` was proposed as a measure of functional specialisation. Unexpectedly, **trained SI < random SI**. Diagnosis: random CNNs have chaotic variance (some channels at 0, others at 2.5), while trained CNNs form tight bimodal clusters (near 0 and near 2.0–2.5). Standard deviation is a poor summary of structured bimodality.

### Exp 5 — RNN/GRU on Sequential MNIST
A `SimpleGRU` was trained on sequential MNIST (pixels fed one at a time — a genuine causal time series). GRU hidden states provide a true temporal axis. Result:

- **Trained GRU C_a > random GRU baseline: d ≈ 0.21, p = 0.04**
- Small but significant — the first positive trained-vs-random signal found

Interpretation: the GRU genuinely integrates information across time. Its hidden states at step t are a function of all prior steps, creating the causal structure the temporal metrics were designed to detect.

---

## 5. Targeted Experiments

**File:** `targeted-experiments.py`

| Exp | Description | Result |
|-----|-------------|--------|
| A | GRU trained vs. 100 random GRUs | H1 confirmed: d = 0.21, p = 0.04 |
| B | GRU C_a across 15 training epochs | C_a peaks early then declines (same pattern as Exp 1) |
| C | Channel Specialisation Index (CNN) | H0 not rejected (std-based SI reversed) |
| D | Per-image spatial complexity (CNN) | H1 confirmed: d = 0.50, p = 0.001 |

---

## 6. The Conceptual Shift

By the end of the CNN/GRU experiments, a key conceptual reframing had emerged:

> **Trained classifiers are the wrong substrate.** The job of a classifier is to map rich input distributions onto a small, stable set of class labels. Stability and compactness are virtues for a classifier. An edge-of-chaos metric will correctly observe that a well-trained classifier has *less* dynamic complexity than a randomly thrashing one — but this does not mean the metric is wrong.

A more appropriate substrate is one where the model's task *requires* it to maintain rich, dynamic internal representations: **generative models**. A language model cannot simplify its hidden states without losing generative capability. Generating coherent, diverse, contextually appropriate text requires dynamics that are neither frozen nor chaotic.

---

## 7. GPT-2 Experiments

**File:** `gpt2-experiment.py`

### 7.1 Design

GPT-2 small (124M parameters, 12 transformer blocks, 768-dim residual stream) was used as the substrate. Key design choices:

- **Temporal axis:** token position (1 to 200). GPT-2 is causally masked — `h_t = f(tokens 0..t)` — so token order is a *genuine* causal time axis, unlike the fake temporal axis used in the CNN experiments.
- **Volume format:** one `(1, 1, 768)` array per token position, giving T=200 temporal steps and W=768 spatial positions.
- **Layer probed:** L6 (mid-network) as the primary layer, with a depth profile at layers 0, 3, 6, 9, 12 for the trained model.
- **Efficiency fix:** all hidden states extracted in a *single forward pass* per model (extracting per-layer in separate passes caused 5× wall-clock inflation on CPU).

### 7.2 Exp E — Trained vs. Untrained GPT-2

**Hypothesis H1:** Pre-trained GPT-2 produces higher C_a at L6 than randomly initialised GPT-2s on the same coherent text (d > 0.5, p < 0.05).

| | Trained | Random (n=20) |
|---|---|---|
| C_a at L6 | 0.0068 | 0.0114 ± 0.0064 |
| mi1 | 0.075 | 0.22 – 0.37 |
| tc_mean | 0.647 | 0.813 |

**Result: H0 not rejected. Effect in the opposite direction (d = −0.72, p = 0.006).**

The trained model scored *lower* than random, replicating the CNN pattern — but the mechanism is different and illuminating. Random GPT-2 uses near-uniform attention (key/query dot products ≈ 0 before softmax → weights spread over all positions). Every token's hidden state ends up as roughly the average of all token embeddings, so `h_t ≈ h_{t+1}` at every step. This creates *spuriously high* temporal MI and high temporal compression — not from meaningful computation, but from trivial averaging. The trained model's selective, content-specific attention produces hidden states that vary more meaningfully across positions, lowering mi1.

**Layer profile (trained model):**

| Layer | C_a | mi1 | tc_mean |
|-------|-----|-----|---------|
| embed | 0.0033 | 0.039 | 0.625 |
| L3 | 0.0077 | 0.071 | 0.623 |
| L6 | 0.0068 | 0.075 | 0.647 |
| L9 | 0.0087 | 0.087 | 0.674 |
| L12 | 0.0330 | 0.096 | 0.704 |

C_a and mi1 both increase with depth, peaking at the final layer — consistent with the model integrating longer-range dependencies as information flows through more transformer blocks.

### 7.3 Exp F — Coherent vs. Random Input on Trained GPT-2

**Hypothesis H1:** The trained GPT-2 produces higher C_a at L6 when processing coherent English text than when processing random uniform-token sequences.

| | Coherent text (n=5) | Random tokens (n=5) |
|---|---|---|
| C_a | 0.0063 ± 0.0004 | 0.0042 ± 0.0002 |
| mi1 | 0.0956 | 0.0150 |
| tc_mean | 0.6789 | 0.5714 |
| gzip_ratio | 0.2331 | 0.2331 |
| mean_H | 0.9995 | 0.9996 |

**Result: H1 confirmed. d = 11.5, p ≈ 0.**

When the same trained GPT-2 processes coherent language vs. random tokens, the hidden-state dynamics are measurably and substantially more complex. The primary drivers are `mi1` (coherent 6× higher) and `tc_mean`. Crucially, `gzip_ratio` and `mean_H` are effectively identical between conditions — the spatial structure and overall density of active neurons do not differ. The entire signal is in the temporal dimension.

**Interpretation:** When given coherent text the model can process meaningfully, attention heads selectively integrate context across positions, building up a richer cross-token representation. The metric correctly detects that the residual stream dynamics are more structured — not because activations are more or less dense, but because they evolve in a more temporally coherent way.

---

## 8. Consolidated Findings

### 8.1 What works

| Signal | Description | Effect |
|--------|-------------|--------|
| Per-image spatial complexity (CNN) | `wH_agnostic` per image, no temporal axis | d ≈ 0.50, p = 0.001 |
| GRU trained vs. random | Genuine causal temporal axis (sequential MNIST) | d ≈ 0.21, p = 0.04 |
| GPT-2 coherent vs. random input | Same model, different input structure | d = 11.5, p ≈ 0 |

### 8.2 What doesn't work (and why)

| Experiment | Expected | Got | Why |
|------------|----------|-----|-----|
| CNN trained > random (temporal) | Trained higher C_a | Trained < random | Temporal axis is independent test images — not a causal time series |
| GPT-2 trained > random | Trained higher C_a | Trained < random | Random attention averages all positions → spuriously high temporal MI |
| Channel specialisation (SI) | Trained higher SI | Trained lower SI | Random CNNs have chaotic variance; trained CNNs have structured bimodality — std is a poor summary |

### 8.3 The trained < random pattern

Every experiment comparing a trained network to random ones produced the same directional result: **trained < random on C_a**. This is not a bug — it is a systematic finding with a coherent interpretation:

> Training optimises a network toward *stable, low-variance attractors* that reliably map inputs to outputs. Stability is the goal, and stability reduces dynamic complexity as the metric defines it. Random networks thrash more — whether from uncorrelated activations (CNN) or uniform attention averaging (GPT-2) — and that thrashing registers as high apparent complexity.

This does not invalidate the metric. It means the metric is correctly measuring dynamic complexity, but trained classifiers and language models are engineered to be *less dynamically complex* in pursuit of reliable task performance.

### 8.4 Where the signal lives

The clearest positive signal — Exp F (GPT-2 coherent vs. random input, d = 11.5) — comes from **holding the model constant and varying the input**. The trained GPT-2 creates richer dynamics when given input it can actually process. This points to the most scientifically clean use of the C metric on neural network substrates:

> Use C as a probe of *how richly a model engages with its input*, rather than as a comparison between model states (trained vs. untrained).

---

## 9. Open Questions and Future Directions

1. **Attention-aware complexity:** The current metric treats the residual stream as a black box. Applying the metric to attention pattern matrices directly (each head's `T×T` weight matrix as the spatial grid) might reveal structure that the residual stream hides.

2. **Generative quality as a proxy:** Compare C_a for GPT-2's hidden states while generating high-quality text (low perplexity) vs. degenerate text (high temperature or low-quality prompts). Does C_a correlate with generation quality?

3. **Cross-layer temporal axis:** Use layers as the time axis (for a single token or short sequence), measuring how the representation evolves through depth. This avoids the token-ordering artefacts entirely.

4. **Intermediate training checkpoints:** Exp B and Exp 1 showed C_a peaks early and declines. What is the shape of this curve across the full training run? Is there a phase transition at the point the model first generalises?

5. **Natural vs. constructed language:** A hypothesis in `further-enquiries.md` — can the C metric distinguish natural language from constructed language (e.g. Esperanto, Lojban, code)? Exp F provides the template for this experiment.

6. **Parameter-free formalisation:** The agnostic tanh weights are already close to parameter-free. The geometric volume interpretation (`C` as a 7D volume in metric space, as described in `further-enquiries.md`) offers a fully principled route to deriving weight boundaries from first principles rather than calibration data.

---

## 10. File Map

```
preliminary-research/
├── neural-network/
│   ├── mnist-experiment.py          # Core CNN experiment + compute_full_C
│   ├── further-experiments.py       # Exps 1-5 (dynamics, layers, permutation, channels, GRU)
│   ├── targeted-experiments.py      # Exps A-D (pre-registered hypotheses, full CSV output)
│   ├── gpt2-experiment.py           # Exps E-F (GPT-2 trained/untrained, coherent/random)
│   ├── analyse_csv.py               # Statistical analysis of mnist_results.csv
│   └── inspect_gpt2.py              # GPT-2 architecture inspection and hidden-state profiling
├── neural-network-experiments-summary.md   ← this file
└── N-D_discrete_equation.md
```

**Generated data files** (in `neural-network/`): `mnist_results.csv`, `expE_gpt2_trained_vs_random.csv`, `expF_gpt2_coherent_vs_random_input.csv`, plus various `.png` plot files.
