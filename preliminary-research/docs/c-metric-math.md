# The C Metric — Mathematical Reference

**Purpose.** This document is the single source of truth for how the
complexity metric $C$ is computed, across all three variants currently in
use:

1. **Binary $C$** — the original v9 pipeline, for 2-state spatiotemporal
   fields.
2. **q-ary $C$** — multi-state generalisation for fields with values in
   $\{0, 1, \ldots, q-1\}$.
3. **Multi-scale $C$** — either variant above, evaluated at multiple
   spatial pool factors.

Every experiment in the framework should cite which variant it used and
point back to this document. If you think an experiment is computing $C$
a different way, either it is wrong, this doc is out of date, or a new
variant needs to be documented here.

---

## 1. Shared notation

Let $X$ be a spatiotemporal field sampled on a 2D spatial domain of $N$
cells over $T$ timesteps after burn-in:

$$
X = \{x_{t,i} : 1 \le t \le T,\ 1 \le i \le N\}, \quad x_{t,i} \in \mathcal{S}
$$

where $\mathcal{S}$ is the state alphabet:

- Binary: $\mathcal{S} = \{0, 1\}$, so $|\mathcal{S}| = 2$.
- q-ary: $\mathcal{S} = \{0, 1, \ldots, q-1\}$, so $|\mathcal{S}| = q$.

All logarithms in entropy/MI formulas are natural logs ($\log \equiv \ln$)
unless otherwise noted. Normalisation factors use $\log |\mathcal{S}|$ so
entropies are in $[0, 1]$.

---

## 2. Binary $C$ (the v9 pipeline)

Source of truth: `complexity_framework_v9.py::compute_C`.

The binary composite is

$$
C_{\text{binary}} = w_H \cdot (w_{\text{OP},s} + w_{\text{OP},t}) \cdot w_T \cdot w_G
$$

with optional fractal-dimension multiplier $w_{\dim}$ for 2D substrates,
which we drop here for brevity (see source for its formula).

### 2.1 Entropy weight $w_H$

Let $p_1(t) = \tfrac{1}{N}\sum_i x_{t,i}$ be the mean activation at time
$t$. The time-averaged Shannon entropy of the binary marginal is

$$
H(t) = -p_1(t) \log_2 p_1(t) - (1-p_1(t)) \log_2 (1-p_1(t))
$$

$$
\bar H = \frac{1}{T}\sum_{t=1}^{T} H(t), \qquad
w_H = \bar H \in [0, 1]
$$

(Here $\log_2$ normalises to $[0,1]$ since max entropy for binary is
$\log_2 2 = 1$.)

### 2.2 Spatial opacity $w_{\text{OP},s}$

Opacity measures how compressible/typical the bit-distribution is
compared to an all-zeros or all-ones reference. Let
$p_1(t) = \tfrac{1}{N}\sum_i x_{t,i}$ be the fraction of 1s at time $t$.
Define

$$
\text{op}_{\uparrow}(t) = 1 - \Pr(\text{runs} \ge \ell_0 \mid \text{frame } t)
$$

computed as the fraction of frames whose longest run of 1s is below a
threshold $\ell_0$ determined from the stationary binomial distribution.
(Implementation in `_opacity_both`; not fully re-derived here because the
q-ary pipeline replaces this with a mutual-information measure that is
cleaner to state.)

### 2.3 Temporal opacity $w_{\text{OP},t}$

The normalised mutual information between consecutive time slices,
sampled on a stride-$s$ grid of cells for efficiency:

$$
I(X_t ; X_{t+1}) = H(X_t) + H(X_{t+1}) - H(X_t, X_{t+1})
$$

$$
w_{\text{OP},t} = \text{clip}\!\left(\frac{I(X_t ; X_{t+1})}{\max(H(X_t), \varepsilon)}, 0, 1\right)
$$

Joint and marginal entropies computed from empirical counts over
$\{0,1\}^2$ transition pairs.

### 2.4 Temporal-compression weight $w_T$

From a Lempel-Ziv / temporal-compression ratio $r$ of the
time-concatenated bit string; the weight is a tent/bump function peaked
at an intermediate compression ratio:

$$
w_T = \text{tent}(r;\, r_{\min}, r^\ast, r_{\max})
$$

with $r^\ast$ calibrated on Conway's Game of Life as the "interesting
compression regime." (Not critical for cross-variant equivalence; see
`weight_tcomp` for the exact piecewise form.)

### 2.5 Gzip weight $w_G$

$$
g = \frac{|\text{gzip}(\text{bitpack}(X))|}{|\text{bitpack}(X)|}
$$

$$
w_G = \text{tent}(g;\, g_{\min}, g^\ast, g_{\max})
$$

---

## 3. q-ary $C$

Source of truth: `experiments/c-profile-fingerprint/compute_C_qary.py`.

Designed so that **at $q = 2$ the q-ary pipeline reduces to the binary
pipeline up to definitional choices of $w_{\text{OP},s}$ and $w_T$/$w_G$**
(the information-theoretic weights are identical; the opacity and
compression weights are redefined to have a clean q-ary form).

### 3.1 Global state distribution

Let $n_k$ be the count of state $k$ in all $T \cdot N$ samples of the
measurement window, and $p_k = n_k / (T N)$.

$$
H = -\sum_{k=0}^{q-1} p_k \log p_k
$$

$$
w_H = \text{clip}\!\left(\frac{H}{\log q}, 0, 1\right)
$$

Maximum $w_H = 1$ when all $q$ states are equally populated.

### 3.2 Spatial mutual information $w_{\text{OP},s}$

Horizontal neighbour pairs pooled across all frames. Let $M_s \in
\mathbb{R}^{q \times q}$ be the joint-count matrix

$$
M_s[a, b] = \#\{(t, i) : x_{t,i} = a \text{ and } x_{t, i+1} = b\}
$$

Normalise to joint distribution $P_s = M_s / \sum M_s$, with row marginal
$p^{(r)}_a = \sum_b P_s[a,b]$ and column marginal $p^{(c)}_b = \sum_a P_s[a,b]$.

$$
I_s = \sum_{a,b} P_s[a,b] \log\!\left(\frac{P_s[a,b]}{p^{(r)}_a\, p^{(c)}_b}\right)
$$

$$
H_s^{(r)} = -\sum_a p^{(r)}_a \log p^{(r)}_a
$$

$$
w_{\text{OP},s} = \text{clip}\!\left(\frac{I_s}{\max(H_s^{(r)}, \varepsilon)}, 0, 1\right)
$$

### 3.3 Temporal mutual information $w_{\text{OP},t}$

Same construction, but joint counts over *temporal* neighbour pairs:

$$
M_t[a, b] = \#\{(t, i) : x_{t,i} = a \text{ and } x_{t+1, i} = b\}
$$

and $w_{\text{OP},t}$ is the normalised MI of $P_t = M_t / \sum M_t$
analogous to 3.2.

### 3.4 Temporal change fraction $w_T$

Let $f_T$ be the fraction of cells whose state changes between
consecutive frames:

$$
f_T = \frac{1}{(T-1) N} \sum_{t=1}^{T-1} \sum_{i=1}^{N} \mathbb{1}[x_{t,i} \ne x_{t+1,i}]
$$

$$
w_T = 1 - 2\,|f_T - \tfrac{1}{2}|
$$

Peaked at $f_T = 0.5$ (balanced between frozen and noisy).

### 3.5 Spatial boundary fraction $w_G$

Let $f_G$ be the fraction of horizontally-adjacent cell pairs in different states:

$$
f_G = \frac{1}{T N'} \sum_t \sum_{i} \mathbb{1}[x_{t,i} \ne x_{t,i+1}],
\qquad N' = N - T
$$

Rescale by the maximum value $f_G$ can take under a uniform random field,
$f_G^{\max} = (q-1)/q$, then apply a tent function centred at half-max:

$$
\tilde f_G = \frac{f_G}{f_G^{\max}},\qquad
w_G = 1 - 2\,|\tilde f_G - \tfrac{1}{2}|
$$

### 3.6 Composite

$$
\boxed{\;
C_{\text{q-ary}} \;=\; w_H \cdot \tfrac{1}{2}(w_{\text{OP},s} + w_{\text{OP},t}) \cdot w_T \cdot w_G
\;}
$$

The factor $\tfrac{1}{2}$ keeps the averaged opacity term in $[0, 1]$.

### 3.7 Degenerate cases (sanity)

- **Uniform random field** (all cells IID uniform over $\mathcal{S}$):
  $w_H = 1$, $w_{\text{OP},s} = w_{\text{OP},t} = 0$ (no structure in MI),
  so $C = 0$. ✓
- **Frozen field** (all cells fixed in one state):
  $w_H = 0$, $w_T = 0$, $w_G = 0$, so $C = 0$. ✓

---

## 4. Multi-scale $C$

Source of truth:
- `experiments/c-profile-fingerprint/multiscale_diagnostic.py::coarsen_history`
  (binary coarsening — block-mean + threshold)
- `experiments/c-profile-fingerprint/compute_C_qary.py::coarsen_history_qary`
  (q-ary coarsening — per-block mode)

### 4.1 Spatial coarsening operator $\mathcal{C}_s$

Given a history tensor $X \in \mathcal{S}^{T \times G \times G}$ and a
pool factor $s \in \{1, 2, 4, 8, \ldots\}$, the coarsened history
$\mathcal{C}_s(X) \in \mathcal{S}^{T \times (G/s) \times (G/s)}$ is

$$
\big[\mathcal{C}_s(X)\big]_{t, I, J} = \phi\!\big(\{x_{t,i,j}\}_{i \in [sI, s(I+1)),\ j \in [sJ, s(J+1))}\big)
$$

for a block-aggregation function $\phi$. Two choices, matched to the
target state alphabet:

**Binary ($\phi_{\text{bin}}$):** block mean then threshold at the
per-frame 75th percentile:

$$
\phi_{\text{bin}}(B) = \mathbb{1}\big[\,\overline{B}\, > \,\text{quantile}_{0.75}(\overline{B}_{\text{all blocks}})\,\big]
$$

**q-ary ($\phi_{\text{qary}}$):** per-block mode (most common state):

$$
\phi_{\text{qary}}(B) = \arg\max_{k \in \mathcal{S}} \#\{b \in B : b = k\}
$$

### 4.2 Multi-scale evaluation

For a set of pool factors $\mathcal{P} = \{1, 2, 4, 8\}$:

$$
C_s(X) = \mathrm{compute\_C}\big(\mathcal{C}_s(X)\big), \quad s \in \mathcal{P}
$$

yielding a **C-profile** vector $(C_1, C_2, C_4, C_8)$ per parameter value.

### 4.3 RG $\beta$ function (discrete)

From the C-profile we define the discrete renormalisation-group
$\beta$-function

$$
\beta(s \to 2s) = C_{2s} - C_s, \qquad s \in \{1, 2, 4\}
$$

- $\beta > 0$ (monotone-growing profile): structure is scale-free /
  scale-invariant across that RG step.
- $\beta < 0$ (peaked profile): structure is lost by coarsening; system
  has a characteristic scale.
- $\beta \approx 0$: RG fixed point — classical signature of
  second-order criticality.

### 4.4 Temporal coarsening (for non-spatial substrates)

For substrates without a spatial lattice (e.g. RBN), one can coarsen the
time axis instead:

$$
\big[\mathcal{C}_\tau(X)\big]_{T', i} = \phi\!\big(\{x_{t, i}\}_{t \in [\tau T', \tau(T'+1))}\big)
$$

Currently implemented only for binary fields; see
`multiscale_diagnostic_extended.py::coarsen_temporal`.

---

## 5. The criticality detector

Source: `experiments/c-profile-fingerprint/criticality_detector.py`.

Given a parameter sweep $\{(\theta_k, C_1^{(k)}, C_2^{(k)}, C_4^{(k)}, C_8^{(k)})\}_{k=1}^K$,
three indicators estimate the critical parameter $\theta^\ast$:

1. **Scale-collapse**: argmax over $k$ of a score
  $\mathcal{S}_{\text{coll}}(k) = \bar C_k / (1 + \text{CV}_k)$
  restricted to the 5%-of-peak "alive" region, where
  $\bar C_k = \frac{1}{|\mathcal{P}|}\sum_{s \in \mathcal{P}} C_s^{(k)}$
  and $\text{CV}_k = \sigma_k / \bar C_k$.

2. **$\beta$-zero**: argmin over $k$ of
  $\sum_{s \in \{1,2,4\}} |\beta(s \to 2s; \theta_k)|$,
  restricted to the 30%-of-peak region to avoid noise-tail contamination.

3. **Coarsest-peak**: argmax over $k$ of $C_8^{(k)}$.

**Consensus**: $\hat\theta^\ast = \tfrac{1}{3}(\theta_{\text{coll}} + \theta_\beta + \theta_{\text{peak}})$.

**Confidence**: $\text{conf} = \text{clip}\!\left(1 - \tfrac{\sigma(\hat\theta)}{0.15\,\Delta\theta_{\text{range}}}, 0, 1\right)$,
capped at $0.10$ when $\hat\theta$ lies within one grid step of either
endpoint of the swept range (edge-pinning guard).

**Known calibration limitation.** On synthetic sweeps with known truth,
reported confidence does NOT reliably correlate with accuracy (see
`detector_calibration.py`). Treat confidence as a weak indicator, not a
probability.

---

## 6. Canonical substrate recipe

When a new substrate is added, the pipeline is:

1. Produce a history tensor $X \in \mathcal{S}^{T \times G \times G}$ with
   state alphabet $\mathcal{S}$ appropriate to the substrate.
2. Discard the first $T_{\text{burn}}$ frames.
3. For each pool factor $s$, compute $\mathcal{C}_s(X)$ using the
   coarsening operator matched to $\mathcal{S}$.
4. Flatten each coarsened tensor to shape $(T_{\text{window}}, (G/s)^2)$
   and call `compute_C` (binary) or `compute_C_qary(q=|\mathcal{S}|)`
   as appropriate.
5. Average across seeds to get one row per parameter $\theta$.
6. (Optional) Pass all rows to `criticality_detector.estimate_critical`.

---

## 7. Version history

- **v9 (current binary)** — `complexity_framework_v9.py::compute_C`.
  Fixed as of 2026-04.
- **q-ary (2026-04)** — `compute_C_qary.py`. Reduces to binary at $q=2$
  for $w_H$, $w_{\text{OP},s}$, $w_{\text{OP},t}$. The $w_T$ and $w_G$
  weights are replaced with peaked tent functions (see §3.4–3.5), which
  differ in form from the v9 piecewise `weight_tcomp` / `weight_gzip`
  but are simpler and cleaner to reason about. Full numerical equivalence
  at $q=2$ is therefore *not* guaranteed; the two variants agree on the
  qualitative shape of $C(\theta)$ but absolute values may differ by
  a factor of $\sim$2.
- **Multi-scale** — introduced earlier; see `multiscale-c-summary.md` for
  the narrative. This doc is the formal reference.

---

## 8. What this document does NOT cover

- The exact piecewise forms of `weight_tcomp`, `weight_gzip`,
  `_opacity_both`, and `weight_fractal_dim` in the v9 binary pipeline.
  These are calibrated on Conway's Game of Life and their exact
  tent-function break-points live in source only. They are stable.
- Continuous-state $C$ (differential entropy, KSG MI). Not yet
  implemented.
- Graph-aware coarsening (for RBN / non-lattice). Not yet implemented.
