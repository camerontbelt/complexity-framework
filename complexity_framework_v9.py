"""
complexity_framework.py — v9
================================
Unified Complexity Measurement Framework
Updated: April 2026

Implements the eight candidate properties of complexity across five
simulation substrates using a single set of IC-independent metrics.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUBSTRATES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  eca    1D binary CA, radius-1 (256 Wolfram rules)
  k3     1D binary CA, radius-2 totalistic (32 rules)
  life   2D outer-totalistic CA (Life-like B/S rules)
  nbody  N-body particle simulation (α × αs scan)
  pd     Spatial Prisoner's Dilemma (temptation sweep)
           also: stag   Stag Hunt coordination game
                 minority  Minority anti-coordination game

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PD SUBSTRATE — P6 CO-EVOLUTIONARY DYNAMICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  The spatial Prisoner's Dilemma directly operationalises candidate
  property P6 (co-evolutionary dynamics).

  Setup (Nowak-May 1992):
    100×100 grid of agents, Moore neighbourhood (8 neighbours).
    Each agent plays PD simultaneously with all neighbours.
    Update: each agent copies the strategy of the highest-scoring
    neighbour (including self).  Payoffs: CC=R=1, CD=S=0, DC=T, DD=P=0.
    Initial condition: random 50% cooperators.

  Hypothesis (stated prior to running):
    Composite C will peak at T≈1.80, the Nowak-May critical temptation
    where mutual evolutionary pressure between cooperators and defectors
    produces propagating cooperation-defection fronts (glider-like dynamics).
    Below T=1.78: cooperation trivially dominates (C≈0).
    Above T=2.00: defection trivially dominates (C≈0).

  Confirmed result:
    Sharp transition at T=1.79→1.80.  Mean C=1.062 at T=1.80.
    90% of random ICs reach the complex attractor basin.
    Temporal opacity (w_OP_t) is the primary signal: rises from ~0
    across the cooperation-dominated regime to 0.93 at T=1.80.
    Exactly matches Nowak-May critical T — recovered from information-
    theoretic measurement alone, without knowledge of the payoff matrix.

  Control games:
    Stag Hunt    (coordination):    C=0.000 everywhere — trivially resolves
    Minority     (anti-coord.):     C≈0.001 everywhere — static checkerboard
    These controls confirm C is specific to co-evolutionary dynamics,
    not a general detector of spatial structure or phase transitions.

  Langton λ connection:
    r(λ, C) = -0.001 across 256 ECA rules — C is NOT rediscovering λ.
    All C4 rules sit at λ∈{0.375, 0.625} (adjacent to critical λ=0.5).
    λ locates the critical neighbourhood (necessary condition).
    C discriminates within it with 32× C4/C3 separation (sufficient).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGES FROM v8
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  + Spatial game substrate (PD, Stag Hunt, Minority game)
      pd_run()           — single game run, returns flat binary array
      pd_evaluate()      — metrics for one (game, T, seeds) combination
      pd_sweep()         — full parameter sweep over T∈[T_min, T_max]
      pd_controls()      — runs all three control games for comparison
      pd_langton_lambda()— correlates Langton λ with composite C on ECA
      PD_CFG             — default configuration dict
  + CLI: python complexity_framework.py pd [--T-min 1.0] [--T-max 2.0]
                                            [--game prisoner|stag|minority]
                                            [--fine-sweep] [--controls]
                                            [--csv results.csv]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UNIFIED OPACITY FRAMEWORK (v6 — key insight)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Opacity is a single property (P1 — opaque hierarchical layering)
  that exists along two orthogonal axes:

    SPATIAL opacity  — information hiding across scale at a moment in time
      opacity_up   = H(global | local)   upward:   local → global
      opacity_down = H(local  | global)  downward: global → local

    TEMPORAL opacity — information hiding across time at a point in space
      opacity_temp = I(past ; future) with decay
                   = tanh(MI₁) · tanh(1−MI₁) · tanh(decay)
      MI₁  = normalised mutual information between state at t and t+1
      decay = MI₁ − MI_max_lag  (how fast temporal correlation falls off)

  Together they capture whether a system hides information in BOTH
  space and time — the full P1 criterion.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
METRICS (all IC-independent, all parameter-free boundaries)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  mean_H / std_H   Spatial entropy         (P7)
  opacity_up       H(global | local)       (P1 spatial ↑)
  opacity_down     H(local  | global)      (P1 spatial ↓)
  opacity_temp     I(past ; future)+decay  (P1 temporal)
  tcomp            Temporal compression    (P4/P5 — retained)
  gzip             Kolmogorov proxy        (P2)

COMPOSITE  C = w_H × w_OP_spatial × w_OP_temporal × w_T × w_G
  Multiplicative — a rule must score well on ALL dimensions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PARAMETER-FREE BOUNDARIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  w_H:        tanh gates at H=0 (dead) and H=1 (noise) — info theory
  w_OP_up:    Gaussian — only remaining fitted parameter (mu=0.14)
  w_OP_down:  Gaussian — only remaining fitted parameter (mu=0.97)
  w_OP_temp:  tanh(MI)·tanh(1−MI)·tanh(decay) — fully parameter-free
              zeros at: random (MI→0), frozen (MI→1), no-decay
              peaks wherever complex systems actually sit
  w_T:        dual-Gaussian — retains tcomp IC-independence property
  w_G:        Gaussian — intermediate compressibility

  Temporal opacity has NO fitted parameters. Its peak location
  emerges from where complex systems sit on the MI-decay plane.
  This is the theoretical advance from v5: opacity_temp is the
  first fully parameter-free discriminating metric in the framework.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIMENSIONAL SCALING BEHAVIOUR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  In 1D: temporal MI across a binary CA field is near-zero for all
  rules — the field is too small for temporal correlations to build up.
  w_OP_temp ≈ 0 for all 1D classes. Because the composite is
  multiplicative this would collapse 1D scores entirely.
  Solution: w_OP_temp enters via geometric mean with the spatial
  composite, so it contributes without nullifying:
    C = (w_H × w_OP_s × w_T × w_G) × sqrt(1 + w_OP_temp)
  In 1D:  sqrt(1+0)   = 1.0  → no change to existing 1D results
  In 2D:  sqrt(1+0.93)= 1.39 → boosts C4 scores by ~39%
          sqrt(1+0.00)= 1.00 → no change for C2/C3
  This is the dimensional scaling: temporal opacity activates
  naturally when the substrate is rich enough to support it.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHANGES FROM v7
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  + weight_opacity_geometric(): 3D volume filter (Grok formulation)
      w_geom = tanh(k·min(op↑,op↓,op_t)) × (op↑·op↓·op_t)^{1/3}
      op_t = weight_opacity_temporal(mi1, decay)  — gated, not raw MI
      Derived from the 7D volume interpretation of C as a product of
      orthogonal opacity extents. Parameter-free (k=20 is gate sharpness).
      Validated finding: in 1D, w_geom does NOT replace Gaussian spatial
      weights — temporal opacity is near-zero for all 1D classes, so
      anomalous C2/C3 rules with intermediate MI outscore C4.
      Correct use: parallel 'score_geom' track for comparison and as
      primary opacity metric for 2D+ substrates.
  + score_geom field added to all result dicts (w_H × w_geom × w_T × w_G)
  + w_geom field added to all result dicts for inspection
  + Docstring: clarifies geometric filter scope and 1D limitation
  + All v7 results unchanged (composite() is unmodified)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONFIRMED RESULTS (W=150, random 50% IC, 5 seeds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1D ECA   C4/C3 sep=81×  F1=1.000  Cohen's d=38.4  |r|=1.000
           (temporal opacity near-zero in 1D — results unchanged)
  2D Life  TBD after v6 run
  N-body   TBD after v6 run

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python complexity_framework.py eca
  python complexity_framework.py eca --scale-test
  python complexity_framework.py eca --diagnose 110
  python complexity_framework.py eca --csv results.csv

  python complexity_framework.py k3
  python complexity_framework.py k3 --width 200

  python complexity_framework.py life
  python complexity_framework.py life --grid 64

  python complexity_framework.py nbody
  python complexity_framework.py nbody --scan --csv scan.csv
  python complexity_framework.py nbody --heatmap scan.json

  python complexity_framework.py all --csv all_results.csv

REQUIREMENTS
  pip install numpy matplotlib
  pip install scipy   # optional, only used for nbody live visualisation
"""

import numpy as np
import zlib, csv, json, os, sys, time, argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==============================================================================
# IC-INDEPENDENT WEIGHT FUNCTIONS
# ==============================================================================

def _gauss(x, mu, sig):
    return float(np.exp(-0.5 * ((x - mu) / sig) ** 2))


def weight_H(mean_H, std_H):
    """
    P7 Thermodynamic drive — spatial entropy attractor weight.

    tanh(50·H)·tanh(50·(1−H)) zeroes at H=0 (trivial order) and H=1
    (maximum chaos). Peak location is PARAMETER-FREE — it emerges from
    wherever the system sits on the entropy axis.

    × (1 + Gaussian(std_H, 0.012, 0.008)) rewards the entropy VARIANCE
    that distinguishes C4 from C3 under random IC:
      C4: std_H ≈ 0.013  (structures forming/dissolving — critical point)
      C3: std_H ≈ 0.010  (frozen near-maximal entropy)
    The variance bonus peak (0.012) was located by observing the C4 cluster
    and is the only fitted parameter in this weight.

    Max ≈ 2.0 at the edge-of-chaos attractor.
    """
    w_gates = float(np.tanh(50.0 * mean_H) * np.tanh(50.0 * (1.0 - mean_H)))
    w_var   = _gauss(std_H, 0.012, 0.008)
    return w_gates * (1.0 + w_var)


def weight_opacity_spatial(op_up, op_down):
    """
    P1 Opaque hierarchical layering — spatial opacity weight (v5/v6).

    Measures information hiding across spatial scale at a moment in time.
    Two complementary directions:

      opacity_up   = H(global density | local patch)
        Upward: given a local 3-cell window, how uncertain is the global
        state? C4~0.14, C3~0.33 (penalised), C1~0.00 (penalised).

      opacity_down = H(local patch | global density)
        Downward: given the global density, how underdetermined are local
        patterns? C4~0.97, C2~0.71 (penalised), C1~0.00 (penalised).

    Peaks calibrated to 1D ECA C4 cluster; held fixed across substrates.
    Note: opacity_down peak shifts by substrate (1D: 0.97, 2D: ~0.67).
    """
    w_up   = _gauss(op_up,   0.14, 0.10)
    w_down = _gauss(op_down, 0.97, 0.05)
    return w_up * w_down


# Keep old name as alias for backward compatibility
def weight_opacity(op_up, op_down):
    return weight_opacity_spatial(op_up, op_down)


def weight_opacity_temporal(mi1, decay, k=10):
    """
    P1 Opaque hierarchical layering — temporal opacity weight (v6, NEW).

    Measures information hiding across time: does the past predict the
    future in a structured, non-trivial way?

      MI₁   = I(X_t ; X_{t+1}) / H(X_t)   normalised mutual information
               at lag 1. How much does knowing now tell you about next step?

      decay = MI₁ − MI_k   where k = max_lag (default 10)
               How fast does temporal correlation fall off?

    Three parameter-free tanh gates — boundaries from information theory,
    not empirical curve fitting:

      tanh(k·MI₁):        NOT random   — MI→0 means no temporal structure
      tanh(k·(1−MI₁)):   NOT frozen   — MI→1 means trivially predictable
      tanh(k·decay):     HAS structure — decay→0 means no causal dynamics

    Only complex systems pass all three simultaneously:
      C3 (chaotic):  MI₁≈0.002 → gate 1 kills it  (no temporal structure)
      C2 (frozen):   MI₁≈1.000 → gate 2 kills it  (trivially predictable)
      C2 (periodic): decay≈0   → gate 3 kills it  (no causal dynamics)
      C4 (complex):  MI₁≈0.20-0.26, decay≈0.16-0.23 → all gates open

    Dimensional scaling: in 1D binary CAs the field MI is near-zero for
    all classes (~0.01-0.02) because single-cell binary states carry
    minimal mutual information. The metric activates naturally as
    substrate dimensionality increases — no substrate flag needed.

    Fully parameter-free: the peak location is not specified.
    k=10 controls gate sharpness (sigmoid steepness) only.
    """
    g1 = np.tanh(k * float(mi1))
    g2 = np.tanh(k * (1.0 - float(mi1)))
    g3 = np.tanh(k * max(float(decay), 0.0))
    return float(g1 * g2 * g3)


def weight_opacity_geometric(op_up, op_down, op_t_mi1, op_t_decay, k=20):
    """
    P1 Opaque hierarchical layering — fully parameter-free geometric filter (v8).

    Derived from the 7D volume interpretation of the composite score C.
    When C is viewed as a volume in opacity-vector space:

      vec_O = (op_up, op_down, op_t)

    complex systems occupy the interior of the positive octant — positive extent
    along ALL three opacity axes simultaneously. This motivates:

      w_geom = tanh(k · min(op_up, op_down, op_t)) × (op_up · op_down · op_t)^{1/3}

    where op_t = weight_opacity_temporal(mi1, decay) — the GATED temporal
    opacity weight, not the raw MI value.

    Why use gated op_t (not raw MI1)?
      This is Grok's formulation. op_t here is weight_opacity_temporal(mi1, decay),
      the tanh³ gated weight — not the raw MI value. The gate correctly zeroes
      frozen (MI→1) and chaotic (MI→0) systems, so any rule that fails temporal
      opacity gates collapses the 3D volume to zero.

    Important finding (validated on 256-rule ECA):
      In 1D, ALL classes have small temporal opacity (MI₁ ≈ 0.01-0.02) because
      binary field states carry minimal mutual information. This means w_OP_t ≈ 0.02
      for C4 AND C3, so the temporal gate provides little discrimination.
      Anomalous C2/C3 rules (e.g. 182, 183) have higher temporal MI (~0.16) and
      score HIGHER than C4 on the 3D geometric filter.

      CONCLUSION: w_geom(op_up, op_down, w_OP_t) does NOT replace the Gaussian
      spatial weights in 1D. It is a theoretically motivated alternative for 2D+
      substrates where temporal MI is large and substrate-discriminating.

    Use cases:
      - As a parallel 'geometric complexity score' for comparison
      - As a primary opacity metric for 2D+ substrates (N-body, 2D Life)
      - As a research tool to study the 7D volume geometry

    Returns w_geom in [0, ~0.8].
    """
    o_t       = weight_opacity_temporal(op_t_mi1, op_t_decay)
    min_extent = min(float(op_up), float(op_down), float(o_t))
    if min_extent <= 0.0:
        return 0.0
    balanced = (float(op_up) * float(op_down) * float(o_t)) ** (1.0 / 3.0)
    return float(np.tanh(k * min_extent) * balanced)


def weight_tcomp(mean_tc):
    """
    P4/P5 Temporal compression / selection pressure.

    Triple-Gaussian: max(G(tc,0.58,0.08), G(tc,0.73,0.08), G(tc,0.90,0.05))

    Three confirmed tcomp attractors across substrates:

      Peak 1 — 0.58 (σ=0.08): dynamic attractor, random IC
        1D C4 under random IC and 2D chaotic-complex rules land here.

      Peak 2 — 0.73 (σ=0.08): dynamic attractor, single-cell IC
        1D C4 under single-cell IC. Makes weight IC-independent in 1D.

      Peak 3 — 0.90 (σ=0.05): static-ecosystem attractor (v7, NEW)
        2D C4 rules (Conway, HighLife, Morley, Day&Night) converge here.
        Glider ecosystems form stable backgrounds — high tcomp.
        Confirmed IC-independent across density d=0.1–0.8.
        Narrower sigma because the 2D attractor is tighter.

    1D: C4 hits peaks 1 or 2 as before — peak 3 too high to affect them.
    2D: C4 hits peak 3. Frozen C2 (tc≈0.99) remains off all peaks.
    Fully backward-compatible with all existing 1D results.
    """
    return float(max(_gauss(mean_tc, 0.58, 0.08),
                     _gauss(mean_tc, 0.73, 0.08),
                     _gauss(mean_tc, 0.90, 0.05)))


def weight_gzip(mean_gz):
    """
    P2 Modular interconnection — Kolmogorov complexity proxy.

    Gaussian(gz, µ=0.10, σ=0.05) peaked at the C4 compression attractor.
    C4: gz ≈ 0.10–0.12 (large stable regions + complex glider boundaries)
    C3: gz ≈ 0.07–0.16 (varies by IC)
    C1: gz ≈ 0.008–0.02 (trivially compressible)

    The intermediate gzip regime encodes the hypothesis: complexity is
    neither trivially compressible (dead) nor incompressible (random).
    Peak value confirmed stable across W=100–400 and both IC types.
    """
    return _gauss(mean_gz, 0.10, 0.05)


def weight_fractal_dim(frac_dim, substrate_spatial_dim=2):
    """
    Dimensional complexity — fractal (box-counting) dimension weight (v7).

    Measures how much of the available spatial dimension the pattern
    actually uses. Complex systems occupy an intermediate fractal dimension
    — neither a low-dimensional line nor filling the full available space.

    excess = fractal_dim - (substrate_spatial_dim - 1)
      In 1D substrate (dim=1): patterns are 1D lines → excess ≈ 0
      In 2D substrate (dim=2): C4 rules → excess ≈ 0.3-0.4 (dim≈1.3-1.4)
                                C3/C2 space-filling → excess ≈ 0.9-1.0

    Weight: Gaussian(excess, µ=0.35, σ=0.20)
      Peak at excess=0.35 (fractal dim ≈ 1.35) — the 2D C4 cluster.
      Falls toward zero at excess=0 (1D line — no spatial structure)
      and at excess=1.0 (space-filling — chaos or frozen).

    Dimensional scaling:
      1D substrate: fractal_dim≈1.0, excess=0.0 → w_dim≈0.28 (small)
      2D C4: fractal_dim≈1.35, excess=0.35 → w_dim≈1.00 (strong)
      2D C3/C2: fractal_dim≈1.9, excess=0.9 → w_dim≈0.01 (near-zero)

    The weight is multiplicative — a system that fails to use spatial
    dimensions in a structured way is penalised regardless of other metrics.

    frac_dim:             box-counting fractal dimension of the late-time pattern
    substrate_spatial_dim: spatial dimensions of the substrate (1 for ECA, 2 for Life)
    """
    excess = float(np.clip(frac_dim - (substrate_spatial_dim - 1), 0.0, 1.0))
    return _gauss(excess, 0.35, 0.20)


def composite(mean_H, std_H, op_up, op_down, op_temp_mi1, op_temp_decay,
              mean_tc, mean_gz, frac_dim=None, substrate_spatial_dim=1):
    """
    C = w_H × (w_OP_s + w_OP_t) × w_T × w_G  [× w_dim if frac_dim provided]

    v7 architecture — three changes from v6:

    1. ADDITIVE opacity (your insight):
       (w_OP_spatial + w_OP_temporal) replaces w_OP_spatial × sqrt(1+w_OP_t)
       Logic: P1 opacity can manifest in EITHER space OR time — satisfying
       one direction is sufficient evidence of hierarchical layering.
       In 1D: spatial is strong, temporal near-zero → sum ≈ spatial (unchanged)
       In 2D: spatial zeroes (wrong peaks), temporal carries the composite
       OR logic for opacity channels; AND logic retained for all other properties.

    2. TRIPLE tcomp Gaussian (third peak at 0.90):
       Adds the 2D static-ecosystem attractor (Conway/HighLife/Morley/Day&Night).
       1D results unchanged — those rules still hit the 0.58/0.73 peaks.

    3. FRACTAL DIMENSION (optional multiplicative term):
       w_dim activates when frac_dim is provided. Near-zero in 1D (excess≈0),
       strong in 2D C4 (excess≈0.35), near-zero for space-filling C3/C2.
       When frac_dim=None the term is omitted — 1D experiments are unaffected.
    """
    w_OP_s = weight_opacity_spatial(op_up, op_down)
    w_OP_t = weight_opacity_temporal(op_temp_mi1, op_temp_decay)
    w_opacity = w_OP_s + w_OP_t          # additive: either channel suffices

    C = (weight_H(mean_H, std_H) *
         w_opacity *
         weight_tcomp(mean_tc) *
         weight_gzip(mean_gz))

    if frac_dim is not None:
        C *= weight_fractal_dim(frac_dim, substrate_spatial_dim)

    return float(C)


# ==============================================================================
# SHARED LOW-LEVEL METRICS
# (substrate-independent: work on any 2D array of shape (T, N_cells))
# ==============================================================================

def _entropy_stats(grid, burnin, window):
    """Mean and std of row-wise binary entropy over the measurement window."""
    post = grid[burnin:burnin + window]
    d = np.clip(post.mean(axis=1), 1e-12, 1 - 1e-12)
    H = -(d * np.log2(d) + (1 - d) * np.log2(1 - d))
    return float(H.mean()), float(H.std())


def _opacity_both(grid, burnin, window, n_bins=8):
    """
    Vectorised computation of both spatial opacity directions simultaneously.
    120× faster than the original per-cell Python loops.

    Encodes each 3-cell local patch as an integer 0-7 (left*4 + centre*2 + right).
    Builds the joint (patch, global_bin) distribution using numpy.add.at,
    then derives both conditional entropies from the same joint table.

      op_up   = H(global | local) = [H(patch,global) - H(patch)]   / log2(n_bins)
      op_down = H(local | global) = [H(patch,global) - H(global)]  / log2(8)
    """
    post  = grid[burnin:burnin + window]        # (T, W)
    T, W  = post.shape

    # Global density bin per time step
    dens  = post.mean(axis=1)                                          # (T,)
    gbins = np.clip((dens * n_bins).astype(int), 0, n_bins - 1)       # (T,)

    # 3-cell patch encoded as int 0-7
    left      = np.roll(post, 1,  axis=1)
    right     = np.roll(post, -1, axis=1)
    patch_int = (left * 4 + post * 2 + right).astype(np.int16)        # (T, W)

    # Repeat global bin for every cell in the row
    g_flat = np.repeat(gbins, W)                                       # (T*W,)
    p_flat = patch_int.ravel()                                         # (T*W,)

    # Joint count table: shape (8, n_bins)
    joint = np.zeros((8, n_bins), dtype=np.int64)
    np.add.at(joint, (p_flat, g_flat), 1)

    marg_p = joint.sum(axis=1)   # marginal over patches    (8,)
    marg_g = joint.sum(axis=0)   # marginal over global bins (n_bins,)
    total  = float(joint.sum())
    if total == 0:
        return 0.0, 0.0

    def _H(counts):
        c = counts[counts > 0].astype(float)
        p = c / c.sum()
        return float(-np.sum(p * np.log2(p)))

    H_joint = _H(joint.ravel())
    H_patch = _H(marg_p)
    H_glob  = _H(marg_g)

    op_up   = float(np.clip((H_joint - H_patch) / np.log2(n_bins), 0.0, 1.0))
    op_down = float(np.clip((H_joint - H_glob)  / np.log2(8),      0.0, 1.0))
    return op_up, op_down


# Thin wrappers kept for any external callers
def _opacity_upward(grid, burnin, window, n_bins=8, half=1):
    return _opacity_both(grid, burnin, window, n_bins)[0]

def _opacity_downward(grid, burnin, window, n_bins=8, half=1):
    return _opacity_both(grid, burnin, window, n_bins)[1]


def _tcomp(grid, burnin, window):
    """Mean temporal compression across all cells (scale-invariant)."""
    post  = grid[burnin:burnin + window]
    flips = np.sum(post[1:] != post[:-1], axis=0)
    return float(np.clip(1.0 - (1 + flips) / window, 0.0, 1.0).mean())


def _gzip(grid, burnin, window):
    """Gzip compression ratio of the spatiotemporal history."""
    raw = grid[burnin:burnin + window].tobytes()
    return len(zlib.compress(raw, 6)) / len(raw)


def _fractal_dim_2d(hist_3d, burnin, window, n_frames=5):
    """
    Box-counting (Minkowski-Bouligand) fractal dimension of the 2D spatial pattern.

    Averages over n_frames late-time frames (post-transient) for stability.
    Works on the 3D history array (T, H, W) — the unreflattened 2D frames.

    Returns fractal dimension in [0, 2]:
      ~1.0 = sparse 1D-like structures (glider streams)
      ~1.3-1.4 = intermediate (C4 target zone — Conway, HighLife)
      ~1.9-2.0 = space-filling (chaotic C3 or frozen C2)

    Only meaningful for 2D substrates. For 1D, pass None to composite()
    and the w_dim term is omitted.
    """
    T = hist_3d.shape[0]
    dims = []
    # Sample n_frames evenly spaced in the late measurement window
    t_start = burnin + window // 2
    t_end   = min(burnin + window, T)
    frame_indices = np.linspace(t_start, t_end - 1, n_frames, dtype=int)

    for t in frame_indices:
        frame = hist_3d[t]
        size  = frame.shape[0]
        if frame.sum() == 0:
            dims.append(0.0)
            continue
        sizes, counts = [], []
        box = size // 2
        while box >= 2:
            n   = size // box
            cnt = sum(1 for i in range(n) for j in range(n)
                      if frame[i*box:(i+1)*box, j*box:(j+1)*box].any())
            if cnt > 0:
                sizes.append(box)
                counts.append(cnt)
            box = box // 2
        if len(sizes) >= 3:
            log_s = np.log(1.0 / np.array(sizes, dtype=float))
            log_c = np.log(np.array(counts, dtype=float))
            slope, _ = np.polyfit(log_s, log_c, 1)
            dims.append(float(np.clip(slope, 0.0, 2.5)))

    return float(np.mean(dims)) if dims else 1.0


def _opacity_temporal(grid, burnin, window, max_lag=10, stride=3):
    """
    Temporal opacity: I(X_t ; X_{t+lag}) normalised by H(X_t).

    Measures how much temporal mutual information is present at lag=1
    and how fast it decays across lags 1..max_lag.

    Works on any (T, N_cells) grid — substrate-agnostic.

    Returns (mi1, decay) where:
      mi1   = normalised MI at lag 1
      decay = mi1 − mi_at_max_lag  (positive = decaying correlation)

    In 1D binary CAs: mi1 ≈ 0.01-0.02 for all classes (near-zero).
    In 2D CAs:        mi1 ≈ 0.20-0.26 for C4, ≈0.002 for C3, ≈1.0 for frozen.
    Dimensional scaling is therefore automatic — no substrate flag needed.
    """
    T, W = grid.shape

    def _mi_at_lag(lag):
        joint   = defaultdict(int)
        marg_t  = defaultdict(int)
        marg_tl = defaultdict(int)
        for t in range(burnin, burnin + window - lag):
            for x in range(0, W, stride):
                a = int(grid[t,     x])
                b = int(grid[t+lag, x])
                joint[(a, b)]  += 1
                marg_t[a]      += 1
                marg_tl[b]     += 1
        total = sum(joint.values())
        if total == 0: return 0.0
        Ht  = sum(-c/total * np.log2(c/total) for c in marg_t.values()  if c > 0)
        Htl = sum(-c/total * np.log2(c/total) for c in marg_tl.values() if c > 0)
        Hj  = sum(-c/total * np.log2(c/total) for c in joint.values()   if c > 0)
        MI  = Ht + Htl - Hj
        return float(np.clip(MI / max(Ht, 1e-9), 0.0, 1.0))

    mi1  = _mi_at_lag(1)
    mi_k = _mi_at_lag(max_lag)
    decay = float(np.clip(mi1 - mi_k, 0.0, 1.0))
    return mi1, decay


def _evaluate_grid(grid, burnin, window):
    """
    Compute all metrics and composite from a single grid run.
    Returns a dict of raw metric values, weights, and composite score.
    fractal_dim is not computed here (needs 3D hist) — set to None for 1D/k3.
    """
    mH, sH         = _entropy_stats(grid, burnin, window)
    op_up, op_down = _opacity_both(grid,  burnin, window)
    mi1, dec       = _opacity_temporal(grid, burnin, window)
    tc             = _tcomp(grid, burnin, window)
    gz             = _gzip(grid,  burnin, window)
    C              = composite(mH, sH, op_up, op_down, mi1, dec, tc, gz,
                               frac_dim=None)   # no fractal dim in 1D
    return dict(
        mean_H=mH, std_H=sH,
        opacity_up=op_up, opacity_down=op_down,
        opacity_temp_mi1=mi1, opacity_temp_decay=dec,
        tcomp=tc, gzip=gz, fractal_dim=None, score=C,
        w_H    = weight_H(mH, sH),
        w_OP_s = weight_opacity_spatial(op_up, op_down),
        w_OP_t = weight_opacity_temporal(mi1, dec),
        w_geom = weight_opacity_geometric(op_up, op_down, mi1, dec),
        w_T    = weight_tcomp(tc),
        w_G    = weight_gzip(gz),
        w_dim  = None,
    )


def _average_seeds(seed_results):
    """Average metric dicts across seeds, recompute composite."""
    keys = ['mean_H', 'std_H', 'opacity_up', 'opacity_down',
            'opacity_temp_mi1', 'opacity_temp_decay', 'tcomp', 'gzip']
    avgs = {k: float(np.mean([r[k] for r in seed_results])) for k in keys}

    # fractal_dim is None for 1D substrates, float for 2D
    fd_vals = [r.get('fractal_dim') for r in seed_results
               if r.get('fractal_dim') is not None]
    avgs['fractal_dim'] = float(np.mean(fd_vals)) if fd_vals else None

    # substrate_spatial_dim: infer from whether fractal_dim is present
    sdim = 2 if avgs['fractal_dim'] is not None else 1

    C = composite(avgs['mean_H'], avgs['std_H'],
                  avgs['opacity_up'], avgs['opacity_down'],
                  avgs['opacity_temp_mi1'], avgs['opacity_temp_decay'],
                  avgs['tcomp'], avgs['gzip'],
                  frac_dim=avgs['fractal_dim'],
                  substrate_spatial_dim=sdim)
    avgs['score']  = C
    avgs['w_H']    = weight_H(avgs['mean_H'], avgs['std_H'])
    avgs['w_OP_s'] = weight_opacity_spatial(avgs['opacity_up'], avgs['opacity_down'])
    avgs['w_OP_t'] = weight_opacity_temporal(avgs['opacity_temp_mi1'],
                                              avgs['opacity_temp_decay'])
    avgs['w_geom'] = weight_opacity_geometric(avgs['opacity_up'], avgs['opacity_down'],
                                               avgs['opacity_temp_mi1'],
                                               avgs['opacity_temp_decay'])
    avgs['w_T']    = weight_tcomp(avgs['tcomp'])
    avgs['w_G']    = weight_gzip(avgs['gzip'])
    avgs['w_dim']  = (weight_fractal_dim(avgs['fractal_dim'], sdim)
                      if avgs['fractal_dim'] is not None else None)
    # Geometric composite: replaces w_OP_s with w_geom in the product
    # w_OP_t already baked into w_geom, so use: w_H × w_geom × w_T × w_G
    avgs['score_geom'] = float(
        avgs['w_H'] * avgs['w_geom'] * avgs['w_T'] * avgs['w_G']
        * (weight_fractal_dim(avgs['fractal_dim'], sdim)
           if avgs['fractal_dim'] is not None else 1.0)
    )
    return avgs


# ==============================================================================
# SUBSTRATE 1 — ECA (1D binary CA, radius 1, 256 Wolfram rules)
# ==============================================================================

ECA_CLASS4 = {110, 124, 137, 193}
ECA_CLASS3 = {30, 45, 86, 89, 101, 105, 106, 150, 153, 165, 169, 182}
ECA_CLASS1 = {0, 8, 32, 40, 128, 136, 160, 168, 255}
ECA_CLASS2 = set(range(256)) - ECA_CLASS4 - ECA_CLASS3 - ECA_CLASS1

ECA_CFG = dict(WIDTH=150, STEPS=300, BURNIN=20, WINDOW=150,
               N_SEEDS=5, DENSITY=0.5)


def _eca_class(rule):
    if rule in ECA_CLASS4: return 'C4'
    if rule in ECA_CLASS3: return 'C3'
    if rule in ECA_CLASS1: return 'C1'
    if rule in ECA_CLASS2: return 'C2'
    return '-'


def _eca_run(rule_number, cfg, seed=42):
    lookup = np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.uint8)
    rng = np.random.RandomState(seed)
    W, S = cfg['WIDTH'], cfg['STEPS']
    grid = np.zeros((S, W), dtype=np.uint8)
    d = cfg.get('DENSITY', 0.5)
    if d is None:
        grid[0, W // 2] = 1
    else:
        grid[0] = (rng.rand(W) < d).astype(np.uint8)
    for t in range(1, S):
        row  = grid[t - 1]
        left = np.roll(row, 1); right = np.roll(row, -1)
        grid[t] = lookup[(left << 2) | (row << 1) | right]
    return grid


def eca_evaluate(rule_number, cfg):
    seeds  = [_evaluate_grid(_eca_run(rule_number, cfg, seed=s * 17 + 3),
                              cfg['BURNIN'], cfg['WINDOW'])
              for s in range(cfg['N_SEEDS'])]
    result = _average_seeds(seeds)
    result['rule']  = rule_number
    result['class'] = _eca_class(rule_number)
    return result


def eca_run_all(cfg, verbose=True):
    density  = cfg.get('DENSITY', 0.5)
    ic_label = 'single-cell' if density is None else f'random {density*100:.0f}%'
    if verbose:
        print(f"\nECA | 256 rules | IC: {ic_label} | "
              f"W={cfg['WIDTH']} steps={cfg['STEPS']} seeds={cfg['N_SEEDS']}")
    results = []; t0 = time.time()
    for rule in range(256):
        results.append(eca_evaluate(rule, cfg))
        if verbose and rule % 64 == 0:
            print(f"  {rule:3d}/255  ({time.time()-t0:.0f}s)")
    results.sort(key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(results): r['rank'] = i + 1
    return results


def eca_separation(results):
    c4 = [r['score'] for r in results if r['rule'] in ECA_CLASS4]
    c3 = [r['score'] for r in results if r['rule'] in ECA_CLASS3]
    return float(np.mean(c4)) / max(float(np.mean(c3)), 1e-12) if c4 and c3 else 0.0


def eca_scale_test(cfg, widths=None, csv_out=None):
    if widths is None: widths = [100, 150, 200, 300, 400]
    print(f"\n{'='*65}")
    print(f"  ECA scale sweep | IC: random {cfg.get('DENSITY',0.5)*100:.0f}%")
    print(f"{'='*65}")
    saved = {k: cfg[k] for k in ['WIDTH','STEPS','BURNIN','WINDOW']}
    rows  = []
    for W in widths:
        cfg['WIDTH']  = W
        cfg['STEPS']  = max(300, W * 2)
        cfg['BURNIN'] = 20
        cfg['WINDOW'] = min(150, cfg['STEPS'] - 30)
        t0  = time.time()
        res = eca_run_all(cfg, verbose=False)
        sep = eca_separation(res)
        rm  = {r['rule']: r['rank'] for r in res}
        c4r = sorted([rm[r] for r in ECA_CLASS4])
        r110 = next(r for r in res if r['rule'] == 110)
        print(f"  W={W:3d}: sep={sep:6.1f}× C4={c4r}  "
              f"R110=#{rm[110]:3d}  "
              f"op_up={r110['opacity_up']:.4f}  "
              f"op_dn={r110['opacity_down']:.4f}  "
              f"tc={r110['tcomp']:.4f}  gz={r110['gzip']:.4f}  "
              f"({time.time()-t0:.0f}s)")
        rows.append(dict(width=W, c4_c3_sep=round(sep,2), c4_ranks=str(c4r),
                         R110_rank=rm[110],
                         R110_op_up=round(r110['opacity_up'],4),
                         R110_op_dn=round(r110['opacity_down'],4),
                         R110_tc=round(r110['tcomp'],4),
                         R110_gz=round(r110['gzip'],4),
                         R110_wOP=round(r110['w_OP'],4)))
    for k, v in saved.items(): cfg[k] = v
    if csv_out: _save_csv(rows, csv_out)
    return rows


def eca_diagnose(rule_number, cfg):
    density  = cfg.get('DENSITY', 0.5)
    ic_label = 'single-cell' if density is None else f'random {density*100:.0f}%'
    cls      = _eca_class(rule_number)
    print(f"\nDiagnose ECA Rule {rule_number} [{cls}] IC:{ic_label} W={cfg['WIDTH']}")
    print('─' * 100)
    Cs = []
    for s in range(cfg['N_SEEDS']):
        r = _evaluate_grid(_eca_run(rule_number, cfg, seed=s*17+3),
                           cfg['BURNIN'], cfg['WINDOW'])
        Cs.append(r['score'])
        print(f"  seed {s*17+3:3d}: "
              f"H={r['mean_H']:.4f}±{r['std_H']:.4f}  "
              f"op↑={r['opacity_up']:.4f}  op↓={r['opacity_down']:.4f}  "
              f"tc={r['tcomp']:.4f}  gz={r['gzip']:.4f}  "
              f"wH={r['w_H']:.4f}  wOP={r['w_OP']:.4f}  "
              f"wT={r['w_T']:.4f}  wG={r['w_G']:.4f}  "
              f"C={r['score']:.6f}")
    print('─' * 100)
    print(f"  Mean C = {np.mean(Cs):.6f}  (std={np.std(Cs):.6f})")


def eca_print_results(results, label=''):
    sep     = eca_separation(results)
    c4_ranks = sorted([r['rank'] for r in results if r['rule'] in ECA_CLASS4])
    print(f"\n{'─'*100}")
    if label: print(f"  {label}")
    print(f"  C4/C3 separation: {sep:.1f}×    C4 ranks: {c4_ranks}")
    print(f"{'─'*100}")
    print(f"  {'Rk':>3} {'Rule':>4} {'Cls':>3} {'Score':>8}  "
          f"{'H':>5} {'sH':>5} {'op↑':>5} {'op↓':>5} "
          f"{'MI₁':>6} {'dec':>6} {'tc':>5} {'gz':>5}  "
          f"{'wH':>6} {'wOPs':>6} {'wOPt':>6} {'wT':>6} {'wG':>6}")
    for r in results[:20]:
        m = '*' if r['rule'] in ECA_CLASS4 else ' '
        print(f"  {m}{r['rank']:3d} {r['rule']:4d} {r['class']:>3} "
              f"{r['score']:8.4f}  "
              f"{r['mean_H']:5.3f} {r['std_H']:5.3f} "
              f"{r['opacity_up']:5.3f} {r['opacity_down']:5.3f} "
              f"{r['opacity_temp_mi1']:6.4f} {r['opacity_temp_decay']:6.4f} "
              f"{r['tcomp']:5.3f} {r['gzip']:5.3f}  "
              f"{r['w_H']:6.4f} {r['w_OP_s']:6.4f} {r['w_OP_t']:6.4f} "
              f"{r['w_T']:6.4f} {r['w_G']:6.4f}")


# ==============================================================================
# SUBSTRATE 2 — k=3 CA (1D binary, radius-2 totalistic, 32 rules)
# ==============================================================================

K3_COMPLEX = {22, 26}
K3_CHAOTIC = {6, 14, 18, 30}
K3_SIMPLE  = {0, 1, 31}

K3_CFG = dict(WIDTH=200, STEPS=500, BURNIN=50, WINDOW=200,
              N_SEEDS=5, DENSITY=0.5)


def _k3_class(rule):
    if rule in K3_COMPLEX: return 'C4'
    if rule in K3_CHAOTIC: return 'C3'
    if rule in K3_SIMPLE:  return 'C1'
    return '-'


def _k3_run(rule_number, cfg, seed=42):
    lookup = np.array([(rule_number >> i) & 1 for i in range(5)], dtype=np.uint8)
    rng    = np.random.RandomState(seed)
    W, S   = cfg['WIDTH'], cfg['STEPS']
    grid   = np.zeros((S, W), dtype=np.uint8)
    d = cfg.get('DENSITY', 0.5)
    if d is None:
        grid[0, W // 2] = 1
    else:
        grid[0] = (rng.rand(W) < d).astype(np.uint8)
    for t in range(1, S):
        row   = grid[t - 1]
        nbsum = (np.roll(row,2) + np.roll(row,1) + row +
                 np.roll(row,-1) + np.roll(row,-2))
        grid[t] = lookup[np.clip(nbsum, 0, 4)]
    return grid


def k3_evaluate(rule_number, cfg):
    seeds  = [_evaluate_grid(_k3_run(rule_number, cfg, seed=s*17+3),
                              cfg['BURNIN'], cfg['WINDOW'])
              for s in range(cfg['N_SEEDS'])]
    result = _average_seeds(seeds)
    result['rule']  = rule_number
    result['class'] = _k3_class(rule_number)
    return result


def k3_run_all(cfg, verbose=True):
    density = cfg.get('DENSITY', 0.5)
    if verbose:
        print(f"\nk=3 CA | 32 totalistic rules | IC: random {density*100:.0f}% | "
              f"W={cfg['WIDTH']} seeds={cfg['N_SEEDS']}")
    results = []
    for rule in range(32):
        results.append(k3_evaluate(rule, cfg))
    results.sort(key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(results): r['rank'] = i + 1
    if verbose: _print_substrate_top(results, 'k=3 CA', K3_COMPLEX, K3_CHAOTIC)
    return results


# ==============================================================================
# SUBSTRATE 3 — 2D Life-like CA (outer totalistic, B/S notation)
# ==============================================================================

LIFE_CFG = dict(GRID=64, STEPS=200, BURNIN=20, WINDOW=100,
                N_SEEDS=5, DENSITY=0.35)

# (name, birth_set, survive_set, class_label)
LIFE_RULES = [
    # C4 — complex / self-organising
    ('Conway',      frozenset([3]),         frozenset([2,3]),         'C4'),
    ('HighLife',    frozenset([3,6]),        frozenset([2,3]),         'C4'),
    ('Day&Night',   frozenset([3,6,7,8]),   frozenset([3,4,6,7,8]),   'C4'),
    ('Morley',      frozenset([3,6,8]),      frozenset([2,4,5]),       'C4'),
    # C3 — chaotic / explosive
    ('34Life',      frozenset([3,4]),        frozenset([3,4]),         'C3'),
    ('Amoeba',      frozenset([3,5,7]),      frozenset([1,3,5,8]),     'C3'),
    ('Anneal',      frozenset([4,6,7,8]),    frozenset([3,5,6,7,8]),   'C3'),
    ('Gnarl',       frozenset([1]),          frozenset([1]),           'C3'),
    ('Coagulations',frozenset([3,7,8]),      frozenset([2,3,5,6,7,8]),'C3'),
    # C2 — periodic / structured
    ('2x2',         frozenset([3,6]),        frozenset([1,2,5]),       'C2'),
    ('Seeds',       frozenset([2]),          frozenset([]),            'C2'),
    ('Replicator',  frozenset([1,3,5,7]),    frozenset([1,3,5,7]),     'C2'),
    ('Maze',        frozenset([3]),          frozenset([1,2,3,4,5]),   'C2'),
    ('Coral',       frozenset([3]),          frozenset([4,5,6,7,8]),   'C2'),
    # C1 — static / trivial
    ('Static',      frozenset([3]),          frozenset([2,3,4,5,6,7,8]),'C1'),
    ('Flakes',      frozenset([3]),          frozenset([0,1,2,3,4,5,6,7,8]),'C1'),
    ('AllDead',     frozenset([]),           frozenset([2,3]),         'C1'),
]


def _life_run(birth, survive, cfg, seed=42):
    rng   = np.random.RandomState(seed)
    G     = cfg['GRID']
    grid  = (rng.rand(G, G) < cfg['DENSITY']).astype(np.uint8)
    S     = cfg['STEPS']
    hist  = np.zeros((S, G, G), dtype=np.uint8)
    hist[0] = grid
    b_arr = np.zeros(9, dtype=bool); s_arr = np.zeros(9, dtype=bool)
    for b in birth:   b_arr[b] = True
    for s in survive: s_arr[s] = True
    for t in range(1, S):
        g  = hist[t-1].astype(np.int32)
        nb = np.zeros_like(g)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nb += np.roll(np.roll(g, dr, axis=0), dc, axis=1)
        hist[t] = ((g == 0) & b_arr[nb] | (g == 1) & s_arr[nb]).astype(np.uint8)
    flat = hist.reshape(S, G * G)
    return flat, hist


def _life_entropy(hist, burnin, window):
    post = hist[burnin:burnin+window]
    d    = np.clip(post.mean(axis=(1,2)), 1e-12, 1 - 1e-12)
    H    = -(d * np.log2(d) + (1 - d) * np.log2(1 - d))
    return float(H.mean()), float(H.std())


def life_evaluate(name, birth, survive, cfg):
    seeds_results = []
    for s in range(cfg['N_SEEDS']):
        flat, hist = _life_run(birth, survive, cfg, seed=s*17+3)
        mH, sH         = _life_entropy(hist, cfg['BURNIN'], cfg['WINDOW'])
        op_up, op_down = _opacity_both(flat, cfg['BURNIN'], cfg['WINDOW'])
        mi1, dec       = _opacity_temporal(flat, cfg['BURNIN'], cfg['WINDOW'])
        tc             = _tcomp(flat, cfg['BURNIN'], cfg['WINDOW'])
        gz             = _gzip(flat,  cfg['BURNIN'], cfg['WINDOW'])
        fd             = _fractal_dim_2d(hist, cfg['BURNIN'], cfg['WINDOW'])
        C              = composite(mH, sH, op_up, op_down, mi1, dec, tc, gz,
                                   frac_dim=fd, substrate_spatial_dim=2)
        seeds_results.append(dict(
            mean_H=mH, std_H=sH,
            opacity_up=op_up, opacity_down=op_down,
            opacity_temp_mi1=mi1, opacity_temp_decay=dec,
            tcomp=tc, gzip=gz, fractal_dim=fd, score=C,
            w_H   =weight_H(mH, sH),
            w_OP_s=weight_opacity_spatial(op_up, op_down),
            w_OP_t=weight_opacity_temporal(mi1, dec),
            w_T   =weight_tcomp(tc),
            w_G   =weight_gzip(gz),
            w_dim =weight_fractal_dim(fd, 2),
        ))
    result = _average_seeds(seeds_results)
    result['name'] = name
    return result


def life_run_all(cfg, verbose=True):
    if verbose:
        print(f"\n2D Life-like CA | {len(LIFE_RULES)} rules | "
              f"grid={cfg['GRID']}×{cfg['GRID']}  "
              f"density={cfg['DENSITY']}  seeds={cfg['N_SEEDS']}")
    results = []
    for name, birth, survive, cls_label in LIFE_RULES:
        r = life_evaluate(name, birth, survive, cfg)
        r['class'] = cls_label
        results.append(r)
        if verbose:
            fd  = r.get('fractal_dim', 0) or 0
            wd  = r.get('w_dim', 0) or 0
            print(f"  {name:14s} [{cls_label}]  C={r['score']:.4f}  "
                  f"H={r['mean_H']:.3f}  "
                  f"op↑={r['opacity_up']:.3f}  op↓={r['opacity_down']:.3f}  "
                  f"MI₁={r['opacity_temp_mi1']:.3f}  wOPt={r['w_OP_t']:.3f}  "
                  f"tc={r['tcomp']:.3f}  fd={fd:.3f}  wD={wd:.3f}")
    results.sort(key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(results): r['rank'] = i + 1
    if verbose:
        c4 = [r for r in results if r['class'] == 'C4']
        print(f"\n  C4 rules: {[r['name'] for r in c4]}")
        print(f"  C4 ranks: {[r['rank'] for r in c4]}")
    return results


# ==============================================================================
# SUBSTRATE 4 — N-body particle simulation
# ==============================================================================

NBODY_CFG = dict(
    N=80, BOX=18.0, DT=0.025, DAMPING=0.998, FORCE_CLAMP=25.0,
    SCAN_N_ALPHA=16,  SCAN_N_ALPHAS=16,
    SCAN_ALPHA_MIN=0.1,  SCAN_ALPHA_MAX=5.0,
    SCAN_ALPHAS_MIN=0.3, SCAN_ALPHAS_MAX=3.2,
    SCAN_STEPS=200, SCAN_SKIP=40, SCAN_SAMPLE=5,
    SCAN_SEEDS=[42, 123, 7],
)


def _nbody_init(N, box, seed=42):
    np.random.seed(seed)
    return np.random.rand(N, 2) * box, np.random.randn(N, 2) * 0.35


def _nbody_forces(pos, alpha, alpha_s, box, clamp):
    dx = pos[:,None,0] - pos[None,:,0]
    dy = pos[:,None,1] - pos[None,:,1]
    dx -= box * np.round(dx / box)
    dy -= box * np.round(dy / box)
    r2   = np.maximum(dx**2 + dy**2, 0.04)
    mask = r2 < (alpha_s * 3.5)**2
    np.fill_diagonal(mask, False)
    sr2  = np.where(mask, (alpha_s**2) / r2, 0)
    sr6  = sr2**3; sr12 = sr6**2
    fm   = np.where(mask, 24 * alpha * (2*sr12 - sr6) / r2, 0)
    np.fill_diagonal(fm, 0)
    return (np.clip((fm * dx).sum(1), -clamp, clamp),
            np.clip((fm * dy).sum(1), -clamp, clamp))


def _nbody_step(pos, vel, alpha, alpha_s, cfg):
    box = cfg['BOX']; dt = cfg['DT']
    damp = cfg['DAMPING']; cl = cfg['FORCE_CLAMP']
    fx, fy = _nbody_forces(pos, alpha, alpha_s, box, cl)
    vel[:,0] += 0.5*fx*dt; vel[:,1] += 0.5*fy*dt
    pos = (pos + vel * dt) % box
    fx2, fy2 = _nbody_forces(pos, alpha, alpha_s, box, cl)
    vel[:,0] += 0.5*fx2*dt; vel[:,1] += 0.5*fy2*dt
    vel *= damp
    return pos, vel


def _nbody_run(alpha, alpha_s, cfg, seed=42):
    pos, vel = _nbody_init(cfg['N'], cfg['BOX'], seed=seed)
    frames   = []
    for s in range(cfg['SCAN_STEPS']):
        pos, vel = _nbody_step(pos, vel, alpha, alpha_s, cfg)
        if s >= cfg['SCAN_SKIP'] and s % cfg['SCAN_SAMPLE'] == 0:
            frames.append(pos.copy())
    return frames


def _nbody_metrics(frames, cfg):
    """
    Map N-body simulation frames to the four metrics via spatial quantisation.
    Particle positions are discretised to binary occupancy grids, then the
    standard CA metrics (including bidirectional opacity) are applied.
    Same weight functions, same composite — substrate-agnostic by design.
    """
    if len(frames) < 4:
        return dict(mean_H=0, std_H=0, opacity_up=0, opacity_down=0,
                    tcomp=0, gzip=0, score=0,
                    w_H=0, w_OP=0, w_T=0, w_G=0)
    box  = cfg['BOX']; bins = 20; T = len(frames)
    occ  = np.zeros((T, bins*bins), dtype=np.uint8)
    for t, pos in enumerate(frames):
        gx = np.clip((pos[:,0] / box * bins).astype(int), 0, bins-1)
        gy = np.clip((pos[:,1] / box * bins).astype(int), 0, bins-1)
        for x, y in zip(gx, gy):
            occ[t, x*bins+y] = 1
    burnin = 0; window = T
    mH, sH    = _entropy_stats(occ, burnin, window)
    op_up     = _opacity_upward(occ,   burnin, window)
    op_down   = _opacity_downward(occ, burnin, window)
    mi1, dec  = _opacity_temporal(occ, burnin, window)
    tc        = _tcomp(occ, burnin, window)
    gz        = _gzip(occ,  burnin, window)
    C         = composite(mH, sH, op_up, op_down, mi1, dec, tc, gz)
    return dict(mean_H=mH, std_H=sH,
                opacity_up=op_up, opacity_down=op_down,
                opacity_temp_mi1=mi1, opacity_temp_decay=dec,
                tcomp=tc, gzip=gz, score=C,
                w_H   =weight_H(mH,sH),
                w_OP_s=weight_opacity_spatial(op_up, op_down),
                w_OP_t=weight_opacity_temporal(mi1, dec),
                w_T   =weight_tcomp(tc),
                w_G   =weight_gzip(gz))


def nbody_scan(cfg, csv_out=None, json_out=None, verbose=True):
    N_A  = cfg['SCAN_N_ALPHA']; N_AS = cfg['SCAN_N_ALPHAS']
    AR   = np.logspace(np.log10(cfg['SCAN_ALPHA_MIN']),
                       np.log10(cfg['SCAN_ALPHA_MAX']), N_A)
    ASR  = np.logspace(np.log10(cfg['SCAN_ALPHAS_MIN']),
                       np.log10(cfg['SCAN_ALPHAS_MAX']), N_AS)
    total = N_A * N_AS
    if verbose:
        print(f"\nN-body scan | {N_A}×{N_AS}={total} points | "
              f"seeds={cfg['SCAN_SEEDS']}")
    cmap = np.zeros((N_A, N_AS)); rows = []
    t0   = time.time()
    for i, alpha in enumerate(AR):
        for j, alpha_s in enumerate(ASR):
            seed_m = [_nbody_metrics(_nbody_run(alpha, alpha_s, cfg, s), cfg)
                      for s in cfg['SCAN_SEEDS']]
            avg = {k: float(np.mean([m[k] for m in seed_m])) for k in seed_m[0]}
            cmap[i, j] = avg['score']
            rows.append(dict(alpha=round(alpha, 4), alpha_s=round(alpha_s, 4),
                             **{k: round(v, 6) for k, v in avg.items()}))
            if verbose:
                done = i*N_AS + j + 1; pct = done/total*100
                bar  = '█'*int(pct//5) + '░'*(20-int(pct//5))
                print(f"\r  [{bar}] {pct:.0f}%  α={alpha:.3f} αs={alpha_s:.3f}  "
                      f"C={avg['score']:.4f}", end='', flush=True)
    if verbose:
        print()
        pk  = np.unravel_index(np.argmax(cmap), cmap.shape)
        oi  = np.argmin(np.abs(AR  - 1.0))
        oj  = np.argmin(np.abs(ASR - 1.0))
        pct = float(np.mean(cmap < cmap[oi, oj]) * 100)
        print(f"  Peak        : α={AR[pk[0]]:.3f}  αs={ASR[pk[1]]:.3f}  "
              f"C={cmap.max():.4f}")
        print(f"  Our universe: C={cmap[oi,oj]:.4f}  ({pct:.0f}th percentile)")
        print(f"  Elapsed     : {time.time()-t0:.0f}s")
    if csv_out:  _save_csv(rows, csv_out)
    if json_out:
        pk = np.unravel_index(np.argmax(cmap), cmap.shape)
        oi = np.argmin(np.abs(AR - 1.0)); oj = np.argmin(np.abs(ASR - 1.0))
        with open(json_out, 'w') as f:
            json.dump({
                'alpha_range':   AR.tolist(),
                'alphas_range':  ASR.tolist(),
                'complexity_map': cmap.tolist(),
                'peak':  {'alpha': float(AR[pk[0]]), 'alpha_s': float(ASR[pk[1]]),
                           'score': float(cmap.max())},
                'our_universe': {'score': float(cmap[oi,oj]),
                                  'percentile': float(np.mean(cmap<cmap[oi,oj])*100)},
            }, f, indent=2)
        print(f"  Saved → {json_out}")
    return rows, cmap, AR, ASR


# ==============================================================================
# SHARED OUTPUT HELPERS
# ==============================================================================

def _save_csv(rows, filename):
    if not rows: return
    with open(filename, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 6) if isinstance(v, float) else v
                        for k, v in r.items()})
    print(f"  Saved → {filename}")


def _print_substrate_top(results, label, complex_set, chaotic_set):
    c4s = [r['score'] for r in results
           if r.get('rule') in complex_set or r.get('name') in complex_set]
    c3s = [r['score'] for r in results
           if r.get('rule') in chaotic_set or r.get('name') in chaotic_set]
    sep = float(np.mean(c4s) / np.mean(c3s)) if c4s and c3s else 0.0
    print(f"\n  {label}  C4/C3 sep={sep:.1f}×")
    for r in results[:10]:
        name = r.get('name') or f"Rule {r.get('rule','?')}"
        print(f"  Rank {r['rank']:3d}  {name:16s} [{r.get('class','-'):>3}]  "
              f"C={r['score']:.4f}  "
              f"H={r['mean_H']:.3f}  op↑={r['opacity_up']:.3f}  "
              f"op↓={r['opacity_down']:.3f}  "
              f"tc={r['tcomp']:.3f}  gz={r['gzip']:.3f}")



# ==============================================================================
# SPATIAL GAME THEORY SUBSTRATE — P6 CO-EVOLUTIONARY DYNAMICS
# ==============================================================================

# ── Configuration ──────────────────────────────────────────────────────────────

PD_CFG = {
    'GRID':       100,    # grid edge length (GRID × GRID agents)
    'STEPS':      250,    # total simulation steps
    'BURNIN':     50,     # steps discarded before measurement
    'WINDOW':     150,    # measurement window length
    'N_SEEDS':    10,     # random initial conditions per T value
    'IC_DENSITY': 0.5,    # initial cooperator fraction
    # Payoff matrix constants (S and P fixed; T is the swept parameter)
    'R': 1.0,             # reward for mutual cooperation
    'S': 0.0,             # sucker's payoff (cooperator vs defector)
    'P': 0.0,             # punishment for mutual defection
    # Sweep range
    'T_MIN':  1.0,        # temptation sweep start
    'T_MAX':  2.0,        # temptation sweep end
    'T_STEP': 0.05,       # coarse step
    'T_FINE_MIN':  1.70,  # fine sweep start
    'T_FINE_MAX':  1.95,  # fine sweep end
    'T_FINE_STEP': 0.01,  # fine step
    # Classification threshold
    'COMPLEX_THRESHOLD': 0.1,  # C > this → complex attractor basin
}


# ── Payoff factories ────────────────────────────────────────────────────────────

def _pd_payoff(grid, nb, T, R=1.0, S=0.0, P=0.0):
    """
    Prisoner's Dilemma payoff for each cell given neighbour state.
    grid, nb: float arrays in {0, 1} (0=defect, 1=cooperate).
    Returns payoff array (same shape as grid).
    """
    return (grid * (nb * R + (1 - nb) * S) +
            (1 - grid) * (nb * T + (1 - nb) * P))


def _stag_payoff(grid, nb, alpha=0.5):
    """
    Stag Hunt (coordination game).
    CC=1, CD=0 (sucker), DC=alpha<1 (hare), DD=0.
    Coordination is beneficial; defection is safe but suboptimal.
    Prediction: trivially resolves to full cooperation on grid. C≈0.
    """
    return (grid * (nb * 1.0 + (1 - nb) * 0.0) +
            (1 - grid) * (nb * alpha + (1 - nb) * 0.0))


def _minority_payoff(grid, nb, beta=1.0):
    """
    Minority game (anti-coordination).
    Reward for being the minority strategy among neighbours.
    Prediction: produces static checkerboard. C≈0.
    """
    return beta * (grid * (1 - nb) + (1 - grid) * nb)


# ── Simulation runner ───────────────────────────────────────────────────────────

def pd_run(payoff_fn, cfg, seed=42):
    """
    Run a spatial game for one seed.

    Parameters
    ----------
    payoff_fn : callable(grid, nb) → payoff_array
        Vectorised payoff function; called once per Moore neighbour.
    cfg : dict
        Must contain GRID, STEPS, IC_DENSITY.
    seed : int

    Returns
    -------
    np.ndarray, shape (STEPS, GRID*GRID), dtype uint8
        Binary cooperator (1) / defector (0) grid history, flattened.
    """
    G = cfg['GRID']
    rng = np.random.RandomState(seed)
    grid = (rng.rand(G, G) < cfg['IC_DENSITY']).astype(np.float32)

    history = []
    for _ in range(cfg['STEPS']):
        history.append(grid.copy().flatten().astype(np.uint8))

        # Accumulate payoffs from all 8 Moore neighbours + self
        payoffs = np.zeros((G, G), dtype=np.float32)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                nb = np.roll(np.roll(grid, di, axis=0), dj, axis=1)
                payoffs += payoff_fn(grid, nb)

        # Each cell adopts the strategy of its highest-scoring neighbour
        best_pay = payoffs.copy()
        best_str = grid.copy()
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                nb_pay = np.roll(np.roll(payoffs, di, axis=0), dj, axis=1)
                nb_str = np.roll(np.roll(grid,    di, axis=0), dj, axis=1)
                mask = nb_pay > best_pay
                best_pay = np.where(mask, nb_pay, best_pay)
                best_str = np.where(mask, nb_str, best_str)
        grid = best_str

    return np.array(history)  # (STEPS, G*G)


# ── Metrics extractor ───────────────────────────────────────────────────────────

def _pd_metrics(flat, cfg):
    """
    Apply the standard v8 metric pipeline to a flat (STEPS, W) binary array.
    Uses the same weight functions as all other substrates — no PD-specific
    modifications.

    Returns dict with all raw metrics and composite C.
    Returns None if insufficient data.
    """
    burnin = cfg['BURNIN']
    window = cfg['WINDOW']
    post   = flat[burnin:burnin + window]
    T2, W  = post.shape
    if T2 < 10:
        return None

    # Entropy stats
    d = np.clip(post.mean(axis=1), 1e-12, 1 - 1e-12)
    H_rows = -(d * np.log2(d) + (1 - d) * np.log2(1 - d))
    mean_H = float(H_rows.mean())
    std_H  = float(H_rows.std())

    # Spatial opacity (vectorised joint entropy)
    left  = np.roll(post, 1, axis=1)
    right = np.roll(post, -1, axis=1)
    p_int = (left * 4 + post * 2 + right).astype(np.int16)
    gbins = np.clip((post.mean(axis=1) * 8).astype(int), 0, 7)
    joint = np.zeros((8, 8), dtype=np.int64)
    np.add.at(joint, (p_int.ravel(), np.repeat(gbins, W)), 1)
    marg_p = joint.sum(axis=1)
    marg_g = joint.sum(axis=0)

    def _H(counts):
        c = counts[counts > 0].astype(float)
        p = c / c.sum()
        return float(-np.sum(p * np.log2(p)))

    if joint.sum() == 0:
        return None
    op_up = float(np.clip((_H(joint.ravel()) - _H(marg_p)) / np.log2(8), 0, 1))
    op_dn = float(np.clip((_H(joint.ravel()) - _H(marg_g)) / np.log2(8), 0, 1))

    # Temporal opacity
    mi1, decay = _opacity_temporal(post, 0, T2, max_lag=10)

    # Temporal compression
    flips = np.sum(post[1:] != post[:-1], axis=0)
    tc    = float(np.clip(1 - (1 + flips) / T2, 0, 1).mean())

    # Gzip
    gz = len(zlib.compress(post.tobytes(), 6)) / len(post.tobytes())

    # Composite weights
    w_H   = weight_H(mean_H, std_H)
    w_OPs = weight_opacity_spatial(op_up, op_dn)
    w_OPt = weight_opacity_temporal(mi1, decay)
    w_T   = weight_tcomp(tc)
    w_G   = weight_gzip(gz)
    C     = w_H * (w_OPs + w_OPt) * w_T * w_G

    return dict(
        mean_H=mean_H, std_H=std_H,
        op_up=op_up, op_dn=op_dn,
        mi1=mi1, decay=decay,
        tcomp=tc, gzip=gz,
        w_H=w_H, w_OPs=w_OPs, w_OPt=w_OPt, w_T=w_T, w_G=w_G,
        score=C,
        coop=float(post.mean()),
    )


# ── Single parameter-point evaluator ───────────────────────────────────────────

def pd_evaluate(payoff_fn, param_value, param_name, cfg, verbose=False):
    """
    Evaluate the composite C for one parameter value, averaged over N_SEEDS.

    Returns
    -------
    dict with param, mean metrics, std_C, p_complex
    """
    seed_results = []
    for seed in range(cfg['N_SEEDS']):
        flat = pd_run(payoff_fn, cfg, seed=seed * 7 + 3)
        m    = _pd_metrics(flat, cfg)
        if m:
            seed_results.append(m)

    if not seed_results:
        return None

    avg = {k: float(np.mean([r[k] for r in seed_results]))
           for k in seed_results[0]}
    std_C     = float(np.std([r['score'] for r in seed_results]))
    p_complex = sum(1 for r in seed_results
                    if r['score'] > cfg['COMPLEX_THRESHOLD']) / len(seed_results)

    result = {param_name: round(float(param_value), 4),
              **{k: round(v, 5) for k, v in avg.items()},
              'std_C':     round(std_C, 5),
              'p_complex': round(p_complex, 3)}

    if verbose:
        coop = avg['coop']
        regime = ('full cooperation' if coop > 0.92 else
                  'partial coop'     if coop > 0.55 else
                  'edge / mixed  ← INTERESTING' if coop > 0.08 else
                  'full defection')
        print(f"  {param_name}={param_value:5.2f}  "
              f"C={avg['score']:8.5f}±{std_C:.4f}  "
              f"P(cx)={p_complex:.1f}  "
              f"coop={coop:.3f}  wOPt={avg['w_OPt']:.4f}  {regime}")

    return result


# ── Full parameter sweep ────────────────────────────────────────────────────────

def pd_sweep(cfg, fine=False, csv_out=None, verbose=True):
    """
    Sweep the temptation parameter T across [T_MIN, T_MAX].

    Parameters
    ----------
    cfg   : dict (PD_CFG or override)
    fine  : bool — if True, also run the fine sweep around T=1.70-1.95
    csv_out : str or None

    Returns
    -------
    list of dicts, one per T value
    """
    R, S, P = cfg['R'], cfg['S'], cfg['P']

    if fine:
        T_values = np.round(np.arange(cfg['T_FINE_MIN'],
                                      cfg['T_FINE_MAX'] + 1e-9,
                                      cfg['T_FINE_STEP']), 3)
    else:
        T_values = np.round(np.arange(cfg['T_MIN'],
                                      cfg['T_MAX'] + 1e-9,
                                      cfg['T_STEP']), 3)

    if verbose:
        scope = 'fine' if fine else 'coarse'
        print(f"\n{'='*65}")
        print(f"  Spatial Prisoner's Dilemma — {scope} T sweep")
        print(f"  Grid: {cfg['GRID']}×{cfg['GRID']}  "
              f"Steps: {cfg['STEPS']}  Seeds: {cfg['N_SEEDS']}")
        print(f"{'='*65}")
        print(f"  {'T':>5}  {'C':>10}  {'±':>6}  "
              f"{'P(cx)':>6}  {'coop':>6}  {'wOPt':>7}  regime")
        print(f"  {'─'*65}")

    rows = []
    for T in T_values:
        def pf(g, nb, _T=float(T)): return _pd_payoff(g, nb, _T, R, S, P)
        row = pd_evaluate(pf, T, 'T', cfg, verbose=verbose)
        if row:
            rows.append(row)

    if rows and verbose:
        peak = max(rows, key=lambda r: r['score'])
        print(f"\n  Peak C={peak['score']:.5f} ± {peak['std_C']:.4f} "
              f"at T={peak['T']:.2f}  P(cx)={peak['p_complex']}")
        print(f"  Known Nowak-May critical T ≈ 1.80  "
              f"offset={abs(peak['T'] - 1.80):.2f}")

    if csv_out:
        _save_csv(rows, csv_out)

    return rows


# ── Control game sweeps ─────────────────────────────────────────────────────────

def pd_controls(cfg, csv_out=None, verbose=True):
    """
    Run the three control games and return a combined results list.

    Control 1 — Stag Hunt (coordination game)
      Prediction: C=0 everywhere. Grid resolves to ordered cooperation.
      Answer: does C peak on any spatial ordering? No.

    Control 2 — Minority game (anti-coordination)
      Prediction: C≈0 everywhere. Produces static checkerboard.
      Answer: does C peak on any non-trivial equilibrium? No.

    Returns
    -------
    list of dicts with 'game' field identifying each control
    """
    all_rows = []

    # ── Stag Hunt ─────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*65}")
        print(f"  CONTROL 1: Stag Hunt (coordination game)")
        print(f"  Prediction: C=0 everywhere — grid resolves to full cooperation")
        print(f"{'='*65}")
        print(f"  {'alpha':>6}  {'C':>10}  {'coop':>6}  regime")
        print(f"  {'─'*40}")

    for alpha in np.round(np.arange(0.1, 1.05, 0.2), 2):
        def pf(g, nb, a=float(alpha)): return _stag_payoff(g, nb, a)
        row = pd_evaluate(pf, alpha, 'alpha', cfg, verbose=False)
        if row:
            row['game'] = 'stag_hunt'
            if verbose:
                coop = row['coop']
                regime = 'coop wins' if coop > 0.9 else 'mixed' if coop > 0.1 else 'defect'
                print(f"  {alpha:6.2f}  {row['score']:10.5f}  {coop:6.3f}  {regime}")
            all_rows.append(row)

    # ── Minority game ──────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*65}")
        print(f"  CONTROL 2: Minority game (anti-coordination)")
        print(f"  Prediction: C≈0 — produces static checkerboard, no dynamics")
        print(f"{'='*65}")
        print(f"  {'beta':>5}  {'C':>10}  {'coop':>6}  regime")
        print(f"  {'─'*40}")

    for beta in np.round(np.arange(0.2, 1.25, 0.2), 2):
        def pf(g, nb, b=float(beta)): return _minority_payoff(g, nb, b)
        row = pd_evaluate(pf, beta, 'beta', cfg, verbose=False)
        if row:
            row['game'] = 'minority'
            if verbose:
                coop = row['coop']
                regime = 'checkerboard' if abs(coop - 0.5) < 0.15 else 'other'
                print(f"  {beta:5.2f}  {row['score']:10.5f}  {coop:6.3f}  {regime}")
            all_rows.append(row)

    # ── Summary ────────────────────────────────────────────────────
    if verbose and all_rows:
        sh_peak = max((r for r in all_rows if r['game'] == 'stag_hunt'),
                      key=lambda r: r['score'], default=None)
        mn_peak = max((r for r in all_rows if r['game'] == 'minority'),
                      key=lambda r: r['score'], default=None)
        print(f"\n  CONTROL SUMMARY")
        print(f"  {'Game':24s}  {'Peak C':>8}  {'interpretation'}")
        print(f"  {'─'*60}")
        if sh_peak:
            print(f"  {'Stag Hunt':24s}  {sh_peak['score']:8.5f}  "
                  f"trivial — spatial coordination")
        if mn_peak:
            print(f"  {'Minority game':24s}  {mn_peak['score']:8.5f}  "
                  f"checkerboard — no temporal dynamics")
        print(f"  (Compare: PD peak C ≈ 1.062 at T=1.80)")

    if csv_out:
        _save_csv(all_rows, csv_out)

    return all_rows


# ── Langton λ analysis ──────────────────────────────────────────────────────────

def pd_langton_lambda(eca_results, verbose=True):
    """
    Correlate Langton's λ with composite C across all 256 ECA rules.

    λ = fraction of rule-table entries that map to state 1.

    Key finding: r(λ, C) ≈ 0 — C is not rediscovering λ.
    All four C4 rules sit at λ∈{0.375, 0.625}, adjacent to the
    critical point λ=0.5 but not at it.  λ is a necessary condition
    for complex dynamics; C provides the sufficient condition.

    Parameters
    ----------
    eca_results : list of dicts from eca_run_all()

    Returns
    -------
    dict with correlation and per-λ statistics
    """
    def langton(rule_int):
        return bin(int(rule_int)).count('1') / 8.0

    scores  = np.array([r['score']  for r in eca_results])
    lambdas = np.array([langton(r['rule']) for r in eca_results])
    classes = [r.get('class', '-') for r in eca_results]

    corr = float(np.corrcoef(lambdas, scores)[0, 1])

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Langton λ vs Composite C")
        print(f"{'='*60}")
        print(f"  Pearson r(λ, C) = {corr:+.4f}  (≈0 means C ≠ λ)")
        print()
        print(f"  {'λ':>6}  {'n_rules':>8}  {'C4':>5}  {'C3':>5}  "
              f"{'mean_C':>8}  {'max_C':>8}")
        print(f"  {'─'*55}")

    per_lambda = {}
    for lam_val in [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]:
        idx = [i for i, r in enumerate(eca_results)
               if abs(langton(r['rule']) - lam_val) < 0.001]
        if not idx:
            continue
        lam_scores  = scores[idx]
        lam_classes = [classes[i] for i in idx]
        n4 = lam_classes.count('C4')
        n3 = lam_classes.count('C3')
        mc = float(lam_scores.mean())
        mx = float(lam_scores.max())
        per_lambda[lam_val] = dict(n=len(idx), n4=n4, n3=n3, mean_C=mc, max_C=mx)
        if verbose:
            print(f"  {lam_val:6.3f}  {len(idx):8d}  {n4:5d}  {n3:5d}  "
                  f"{mc:8.5f}  {mx:8.5f}")

    # Within-band discrimination
    band = [(r, langton(r['rule'])) for r in eca_results
            if 0.3 < langton(r['rule']) < 0.7]
    c4_band = [r['score'] for r, _ in band if r.get('class') == 'C4']
    c3_band = [r['score'] for r, _ in band if r.get('class') == 'C3']
    sep = float(np.mean(c4_band) / np.mean(c3_band)) if c4_band and c3_band else 0.0

    if verbose:
        print(f"\n  Within λ∈[0.375,0.625] band ({len(band)} rules):")
        print(f"    C4 mean C = {np.mean(c4_band):.5f}")
        print(f"    C3 mean C = {np.mean(c3_band):.5f}")
        print(f"    C4/C3 separation = {sep:.1f}×")
        print(f"\n  Interpretation:")
        print(f"    λ locates the critical neighbourhood (necessary condition).")
        print(f"    C discriminates within it with {sep:.0f}× separation (sufficient).")

    return dict(correlation=corr, per_lambda=per_lambda,
                band_separation=sep, c4_in_band=len(c4_band))


# ==============================================================================
# SUBSTRATE 6 — 2D ISING MODEL (ferromagnetic phase transition)
# ==============================================================================
#
# HYPOTHESIS:
#   Composite C will peak at or near the analytically known critical temperature
#   Tc = 2.2692 J/kB (Onsager 1944), without being told where Tc is.
#
#   The three regimes map directly onto the framework's existing taxonomy:
#     T << Tc : ferromagnetic order   → spins align   → frozen   (Class 1/2 analogue)
#     T ≈  Tc : critical point        → scale-free fluctuations  (Class 4 analogue)
#     T >> Tc : paramagnetic disorder → random spins  → chaotic  (Class 3 analogue)
#
# NULL HYPOTHESIS:
#   C shows no peak at Tc — it either increases monotonically with T (tracking
#   entropy alone), decreases monotonically, or peaks at a temperature
#   inconsistent with the known critical point.
#
# EXTERNAL GROUND TRUTH:
#   Tc = 2.2692 J/kB is Onsager's exact analytical result (1944).
#   It has nothing to do with information theory — it comes from the partition
#   function of the 2D square-lattice Ising model.  If C peaks there it means
#   the framework independently recovers the thermodynamic critical point from
#   information-theoretic measurement alone.
#
# SUBSTRATE NOTES:
#   - Metropolis-Hastings Monte Carlo (standard, well-validated algorithm)
#   - Binary spin field: +1 → cell state 1, -1 → cell state 0
#   - IC: random 50% density (framework standard)
#   - Burnin discards the thermalisation transient
#   - Measurement window over equilibrated configurations
#   - No substrate-specific modifications to weight functions
#
# ==============================================================================

ISING_CFG = dict(
    GRID       = 16,        # Lattice size will be set per run
    N_STEPS    = 40000,     # More sweeps for better statistics
    BURNIN     = 10000,     # Safer thermalization (especially important for small L)
    WINDOW     = 200,       # Target number of snapshots
    SNAP_EVERY = 150,       # Good balance: low autocorrelation + enough samples
    N_SEEDS    = 20,        # Keep 20 seeds
    
    # Temperature sweep (fine resolution near criticality)
    T_MIN      = 2.22,
    T_MAX      = 2.32,
    T_STEP     = 0.005,     # Finer grid! Much better peak location accuracy
    
    TC_EXACT   = 2.2692,
    SWEEP_WORKERS = None,
)


def _decimal_places_for_step(step, cap=12):
    """Fraction digits for printing T at the resolution implied by T_STEP."""
    try:
        from decimal import Decimal
        d = Decimal(str(float(step))).normalize()
        e = d.as_tuple().exponent
        if e >= 0:
            return 0
        return min(-e, cap)
    except Exception:
        return 4


def _ising_run(T, cfg, seed=42):
    """
    Metropolis-Hastings Monte Carlo simulation of 2D ferromagnetic Ising model.

    Returns history array of shape (WINDOW, G*G) — binary (1=spin up, 0=spin down).
    Each row is a snapshot taken every SNAP_EVERY sweeps after thermalisation.

    The Metropolis acceptance probability:
      ΔE = 2 * spin[i,j] * sum_of_neighbours
      P(accept flip) = min(1, exp(-ΔE / T))
    """
    rng  = np.random.default_rng(seed)
    G    = cfg['GRID']
    N    = G * G

    # IC: random 50% density (framework standard)
    spins = rng.choice([-1, 1], size=(G, G)).astype(np.int8)

    burnin     = cfg['BURNIN']
    window     = cfg['WINDOW']
    snap_every = cfg['SNAP_EVERY']
    total_snaps= burnin + window * snap_every

    history = []
    snap_count = 0

    for sweep in range(total_snaps):
        # One sweep = G*G random spin-flip attempts
        idx_i = rng.integers(0, G, size=N)
        idx_j = rng.integers(0, G, size=N)

        for k in range(N):
            i, j = idx_i[k], idx_j[k]
            s    = int(spins[i, j])
            # Sum of 4 nearest neighbours (toroidal)
            nb_sum = (int(spins[(i-1)%G, j]) + int(spins[(i+1)%G, j]) +
                      int(spins[i, (j-1)%G]) + int(spins[i, (j+1)%G]))
            dE = 2 * s * nb_sum
            if dE <= 0 or rng.random() < np.exp(-dE / T):
                spins[i, j] = -s

        # Snapshot after burnin
        if sweep >= burnin and (sweep - burnin) % snap_every == 0:
            if snap_count < window:
                # Convert to binary: +1 → 1, -1 → 0
                history.append(((spins + 1) // 2).astype(np.int8).ravel())
                snap_count += 1

    return np.array(history, dtype=np.int8)   # (WINDOW, G*G)


def _ising_run_fast(T, cfg, seed=42):
    """
    Vectorised checkerboard Metropolis — ~10× faster than per-spin loop.

    Splits lattice into two interlaced sublattices (black/white squares).
    All spins on one sublattice can be updated simultaneously because
    they share no neighbours within the same sublattice.
    """
    rng  = np.random.default_rng(seed)
    G    = cfg['GRID']

    spins = rng.choice([-1, 1], size=(G, G)).astype(np.float32)

    burnin     = cfg['BURNIN']
    window     = cfg['WINDOW']
    snap_every = cfg['SNAP_EVERY']
    history    = []
    snap_count = 0
    total      = burnin + window * snap_every

    # Checkerboard masks
    ii, jj   = np.mgrid[0:G, 0:G]
    black    = ((ii + jj) % 2 == 0)
    white    = ~black

    for sweep in range(total):
        for mask in (black, white):
            # Neighbour sum for all sites simultaneously
            nb = (np.roll(spins, 1,  axis=0) + np.roll(spins, -1, axis=0) +
                  np.roll(spins, 1,  axis=1) + np.roll(spins, -1, axis=1))
            dE    = 2.0 * spins * nb
            # Accept flip where dE <= 0 OR random < exp(-dE/T)
            rand  = rng.random((G, G)).astype(np.float32)
            flip  = mask & ((dE <= 0) | (rand < np.exp(np.clip(-dE / T, -30, 0))))
            spins = np.where(flip, -spins, spins)

        if sweep >= burnin and (sweep - burnin) % snap_every == 0:
            if snap_count < window:
                history.append(((spins + 1) / 2).astype(np.int8).ravel())
                snap_count += 1

    return np.array(history, dtype=np.int8)


def _ising_metrics(history, cfg):
    """
    Apply full framework composite to an Ising snapshot history.
    history: (WINDOW, G*G) binary array — 0=down, 1=up
    """
    burnin = 0       # already thermalised — no additional burnin
    window = cfg['WINDOW']

    mH, sH       = _entropy_stats(history, burnin, window)
    op_up, op_dn = _opacity_both(history,  burnin, window)
    mi1, decay   = _opacity_temporal(history, burnin, window)
    tc           = _tcomp(history, burnin, window)
    post         = history[burnin:burnin + window]
    gz           = len(zlib.compress(post.tobytes(), 6)) / max(len(post.tobytes()), 1)
    C = composite(mH, sH, op_up, op_dn, mi1, decay, tc, gz)

    return dict(
        mean_H   = mH,    std_H  = sH,
        op_up    = op_up, op_dn  = op_dn,
        mi1      = mi1,   decay  = decay,
        tcomp    = tc,    gzip   = gz,
        w_H      = weight_H(mH, sH),
        w_OP_s   = weight_opacity_spatial(op_up, op_dn),
        w_OP_t   = weight_opacity_temporal(mi1, decay),
        w_T      = weight_tcomp(tc),
        w_G      = weight_gzip(gz),
        score    = C,
    )


def _ising_sweep_one_T(T, cfg):
    """All seeds at one temperature. Safe for parallel calls (read-only cfg)."""
    t_step0 = time.perf_counter()
    seed_results = []
    Tf = float(T)
    for s in range(cfg['N_SEEDS']):
        hist = _ising_run_fast(Tf, cfg, seed=s)
        m    = _ising_metrics(hist, cfg)
        m['T'] = Tf
        m['seed'] = s
        mid  = hist.astype(np.float32) * 2 - 1   # back to ±1
        m['magnetisation'] = float(np.abs(mid.mean()))
        seed_results.append(m)
    return Tf, seed_results, time.perf_counter() - t_step0


def _ising_pool_task(task):
    """Top-level (picklable) entry for ProcessPoolExecutor: ``(T, cfg)`` → result."""
    T, cfg = task
    return _ising_sweep_one_T(T, cfg)


def ising_sweep(cfg=None, csv_out=None, verbose=True):
    """
    Sweep temperature from T_MIN to T_MAX, compute C at each point.
    Returns list of result dicts.

    Temperature points run in parallel worker *processes* when
    cfg['SWEEP_WORKERS'] > 1 (default: min(number of T values, os.cpu_count())).
    Processes avoid a common deadlock from multi-threaded BLAS (OpenMP/MKL)
    nested inside several Python threads. Set SWEEP_WORKERS to 1 for sequential
    execution.
    """
    if cfg is None:
        cfg = ISING_CFG.copy()

    T_vals = np.round(np.arange(cfg['T_MIN'], cfg['T_MAX'] + 1e-9, cfg['T_STEP']), 4)
    TC     = cfg['TC_EXACT']
    rows   = []
    t_nd   = _decimal_places_for_step(cfg['T_STEP'])
    n_T    = len(T_vals)
    wcfg   = cfg.get('SWEEP_WORKERS')
    if wcfg is None:
        max_workers = min(n_T, os.cpu_count() or 1)
    else:
        max_workers = max(1, min(int(wcfg), n_T)) if n_T else 1

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Ising model — temperature sweep")
        print(f"  Grid: {cfg['GRID']}×{cfg['GRID']}  "
              f"Sweeps: {cfg['N_STEPS']}  Seeds: {cfg['N_SEEDS']}")
        print(f"  T range: {cfg['T_MIN']:.{t_nd}f} → {cfg['T_MAX']:.{t_nd}f}  "
              f"step={cfg['T_STEP']:.{t_nd}f}")
        print(f"  Tc (Onsager exact) = {TC}")
        print(f"  Parallel T processes: {max_workers}")
        print(f"{'─'*60}")

    T_list = [float(T) for T in T_vals]
    tasks = [(T, cfg) for T in T_list]

    def _print_T_line(T, seed_results, dt_wall):
        avg_C = float(np.mean([r['score'] for r in seed_results]))
        avg_M = float(np.mean([r['magnetisation'] for r in seed_results]))
        dist = abs(T - TC)
        regime = ('ordered' if T < TC - 0.15 else
                  'critical' if dist <= 0.15 else
                  'disordered')
        marker = ' ◄ Tc' if dist < cfg['T_STEP'] / 2 else ''
        dt_min = dt_wall / 60.0
        print(f"  T={T:.{t_nd}f}  [{regime:10s}]  "
              f"C={avg_C:.4f}  M={avg_M:.3f}  "
              f"time={dt_min:.4f} min{marker}", flush=True)

    if max_workers <= 1:
        per_T = [_ising_sweep_one_T(T, cfg) for T in T_list]
        for T, seed_results, dt_wall in per_T:
            rows.extend(seed_results)
            if verbose:
                _print_T_line(T, seed_results, dt_wall)
    else:
        if verbose:
            print(f"  Starting {max_workers} worker processes (first output may take a minute)…",
                  flush=True)
        done = {}
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            future_to_T = {ex.submit(_ising_pool_task, t): t[0] for t in tasks}
            for fut in as_completed(future_to_T):
                T_expect = future_to_T[fut]
                Tf, seed_results, dt_wall = fut.result()
                done[T_expect] = (seed_results, dt_wall)
                if verbose:
                    _print_T_line(T_expect, seed_results, dt_wall)
        per_T = [(T, done[T][0], done[T][1]) for T in T_list]
        for T, seed_results, dt_wall in per_T:
            rows.extend(seed_results)

    if csv_out:
        _save_csv(rows, csv_out)
        if verbose:
            print(f"\n  CSV → {csv_out}")

    return rows


def ising_plot(rows, cfg=None, save_path=None):
    """
    Six-panel analysis figure for Ising sweep results.
    """
    if cfg is None:
        cfg = ISING_CFG.copy()

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    TC = cfg['TC_EXACT']

    # Average over seeds per temperature
    T_vals = sorted(set(r['T'] for r in rows))
    def avg(key):
        return [float(np.mean([r[key] for r in rows if r['T'] == T])) for T in T_vals]
    def std(key):
        return [float(np.std( [r[key] for r in rows if r['T'] == T])) for T in T_vals]

    C_mean  = avg('score');        C_std   = std('score')
    M_mean  = avg('magnetisation')
    wH_mean = avg('w_H');          wT_mean = avg('w_T')
    wG_mean = avg('w_G');          wOPs_mean = avg('w_OP_s')
    wOPt_mean = avg('w_OP_t')
    H_mean  = avg('mean_H');       mi1_mean = avg('mi1')

    T_arr   = np.array(T_vals)
    peak_T  = T_vals[int(np.argmax(C_mean))]
    peak_C  = max(C_mean)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f'2D Ising Model — Complexity vs Temperature\n'
        f'Tc (Onsager exact) = {TC}  |  Grid: {cfg["GRID"]}×{cfg["GRID"]}  '
        f'Seeds: {cfg["N_SEEDS"]}',
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── Panel 1: Composite C vs T ─────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(T_arr, C_mean, yerr=C_std, fmt='o-', color='#2ecc71',
                 ms=5, lw=1.8, elinewidth=1, capsize=3, label='Composite C')
    ax1.axvline(TC,     color='red',    lw=2,   ls='--', label=f'Tc={TC}')
    ax1.axvline(peak_T, color='orange', lw=1.5, ls=':',  label=f'C peak={peak_T:.2f}')
    ax1.set_xlabel('Temperature T  (J/kB)')
    ax1.set_ylabel('Composite C')
    ax1.set_title('Composite C vs Temperature')
    ax1.legend(fontsize=8)

    # Shade regimes
    ax1.axvspan(T_arr[0], TC,         alpha=0.06, color='blue',  label='Ordered')
    ax1.axvspan(TC,        T_arr[-1], alpha=0.06, color='red',   label='Disordered')

    # ── Panel 2: Magnetisation vs T ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(T_arr, M_mean, 's-', color='#3498db', ms=5, lw=1.8)
    ax2.axvline(TC, color='red', lw=2, ls='--', label=f'Tc={TC}')
    ax2.set_xlabel('Temperature T  (J/kB)')
    ax2.set_ylabel('|Magnetisation|')
    ax2.set_title('Order Parameter vs Temperature\n(independent physical observable)')
    ax2.legend(fontsize=8)

    # ── Panel 3: Weight heatmap vs T ──────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    weight_data = np.array([wH_mean, wOPs_mean, wOPt_mean, wT_mean, wG_mean])
    im = ax3.imshow(weight_data, aspect='auto', cmap='RdYlGn',
                    vmin=0, vmax=1,
                    extent=[T_arr[0], T_arr[-1], -0.5, 4.5])
    ax3.set_yticks([0,1,2,3,4])
    ax3.set_yticklabels(['w_G','w_T','w_OP_t','w_OP_s','w_H'], fontsize=9)
    ax3.axvline(TC, color='red', lw=2, ls='--')
    ax3.set_xlabel('Temperature T')
    ax3.set_title('Metric Weights vs Temperature')
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # ── Panel 4: Individual weights vs T ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(T_arr, wH_mean,   'o-', label='w_H',    ms=4, lw=1.5)
    ax4.plot(T_arr, wOPt_mean, 's-', label='w_OP_t', ms=4, lw=1.5)
    ax4.plot(T_arr, wT_mean,   '^-', label='w_T',    ms=4, lw=1.5)
    ax4.plot(T_arr, wG_mean,   'D-', label='w_G',    ms=4, lw=1.5)
    ax4.axvline(TC, color='red', lw=2, ls='--', label=f'Tc={TC}')
    ax4.set_xlabel('Temperature T')
    ax4.set_ylabel('Weight value')
    ax4.set_title('Individual Metric Weights vs T')
    ax4.legend(fontsize=7)

    # ── Panel 5: Entropy and MI vs T ──────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5_r = ax5.twinx()
    ax5.plot(T_arr, H_mean,   'o-', color='#e74c3c', ms=4, lw=1.5, label='mean H')
    ax5_r.plot(T_arr, mi1_mean, 's--', color='#9b59b6', ms=4, lw=1.5, label='MI₁')
    ax5.axvline(TC, color='red', lw=2, ls='--')
    ax5.set_xlabel('Temperature T')
    ax5.set_ylabel('Mean Entropy', color='#e74c3c')
    ax5_r.set_ylabel('Temporal MI₁', color='#9b59b6')
    ax5.set_title('Entropy & Temporal MI vs T')
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_r.get_legend_handles_labels()
    ax5.legend(lines1+lines2, labels1+labels2, fontsize=8)

    # ── Panel 6: Summary ──────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    dist_to_tc = abs(peak_T - TC)
    verdict = ('CONFIRMED' if dist_to_tc <= cfg['T_STEP'] * 1.5
               else f'OFFSET by {dist_to_tc:.3f}')

    summary = (
        f"RESULTS SUMMARY\n"
        f"{'─'*32}\n\n"
        f"Tc (Onsager exact) = {TC}\n"
        f"C peak at T        = {peak_T:.4f}\n"
        f"Distance to Tc     = {dist_to_tc:.4f}\n\n"
        f"Peak C             = {peak_C:.5f}\n\n"
        f"Verdict: {verdict}\n\n"
        f"{'─'*32}\n"
        f"Hypothesis:\n"
        f"  C peaks at Tc without\n"
        f"  being told where Tc is.\n\n"
        f"Null hypothesis:\n"
        f"  C shows no peak at Tc\n"
        f"  (monotonic or wrong peak)\n\n"
        f"External ground truth:\n"
        f"  Onsager (1944) exact\n"
        f"  solution — independent\n"
        f"  of information theory.\n\n"
        f"Grid: {cfg['GRID']}×{cfg['GRID']}\n"
        f"Seeds: {cfg['N_SEEDS']}\n"
        f"T step: {cfg['T_STEP']}"
    )
    color = '#2ecc71' if dist_to_tc <= cfg['T_STEP'] * 1.5 else '#e74c3c'
    ax6.text(0.04, 0.97, summary, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85,
                       edgecolor=color, linewidth=2))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot → {save_path}")
    else:
        plt.show()


# ==============================================================================
# SUBSTRATE 7 — SIR EPIDEMIC MODEL (R₀ phase transition)
# ==============================================================================
#
# HYPOTHESIS:
#   Composite C will peak at or near the analytically known epidemic threshold
#   R₀ = 1 (basic reproduction number), without being told where it is.
#
#   The three regimes map onto the framework's existing taxonomy:
#     R₀ < 1 : sub-critical  — epidemic dies out        (ordered/dead)
#     R₀ = 1 : critical      — spreading wavefront      (Class 4 analogue)
#     R₀ > 1 : super-critical— epidemic sweeps to fixation (chaotic/uniform)
#
# NULL HYPOTHESIS:
#   C shows no peak at R₀=1 — either monotonic with R₀, or peaks elsewhere.
#
# EXTERNAL GROUND TRUTH:
#   R₀=1 is derived from branching process theory and validated by a century
#   of epidemiological data. It has nothing to do with information theory.
#
# SYMBOLIZATION — TWO MAPPINGS COMPARED:
#   This experiment deliberately runs BOTH binary projections of the 3-state
#   SIR field and compares them. If the signal is real it should appear in
#   both; if only one shows it, that tells us which channel carries complexity.
#
#   Mapping A — INFECTED field:   I=1, (S,R)=0
#     Captures the active wavefront dynamics directly.
#
#   Mapping B — SUSCEPTIBLE field: S=1, (I,R)=0
#     Captures the "landscape" being carved by the epidemic — the holes
#     left behind reveal spatial structure as the wave passes.
#
# SUBSTRATE NOTES:
#   - Discrete-time SIR on 2D toroidal grid (Moore neighbourhood)
#   - Infection probability β sweeps R₀ from sub- to super-critical
#   - Recovery probability γ fixed; R₀ = β * mean_neighbours / γ
#   - IC: random 50% susceptible density, 1% infected seed (framework standard)
#   - Multiple seeds for robustness
#
# ==============================================================================

SIR_CFG = dict(
    GRID        = 128,      # lattice size
    STEPS       = 300,      # simulation steps
    BURNIN      = 20,       # discard early transient
    WINDOW      = 200,      # measurement window
    N_SEEDS     = 5,
    GAMMA       = 0.10,     # recovery probability per step
    INIT_I_FRAC = 0.01,     # initial infected fraction (seed)
    INIT_S_FRAC = 0.99,     # initial susceptible fraction
    # β sweep — R₀ = β * 8 * (1-γ) / γ approximately
    # We sweep β directly and compute R₀ for labelling
    BETA_MIN    = 0.005,
    BETA_MAX    = 0.080,
    BETA_STEP   = 0.005,
    R0_CRITICAL = 1.0,      # known threshold — external ground truth
)

# SIR cell states
_S, _I, _R = 0, 1, 2


def _sir_run(beta, cfg, seed=42):
    """
    Discrete-time stochastic SIR on 2D toroidal Moore-neighbourhood grid.

    At each step for each susceptible cell:
      P(become infected) = 1 - (1-β)^n_infected_neighbours
    Each infected cell recovers with probability γ per step.

    Returns:
      hist_I: (STEPS, G*G) binary — infected field (mapping A)
      hist_S: (STEPS, G*G) binary — susceptible field (mapping B)
    """
    rng   = np.random.default_rng(seed)
    G     = cfg['GRID']
    gamma = cfg['GAMMA']

    # Initialise: almost all susceptible, small infected seed
    state = np.zeros((G, G), dtype=np.int8)   # 0=S
    infected_mask = rng.random((G, G)) < cfg['INIT_I_FRAC']
    state[infected_mask] = _I

    hist_I = []
    hist_S = []

    for t in range(cfg['STEPS']):
        hist_I.append((state == _I).astype(np.int8).ravel())
        hist_S.append((state == _S).astype(np.int8).ravel())

        n_infected = np.zeros((G, G), dtype=np.int32)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                n_infected += (np.roll(np.roll(state, di, axis=0), dj, axis=1) == _I)

        new_state = state.copy()

        # S → I
        susceptible = (state == _S)
        p_infect    = 1.0 - (1.0 - beta) ** n_infected.astype(np.float32)
        infect_mask = susceptible & (rng.random((G, G)).astype(np.float32) < p_infect)
        new_state[infect_mask] = _I

        # I → R
        recover_mask = (state == _I) & (rng.random((G, G)) < gamma)
        new_state[recover_mask] = _R

        state = new_state

    return (np.array(hist_I, dtype=np.int8),
            np.array(hist_S, dtype=np.int8))


def _sir_metrics(history, cfg):
    """Apply full framework composite to a binary SIR field history."""
    burnin = cfg['BURNIN']
    window = cfg['WINDOW']
    mH, sH       = _entropy_stats(history, burnin, window)
    op_up, op_dn = _opacity_both(history,  burnin, window)
    mi1, decay   = _opacity_temporal(history, burnin, window)
    tc           = _tcomp(history, burnin, window)
    post         = history[burnin:burnin + window]
    gz           = len(zlib.compress(post.tobytes(), 6)) / max(len(post.tobytes()), 1)
    C = composite(mH, sH, op_up, op_dn, mi1, decay, tc, gz)
    return dict(
        mean_H=mH, std_H=sH, op_up=op_up, op_dn=op_dn,
        mi1=mi1, decay=decay, tcomp=tc, gzip=gz,
        w_H    = weight_H(mH, sH),
        w_OP_s = weight_opacity_spatial(op_up, op_dn),
        w_OP_t = weight_opacity_temporal(mi1, decay),
        w_T    = weight_tcomp(tc),
        w_G    = weight_gzip(gz),
        score  = C,
    )


def _sir_r0(beta, cfg):
    """Approximate R₀ = β * k / γ where k=8 (Moore neighbourhood)."""
    return float(beta * 8 / cfg['GAMMA'])


def sir_sweep(cfg=None, csv_out=None, verbose=True):
    """
    Sweep β (infection probability), compute C for both binary mappings.
    Returns list of result rows.
    """
    if cfg is None:
        cfg = SIR_CFG.copy()

    betas  = np.round(np.arange(cfg['BETA_MIN'],
                                cfg['BETA_MAX'] + 1e-9,
                                cfg['BETA_STEP']), 4)
    rows   = []

    if verbose:
        print(f"\n{'─'*65}")
        print(f"SIR epidemic — β sweep  (R₀ = β×8/γ,  γ={cfg['GAMMA']})")
        print(f"  Grid: {cfg['GRID']}×{cfg['GRID']}  "
              f"Steps: {cfg['STEPS']}  Seeds: {cfg['N_SEEDS']}")
        print(f"  β range: {cfg['BETA_MIN']} → {cfg['BETA_MAX']}  "
              f"step={cfg['BETA_STEP']}")
        print(f"  R₀ critical = {cfg['R0_CRITICAL']} (external ground truth)")
        print(f"{'─'*65}")
        print(f"  {'β':>6}  {'R₀':>5}  {'regime':10}  "
              f"{'C_infected':>11}  {'C_suscept':>10}  {'I_density':>9}")

    for beta in betas:
        R0      = _sir_r0(beta, cfg)
        dist    = abs(R0 - cfg['R0_CRITICAL'])
        regime  = ('sub-critical'  if R0 < 0.85 else
                   'critical'      if dist < 0.3 else
                   'super-critical')

        sr_I, sr_S = [], []

        for s in range(cfg['N_SEEDS']):
            hI, hS = _sir_run(beta, cfg, seed=s)

            mI = _sir_metrics(hI, cfg)
            mI.update(dict(beta=float(beta), R0=R0, seed=s,
                           mapping='infected', regime=regime,
                           final_I_density=float(hI[-1].mean()),
                           final_S_density=float(hS[-1].mean())))
            rows.append(mI)
            sr_I.append(mI['score'])

            mS = _sir_metrics(hS, cfg)
            mS.update(dict(beta=float(beta), R0=R0, seed=s,
                           mapping='susceptible', regime=regime,
                           final_I_density=float(hI[-1].mean()),
                           final_S_density=float(hS[-1].mean())))
            rows.append(mS)
            sr_S.append(mS['score'])

        avg_I = float(np.mean(sr_I))
        avg_S = float(np.mean(sr_S))
        avg_I_dens = float(np.mean([r['final_I_density']
                                    for r in rows if r['beta']==float(beta)
                                    and r['mapping']=='infected']))
        marker = ' ◄ R₀≈1' if dist < cfg['BETA_STEP'] * 8 * 1.5 else ''

        if verbose:
            print(f"  β={beta:.3f}  R₀={R0:.2f}  [{regime:12s}]  "
                  f"C_I={avg_I:.5f}  C_S={avg_S:.5f}  "
                  f"Idens={avg_I_dens:.3f}{marker}")

    if csv_out:
        _save_csv(rows, csv_out)
        if verbose:
            print(f"\n  CSV → {csv_out}")

    return rows


def sir_plot(rows, cfg=None, save_path=None):
    """
    Six-panel analysis figure comparing both binary mappings across R₀.
    """
    if cfg is None:
        cfg = SIR_CFG.copy()

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    R0_CRIT = cfg['R0_CRITICAL']

    # Separate by mapping
    rows_I = [r for r in rows if r['mapping'] == 'infected']
    rows_S = [r for r in rows if r['mapping'] == 'susceptible']

    betas  = sorted(set(r['beta'] for r in rows_I))
    R0s    = [_sir_r0(b, cfg) for b in betas]

    def avg_by_beta(rlist, key):
        return [float(np.mean([r[key] for r in rlist if r['beta']==b]))
                for b in betas]
    def sem_by_beta(rlist, key):
        vals = [np.std([r[key] for r in rlist if r['beta']==b]) /
                np.sqrt(cfg['N_SEEDS']) for b in betas]
        return [float(v) for v in vals]

    CI_mean  = avg_by_beta(rows_I, 'score')
    CI_err   = sem_by_beta(rows_I, 'score')
    CS_mean  = avg_by_beta(rows_S, 'score')
    CS_err   = sem_by_beta(rows_S, 'score')
    Id_mean  = avg_by_beta(rows_I, 'final_I_density')
    Sd_mean  = avg_by_beta(rows_I, 'final_S_density')

    # Per-metric for infected mapping
    wH_I    = avg_by_beta(rows_I, 'w_H')
    wOPt_I  = avg_by_beta(rows_I, 'w_OP_t')
    wT_I    = avg_by_beta(rows_I, 'w_T')
    wG_I    = avg_by_beta(rows_I, 'w_G')
    wOPs_I  = avg_by_beta(rows_I, 'w_OP_s')

    R0_arr   = np.array(R0s)
    peak_I   = R0s[int(np.argmax(CI_mean))]
    peak_S   = R0s[int(np.argmax(CS_mean))]
    peak_CI  = max(CI_mean)
    peak_CS  = max(CS_mean)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor('#0e0e16')
    fig.suptitle(
        f'SIR Epidemic Model — Complexity vs R₀\n'
        f'R₀ critical = {R0_CRIT} (branching process theory)   |   '
        f'Grid: {cfg["GRID"]}×{cfg["GRID"]}   Seeds: {cfg["N_SEEDS"]}   '
        f'γ = {cfg["GAMMA"]}',
        fontsize=13, fontweight='bold', color='white', y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    PANEL_BG = '#1a1a2a'
    AXIS_COL = '#ccccdd'
    GRID_COL = '#333344'
    TC_COL   = '#ff4444'

    def style_ax(ax):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=AXIS_COL)
        ax.xaxis.label.set_color(AXIS_COL)
        ax.yaxis.label.set_color(AXIS_COL)
        ax.title.set_color(AXIS_COL)
        for sp in ax.spines.values(): sp.set_color(GRID_COL)
        ax.grid(True, color=GRID_COL, lw=0.5, alpha=0.6)

    # ── Panel 1: Both C curves vs R₀ — HERO ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    style_ax(ax1)

    ax1.fill_between(R0_arr,
                     [c-e for c,e in zip(CI_mean,CI_err)],
                     [c+e for c,e in zip(CI_mean,CI_err)],
                     alpha=0.15, color='#e74c3c')
    ax1.fill_between(R0_arr,
                     [c-e for c,e in zip(CS_mean,CS_err)],
                     [c+e for c,e in zip(CS_mean,CS_err)],
                     alpha=0.15, color='#3498db')

    ax1.errorbar(R0_arr, CI_mean, yerr=CI_err, fmt='o-',
                 color='#e74c3c', ms=6, lw=2, elinewidth=1.2, capsize=4,
                 label=f'C — Infected field (peak R₀={peak_I:.2f})')
    ax1.errorbar(R0_arr, CS_mean, yerr=CS_err, fmt='s--',
                 color='#3498db', ms=6, lw=2, elinewidth=1.2, capsize=4,
                 label=f'C — Susceptible field (peak R₀={peak_S:.2f})')

    ymax = max(max(CI_mean), max(CS_mean)) * 1.35
    ax1.axvspan(R0_arr[0], R0_CRIT,     alpha=0.07, color='#3498db',
                label='Sub-critical (epidemic dies)')
    ax1.axvspan(R0_CRIT,   R0_arr[-1],  alpha=0.07, color='#e74c3c',
                label='Super-critical (epidemic spreads)')
    ax1.axvline(R0_CRIT, color=TC_COL, lw=2.5, ls='--', zorder=5,
                label=f'R₀=1 (external ground truth)')

    ax1.set_xlim(R0_arr[0]-0.05, R0_arr[-1]+0.05)
    ax1.set_ylim(-0.005, ymax)
    ax1.set_xlabel('Basic Reproduction Number  R₀  =  β × 8 / γ', fontsize=11)
    ax1.set_ylabel('Composite C', fontsize=11)
    ax1.set_title('Composite C vs R₀  —  Both Binary Mappings', fontsize=11,
                  fontweight='bold')
    ax1.legend(fontsize=8, facecolor='#1a1a2a', labelcolor=AXIS_COL,
               edgecolor=GRID_COL, loc='upper left')

    # ── Panel 2: Epidemic dynamics (density over R₀) ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2)
    ax2.plot(R0_arr, Id_mean, 'o-', color='#e74c3c', ms=5, lw=1.8,
             label='Final I density')
    ax2.plot(R0_arr, Sd_mean, 's-', color='#3498db', ms=5, lw=1.8,
             label='Final S density')
    ax2.axvline(R0_CRIT, color=TC_COL, lw=2, ls='--',
                label=f'R₀=1')
    ax2.set_xlabel('R₀')
    ax2.set_ylabel('Final cell density')
    ax2.set_title('Epidemic Dynamics\n(independent physical observable)', fontsize=10)
    ax2.legend(fontsize=8, facecolor='#1a1a2a', labelcolor=AXIS_COL,
               edgecolor=GRID_COL)

    # ── Panel 3: Metric weights for infected mapping ──────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3)
    ax3.plot(R0_arr, wH_I,   'o-', color='#e74c3c', ms=4, lw=1.5, label='w_H')
    ax3.plot(R0_arr, wOPt_I, 's-', color='#9b59b6', ms=4, lw=1.5, label='w_OP_t')
    ax3.plot(R0_arr, wT_I,   '^-', color='#f39c12', ms=4, lw=1.5, label='w_T')
    ax3.plot(R0_arr, wG_I,   'D-', color='#1abc9c', ms=4, lw=1.5, label='w_G')
    ax3.plot(R0_arr, wOPs_I, 'v-', color='#e67e22', ms=4, lw=1.5, label='w_OP_s')
    ax3.axvline(R0_CRIT, color=TC_COL, lw=2, ls='--')
    ax3.set_xlabel('R₀')
    ax3.set_ylabel('Weight value')
    ax3.set_title('Metric Weights vs R₀\n(Infected mapping)', fontsize=10)
    ax3.legend(fontsize=7, facecolor='#1a1a2a', labelcolor=AXIS_COL,
               edgecolor=GRID_COL, ncol=2)

    # ── Panel 4: Direct C comparison bar at nearest R₀ values ────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4)
    regimes     = ['Sub-critical\n(R₀<0.85)',
                   'Critical\n(R₀≈1)',
                   'Super-critical\n(R₀>1.3)']
    sub_I  = float(np.mean([r['score'] for r in rows_I if r['R0'] < 0.85]))
    crit_I = float(np.mean([r['score'] for r in rows_I
                             if abs(r['R0']-R0_CRIT) < 0.35]))
    sup_I  = float(np.mean([r['score'] for r in rows_I if r['R0'] > 1.3]))
    sub_S  = float(np.mean([r['score'] for r in rows_S if r['R0'] < 0.85]))
    crit_S = float(np.mean([r['score'] for r in rows_S
                             if abs(r['R0']-R0_CRIT) < 0.35]))
    sup_S  = float(np.mean([r['score'] for r in rows_S if r['R0'] > 1.3]))

    x   = np.arange(3)
    w   = 0.35
    ax4.bar(x - w/2, [sub_I, crit_I, sup_I], width=w, color='#e74c3c',
            alpha=0.85, edgecolor='k', lw=0.7, label='Infected')
    ax4.bar(x + w/2, [sub_S, crit_S, sup_S], width=w, color='#3498db',
            alpha=0.85, edgecolor='k', lw=0.7, label='Susceptible')
    ax4.set_xticks(x)
    ax4.set_xticklabels(regimes, fontsize=8)
    ax4.set_ylabel('Mean C')
    ax4.set_title('C by Regime — Both Mappings', fontsize=10)
    ax4.legend(fontsize=8, facecolor='#1a1a2a', labelcolor=AXIS_COL,
               edgecolor=GRID_COL)

    # ── Panel 5: Verdict summary ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(PANEL_BG)
    ax5.axis('off')

    dist_I    = abs(peak_I - R0_CRIT)
    dist_S    = abs(peak_S - R0_CRIT)
    verdict_I = 'CONFIRMED' if dist_I < 0.35 else f'OFFSET {dist_I:.2f}'
    verdict_S = 'CONFIRMED' if dist_S < 0.35 else f'OFFSET {dist_S:.2f}'
    vcolor    = '#2ecc71' if (dist_I < 0.35 or dist_S < 0.35) else '#e74c3c'

    # Agreement between mappings
    peak_agree = abs(peak_I - peak_S) < 0.35
    agree_str  = 'AGREE' if peak_agree else 'DISAGREE'
    agree_col  = '#2ecc71' if peak_agree else '#f39c12'

    summary = (
        f"RESULTS SUMMARY\n"
        f"{'─'*32}\n\n"
        f"R₀ critical (theory) = {R0_CRIT:.2f}\n\n"
        f"Infected mapping:\n"
        f"  C peaks at R₀ = {peak_I:.2f}\n"
        f"  Peak C = {peak_CI:.5f}\n"
        f"  Verdict: {verdict_I}\n\n"
        f"Susceptible mapping:\n"
        f"  C peaks at R₀ = {peak_S:.2f}\n"
        f"  Peak C = {peak_CS:.5f}\n"
        f"  Verdict: {verdict_S}\n\n"
        f"Mappings: {agree_str}\n\n"
        f"{'─'*32}\n"
        f"H:  C peaks at R₀=1\n"
        f"    without being told\n\n"
        f"H0: No peak at R₀=1\n\n"
        f"Grid: {cfg['GRID']}×{cfg['GRID']}\n"
        f"Seeds: {cfg['N_SEEDS']}  γ={cfg['GAMMA']}"
    )
    ax5.text(0.05, 0.97, summary, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace', color=AXIS_COL,
             bbox=dict(boxstyle='round', facecolor='#0e0e16', alpha=0.9,
                       edgecolor=vcolor, linewidth=2.5))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#0e0e16')
        plt.close()
        print(f"  Plot → {save_path}")
    else:
        plt.show()


# ==============================================================================
# SUBSTRATE 8 — DIRECTED PERCOLATION (tuned critical threshold)
# ==============================================================================
#
# HYPOTHESIS:
#   Composite C will peak at or near the empirical critical threshold p_c≈0.287
#   for the synchronous contact process on a 2D toroidal grid, without being
#   told where p_c is.
#
# NULL HYPOTHESIS:
#   C shows no peak at p_c — monotonic with p, or peaks elsewhere.
#
# EXTERNAL GROUND TRUTH:
#   The critical threshold separating absorbing (dead) from active phases is
#   an established result from non-equilibrium statistical physics.
#   The synchronous contact process belongs to the directed percolation
#   universality class — same critical exponents as epidemic spreading,
#   turbulence onset, catalytic reactions, neural avalanches.
#   The specific p_c value for this lattice geometry is determined
#   empirically (activity survives iff p > p_c in the infinite-size limit).
#
# NOTE ON p_c VALUE:
#   The commonly cited p_c=0.6447 (Grassberger 1989) is for directed bond
#   percolation on a specific lattice. Our synchronous contact process with
#   von Neumann neighbourhood has a different p_c (~0.287 on 128×128).
#   Both belong to the same universality class — the test is whether C
#   finds the transition, wherever it actually is.
#
# ==============================================================================

DP_CFG = dict(
    GRID      = 128,
    STEPS     = 400,
    BURNIN    = 50,
    WINDOW    = 200,
    N_SEEDS   = 5,
    P_MIN     = 0.10,
    P_MAX     = 0.50,
    P_STEP    = 0.025,
    # p_c for synchronous contact process, von Neumann neighbourhood,
    # 128×128 toroidal grid — empirically determined to be ~0.27-0.29.
    # Note: the commonly cited p_c=0.6447 (Grassberger 1989) is for
    # directed BOND percolation on a different lattice geometry.
    # Our model is the synchronous contact process; p_c differs.
    PC_EXACT  = 0.2873,     # empirical for this specific model variant
    INIT_DENSITY = 0.02,    # sparse IC — lets sub-critical runs die naturally
)


def _dp_run(p, cfg, seed=42):
    """
    2D directed (bond) percolation — synchronous update.

    A cell is active at t+1 if at least one active neighbour
    at time t activates it with probability p (independently per neighbour).
    Equivalent: active if any of 4 neighbours fires.

    Uses von Neumann (4-neighbour) neighbourhood — standard for DP.
    Returns binary history (STEPS, G*G).
    """
    rng = np.random.default_rng(seed)
    G   = cfg['GRID']

    grid    = (rng.random((G, G)) < cfg['INIT_DENSITY']).astype(np.int8)
    history = []

    for t in range(cfg['STEPS']):
        history.append(grid.ravel().copy())

        # Each active neighbour independently tries to activate cell
        # P(cell becomes active) = 1 - (1-p)^(number of active neighbours)
        padded   = np.pad(grid, 1, mode='wrap')
        nb_sum   = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                    padded[1:-1, :-2] + padded[1:-1, 2:]).astype(np.float32)

        p_activate = 1.0 - (1.0 - p) ** nb_sum
        new_grid   = (rng.random((G, G)).astype(np.float32) < p_activate).astype(np.int8)
        grid       = new_grid

        # If fully dead (absorbing state) — pad remainder with zeros
        if grid.sum() == 0:
            for _ in range(cfg['STEPS'] - t - 1):
                history.append(np.zeros(G*G, dtype=np.int8))
            break

    return np.array(history, dtype=np.int8)


def _dp_metrics(history, cfg):
    burnin = cfg['BURNIN']
    window = cfg['WINDOW']
    mH, sH       = _entropy_stats(history, burnin, window)
    op_up, op_dn = _opacity_both(history,  burnin, window)
    mi1, decay   = _opacity_temporal(history, burnin, window)
    tc           = _tcomp(history, burnin, window)
    post         = history[burnin:burnin + window]
    gz           = len(zlib.compress(post.tobytes(), 6)) / max(len(post.tobytes()), 1)
    C = composite(mH, sH, op_up, op_dn, mi1, decay, tc, gz)
    return dict(
        mean_H=mH, std_H=sH, op_up=op_up, op_dn=op_dn,
        mi1=mi1, decay=decay, tcomp=tc, gzip=gz,
        w_H    = weight_H(mH, sH),
        w_OP_s = weight_opacity_spatial(op_up, op_dn),
        w_OP_t = weight_opacity_temporal(mi1, decay),
        w_T    = weight_tcomp(tc),
        w_G    = weight_gzip(gz),
        score  = C,
    )


def dp_sweep(cfg=None, csv_out=None, verbose=True):
    if cfg is None:
        cfg = DP_CFG.copy()

    p_vals = np.round(np.arange(cfg['P_MIN'], cfg['P_MAX']+1e-9, cfg['P_STEP']), 4)
    PC     = cfg['PC_EXACT']
    rows   = []

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Directed Percolation — p sweep")
        print(f"  Grid: {cfg['GRID']}×{cfg['GRID']}  "
              f"Steps: {cfg['STEPS']}  Seeds: {cfg['N_SEEDS']}")
        print(f"  p range: {cfg['P_MIN']} → {cfg['P_MAX']}  step={cfg['P_STEP']}")
        print(f"  p_c (Grassberger exact) = {PC}")
        print(f"{'─'*60}")

    for p in p_vals:
        sr = []
        for s in range(cfg['N_SEEDS']):
            hist = _dp_run(p, cfg, seed=s)
            m    = _dp_metrics(hist, cfg)
            m.update(dict(p=float(p), seed=s,
                          mean_density=float(hist[cfg['BURNIN']:].mean())))
            rows.append(m)
            sr.append(m['score'])

        avg_C  = float(np.mean(sr))
        dist   = abs(p - PC)
        regime = ('sub-critical'  if p < PC - 0.05 else
                  'critical'      if dist <= 0.05   else
                  'super-critical')
        marker = ' ◄ p_c' if dist < cfg['P_STEP']/2 else ''
        if verbose:
            avg_d = float(np.mean([r['mean_density'] for r in rows
                                   if r['p']==float(p)]))
            print(f"  p={p:.3f}  [{regime:13s}]  "
                  f"C={avg_C:.5f}  density={avg_d:.3f}{marker}")

    if csv_out:
        _save_csv(rows, csv_out)
        if verbose:
            print(f"\n  CSV → {csv_out}")

    return rows


# ==============================================================================
# SUBSTRATE 9 — FOREST FIRE MODEL (self-organized criticality)
# ==============================================================================
#
# HYPOTHESIS:
#   C will be elevated across a broad intermediate p/f ratio range,
#   consistent with the known SOC regime (Drossel & Schwabl 1992).
#   Unlike Ising/DP, the prediction is a PLATEAU not a spike — because
#   SOC means the system self-tunes to criticality across a wide parameter range.
#
# NULL HYPOTHESIS:
#   C tracks monotonically with tree density, or shows no elevated plateau
#   at intermediate p/f — no SOC signal detectable.
#
# EXTERNAL GROUND TRUTH:
#   Drossel & Schwabl (1992) established that the forest fire model exhibits
#   SOC at intermediate p/f, with power-law fire size distributions.
#   Independent of information theory — confirmed by cluster size analysis.
#
# KEY DISTINCTION FROM ISING/DP:
#   This is self-organized criticality — no parameter tuning required.
#   The system finds its own critical state. The framework should detect
#   elevated C across a range, not at a single point.
#
# ==============================================================================

FF_CFG = dict(
    GRID      = 128,
    STEPS     = 500,
    BURNIN    = 100,
    WINDOW    = 200,
    N_SEEDS   = 5,
    # Sweep p/f ratio — the control parameter for SOC
    # p = tree growth probability per empty cell per step
    # f = lightning strike probability per tree per step
    # SOC emerges at large p/f (slow lightning relative to growth)
    P_TREE    = 0.05,        # fixed tree growth rate
    F_VALUES  = np.array([0.500, 0.200, 0.100, 0.050, 0.020,
                           0.010, 0.005, 0.002, 0.001]),  # lightning rates
    # p/f ratios: 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50
)


def _ff_run(p_tree, f_lightning, cfg, seed=42):
    """
    Drossel-Schwabl forest fire model on 2D toroidal grid.

    States: 0=empty, 1=tree, 2=burning
    Rules (applied synchronously):
      burning → empty
      tree adjacent to burning → burning  (fire spreads)
      tree → burning with prob f          (lightning)
      empty → tree with prob p            (growth)

    Returns binary history: tree=1, (empty+burning)=0
    """
    rng = np.random.default_rng(seed)
    G   = cfg['GRID']

    # IC: 50% trees
    grid    = (rng.random((G, G)) < 0.5).astype(np.int8)
    history = []

    EMPTY, TREE, BURN = 0, 1, 2

    for t in range(cfg['STEPS']):
        # Record binary tree field
        history.append((grid == TREE).astype(np.int8).ravel())

        new_grid = grid.copy()

        # burning → empty
        new_grid[grid == BURN] = EMPTY

        # tree adjacent to burning → burning (von Neumann)
        padded  = np.pad((grid == BURN).astype(np.int8), 1, mode='wrap')
        adj_fire = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                    padded[1:-1, :-2] + padded[1:-1, 2:]) > 0
        new_grid[(grid == TREE) & adj_fire] = BURN

        # lightning: tree → burning with prob f
        lightning = (grid == TREE) & (rng.random((G, G)) < f_lightning)
        new_grid[lightning] = BURN

        # growth: empty → tree with prob p
        growth = (grid == EMPTY) & (rng.random((G, G)) < p_tree)
        new_grid[growth] = TREE

        grid = new_grid

    return np.array(history, dtype=np.int8)


def _ff_metrics(history, cfg):
    burnin = cfg['BURNIN']
    window = cfg['WINDOW']
    mH, sH       = _entropy_stats(history, burnin, window)
    op_up, op_dn = _opacity_both(history,  burnin, window)
    mi1, decay   = _opacity_temporal(history, burnin, window)
    tc           = _tcomp(history, burnin, window)
    post         = history[burnin:burnin + window]
    gz           = len(zlib.compress(post.tobytes(), 6)) / max(len(post.tobytes()), 1)
    C = composite(mH, sH, op_up, op_dn, mi1, decay, tc, gz)
    return dict(
        mean_H=mH, std_H=sH, op_up=op_up, op_dn=op_dn,
        mi1=mi1, decay=decay, tcomp=tc, gzip=gz,
        w_H    = weight_H(mH, sH),
        w_OP_s = weight_opacity_spatial(op_up, op_dn),
        w_OP_t = weight_opacity_temporal(mi1, decay),
        w_T    = weight_tcomp(tc),
        w_G    = weight_gzip(gz),
        score  = C,
    )


def ff_sweep(cfg=None, csv_out=None, verbose=True):
    if cfg is None:
        cfg = FF_CFG.copy()

    p     = cfg['P_TREE']
    f_vals= cfg['F_VALUES']
    rows  = []

    if verbose:
        print(f"\n{'─'*60}")
        print(f"Forest Fire — p/f sweep  (p={p} fixed, f varies)")
        print(f"  Grid: {cfg['GRID']}×{cfg['GRID']}  "
              f"Steps: {cfg['STEPS']}  Seeds: {cfg['N_SEEDS']}")
        print(f"  SOC prediction: elevated C at intermediate p/f")
        print(f"{'─'*60}")

    for f in f_vals:
        pf_ratio = p / f
        sr = []
        for s in range(cfg['N_SEEDS']):
            hist = _ff_run(p, f, cfg, seed=s)
            m    = _ff_metrics(hist, cfg)
            m.update(dict(f=float(f), p_over_f=float(pf_ratio), seed=s,
                          tree_density=float(hist[cfg['BURNIN']:].mean())))
            rows.append(m)
            sr.append(m['score'])

        avg_C    = float(np.mean(sr))
        avg_dens = float(np.mean([r['tree_density'] for r in rows
                                  if abs(r['f']-float(f))<1e-9]))
        regime   = ('fire-dominated' if pf_ratio < 1   else
                    'SOC-regime'     if pf_ratio < 15  else
                    'growth-dominated')

        if verbose:
            print(f"  f={f:.3f}  p/f={pf_ratio:5.1f}  [{regime:16s}]  "
                  f"C={avg_C:.5f}  tree_dens={avg_dens:.3f}")

    if csv_out:
        _save_csv(rows, csv_out)
        if verbose:
            print(f"\n  CSV → {csv_out}")

    return rows


# ==============================================================================
# COMBINED CRITICALITY PLOT (DP + Forest Fire side by side)
# ==============================================================================

def criticality_plot(dp_rows, ff_rows, dp_cfg=None, ff_cfg=None, save_path=None):
    """
    Combined six-panel figure: DP on left, Forest Fire on right.
    Designed to show the contrast between tuned criticality (DP)
    and self-organized criticality (Forest Fire).
    """
    if dp_cfg is None: dp_cfg = DP_CFG.copy()
    if ff_cfg is None: ff_cfg = FF_CFG.copy()

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches

    PC    = dp_cfg['PC_EXACT']
    p_val = ff_cfg['P_TREE']

    # ── DP averages ───────────────────────────────────────────────────────────
    p_vals   = sorted(set(r['p'] for r in dp_rows))
    def dp_avg(key):
        return [float(np.mean([r[key] for r in dp_rows if r['p']==p]))
                for p in p_vals]
    def dp_sem(key):
        return [float(np.std([r[key] for r in dp_rows if r['p']==p]) /
                       np.sqrt(dp_cfg['N_SEEDS'])) for p in p_vals]

    dp_C    = dp_avg('score');      dp_Cerr = dp_sem('score')
    dp_dens = dp_avg('mean_density')
    dp_wH   = dp_avg('w_H');        dp_wT   = dp_avg('w_T')
    dp_wG   = dp_avg('w_G');        dp_wOPt = dp_avg('w_OP_t')

    dp_peak_p = p_vals[int(np.argmax(dp_C))]
    dp_peak_C = max(dp_C)
    dp_dist   = abs(dp_peak_p - PC)

    # ── FF averages ───────────────────────────────────────────────────────────
    pf_vals  = sorted(set(r['p_over_f'] for r in ff_rows))
    def ff_avg(key):
        return [float(np.mean([r[key] for r in ff_rows
                               if abs(r['p_over_f']-pf)<1e-9]))
                for pf in pf_vals]
    def ff_sem(key):
        return [float(np.std([r[key] for r in ff_rows
                              if abs(r['p_over_f']-pf)<1e-9]) /
                       np.sqrt(ff_cfg['N_SEEDS'])) for pf in pf_vals]

    ff_C    = ff_avg('score');      ff_Cerr = ff_sem('score')
    ff_dens = ff_avg('tree_density')
    ff_wH   = ff_avg('w_H');        ff_wT   = ff_avg('w_T')
    ff_wG   = ff_avg('w_G');        ff_wOPt = ff_avg('w_OP_t')

    ff_peak_pf = pf_vals[int(np.argmax(ff_C))]
    ff_peak_C  = max(ff_C)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('#0e0e16')
    fig.suptitle(
        'Phase Transitions — Directed Percolation vs Forest Fire\n'
        'Tuned Criticality (sharp spike)  vs  Self-Organized Criticality (broad plateau)',
        fontsize=13, fontweight='bold', color='white', y=0.98
    )
    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.52, wspace=0.38,
                           left=0.06, right=0.97)

    PANEL_BG = '#1a1a2a'
    AXIS_COL = '#ccccdd'
    GRID_COL = '#333344'
    TC_COL   = '#ff4444'

    def style_ax(ax):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=AXIS_COL)
        ax.xaxis.label.set_color(AXIS_COL)
        ax.yaxis.label.set_color(AXIS_COL)
        ax.title.set_color(AXIS_COL)
        for sp in ax.spines.values(): sp.set_color(GRID_COL)
        ax.grid(True, color=GRID_COL, lw=0.5, alpha=0.6)

    # ── Row 0: Hero panels ────────────────────────────────────────────────────
    # DP hero (spans 2 cols)
    ax_dp = fig.add_subplot(gs[0, 0:2])
    style_ax(ax_dp)
    p_arr = np.array(p_vals)
    ax_dp.fill_between(p_arr,
                       [c-e for c,e in zip(dp_C,dp_Cerr)],
                       [c+e for c,e in zip(dp_C,dp_Cerr)],
                       alpha=0.2, color='#2ecc71')
    ax_dp.errorbar(p_arr, dp_C, yerr=dp_Cerr, fmt='o-',
                   color='#2ecc71', ms=6, lw=2, elinewidth=1.2, capsize=4,
                   label='Composite C')
    ax_dp.axvspan(p_arr[0], PC,       alpha=0.07, color='#3498db',
                  label='Sub-critical (activity dies)')
    ax_dp.axvspan(PC,       p_arr[-1],alpha=0.07, color='#e74c3c',
                  label='Super-critical (activity survives)')
    ax_dp.axvline(PC, color=TC_COL, lw=2.5, ls='--', zorder=5,
                  label=f'p_c={PC} (Grassberger 1989)')
    ax_dp.axvline(dp_peak_p, color='#ffaa00', lw=1.8, ls=':', zorder=5,
                  label=f'C peak at p={dp_peak_p:.3f}')
    ymax_dp = max(dp_C)*1.35 if max(dp_C)>0 else 0.01
    ax_dp.set_ylim(-0.002, ymax_dp)
    ax_dp.set_xlabel('Activation probability  p', fontsize=10)
    ax_dp.set_ylabel('Composite C', fontsize=10)
    ax_dp.set_title('DIRECTED PERCOLATION — Tuned Critical Threshold\n'
                    'Prediction: sharp spike at p_c', fontsize=10, fontweight='bold')
    ax_dp.legend(fontsize=7, facecolor='#1a1a2a', labelcolor=AXIS_COL,
                 edgecolor=GRID_COL)

    # FF hero (spans 2 cols)
    ax_ff = fig.add_subplot(gs[0, 2:4])
    style_ax(ax_ff)
    pf_arr = np.array(pf_vals)
    ax_ff.fill_between(np.log10(pf_arr),
                       [c-e for c,e in zip(ff_C,ff_Cerr)],
                       [c+e for c,e in zip(ff_C,ff_Cerr)],
                       alpha=0.2, color='#f39c12')
    ax_ff.errorbar(np.log10(pf_arr), ff_C, yerr=ff_Cerr, fmt='s-',
                   color='#f39c12', ms=6, lw=2, elinewidth=1.2, capsize=4,
                   label='Composite C')
    # Shade SOC regime (p/f = 1-15, log10 = 0-1.18)
    ax_ff.axvspan(0, np.log10(15), alpha=0.1, color='#f39c12',
                  label='SOC regime (predicted)')
    ymax_ff = max(ff_C)*1.35 if max(ff_C)>0 else 0.01
    ax_ff.set_ylim(-0.002, ymax_ff)
    ax_ff.set_xlabel('log₁₀(p/f)  — environment richness ratio', fontsize=10)
    ax_ff.set_ylabel('Composite C', fontsize=10)
    ax_ff.set_title('FOREST FIRE — Self-Organized Criticality\n'
                    'Prediction: broad plateau at intermediate p/f', fontsize=10,
                    fontweight='bold')
    ax_ff.legend(fontsize=7, facecolor='#1a1a2a', labelcolor=AXIS_COL,
                 edgecolor=GRID_COL)

    # ── Row 1: Density + weights ──────────────────────────────────────────────
    ax_dp_d = fig.add_subplot(gs[1, 0])
    style_ax(ax_dp_d)
    ax_dp_d.plot(p_arr, dp_dens, 'o-', color='#3498db', ms=4, lw=1.5)
    ax_dp_d.axvline(PC, color=TC_COL, lw=2, ls='--')
    ax_dp_d.set_xlabel('p'); ax_dp_d.set_ylabel('Mean active density')
    ax_dp_d.set_title('DP: Activity Density vs p', fontsize=9)

    ax_dp_w = fig.add_subplot(gs[1, 1])
    style_ax(ax_dp_w)
    dp_wOPs = dp_avg('w_OP_s')
    ax_dp_w.plot(p_arr, dp_wH,  'o-', color='#e74c3c', ms=3, lw=1.3, label='w_H')
    ax_dp_w.plot(p_arr, dp_wOPs,'v-', color='#e67e22', ms=3, lw=1.3, label='w_OP_s')
    ax_dp_w.plot(p_arr, dp_wOPt,'s-', color='#9b59b6', ms=3, lw=1.3, label='w_OP_t')
    ax_dp_w.plot(p_arr, dp_wT,  '^-', color='#f39c12', ms=3, lw=1.3, label='w_T')
    ax_dp_w.plot(p_arr, dp_wG,  'D-', color='#1abc9c', ms=3, lw=1.3, label='w_G')
    ax_dp_w.axvline(PC, color=TC_COL, lw=2, ls='--')
    ax_dp_w.set_xlabel('p'); ax_dp_w.set_ylabel('Weight')
    ax_dp_w.set_title('DP: Metric Weights vs p', fontsize=9)
    ax_dp_w.legend(fontsize=6, facecolor='#1a1a2a', labelcolor=AXIS_COL,
                   edgecolor=GRID_COL, ncol=2)

    ax_ff_d = fig.add_subplot(gs[1, 2])
    style_ax(ax_ff_d)
    ax_ff_d.plot(np.log10(pf_arr), ff_dens, 's-', color='#2ecc71', ms=4, lw=1.5)
    ax_ff_d.axvspan(0, np.log10(15), alpha=0.1, color='#f39c12')
    ax_ff_d.set_xlabel('log₁₀(p/f)'); ax_ff_d.set_ylabel('Mean tree density')
    ax_ff_d.set_title('FF: Tree Density vs p/f', fontsize=9)

    ax_ff_w = fig.add_subplot(gs[1, 3])
    style_ax(ax_ff_w)
    ff_wOPs = ff_avg('w_OP_s')
    ax_ff_w.plot(np.log10(pf_arr), ff_wH,  'o-', color='#e74c3c', ms=3, lw=1.3,
                 label='w_H')
    ax_ff_w.plot(np.log10(pf_arr), ff_wOPs,'v-', color='#e67e22', ms=3, lw=1.3,
                 label='w_OP_s')
    ax_ff_w.plot(np.log10(pf_arr), ff_wOPt,'s-', color='#9b59b6', ms=3, lw=1.3,
                 label='w_OP_t')
    ax_ff_w.plot(np.log10(pf_arr), ff_wT,  '^-', color='#f39c12', ms=3, lw=1.3,
                 label='w_T')
    ax_ff_w.plot(np.log10(pf_arr), ff_wG,  'D-', color='#1abc9c', ms=3, lw=1.3,
                 label='w_G')
    ax_ff_w.axvspan(0, np.log10(15), alpha=0.1, color='#f39c12')
    ax_ff_w.set_xlabel('log₁₀(p/f)'); ax_ff_w.set_ylabel('Weight')
    ax_ff_w.set_title('FF: Metric Weights vs p/f', fontsize=9)
    ax_ff_w.legend(fontsize=6, facecolor='#1a1a2a', labelcolor=AXIS_COL,
                   edgecolor=GRID_COL, ncol=2)

    # ── Row 2: Verdict panels ─────────────────────────────────────────────────
    ax_dp_v = fig.add_subplot(gs[2, 0:2])
    ax_dp_v.set_facecolor(PANEL_BG)
    ax_dp_v.axis('off')

    dp_verdict = 'CONFIRMED' if dp_dist <= dp_cfg['P_STEP']*1.5 else \
                 f'OFFSET {dp_dist:.4f}'
    dp_color   = '#2ecc71' if dp_dist <= dp_cfg['P_STEP']*1.5 else '#e74c3c'

    dp_summary = (
        f"DIRECTED PERCOLATION — VERDICT\n"
        f"{'─'*36}\n\n"
        f"p_c (Grassberger 1989)  = {PC}\n"
        f"C peaks at p            = {dp_peak_p:.4f}\n"
        f"Distance to p_c         = {dp_dist:.4f}\n"
        f"Peak C                  = {dp_peak_C:.5f}\n\n"
        f"Hypothesis:  sharp C spike at p_c\n"
        f"Null:        no peak at p_c\n\n"
        f"VERDICT: {dp_verdict}\n\n"
        f"Note: DP universality class covers\n"
        f"epidemic spreading, turbulence onset,\n"
        f"catalytic reactions, neural avalanches."
    )
    ax_dp_v.text(0.05, 0.95, dp_summary, transform=ax_dp_v.transAxes,
                 fontsize=9, verticalalignment='top',
                 fontfamily='monospace', color=AXIS_COL,
                 bbox=dict(boxstyle='round', facecolor='#0e0e16', alpha=0.9,
                           edgecolor=dp_color, linewidth=2.5))

    ax_ff_v = fig.add_subplot(gs[2, 2:4])
    ax_ff_v.set_facecolor(PANEL_BG)
    ax_ff_v.axis('off')

    # Check if C is elevated across SOC range vs extremes
    soc_C    = float(np.mean([r['score'] for r in ff_rows
                              if 1 <= r['p_over_f'] <= 15]))
    low_C    = float(np.mean([r['score'] for r in ff_rows
                              if r['p_over_f'] < 1]))
    high_C   = float(np.mean([r['score'] for r in ff_rows
                              if r['p_over_f'] > 15]))
    soc_elevated = soc_C > max(low_C, high_C) * 1.5
    ff_verdict   = 'CONFIRMED' if soc_elevated else 'NOT CONFIRMED'
    ff_color     = '#2ecc71' if soc_elevated else '#e74c3c'

    ff_summary = (
        f"FOREST FIRE (SOC) — VERDICT\n"
        f"{'─'*36}\n\n"
        f"SOC regime C (p/f 1-15) = {soc_C:.5f}\n"
        f"Fire-dominated C (p/f<1)= {low_C:.5f}\n"
        f"Growth-dominated C(p/f>15)={high_C:.5f}\n\n"
        f"SOC elevation ratio     = {soc_C/max(max(low_C,high_C),1e-9):.2f}×\n"
        f"C peak at p/f           = {ff_peak_pf:.1f}\n"
        f"Peak C                  = {ff_peak_C:.5f}\n\n"
        f"Hypothesis:  broad C plateau at\n"
        f"             intermediate p/f (SOC)\n"
        f"Null:        C tracks tree density\n\n"
        f"VERDICT: {ff_verdict}"
    )
    ax_ff_v.text(0.05, 0.95, ff_summary, transform=ax_ff_v.transAxes,
                 fontsize=9, verticalalignment='top',
                 fontfamily='monospace', color=AXIS_COL,
                 bbox=dict(boxstyle='round', facecolor='#0e0e16', alpha=0.9,
                           edgecolor=ff_color, linewidth=2.5))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#0e0e16')
        plt.close()
        print(f"  Plot → {save_path}")
    else:
        plt.show()


# ==============================================================================
# SUBSTRATE 10 — SCHELLING SEGREGATION MODEL
# ==============================================================================
#
# BACKGROUND:
#   Schelling (1971) showed that mild individual preferences for similarity
#   can produce macro-level segregation that no individual intended.
#   Canonical model in agent-based social simulation and complexity science.
#
# HYPOTHESIS:
#   Composite C will be elevated at intermediate similarity thresholds
#   (~30-50%), where the grid exhibits active cluster formation and
#   dissolution — neither stably mixed nor crystallised into frozen
#   segregated patches.
#
#   Both group mappings (Group A = 1, Group B = 1) should show the same
#   signal. If the signal is real it is substrate-level, not an artifact
#   of which group we observe.
#
# NULL HYPOTHESIS:
#   C tracks monotonically with threshold (rising with segregation order,
#   or falling with entropy), or peaks at extreme threshold values rather
#   than the known transition band.
#
# EXTERNAL GROUND TRUTH:
#   Zhang (2004), Pancs & Vriend (2007), and follow-up computational
#   studies consistently place the mixed→segregated transition at
#   similarity threshold 30-50%. This is established by sociological
#   segregation indices (e.g. dissimilarity index), completely independent
#   of information theory.
#
# NOTE ON TRANSITION TYPE:
#   Unlike Ising (sharp critical point), Schelling has a gradual crossover
#   whose location depends on density and grid size. We expect a BROAD
#   elevated C band rather than a sharp spike — more like Forest Fire SOC
#   than Ising. The verdict criterion is therefore "elevated in 30-50% band"
#   not "peaks at single precise threshold."
#
# SYMBOLIZATION — TWO MAPPINGS:
#   Mapping A — Group A field:  agent_A=1, (agent_B, empty)=0
#   Mapping B — Group B field:  agent_B=1, (agent_A, empty)=0
#   Both run on identical simulations. Agreement confirms signal is real.
#
# ==============================================================================

SCHELLING_CFG = dict(
    GRID         = 64,       # 64×64 grid
    STEPS        = 500,      # simulation steps
    BURNIN       = 50,       # discard early transient
    WINDOW       = 200,      # measurement window
    N_SEEDS      = 5,
    DENSITY      = 0.80,     # fraction of cells occupied (0.2 empty)
    # Threshold sweep — similarity fraction required to be satisfied
    THRESH_MIN   = 0.10,
    THRESH_MAX   = 0.80,
    THRESH_STEP  = 0.05,
    # Known transition band from literature
    TRANSITION_LOW  = 0.30,
    TRANSITION_HIGH = 0.50,
)

# Cell states
_EMPTY = 0
_A     = 1
_B     = 2


def _schelling_run(threshold, cfg, seed=42):
    """
    Schelling segregation model on 2D toroidal grid.

    Two groups (A and B) occupy DENSITY fraction of cells, remainder empty.
    At each step:
      - Find all unsatisfied agents (similarity fraction < threshold)
      - Randomly move each to a random empty cell
      - Record grid state

    Similarity fraction = (same-group neighbours) / (total occupied neighbours)
    Uses Moore neighbourhood (8 cells).

    Returns:
      hist_A: (STEPS, G*G) binary — group A field (mapping A)
      hist_B: (STEPS, G*G) binary — group B field (mapping B)
    """
    rng = np.random.default_rng(seed)
    G   = cfg['GRID']
    N   = G * G

    # Initialise: fill DENSITY fraction with equal A and B
    n_agents  = int(N * cfg['DENSITY'])
    n_A       = n_agents // 2
    n_B       = n_agents - n_A

    flat = np.zeros(N, dtype=np.int8)
    occupied = rng.choice(N, size=n_agents, replace=False)
    flat[occupied[:n_A]] = _A
    flat[occupied[n_A:]] = _B
    grid = flat.reshape(G, G)

    hist_A = []
    hist_B = []

    for t in range(cfg['STEPS']):
        hist_A.append((grid == _A).astype(np.int8).ravel())
        hist_B.append((grid == _B).astype(np.int8).ravel())

        # Compute neighbour counts using convolution
        padded   = np.pad(grid, 1, mode='wrap')
        # Count A neighbours
        a_nbrs = np.zeros((G, G), dtype=np.int32)
        b_nbrs = np.zeros((G, G), dtype=np.int32)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                slab = padded[1+di:G+1+di, 1+dj:G+1+dj]
                a_nbrs += (slab == _A).astype(np.int32)
                b_nbrs += (slab == _B).astype(np.int32)

        total_nbrs = a_nbrs + b_nbrs

        # Similarity fraction per cell
        same = np.where(grid == _A, a_nbrs,
               np.where(grid == _B, b_nbrs, 0))
        # Avoid division by zero — isolated agents (no neighbours) are satisfied
        with np.errstate(divide='ignore', invalid='ignore'):
            similarity = np.where(total_nbrs > 0,
                                  same.astype(np.float32) / total_nbrs,
                                  1.0)

        # Unsatisfied: occupied AND similarity < threshold
        unsatisfied = ((grid != _EMPTY) &
                       (similarity < threshold))

        # Empty cells
        empty_mask  = (grid == _EMPTY)
        empty_cells = np.argwhere(empty_mask)

        if len(empty_cells) == 0:
            continue

        # Move each unsatisfied agent to a random empty cell
        unsat_cells = np.argwhere(unsatisfied)
        rng.shuffle(unsat_cells)

        new_grid = grid.copy()
        # Track which cells are now empty (updated dynamically)
        is_empty = empty_mask.copy()

        for (i, j) in unsat_cells:
            empty_now = np.argwhere(is_empty)
            if len(empty_now) == 0:
                break
            # Pick random empty destination
            idx  = rng.integers(len(empty_now))
            ni, nj = empty_now[idx]
            # Move agent
            new_grid[ni, nj] = new_grid[i, j]
            new_grid[i, j]   = _EMPTY
            is_empty[ni, nj] = False
            is_empty[i, j]   = True

        grid = new_grid

    return (np.array(hist_A, dtype=np.int8),
            np.array(hist_B, dtype=np.int8))


def _schelling_metrics(history, cfg):
    """Apply full framework composite to a binary Schelling field history."""
    burnin = cfg['BURNIN']
    window = cfg['WINDOW']
    mH, sH       = _entropy_stats(history, burnin, window)
    op_up, op_dn = _opacity_both(history,  burnin, window)
    mi1, decay   = _opacity_temporal(history, burnin, window)
    tc           = _tcomp(history, burnin, window)
    post         = history[burnin:burnin + window]
    gz           = len(zlib.compress(post.tobytes(), 6)) / max(len(post.tobytes()), 1)
    C = composite(mH, sH, op_up, op_dn, mi1, decay, tc, gz)
    return dict(
        mean_H=mH, std_H=sH, op_up=op_up, op_dn=op_dn,
        mi1=mi1, decay=decay, tcomp=tc, gzip=gz,
        w_H    = weight_H(mH, sH),
        w_OP_s = weight_opacity_spatial(op_up, op_dn),
        w_OP_t = weight_opacity_temporal(mi1, decay),
        w_T    = weight_tcomp(tc),
        w_G    = weight_gzip(gz),
        score  = C,
    )


def schelling_sweep(cfg=None, csv_out=None, verbose=True):
    """
    Sweep similarity threshold, compute C for both group mappings.
    Returns list of result rows.
    """
    if cfg is None:
        cfg = SCHELLING_CFG.copy()

    thresholds = np.round(np.arange(cfg['THRESH_MIN'],
                                    cfg['THRESH_MAX'] + 1e-9,
                                    cfg['THRESH_STEP']), 3)
    rows = []

    if verbose:
        print(f"\n{'─'*65}")
        print(f"Schelling Segregation — threshold sweep")
        print(f"  Grid: {cfg['GRID']}×{cfg['GRID']}  "
              f"Steps: {cfg['STEPS']}  Seeds: {cfg['N_SEEDS']}")
        print(f"  Threshold: {cfg['THRESH_MIN']} → {cfg['THRESH_MAX']}  "
              f"step={cfg['THRESH_STEP']}")
        print(f"  Density: {cfg['DENSITY']}  "
              f"Transition band: {cfg['TRANSITION_LOW']}–{cfg['TRANSITION_HIGH']}")
        print(f"{'─'*65}")
        print(f"  {'thresh':>7}  {'regime':14}  "
              f"{'C_A':>8}  {'C_B':>8}  {'seg_A':>7}")

    for thresh in thresholds:
        regime = ('mixed'       if thresh < cfg['TRANSITION_LOW']  else
                  'transition'  if thresh <= cfg['TRANSITION_HIGH'] else
                  'segregated')

        sr_A, sr_B, seg_vals = [], [], []

        for s in range(cfg['N_SEEDS']):
            hA, hB = _schelling_run(thresh, cfg, seed=s)

            mA = _schelling_metrics(hA, cfg)
            mA.update(dict(threshold=float(thresh), seed=s,
                           mapping='group_A', regime=regime))
            rows.append(mA)
            sr_A.append(mA['score'])

            mB = _schelling_metrics(hB, cfg)
            mB.update(dict(threshold=float(thresh), seed=s,
                           mapping='group_B', regime=regime))
            rows.append(mB)
            sr_B.append(mB['score'])

            # Segregation index: mean fraction of same-group neighbours
            # Measured on final grid state
            seg_vals.append(float(hA[-1].mean()))

        avg_A   = float(np.mean(sr_A))
        avg_B   = float(np.mean(sr_B))
        avg_seg = float(np.mean(seg_vals))
        marker  = ' ◄' if cfg['TRANSITION_LOW'] <= thresh <= cfg['TRANSITION_HIGH'] else ''

        if verbose:
            print(f"  t={thresh:.2f}  [{regime:12s}]  "
                  f"C_A={avg_A:.5f}  C_B={avg_B:.5f}  "
                  f"A_dens={avg_seg:.3f}{marker}")

    if csv_out:
        _save_csv(rows, csv_out)
        if verbose:
            print(f"\n  CSV → {csv_out}")

    return rows


def schelling_plot(rows, cfg=None, save_path=None):
    """Five-panel analysis figure for Schelling sweep."""
    if cfg is None:
        cfg = SCHELLING_CFG.copy()

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches

    TL = cfg['TRANSITION_LOW']
    TH = cfg['TRANSITION_HIGH']

    rows_A = [r for r in rows if r['mapping'] == 'group_A']
    rows_B = [r for r in rows if r['mapping'] == 'group_B']
    thresholds = sorted(set(r['threshold'] for r in rows_A))

    def avg(rlist, key):
        return [float(np.mean([r[key] for r in rlist
                               if r['threshold']==t])) for t in thresholds]
    def sem(rlist, key):
        return [float(np.std([r[key] for r in rlist
                              if r['threshold']==t]) /
                       np.sqrt(cfg['N_SEEDS'])) for t in thresholds]

    CA    = avg(rows_A, 'score');    CA_err  = sem(rows_A, 'score')
    CB    = avg(rows_B, 'score');    CB_err  = sem(rows_B, 'score')
    wH_A  = avg(rows_A, 'w_H')
    wOPs_A= avg(rows_A, 'w_OP_s')
    wOPt_A= avg(rows_A, 'w_OP_t')
    wT_A  = avg(rows_A, 'w_T')
    wG_A  = avg(rows_A, 'w_G')
    densA = avg(rows_A, 'mean_H')   # entropy as proxy for density pattern

    T_arr    = np.array(thresholds)
    peak_A   = thresholds[int(np.argmax(CA))]
    peak_B   = thresholds[int(np.argmax(CB))]
    peak_CA  = max(CA)
    peak_CB  = max(CB)

    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor('#0e0e16')
    fig.suptitle(
        'Schelling Segregation Model — Complexity vs Similarity Threshold\n'
        f'Transition band: {TL}–{TH} (Zhang 2004, Pancs & Vriend 2007)   |   '
        f'Grid: {cfg["GRID"]}×{cfg["GRID"]}   Density: {cfg["DENSITY"]}   '
        f'Seeds: {cfg["N_SEEDS"]}',
        fontsize=12, fontweight='bold', color='white', y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    PANEL_BG = '#1a1a2a'
    AXIS_COL = '#ccccdd'
    GRID_COL = '#333344'

    def style_ax(ax):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=AXIS_COL)
        ax.xaxis.label.set_color(AXIS_COL)
        ax.yaxis.label.set_color(AXIS_COL)
        ax.title.set_color(AXIS_COL)
        for sp in ax.spines.values(): sp.set_color(GRID_COL)
        ax.grid(True, color=GRID_COL, lw=0.5, alpha=0.6)

    # ── Panel 1: Hero — both C curves ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0:2])
    style_ax(ax1)

    ax1.fill_between(T_arr,
                     [c-e for c,e in zip(CA,CA_err)],
                     [c+e for c,e in zip(CA,CA_err)],
                     alpha=0.15, color='#e74c3c')
    ax1.fill_between(T_arr,
                     [c-e for c,e in zip(CB,CB_err)],
                     [c+e for c,e in zip(CB,CB_err)],
                     alpha=0.15, color='#3498db')
    ax1.errorbar(T_arr, CA, yerr=CA_err, fmt='o-',
                 color='#e74c3c', ms=6, lw=2, elinewidth=1.2, capsize=4,
                 label=f'C — Group A (peak={peak_A:.2f})')
    ax1.errorbar(T_arr, CB, yerr=CB_err, fmt='s--',
                 color='#3498db', ms=6, lw=2, elinewidth=1.2, capsize=4,
                 label=f'C — Group B (peak={peak_B:.2f})')

    # Shade regimes
    ax1.axvspan(T_arr[0], TL,        alpha=0.07, color='#3498db',
                label='Mixed regime')
    ax1.axvspan(TL,       TH,        alpha=0.12, color='#f39c12',
                label='Transition band (literature)')
    ax1.axvspan(TH,       T_arr[-1], alpha=0.07, color='#e74c3c',
                label='Segregated regime')

    ymax = max(max(CA), max(CB)) * 1.35 if max(max(CA),max(CB)) > 0 else 0.01
    ax1.set_ylim(-0.002, ymax)
    ax1.set_xlabel('Similarity Threshold', fontsize=11)
    ax1.set_ylabel('Composite C', fontsize=11)
    ax1.set_title('Composite C vs Similarity Threshold — Both Group Mappings',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8, facecolor='#1a1a2a', labelcolor=AXIS_COL,
               edgecolor=GRID_COL, loc='upper right')

    # ── Panel 2: Segregation dynamics ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2)
    # Use mean_H as a proxy for how mixed/segregated the field is
    ax2.plot(T_arr, wH_A, 'o-', color='#f39c12', ms=5, lw=1.8,
             label='mean entropy (w_H numerator)')
    ax2.axvspan(TL, TH, alpha=0.12, color='#f39c12')
    ax2.set_xlabel('Similarity Threshold')
    ax2.set_ylabel('Mean Spatial Entropy')
    ax2.set_title('Spatial Entropy vs Threshold\n(proxy for mix/segregation state)',
                  fontsize=10)
    ax2.legend(fontsize=8, facecolor='#1a1a2a', labelcolor=AXIS_COL,
               edgecolor=GRID_COL)

    # ── Panel 3: Metric weights breakdown ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3)
    ax3.plot(T_arr, wH_A,   'o-', color='#e74c3c', ms=4, lw=1.5, label='w_H')
    ax3.plot(T_arr, wOPs_A, 'v-', color='#e67e22', ms=4, lw=1.5, label='w_OP_s')
    ax3.plot(T_arr, wOPt_A, 's-', color='#9b59b6', ms=4, lw=1.5, label='w_OP_t')
    ax3.plot(T_arr, wT_A,   '^-', color='#f39c12', ms=4, lw=1.5, label='w_T')
    ax3.plot(T_arr, wG_A,   'D-', color='#1abc9c', ms=4, lw=1.5, label='w_G')
    ax3.axvspan(TL, TH, alpha=0.12, color='#f39c12')
    ax3.set_xlabel('Similarity Threshold')
    ax3.set_ylabel('Weight value')
    ax3.set_title('Metric Weights vs Threshold\n(Group A mapping)', fontsize=10)
    ax3.legend(fontsize=7, facecolor='#1a1a2a', labelcolor=AXIS_COL,
               edgecolor=GRID_COL, ncol=2)

    # ── Panel 4: Regime bar comparison ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4)
    regimes   = ['Mixed\n(t<0.30)', 'Transition\n(0.30-0.50)', 'Segregated\n(t>0.50)']
    mixed_A   = float(np.mean([r['score'] for r in rows_A
                               if r['threshold'] < TL]))
    trans_A   = float(np.mean([r['score'] for r in rows_A
                               if TL <= r['threshold'] <= TH]))
    seg_A     = float(np.mean([r['score'] for r in rows_A
                               if r['threshold'] > TH]))
    mixed_B   = float(np.mean([r['score'] for r in rows_B
                               if r['threshold'] < TL]))
    trans_B   = float(np.mean([r['score'] for r in rows_B
                               if TL <= r['threshold'] <= TH]))
    seg_B     = float(np.mean([r['score'] for r in rows_B
                               if r['threshold'] > TH]))
    x = np.arange(3); w = 0.35
    ax4.bar(x - w/2, [mixed_A, trans_A, seg_A], width=w,
            color='#e74c3c', alpha=0.85, edgecolor='k', lw=0.7, label='Group A')
    ax4.bar(x + w/2, [mixed_B, trans_B, seg_B], width=w,
            color='#3498db', alpha=0.85, edgecolor='k', lw=0.7, label='Group B')
    ax4.set_xticks(x)
    ax4.set_xticklabels(regimes, fontsize=8)
    ax4.set_ylabel('Mean C')
    ax4.set_title('C by Regime — Both Mappings', fontsize=10)
    ax4.legend(fontsize=8, facecolor='#1a1a2a', labelcolor=AXIS_COL,
               edgecolor=GRID_COL)

    # ── Panel 5: Verdict ──────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(PANEL_BG)
    ax5.axis('off')

    in_band_A = TL <= peak_A <= TH
    in_band_B = TL <= peak_B <= TH
    elevated  = trans_A > max(mixed_A, seg_A) * 1.3
    agree     = abs(peak_A - peak_B) < cfg['THRESH_STEP'] * 2

    if (in_band_A or in_band_B) and elevated:
        verdict = 'CONFIRMED'
        vcolor  = '#2ecc71'
    elif elevated:
        verdict = 'PARTIALLY CONFIRMED\n(elevated but offset)'
        vcolor  = '#f39c12'
    else:
        verdict = 'NOT CONFIRMED'
        vcolor  = '#e74c3c'

    agree_str = 'AGREE' if agree else 'DISAGREE'
    agree_col = '#2ecc71' if agree else '#f39c12'

    summary = (
        f"RESULTS SUMMARY\n"
        f"{'─'*32}\n\n"
        f"Transition band (lit.) = {TL}–{TH}\n\n"
        f"Group A:\n"
        f"  C peaks at t = {peak_A:.2f}\n"
        f"  Peak C = {peak_CA:.5f}\n"
        f"  In band: {'YES' if in_band_A else 'NO'}\n\n"
        f"Group B:\n"
        f"  C peaks at t = {peak_B:.2f}\n"
        f"  Peak C = {peak_CB:.5f}\n"
        f"  In band: {'YES' if in_band_B else 'NO'}\n\n"
        f"Mappings: {agree_str}\n\n"
        f"Transition elevation:\n"
        f"  trans={trans_A:.5f}\n"
        f"  mixed={mixed_A:.5f}\n"
        f"  seg  ={seg_A:.5f}\n\n"
        f"{'─'*32}\n"
        f"H:  C elevated at 30-50%\n"
        f"H0: C monotonic or wrong peak\n\n"
        f"VERDICT: {verdict}"
    )
    ax5.text(0.04, 0.97, summary, transform=ax5.transAxes, fontsize=8.5,
             verticalalignment='top', fontfamily='monospace', color=AXIS_COL,
             bbox=dict(boxstyle='round', facecolor='#0e0e16', alpha=0.9,
                       edgecolor=vcolor, linewidth=2.5))

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='#0e0e16')
        plt.close()
        print(f"  Plot → {save_path}")
    else:
        plt.show()


# ==============================================================================
# CLI
# ==============================================================================

def main():
    p = argparse.ArgumentParser(
        description='Complexity Framework v9 — unified framework + spatial game theory',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('substrate', nargs='?', default='eca',
                   choices=['eca','k3','life','nbody','pd','ising','sir',
                            'dp','ff','schelling','all'])
    p.add_argument('--csv',       metavar='FILE')
    p.add_argument('--no-plot',   action='store_true')
    p.add_argument('--save-plot', metavar='FILE')
    p.add_argument('--seeds',     type=int)
    # ECA / k3
    p.add_argument('--width',      type=int)
    p.add_argument('--ic',         choices=['random','single'], default='random')
    p.add_argument('--density',    type=float, default=0.5)
    p.add_argument('--diagnose',   type=int, metavar='RULE')
    p.add_argument('--scale-test', action='store_true')
    p.add_argument('--widths',     type=int, nargs='+')
    p.add_argument('--scale-csv',  metavar='FILE')
    # Life
    p.add_argument('--grid',       type=int)
    p.add_argument('--list-rules', action='store_true')
    # N-body
    p.add_argument('--scan',       action='store_true')
    p.add_argument('--heatmap',    metavar='FILE')
    p.add_argument('--alpha',      type=float)
    p.add_argument('--alpha-s',    type=float)
    # Spatial game theory (PD)
    p.add_argument('--game',       choices=['prisoner','stag','minority'],
                   default='prisoner',
                   help='Which spatial game to run (default: prisoner)')
    p.add_argument('--T-min',      type=float, dest='T_min')
    p.add_argument('--T-max',      type=float, dest='T_max')
    p.add_argument('--T-step',     type=float, dest='T_step')
    p.add_argument('--fine-sweep', action='store_true',
                   help='Run fine sweep around T=1.70-1.95')
    p.add_argument('--controls',   action='store_true',
                   help='Run Stag Hunt and Minority game controls')
    p.add_argument('--lambda-analysis', action='store_true',
                   help='Compute Langton λ correlation (requires ECA results)')
    # Ising
    p.add_argument('--T-min-ising', type=float, dest='T_min_ising')
    p.add_argument('--T-max-ising', type=float, dest='T_max_ising')
    p.add_argument('--T-step-ising',type=float, dest='T_step_ising')
    p.add_argument('--ising-workers', type=int, default=None, dest='ising_workers',
                   help='Parallel Ising T points via subprocesses (default: min(#T, CPUs); 1=sequential)')
    # SIR
    p.add_argument('--beta-min',  type=float, dest='beta_min')
    p.add_argument('--beta-max',  type=float, dest='beta_max')
    p.add_argument('--beta-step', type=float, dest='beta_step')
    p.add_argument('--gamma',     type=float, dest='gamma')
    args = p.parse_args()

    # ── ECA ──────────────────────────────────────────────────────────────────
    if args.substrate in ('eca', 'all'):
        cfg = ECA_CFG.copy()
        if args.width:  cfg['WIDTH']   = args.width
        if args.seeds:  cfg['N_SEEDS'] = args.seeds
        cfg['DENSITY'] = None if args.ic == 'single' else args.density
        if args.diagnose is not None:
            eca_diagnose(args.diagnose, cfg); return
        if args.scale_test:
            eca_scale_test(cfg, widths=args.widths, csv_out=args.scale_csv)
            return
        results = eca_run_all(cfg, verbose=True)
        eca_print_results(results,
            label=f"IC={'random' if cfg['DENSITY'] else 'single'}  W={cfg['WIDTH']}")
        if args.csv: _save_csv(results, args.csv)
        # Optionally run Langton λ analysis on ECA results
        if args.lambda_analysis:
            pd_langton_lambda(results, verbose=True)

    # ── k=3 CA ───────────────────────────────────────────────────────────────
    if args.substrate in ('k3', 'all'):
        cfg = K3_CFG.copy()
        if args.width:  cfg['WIDTH']   = args.width
        if args.seeds:  cfg['N_SEEDS'] = args.seeds
        cfg['DENSITY'] = None if args.ic == 'single' else args.density
        results = k3_run_all(cfg, verbose=True)
        if args.csv: _save_csv(results, args.csv.replace('.csv','_k3.csv'))

    # ── 2D Life-like ─────────────────────────────────────────────────────────
    if args.substrate in ('life', 'all'):
        if args.list_rules:
            print("\nLife-like rules:")
            for name, birth, survive, cls in LIFE_RULES:
                print(f"  {name:14s}  B={''.join(str(b) for b in sorted(birth)):8s}"
                      f"  S={''.join(str(s) for s in sorted(survive)):10s}  [{cls}]")
            return
        cfg = LIFE_CFG.copy()
        if args.grid:  cfg['GRID']    = args.grid
        if args.seeds: cfg['N_SEEDS'] = args.seeds
        results = life_run_all(cfg, verbose=True)
        if args.csv: _save_csv(results, args.csv.replace('.csv','_life.csv'))

    # ── N-body ───────────────────────────────────────────────────────────────
    if args.substrate in ('nbody', 'all'):
        cfg = NBODY_CFG.copy()
        if args.seeds: cfg['SCAN_SEEDS'] = list(range(args.seeds))
        if args.heatmap:
            try:
                import matplotlib; import matplotlib.pyplot as plt
                import matplotlib.colors as mcolors
                from matplotlib.patches import Rectangle
                if args.save_plot: matplotlib.use('Agg')
                with open(args.heatmap) as f: data = json.load(f)
                cmap_data = np.array(data['complexity_map'])
                AR  = np.array(data['alpha_range'])
                ASR = np.array(data['alphas_range'])
                fig, ax = plt.subplots(figsize=(7, 6), facecolor='#0e0e16')
                cm = mcolors.LinearSegmentedColormap.from_list(
                    'cx', ['#0e0e16','#1a2a3a','#1D9E75','#ffffff'], N=256)
                ax.set_facecolor('#0e0e16')
                for sp in ax.spines.values(): sp.set_color('#333344')
                ax.tick_params(colors='#888899')
                im = ax.imshow(cmap_data.T, origin='lower', aspect='auto',
                               cmap=cm, extent=[AR[0],AR[-1],ASR[0],ASR[-1]])
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.set_xlabel('α (EM coupling)', color='#888899')
                ax.set_ylabel('αs (strong force)', color='#888899')
                ax.set_title('N-body Complexity Landscape (v9)', color='#aaaacc')
                ax.add_patch(Rectangle((0.5,0.5), 1.5, 1.5, fill=False,
                             edgecolor='#7F77DD', lw=1.5, ls='--', zorder=5,
                             label='Adams island'))
                ax.plot(1.0, 1.0, 'o', color='#1D9E75', ms=9,
                        mec='white', mew=1.2, label='Our universe', zorder=6)
                ax.legend(fontsize=8, facecolor='#1a1a2a',
                          labelcolor='#ccccdd', edgecolor='#333344')
                plt.colorbar(im, ax=ax).ax.tick_params(colors='#888899')
                plt.tight_layout()
                if args.save_plot:
                    plt.savefig(args.save_plot, dpi=130,
                                facecolor='#0e0e16', bbox_inches='tight')
                    print(f"  Plot → {args.save_plot}")
                else:
                    plt.show()
            except ImportError:
                print("  pip install matplotlib")
            return
        if args.scan:
            csv_out  = args.csv or 'nbody_scan.csv'
            json_out = (args.csv.replace('.csv','.json')
                        if args.csv else 'nbody_scan.json')
            _, cmap, AR, ASR = nbody_scan(cfg, csv_out=csv_out,
                                           json_out=json_out, verbose=True)
            return
        # Single-point evaluation
        alpha   = args.alpha   or 1.0
        alpha_s = args.alpha_s or 1.0
        print(f"\nN-body single point: α={alpha:.3f}  αs={alpha_s:.3f}")
        for s in cfg['SCAN_SEEDS']:
            m = _nbody_metrics(_nbody_run(alpha, alpha_s, cfg, seed=s), cfg)
            print(f"  seed {s}: C={m['score']:.4f}  "
                  f"H={m['mean_H']:.3f}  op↑={m['opacity_up']:.3f}  "
                  f"op↓={m['opacity_down']:.3f}  "
                  f"MI₁={m['opacity_temp_mi1']:.3f}  "
                  f"wOPt={m['w_OP_t']:.3f}  "
                  f"tc={m['tcomp']:.3f}  gz={m['gzip']:.3f}")

    # ── Spatial game theory ───────────────────────────────────────────────────
    if args.substrate in ('pd', 'all'):
        cfg = PD_CFG.copy()
        if args.seeds: cfg['N_SEEDS'] = args.seeds
        if args.grid:  cfg['GRID']    = args.grid
        if args.T_min is not None: cfg['T_MIN']       = args.T_min
        if args.T_max is not None: cfg['T_MAX']       = args.T_max
        if args.T_step is not None: cfg['T_STEP']     = args.T_step

        if args.controls:
            # Run Stag Hunt + Minority game controls
            csv_out = args.csv.replace('.csv', '_controls.csv') if args.csv else None
            pd_controls(cfg, csv_out=csv_out, verbose=True)

        elif args.game == 'prisoner' or args.substrate == 'pd':
            # Default: Prisoner's Dilemma sweep
            csv_out = args.csv or 'spatial_pd_results.csv'
            rows = pd_sweep(cfg, fine=args.fine_sweep,
                            csv_out=csv_out, verbose=True)

            # Optionally also run controls
            if args.controls:
                ctrl_csv = csv_out.replace('.csv', '_controls.csv')
                pd_controls(cfg, csv_out=ctrl_csv, verbose=True)

        elif args.game == 'stag':
            print("\nRunning Stag Hunt control...")
            cfg_ctrl = cfg.copy()
            rows = []
            for alpha in np.round(np.arange(0.1, 1.05, 0.1), 2):
                def pf(g, nb, a=float(alpha)): return _stag_payoff(g, nb, a)
                row = pd_evaluate(pf, alpha, 'alpha', cfg_ctrl, verbose=True)
                if row:
                    row['game'] = 'stag_hunt'
                    rows.append(row)
            if args.csv: _save_csv(rows, args.csv)

        elif args.game == 'minority':
            print("\nRunning Minority game control...")
            cfg_ctrl = cfg.copy()
            rows = []
            for beta in np.round(np.arange(0.2, 1.25, 0.2), 2):
                def pf(g, nb, b=float(beta)): return _minority_payoff(g, nb, b)
                row = pd_evaluate(pf, beta, 'beta', cfg_ctrl, verbose=True)
                if row:
                    row['game'] = 'minority'
                    rows.append(row)
            if args.csv: _save_csv(rows, args.csv)

    # ── ECA ──────────────────────────────────────────────────────────────────
    if args.substrate in ('eca', 'all'):
        cfg = ECA_CFG.copy()
        if args.width:  cfg['WIDTH']   = args.width
        if args.seeds:  cfg['N_SEEDS'] = args.seeds
        cfg['DENSITY'] = None if args.ic == 'single' else args.density
        if args.diagnose is not None:
            eca_diagnose(args.diagnose, cfg); return
        if args.scale_test:
            eca_scale_test(cfg, widths=args.widths, csv_out=args.scale_csv)
            return
        results = eca_run_all(cfg, verbose=True)
        eca_print_results(results,
            label=f"IC={'random' if cfg['DENSITY'] else 'single'}  W={cfg['WIDTH']}")
        if args.csv: _save_csv(results, args.csv)

    # ── k=3 CA ───────────────────────────────────────────────────────────────
    if args.substrate in ('k3', 'all'):
        cfg = K3_CFG.copy()
        if args.width:  cfg['WIDTH']   = args.width
        if args.seeds:  cfg['N_SEEDS'] = args.seeds
        cfg['DENSITY'] = None if args.ic == 'single' else args.density
        results = k3_run_all(cfg, verbose=True)
        if args.csv: _save_csv(results, args.csv.replace('.csv','_k3.csv'))

    # ── 2D Life-like ─────────────────────────────────────────────────────────
    if args.substrate in ('life', 'all'):
        if args.list_rules:
            print("\nLife-like rules:")
            for name, birth, survive, cls in LIFE_RULES:
                print(f"  {name:14s}  B={''.join(str(b) for b in sorted(birth)):8s}"
                      f"  S={''.join(str(s) for s in sorted(survive)):10s}  [{cls}]")
            return
        cfg = LIFE_CFG.copy()
        if args.grid:  cfg['GRID']    = args.grid
        if args.seeds: cfg['N_SEEDS'] = args.seeds
        results = life_run_all(cfg, verbose=True)
        if args.csv: _save_csv(results, args.csv.replace('.csv','_life.csv'))

    # ── N-body ───────────────────────────────────────────────────────────────
    if args.substrate in ('nbody', 'all'):
        cfg = NBODY_CFG.copy()
        if args.seeds: cfg['SCAN_SEEDS'] = list(range(args.seeds))
        if args.heatmap:
            try:
                import matplotlib; import matplotlib.pyplot as plt
                import matplotlib.colors as mcolors
                from matplotlib.patches import Rectangle
                if args.save_plot: matplotlib.use('Agg')
                with open(args.heatmap) as f: data = json.load(f)
                cmap_data = np.array(data['complexity_map'])
                AR  = np.array(data['alpha_range'])
                ASR = np.array(data['alphas_range'])
                fig, ax = plt.subplots(figsize=(7, 6), facecolor='#0e0e16')
                cm = mcolors.LinearSegmentedColormap.from_list(
                    'cx', ['#0e0e16','#1a2a3a','#1D9E75','#ffffff'], N=256)
                ax.set_facecolor('#0e0e16')
                for sp in ax.spines.values(): sp.set_color('#333344')
                ax.tick_params(colors='#888899')
                im = ax.imshow(cmap_data.T, origin='lower', aspect='auto',
                               cmap=cm, extent=[AR[0],AR[-1],ASR[0],ASR[-1]])
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.set_xlabel('α (EM coupling)', color='#888899')
                ax.set_ylabel('αs (strong force)', color='#888899')
                ax.set_title('N-body Complexity Landscape (v5)', color='#aaaacc')
                ax.add_patch(Rectangle((0.5,0.5), 1.5, 1.5, fill=False,
                             edgecolor='#7F77DD', lw=1.5, ls='--', zorder=5,
                             label='Adams island'))
                ax.plot(1.0, 1.0, 'o', color='#1D9E75', ms=9,
                        mec='white', mew=1.2, label='Our universe', zorder=6)
                ax.legend(fontsize=8, facecolor='#1a1a2a',
                          labelcolor='#ccccdd', edgecolor='#333344')
                plt.colorbar(im, ax=ax).ax.tick_params(colors='#888899')
                plt.tight_layout()
                if args.save_plot:
                    plt.savefig(args.save_plot, dpi=130,
                                facecolor='#0e0e16', bbox_inches='tight')
                    print(f"  Plot → {args.save_plot}")
                else:
                    plt.show()
            except ImportError:
                print("  pip install matplotlib")
            return
        if args.scan:
            csv_out  = args.csv or 'nbody_scan.csv'
            json_out = (args.csv.replace('.csv','.json')
                        if args.csv else 'nbody_scan.json')
            _, cmap, AR, ASR = nbody_scan(cfg, csv_out=csv_out,
                                           json_out=json_out, verbose=True)
            return
        # Single-point evaluation
        alpha   = args.alpha   or 1.0
        alpha_s = args.alpha_s or 1.0
        print(f"\nN-body single point: α={alpha:.3f}  αs={alpha_s:.3f}")
        for s in cfg['SCAN_SEEDS']:
            m = _nbody_metrics(_nbody_run(alpha, alpha_s, cfg, seed=s), cfg)
            print(f"  seed {s}: C={m['score']:.4f}  "
                  f"H={m['mean_H']:.3f}  op↑={m['opacity_up']:.3f}  "
                  f"op↓={m['opacity_down']:.3f}  "
                  f"MI₁={m['opacity_temp_mi1']:.3f}  "
                  f"wOPt={m['w_OP_t']:.3f}  "
                  f"tc={m['tcomp']:.3f}  gz={m['gzip']:.3f}")


    # ── Ising model ──────────────────────────────────────────────────────────
    if args.substrate in ('ising', 'all'):
        cfg = ISING_CFG.copy()
        if args.seeds:           cfg['N_SEEDS'] = args.seeds
        if args.grid:            cfg['GRID']    = args.grid
        if args.T_min_ising is not None: cfg['T_MIN']   = args.T_min_ising
        if args.T_max_ising is not None: cfg['T_MAX']   = args.T_max_ising
        if args.T_step_ising is not None: cfg['T_STEP'] = args.T_step_ising
        if args.ising_workers is not None: cfg['SWEEP_WORKERS'] = args.ising_workers

        csv_out  = args.csv or 'ising_results.csv'
        rows     = ising_sweep(cfg, csv_out=csv_out, verbose=True)

        if not args.no_plot:
            plot_path = args.save_plot or 'ising_analysis.png'
            ising_plot(rows, cfg, save_path=plot_path)


    # ── SIR epidemic model ────────────────────────────────────────────────────
    if args.substrate in ('sir', 'all'):
        cfg = SIR_CFG.copy()
        if args.seeds:    cfg['N_SEEDS']   = args.seeds
        if args.grid:     cfg['GRID']      = args.grid
        if args.beta_min  is not None: cfg['BETA_MIN']  = args.beta_min
        if args.beta_max  is not None: cfg['BETA_MAX']  = args.beta_max
        if args.beta_step is not None: cfg['BETA_STEP'] = args.beta_step
        if args.gamma     is not None: cfg['GAMMA']     = args.gamma

        csv_out  = args.csv or 'sir_results.csv'
        rows     = sir_sweep(cfg, csv_out=csv_out, verbose=True)

        if not args.no_plot:
            plot_path = args.save_plot or 'sir_analysis.png'
            sir_plot(rows, cfg, save_path=plot_path)


    # ── Directed Percolation ──────────────────────────────────────────────────
    if args.substrate in ('dp', 'all'):
        cfg      = DP_CFG.copy()
        if args.seeds: cfg['N_SEEDS'] = args.seeds
        if args.grid:  cfg['GRID']    = args.grid
        csv_out  = args.csv or 'dp_results.csv'
        dp_rows  = dp_sweep(cfg, csv_out=csv_out, verbose=True)
        if not args.no_plot:
            ff_rows_for_plot = getattr(args, '_ff_rows', None)

    # ── Forest Fire ───────────────────────────────────────────────────────────
    if args.substrate in ('ff', 'all'):
        cfg      = FF_CFG.copy()
        if args.seeds: cfg['N_SEEDS'] = args.seeds
        if args.grid:  cfg['GRID']    = args.grid
        csv_out  = args.csv.replace('.csv','_ff.csv') if args.csv else 'ff_results.csv'
        ff_rows  = ff_sweep(cfg, csv_out=csv_out, verbose=True)
        if not args.no_plot:
            plot_path = args.save_plot or 'criticality_analysis.png'
            # Use dp_rows if we just ran DP, else empty
            _dp = dp_rows if args.substrate == 'all' else []
            criticality_plot(_dp if _dp else dp_rows if 'dp_rows' in dir() else [],
                             ff_rows, save_path=plot_path)

    # Standalone DP plot (if only dp was run)
    if args.substrate == 'dp' and not args.no_plot:
        criticality_plot(dp_rows, [], save_path=args.save_plot or 'dp_analysis.png')


    # ── Schelling segregation ─────────────────────────────────────────────────
    if args.substrate in ('schelling', 'all'):
        cfg     = SCHELLING_CFG.copy()
        if args.seeds: cfg['N_SEEDS'] = args.seeds
        if args.grid:  cfg['GRID']    = args.grid
        csv_out = args.csv or 'schelling_results.csv'
        rows    = schelling_sweep(cfg, csv_out=csv_out, verbose=True)
        if not args.no_plot:
            plot_path = args.save_plot or 'schelling_analysis.png'
            schelling_plot(rows, cfg, save_path=plot_path)


if __name__ == '__main__':
    main()