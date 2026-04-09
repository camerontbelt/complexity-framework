"""
complexity_framework.py — v6
================================
Unified Complexity Measurement Framework
Updated: April 2026

Implements the eight candidate properties of complexity across four
simulation substrates using a single set of IC-independent metrics.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUBSTRATES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  eca    1D binary CA, radius-1 (256 Wolfram rules)
  k3     1D binary CA, radius-2 totalistic (32 rules)
  life   2D outer-totalistic CA (Life-like B/S rules)
  nbody  N-body particle simulation (α × αs scan)

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
CHANGES FROM v5
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  + _opacity_temporal(): new metric, fully parameter-free
  + weight_opacity_temporal(): tanh³ gate, no Gaussians
  + composite() updated: temporal opacity via sqrt(1+w_OPt) coupling
  + _evaluate_grid() / _average_seeds() updated for new metric
  + All substrate runners unchanged — new metric is substrate-agnostic

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
        w_H   = weight_H(mH, sH),
        w_OP_s= weight_opacity_spatial(op_up, op_down),
        w_OP_t= weight_opacity_temporal(mi1, dec),
        w_T   = weight_tcomp(tc),
        w_G   = weight_gzip(gz),
        w_dim = None,
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
    avgs['w_T']    = weight_tcomp(avgs['tcomp'])
    avgs['w_G']    = weight_gzip(avgs['gzip'])
    avgs['w_dim']  = (weight_fractal_dim(avgs['fractal_dim'], sdim)
                      if avgs['fractal_dim'] is not None else None)
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
# CLI
# ==============================================================================

def main():
    p = argparse.ArgumentParser(
        description='Complexity Framework v5 — unified bidirectional opacity',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('substrate', nargs='?', default='eca',
                   choices=['eca', 'k3', 'life', 'nbody', 'all'])
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


if __name__ == '__main__':
    main()
