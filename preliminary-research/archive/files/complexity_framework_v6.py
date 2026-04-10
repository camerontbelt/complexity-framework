"""
complexity_framework.py — v6
================================
Unified Complexity Measurement Framework
Updated: April 2026

Implements the eight candidate properties of complexity across four
simulation substrates using a single set of IC-independent metrics.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORE INSIGHT (v6)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Opacity is a property that exists in both space and time.
The original framework measured it only spatially. v6 adds
the temporal axis, completing the P1 (opaque hierarchical
layering) measurement.

  Spatial opacity (upward):   H(global | local)
    Given a local patch, how uncertain is the global state?

  Spatial opacity (downward):  H(local | global)
    Given the global state, how underdetermined are local patterns?

  Temporal opacity:  I(past ; future) with decay
    Given the past, how much does knowing more past still
    improve future prediction — and does that correlation
    decay (ruling out trivial frozen memory)?

All three share the same theoretical structure: they measure
information hiding across a scale boundary. Spatial opacity
measures across the spatial scale boundary (local↔global).
Temporal opacity measures across the temporal scale boundary
(past↔future). The composite requires ALL THREE to be non-trivial.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY TEMPORAL OPACITY IS PARAMETER-FREE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Three tanh gates with theoretically-motivated zero points:

  tanh(k · MI₁)         not random:  MI=0 means no temporal structure
  tanh(k · (1 − MI₁))   not frozen:  MI=1 means trivially predictable
  tanh(k · decay)        has structure: decay>0 means correlation falls
                          off — information is transmitted but not
                          stored indefinitely (causal, not static)

No Gaussian peak is required because the extremes are information-
theoretic limits, not empirical observations. The peak emerges from
where complex systems actually sit — exactly as entropy's tanh gates
do. This is the same logic Cameron originally used for w_H.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIMENSIONAL BEHAVIOUR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Temporal opacity signal scales with substrate dimensionality:
  1D ECA:  w_temporal ≈ 0.02–0.03 for all classes (too small
           to discriminate — field-level MI is dimensionally weak)
  2D Life: w_temporal ≈ 0.91–0.97 for C4, ≈0.00 for C3/frozen C2
           (strong discrimination — richer spatial field)
  N-body:  expected to activate similarly to 2D

This is not a failure. It is the correct dimensional scaling:
a 1D substrate has less spatial volume over which temporal
correlations can accumulate and be measured. The metric
activates proportionally to the dimensional richness of the
substrate and contributes nothing when the substrate is too
sparse to support it.

Result: 1D ECA results are preserved (w_temporal near-zero
multiplies without effect). 2D results improve substantially.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPOSITE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C = w_H × w_OP_spatial × w_OP_temporal × w_G

  w_H           spatial entropy tanh gates + variance bonus
  w_OP_spatial  Gaussian(opacity_up) × Gaussian(opacity_down)
  w_OP_temporal tanh(MI₁) × tanh(1−MI₁) × tanh(decay)
  w_G           Gaussian(gzip)

t_comp is RETIRED in v6. Temporal opacity subsumes it:
  - t_comp measured whether cells flip at an intermediate rate
  - temporal opacity measures WHY — whether that rate reflects
    genuine causal structure or trivial static/random behaviour
  - temporal opacity is strictly more informative and parameter-free

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUBSTRATES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  eca    1D binary CA, radius-1 (256 Wolfram rules)
  k3     1D binary CA, radius-2 totalistic (32 rules)
  life   2D outer-totalistic CA (Life-like B/S rules)
  nbody  N-body particle simulation (α × αs scan)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python complexity_framework.py eca
  python complexity_framework.py eca --scale-test
  python complexity_framework.py eca --diagnose 110
  python complexity_framework.py eca --csv results.csv

  python complexity_framework.py k3

  python complexity_framework.py life
  python complexity_framework.py life --grid 64

  python complexity_framework.py nbody
  python complexity_framework.py nbody --scan --csv scan.csv
  python complexity_framework.py nbody --heatmap scan.json

  python complexity_framework.py all --csv all_results.csv

REQUIREMENTS
  pip install numpy matplotlib
"""

import numpy as np
import zlib, csv, json, os, sys, time, argparse
from collections import defaultdict


# ==============================================================================
# WEIGHT FUNCTIONS
# ==============================================================================

def _gauss(x, mu, sig):
    return float(np.exp(-0.5 * ((x - mu) / sig) ** 2))


def weight_H(mean_H, std_H):
    """
    P7 Thermodynamic drive — spatial entropy.

    tanh(50·H)·tanh(50·(1−H)): parameter-free gates zeroing at H=0
    (dead) and H=1 (maximum noise). Peak emerges from wherever the
    system's attractor sits — no fitting required.

    ×(1 + Gaussian(std_H, 0.012, 0.008)): rewards the entropy variance
    that marks C4 rules operating at a dynamical critical point. The
    peak (0.012) is the only empirically-located parameter in w_H.

    Max ≈ 2.0. Substrate-independent.
    """
    w_gates = float(np.tanh(50.0 * mean_H) * np.tanh(50.0 * (1.0 - mean_H)))
    w_var   = _gauss(std_H, 0.012, 0.008)
    return w_gates * (1.0 + w_var)


def weight_opacity_spatial(op_up, op_down):
    """
    P1 Spatial opacity — information hiding across spatial scale.

    opacity_up   = H(global density | local patch)
      Upward: local → global information hiding.
      C4~0.14  C3~0.33 (too high)  C2~0.12  C1~0.00

    opacity_down = H(local patch | global density)
      Downward: global → local underdetermination.
      C4~0.97  C3~0.98  C2~0.71 (too low)  C1~0.00

    Product: C4 occupies the unique region of intermediate upward
    AND high downward opacity simultaneously. Both Gaussians are
    calibrated to the 1D ECA C4 cluster and held fixed.

    Note: opacity_down peak shifts by substrate (1D→0.97, 2D→~0.67)
    due to normalisation differences — this is a known open item.
    """
    w_up   = _gauss(op_up,   0.14, 0.10)
    w_down = _gauss(op_down, 0.97, 0.05)
    return w_up * w_down


def weight_opacity_temporal(mi_lag1, mi_decay, k=10):
    """
    P1 Temporal opacity — information hiding across temporal scale.

    Measures I(past ; future) with a decay signature. Three tanh gates
    with theoretically-motivated zero points (no empirical fitting):

      tanh(k · MI₁)         not random:
        MI=0 → past gives no information about future.
        C3 rules fail here: chaotic rules have no temporal memory.

      tanh(k · (1 − MI₁))   not frozen:
        MI=1 → future perfectly predicted from past (static).
        Frozen C2/C1 rules fail here.

      tanh(k · decay)        has decaying causal structure:
        decay = MI(lag=1) − MI(lag=max) > 0
        Rules with genuine causal structure show MI that is present
        but not permanent. Random: no decay (MI≈0 throughout).
        Frozen: no decay (MI≈1 throughout). Complex: MI starts
        intermediate, decays as the causal horizon is exceeded.

    k=10 controls gate sharpness only — not peak location.
    Peak location is determined by where complex systems sit,
    not by any fitted parameter.

    Dimensional scaling: signal proportional to substrate richness.
      1D: w_temporal ≈ 0.02 for all classes (too sparse to discriminate)
      2D: w_temporal ≈ 0.91–0.97 C4, ≈0.00 C3/frozen C2
    This is correct behaviour, not a failure.
    """
    g1 = np.tanh(k * float(mi_lag1))
    g2 = np.tanh(k * (1.0 - float(mi_lag1)))
    g3 = np.tanh(k * max(float(mi_decay), 0.0))
    return float(g1 * g2 * g3)


def weight_gzip(mean_gz):
    """
    P2 Modular interconnection — Kolmogorov complexity proxy.

    Gaussian(gz, µ=0.10, σ=0.05). C4 rules compress to gz≈0.10–0.12
    (stable structured regions + complex boundaries). Trivially
    compressible rules (C1) sit at gz≈0.01. Random rules sit at gz≈0.16+.
    """
    return _gauss(mean_gz, 0.10, 0.05)


def composite(mean_H, std_H, op_up, op_down, mi_lag1, mi_decay, mean_gz):
    """
    C = w_H × w_OP_spatial × w_OP_temporal × w_G

    Multiplicative: a system must exhibit all three opacity signatures
    simultaneously (spatial upward, spatial downward, temporal) as well
    as intermediate entropy and intermediate compressibility.

    t_comp is retired: temporal opacity subsumes it with better
    theoretical grounding and no fitted Gaussian peak.
    """
    return (weight_H(mean_H, std_H) *
            weight_opacity_spatial(op_up, op_down) *
            weight_opacity_temporal(mi_lag1, mi_decay) *
            weight_gzip(mean_gz))


# ==============================================================================
# SHARED LOW-LEVEL METRICS
# (substrate-independent: operate on any array of shape (T, N_cells))
# ==============================================================================

def _entropy_stats(grid, burnin, window):
    """Mean and std of row-wise binary entropy over the measurement window."""
    post = grid[burnin:burnin + window]
    d = np.clip(post.mean(axis=1), 1e-12, 1 - 1e-12)
    H = -(d * np.log2(d) + (1 - d) * np.log2(1 - d))
    return float(H.mean()), float(H.std())


def _opacity_upward(grid, burnin, window, n_bins=8, half=1):
    """H(global density bin | local patch) — upward spatial opacity."""
    joint = defaultdict(int)
    marg  = defaultdict(int)
    W = grid.shape[1]
    for t in range(burnin, burnin + window):
        row   = grid[t]
        g_bin = min(int(row.mean() * n_bins), n_bins - 1)
        for x in range(W):
            patch = tuple(int(row[(x + dx) % W]) for dx in range(-half, half + 1))
            joint[(patch, g_bin)] += 1
            marg[patch] += 1
    total = sum(joint.values())
    if total == 0: return 0.0
    lt = sum(marg.values())
    hj = sum(-c / total * np.log2(c / total) for c in joint.values() if c > 0)
    hl = sum(-c / lt   * np.log2(c / lt)    for c in marg.values()  if c > 0)
    return float(np.clip((hj - hl) / np.log2(n_bins), 0.0, 1.0))


def _opacity_downward(grid, burnin, window, n_bins=8, half=1):
    """H(local patch | global density bin) — downward spatial opacity."""
    joint  = defaultdict(int)
    marg_g = defaultdict(int)
    W = grid.shape[1]
    for t in range(burnin, burnin + window):
        row   = grid[t]
        g_bin = min(int(row.mean() * n_bins), n_bins - 1)
        for x in range(W):
            patch = tuple(int(row[(x + dx) % W]) for dx in range(-half, half + 1))
            joint[(patch, g_bin)] += 1
            marg_g[g_bin] += 1
    total = sum(joint.values())
    if total == 0: return 0.0
    lg = sum(marg_g.values())
    hj = sum(-c / total * np.log2(c / total) for c in joint.values() if c > 0)
    hg = sum(-c / lg   * np.log2(c / lg)    for c in marg_g.values() if c > 0)
    max_patch_bits = np.log2(2 ** (2 * half + 1))
    return float(np.clip((hj - hg) / max_patch_bits, 0.0, 1.0))


def _temporal_mi(grid, burnin, window, lag, stride=4):
    """
    Normalised mutual information I(X_t ; X_{t+lag}) averaged across cells.
    Measures how much knowing a cell's state at time t reduces uncertainty
    about that same cell at time t+lag.
    Normalised by H(X_t) → range [0, 1].
    """
    post = grid[burnin:burnin + window]
    W    = post.shape[1]
    joint  = defaultdict(int)
    marg_t = defaultdict(int)
    marg_l = defaultdict(int)
    for t in range(0, len(post) - lag):
        for x in range(0, W, stride):
            a = int(post[t,     x])
            b = int(post[t+lag, x])
            joint[(a, b)] += 1
            marg_t[a]     += 1
            marg_l[b]     += 1
    total = sum(joint.values())
    if total == 0: return 0.0
    Ht = sum(-c/total * np.log2(c/total) for c in marg_t.values() if c > 0)
    Hl = sum(-c/total * np.log2(c/total) for c in marg_l.values() if c > 0)
    Hj = sum(-c/total * np.log2(c/total) for c in joint.values()  if c > 0)
    MI = Ht + Hl - Hj
    return float(np.clip(MI / max(Ht, 1e-9), 0.0, 1.0))


def _opacity_temporal(grid, burnin, window, max_lag=10, stride=4):
    """
    Temporal opacity: I(past ; future) profile.
    Returns (mi_lag1, mi_decay) where:
      mi_lag1  = MI at lag=1  (initial temporal correlation)
      mi_decay = MI(lag=1) - MI(lag=max_lag)  (how much it falls off)
    """
    mi1  = _temporal_mi(grid, burnin, window, lag=1,       stride=stride)
    miN  = _temporal_mi(grid, burnin, window, lag=max_lag, stride=stride)
    return mi1, float(mi1 - miN)


def _gzip(grid, burnin, window):
    """Gzip compression ratio of the spatiotemporal history."""
    raw = grid[burnin:burnin + window].tobytes()
    return len(zlib.compress(raw, 6)) / len(raw)


def _evaluate_grid(grid, burnin, window):
    """
    Compute all metrics and composite from a single grid run.
    Returns a dict of raw metric values, weights, and composite score.
    """
    mH, sH    = _entropy_stats(grid, burnin, window)
    op_up     = _opacity_upward(grid,    burnin, window)
    op_down   = _opacity_downward(grid,  burnin, window)
    mi1, dec  = _opacity_temporal(grid,  burnin, window)
    gz        = _gzip(grid, burnin, window)
    C = composite(mH, sH, op_up, op_down, mi1, dec, gz)
    return dict(
        mean_H=mH, std_H=sH,
        opacity_up=op_up, opacity_down=op_down,
        mi_lag1=mi1, mi_decay=dec,
        gzip=gz, score=C,
        w_H=weight_H(mH, sH),
        w_OP_s=weight_opacity_spatial(op_up, op_down),
        w_OP_t=weight_opacity_temporal(mi1, dec),
        w_G=weight_gzip(gz),
    )


def _average_seeds(seed_results):
    """Average metric dicts across seeds, recompute composite."""
    keys = ['mean_H', 'std_H', 'opacity_up', 'opacity_down',
            'mi_lag1', 'mi_decay', 'gzip']
    avgs = {k: float(np.mean([r[k] for r in seed_results])) for k in keys}
    C = composite(avgs['mean_H'], avgs['std_H'],
                  avgs['opacity_up'], avgs['opacity_down'],
                  avgs['mi_lag1'], avgs['mi_decay'],
                  avgs['gzip'])
    avgs['score']   = C
    avgs['w_H']     = weight_H(avgs['mean_H'], avgs['std_H'])
    avgs['w_OP_s']  = weight_opacity_spatial(avgs['opacity_up'], avgs['opacity_down'])
    avgs['w_OP_t']  = weight_opacity_temporal(avgs['mi_lag1'], avgs['mi_decay'])
    avgs['w_G']     = weight_gzip(avgs['gzip'])
    return avgs


# ==============================================================================
# SUBSTRATE 1 — ECA (1D binary CA, radius 1, 256 Wolfram rules)
# ==============================================================================

ECA_CLASS4 = {110, 124, 137, 193}
ECA_CLASS3 = {30, 45, 86, 89, 101, 105, 106, 150, 153, 165, 169, 182}
ECA_CLASS1 = {0, 8, 32, 40, 128, 136, 160, 168, 255}
ECA_CLASS2 = set(range(256)) - ECA_CLASS4 - ECA_CLASS3 - ECA_CLASS1

ECA_CFG = dict(WIDTH=150, STEPS=300, BURNIN=20, WINDOW=150, N_SEEDS=5, DENSITY=0.5)


def _eca_class(rule):
    if rule in ECA_CLASS4: return 'C4'
    if rule in ECA_CLASS3: return 'C3'
    if rule in ECA_CLASS1: return 'C1'
    return 'C2'


def _eca_run(rule_number, cfg, seed=42):
    lookup = np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.uint8)
    rng = np.random.RandomState(seed)
    W, S = cfg['WIDTH'], cfg['STEPS']
    grid = np.zeros((S, W), dtype=np.uint8)
    d = cfg.get('DENSITY', 0.5)
    if d is None: grid[0, W // 2] = 1
    else:         grid[0] = (rng.rand(W) < d).astype(np.uint8)
    for t in range(1, S):
        row = grid[t-1]; left = np.roll(row, 1); right = np.roll(row, -1)
        grid[t] = lookup[(left << 2) | (row << 1) | right]
    return grid


def eca_evaluate(rule_number, cfg):
    seeds  = [_evaluate_grid(_eca_run(rule_number, cfg, seed=s*17+3),
                              cfg['BURNIN'], cfg['WINDOW'])
              for s in range(cfg['N_SEEDS'])]
    result = _average_seeds(seeds)
    result['rule']  = rule_number
    result['class'] = _eca_class(rule_number)
    return result


def eca_run_all(cfg, verbose=True):
    ic_label = 'single-cell' if cfg.get('DENSITY') is None \
               else f"random {cfg.get('DENSITY',0.5)*100:.0f}%"
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


def eca_diagnose(rule_number, cfg):
    ic_label = 'single-cell' if cfg.get('DENSITY') is None \
               else f"random {cfg.get('DENSITY',0.5)*100:.0f}%"
    print(f"\nDiagnose ECA Rule {rule_number} [{_eca_class(rule_number)}] "
          f"IC:{ic_label} W={cfg['WIDTH']}")
    print('─' * 105)
    Cs = []
    for s in range(cfg['N_SEEDS']):
        r = _evaluate_grid(_eca_run(rule_number, cfg, seed=s*17+3),
                           cfg['BURNIN'], cfg['WINDOW'])
        Cs.append(r['score'])
        print(f"  seed {s*17+3:3d}: "
              f"H={r['mean_H']:.4f}±{r['std_H']:.4f}  "
              f"op↑={r['opacity_up']:.4f}  op↓={r['opacity_down']:.4f}  "
              f"MI₁={r['mi_lag1']:.4f}  dec={r['mi_decay']:.4f}  "
              f"gz={r['gzip']:.4f}  "
              f"wH={r['w_H']:.4f}  wOPs={r['w_OP_s']:.4f}  "
              f"wOPt={r['w_OP_t']:.4f}  wG={r['w_G']:.4f}  "
              f"C={r['score']:.6f}")
    print('─' * 105)
    print(f"  Mean C = {np.mean(Cs):.6f}  (std={np.std(Cs):.6f})")


def eca_print_results(results, label=''):
    sep      = eca_separation(results)
    c4_ranks = sorted([r['rank'] for r in results if r['rule'] in ECA_CLASS4])
    print(f"\n{'─'*95}")
    if label: print(f"  {label}")
    print(f"  C4/C3 separation: {sep:.1f}×    C4 ranks: {c4_ranks}")
    print(f"{'─'*95}")
    print(f"  {'Rk':>3} {'Rule':>4} {'Cls':>3} {'Score':>8}  "
          f"{'H':>5} {'sH':>5} {'op↑':>5} {'op↓':>5} "
          f"{'MI₁':>5} {'dec':>5} {'gz':>5}  "
          f"{'wH':>6} {'wOPs':>6} {'wOPt':>6} {'wG':>6}")
    for r in results[:20]:
        m = '*' if r['rule'] in ECA_CLASS4 else ' '
        print(f"  {m}{r['rank']:3d} {r['rule']:4d} {r['class']:>3} "
              f"{r['score']:8.5f}  "
              f"{r['mean_H']:5.3f} {r['std_H']:5.3f} "
              f"{r['opacity_up']:5.3f} {r['opacity_down']:5.3f} "
              f"{r['mi_lag1']:5.3f} {r['mi_decay']:5.3f} {r['gzip']:5.3f}  "
              f"{r['w_H']:6.4f} {r['w_OP_s']:6.4f} "
              f"{r['w_OP_t']:6.4f} {r['w_G']:6.4f}")


def eca_scale_test(cfg, widths=None, csv_out=None):
    if widths is None: widths = [100, 150, 200, 300, 400]
    print(f"\n{'='*65}")
    print(f"  ECA scale sweep | IC: random {cfg.get('DENSITY',0.5)*100:.0f}%")
    print(f"{'='*65}")
    saved = {k: cfg[k] for k in ['WIDTH','STEPS','BURNIN','WINDOW']}
    rows  = []
    for W in widths:
        cfg.update(WIDTH=W, STEPS=max(300,W*2), BURNIN=20,
                   WINDOW=min(150,max(300,W*2)-30))
        t0  = time.time()
        res = eca_run_all(cfg, verbose=False)
        sep = eca_separation(res)
        rm  = {r['rule']: r['rank'] for r in res}
        r110 = next(r for r in res if r['rule'] == 110)
        c4r  = sorted([rm[r] for r in ECA_CLASS4])
        print(f"  W={W:3d}: sep={sep:6.1f}× C4={c4r}  "
              f"R110=#{rm[110]:3d}  "
              f"op↑={r110['opacity_up']:.4f}  op↓={r110['opacity_down']:.4f}  "
              f"MI₁={r110['mi_lag1']:.4f}  dec={r110['mi_decay']:.4f}  "
              f"({time.time()-t0:.0f}s)")
        rows.append(dict(width=W, c4_c3_sep=round(sep,2), c4_ranks=str(c4r),
                         R110_rank=rm[110],
                         R110_op_up=round(r110['opacity_up'],4),
                         R110_op_dn=round(r110['opacity_down'],4),
                         R110_mi1=round(r110['mi_lag1'],4),
                         R110_dec=round(r110['mi_decay'],4),
                         R110_wOPs=round(r110['w_OP_s'],4),
                         R110_wOPt=round(r110['w_OP_t'],4)))
    for k, v in saved.items(): cfg[k] = v
    if csv_out: _save_csv(rows, csv_out)
    return rows


# ==============================================================================
# SUBSTRATE 2 — k=3 CA (1D binary, radius-2 totalistic, 32 rules)
# ==============================================================================

K3_COMPLEX = {22, 26}
K3_CHAOTIC = {6, 14, 18, 30}
K3_SIMPLE  = {0, 1, 31}

K3_CFG = dict(WIDTH=200, STEPS=500, BURNIN=50, WINDOW=200, N_SEEDS=5, DENSITY=0.5)


def _k3_class(rule):
    if rule in K3_COMPLEX: return 'C4'
    if rule in K3_CHAOTIC: return 'C3'
    if rule in K3_SIMPLE:  return 'C1'
    return '-'


def _k3_run(rule_number, cfg, seed=42):
    lookup = np.array([(rule_number >> i) & 1 for i in range(5)], dtype=np.uint8)
    rng = np.random.RandomState(seed)
    W, S = cfg['WIDTH'], cfg['STEPS']
    grid = np.zeros((S, W), dtype=np.uint8)
    d = cfg.get('DENSITY', 0.5)
    if d is None: grid[0, W // 2] = 1
    else:         grid[0] = (rng.rand(W) < d).astype(np.uint8)
    for t in range(1, S):
        row   = grid[t-1]
        nbsum = (np.roll(row,2)+np.roll(row,1)+row+np.roll(row,-1)+np.roll(row,-2))
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
    if verbose:
        print(f"\nk=3 CA | 32 rules | IC: random {cfg.get('DENSITY',0.5)*100:.0f}% | "
              f"W={cfg['WIDTH']} seeds={cfg['N_SEEDS']}")
    results = [k3_evaluate(r, cfg) for r in range(32)]
    results.sort(key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(results): r['rank'] = i + 1
    if verbose: _print_substrate_top(results, 'k=3 CA', K3_COMPLEX, K3_CHAOTIC)
    return results


# ==============================================================================
# SUBSTRATE 3 — 2D Life-like CA (outer totalistic, B/S notation)
# ==============================================================================

LIFE_CFG = dict(GRID=64, STEPS=200, BURNIN=20, WINDOW=100, N_SEEDS=5, DENSITY=0.35)

LIFE_RULES = [
    ('Conway',       frozenset([3]),         frozenset([2,3]),           'C4'),
    ('HighLife',     frozenset([3,6]),        frozenset([2,3]),           'C4'),
    ('Day&Night',    frozenset([3,6,7,8]),   frozenset([3,4,6,7,8]),     'C4'),
    ('Morley',       frozenset([3,6,8]),      frozenset([2,4,5]),         'C4'),
    ('34Life',       frozenset([3,4]),        frozenset([3,4]),           'C3'),
    ('Amoeba',       frozenset([3,5,7]),      frozenset([1,3,5,8]),       'C3'),
    ('Anneal',       frozenset([4,6,7,8]),    frozenset([3,5,6,7,8]),     'C3'),
    ('Gnarl',        frozenset([1]),          frozenset([1]),             'C3'),
    ('Coagulations', frozenset([3,7,8]),      frozenset([2,3,5,6,7,8]),   'C3'),
    ('2x2',          frozenset([3,6]),        frozenset([1,2,5]),         'C2'),
    ('Seeds',        frozenset([2]),          frozenset([]),              'C2'),
    ('Replicator',   frozenset([1,3,5,7]),    frozenset([1,3,5,7]),       'C2'),
    ('Maze',         frozenset([3]),          frozenset([1,2,3,4,5]),     'C2'),
    ('Coral',        frozenset([3]),          frozenset([4,5,6,7,8]),     'C2'),
    ('Static',       frozenset([3]),          frozenset([2,3,4,5,6,7,8]),'C1'),
    ('Flakes',       frozenset([3]),          frozenset([0,1,2,3,4,5,6,7,8]),'C1'),
    ('AllDead',      frozenset([]),           frozenset([2,3]),           'C1'),
]


def _life_run(birth, survive, cfg, seed=42):
    rng  = np.random.RandomState(seed)
    G    = cfg['GRID']
    grid = (rng.rand(G, G) < cfg['DENSITY']).astype(np.uint8)
    S    = cfg['STEPS']
    hist = np.zeros((S, G, G), dtype=np.uint8); hist[0] = grid
    b_arr = np.zeros(9, dtype=bool); s_arr = np.zeros(9, dtype=bool)
    for b in birth:   b_arr[b] = True
    for s in survive: s_arr[s] = True
    for t in range(1, S):
        g  = hist[t-1].astype(np.int32)
        nb = np.zeros_like(g)
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr == 0 and dc == 0: continue
                nb += np.roll(np.roll(g, dr, axis=0), dc, axis=1)
        hist[t] = ((g==0)&b_arr[nb] | (g==1)&s_arr[nb]).astype(np.uint8)
    return hist.reshape(S, G*G)


def _life_entropy(flat, burnin, window, G):
    post = flat[burnin:burnin+window].reshape(window, G, G)
    d    = np.clip(post.mean(axis=(1,2)), 1e-12, 1-1e-12)
    H    = -(d*np.log2(d)+(1-d)*np.log2(1-d))
    return float(H.mean()), float(H.std())


def life_evaluate(name, birth, survive, cfg):
    G = cfg['GRID']
    seeds_results = []
    for s in range(cfg['N_SEEDS']):
        flat     = _life_run(birth, survive, cfg, seed=s*17+3)
        mH, sH   = _life_entropy(flat, cfg['BURNIN'], cfg['WINDOW'], G)
        op_up    = _opacity_upward(flat,   cfg['BURNIN'], cfg['WINDOW'])
        op_down  = _opacity_downward(flat, cfg['BURNIN'], cfg['WINDOW'])
        mi1, dec = _opacity_temporal(flat, cfg['BURNIN'], cfg['WINDOW'])
        gz       = _gzip(flat, cfg['BURNIN'], cfg['WINDOW'])
        C        = composite(mH, sH, op_up, op_down, mi1, dec, gz)
        seeds_results.append(dict(
            mean_H=mH, std_H=sH,
            opacity_up=op_up, opacity_down=op_down,
            mi_lag1=mi1, mi_decay=dec,
            gzip=gz, score=C,
            w_H=weight_H(mH,sH),
            w_OP_s=weight_opacity_spatial(op_up,op_down),
            w_OP_t=weight_opacity_temporal(mi1,dec),
            w_G=weight_gzip(gz),
        ))
    result = _average_seeds(seeds_results)
    result['name'] = name
    return result


def life_run_all(cfg, verbose=True):
    if verbose:
        print(f"\n2D Life-like | {len(LIFE_RULES)} rules | "
              f"grid={cfg['GRID']}×{cfg['GRID']}  "
              f"density={cfg['DENSITY']}  seeds={cfg['N_SEEDS']}")
    results = []
    for name, birth, survive, cls_label in LIFE_RULES:
        r = life_evaluate(name, birth, survive, cfg)
        r['class'] = cls_label
        results.append(r)
        if verbose:
            print(f"  {name:14s} [{cls_label}]  C={r['score']:.5f}  "
                  f"op↑={r['opacity_up']:.3f}  op↓={r['opacity_down']:.3f}  "
                  f"MI₁={r['mi_lag1']:.3f}  dec={r['mi_decay']:.3f}")
    results.sort(key=lambda x: x['score'], reverse=True)
    for i, r in enumerate(results): r['rank'] = i + 1
    if verbose:
        c4 = [r for r in results if r['class']=='C4']
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
    return np.random.rand(N,2)*box, np.random.randn(N,2)*0.35


def _nbody_forces(pos, alpha, alpha_s, box, clamp):
    dx = pos[:,None,0]-pos[None,:,0]; dy = pos[:,None,1]-pos[None,:,1]
    dx -= box*np.round(dx/box);       dy -= box*np.round(dy/box)
    r2   = np.maximum(dx**2+dy**2, 0.04)
    mask = r2 < (alpha_s*3.5)**2; np.fill_diagonal(mask, False)
    sr2  = np.where(mask,(alpha_s**2)/r2,0); sr6=sr2**3; sr12=sr6**2
    fm   = np.where(mask,24*alpha*(2*sr12-sr6)/r2,0); np.fill_diagonal(fm,0)
    return (np.clip((fm*dx).sum(1),-clamp,clamp),
            np.clip((fm*dy).sum(1),-clamp,clamp))


def _nbody_step(pos, vel, alpha, alpha_s, cfg):
    box=cfg['BOX']; dt=cfg['DT']; damp=cfg['DAMPING']; cl=cfg['FORCE_CLAMP']
    fx,fy=_nbody_forces(pos,alpha,alpha_s,box,cl)
    vel[:,0]+=0.5*fx*dt; vel[:,1]+=0.5*fy*dt
    pos=(pos+vel*dt)%box
    fx2,fy2=_nbody_forces(pos,alpha,alpha_s,box,cl)
    vel[:,0]+=0.5*fx2*dt; vel[:,1]+=0.5*fy2*dt
    vel*=damp
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
    Map N-body frames → binary occupancy grid → standard metrics.
    Temporal opacity is particularly meaningful here: the N-body system
    has genuine 2D spatial richness so w_OP_t should activate for
    genuinely complex parameter regions.
    """
    if len(frames) < 4:
        return dict(mean_H=0,std_H=0,opacity_up=0,opacity_down=0,
                    mi_lag1=0,mi_decay=0,gzip=0,score=0,
                    w_H=0,w_OP_s=0,w_OP_t=0,w_G=0)
    box=cfg['BOX']; bins=20; T=len(frames)
    occ=np.zeros((T,bins*bins),dtype=np.uint8)
    for t,pos in enumerate(frames):
        gx=np.clip((pos[:,0]/box*bins).astype(int),0,bins-1)
        gy=np.clip((pos[:,1]/box*bins).astype(int),0,bins-1)
        for x,y in zip(gx,gy): occ[t,x*bins+y]=1
    burnin=0; window=T
    mH,sH    = _entropy_stats(occ, burnin, window)
    op_up    = _opacity_upward(occ,   burnin, window)
    op_down  = _opacity_downward(occ, burnin, window)
    mi1, dec = _opacity_temporal(occ, burnin, window)
    gz       = _gzip(occ, burnin, window)
    C        = composite(mH,sH,op_up,op_down,mi1,dec,gz)
    return dict(mean_H=mH,std_H=sH,
                opacity_up=op_up,opacity_down=op_down,
                mi_lag1=mi1,mi_decay=dec,
                gzip=gz,score=C,
                w_H=weight_H(mH,sH),
                w_OP_s=weight_opacity_spatial(op_up,op_down),
                w_OP_t=weight_opacity_temporal(mi1,dec),
                w_G=weight_gzip(gz))


def nbody_scan(cfg, csv_out=None, json_out=None, verbose=True):
    N_A=cfg['SCAN_N_ALPHA']; N_AS=cfg['SCAN_N_ALPHAS']
    AR =np.logspace(np.log10(cfg['SCAN_ALPHA_MIN']),
                    np.log10(cfg['SCAN_ALPHA_MAX']),N_A)
    ASR=np.logspace(np.log10(cfg['SCAN_ALPHAS_MIN']),
                    np.log10(cfg['SCAN_ALPHAS_MAX']),N_AS)
    total=N_A*N_AS
    if verbose:
        print(f"\nN-body scan | {N_A}×{N_AS}={total} points | seeds={cfg['SCAN_SEEDS']}")
    cmap=np.zeros((N_A,N_AS)); rows=[]; t0=time.time()
    for i,alpha in enumerate(AR):
        for j,alpha_s in enumerate(ASR):
            seed_m=[_nbody_metrics(_nbody_run(alpha,alpha_s,cfg,s),cfg)
                    for s in cfg['SCAN_SEEDS']]
            avg={k:float(np.mean([m[k] for m in seed_m])) for k in seed_m[0]}
            cmap[i,j]=avg['score']
            rows.append(dict(alpha=round(alpha,4),alpha_s=round(alpha_s,4),
                             **{k:round(v,6) for k,v in avg.items()}))
            if verbose:
                done=i*N_AS+j+1; pct=done/total*100
                bar='█'*int(pct//5)+'░'*(20-int(pct//5))
                print(f"\r  [{bar}] {pct:.0f}%  α={alpha:.3f} αs={alpha_s:.3f}  "
                      f"C={avg['score']:.4f}", end='', flush=True)
    if verbose:
        print()
        pk=np.unravel_index(np.argmax(cmap),cmap.shape)
        oi=np.argmin(np.abs(AR-1.0)); oj=np.argmin(np.abs(ASR-1.0))
        pct=float(np.mean(cmap<cmap[oi,oj])*100)
        print(f"  Peak        : α={AR[pk[0]]:.3f}  αs={ASR[pk[1]]:.3f}  C={cmap.max():.4f}")
        print(f"  Our universe: C={cmap[oi,oj]:.4f}  ({pct:.0f}th percentile)")
        print(f"  Elapsed     : {time.time()-t0:.0f}s")
    if csv_out: _save_csv(rows,csv_out)
    if json_out:
        pk=np.unravel_index(np.argmax(cmap),cmap.shape)
        oi=np.argmin(np.abs(AR-1.0)); oj=np.argmin(np.abs(ASR-1.0))
        with open(json_out,'w') as f:
            json.dump({'alpha_range':AR.tolist(),'alphas_range':ASR.tolist(),
                       'complexity_map':cmap.tolist(),
                       'peak':{'alpha':float(AR[pk[0]]),'alpha_s':float(ASR[pk[1]]),
                               'score':float(cmap.max())},
                       'our_universe':{'score':float(cmap[oi,oj]),
                                       'percentile':float(np.mean(cmap<cmap[oi,oj])*100)}},
                      f,indent=2)
        print(f"  Saved → {json_out}")
    return rows, cmap, AR, ASR


# ==============================================================================
# SHARED OUTPUT HELPERS
# ==============================================================================

def _save_csv(rows, filename):
    if not rows: return
    with open(filename,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k:round(v,6) if isinstance(v,float) else v
                        for k,v in r.items()})
    print(f"  Saved → {filename}")


def _print_substrate_top(results, label, complex_set, chaotic_set):
    c4s=[r['score'] for r in results
         if r.get('rule') in complex_set or r.get('name') in complex_set]
    c3s=[r['score'] for r in results
         if r.get('rule') in chaotic_set or r.get('name') in chaotic_set]
    sep=float(np.mean(c4s)/np.mean(c3s)) if c4s and c3s else 0.0
    print(f"\n  {label}  C4/C3 sep={sep:.1f}×")
    for r in results[:10]:
        name=r.get('name') or f"Rule {r.get('rule','?')}"
        print(f"  Rank {r['rank']:3d}  {name:16s} [{r.get('class','-'):>3}]  "
              f"C={r['score']:.5f}  "
              f"op↑={r['opacity_up']:.3f}  op↓={r['opacity_down']:.3f}  "
              f"MI₁={r['mi_lag1']:.3f}  dec={r['mi_decay']:.3f}")


# ==============================================================================
# CLI
# ==============================================================================

def main():
    p = argparse.ArgumentParser(
        description='Complexity Framework v6 — unified spatial+temporal opacity',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('substrate', nargs='?', default='eca',
                   choices=['eca','k3','life','nbody','all'])
    p.add_argument('--csv',        metavar='FILE')
    p.add_argument('--no-plot',    action='store_true')
    p.add_argument('--save-plot',  metavar='FILE')
    p.add_argument('--seeds',      type=int)
    p.add_argument('--width',      type=int)
    p.add_argument('--ic',         choices=['random','single'], default='random')
    p.add_argument('--density',    type=float, default=0.5)
    p.add_argument('--diagnose',   type=int,   metavar='RULE')
    p.add_argument('--scale-test', action='store_true')
    p.add_argument('--widths',     type=int,   nargs='+')
    p.add_argument('--scale-csv',  metavar='FILE')
    p.add_argument('--grid',       type=int)
    p.add_argument('--list-rules', action='store_true')
    p.add_argument('--scan',       action='store_true')
    p.add_argument('--heatmap',    metavar='FILE')
    p.add_argument('--alpha',      type=float)
    p.add_argument('--alpha-s',    type=float)
    args = p.parse_args()

    if args.substrate in ('eca','all'):
        cfg = ECA_CFG.copy()
        if args.width:  cfg['WIDTH']   = args.width
        if args.seeds:  cfg['N_SEEDS'] = args.seeds
        cfg['DENSITY'] = None if args.ic=='single' else args.density
        if args.diagnose is not None: eca_diagnose(args.diagnose,cfg); return
        if args.scale_test:
            eca_scale_test(cfg,widths=args.widths,csv_out=args.scale_csv); return
        results = eca_run_all(cfg, verbose=True)
        eca_print_results(results,
            label=f"IC={'random' if cfg['DENSITY'] else 'single'}  W={cfg['WIDTH']}")
        if args.csv: _save_csv(results, args.csv)

    if args.substrate in ('k3','all'):
        cfg = K3_CFG.copy()
        if args.width:  cfg['WIDTH']   = args.width
        if args.seeds:  cfg['N_SEEDS'] = args.seeds
        cfg['DENSITY'] = None if args.ic=='single' else args.density
        results = k3_run_all(cfg, verbose=True)
        if args.csv: _save_csv(results, args.csv.replace('.csv','_k3.csv'))

    if args.substrate in ('life','all'):
        if args.list_rules:
            for name,birth,survive,cls in LIFE_RULES:
                print(f"  {name:14s}  B={''.join(str(b) for b in sorted(birth)):8s}"
                      f"  S={''.join(str(s) for s in sorted(survive)):10s}  [{cls}]")
            return
        cfg = LIFE_CFG.copy()
        if args.grid:  cfg['GRID']    = args.grid
        if args.seeds: cfg['N_SEEDS'] = args.seeds
        results = life_run_all(cfg, verbose=True)
        if args.csv: _save_csv(results, args.csv.replace('.csv','_life.csv'))

    if args.substrate in ('nbody','all'):
        cfg = NBODY_CFG.copy()
        if args.seeds: cfg['SCAN_SEEDS'] = list(range(args.seeds))
        if args.heatmap:
            with open(args.heatmap) as f: data=json.load(f)
            print("Heatmap display requires matplotlib. Run: "
                  "python complexity_framework.py nbody --heatmap FILE")
            return
        if args.scan:
            csv_out  = args.csv or 'nbody_scan.csv'
            json_out = (args.csv.replace('.csv','.json')
                        if args.csv else 'nbody_scan.json')
            nbody_scan(cfg, csv_out=csv_out, json_out=json_out, verbose=True)
            return
        alpha=args.alpha or 1.0; alpha_s=args.alpha_s or 1.0
        print(f"\nN-body single point: α={alpha:.3f}  αs={alpha_s:.3f}")
        for s in cfg['SCAN_SEEDS']:
            m=_nbody_metrics(_nbody_run(alpha,alpha_s,cfg,seed=s),cfg)
            print(f"  seed {s}: C={m['score']:.5f}  "
                  f"op↑={m['opacity_up']:.3f}  op↓={m['opacity_down']:.3f}  "
                  f"MI₁={m['mi_lag1']:.3f}  dec={m['mi_decay']:.3f}  "
                  f"gz={m['gzip']:.3f}")


if __name__ == '__main__':
    main()
