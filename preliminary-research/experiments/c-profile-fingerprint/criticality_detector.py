"""
Criticality detector: estimate T_c (or p_c, etc.) from a multi-scale C sweep.

Takes a list of rows {param, C_1, C_2, C_4, C_8} and returns an estimated
critical point with three independent indicators:

  1. SCALE-COLLAPSE point: param where C values across scales agree best
     (std dev across pools minimized, weighted by mean C to avoid picking
     the trivial ordered/disordered tails where C is uniformly small).

  2. β-ZERO point: param where the RG β function β(s→2s)=C(2s)−C(s) crosses
     zero simultaneously across multiple RG steps. Classical RG fixed-point
     definition.

  3. COARSEST-PEAK point: param that maximizes C at the coarsest available
     scale. Least rigorous, strongest finite-size bias, but robust.

The three estimates also function as a CONSISTENCY CHECK: if they agree,
confidence is high. If they disagree strongly, the system is likely NOT
showing clean scale-invariant second-order behaviour -- could be first-order,
crossover, or out-of-equilibrium.
"""
import json, os, sys
import numpy as np

POOLS_DEFAULT = (1, 2, 4, 8)


def _collect(rows, pools):
    params = np.array([r['param'] if 'param' in r else r['T'] for r in rows],
                      dtype=float)
    Cs = np.array([[r.get(f'C_{p}', np.nan) for p in pools] for r in rows])
    return params, Cs


def scale_collapse_point(params, Cs, weight_by_C=True):
    """Param where scales agree: min of std/mean across pools, but only
    where mean C is non-trivial. Returns (param_estimate, score_curve)."""
    mean_C = np.nanmean(Cs, axis=1)
    std_C  = np.nanstd(Cs, axis=1)
    # Coefficient of variation, with floor to avoid /0 on dead regions
    cv = std_C / np.maximum(mean_C, 1e-3)
    # Mask out dead regions (mean C below 5% of max): not critical, just trivially agreeing at 0
    alive = mean_C > 0.05 * np.nanmax(mean_C)
    score = np.where(alive, 1.0 / (1.0 + cv), 0.0)
    if weight_by_C:
        # multiply by mean_C so we prefer points that are BOTH scale-agreeing AND high-C
        score = score * mean_C
    idx = int(np.nanargmax(score))
    return params[idx], score


def beta_zero_point(params, Cs):
    """Param closest to simultaneous β~=0 across all RG steps.
    Uses sum-of-absolute-betas, minimized."""
    betas = np.diff(Cs, axis=1)          # shape (n_params, n_pools-1)
    total_abs = np.nansum(np.abs(betas), axis=1)
    mean_C = np.nanmean(Cs, axis=1)
    # Mask to 30% of peak -- avoids noise-tail pathology where |beta| is
    # trivially small just because C is uniformly small.
    alive = mean_C > 0.30 * np.nanmax(mean_C)
    total_abs = np.where(alive, total_abs, np.inf)
    if not np.isfinite(total_abs).any():
        alive = mean_C > 0.05 * np.nanmax(mean_C)   # fallback
        total_abs = np.where(alive, total_abs, np.inf)
    idx = int(np.nanargmin(total_abs))
    return params[idx], total_abs


def coarsest_peak_point(params, Cs):
    """Param maximizing C at coarsest pool."""
    coarsest = Cs[:, -1]
    idx = int(np.nanargmax(coarsest))
    return params[idx], coarsest


def estimate_critical(rows, pools=POOLS_DEFAULT, verbose=False, true_value=None):
    params, Cs = _collect(rows, pools)
    p_collapse, _ = scale_collapse_point(params, Cs)
    p_beta,     _ = beta_zero_point(params, Cs)
    p_peak,     _ = coarsest_peak_point(params, Cs)

    estimates = np.array([p_collapse, p_beta, p_peak])
    consensus = float(np.mean(estimates))
    spread    = float(np.std(estimates))
    # Rough confidence: tight spread relative to param range = high confidence
    param_range = params.max() - params.min()
    confidence = float(np.clip(1.0 - spread / (0.15 * param_range), 0.0, 1.0))

    # Boundary penalty: if consensus sits within 1 grid step of either end of the
    # sweep range, the true critical point may lie OUTSIDE the scanned region and
    # all three indicators are pinned against the edge. Cap confidence aggressively.
    grid_step = float(np.median(np.diff(np.sort(params)))) if len(params) > 1 else 0.0
    edge_low  = abs(consensus - params.min()) <= 1.01 * grid_step
    edge_high = abs(consensus - params.max()) <= 1.01 * grid_step
    edge_hit  = bool(edge_low or edge_high)
    if edge_hit:
        confidence = min(confidence, 0.10)

    result = {
        'p_collapse': float(p_collapse),
        'p_beta':     float(p_beta),
        'p_peak':     float(p_peak),
        'consensus':  consensus,
        'spread':     spread,
        'confidence': confidence,
        'edge_hit':   edge_hit,
        'grid_step':  grid_step,
    }
    if true_value is not None:
        result['true']   = float(true_value)
        result['error_collapse'] = float(p_collapse - true_value)
        result['error_beta']     = float(p_beta - true_value)
        result['error_peak']     = float(p_peak - true_value)
        result['error_consensus']= float(consensus - true_value)

    if verbose:
        print(f"  scale-collapse : {p_collapse:.4f}")
        print(f"  beta-zero      : {p_beta:.4f}")
        print(f"  coarsest-peak  : {p_peak:.4f}")
        flag = " [EDGE-PINNED -- likely outside sweep range]" if edge_hit else ""
        print(f"  consensus      : {consensus:.4f}  (spread={spread:.4f}, conf={confidence:.2f}){flag}")
        if true_value is not None:
            print(f"  true value     : {true_value:.4f}")
            print(f"  errors         : collapse={result['error_collapse']:+.4f}, "
                  f"beta={result['error_beta']:+.4f}, peak={result['error_peak']:+.4f}")
    return result


# ── validate on existing data ────────────────────────────────────────────────
if __name__ == '__main__':
    here = os.path.dirname(__file__)

    # Ising
    ms = json.load(open(os.path.join(here, 'multiscale_diagnostic.json')))
    print("=" * 60)
    print("Ising 2D -- known T_c = 2.269")
    print("=" * 60)
    estimate_critical(ms['Ising'], verbose=True, true_value=2.269)

    # DP
    if 'DP' in ms:
        print("\n" + "=" * 60)
        print("Directed Percolation -- known p_c ~= 0.2873")
        print("=" * 60)
        estimate_critical(ms['DP'], verbose=True, true_value=0.2873)

    # CP
    if 'CP' in ms:
        print("\n" + "=" * 60)
        print("Contact Process -- known lambda_c ~= 1.6488")
        print("=" * 60)
        estimate_critical(ms['CP'], verbose=True, true_value=1.6488)

    # Potts v1 -- known T_c(q) = 1/ln(1+√q)
    potts = json.load(open(os.path.join(here, 'potts_q_multiscale.json')))
    for q_str, rows in potts.items():
        q = int(q_str)
        Tc = 1.0 / np.log(1.0 + np.sqrt(q))
        order = 'second-order' if q <= 4 else 'first-order'
        print("\n" + "=" * 60)
        print(f"Potts q={q} ({order}) -- known T_c = {Tc:.4f}")
        print("=" * 60)
        # rows use 'T' not 'param'
        estimate_critical(rows, verbose=True, true_value=Tc)
