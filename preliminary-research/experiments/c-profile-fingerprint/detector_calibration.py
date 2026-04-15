"""Calibrate the confidence score of the criticality detector.

Generates N synthetic multi-scale C sweeps with known T_c, runs the detector,
and measures whether reported confidence correlates with actual estimation
error. Produces a calibration curve: does "confidence 0.8" actually mean
"80 percent chance of being within epsilon"?

Synthetic model for C(T; pool):
  C(T) = A * exp(-((T - T_c)/sigma)^2) * f(pool) + noise
where f(pool) gives a scale-dependent amplitude and noise models seed
variance. Four shape variants are included to stress-test the detector:

  'ising-like'  : symmetric Gaussian, amplitudes grow with pool (scale-invariant)
  'potts-like'  : symmetric Gaussian, amplitudes roughly equal across pools
  'sir-like'    : asymmetric shoulder (sharp rise, slow decay), low amplitude
  'cp-like'     : rising plateau (supercritical phase remains high) -- the
                  pathology we just identified on real CP data
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from criticality_detector import estimate_critical

POOLS = (1, 2, 4, 8)


def synthetic_sweep(shape, T_c, n_points=9, T_range=(None, None),
                    noise=0.02, seed=0):
    """Return rows list suitable for estimate_critical()."""
    rng = np.random.default_rng(seed)
    lo, hi = T_range
    if lo is None: lo = T_c - 0.6
    if hi is None: hi = T_c + 0.8
    Ts = np.linspace(lo, hi, n_points)
    rows = []
    for T in Ts:
        dx = T - T_c
        if shape == 'ising-like':
            # symmetric gaussian, amplitudes grow with pool (scale-invariant)
            amps = {1: 0.6, 2: 0.7, 4: 0.8, 8: 0.9}
            sigma = 0.2
            Cs = {p: amps[p] * np.exp(-(dx/sigma)**2) for p in POOLS}
        elif shape == 'potts-like':
            amps = {1: 0.5, 2: 0.45, 4: 0.40, 8: 0.35}
            sigma = 0.15
            Cs = {p: amps[p] * np.exp(-(dx/sigma)**2) for p in POOLS}
        elif shape == 'sir-like':
            # asymmetric: sharp rise, exponential decay on the supercritical side
            sigma_l, sigma_r = 0.05, 0.35
            amps = {1: 0.08, 2: 0.07, 4: 0.05, 8: 0.03}
            shape_f = np.where(dx < 0,
                                np.exp(-(dx/sigma_l)**2),
                                np.exp(-dx/sigma_r))
            Cs = {p: amps[p] * shape_f for p in POOLS}
        elif shape == 'cp-like':
            # rising plateau: critical point is INFLECTION, not peak
            # C stays near plateau for all T >= T_c
            plateau = {1: 0.6, 2: 0.55, 4: 0.50, 8: 0.48}
            rise = 1.0 / (1.0 + np.exp(-10 * (T - T_c + 0.05)))  # sigmoid
            Cs = {p: plateau[p] * rise for p in POOLS}
        else:
            raise ValueError(shape)

        row = {'param': float(T)}
        for p in POOLS:
            v = float(Cs[p]) + rng.normal(0, noise)
            row[f'C_{p}'] = max(0.0, v)
        rows.append(row)
    return rows


def run_calibration(N=200, noise_levels=(0.01, 0.03, 0.06, 0.10),
                    shapes=('ising-like', 'potts-like', 'sir-like', 'cp-like')):
    results = []
    for shape in shapes:
        for noise in noise_levels:
            for trial in range(N):
                T_c = np.random.uniform(1.0, 3.0)
                rows = synthetic_sweep(shape, T_c, noise=noise, seed=trial)
                r = estimate_critical(rows, true_value=T_c)
                results.append({
                    'shape': shape,
                    'noise': noise,
                    'confidence': r['confidence'],
                    'abs_error': abs(r['error_consensus']),
                    'edge_hit': r['edge_hit'],
                })
    return results


def summarise(results):
    import collections
    # Group by confidence decile, measure mean/median abs error
    confs = np.array([r['confidence'] for r in results])
    errs  = np.array([r['abs_error']  for r in results])
    print(f"\n{'conf bin':>12}  {'n':>5}  {'mean |err|':>12}  {'median |err|':>14}  {'P(|err|<0.1)':>14}")
    print("-" * 70)
    edges = np.linspace(0, 1, 11)
    for i in range(10):
        lo, hi = edges[i], edges[i+1]
        mask = (confs >= lo) & (confs < hi + (1e-9 if i==9 else 0))
        if mask.sum() == 0:
            print(f"  [{lo:.1f},{hi:.1f})  -- empty --")
            continue
        e = errs[mask]
        print(f"  [{lo:.1f},{hi:.1f})  {mask.sum():>5d}  {e.mean():>12.4f}  {np.median(e):>14.4f}  {(e<0.1).mean():>14.2%}")

    # Per-shape summary
    shapes = sorted({r['shape'] for r in results})
    print(f"\n{'shape':>14}  {'mean conf':>10}  {'mean |err|':>12}  {'P(|err|<0.1)':>14}")
    print("-" * 60)
    for s in shapes:
        e = np.array([r['abs_error'] for r in results if r['shape'] == s])
        c = np.array([r['confidence']  for r in results if r['shape'] == s])
        print(f"  {s:>12}  {c.mean():>10.3f}  {e.mean():>12.4f}  {(e<0.1).mean():>14.2%}")


def plot_calibration(results, outfile):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    confs = np.array([r['confidence'] for r in results])
    errs  = np.array([r['abs_error']  for r in results])
    shapes = [r['shape'] for r in results]
    palette = {'ising-like':'tab:blue','potts-like':'tab:orange',
               'sir-like':'tab:green','cp-like':'tab:red'}
    colors = [palette[s] for s in shapes]
    axes[0].scatter(confs, errs, c=colors, alpha=0.4, s=10)
    axes[0].set_xlabel('reported confidence')
    axes[0].set_ylabel('|true - estimate|')
    axes[0].set_title('Raw scatter: confidence vs error')
    axes[0].grid(alpha=0.3)
    for s, c in palette.items():
        axes[0].scatter([], [], c=c, label=s)
    axes[0].legend(fontsize=8)

    # Calibration curve: mean error in each confidence decile
    edges = np.linspace(0, 1, 11)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_err = []
    for i in range(10):
        lo, hi = edges[i], edges[i+1]
        mask = (confs >= lo) & (confs < hi + (1e-9 if i==9 else 0))
        mean_err.append(errs[mask].mean() if mask.sum() > 0 else np.nan)
    axes[1].plot(centers, mean_err, marker='o', lw=2)
    axes[1].axhline(0.1, color='gray', ls='--', lw=0.8, alpha=0.7, label='tolerance eps=0.1')
    axes[1].set_xlabel('reported confidence (decile)')
    axes[1].set_ylabel('mean |error| in bin')
    axes[1].set_title('Calibration curve\n(well-calibrated: error falls as confidence rises)')
    axes[1].grid(alpha=0.3); axes[1].legend()

    plt.suptitle(f'Criticality detector confidence calibration  (N={len(results)} synthetic trials)')
    plt.tight_layout()
    plt.savefig(outfile, dpi=130, bbox_inches='tight')
    print(f'Saved: {outfile}')


if __name__ == '__main__':
    results = run_calibration(N=200)
    summarise(results)
    out = os.path.join(os.path.dirname(__file__), 'detector_calibration.png')
    plot_calibration(results, out)
