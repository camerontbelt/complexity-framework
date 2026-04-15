"""Bootstrap-over-seeds error bars for the criticality detector.

Reads a results file that retains per-seed C values at each parameter,
resamples seeds with replacement B times, runs the detector on each
bootstrap replicate, and reports the 2.5 / 50 / 97.5 percentiles of the
consensus estimate.

Usage:
  python detector_bootstrap.py <results.json>

Expects the JSON to have per-dT packs containing:
  'per_seed': [{'param': T, 'seeds': [{'C_1':..,'C_2':..,...}, ...]}, ...]
as produced by ising_grid_convergence.py.
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from criticality_detector import estimate_critical

POOLS = (1, 2, 4, 8)


def bootstrap(per_seed_rows, B=500, seed=0):
    rng = np.random.default_rng(seed)
    n_seeds = len(per_seed_rows[0]['seeds'])
    estimates = []
    for b in range(B):
        resample_idx = rng.integers(0, n_seeds, size=n_seeds)
        rows = []
        for pack in per_seed_rows:
            seeds = pack['seeds']
            row = {'param': pack['param']}
            for p in POOLS:
                vals = [seeds[i][f'C_{p}'] for i in resample_idx]
                row[f'C_{p}'] = float(np.mean(vals))
            rows.append(row)
        r = estimate_critical(rows)
        estimates.append(r['consensus'])
    return np.array(estimates)


def run_from_file(path, true_value=None):
    data = json.load(open(path))
    print(f"\nBootstrap analysis of {os.path.basename(path)}  (B=500)")
    print(f"{'dT':>8}  {'n':>4}  {'median':>8}  {'2.5%':>8}  {'97.5%':>8}  "
          f"{'CI width':>10}  {'err':>8}")
    print("-" * 70)
    for dT_str, pack in data.items():
        per_seed = pack['per_seed']
        ests = bootstrap(per_seed, B=500)
        lo, med, hi = np.percentile(ests, [2.5, 50, 97.5])
        n = len(per_seed)
        err_str = f"{med - true_value:+.4f}" if true_value is not None else '--'
        print(f"  {dT_str:>6}  {n:>4d}  {med:>8.4f}  {lo:>8.4f}  {hi:>8.4f}  "
              f"{hi-lo:>10.4f}  {err_str:>8}")


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'ising_grid_convergence.json'
    path = os.path.join(os.path.dirname(__file__), path)
    run_from_file(path, true_value=2.269)
