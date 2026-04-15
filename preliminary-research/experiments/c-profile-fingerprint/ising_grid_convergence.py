"""Fine-grid convergence study for the criticality detector on 2D Ising.

Question: as we refine the T-grid, does the detector's consensus estimate
converge monotonically to the true T_c = 2.269, or does it plateau at some
biased value?

If it converges: the grid-step-sized errors we saw on Potts q-ary are
genuinely grid-limited and interpolation would close the gap.

If it plateaus above zero: the detector has a structural bias independent
of grid resolution and we need to understand it before claiming accuracy.

Run 4 grid resolutions: dT = {0.20, 0.10, 0.05, 0.025}. Each centred on
T = 2.3 with range +/- 0.4 (so 5, 9, 17, 33 points respectively).

Also retain per-seed C values so we can bootstrap error bars afterward.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from complexity_framework_v9 import _ising_run_fast, compute_C
from multiscale_diagnostic import coarsen_history
from criticality_detector import estimate_critical

POOLS   = (1, 2, 4, 8)
G       = 64
N_SEEDS = 5

CFG = dict(GRID=G, BURNIN=200, WINDOW=200, SNAP_EVERY=3, N_SEEDS=N_SEEDS)
C_BURNIN, C_WINDOW = 20, 180

T_CENTER = 2.3
T_HALF   = 0.4
TRUE_TC  = 2.269


def run_one(T, seed):
    hist = _ising_run_fast(T, CFG, seed=seed)
    Tn = hist.shape[0]
    hist3d = hist.reshape(Tn, G, G)
    per_pool = {}
    for pool in POOLS:
        grid = coarsen_history(hist3d, pool)
        r = compute_C(grid, C_BURNIN, C_WINDOW)
        per_pool[pool] = r['score']
    return per_pool


def sweep(dT):
    n = int(round(2 * T_HALF / dT)) + 1
    Ts = np.linspace(T_CENTER - T_HALF, T_CENTER + T_HALF, n)
    rows = []
    per_seed_rows = []           # retains per-seed scores for bootstrapping
    print(f"\ndT = {dT:.4f}  (n = {n})")
    for T in Ts:
        seed_scores = []
        for seed in range(N_SEEDS):
            seed_scores.append(run_one(float(T), seed))
        row = {'param': float(T)}
        for pool in POOLS:
            vals = [s[pool] for s in seed_scores]
            row[f'C_{pool}'] = float(np.mean(vals))
        rows.append(row)
        per_seed_rows.append({'param': float(T),
                              'seeds': [{f'C_{p}': s[p] for p in POOLS}
                                        for s in seed_scores]})
        print(f"  T={T:.4f}: " + "  ".join(
            f"C*{p}={row[f'C_{p}']:.3f}" for p in POOLS))
    return rows, per_seed_rows


if __name__ == '__main__':
    t0 = time.time()
    dTs = [0.20, 0.10, 0.05, 0.025]
    results = {}
    for dT in dTs:
        rows, per_seed = sweep(dT)
        est = estimate_critical(rows, true_value=TRUE_TC)
        results[f'{dT:.4f}'] = {
            'rows': rows,
            'per_seed': per_seed,
            'estimate': est,
        }
        print(f"  --> consensus={est['consensus']:.4f}  error={est['error_consensus']:+.4f}")

    elapsed = (time.time() - t0) / 60
    print(f"\nTotal elapsed: {elapsed:.1f} min")

    out = os.path.join(os.path.dirname(__file__), 'ising_grid_convergence.json')
    json.dump(results, open(out, 'w'), indent=2, default=float)
    print(f"Saved: {out}")

    # Convergence table
    print(f"\n{'dT':>8}  {'n':>5}  {'collapse':>10}  {'beta':>10}  {'peak':>10}  "
          f"{'consensus':>10}  {'|err|':>8}")
    for dT_str, pack in results.items():
        e = pack['estimate']
        n = len(pack['rows'])
        print(f"  {dT_str:>6}  {n:>5d}  {e['p_collapse']:>10.4f}  {e['p_beta']:>10.4f}  "
              f"{e['p_peak']:>10.4f}  {e['consensus']:>10.4f}  "
              f"{abs(e['error_consensus']):>8.4f}")
