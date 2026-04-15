"""Blume-Capel blind test at G=128.

Rerun of blume_capel_blind_test.py with a larger lattice to test the
finite-size-rounding hypothesis for the D=1.99 failure.

Hypothesis under test:
  At G=48, D=1.99 gave T_c=0.85 vs true 0.42 (error +0.43). We blamed
  finite-size rounding of the first-order / tricritical transition.
  If true, at G=128 the D=1.99 case should show a sharper peak at
  lower T, closer to the published T_t ≈ 0.42.

Also a correlation-length check: Blume-Capel's correlation length at the
Ising-line critical temperatures is ~lattice scale, so G=48 captures them
well for D <= 1.5. We expect D <= 1.5 results to be similar to G=48.

Config changes vs G=48 version:
  G:       48 -> 128 (cells: 2304 -> 16384, ratio 7.1x)
  BURNIN:  2000 -> 4000 (critical slowing down z~=2)
  WINDOW:  120 (unchanged -- longer windows hit diminishing returns)
  N_SEEDS: 3 (unchanged)

Estimated runtime: ~25-30 min (G=48 took 3.2 min; expect ~7x).
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import numpy as np
from compute_C_qary import compute_C_qary, coarsen_history_qary
from criticality_detector import estimate_critical
from blume_capel_blind_test import blume_capel_run, cliff_ratio

# ── config: bigger lattice, longer burnin ────────────────────────────────────
POOLS   = (1, 2, 4, 8)
G       = 128
SWEEPS  = 9000
BURNIN_M = 4000
WINDOW  = 120
SNAP_EVERY = 20
N_SEEDS = 3

C_BURNIN = 10
C_WINDOW = 100


def sweep_T(D, T_values):
    print(f"\n=== Blume-Capel G={G}, D={D:.3f} ===", flush=True)
    rows = []
    for T in T_values:
        per = {p: [] for p in POOLS}
        for seed in range(N_SEEDS):
            hist3d = blume_capel_run(T, D, G=G, sweeps=SWEEPS,
                                     burnin=BURNIN_M, window=WINDOW,
                                     snap_every=SNAP_EVERY, seed=seed)
            for pool in POOLS:
                grid3d = hist3d if pool == 1 else coarsen_history_qary(hist3d, pool, q=3)
                if grid3d.shape[1] < 4: continue
                Tn, Gn, _ = grid3d.shape
                grid2d = grid3d.reshape(Tn, Gn*Gn).astype(np.int32)
                r = compute_C_qary(grid2d, q=3, burnin=C_BURNIN, window=C_WINDOW)
                per[pool].append(r['score'])
        row = {'param': float(T), 'T': float(T), 'D': float(D)}
        for pool in POOLS:
            if per[pool]:
                row[f'C_{pool}'] = float(np.mean(per[pool]))
        rows.append(row)
        print(f"  T={T:.3f}: " + "  ".join(
            f"C*{p}={row.get(f'C_{p}',0):.3f}" for p in POOLS), flush=True)
    return rows


if __name__ == '__main__':
    t0 = time.time()

    # Denser T-grids near the tricritical region for the D=1.9 and D=1.99 cases,
    # since we now have compute budget. Coarser D=0 to avoid over-spending.
    D_CASES = [
        (0.00, np.linspace(1.2, 2.2, 8)),
        (1.00, np.linspace(0.9, 1.9, 8)),
        (1.50, np.linspace(0.6, 1.5, 8)),
        (1.90, np.linspace(0.3, 1.0, 8)),
        (1.99, np.linspace(0.25, 0.95, 10)),   # denser and extends lower
    ]

    all_results = {}
    for D, Ts in D_CASES:
        rows = sweep_T(D, Ts)
        est = estimate_critical(rows)
        cliff = cliff_ratio(rows, pool=1)
        all_results[f'D={D:.2f}'] = {
            'D': D, 'rows': rows, 'estimate': est, 'cliff_ratio': cliff
        }
        print(f"  --> consensus T_c = {est['consensus']:.3f}  "
              f"(conf={est['confidence']:.2f})  "
              f"cliff = {cliff:.2f}" if cliff else "  cliff = NA", flush=True)

    elapsed = (time.time() - t0) / 60
    print(f"\nTotal elapsed: {elapsed:.1f} min", flush=True)

    out = os.path.join(os.path.dirname(__file__), 'blume_capel_blind_test_G128.json')
    json.dump(all_results, open(out, 'w'), indent=2, default=float)
    print(f"Saved: {out}", flush=True)

    # Comparison table vs published values
    published = {0.00: 1.693, 1.00: 1.40, 1.50: 1.15, 1.90: 0.90, 1.99: 0.422}
    print(f"\n{'D':>6}  {'T_c est':>8}  {'published':>10}  {'err':>8}  "
          f"{'conf':>6}  {'cliff':>6}")
    for key, pack in all_results.items():
        D = pack['D']
        est = pack['estimate']
        cliff = pack['cliff_ratio']
        pub = published.get(D, float('nan'))
        err = est['consensus'] - pub
        cliff_str = f'{cliff:.2f}' if cliff else 'NA'
        print(f"  {D:>6.2f}  {est['consensus']:>8.4f}  {pub:>10.4f}  "
              f"{err:>+8.4f}  {est['confidence']:>6.2f}  {cliff_str:>6}")
