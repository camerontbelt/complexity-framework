"""Potts multi-scale sweep using q-ary C directly on the raw state field.
No binarization. Runs q in {2, 3, 5, 10}, feeds criticality_detector on output.

Using smaller config than v2 for quicker iteration: G=48, 6000 sweeps, 3 seeds.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from compute_C_qary import compute_C_qary, coarsen_history_qary
from criticality_detector import estimate_critical


def potts_run_raw(T, q, G=48, sweeps=6000, burnin=2000, window=120,
                  snap_every=25, seed=42):
    rng = np.random.default_rng(seed)
    spins = rng.integers(0, q, size=(G, G), dtype=np.int8)

    history = []
    snap_count = 0
    total = burnin + window * snap_every

    ii, jj = np.mgrid[0:G, 0:G]
    black = ((ii + jj) % 2 == 0); white = ~black

    for sweep in range(total):
        for mask in (black, white):
            proposed = rng.integers(0, q, size=(G, G), dtype=np.int8)
            match_cur = None; match_new = None
            for di, dj in [(0,1),(0,-1),(1,0),(-1,0)]:
                nb = np.roll(np.roll(spins, di, axis=0), dj, axis=1)
                mc = (spins == nb).astype(np.float32)
                mn = (proposed == nb).astype(np.float32)
                if match_cur is None: match_cur = mc; match_new = mn
                else: match_cur += mc; match_new += mn
            dE = -(match_new - match_cur)
            accept = mask & ((dE <= 0) |
                             (rng.random((G, G)) < np.exp(np.clip(-dE/max(T,1e-10), -30, 0))))
            spins = np.where(accept, proposed, spins)

        if sweep >= burnin and (sweep - burnin) % snap_every == 0:
            if snap_count < window:
                history.append(spins.copy())
                snap_count += 1

    return np.array(history, dtype=np.int8)


Q_VALUES  = [2, 3, 5, 10]
POOLS     = (1, 2, 4, 8)
G         = 48
SWEEPS    = 6000
BURNIN_M  = 2000
WINDOW    = 120
SNAP_EVERY = 25
N_SEEDS   = 3

C_BURNIN = 10
C_WINDOW = 100


def Tc_of_q(q): return 1.0 / np.log(1.0 + np.sqrt(q))


def sweep_q(q, T_values):
    rows = []
    Tc = Tc_of_q(q)
    print(f"\n=== Potts q={q}, T_c = {Tc:.4f} ===")
    for T in T_values:
        per = {p: [] for p in POOLS}
        for seed in range(N_SEEDS):
            hist3d = potts_run_raw(T, q=q, G=G, sweeps=SWEEPS,
                                   burnin=BURNIN_M, window=WINDOW,
                                   snap_every=SNAP_EVERY, seed=seed)
            for pool in POOLS:
                if pool == 1:
                    grid3d = hist3d
                else:
                    grid3d = coarsen_history_qary(hist3d, pool, q)
                if grid3d.shape[1] < 4:
                    continue
                # flatten to (T, W)
                Tn, Gn, _ = grid3d.shape
                grid2d = grid3d.reshape(Tn, Gn*Gn).astype(np.int32)
                r = compute_C_qary(grid2d, q=q, burnin=C_BURNIN, window=C_WINDOW)
                per[pool].append(r['score'])
        row = {'T': float(T), 'param': float(T), 'q': q, 'T_c': Tc}
        for pool in POOLS:
            if per[pool]:
                row[f'C_{pool}'] = float(np.mean(per[pool]))
        rows.append(row)
        print(f"  T={T:.3f}: " + "  ".join(
            f"C*{p}={row.get(f'C_{p}',0):.3f}" for p in POOLS))
    return rows


if __name__ == '__main__':
    t0 = time.time()

    def T_grid(q, n=9):
        Tc = Tc_of_q(q)
        return np.round(np.linspace(0.7*Tc, 1.4*Tc, n), 4).tolist()

    all_results = {}
    for q in Q_VALUES:
        all_results[q] = sweep_q(q, T_grid(q))

    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")

    out_json = os.path.join(os.path.dirname(__file__), 'potts_qary_sweep.json')
    with open(out_json, 'w') as f:
        json.dump({str(q): v for q, v in all_results.items()}, f, indent=2, default=float)
    print(f"Saved: {out_json}")

    # Run detector on each q
    print("\n" + "="*64)
    print("Criticality detector on q-ary C data:")
    print("="*64)
    for q, rows in all_results.items():
        Tc = Tc_of_q(q)
        order = 'second-order' if q <= 4 else 'first-order'
        print(f"\nPotts q={q} ({order}) -- known T_c = {Tc:.4f}")
        estimate_critical(rows, verbose=True, true_value=Tc)
