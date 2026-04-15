"""SIR (q=3 natively: S=0, I=1, R=2) with q-ary C + criticality detector.

Tests whether dropping binarization and using the native 3-state field
gives a cleaner critical-point signal than the binary (infected-only) pipeline.

Known threshold: R_0 = beta * <neighbours> * (1-gamma) / gamma approx 1.
With gamma=0.1, Moore neighbourhood (8), beta_c approx 0.0139.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
from compute_C_qary import compute_C_qary, coarsen_history_qary
from criticality_detector import estimate_critical


def sir_run_raw(beta, G=64, steps=250, gamma=0.10, init_I=0.01, seed=42):
    """Return full (steps, G, G) state history with values in {0=S, 1=I, 2=R}."""
    rng = np.random.default_rng(seed)
    state = np.zeros((G, G), dtype=np.int8)                   # 0 = S
    state[rng.random((G, G)) < init_I] = 1                    # seed infected
    hist = np.zeros((steps, G, G), dtype=np.int8)
    for t in range(steps):
        hist[t] = state
        n_inf = np.zeros((G, G), dtype=np.int32)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0: continue
                n_inf += (np.roll(np.roll(state, di, axis=0), dj, axis=1) == 1)
        new_state = state.copy()
        # S -> I
        susceptible = (state == 0)
        p_inf = 1.0 - (1.0 - beta) ** n_inf.astype(np.float32)
        inf_mask = susceptible & (rng.random((G, G)).astype(np.float32) < p_inf)
        new_state[inf_mask] = 1
        # I -> R
        rec_mask = (state == 1) & (rng.random((G, G)) < gamma)
        new_state[rec_mask] = 2
        state = new_state
    return hist


# ── experiment config ────────────────────────────────────────────────────────
POOLS   = (1, 2, 4, 8)
G       = 64
STEPS   = 250
N_SEEDS = 5
GAMMA   = 0.10

# Measurement window: skip initial super-short transient, capture active epidemic
C_BURNIN = 10
C_WINDOW = 150

# Sweep beta densely around predicted threshold 0.0139
BETA_VALUES = np.round(np.linspace(0.005, 0.060, 12), 4).tolist()


def beta_c_approx(gamma=GAMMA):
    # R_0 = 1 => beta = gamma / ((1-gamma) * <n_neighbours>)
    return gamma / ((1.0 - gamma) * 8.0)


def sweep():
    rows = []
    bc = beta_c_approx()
    print(f"=== SIR q-ary sweep (predicted beta_c ~ {bc:.4f}) ===")
    for beta in BETA_VALUES:
        per = {p: [] for p in POOLS}
        for seed in range(N_SEEDS):
            hist3d = sir_run_raw(beta, G=G, steps=STEPS, gamma=GAMMA, seed=seed)
            for pool in POOLS:
                grid3d = hist3d if pool == 1 else coarsen_history_qary(hist3d, pool, q=3)
                if grid3d.shape[1] < 4: continue
                Tn, Gn, _ = grid3d.shape
                grid2d = grid3d.reshape(Tn, Gn*Gn).astype(np.int32)
                r = compute_C_qary(grid2d, q=3, burnin=C_BURNIN, window=C_WINDOW)
                per[pool].append(r['score'])
        row = {'param': float(beta), 'beta': float(beta)}
        for pool in POOLS:
            if per[pool]:
                row[f'C_{pool}'] = float(np.mean(per[pool]))
        rows.append(row)
        print(f"  beta={beta:.4f}: " + "  ".join(
            f"C*{p}={row.get(f'C_{p}',0):.3f}" for p in POOLS))
    return rows


if __name__ == '__main__':
    t0 = time.time()
    rows = sweep()
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.2f} min")

    out_json = os.path.join(os.path.dirname(__file__), 'sir_qary_sweep.json')
    with open(out_json, 'w') as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"Saved: {out_json}")

    print("\n" + "="*60)
    print(f"Criticality detector (predicted beta_c ~ {beta_c_approx():.4f})")
    print("="*60)
    estimate_critical(rows, verbose=True, true_value=beta_c_approx())
