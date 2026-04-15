"""Blume-Capel blind test for the criticality detector + cliff-ratio.

The Blume-Capel model is a spin-1 Ising generalisation:
    H = -J sum_<ij> s_i s_j + D sum_i s_i^2,   s_i in {-1, 0, +1}

At J=1 (our convention) the 2D phase diagram has:
  - A line of critical points T_c(D) that is second-order for D < D_t
    and first-order for D > D_t
  - A tricritical point at (D_t, T_t) ~ (1.965, 0.608) where the order
    switches (numerical estimates vary in the literature: 1.96-1.98 for
    D_t and 0.60-0.62 for T_t)

Why this is a useful blind test:
  1. SAME model, SAME code -- only D changes. Isolates the effect of
     transition order on the detector's behaviour.
  2. The cliff-ratio prediction says: as D crosses D_t, the ratio
     C(peak)/C(peak+1) should increase sharply.
  3. The detector's T_c estimate should track the published T_c(D) curve.
  4. Natural q=3 system (s=-1,0,+1 -> int 0,1,2) -- q-ary C native.

Honesty note on blind-test protocol:
  I (the assistant) have general memory that the Blume-Capel phase
  diagram has a tricritical point somewhere around D ~ 1.95-2.00 and
  T_c(D=0) is in the ballpark of 1.7 (it's roughly Ising/2.5).
  I have NOT looked up precise published numerical values for this run;
  after the run, we'll compare to literature values (Plascak & Figueiredo,
  Silva et al.) and the user is free to verify those independently.

Pre-registered predictions (written BEFORE running):
  1. Detector consensus T_c falls monotonically as D grows from 0 to 2.
  2. Cliff ratio stays low (~1-2) for D <= 1.5, rises as we approach the
     tricritical point, exceeds ~3 for D >= 1.9.
  3. Detector confidence (after boundary fix) drops for the D >= 1.9
     cases because the first-order cliff violates the symmetric-peak
     assumption baked into the scale-collapse indicator.
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import numpy as np
from compute_C_qary import compute_C_qary, coarsen_history_qary
from criticality_detector import estimate_critical


def blume_capel_run(T, D, G=48, sweeps=6000, burnin=2000,
                    window=120, snap_every=20, seed=42):
    """Metropolis single-spin-flip for Blume-Capel on 2D torus.
    Returns (window, G, G) int array with values in {0,1,2} for s in {-1,0,+1}."""
    rng = np.random.default_rng(seed)
    # Initial random spins in {-1,0,+1}
    s = rng.choice([-1, 0, 1], size=(G, G)).astype(np.int8)
    total = burnin + window * snap_every
    hist = np.zeros((window, G, G), dtype=np.int8)
    snap_count = 0

    # Checkerboard masks
    ii, jj = np.mgrid[0:G, 0:G]
    black = ((ii + jj) % 2 == 0); white = ~black

    invT = 1.0 / max(T, 1e-10)

    for sweep in range(total):
        for mask in (black, white):
            # Neighbour sum for each site (Manhattan 4-neighbour)
            nb = (np.roll(s, 1, axis=0) + np.roll(s, -1, axis=0) +
                  np.roll(s, 1, axis=1) + np.roll(s, -1, axis=1)).astype(np.float32)
            proposed = rng.choice([-1, 0, 1], size=(G, G)).astype(np.int8)
            # Energy change: dH = -(s' - s)*nb + D*(s'^2 - s^2)
            ds = (proposed - s).astype(np.float32)
            ds2 = (proposed.astype(np.float32)**2 - s.astype(np.float32)**2)
            dH = -ds * nb + D * ds2
            rand = rng.random((G, G)).astype(np.float32)
            accept = mask & ((dH <= 0) | (rand < np.exp(np.clip(-dH * invT, -30, 0))))
            s = np.where(accept, proposed, s)

        if sweep >= burnin and (sweep - burnin) % snap_every == 0:
            if snap_count < window:
                hist[snap_count] = s + 1    # map {-1,0,1} -> {0,1,2}
                snap_count += 1

    return hist


# ── config ───────────────────────────────────────────────────────────────────
POOLS   = (1, 2, 4, 8)
G       = 48
SWEEPS  = 6000
BURNIN_M = 2000
WINDOW  = 120
SNAP_EVERY = 20
N_SEEDS = 3

C_BURNIN = 10
C_WINDOW = 100


def sweep_T(D, T_values, tag=''):
    print(f"\n=== Blume-Capel D={D:.3f} {tag} ===")
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
            f"C*{p}={row.get(f'C_{p}',0):.3f}" for p in POOLS))
    return rows


def cliff_ratio(rows, pool=1):
    Cs = [r.get(f'C_{pool}', 0) for r in rows]
    idx = int(np.argmax(Cs))
    if idx + 1 >= len(Cs) or Cs[idx] == 0: return None
    return Cs[idx] / max(Cs[idx+1], 1e-6)


if __name__ == '__main__':
    t0 = time.time()

    # Pre-registered predictions (written BEFORE running, no peeking):
    # Detector should find a T_c that varies smoothly with D.
    # Cliff ratio should be low (~1-2) for D <= 1.5, then rise sharply as
    # we approach the tricritical point near D ~ 1.95.
    D_CASES = [
        (0.0, np.linspace(1.0, 2.4, 8)),
        (1.0, np.linspace(0.8, 2.0, 8)),
        (1.5, np.linspace(0.5, 1.6, 8)),
        (1.9, np.linspace(0.3, 1.1, 8)),
        (1.99, np.linspace(0.2, 1.0, 8)),
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
              f"(conf={est['confidence']:.2f})  cliff_ratio = "
              f"{cliff:.2f}" if cliff else "  cliff_ratio = NA")

    elapsed = (time.time() - t0) / 60
    print(f"\nTotal elapsed: {elapsed:.1f} min")

    out = os.path.join(os.path.dirname(__file__), 'blume_capel_blind_test.json')
    json.dump(all_results, open(out, 'w'), indent=2, default=float)
    print(f"Saved: {out}")

    # Summary
    print(f"\n{'D':>6}  {'T_c est':>10}  {'conf':>6}  {'cliff':>8}  {'predicted order':>18}")
    for key, pack in all_results.items():
        D = pack['D']
        est = pack['estimate']
        cliff = pack['cliff_ratio']
        order = 'first-order?' if (cliff and cliff > 3.0) else 'second-order'
        cliff_str = f'{cliff:.2f}' if cliff else 'NA'
        print(f"  {D:>6.2f}  {est['consensus']:>10.4f}  {est['confidence']:>6.2f}  "
              f"{cliff_str:>8}  {order:>18}")
