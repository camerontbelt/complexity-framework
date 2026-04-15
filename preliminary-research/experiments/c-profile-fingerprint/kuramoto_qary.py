"""Kuramoto q-ary multi-scale C re-run.

Purpose
-------
The binary Kuramoto result in `fingerprint_test.py` encodes each oscillator
as 1 bit (phase > π → 1, else → 0).  That half-plane cut is arbitrary; a
phase in [0, π/q) and [π/q, 2π/q) both collapse to the same bit.  Here we
re-run with q=4 angular sectors of [0, 2π) and compute multi-scale q-ary C.

Question
--------
Does the binary Kuramoto sync-transition peak (K ~ 2–3) survive under q-ary
phase-sector quantisation?  If yes, the Kuramoto paper result is robust.
If no — and combined with the Gray-Scott q-ary disagreement — the binary
pipeline's thresholding is doing real selection work that needs to be
documented in the paper, not assumed away.

Config
------
Q            = 4            # angular sectors of width π/2
POOL_FACTORS = (1, 2, 4, 8)
N_SEEDS      = 3
GRID         = 64
K_vals       = 0.0 .. 10.0 step 0.5 (same as binary sweep)

Estimated runtime: ~3-5 min.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import numpy as np
from compute_C_qary import compute_C_qary, coarsen_history_qary
from criticality_detector import estimate_critical

# Reuse the exact same dynamics as the binary experiment.
import importlib.util
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "fp_test", os.path.join(_HERE, "fingerprint_test.py"))
_ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ft)


def kuramoto_run_phases(K, G=64, steps=400, dt=0.1, seed=42):
    """Same integration as fingerprint_test.kuramoto_run but returns the
    raw phase history (T, G, G) in [0, 2π)."""
    rng   = np.random.default_rng(seed)
    omega = rng.standard_normal((G, G)).astype(np.float32)
    theta = rng.uniform(0, 2*np.pi, (G, G)).astype(np.float32)
    hist  = np.zeros((steps, G, G), dtype=np.float32)
    for t in range(steps):
        hist[t] = theta % (2*np.pi)
        coupling = (np.sin(np.roll(theta, 1, 0) - theta) +
                    np.sin(np.roll(theta, -1, 0) - theta) +
                    np.sin(np.roll(theta, 1, 1) - theta) +
                    np.sin(np.roll(theta, -1, 1) - theta))
        theta = theta + dt * (omega + (K / 4.0) * coupling)
    return hist


def phases_to_qary(phase_hist, q=4):
    """Bin phases into q equal angular sectors."""
    edges = np.linspace(0.0, 2*np.pi, q + 1)[1:-1]
    return np.digitize(phase_hist, edges).astype(np.int32)


# ── config ───────────────────────────────────────────────────────────────────
Q            = 4
POOL_FACTORS = (1, 2, 4, 8)
GRID         = 64
STEPS        = 400
BURNIN       = 100
DT           = 0.1
N_SEEDS      = 3
K_VALS       = np.round(np.arange(0.0, 10.5, 0.5), 1)
C_BURNIN     = 10
C_WINDOW     = 250


def compute_multiscale(hist3d, pools=POOL_FACTORS, q=Q):
    out = {}
    for p in pools:
        grid3d = hist3d if p == 1 else coarsen_history_qary(hist3d, p, q=q)
        if grid3d.shape[1] < 4:
            continue
        Tn, Gn, _ = grid3d.shape
        grid2d = grid3d.reshape(Tn, Gn * Gn).astype(np.int32)
        r = compute_C_qary(grid2d, q=q, burnin=C_BURNIN, window=C_WINDOW)
        out[p] = float(r['score'])
    return out


def run():
    print(f"\n=== Kuramoto q-ary (q={Q}) multi-scale, G={GRID} ===")
    rows = []
    for K in K_VALS:
        per = {p: [] for p in POOL_FACTORS}
        for seed in range(N_SEEDS):
            phase_hist = kuramoto_run_phases(K, G=GRID, steps=STEPS,
                                             dt=DT, seed=seed*11+5)
            hist3d = phases_to_qary(phase_hist[BURNIN:], q=Q)
            scores = compute_multiscale(hist3d)
            for p, s in scores.items():
                per[p].append(s)
        row = {'param': float(K), 'K': float(K)}
        for p in POOL_FACTORS:
            if per[p]:
                row[f'C_{p}'] = float(np.mean(per[p]))
        rows.append(row)
        print(f"  K={K:>5.1f}  " +
              "  ".join(f"C*{p}={row.get(f'C_{p}',0):.3f}" for p in POOL_FACTORS))
    return rows


if __name__ == '__main__':
    t0 = time.time()
    rows = run()
    est  = estimate_critical(rows)
    out  = os.path.join(_HERE, 'kuramoto_qary.json')
    json.dump({'rows': rows, 'estimate': est}, open(out, 'w'), indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")
    print(f"\nDetector consensus K_c = {est['consensus']:.3f}  (conf={est['confidence']:.2f})")
    print(f"  collapse={est['p_collapse']:.3f}  beta={est['p_beta']:.3f}  peak={est['p_peak']:.3f}")

    # Binary comparison (hardcoded values from fingerprint_phase_space.py)
    binary_peak_K = 3.5   # argmax of hardcoded C array ~ 3.5
    print(f"\nBinary C (archived) peaked at K ~ {binary_peak_K}")
    print(f"q-ary detector-consensus delta: {est['consensus'] - binary_peak_K:+.3f}")
