"""Gray-Scott q-ary multi-scale C re-run.

Purpose
-------
The original gray-scott-multiscale.py binarises the continuous v-field via
a per-frame 75th-percentile threshold.  That throws away the distinction
between "barely above threshold" and "near-saturation" cells.  Here we
re-run the same 6 Pearson/Munafo parameter sets but with a q=4 quantile
quantisation of the v-field, then compute multi-scale q-ary C.

Question
--------
Does the regime ordering (trivial < ordered < complex; complex > chaotic)
established under binary C survive when we keep 4 intensity levels?

If yes: the binary result was robust to quantisation — the original paper
claim is reproducible.  If no: the binary result depended on the
thresholding, and we need to update the paper.

Config
------
POOL_FACTORS = (1, 2, 4, 8)   # matches c-metric-math multi-scale spec
Q            = 4              # 4 quantile bins
N_FRAMES     = 50
N_SEEDS      = 2
GRID         = 128

Estimated runtime: ~5-10 min (dominated by simulation, not C computation).
"""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'c-profile-fingerprint'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import numpy as np
from compute_C_qary import compute_C_qary, coarsen_history_qary

# Import simulation from the existing module without executing main().
import importlib.util
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "gs_ms", os.path.join(_HERE, "gray-scott-multiscale.py"))
_gs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gs)

simulate       = _gs.simulate
PARAMETER_SETS = _gs.PARAMETER_SETS
GRID           = _gs.GRID

# ── q-ary config ─────────────────────────────────────────────────────────────
Q            = 4
POOL_FACTORS = (1, 2, 4, 8)
N_SEEDS      = 2
C_BURNIN     = 5      # frames; we only have 50 frames total
C_WINDOW     = 40


def quantise_qary(frames, q=Q):
    """Stack frames → (T, G, G), quantise to {0,..,q-1} via global quantile
    bins computed from the pooled values so each state is roughly equally
    populated across the whole window.  Returns int32 (T, G, G)."""
    arr = np.stack(frames, axis=0).astype(np.float32)   # (T, G, G)
    # Global quantile edges -> q-1 cuts
    qs    = np.linspace(0.0, 1.0, q + 1)[1:-1]
    edges = np.quantile(arr, qs)
    out   = np.digitize(arr, edges).astype(np.int32)    # values in 0..q-1
    return out


def compute_multiscale(hist3d, pools=POOL_FACTORS, q=Q):
    """Return dict pool -> C score."""
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
    print(f"\n=== Gray-Scott q-ary (q={Q}) multi-scale ===")
    results = {}
    for name, f, k, desc, expected in PARAMETER_SETS:
        per_pool = {p: [] for p in POOL_FACTORS}
        for seed in range(N_SEEDS):
            frames, _ = simulate(f, k, seed=seed)
            hist3d = quantise_qary(frames, q=Q)
            scores = compute_multiscale(hist3d)
            for p, s in scores.items():
                per_pool[p].append(s)
        mean_by = {p: float(np.mean(v)) for p, v in per_pool.items() if v}
        peak = max(mean_by, key=lambda p: mean_by[p]) if mean_by else None
        results[name] = {
            'f': f, 'k': k, 'expected': expected, 'desc': desc,
            'C_by_pool': mean_by,
            'peak_pool': int(peak) if peak is not None else None,
            'peak_C':    float(mean_by[peak]) if peak is not None else 0.0,
        }
        print(f"  {name:18s} [{expected:8s}]  " +
              "  ".join(f"x{p}={mean_by.get(p, 0):.3f}" for p in POOL_FACTORS) +
              f"  peak=x{peak}")
    return results


if __name__ == '__main__':
    t0 = time.time()
    res = run()
    out = os.path.join(_HERE, 'gray_scott_qary.json')
    json.dump(res, open(out, 'w'), indent=2, default=float)
    print(f"\nSaved: {out}")
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")

    # Summary vs. class order: trivial < ordered < complex > chaotic
    print("\n--- Ordering by peak_C ---")
    ordered = sorted(res.items(), key=lambda kv: -kv[1]['peak_C'])
    for name, pack in ordered:
        print(f"  {pack['peak_C']:.3f}  {name:18s} [{pack['expected']}]")
