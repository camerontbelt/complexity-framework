"""
Multi-scale C diagnostic.

For each of {Ising, DP, Contact Process, Sandpile}, sweep the control
parameter and compute C at pool factors {1, 2, 4, 8}. Check whether:
  (a) Type-A substrates (Ising) retain their peak at p_c under coarsening
  (b) DP's peak moves toward p_c at coarser scales
  (c) Contact Process stays peaked at λ_c at coarser scales
  (d) Sandpile lights up at coarser scales (SOC = scale invariant)

Also check whether w_OP_t (temporal opacity) is the gate that opens at
coarser scales — that's the specific prediction.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from complexity_framework_v9 import (
    compute_C, _dp_run, DP_CFG, _sandpile_run, SANDPILE_CFG
)

# We need an Ising runner and a CP runner too
from fingerprint_test import contact_process_run


# ═══════════════════════════════════════════════════════════════════════════════
# Ising runner (Metropolis, 2D, sync-sweep)
# ═══════════════════════════════════════════════════════════════════════════════
def ising_run(T, G=64, steps=400, seed=42):
    rng = np.random.default_rng(seed)
    grid = rng.choice([-1, 1], size=(G, G)).astype(np.int8)
    beta = 1.0 / max(T, 1e-6)
    history = []
    for t in range(steps):
        # one sweep = G*G single-site updates (checkerboard)
        for parity in (0, 1):
            mask = ((np.add.outer(np.arange(G), np.arange(G)) & 1) == parity)
            padded = np.pad(grid, 1, mode='wrap')
            nb = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                  padded[1:-1, :-2] + padded[1:-1, 2:])
            dE = 2 * grid * nb
            flip_prob = np.exp(-beta * dE.clip(0))
            r = rng.random((G, G))
            flip = mask & ((dE <= 0) | (r < flip_prob))
            grid = np.where(flip, -grid, grid).astype(np.int8)
        # binarise (+1 -> 1, -1 -> 0)
        history.append(((grid + 1) // 2).ravel().copy())
    return np.array(history, dtype=np.int8)


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-scale wrapper
# ═══════════════════════════════════════════════════════════════════════════════
def coarsen_history(history_3d, pool):
    """history_3d: (T, G, G) binary. Returns (T, (G/pool)^2) binary at active-pct threshold."""
    T, G, _ = history_3d.shape
    if pool == 1:
        return history_3d.reshape(T, G*G).astype(np.int8)
    n = G // pool
    hist_trim = history_3d[:, :n*pool, :n*pool]
    # block-mean: reshape → mean over block axes
    blocks = hist_trim.reshape(T, n, pool, n, pool).mean(axis=(2, 4))
    # binarise with per-frame threshold keeping ~25% active
    thresh = np.percentile(blocks, 75, axis=(1, 2), keepdims=True)
    binary = (blocks > thresh).astype(np.int8)
    return binary.reshape(T, n*n)


def compute_C_at_scales(history_flat_unused, history_3d, burnin, window, pools=(1, 2, 4, 8)):
    """Returns dict pool -> compute_C result."""
    results = {}
    for p in pools:
        grid = coarsen_history(history_3d, p)
        if grid.shape[1] < 4:  # too small
            continue
        results[p] = compute_C(grid, burnin, window)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Sweep
# ═══════════════════════════════════════════════════════════════════════════════
POOLS = (1, 2, 4, 8)
N_SEEDS = 3
BURNIN, WINDOW = 50, 200

def run_substrate(name, runner, param_vals, G, grid_key='G'):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    rows = []
    for p in param_vals:
        per_seed = {pool: [] for pool in POOLS}
        wOPt_per_seed = {pool: [] for pool in POOLS}
        wOPs_per_seed = {pool: [] for pool in POOLS}
        for s in range(N_SEEDS):
            hist_flat = runner(p, seed=s)
            # reshape to (T, G, G)
            T = hist_flat.shape[0]
            hist_3d = hist_flat.reshape(T, G, G)
            res = compute_C_at_scales(None, hist_3d, BURNIN, WINDOW, POOLS)
            for pool, r in res.items():
                per_seed[pool].append(r['score'])
                wOPt_per_seed[pool].append(r['w_OP_t'])
                wOPs_per_seed[pool].append(r['w_OP_s'])
        row = dict(param=p)
        for pool in POOLS:
            if per_seed[pool]:
                row[f'C_{pool}']    = float(np.mean(per_seed[pool]))
                row[f'wOPt_{pool}'] = float(np.mean(wOPt_per_seed[pool]))
                row[f'wOPs_{pool}'] = float(np.mean(wOPs_per_seed[pool]))
        rows.append(row)
        print(f"  {p:>7.3f}: " + "  ".join(
            f"C×{pool}={row.get(f'C_{pool}', 0):.3f}(wOPt={row.get(f'wOPt_{pool}', 0):.2f})"
            for pool in POOLS if f'C_{pool}' in row))
    return rows


# ── substrate runners (each returns flat (T, G*G)) ────────────────────────────
def ising_runner(T, seed): return ising_run(T, G=64, steps=300, seed=seed)
def dp_runner(p, seed):
    cfg = DP_CFG.copy(); cfg['GRID']=64; cfg['STEPS']=300
    return _dp_run(p, cfg, seed=seed)
def cp_runner(lam, seed): return contact_process_run(lam, G=64, steps=300, seed=seed)
def sandpile_runner(eps, seed):
    cfg = SANDPILE_CFG.copy(); cfg['GRID']=64; cfg['STEPS']=300
    return _sandpile_run(eps, cfg, seed=seed)


if __name__ == '__main__':
    t0 = time.time()

    # Focused sweeps — tight around the critical point
    ising_params    = [1.8, 2.0, 2.1, 2.2, 2.269, 2.35, 2.5, 2.8]       # T_c = 2.269
    dp_params       = [0.20, 0.25, 0.275, 0.2873, 0.30, 0.325, 0.35, 0.375, 0.40]  # p_c = 0.2873
    cp_params       = [1.0, 1.3, 1.5, 1.6489, 1.8, 2.0, 2.3]             # λ_c = 1.6489
    sandpile_params = [0.0, 0.005, 0.01, 0.05, 0.1]                     # natural state = 0

    all_results = {}
    all_results['Ising']    = run_substrate('Ising (T_c=2.269)', ising_runner, ising_params, G=64)
    all_results['DP']       = run_substrate('DP (p_c=0.2873)',  dp_runner,     dp_params, G=64)
    all_results['CP']       = run_substrate('Contact Process (λ_c=1.6489)', cp_runner, cp_params, G=64)
    all_results['Sandpile'] = run_substrate('Sandpile (SOC at eps=0)', sandpile_runner, sandpile_params, G=64)

    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")

    # ── plot: C profile at each pool factor, per substrate ───────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    substrates = list(all_results.keys())
    p_cs = {'Ising': 2.269, 'DP': 0.2873, 'CP': 1.6489, 'Sandpile': 0.0}

    for col, name in enumerate(substrates):
        rows = all_results[name]
        params = [r['param'] for r in rows]
        ax_C = axes[0, col]
        ax_W = axes[1, col]
        for pool in POOLS:
            key_C = f'C_{pool}'; key_W = f'wOPt_{pool}'
            Cs = [r.get(key_C, 0) for r in rows]
            Ws = [r.get(key_W, 0) for r in rows]
            ax_C.plot(params, Cs, marker='o', label=f'×{pool}')
            ax_W.plot(params, Ws, marker='s', label=f'×{pool}')
        ax_C.axvline(p_cs[name], color='red', ls='--', lw=0.8, alpha=0.6)
        ax_W.axvline(p_cs[name], color='red', ls='--', lw=0.8, alpha=0.6)
        ax_C.set_title(f'{name}: C vs param at each scale')
        ax_W.set_title(f'{name}: w_OP_t vs param at each scale')
        ax_C.set_xlabel('control param');  ax_C.set_ylabel('C')
        ax_W.set_xlabel('control param');  ax_W.set_ylabel('w_OP_t')
        ax_C.legend(fontsize=8);           ax_W.legend(fontsize=8)
        ax_C.grid(alpha=0.3);              ax_W.grid(alpha=0.3)

    plt.suptitle('Multi-scale diagnostic: C and temporal-opacity weight at pool factors {1, 2, 4, 8}',
                 y=1.00, fontsize=12, weight='bold')
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'multiscale_diagnostic.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"Saved: {out}")

    # save raw data
    import json
    out_json = os.path.join(os.path.dirname(__file__), 'multiscale_diagnostic.json')
    with open(out_json, 'w') as f:
        json.dump({k: [dict(r) for r in v] for k, v in all_results.items()}, f, indent=2, default=float)
    print(f"Saved: {out_json}")
