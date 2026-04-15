"""
Extended multi-scale C diagnostic — SIR, RBN, Voter.

Follow-up to multiscale_diagnostic.py. Tests whether the "mesoscale rescue"
effect seen for DP also appears for:
  - SIR (absorbing-state, epidemic threshold at R0=1, i.e. β ~ 0.0125 for γ=0.1)
  - RBN (Kauffman edge-of-chaos at K=2; uses TEMPORAL coarsening since no spatial lattice)
  - Voter (Type C, absorbing consensus at μ=0)

Predictions:
  SIR:    peak shifts toward β_c at coarse spatial scales (like DP)
  RBN:    peak sharpens at K=2 under temporal coarsening
  Voter:  signal amplifies at coarser scales (scale-invariant coarsening)
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from complexity_framework_v9 import compute_C, _sir_run, SIR_CFG, _rbn_run, RBN_CFG
from fingerprint_test import voter_model_run
from multiscale_diagnostic import coarsen_history

POOLS   = (1, 2, 4, 8)
N_SEEDS = 3
BURNIN, WINDOW = 50, 200


# ── runners returning flat (T, cells) binary histories ────────────────────────
def sir_runner(beta, seed):
    cfg = SIR_CFG.copy(); cfg['GRID']=64; cfg['STEPS']=300
    # _sir_run returns (hist_I, hist_S); use the infected field as the binary grid
    hist_I, hist_S = _sir_run(beta, cfg, seed=seed)
    return hist_I

def rbn_runner(K, seed):
    cfg = RBN_CFG.copy(); cfg['N']=400; cfg['STEPS']=300
    return _rbn_run(K, cfg, seed=seed)   # (T, N)

def voter_runner(mu, seed):
    return voter_model_run(mu, G=64, steps=300, burnin=50, window=200, seed=seed)


# ── temporal coarsening (for non-spatial substrates like RBN) ─────────────────
def coarsen_temporal(hist_flat, stride):
    """Average blocks of `stride` consecutive time steps and re-binarise.

    hist_flat : (T, N) binary
    stride    : int  — how many time steps to average together
    returns   : (T//stride, N) binary
    """
    if stride == 1:
        return hist_flat.astype(np.int8)
    T, N = hist_flat.shape
    n = T // stride
    trimmed = hist_flat[:n*stride]
    blocks  = trimmed.reshape(n, stride, N).mean(axis=1)
    thresh  = np.percentile(blocks, 75, axis=1, keepdims=True)
    return (blocks > thresh).astype(np.int8)


# ── per-substrate sweep ───────────────────────────────────────────────────────
def sweep_spatial(name, runner, params, G):
    print(f"\n{'='*60}\n{name} (spatial coarsening)\n{'='*60}")
    rows = []
    for p in params:
        per = {pool: [] for pool in POOLS}
        wopt = {pool: [] for pool in POOLS}
        for s in range(N_SEEDS):
            hist_flat = runner(p, seed=s)
            T = hist_flat.shape[0]
            hist_3d = hist_flat.reshape(T, G, G)
            for pool in POOLS:
                grid = coarsen_history(hist_3d, pool)
                if grid.shape[1] < 4:
                    continue
                r = compute_C(grid, BURNIN, WINDOW)
                per[pool].append(r['score'])
                wopt[pool].append(r['w_OP_t'])
        row = {'param': p}
        for pool in POOLS:
            if per[pool]:
                row[f'C_{pool}']    = float(np.mean(per[pool]))
                row[f'wOPt_{pool}'] = float(np.mean(wopt[pool]))
        rows.append(row)
        print(f"  {p:>7.4f}: " + "  ".join(
            f"C×{pool}={row.get(f'C_{pool}', 0):.3f}(wOPt={row.get(f'wOPt_{pool}', 0):.2f})"
            for pool in POOLS if f'C_{pool}' in row))
    return rows


def sweep_temporal(name, runner, params, STRIDES=(1, 2, 4, 8)):
    print(f"\n{'='*60}\n{name} (temporal coarsening)\n{'='*60}")
    rows = []
    for p in params:
        per = {s: [] for s in STRIDES}
        wopt = {s: [] for s in STRIDES}
        for seed in range(N_SEEDS):
            hist = runner(p, seed=seed)
            for stride in STRIDES:
                grid = coarsen_temporal(hist, stride)
                T_new = grid.shape[0]
                if T_new < BURNIN + 20:
                    continue
                new_burnin = max(5, BURNIN // stride)
                new_window = min(WINDOW // stride, T_new - new_burnin - 1)
                r = compute_C(grid, new_burnin, new_window)
                per[stride].append(r['score'])
                wopt[stride].append(r['w_OP_t'])
        row = {'param': p}
        for stride in STRIDES:
            if per[stride]:
                row[f'C_t{stride}']    = float(np.mean(per[stride]))
                row[f'wOPt_t{stride}'] = float(np.mean(wopt[stride]))
        rows.append(row)
        print(f"  {p:>7.3f}: " + "  ".join(
            f"C/t{s}={row.get(f'C_t{s}', 0):.3f}(wOPt={row.get(f'wOPt_t{s}', 0):.2f})"
            for s in STRIDES if f'C_t{s}' in row))
    return rows


if __name__ == '__main__':
    t0 = time.time()

    # β_c for this SIR setup: R0=1 → β ≈ γ/8 (Moore nbhd, ~8 nbrs) = 0.0125
    sir_params   = [0.005, 0.010, 0.0125, 0.015, 0.020, 0.030, 0.050, 0.080]
    rbn_params   = [1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0]   # K_c = 2
    voter_params = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3]

    all_results = {}
    all_results['SIR']   = sweep_spatial('SIR (β_c ~ 0.0125)', sir_runner, sir_params, G=64)
    all_results['RBN']   = sweep_temporal('RBN (K_c = 2)',     rbn_runner, rbn_params)
    all_results['Voter'] = sweep_spatial('Voter (μ = 0 attractor)', voter_runner, voter_params, G=64)

    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    p_cs   = {'SIR': 0.0125, 'RBN': 2.0, 'Voter': 0.0}
    # which pool/stride keys to use
    keys = {'SIR':   [('C_1','wOPt_1'), ('C_2','wOPt_2'), ('C_4','wOPt_4'), ('C_8','wOPt_8')],
            'RBN':   [('C_t1','wOPt_t1'), ('C_t2','wOPt_t2'), ('C_t4','wOPt_t4'), ('C_t8','wOPt_t8')],
            'Voter': [('C_1','wOPt_1'), ('C_2','wOPt_2'), ('C_4','wOPt_4'), ('C_8','wOPt_8')]}
    labels = {'SIR':['×1','×2','×4','×8'], 'RBN':['/t1','/t2','/t4','/t8'], 'Voter':['×1','×2','×4','×8']}

    for col, name in enumerate(['SIR','RBN','Voter']):
        rows = all_results[name]
        params = [r['param'] for r in rows]
        for (ck, wk), lbl in zip(keys[name], labels[name]):
            Cs = [r.get(ck, 0) for r in rows]
            Ws = [r.get(wk, 0) for r in rows]
            axes[0, col].plot(params, Cs, marker='o', label=lbl)
            axes[1, col].plot(params, Ws, marker='s', label=lbl)
        for ax in (axes[0, col], axes[1, col]):
            ax.axvline(p_cs[name], color='red', ls='--', lw=0.8, alpha=0.6)
            ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlabel('control param')
        axes[0, col].set_title(f'{name}: C vs param')
        axes[1, col].set_title(f'{name}: w_OP_t vs param')
        axes[0, col].set_ylabel('C')
        axes[1, col].set_ylabel('w_OP_t')

    plt.suptitle('Extended multi-scale diagnostic — SIR + RBN + Voter\n'
                 '(SIR, Voter: spatial coarsening ×pool. RBN: temporal coarsening /stride)',
                 y=1.00, fontsize=12, weight='bold')
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'multiscale_diagnostic_extended.png')
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"Saved: {out}")

    out_json = os.path.join(os.path.dirname(__file__), 'multiscale_diagnostic_extended.json')
    with open(out_json, 'w') as f:
        json.dump({k: [dict(r) for r in v] for k, v in all_results.items()}, f, indent=2, default=float)
    print(f"Saved: {out_json}")
