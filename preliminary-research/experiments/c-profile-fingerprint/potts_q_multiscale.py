"""
Potts Model — multi-scale C across q values.

Tests whether the RG (Renormalization Group) beta function
  beta(s -> 2s) = C(2s) - C(s)
discriminates between second-order (q <= 4) and first-order (q >= 5) Potts
transitions.

Predictions (recorded before running):
  q = 2  (second-order, Ising universality in Potts convention):
         beta > 0 across scales at T_c (scale-invariant)
  q = 3  (weakly second-order):
         beta > 0 across scales at T_c
  q = 5  (first-order):
         beta peaks at some finite scale then declines → characteristic size
  q = 10 (strongly first-order):
         beta peaks at a small/medium scale — nucleation domains

T_c(q) = 1 / ln(1 + sqrt(q)).
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from complexity_framework_v9 import compute_C
from fingerprint_test import potts_run
from multiscale_diagnostic import coarsen_history

# ── experiment setup ──────────────────────────────────────────────────────────
Q_VALUES  = [2, 3, 5, 10]
POOLS     = (1, 2, 4, 8)
G         = 48
SWEEPS    = 4000
BURNIN_M  = 1500             # in Metropolis sweeps
WINDOW    = 100              # snapshots
SNAP_EVERY = 25
N_SEEDS   = 3

# C-compute burnin/window operate on the snapshot history, not the raw sweeps
C_BURNIN = 10
C_WINDOW = 80

def Tc_of_q(q):
    return 1.0 / np.log(1.0 + np.sqrt(q))

def sweep_q(q, T_values):
    rows = []
    Tc = Tc_of_q(q)
    print(f"\n{'='*60}\nPotts q={q}, T_c = {Tc:.4f}\n{'='*60}")
    for T in T_values:
        per  = {pool: [] for pool in POOLS}
        wopt = {pool: [] for pool in POOLS}
        for seed in range(N_SEEDS):
            hist = potts_run(T, q=q, G=G,
                             sweeps=SWEEPS, burnin=BURNIN_M,
                             window=WINDOW, snap_every=SNAP_EVERY,
                             seed=seed)
            T_len = hist.shape[0]
            hist_3d = hist.reshape(T_len, G, G)
            for pool in POOLS:
                grid = coarsen_history(hist_3d, pool)
                if grid.shape[1] < 4:
                    continue
                r = compute_C(grid, C_BURNIN, C_WINDOW)
                per[pool].append(r['score'])
                wopt[pool].append(r['w_OP_t'])
        row = {'T': T, 'q': q, 'T_c': Tc}
        for pool in POOLS:
            if per[pool]:
                row[f'C_{pool}']    = float(np.mean(per[pool]))
                row[f'wOPt_{pool}'] = float(np.mean(wopt[pool]))
        rows.append(row)
        print(f"  T={T:.3f}: " + "  ".join(
            f"C×{p}={row.get(f'C_{p}', 0):.3f}" for p in POOLS))
    return rows


def rg_beta(row):
    """Return list of beta(s -> 2s) = C(2s) - C(s) across pool factors."""
    Cs = [row.get(f'C_{p}') for p in POOLS]
    if None in Cs:
        return []
    return [Cs[i+1] - Cs[i] for i in range(len(Cs)-1)]


if __name__ == '__main__':
    t0 = time.time()

    # T grid: centred on T_c(q), spanning 0.6 Tc to 1.6 Tc
    def T_grid(q, n=9):
        Tc = Tc_of_q(q)
        return np.round(np.linspace(0.6*Tc, 1.6*Tc, n), 4).tolist()

    all_results = {}
    for q in Q_VALUES:
        all_results[q] = sweep_q(q, T_grid(q))

    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")

    # Save raw data
    out_json = os.path.join(os.path.dirname(__file__), 'potts_q_multiscale.json')
    with open(out_json, 'w') as f:
        json.dump({str(q): [dict(r) for r in v] for q, v in all_results.items()},
                  f, indent=2, default=float)
    print(f"Saved: {out_json}")

    # ── figure 1: C vs T at each scale, one panel per q ──────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for col, q in enumerate(Q_VALUES):
        rows = all_results[q]
        Ts = [r['T'] for r in rows]
        Tc = Tc_of_q(q)
        for pool in POOLS:
            Cs = [r.get(f'C_{pool}', 0) for r in rows]
            Ws = [r.get(f'wOPt_{pool}', 0) for r in rows]
            axes[0, col].plot(Ts, Cs, marker='o', label=f'×{pool}')
            axes[1, col].plot(Ts, Ws, marker='s', label=f'×{pool}')
        for ax in (axes[0, col], axes[1, col]):
            ax.axvline(Tc, color='red', ls='--', lw=0.8, alpha=0.6,
                       label=f'T_c={Tc:.3f}')
            ax.legend(fontsize=7); ax.grid(alpha=0.3); ax.set_xlabel('T')
        order_label = 'second-order' if q <= 4 else 'first-order'
        axes[0, col].set_title(f'Potts q={q} ({order_label}): C vs T')
        axes[1, col].set_title(f'Potts q={q}: w_OP_t vs T')
        axes[0, col].set_ylabel('C')
        axes[1, col].set_ylabel('w_OP_t')

    plt.suptitle('Potts model — C(T) profile at pool factors {1, 2, 4, 8} across q',
                 y=1.00, fontsize=12, weight='bold')
    plt.tight_layout()
    out1 = os.path.join(os.path.dirname(__file__), 'potts_q_multiscale.png')
    plt.savefig(out1, dpi=130, bbox_inches='tight')
    print(f"Saved: {out1}")

    # ── figure 2: C vs pool factor AT T_c, one curve per q — the key plot ────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    pool_axis = list(POOLS)
    for q in Q_VALUES:
        rows = all_results[q]
        Tc = Tc_of_q(q)
        # nearest row to Tc
        idx = int(np.argmin([abs(r['T'] - Tc) for r in rows]))
        r = rows[idx]
        Cs = [r.get(f'C_{p}', 0) for p in POOLS]
        axes2[0].plot(pool_axis, Cs, marker='o', label=f'q={q} (T={r["T"]:.3f})', lw=2)
        # RG beta
        beta = rg_beta(r)
        beta_axis = [f'{POOLS[i]}→{POOLS[i+1]}' for i in range(len(POOLS)-1)]
        axes2[1].plot(range(len(beta)), beta, marker='s', label=f'q={q}', lw=2)

    axes2[0].set_xscale('log', base=2)
    axes2[0].set_xticks(pool_axis); axes2[0].set_xticklabels([f'×{p}' for p in pool_axis])
    axes2[0].set_xlabel('spatial pool factor')
    axes2[0].set_ylabel('C at T ≈ T_c')
    axes2[0].set_title('C vs scale at the critical point\n(flat/growing = scale-invariant; peaked = characteristic scale)')
    axes2[0].grid(alpha=0.3); axes2[0].legend()

    axes2[1].axhline(0, color='gray', lw=0.8, ls='--')
    axes2[1].set_xticks(range(len(beta_axis))); axes2[1].set_xticklabels(beta_axis)
    axes2[1].set_xlabel('RG step (s → 2s)')
    axes2[1].set_ylabel('β(s→2s) = C(2s) − C(s)')
    axes2[1].set_title('RG β function at T_c\n(β>0 = scale-invariant; β<0 = structure lost at that scale)')
    axes2[1].grid(alpha=0.3); axes2[1].legend()

    plt.suptitle('Potts multi-scale: discriminating transition order via β function',
                 y=1.02, fontsize=12, weight='bold')
    plt.tight_layout()
    out2 = os.path.join(os.path.dirname(__file__), 'potts_q_beta.png')
    plt.savefig(out2, dpi=130, bbox_inches='tight')
    print(f"Saved: {out2}")

    # ── summary table ────────────────────────────────────────────────────────
    print(f"\n{'q':>4} {'T_c':>8} {'C(×1)':>8} {'C(×2)':>8} {'C(×4)':>8} {'C(×8)':>8}  {'β sign pattern':>20}")
    for q in Q_VALUES:
        rows = all_results[q]
        Tc = Tc_of_q(q)
        idx = int(np.argmin([abs(r['T'] - Tc) for r in rows]))
        r = rows[idx]
        Cs = [r.get(f'C_{p}', 0) for p in POOLS]
        beta = rg_beta(r)
        sign_pattern = ''.join('+' if b > 0 else '-' if b < 0 else '0' for b in beta)
        print(f"{q:>4} {Tc:>8.3f} " + ' '.join(f'{c:>8.3f}' for c in Cs) + f"   {sign_pattern:>20}")
