"""
Potts Model — multi-scale C across q values — v2 with improvements.

Changes from v1:
  1. MAJORITY-CLUSTER BINARIZATION: at each snapshot, cell = 1 if it matches
     the most common state at that moment, else 0. Keeps split near 50/50
     regardless of q. Replaces the broken "state 0 vs rest" encoding that
     made q>=5 degenerate.
  2. Longer simulations: 10k sweeps / 4k burnin / 150 snapshots, 5 seeds.
  3. Larger grid: 64×64 (was 48×48).

Hypothesis we're testing: RG beta function sign pattern at T_c
  q = 2, 3  (second-order): beta > 0 across scales (flat / growing profile)
  q = 5, 10 (first-order):  beta flips at some finite scale (peaked profile)
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from complexity_framework_v9 import compute_C
from multiscale_diagnostic import coarsen_history


def potts_run_majority(T, q=3, G=64, sweeps=10000, burnin=4000, window=150,
                       snap_every=25, seed=42):
    """Potts with Metropolis, but binarized as 'matches majority state' per snapshot.

    Ensures the binary field is roughly balanced at high q.
    """
    rng = np.random.default_rng(seed)
    spins = rng.integers(0, q, size=(G, G), dtype=np.int8)

    history = []
    snap_count = 0
    total = burnin + window * snap_every

    ii, jj = np.mgrid[0:G, 0:G]
    black = ((ii + jj) % 2 == 0)
    white = ~black

    for sweep in range(total):
        for mask in (black, white):
            proposed = rng.integers(0, q, size=(G, G), dtype=np.int8)
            match_cur = None
            match_new = None
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nb = np.roll(np.roll(spins, di, axis=0), dj, axis=1)
                mc = (spins == nb).astype(np.float32)
                mn = (proposed == nb).astype(np.float32)
                if match_cur is None:
                    match_cur = mc; match_new = mn
                else:
                    match_cur += mc; match_new += mn
            dE = -(match_new - match_cur)
            accept = mask & ((dE <= 0) |
                             (rng.random((G, G)) < np.exp(np.clip(-dE / max(T, 1e-10), -30, 0))))
            spins = np.where(accept, proposed, spins)

        if sweep >= burnin and (sweep - burnin) % snap_every == 0:
            if snap_count < window:
                # MAJORITY-CLUSTER BINARIZATION: which state is most populous
                # in this snapshot? cells matching → 1, others → 0.
                counts = np.bincount(spins.ravel(), minlength=q)
                majority_state = counts.argmax()
                binary = (spins == majority_state).astype(np.int8)
                history.append(binary.ravel())
                snap_count += 1

    return np.array(history, dtype=np.int8)


# ── experiment setup ──────────────────────────────────────────────────────────
Q_VALUES  = [2, 3, 5, 10]
POOLS     = (1, 2, 4, 8)
G         = 64
SWEEPS    = 10000
BURNIN_M  = 4000
WINDOW    = 150
SNAP_EVERY = 25
N_SEEDS   = 5

C_BURNIN = 15
C_WINDOW = 120


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
            hist = potts_run_majority(T, q=q, G=G,
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
    Cs = [row.get(f'C_{p}') for p in POOLS]
    if None in Cs:
        return []
    return [Cs[i+1] - Cs[i] for i in range(len(Cs)-1)]


if __name__ == '__main__':
    t0 = time.time()

    def T_grid(q, n=7):
        Tc = Tc_of_q(q)
        # Tighter window: 0.7 Tc → 1.4 Tc
        return np.round(np.linspace(0.7*Tc, 1.4*Tc, n), 4).tolist()

    all_results = {}
    for q in Q_VALUES:
        all_results[q] = sweep_q(q, T_grid(q))

    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")

    out_json = os.path.join(os.path.dirname(__file__), 'potts_q_multiscale_v2.json')
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

    plt.suptitle('Potts v2 (majority-cluster binarization) — C(T) profile at pool factors {1, 2, 4, 8} across q',
                 y=1.00, fontsize=12, weight='bold')
    plt.tight_layout()
    out1 = os.path.join(os.path.dirname(__file__), 'potts_q_multiscale_v2.png')
    plt.savefig(out1, dpi=130, bbox_inches='tight')
    print(f"Saved: {out1}")

    # ── figure 2: C vs pool factor at T_c + beta function ────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    pool_axis = list(POOLS)
    for q in Q_VALUES:
        rows = all_results[q]
        Tc = Tc_of_q(q)
        idx = int(np.argmin([abs(r['T'] - Tc) for r in rows]))
        r = rows[idx]
        Cs = [r.get(f'C_{p}', 0) for p in POOLS]
        axes2[0].plot(pool_axis, Cs, marker='o', label=f'q={q} (T={r["T"]:.3f})', lw=2)
        beta = rg_beta(r)
        axes2[1].plot(range(len(beta)), beta, marker='s', label=f'q={q}', lw=2)

    axes2[0].set_xscale('log', base=2)
    axes2[0].set_xticks(pool_axis)
    axes2[0].set_xticklabels([f'×{p}' for p in pool_axis])
    axes2[0].set_xlabel('spatial pool factor')
    axes2[0].set_ylabel('C at T ≈ T_c')
    axes2[0].set_title('C vs scale at the critical point\n(flat/growing = scale-invariant; peaked = characteristic scale)')
    axes2[0].grid(alpha=0.3); axes2[0].legend()

    beta_axis = [f'{POOLS[i]}→{POOLS[i+1]}' for i in range(len(POOLS)-1)]
    axes2[1].axhline(0, color='gray', lw=0.8, ls='--')
    axes2[1].set_xticks(range(len(beta_axis)))
    axes2[1].set_xticklabels(beta_axis)
    axes2[1].set_xlabel('RG step (s → 2s)')
    axes2[1].set_ylabel('β(s→2s) = C(2s) − C(s)')
    axes2[1].set_title('RG β function at T_c\n(β>0 = scale-invariant; β<0 = structure lost at that scale)')
    axes2[1].grid(alpha=0.3); axes2[1].legend()

    plt.suptitle('Potts v2 multi-scale: discriminating transition order via β function',
                 y=1.02, fontsize=12, weight='bold')
    plt.tight_layout()
    out2 = os.path.join(os.path.dirname(__file__), 'potts_q_beta_v2.png')
    plt.savefig(out2, dpi=130, bbox_inches='tight')
    print(f"Saved: {out2}")

    print(f"\n{'q':>4} {'T_c':>8} {'C(×1)':>8} {'C(×2)':>8} {'C(×4)':>8} {'C(×8)':>8}  {'β pattern':>14}")
    for q in Q_VALUES:
        rows = all_results[q]
        Tc = Tc_of_q(q)
        idx = int(np.argmin([abs(r['T'] - Tc) for r in rows]))
        r = rows[idx]
        Cs = [r.get(f'C_{p}', 0) for p in POOLS]
        beta = rg_beta(r)
        sign = ''.join('+' if b > 0.01 else '-' if b < -0.01 else '0' for b in beta)
        print(f"{q:>4} {Tc:>8.3f} " + ' '.join(f'{c:>8.3f}' for c in Cs) + f"   {sign:>14}")
