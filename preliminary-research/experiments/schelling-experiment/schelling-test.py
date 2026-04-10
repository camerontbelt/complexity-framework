"""
schelling_3d_analysis.py
=========================
Full 3D analysis of Schelling segregation model complexity.

Produces:
  1. C(threshold, time) heatmap -- full 2D matrix
  2. 3D surface plot of C across threshold x time
  3. Per-threshold C trajectories
  4. Peak C location analysis
  5. CSV of full matrix for further analysis

Run locally (no timeout):
  python schelling_3d_analysis.py

Output files:
  schelling_3d_heatmap.png    -- 2D heatmap (threshold x time)
  schelling_3d_surface.png    -- 3D surface plot
  schelling_3d_trajectories.png -- C over time per threshold
  schelling_3d_results.csv    -- full matrix data
  schelling_3d_summary.csv    -- peak C location per threshold

Takes approximately 15-20 minutes on a modern laptop.
"""

import sys, os, zlib, csv, time, argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import convolve
from collections import defaultdict

# -- Import framework v8 (metric functions only) -------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import complexity_framework_v8 as fw

# -- Schelling cell states -----------------------------------------------------
_EMPTY, _A, _B = 0, 1, 2

SCHELLING_CFG = dict(
    GRID         = 64,
    STEPS        = 100,
    DENSITY      = 0.80,
    TRANSITION_LOW  = 0.30,
    TRANSITION_HIGH = 0.50,
)

def _schelling_run(threshold, cfg, seed=42, intervention=None):
    """
    Schelling segregation model on 2D toroidal grid.
    Returns (hist_A, hist_B) -- binary histories for each group.
    """
    rng = np.random.default_rng(seed)
    G   = cfg['GRID']
    N   = G * G

    n_agents = int(N * cfg['DENSITY'])
    n_A      = n_agents // 2
    n_B      = n_agents - n_A

    flat = np.zeros(N, dtype=np.int8)
    occupied = rng.choice(N, size=n_agents, replace=False)
    flat[occupied[:n_A]] = _A
    flat[occupied[n_A:]] = _B
    grid = flat.reshape(G, G)
    local_thresholds = None
    intervention = intervention or {}
    hetero_sigma = intervention.get('hetero_threshold_sigma', 0.0)
    if hetero_sigma > 0:
        local_thresholds = np.full((G, G), np.nan, dtype=np.float32)
        occupied_mask = (grid != _EMPTY)
        draws = rng.normal(threshold, hetero_sigma, size=occupied_mask.sum())
        local_thresholds[occupied_mask] = np.clip(draws, 0.0, 1.0)

    hist_A, hist_B = [], []
    MOORE = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int32)

    for t in range(cfg['STEPS']):
        hist_A.append((grid == _A).astype(np.int8).ravel())
        hist_B.append((grid == _B).astype(np.int8).ravel())

        a_nbrs = convolve((grid==_A).astype(np.int32), MOORE, mode='wrap')
        b_nbrs = convolve((grid==_B).astype(np.int32), MOORE, mode='wrap')
        total  = a_nbrs + b_nbrs

        same = np.where(grid==_A, a_nbrs, np.where(grid==_B, b_nbrs, 0))
        with np.errstate(divide='ignore', invalid='ignore'):
            similarity = np.where(total > 0,
                                  same.astype(np.float32) / total, 1.0)

        threshold_field = threshold if local_thresholds is None else local_thresholds
        unsatisfied = (grid != _EMPTY) & (similarity < threshold_field)
        p_random_move = intervention.get('p_random_move', 0.0)
        if p_random_move > 0:
            random_movers = (grid != _EMPTY) & (~unsatisfied) & (
                rng.random((G, G)) < p_random_move
            )
            unsatisfied = unsatisfied | random_movers
        empty_mask  = (grid == _EMPTY)
        unsat_cells = np.argwhere(unsatisfied)
        rng.shuffle(unsat_cells)

        new_grid = grid.copy()
        is_empty = empty_mask.copy()
        new_thresholds = None if local_thresholds is None else local_thresholds.copy()

        for (i, j) in unsat_cells:
            empty_now = np.argwhere(is_empty)
            if len(empty_now) == 0:
                break
            idx  = rng.integers(len(empty_now))
            ni, nj = empty_now[idx]
            new_grid[ni, nj] = new_grid[i, j]
            new_grid[i, j]   = _EMPTY
            if new_thresholds is not None:
                new_thresholds[ni, nj] = new_thresholds[i, j]
                new_thresholds[i, j] = np.nan
            is_empty[ni, nj] = False
            is_empty[i, j]   = True

        p_vacancy = intervention.get('p_vacancy_churn', 0.0)
        if p_vacancy > 0:
            vacate_mask = (new_grid != _EMPTY) & (rng.random((G, G)) < p_vacancy)
            vacated = np.argwhere(vacate_mask)
            for i, j in vacated:
                new_grid[i, j] = _EMPTY
                if new_thresholds is not None:
                    new_thresholds[i, j] = np.nan
            fill_mask = (new_grid == _EMPTY) & (rng.random((G, G)) < p_vacancy)
            fills = np.argwhere(fill_mask)
            for i, j in fills:
                new_grid[i, j] = _A if rng.random() < 0.5 else _B
                if new_thresholds is not None:
                    th_new = rng.normal(threshold, hetero_sigma)
                    new_thresholds[i, j] = np.clip(th_new, 0.0, 1.0)

        grid = new_grid
        if new_thresholds is not None:
            local_thresholds = new_thresholds

    return np.array(hist_A, dtype=np.int8), np.array(hist_B, dtype=np.int8)


def _schelling_metrics(history, window):
    """Apply full framework composite to a binary history chunk."""
    mH, sH       = fw._entropy_stats(history, 0, window)
    op_up, op_dn = fw._opacity_both(history, 0, window)
    mi1, decay   = fw._opacity_temporal(history, 0, window)
    tc           = fw._tcomp(history, 0, window)
    raw          = history.tobytes()
    gz           = len(zlib.compress(raw, 6)) / max(len(raw), 1)
    C = fw.composite(mH, sH, op_up, op_dn, mi1, decay, tc, gz)
    return dict(
        C=C,
        w_H    = fw.weight_H(mH, sH),
        w_OPs  = fw.weight_opacity_spatial(op_up, op_dn),
        w_OPt  = fw.weight_opacity_temporal(mi1, decay),
        w_T    = fw.weight_tcomp(tc),
        w_G    = fw.weight_gzip(gz),
        mean_H = mH, gzip = gz,
    )

# -- Configuration -------------------------------------------------------------
CFG = dict(
    GRID        = 64,
    STEPS       = 100,
    BURNIN      = 0,
    WINDOW      = 10,
    STRIDE      = 1,
    N_SEEDS     = 5,
    DENSITY     = 0.80,
    THRESH_MIN  = 0.00,
    THRESH_MAX  = 1.00,
    THRESH_STEP = 0.01,
    TRANSITION_LOW  = 0.30,
    TRANSITION_HIGH = 0.50,
)

SCHELLING_CFG.update(dict(
    STEPS   = CFG['STEPS'],
    DENSITY = CFG['DENSITY'],
    GRID    = CFG['GRID'],
))

TL = CFG['TRANSITION_LOW']
TH = CFG['TRANSITION_HIGH']

EXPERIMENTS = [
    dict(name='baseline', label='Baseline', intervention={}),
    dict(
        name='stochastic',
        label='Option1 Stochastic Moves',
        intervention={'p_random_move': 0.2},
    ),
    dict(
        name='heterogeneous_thresholds',
        label='Option2 Heterogeneous Thresholds',
        intervention={'hetero_threshold_sigma': 0.08},
    ),
    dict(
        name='vacancy_dynamics',
        label='Option3 Vacancy Dynamics',
        intervention={'p_vacancy_churn': 0.01},
    ),
]


def _lag1_autocorr(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    x0 = x[:-1] - x[:-1].mean()
    x1 = x[1:] - x[1:].mean()
    denom = np.sqrt(np.sum(x0 * x0) * np.sum(x1 * x1))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x0 * x1) / denom)


# -- Sliding window C computation ----------------------------------------------

def C_over_time(hist, window, stride):
    """
    Compute composite C using a sliding window across full simulation history.

    hist:   (T, G*G) binary array
    window: number of steps per measurement
    stride: steps between windows

    Returns (time_centers, C_values) arrays.
    """
    T      = len(hist)
    times  = []
    scores = []
    t      = 0

    while t + window <= T:
        chunk        = hist[t:t + window]
        mH, sH       = fw._entropy_stats(chunk, 0, window)
        op_up, op_dn = fw._opacity_both(chunk, 0, window)
        mi1, decay   = fw._opacity_temporal(chunk, 0, window)
        tc           = fw._tcomp(chunk, 0, window)
        raw          = chunk.tobytes()
        gz           = len(zlib.compress(raw, 6)) / max(len(raw), 1)
        C            = fw.composite(mH, sH, op_up, op_dn,
                                    mi1, decay, tc, gz)

        # Also store individual weights for diagnostics
        times.append(t + window // 2)
        scores.append(dict(
            C=C,
            w_H    = fw.weight_H(mH, sH),
            w_OPs  = fw.weight_opacity_spatial(op_up, op_dn),
            w_OPt  = fw.weight_opacity_temporal(mi1, decay),
            w_T    = fw.weight_tcomp(tc),
            w_G    = fw.weight_gzip(gz),
            mean_H = mH,
            gzip   = gz,
        ))
        t += stride

    return np.array(times), scores


def _threshold_job(thresh, n_times, window, experiment):
    seeds_A = np.zeros((CFG['N_SEEDS'], n_times))
    seeds_B = np.zeros((CFG['N_SEEDS'], n_times))
    w_acc = {k: np.zeros((CFG['N_SEEDS'], n_times))
             for k in ['w_H', 'w_OPs', 'w_OPt', 'w_T', 'w_G']}

    for s in range(CFG['N_SEEDS']):
        hA, hB = _schelling_run(
            thresh, SCHELLING_CFG, seed=s, intervention=experiment['intervention']
        )
        _, mA = C_over_time(hA, window, CFG['STRIDE'])
        _, mB = C_over_time(hB, window, CFG['STRIDE'])
        seeds_A[s] = [m['C'] for m in mA]
        seeds_B[s] = [m['C'] for m in mB]
        for k in w_acc:
            w_acc[k][s] = [m[k] for m in mA]

    return dict(
        thresh=float(thresh),
        C_A=seeds_A.mean(axis=0),
        C_B=seeds_B.mean(axis=0),
        w_H=w_acc['w_H'].mean(axis=0),
        w_OPs=w_acc['w_OPs'].mean(axis=0),
        w_OPt=w_acc['w_OPt'].mean(axis=0),
        w_T=w_acc['w_T'].mean(axis=0),
        w_G=w_acc['w_G'].mean(axis=0),
    )


# -- Main sweep ----------------------------------------------------------------

def run_full_sweep(experiment, workers=1):
    window = min(CFG['WINDOW'], CFG['STEPS'])
    if window != CFG['WINDOW']:
        print(
            f"Adjusted WINDOW from {CFG['WINDOW']} to {window} "
            f"(cannot exceed STEPS={CFG['STEPS']})."
        )

    thresholds = np.round(
        np.arange(CFG['THRESH_MIN'], CFG['THRESH_MAX'] + 1e-9, CFG['THRESH_STEP']), 3)

    # Get time axis from one dummy run
    dummy_cfg = SCHELLING_CFG.copy()
    dummy_cfg['N_SEEDS'] = 1
    hA_d, _ = _schelling_run(
        0.35, dummy_cfg, seed=0, intervention=experiment['intervention']
    )
    times_ref, _ = C_over_time(hA_d, window, CFG['STRIDE'])
    n_times = len(times_ref)
    if n_times == 0:
        raise ValueError(
            "No valid time windows produced. Ensure STEPS > 0 and STRIDE > 0."
        )

    print(f"{'='*60}")
    print(f"Schelling 3D Analysis -- {experiment['label']}")
    print(f"  Thresholds: {thresholds[0]:.2f} -> {thresholds[-1]:.2f}  "
          f"({len(thresholds)} values)")
    print(f"  Steps: {CFG['STEPS']}  Window: {CFG['WINDOW']}  "
          f"Stride: {CFG['STRIDE']}")
    if window != CFG['WINDOW']:
        print(f"  Effective Window: {window}")
    print(f"  Seeds: {CFG['N_SEEDS']}  Time points: {n_times}")
    print(f"  Total runs: {len(thresholds) * CFG['N_SEEDS'] * 2}")
    print(f"{'='*60}\n")

    # Storage: shape (n_thresh, n_times)
    C_mat_A   = np.zeros((len(thresholds), n_times))
    C_mat_B   = np.zeros((len(thresholds), n_times))
    # Weight matrices for group A
    wH_mat    = np.zeros((len(thresholds), n_times))
    wOPs_mat  = np.zeros((len(thresholds), n_times))
    wOPt_mat  = np.zeros((len(thresholds), n_times))
    wT_mat    = np.zeros((len(thresholds), n_times))
    wG_mat    = np.zeros((len(thresholds), n_times))

    csv_rows   = []
    t_start    = time.time()

    if workers > 1:
        print(f"  Parallel workers: {workers}")
        jobs = {}
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, thresh in enumerate(thresholds):
                fut = ex.submit(_threshold_job, float(thresh), n_times, window, experiment)
                jobs[fut] = i

            done_count = 0
            for fut in as_completed(jobs):
                i = jobs[fut]
                res = fut.result()
                thresh = res['thresh']
                C_mat_A[i] = res['C_A']
                C_mat_B[i] = res['C_B']
                wH_mat[i] = res['w_H']
                wOPs_mat[i] = res['w_OPs']
                wOPt_mat[i] = res['w_OPt']
                wT_mat[i] = res['w_T']
                wG_mat[i] = res['w_G']

                done_count += 1
                elapsed = time.time() - t_start
                eta = elapsed / done_count * (len(thresholds) - done_count)
                peak_C = C_mat_A[i].max()
                peak_t = times_ref[C_mat_A[i].argmax()]
                regime = ('mixed' if thresh < TL else
                          'transition' if thresh <= TH else
                          'segregated')
                print(f"  [{done_count:2d}/{len(thresholds)}] thresh={thresh:.2f}  "
                      f"[{regime:10s}]  peak_C={peak_C:.4f} at t={peak_t:4d}  "
                      f"ETA={eta/60:.1f}min")
    else:
        for i, thresh in enumerate(thresholds):
            res = _threshold_job(float(thresh), n_times, window, experiment)
            C_mat_A[i] = res['C_A']
            C_mat_B[i] = res['C_B']
            wH_mat[i] = res['w_H']
            wOPs_mat[i] = res['w_OPs']
            wOPt_mat[i] = res['w_OPt']
            wT_mat[i] = res['w_T']
            wG_mat[i] = res['w_G']

            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (len(thresholds) - i - 1)
            peak_C = C_mat_A[i].max()
            peak_t = times_ref[C_mat_A[i].argmax()]
            regime = ('mixed' if thresh < TL else
                      'transition' if thresh <= TH else
                      'segregated')
            print(f"  [{i+1:2d}/{len(thresholds)}] thresh={thresh:.2f}  "
                  f"[{regime:10s}]  peak_C={peak_C:.4f} at t={peak_t:4d}  "
                  f"ETA={eta/60:.1f}min")

    for i, thresh in enumerate(thresholds):
        regime = ('mixed' if thresh < TL else
                  'transition' if thresh <= TH else
                  'segregated')
        for ti, t_val in enumerate(times_ref):
            csv_rows.append(dict(
                experiment=experiment['name'],
                threshold=float(thresh), time_step=int(t_val), regime=regime,
                C_A=float(C_mat_A[i, ti]),
                C_B=float(C_mat_B[i, ti]),
                w_H=float(wH_mat[i, ti]),
                w_OPs=float(wOPs_mat[i, ti]),
                w_OPt=float(wOPt_mat[i, ti]),
                w_T=float(wT_mat[i, ti]),
                w_G=float(wG_mat[i, ti]),
            ))

    return thresholds, times_ref, C_mat_A, C_mat_B, \
           wH_mat, wOPs_mat, wOPt_mat, wT_mat, wG_mat, csv_rows


# -- Save CSV ------------------------------------------------------------------

def save_csv(csv_rows, path):
    keys = ['experiment','threshold','time_step','regime',
            'C_A','C_B','w_H','w_OPs','w_OPt','w_T','w_G']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(csv_rows)
    print(f"  Full CSV -> {path}")


def save_summary_csv(thresholds, times_ref, C_mat_A, C_mat_B, path, experiment_name):
    rows = []
    TL_l = CFG['TRANSITION_LOW']
    TH_l = CFG['TRANSITION_HIGH']
    for i, thresh in enumerate(thresholds):
        peak_C_A = float(C_mat_A[i].max())
        peak_t_A = int(times_ref[C_mat_A[i].argmax()])
        peak_C_B = float(C_mat_B[i].max())
        peak_t_B = int(times_ref[C_mat_B[i].argmax()])
        final_C  = float(C_mat_A[i, -1])
        mean_C_A = float(C_mat_A[i].mean())
        mean_C_B = float(C_mat_B[i].mean())
        var_C_A = float(C_mat_A[i].var())
        var_C_B = float(C_mat_B[i].var())
        ac1_C_A = _lag1_autocorr(C_mat_A[i])
        ac1_C_B = _lag1_autocorr(C_mat_B[i])
        regime   = ('mixed'      if thresh < TL_l else
                    'transition' if thresh <= TH_l else
                    'segregated')
        rows.append(dict(
            experiment=experiment_name,
            threshold=float(thresh), regime=regime,
            peak_C_A=peak_C_A, peak_t_A=peak_t_A,
            peak_C_B=peak_C_B, peak_t_B=peak_t_B,
            final_C_A=final_C,
            mean_C_A=mean_C_A, mean_C_B=mean_C_B,
            var_C_A=var_C_A, var_C_B=var_C_B,
            lag1_ac_C_A=ac1_C_A, lag1_ac_C_B=ac1_C_B,
            mapping_agree=abs(peak_C_A-peak_C_B)<0.05,
        ))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  Summary CSV -> {path}")


# -- Plots ---------------------------------------------------------------------

PANEL_BG = '#1a1a2a'
AXIS_COL = '#ccccdd'
GRID_COL = '#333344'

def style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=AXIS_COL)
    ax.xaxis.label.set_color(AXIS_COL)
    ax.yaxis.label.set_color(AXIS_COL)
    ax.title.set_color(AXIS_COL)
    for sp in ax.spines.values():
        sp.set_color(GRID_COL)
    ax.grid(True, color=GRID_COL, lw=0.5, alpha=0.6)


def plot_heatmap(thresholds, times_ref, C_mat_A, C_mat_B, path):
    """2D heatmap: C(threshold, time) for both group mappings."""
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#0e0e16')
    fig.suptitle(
        'Schelling Segregation -- C(threshold, time) Heatmap\n'
        'Full sliding window analysis  |  Bright = high complexity',
        fontsize=13, fontweight='bold', color='white', y=0.98
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3,
                           left=0.07, right=0.97, top=0.90, bottom=0.10)

    vmax = min(max(C_mat_A.max(), C_mat_B.max()), 1.5)
    if len(times_ref) == 1:
        x_pad = max(1.0, CFG['STRIDE'] * 0.5)
        extent = [times_ref[0] - x_pad, times_ref[0] + x_pad,
                  thresholds[0] - 0.025, thresholds[-1] + 0.025]
    else:
        extent = [times_ref[0], times_ref[-1],
                  thresholds[0] - 0.025, thresholds[-1] + 0.025]

    for ax_idx, (mat, label) in enumerate([(C_mat_A, 'Group A'),
                                            (C_mat_B, 'Group B')]):
        ax = fig.add_subplot(gs[ax_idx])
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=AXIS_COL)
        ax.xaxis.label.set_color(AXIS_COL)
        ax.yaxis.label.set_color(AXIS_COL)
        ax.title.set_color(AXIS_COL)
        for sp in ax.spines.values(): sp.set_color(GRID_COL)

        im = ax.imshow(mat, aspect='auto', origin='lower',
                       cmap='inferno', vmin=0, vmax=vmax, extent=extent)
        ax.axhline(TL, color='#00ff88', lw=2, ls='--',
                   label=f'Transition band ({TL}-{TH})')
        ax.axhline(TH, color='#00ff88', lw=2, ls='--')
        ax.set_xlabel('Simulation Step', fontsize=11)
        ax.set_ylabel('Similarity Threshold', fontsize=11)
        ax.set_title(f'C(threshold, time) -- {label}', fontsize=11,
                     fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label='Composite C')
        ax.legend(fontsize=8, facecolor='#1a1a2a', labelcolor=AXIS_COL,
                  edgecolor=GRID_COL)

    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0e0e16')
    plt.close()
    print(f"  Heatmap -> {path}")


def plot_surface(thresholds, times_ref, C_mat_A, path):
    """3D surface plot of C(threshold, time)."""
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#0e0e16')

    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#0e0e16')
    ax.tick_params(colors=AXIS_COL)
    ax.xaxis.label.set_color(AXIS_COL)
    ax.yaxis.label.set_color(AXIS_COL)
    ax.zaxis.label.set_color(AXIS_COL)
    ax.title.set_color('white')

    times_surface = times_ref
    C_surface = C_mat_A
    if len(times_ref) == 1:
        # Duplicate a nearby x-column so surface rendering remains visible.
        x_pad = max(1.0, CFG['STRIDE'] * 0.5)
        times_surface = np.array([times_ref[0] - x_pad, times_ref[0] + x_pad],
                                 dtype=float)
        C_surface = np.repeat(C_mat_A, 2, axis=1)

    T_grid, Thresh_grid = np.meshgrid(times_surface, thresholds)
    surf = ax.plot_surface(T_grid, Thresh_grid, C_surface,
                           cmap='inferno', alpha=0.9,
                           linewidth=0, antialiased=True)

    # Mark transition band walls
    for thresh_line in [TL, TH]:
        idx = np.argmin(np.abs(thresholds - thresh_line))
        ax.plot(times_surface,
                np.full_like(times_surface, thresholds[idx], dtype=float),
                C_surface[idx],
                color='#00ff88', lw=2, zorder=5)

    if len(times_ref) == 1:
        ax.set_xlim(times_surface[0], times_surface[-1])

    ax.set_xlabel('Simulation Step', fontsize=10, labelpad=10)
    ax.set_ylabel('Similarity Threshold', fontsize=10, labelpad=10)
    ax.set_zlabel('Composite C', fontsize=10, labelpad=10)
    ax.set_title(
        'Schelling Segregation -- 3D Complexity Surface\n'
        'C(threshold, time)  |  Group A mapping',
        fontsize=12, fontweight='bold', pad=15
    )
    fig.colorbar(surf, ax=ax, fraction=0.03, pad=0.1, label='C')

    # Best viewing angle
    ax.view_init(elev=30, azim=225)

    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0e0e16')
    plt.close()
    print(f"  3D Surface -> {path}")


def plot_trajectories(thresholds, times_ref, C_mat_A,
                      wH_mat, wOPs_mat, wOPt_mat, wT_mat, wG_mat, path):
    """C(t) trajectories for each threshold + weight breakdown for key thresholds."""
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0e0e16')
    fig.suptitle(
        'Schelling -- C(t) Trajectories and Weight Breakdown\n'
        'Each line = one similarity threshold value',
        fontsize=13, fontweight='bold', color='white', y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38,
                           left=0.07, right=0.97, top=0.92, bottom=0.07)

    # -- Panel 1: All trajectories coloured by regime --------------------------
    ax1 = fig.add_subplot(gs[0, 0:2])
    style_ax(ax1)
    single_time = len(times_ref) == 1
    cmap_mixed = plt.cm.Blues
    cmap_trans = plt.cm.YlOrBr
    cmap_seg   = plt.cm.Reds

    for i, thresh in enumerate(thresholds):
        if thresh < TL:
            col = cmap_mixed(0.4 + 0.5 * (thresh - thresholds[0]) /
                             max(TL - thresholds[0], 0.01))
            lbl = f't={thresh:.2f} (mixed)' if i == 0 else None
        elif thresh <= TH:
            col = cmap_trans(0.4 + 0.6 * (thresh - TL) / max(TH - TL, 0.01))
            lbl = f't={thresh:.2f} (transition)' if thresh == TL else None
        else:
            col = cmap_seg(0.4 + 0.5 * (thresh - TH) /
                           max(thresholds[-1] - TH, 0.01))
            lbl = f't={thresh:.2f} (segregated)' if thresh == thresholds[
                np.searchsorted(thresholds, TH) + 1] else None
        ax1.plot(
            times_ref,
            C_mat_A[i],
            color=col,
            lw=1.5,
            alpha=0.85,
            marker='o' if single_time else None
        )

    if single_time:
        x_pad = max(1.0, CFG['STRIDE'] * 0.5)
        ax1.set_xlim(times_ref[0] - x_pad, times_ref[0] + x_pad)

    # Add regime legend patches
    from matplotlib.patches import Patch
    legend_els = [
        Patch(facecolor='#3498db', label='Mixed (t < 0.30)'),
        Patch(facecolor='#f39c12', label='Transition (0.30-0.50)'),
        Patch(facecolor='#e74c3c', label='Segregated (t > 0.50)'),
    ]
    ax1.legend(handles=legend_els, fontsize=8, facecolor='#1a1a2a',
               labelcolor=AXIS_COL, edgecolor=GRID_COL)
    ax1.set_xlabel('Simulation Step', fontsize=10)
    ax1.set_ylabel('Composite C', fontsize=10)
    ax1.set_title('C(t) for All Thresholds -- Coloured by Regime', fontsize=10,
                  fontweight='bold')

    # -- Panel 2: Peak C vs threshold (dynamic) --------------------------------
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2)
    peak_C_vals = [C_mat_A[i].max() for i in range(len(thresholds))]
    peak_t_vals = [times_ref[C_mat_A[i].argmax()] for i in range(len(thresholds))]
    cols = ['#3498db' if t < TL else '#f39c12' if t <= TH else '#e74c3c'
            for t in thresholds]
    ax2.bar(thresholds, peak_C_vals, width=0.04,
            color=cols, alpha=0.85, edgecolor='k', lw=0.5)
    ax2.axvspan(TL, TH, alpha=0.15, color='#f39c12')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Peak C')
    ax2.set_title('Peak C per Threshold\n(any time point)', fontsize=10)

    # -- Weight breakdown panels for 4 key thresholds -------------------------
    key_thresholds = [0.25, 0.40, 0.60, 0.75]
    weight_names   = ['w_H', 'w_OPs', 'w_OPt', 'w_T', 'w_G']
    weight_mats    = [wH_mat, wOPs_mat, wOPt_mat, wT_mat, wG_mat]
    weight_colors  = ['#e74c3c', '#e67e22', '#9b59b6', '#f39c12', '#1abc9c']

    for pi, kt in enumerate(key_thresholds):
        row = 1 + pi // 3
        col = pi % 3
        ax = fig.add_subplot(gs[row, col])
        style_ax(ax)
        idx = np.argmin(np.abs(thresholds - kt))
        for wm, wn, wc in zip(weight_mats, weight_names, weight_colors):
            ax.plot(
                times_ref,
                wm[idx],
                color=wc,
                lw=1.5,
                label=wn,
                marker='o' if single_time else None
            )
        ax2b = ax.twinx()
        ax2b.plot(
            times_ref,
            C_mat_A[idx],
            color='white',
            lw=2.5,
            ls='--',
            alpha=0.8,
            label='C',
            marker='o' if single_time else None
        )
        ax2b.set_ylabel('C', color=AXIS_COL)
        ax2b.tick_params(colors=AXIS_COL)
        regime = ('mixed' if kt < TL else 'transition' if kt <= TH
                  else 'segregated')
        ax.set_xlabel('Step')
        ax.set_ylabel('Weight')
        ax.set_title(f't={kt:.2f} ({regime})', fontsize=10, fontweight='bold')
        ax.legend(fontsize=6, facecolor='#1a1a2a', labelcolor=AXIS_COL,
                  edgecolor=GRID_COL, ncol=3, loc='upper right')
        if single_time:
            x_pad = max(1.0, CFG['STRIDE'] * 0.5)
            ax.set_xlim(times_ref[0] - x_pad, times_ref[0] + x_pad)
            ax2b.set_xlim(times_ref[0] - x_pad, times_ref[0] + x_pad)
        ax2b.tick_params(colors=AXIS_COL)
        for sp in ax2b.spines.values(): sp.set_color(GRID_COL)

    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0e0e16')
    plt.close()
    print(f"  Trajectories -> {path}")


# -- Entry point ---------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Schelling 3D sweeps for one or more interventions.'
    )
    parser.add_argument(
        '--experiment',
        default='all',
        choices=['all'] + [e['name'] for e in EXPERIMENTS],
        help='Experiment to run (default: all).',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Parallel workers for threshold sweeps (default: 1).',
    )
    args = parser.parse_args()

    selected_experiments = EXPERIMENTS
    if args.experiment != 'all':
        selected_experiments = [e for e in EXPERIMENTS if e['name'] == args.experiment]
    workers = max(1, args.workers)

    t_global = time.time()
    for exp in selected_experiments:
        print(f"\n{'#' * 74}")
        print(f"Running experiment: {exp['label']} ({exp['name']})")
        print(f"{'#' * 74}")

        (thresholds, times_ref, C_mat_A, C_mat_B,
         wH_mat, wOPs_mat, wOPt_mat, wT_mat, wG_mat,
         csv_rows) = run_full_sweep(exp, workers=workers)

        print(f"\nSweep complete in {(time.time()-t_global)/60:.1f} min")
        peak_idx = np.unravel_index(C_mat_A.argmax(), C_mat_A.shape)
        print(f"\nGlobal peak: C={C_mat_A.max():.4f} at "
              f"threshold={thresholds[peak_idx[0]]:.2f} "
              f"time={times_ref[peak_idx[1]]}")

        out_prefix = f"schelling_3d_{exp['name']}"

        print("\nSaving...")
        save_csv(csv_rows, f'{out_prefix}_results.csv')
        save_summary_csv(
            thresholds,
            times_ref,
            C_mat_A,
            C_mat_B,
            f'{out_prefix}_summary.csv',
            exp['name'],
        )

        print("\nGenerating plots...")
        plot_heatmap(thresholds, times_ref, C_mat_A, C_mat_B,
                     f'{out_prefix}_heatmap.png')
        plot_surface(thresholds, times_ref, C_mat_A,
                     f'{out_prefix}_surface.png')
        plot_trajectories(thresholds, times_ref, C_mat_A,
                          wH_mat, wOPs_mat, wOPt_mat, wT_mat, wG_mat,
                          f'{out_prefix}_trajectories.png')

    print(f"\nAll done. Total time: {(time.time()-t_global)/60:.1f} min")
    print("\nOutput files:")
    print("  schelling_3d_<experiment>_heatmap.png")
    print("  schelling_3d_<experiment>_surface.png")
    print("  schelling_3d_<experiment>_trajectories.png")
    print("  schelling_3d_<experiment>_results.csv")
    print("  schelling_3d_<experiment>_summary.csv")
    print("\nExperiments: baseline, stochastic, heterogeneous_thresholds, vacancy_dynamics")
    print("\nTo send results back: share CSVs and PNGs for the experiment(s) you care about.")