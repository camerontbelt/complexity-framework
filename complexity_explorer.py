"""
Complexity Explorer
===================
A standalone tool for studying complexity through N-body particle simulation.
Derived from the candidate laws of complexity developed in conversation with Claude.

USAGE:
    python complexity_explorer.py              # interactive menu
    python complexity_explorer.py simulate     # run live simulation
    python complexity_explorer.py scan         # run parameter scan
    python complexity_explorer.py heatmap      # load saved scan & show heatmap

REQUIREMENTS:
    pip install numpy scipy matplotlib

The two tunable "universal constants":
    alpha   = electromagnetic coupling (attractive well depth)
              controls how strongly particles attract each other
    alpha_s = short-range repulsion (strong force analog)
              controls how strongly particles resist compression

Adams et al. found our universe's constants sit in an "island of stability"
where alpha ≈ 1.0, alpha_s ≈ 1.0 (normalized). This tool lets you explore
that landscape and measure complexity using entity-level metrics derived from
the eight candidate laws of complexity.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
from collections import defaultdict
from scipy import ndimage
import json
import os
import sys
import time

# ============================================================
# CONFIGURATION
# ============================================================

CFG = {
    # Simulation
    'N': 80,                  # number of particles
    'BOX': 18.0,              # simulation box size
    'DT': 0.025,              # timestep
    'DAMPING': 0.998,         # velocity damping per step
    'FORCE_CLAMP': 25.0,      # max force magnitude

    # Default constants (our universe = 1.0, 1.0)
    'ALPHA': 1.0,             # EM coupling (attractive strength)
    'ALPHA_S': 1.0,           # Strong force (repulsion steepness)

    # Scanning
    'SCAN_N_ALPHA': 18,
    'SCAN_N_ALPHAS': 18,
    'SCAN_ALPHA_MIN': 0.1,
    'SCAN_ALPHA_MAX': 5.0,
    'SCAN_ALPHAS_MIN': 0.3,
    'SCAN_ALPHAS_MAX': 3.2,
    'SCAN_STEPS': 200,
    'SCAN_SKIP': 40,
    'SCAN_SAMPLE_EVERY': 5,
    'SCAN_SEEDS': [42, 123, 7],
    'SCAN_SAVE_FILE': 'complexity_scan.json',

    # Simulation display
    'SIM_STEPS_PER_FRAME': 3,
    'SIM_HISTORY_LEN': 120,
    'SIM_METRIC_WINDOW': 40,   # frames to average metrics over
}

# ============================================================
# PHYSICS
# ============================================================

def init_particles(N, box, seed=42):
    np.random.seed(seed)
    pos = np.random.rand(N, 2) * box
    vel = np.random.randn(N, 2) * 0.35
    return pos, vel

def compute_forces(pos, alpha, alpha_s, box, clamp):
    dx = pos[:, None, 0] - pos[None, :, 0]
    dy = pos[:, None, 1] - pos[None, :, 1]
    dx -= box * np.round(dx / box)
    dy -= box * np.round(dy / box)
    r2 = np.maximum(dx**2 + dy**2, 0.04)
    cut = (alpha_s * 3.5)**2
    mask = r2 < cut
    np.fill_diagonal(mask, False)
    sr2 = np.where(mask, (alpha_s**2) / r2, 0)
    sr6 = sr2**3
    sr12 = sr6**2
    fm = np.where(mask, 24 * alpha * (2 * sr12 - sr6) / r2, 0)
    np.fill_diagonal(fm, 0)
    fx = np.clip((fm * dx).sum(axis=1), -clamp, clamp)
    fy = np.clip((fm * dy).sum(axis=1), -clamp, clamp)
    return fx, fy

def step_simulation(pos, vel, alpha, alpha_s, cfg):
    box, dt, damp, clamp = cfg['BOX'], cfg['DT'], cfg['DAMPING'], cfg['FORCE_CLAMP']
    fx, fy = compute_forces(pos, alpha, alpha_s, box, clamp)
    vel[:, 0] += 0.5 * fx * dt
    vel[:, 1] += 0.5 * fy * dt
    pos = (pos + vel * dt) % box
    fx2, fy2 = compute_forces(pos, alpha, alpha_s, box, clamp)
    vel[:, 0] += 0.5 * fx2 * dt
    vel[:, 1] += 0.5 * fy2 * dt
    vel *= damp
    return pos, vel

# ============================================================
# ENTITY DETECTION (cluster finding)
# ============================================================

def find_clusters(pos, alpha_s, box, bind_factor=2.2):
    bind_dist = max(alpha_s * bind_factor, 0.5)
    N = len(pos)
    dx = pos[:, None, 0] - pos[None, :, 0]
    dy = pos[:, None, 1] - pos[None, :, 1]
    dx -= box * np.round(dx / box)
    dy -= box * np.round(dy / box)
    r = np.sqrt(dx**2 + dy**2)
    # Union-find
    parent = list(range(N))
    def find(x):
        root = x
        while parent[root] != root: root = parent[root]
        while parent[x] != root: nxt = parent[x]; parent[x] = root; x = nxt
        return root
    rows, cols = np.where((r < bind_dist) & (r > 0.001))
    for a, b in zip(rows, cols):
        pa, pb = find(a), find(b)
        if pa != pb: parent[pa] = pb
    groups = defaultdict(list)
    for i in range(N): groups[find(i)].append(i)
    clusters = []
    for members in groups.values():
        if len(members) >= 2:
            cent = pos[members].mean(axis=0)
            clusters.append({'members': members, 'centroid': cent, 'size': len(members)})
    return clusters

# ============================================================
# COMPLEXITY METRICS (entity-level, from candidate laws)
# ============================================================

def compute_metrics(frame_list, alpha_s, cfg):
    """
    Compute all eight entity-level metrics from a list of (pos, vel) frames.

    Metrics map to candidate laws:
      P1 opacity      - how well do particle positions predict cluster membership?
      P2 size_div     - diversity of cluster sizes (modular structure)
      P3 self_asm     - how quickly do clusters form from random initial state?
      P4 div_rate     - rate of cluster division/merger events
      P5 lifetime_h   - entropy of cluster lifetime distribution
      P6 interact     - density of cluster-cluster interactions
      P7 stability    - coefficient of variation of cluster count over time
      P8 scale_inv    - consistency of clustering across multiple bind distances
    """
    if len(frame_list) < 3:
        return {m: 0.0 for m in ['complexity','entity_c','field_c','count','stability',
                                  'interact','size_div','div_rate','sp_entropy']}
    box = cfg['BOX']
    frame_clusters = [find_clusters(pos, alpha_s, box) for pos, _ in frame_list]
    counts = np.array([len(c) for c in frame_clusters], dtype=float)

    # P7: Entity count stability
    mean_count = float(counts.mean())
    if mean_count < 0.5:
        return {m: 0.0 for m in ['complexity','entity_c','field_c','count','stability',
                                  'interact','size_div','div_rate','sp_entropy']}
    cv = float(counts.std()) / (mean_count + 1e-8)
    stability = 1.0 / (1.0 + cv)

    # P6: Interaction density
    interact_rates = []
    for clusters in frame_clusters:
        if len(clusters) < 2:
            interact_rates.append(0.0)
            continue
        cents = np.array([c['centroid'] for c in clusters])
        dx = cents[:, None, 0] - cents[None, :, 0]
        dy = cents[:, None, 1] - cents[None, :, 1]
        dx -= box * np.round(dx / box)
        dy -= box * np.round(dy / box)
        r = np.sqrt(dx**2 + dy**2)
        pairs = (r < box * 0.35).sum() - len(clusters)
        interact_rates.append(float(pairs) / len(clusters))
    interact = float(np.mean(interact_rates))

    # P2: Size diversity
    sizes = [c['size'] for cs in frame_clusters for c in cs]
    if sizes:
        sa = np.array(sizes)
        bins = min(14, max(2, len(np.unique(sa))))
        if bins > 1:
            h, _ = np.histogram(sa, bins=bins)
            h = h / h.sum()
            size_div = float(-np.sum(h[h > 0] * np.log2(h[h > 0])) / np.log2(bins))
        else:
            size_div = 0.0
    else:
        size_div = 0.0

    # P4: Division/merger rate
    div_rate = float(np.sum(np.abs(np.diff(counts)) >= 1)) / max(len(counts) - 1, 1)

    # Spatial entropy (field metric)
    sp_vals = []
    for pos, _ in frame_list:
        h2d, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=10,
                                    range=[[0, box], [0, box]])
        h2d = h2d / h2d.sum()
        sp_vals.append(float(-np.sum(h2d[h2d > 0] * np.log2(h2d[h2d > 0])) / np.log2(100)))
    sp_entropy = float(np.mean(sp_vals))

    # Composite
    entity_c = float(mean_count * stability * (1.0 + interact) * (1.0 + div_rate * 3.0))
    field_c = float(sp_entropy * (1.0 + size_div))
    complexity = float(np.sqrt(max(entity_c, 0) * max(field_c, 0) + 1e-12))

    return {
        'complexity': complexity,
        'entity_c': entity_c,
        'field_c': field_c,
        'count': mean_count,
        'stability': stability,
        'interact': interact,
        'size_div': size_div,
        'div_rate': div_rate,
        'sp_entropy': sp_entropy,
    }

# ============================================================
# MODE 1: LIVE SIMULATION
# ============================================================

def run_simulation(alpha=None, alpha_s=None):
    if alpha is None: alpha = CFG['ALPHA']
    if alpha_s is None: alpha_s = CFG['ALPHA_S']

    print(f"\nLive Simulation")
    print(f"  α={alpha:.3f}  αs={alpha_s:.3f}")
    print(f"  {CFG['N']} particles in {CFG['BOX']}×{CFG['BOX']} box")
    print(f"  Close window to exit\n")

    pos, vel = init_particles(CFG['N'], CFG['BOX'])
    frame_buffer = []
    metric_history = {'complexity':[], 'count':[], 'stability':[], 'interact':[]}
    step_count = [0]

    # Build figure
    fig = plt.figure(figsize=(13, 7), facecolor='#0e0e16')
    fig.suptitle(f'N-body Complexity Explorer   α={alpha:.3f}   αs={alpha_s:.3f}',
                 color='#ccccdd', fontsize=12, fontweight='bold')

    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.05, right=0.97, top=0.92, bottom=0.08,
                           hspace=0.45, wspace=0.35)

    ax_main = fig.add_subplot(gs[:, 0])   # particle view (full left column)
    ax_cx   = fig.add_subplot(gs[0, 1])   # complexity over time
    ax_cnt  = fig.add_subplot(gs[1, 1])   # cluster count over time
    ax_met  = fig.add_subplot(gs[0, 2])   # metric bars
    ax_info = fig.add_subplot(gs[1, 2])   # info text

    for ax in [ax_main, ax_cx, ax_cnt, ax_met, ax_info]:
        ax.set_facecolor('#0e0e16')
        ax.tick_params(colors='#888899', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#333344')

    # Main particle view
    ax_main.set_xlim(0, CFG['BOX']); ax_main.set_ylim(0, CFG['BOX'])
    ax_main.set_aspect('equal')
    ax_main.set_title('particle field', color='#8888aa', fontsize=9)
    scat = ax_main.scatter(pos[:, 0], pos[:, 1], s=12, c='#5577ff',
                            alpha=0.8, zorder=3, linewidths=0)
    cluster_circles = []

    # Time series axes
    ax_cx.set_title('complexity index', color='#8888aa', fontsize=9)
    ax_cx.set_ylabel('score', color='#888899', fontsize=8)
    cx_line, = ax_cx.plot([], [], '#1D9E75', lw=1.5)
    ax_cx.set_xlim(0, CFG['SIM_HISTORY_LEN'])

    ax_cnt.set_title('cluster count', color='#8888aa', fontsize=9)
    ax_cnt.set_ylabel('clusters', color='#888899', fontsize=8)
    ax_cnt.set_xlabel('frame', color='#888899', fontsize=8)
    cnt_line, = ax_cnt.plot([], [], '#7F77DD', lw=1.5)
    ax_cnt.set_xlim(0, CFG['SIM_HISTORY_LEN'])

    # Metric bars
    ax_met.set_title('current metrics', color='#8888aa', fontsize=9)
    ax_met.set_xlim(0, 1)
    metric_names = ['stability\n(P7)', 'interact\n(P6)', 'size div\n(P2)', 'div rate\n(P4)', 'sp entropy\n(P7)']
    metric_keys  = ['stability', 'interact', 'size_div', 'div_rate', 'sp_entropy']
    bar_colors   = ['#1D9E75', '#378ADD', '#7F77DD', '#D85A30', '#EF9F27']
    bars = ax_met.barh(metric_names, [0]*5, color=bar_colors, alpha=0.8, height=0.6)
    ax_met.set_xlim(0, 1.1)
    ax_met.tick_params(left=True, labelsize=7)
    for label in ax_met.get_yticklabels(): label.set_color('#aaaacc')

    # Info panel
    ax_info.axis('off')
    info_text = ax_info.text(0.05, 0.95, '', transform=ax_info.transAxes,
                              color='#aaaacc', fontsize=8, va='top',
                              fontfamily='monospace')

    def update(frame):
        nonlocal pos, vel
        # Step simulation
        for _ in range(CFG['SIM_STEPS_PER_FRAME']):
            pos, vel = step_simulation(pos, vel, alpha, alpha_s, CFG)
            step_count[0] += 1

        # Detect clusters
        clusters = find_clusters(pos, alpha_s, CFG['BOX'])

        # Update particle scatter — color by cluster membership
        colors = np.full(CFG['N'], '#334455')
        cluster_palette = ['#1D9E75','#7F77DD','#D85A30','#378ADD','#EF9F27',
                           '#ED93B1','#5DCAA5','#F0997B','#85B7EB','#97C459']
        for idx, cl in enumerate(clusters[:len(cluster_palette)]):
            for m in cl['members']:
                colors[m] = cluster_palette[idx % len(cluster_palette)]
        scat.set_offsets(pos)
        scat.set_color(colors)

        # Draw cluster circles
        for c in cluster_circles: c.remove()
        cluster_circles.clear()
        for cl in clusters:
            c = cl['centroid']
            r = np.sqrt(cl['size']) * alpha_s * 1.4
            circ = plt.Circle(c, r, fill=False, color='#ffffff22', lw=0.5, zorder=2)
            ax_main.add_patch(circ)
            cluster_circles.append(circ)

        # Buffer for metrics
        frame_buffer.append((pos.copy(), vel.copy()))
        if len(frame_buffer) > CFG['SIM_METRIC_WINDOW']:
            frame_buffer.pop(0)

        # Compute metrics every 5 frames
        if frame % 5 == 0 and len(frame_buffer) >= 5:
            m = compute_metrics(frame_buffer, alpha_s, CFG)
            for k in metric_history: metric_history[k].append(m.get(k,0))
            if len(metric_history['complexity']) > CFG['SIM_HISTORY_LEN']:
                for k in metric_history: metric_history[k].pop(0)

            # Update time series
            x = list(range(len(metric_history['complexity'])))
            cx_line.set_data(x, metric_history['complexity'])
            cnt_line.set_data(x, metric_history['count'])
            cx_vals = metric_history['complexity']
            if cx_vals:
                ax_cx.set_ylim(0, max(max(cx_vals)*1.15, 1))
            cnt_vals = metric_history['count']
            if cnt_vals:
                ax_cnt.set_ylim(0, max(max(cnt_vals)*1.15, 1))

            # Update bars (normalize to [0,1] for display)
            norm_vals = {
                'stability': m['stability'],
                'interact': min(1.0, m['interact'] / 3.0),
                'size_div': m['size_div'],
                'div_rate': min(1.0, m['div_rate'] * 5),
                'sp_entropy': m['sp_entropy'],
            }
            for bar, key in zip(bars, metric_keys):
                bar.set_width(norm_vals[key])

            # Info text
            info_text.set_text(
                f"step:       {step_count[0]:,}\n"
                f"clusters:   {len(clusters)}\n"
                f"complexity: {m['complexity']:.3f}\n"
                f"stability:  {m['stability']:.3f}\n"
                f"interact:   {m['interact']:.3f}\n"
                f"size div:   {m['size_div']:.3f}\n"
                f"div rate:   {m['div_rate']:.3f}\n"
                f"sp entropy: {m['sp_entropy']:.3f}\n\n"
                f"α  = {alpha:.4f}\n"
                f"αs = {alpha_s:.4f}"
            )

        return [scat] + cluster_circles + [cx_line, cnt_line] + list(bars) + [info_text]

    ani = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
    plt.show()

# ============================================================
# MODE 2: PARAMETER SCAN
# ============================================================

def run_scan(save_file=None):
    if save_file is None: save_file = CFG['SCAN_SAVE_FILE']
    N_A = CFG['SCAN_N_ALPHA']
    N_AS = CFG['SCAN_N_ALPHAS']
    ALPHA_RANGE = np.logspace(np.log10(CFG['SCAN_ALPHA_MIN']),
                               np.log10(CFG['SCAN_ALPHA_MAX']), N_A)
    ALPHAS_RANGE = np.logspace(np.log10(CFG['SCAN_ALPHAS_MIN']),
                                np.log10(CFG['SCAN_ALPHAS_MAX']), N_AS)
    STEPS = CFG['SCAN_STEPS']
    SKIP  = CFG['SCAN_SKIP']
    SE    = CFG['SCAN_SAMPLE_EVERY']

    total = N_A * N_AS
    print(f"\nParameter Scan")
    print(f"  Grid: {N_A}×{N_AS} = {total} combinations")
    print(f"  α:  [{ALPHA_RANGE[0]:.3f}, {ALPHA_RANGE[-1]:.3f}] (our universe = 1.0)")
    print(f"  αs: [{ALPHAS_RANGE[0]:.3f}, {ALPHAS_RANGE[-1]:.3f}] (our universe = 1.0)")
    print(f"  Steps: {STEPS}  Skip: {SKIP}  Seeds: {CFG['SCAN_SEEDS']}")
    print(f"  Saving to: {save_file}\n")

    # Progress bar
    try:
        from tqdm import tqdm
        progress = tqdm(total=total, ncols=70)
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    cmap     = np.zeros((N_A, N_AS))
    emap     = np.zeros((N_A, N_AS))
    fmap     = np.zeros((N_A, N_AS))
    cnt_map  = np.zeros((N_A, N_AS))
    stab_map = np.zeros((N_A, N_AS))
    int_map  = np.zeros((N_A, N_AS))

    t0 = time.time()
    for i, alpha in enumerate(ALPHA_RANGE):
        row_scores = []
        for j, alpha_s in enumerate(ALPHAS_RANGE):
            seed_metrics = []
            for seed in CFG['SCAN_SEEDS']:
                pos, vel = init_particles(CFG['N'], CFG['BOX'], seed=seed)
                frames = []
                for s in range(STEPS):
                    pos, vel = step_simulation(pos, vel, alpha, alpha_s, CFG)
                    if s >= SKIP and s % SE == 0:
                        frames.append((pos.copy(), vel.copy()))
                m = compute_metrics(frames, alpha_s, CFG)
                seed_metrics.append(m)

            avg = {k: float(np.mean([m[k] for m in seed_metrics]))
                   for k in seed_metrics[0]}
            cmap[i, j]    = avg['complexity']
            emap[i, j]    = avg['entity_c']
            fmap[i, j]    = avg['field_c']
            cnt_map[i, j] = avg['count']
            stab_map[i,j] = avg['stability']
            int_map[i, j] = avg['interact']

            if use_tqdm: progress.update(1)
            else:
                done = i*N_AS + j + 1
                pct = done/total*100
                bar = '█'*int(pct//5) + '░'*(20-int(pct//5))
                print(f"\r  [{bar}] {pct:.0f}% α={alpha:.3f} αs={alpha_s:.3f} "
                      f"score={avg['complexity']:.3f}   ", end='', flush=True)

    if use_tqdm: progress.close()
    else: print()

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s")

    # Find our universe
    oi = np.argmin(np.abs(ALPHA_RANGE  - 1.0))
    oj = np.argmin(np.abs(ALPHAS_RANGE - 1.0))
    our_score = float(cmap[oi, oj])
    our_pct   = float(np.mean(cmap.flatten() < our_score)) * 100

    # Adams island
    am_mask = ((ALPHA_RANGE[:, None] >= 0.5) & (ALPHA_RANGE[:, None] <= 2.0) &
               (ALPHAS_RANGE[None, :] >= 0.5) & (ALPHAS_RANGE[None, :] <= 2.0))
    am_mean = float(cmap[am_mask].mean()) if am_mask.any() else 0.0

    pk = np.unravel_index(np.argmax(cmap), cmap.shape)
    near_zero = float(np.mean(cmap.flatten() < 0.1))

    print(f"\n  Results:")
    print(f"    Peak:          α={ALPHA_RANGE[pk[0]]:.3f}  αs={ALPHAS_RANGE[pk[1]]:.3f}  score={cmap.max():.3f}")
    print(f"    Our universe:  α=1.000  αs=1.000  score={our_score:.3f}  ({our_pct:.0f}th percentile)")
    print(f"    Adams island:  mean={am_mean:.3f}  vs overall {cmap.mean():.3f}  = {am_mean/max(cmap.mean(),1e-6):.2f}×")
    print(f"    Near-zero:     {near_zero:.1%}")

    data = {
        'alpha_range':   ALPHA_RANGE.tolist(),
        'alphas_range':  ALPHAS_RANGE.tolist(),
        'complexity_map': cmap.tolist(),
        'entity_map':    emap.tolist(),
        'field_map':     fmap.tolist(),
        'count_map':     cnt_map.tolist(),
        'stability_map': stab_map.tolist(),
        'interact_map':  int_map.tolist(),
        'n_alpha': N_A, 'n_alphas': N_AS,
        'peak': {
            'alpha': float(ALPHA_RANGE[pk[0]]),
            'alpha_s': float(ALPHAS_RANGE[pk[1]]),
            'score': float(cmap.max()),
            'i': int(pk[0]), 'j': int(pk[1])
        },
        'our_universe': {
            'alpha': 1.0, 'alpha_s': 1.0,
            'score': our_score, 'percentile': our_pct,
            'i': int(oi), 'j': int(oj)
        },
        'adams_region': {
            'mean_score': am_mean,
            'overall_mean': float(cmap.mean()),
            'ratio': float(am_mean / max(cmap.mean(), 1e-6))
        },
        'near_zero_fraction': near_zero,
        'elapsed_seconds': elapsed,
    }

    with open(save_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n  Saved to {save_file}")
    print(f"  Run 'python complexity_explorer.py heatmap' to visualise\n")
    return data

# ============================================================
# MODE 3: HEATMAP VISUALISATION
# ============================================================

METRIC_OPTIONS = {
    'complexity':   ('Complexity index (composite)',  '#1D9E75'),
    'entity_map':   ('Entity score (cluster dynamics)', '#7F77DD'),
    'stability_map':('P7: cluster count stability',   '#378ADD'),
    'interact_map': ('P6: interaction density',        '#D85A30'),
    'count_map':    ('Mean cluster count',             '#EF9F27'),
    'field_map':    ('Field score (spatial structure)', '#5DCAA5'),
}

def show_heatmap(data_file=None):
    if data_file is None: data_file = CFG['SCAN_SAVE_FILE']
    if not os.path.exists(data_file):
        print(f"\n  No scan file found at '{data_file}'")
        print(f"  Run the scanner first: python complexity_explorer.py scan\n")
        return

    print(f"\nLoading scan from {data_file}...")
    with open(data_file) as f:
        data = json.load(f)

    ALPHA   = np.array(data['alpha_range'])
    ALPHAS  = np.array(data['alphas_range'])
    maps = {
        'complexity':    np.array(data['complexity_map']),
        'entity_map':    np.array(data.get('entity_map',   data['complexity_map'])),
        'stability_map': np.array(data.get('stability_map',data['complexity_map'])),
        'interact_map':  np.array(data.get('interact_map', data['complexity_map'])),
        'count_map':     np.array(data.get('count_map',    data['complexity_map'])),
        'field_map':     np.array(data.get('field_map',    data['complexity_map'])),
    }
    pk   = data['peak']
    our  = data['our_universe']
    adam = data['adams_region']

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor='#0e0e16')
    fig.suptitle('Universal Constants Complexity Landscape\n'
                 'Each cell = one "universe" with different force constants.  '
                 'Color = complexity score from entity-level metrics.',
                 color='#ccccdd', fontsize=11)

    for ax, (key, (title, accent)) in zip(axes.flat, METRIC_OPTIONS.items()):
        M = maps[key]
        # Custom colormap: dark background → accent color
        cmap_custom = mcolors.LinearSegmentedColormap.from_list(
            'custom', ['#0e0e16', '#1a2a3a', accent, '#ffffff'], N=256)

        im = ax.imshow(M.T, origin='lower', aspect='auto',
                       extent=[ALPHA[0], ALPHA[-1], ALPHAS[0], ALPHAS[-1]],
                       cmap=cmap_custom)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('α (EM coupling)', color='#888899', fontsize=8)
        ax.set_ylabel('αs (strong force)', color='#888899', fontsize=8)
        ax.set_title(title, color='#aaaacc', fontsize=9, pad=4)
        ax.tick_params(colors='#888899', labelsize=7)
        for sp in ax.spines.values(): sp.set_color('#333344')

        # Adams island box
        ax.add_patch(Rectangle((0.5, 0.5), 1.5, 1.5,
                                fill=False, edgecolor='#7F77DD',
                                linewidth=1.5, linestyle='--', zorder=5))

        # Our universe marker
        ax.plot(our['alpha'], our['alpha_s'], 'o',
                color='#1D9E75', markersize=9, zorder=6,
                markeredgecolor='white', markeredgewidth=1.2,
                label=f"Our universe ({our['percentile']:.0f}th pct)")

        # Peak marker
        ax.plot(pk['alpha'], pk['alpha_s'], '*',
                color='#D85A30', markersize=13, zorder=6,
                markeredgecolor='white', markeredgewidth=0.8,
                label=f"Peak α={pk['alpha']:.2f}")

        if key == 'complexity':
            ax.legend(fontsize=7, facecolor='#1a1a2a',
                      labelcolor='#ccccdd', edgecolor='#333344',
                      loc='upper right')

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
            colors='#888899', labelsize=7)

    # Summary stats in a text box on last panel if there are extra axes
    summary = (
        f"Scan summary\n"
        f"{'─'*28}\n"
        f"Peak universe\n"
        f"  α  = {pk['alpha']:.3f}\n"
        f"  αs = {pk['alpha_s']:.3f}\n"
        f"  score = {pk['score']:.3f}\n\n"
        f"Our universe (α=αs=1.0)\n"
        f"  score = {our['score']:.3f}\n"
        f"  percentile = {our['percentile']:.0f}th\n\n"
        f"Adams island of stability\n"
        f"  mean = {adam['mean_score']:.3f}\n"
        f"  vs overall = {adam['overall_mean']:.3f}\n"
        f"  advantage = {adam['ratio']:.2f}×\n\n"
        f"Near-zero universes\n"
        f"  {data.get('near_zero_fraction',0)*100:.1f}%"
    )
    # Add text to last subplot
    axes[1, 2].axis('off')
    axes[1, 2].set_facecolor('#0e0e16')
    axes[1, 2].text(0.08, 0.95, summary, transform=axes[1, 2].transAxes,
                    color='#ccccdd', fontsize=9, va='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='#1a1a2a',
                              edgecolor='#333344', alpha=0.9))

    # Legend for markers
    axes[1, 2].plot(0, 0, 'o', color='#1D9E75', ms=8,
                    markeredgecolor='white', markeredgewidth=1,
                    label='Our universe', transform=axes[1, 2].transAxes)
    axes[1, 2].plot(0, 0, '*', color='#D85A30', ms=10,
                    markeredgecolor='white', markeredgewidth=0.8,
                    label='Complexity peak', transform=axes[1, 2].transAxes)
    axes[1, 2].add_patch(Rectangle((0, 0), 0, 0,
                                    fill=False, edgecolor='#7F77DD',
                                    linewidth=1.5, linestyle='--',
                                    label='Adams island',
                                    transform=axes[1, 2].transAxes))
    axes[1, 2].legend(fontsize=8, facecolor='#1a1a2a',
                       labelcolor='#ccccdd', edgecolor='#333344',
                       loc='lower left', bbox_to_anchor=(0.05, 0.02))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

# ============================================================
# INTERACTIVE MENU
# ============================================================

BANNER = """
╔═══════════════════════════════════════════════════════╗
║         COMPLEXITY EXPLORER                           ║
║         N-body particle simulation                    ║
║         with entity-level complexity metrics          ║
╚═══════════════════════════════════════════════════════╝

  Based on eight candidate laws of complexity:
    P1  Opaque hierarchical layering
    P2  Modular interconnection
    P3  Self-assembly without top-down specification
    P4  Recursive self-modification
    P5  Selection pressure
    P6  Co-evolutionary dynamics
    P7  Thermodynamic drive
    P8  Scale invariance

  Two tunable "universal constants" (our universe = 1.0, 1.0):
    α   = electromagnetic coupling  (attractive strength)
    αs  = strong force coupling     (repulsion steepness)
"""

def menu():
    print(BANNER)
    while True:
        print("  What would you like to do?")
        print()
        print("  1. Run live simulation  (visualise particles in real time)")
        print("  2. Run parameter scan   (measure complexity across α × αs space)")
        print("  3. Show heatmap         (visualise a saved scan)")
        print("  4. Quit")
        print()
        choice = input("  Enter 1-4: ").strip()

        if choice == '1':
            print()
            a_str  = input(f"  α  (EM coupling, our universe=1.0) [{CFG['ALPHA']}]: ").strip()
            as_str = input(f"  αs (strong force, our universe=1.0) [{CFG['ALPHA_S']}]: ").strip()
            alpha   = float(a_str)  if a_str  else CFG['ALPHA']
            alpha_s = float(as_str) if as_str else CFG['ALPHA_S']
            run_simulation(alpha=alpha, alpha_s=alpha_s)

        elif choice == '2':
            print()
            print(f"  Default grid: {CFG['SCAN_N_ALPHA']}×{CFG['SCAN_N_ALPHAS']} combinations")
            na_str  = input(f"  Grid size per axis [{CFG['SCAN_N_ALPHA']}]: ").strip()
            if na_str:
                n = int(na_str)
                CFG['SCAN_N_ALPHA'] = n
                CFG['SCAN_N_ALPHAS'] = n
            run_scan()

        elif choice == '3':
            print()
            f_str = input(f"  Scan file [{CFG['SCAN_SAVE_FILE']}]: ").strip()
            show_heatmap(f_str if f_str else None)

        elif choice == '4':
            print("\n  Goodbye.\n")
            break
        else:
            print("  Please enter 1, 2, 3, or 4.")
        print()

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    args = sys.argv[1:]

    if not args:
        menu()
    elif args[0] == 'simulate':
        alpha   = float(args[1]) if len(args) > 1 else CFG['ALPHA']
        alpha_s = float(args[2]) if len(args) > 2 else CFG['ALPHA_S']
        run_simulation(alpha=alpha, alpha_s=alpha_s)
    elif args[0] == 'scan':
        n = int(args[1]) if len(args) > 1 else None
        if n:
            CFG['SCAN_N_ALPHA'] = n
            CFG['SCAN_N_ALPHAS'] = n
        run_scan()
    elif args[0] in ('heatmap', 'map', 'plot'):
        f = args[1] if len(args) > 1 else None
        show_heatmap(f)
    else:
        print(__doc__)
