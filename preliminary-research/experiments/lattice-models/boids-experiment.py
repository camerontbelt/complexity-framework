"""
boids-experiment.py
===================
Vicsek flocking model — a minimal self-propelled particle system that has a
well-studied order/disorder phase transition controlled by noise.

WHAT THIS MODELS
----------------
N agents (particles) each move at constant speed in a 2-D periodic box.
At each step an agent adopts the *average heading* of all agents within
radius r, plus a random noise angle drawn from [-eta/2, +eta/2].

  eta ~ 0 (low noise) : all agents align → ordered flock sweeping box
  eta ~ pi (high noise): agents ignore neighbours → random walk
  eta ~ eta_c (~2.0)  : many small, competing sub-flocks, turbulent
                        edges, intermittent merging/splitting → complex

The binarised spatial grid (cell occupied/empty) is fed to compute_full_C.
The rich cluster dynamics at intermediate noise should produce a peak in C.

HYPOTHESIS
----------
H1: C peaks at an intermediate noise level eta*, neither fully ordered nor
    fully disordered, corresponding to the edge-of-chaos flocking regime.
H0: C is flat or monotone across eta — the metric does not detect
    the order-disorder transition in collective motion.

DESIGN
------
  Grid:    32 x 32  (periodic)
  Agents:  200      (density ~ 20 %, keeps entropy in useful range)
  Speed:   v = 0.3 cells/step
  Radius:  r = 1.5  (interaction neighbourhood)
  Warmup:  500 steps
  Frames:  120 frames per run (one frame per 5 simulation steps)
  Seeds:   6 per noise level
  Noise:   eta in {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, pi}
"""

import os, sys, csv as _csv, importlib.util, time, warnings
import numpy as np
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Bootstrap compute_full_C
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_nn   = os.path.join(os.path.dirname(_HERE), "neural-network")
_spec = importlib.util.spec_from_file_location(
            "mnist_exp", os.path.join(_nn, "mnist-experiment.py"))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_full_C = _mod.compute_full_C

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
L         = 32        # box side length (cells)
N_AGENTS  = 200       # number of agents
SPEED     = 0.3       # cells per step
RADIUS    = 1.5       # alignment neighbourhood radius (cells)
WARMUP    = 500       # steps before capturing frames
N_FRAMES  = 120       # frames per run
FRAME_EVERY = 5       # simulation steps between captured frames
N_SEEDS   = 6         # independent realisations per noise level

ETA_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, float(np.pi)]

# ---------------------------------------------------------------------------
# Vicsek simulation
# ---------------------------------------------------------------------------

def vicsek_step(pos, theta, eta):
    """One Vicsek update: align with neighbours + noise, then move."""
    # Pairwise displacement with periodic boundary
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]   # (N,N,2)
    diff -= L * np.round(diff / L)                          # wrap to [-L/2,L/2)
    dist = np.sqrt((diff**2).sum(-1))                       # (N,N)

    # Neighbours within radius (including self)
    nbr = dist < RADIUS                                     # (N,N) bool

    # Average heading of neighbours
    sin_avg = (nbr * np.sin(theta[np.newaxis, :])).sum(-1) / nbr.sum(-1)
    cos_avg = (nbr * np.cos(theta[np.newaxis, :])).sum(-1) / nbr.sum(-1)
    avg_theta = np.arctan2(sin_avg, cos_avg)

    # New heading = average + uniform noise
    new_theta = avg_theta + np.random.uniform(-eta/2, eta/2, N_AGENTS)

    # Move
    new_pos = pos + SPEED * np.column_stack([np.cos(new_theta),
                                             np.sin(new_theta)])
    new_pos %= L   # periodic wrap

    return new_pos, new_theta


def pos_to_frame(pos):
    """Snap agent positions to integer grid → binary (L,L) occupancy array."""
    grid = np.zeros((L, L), dtype=np.float32)
    ix   = np.clip(pos[:, 0].astype(int), 0, L-1)
    iy   = np.clip(pos[:, 1].astype(int), 0, L-1)
    grid[ix, iy] = 1.0
    return grid


def run_vicsek(eta, seed):
    rng = np.random.RandomState(seed)
    pos   = rng.uniform(0, L, (N_AGENTS, 2))
    theta = rng.uniform(-np.pi, np.pi, N_AGENTS)

    for _ in range(WARMUP):
        pos, theta = vicsek_step(pos, theta, eta)

    frames = []
    for _ in range(N_FRAMES):
        for _ in range(FRAME_EVERY):
            pos, theta = vicsek_step(pos, theta, eta)
        frames.append(pos_to_frame(pos))

    return frames


def frames_to_volumes(frames):
    return [f.reshape(1, 1, -1) for f in frames]


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def measure(frames):
    return compute_full_C(frames_to_volumes(frames))


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    s = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2.0)
    return float((a.mean() - b.mean()) / max(s, 1e-9))


def order_param(frames):
    """Mean absolute value of complex order parameter |<e^{i*theta}>|.
    Approximate from velocity field via gradient of positions — not perfect,
    but gives a sense of alignment degree for diagnostics."""
    # We'll compute occupancy-based proxy: std of row-sums (0=uniform, high=clumped)
    row_sums = [f.sum(1) for f in frames]
    return float(np.mean([rs.std() for rs in row_sums]))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Boids / Vicsek flocking experiment")
    print(f"Grid {L}x{L}, {N_AGENTS} agents, r={RADIUS}, v={SPEED}")
    print(f"{N_FRAMES} frames x {FRAME_EVERY} steps, {N_SEEDS} seeds\n")

    results = {}
    t0 = time.time()

    for eta in ETA_VALUES:
        c_vals, ord_vals = [], []
        for seed in range(N_SEEDS):
            frames = run_vicsek(eta, seed)
            res    = measure(frames)
            c_vals.append(res)
            ord_vals.append(order_param(frames))

        mean_C     = np.mean([r["C_a"]    for r in c_vals])
        std_C      = np.std ([r["C_a"]    for r in c_vals])
        mean_ord   = np.mean(ord_vals)
        mean_dens  = np.mean([f.mean()    for f in frames])

        results[eta] = c_vals
        print(f"  eta={eta:.3f}  C={mean_C:.4f} (+/-{std_C:.4f})  "
              f"clump={mean_ord:.3f}  density={mean_dens:.3f}  "
              f"{time.time()-t0:.0f}s")

    # -----------------------------------------------------------------------
    # CSV
    # -----------------------------------------------------------------------
    csv_path = os.path.join(_HERE, "boids_results.csv")
    fields   = ["eta", "seed", "C_a", "mean_H", "std_H",
                "op_up", "op_down", "mi1", "decay", "tc_mean", "gzip_ratio"]
    with open(csv_path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for eta, c_list in results.items():
            for seed, r in enumerate(c_list):
                row = {"eta": round(eta, 5), "seed": seed}
                for f in fields[2:]:
                    row[f] = round(float(r.get(f, 0.0)), 6)
                w.writerow(row)
    print(f"\nCSV -> {csv_path}")

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------
    etas   = sorted(results.keys())
    means  = [np.mean([r["C_a"] for r in results[e]]) for e in etas]
    stds   = [np.std ([r["C_a"] for r in results[e]]) for e in etas]

    peak_eta = etas[int(np.argmax(means))]
    peak_C   = [r["C_a"] for r in results[peak_eta]]
    lo_C     = [r["C_a"] for r in results[etas[0]]]    # most ordered
    hi_C     = [r["C_a"] for r in results[etas[-1]]]   # most disordered

    d_vs_ordered    = cohens_d(peak_C, lo_C)
    d_vs_disordered = cohens_d(peak_C, hi_C)
    _, p_ordered    = scipy_stats.ttest_ind(peak_C, lo_C,    equal_var=False)
    _, p_disordered = scipy_stats.ttest_ind(peak_C, hi_C,    equal_var=False)

    print(f"\n{'='*60}")
    print(f"  STATISTICAL SUMMARY — Boids / Vicsek")
    print(f"{'='*60}")
    print(f"  Peak C at eta={peak_eta:.3f}:  "
          f"{np.mean(peak_C):.4f} +/- {np.std(peak_C):.4f}")
    print(f"  vs ordered   (eta={etas[0]:.3f}): "
          f"{np.mean(lo_C):.4f} +/- {np.std(lo_C):.4f}  "
          f"d={d_vs_ordered:.2f}  p={p_ordered:.4f}")
    print(f"  vs disordered (eta={etas[-1]:.3f}): "
          f"{np.mean(hi_C):.4f} +/- {np.std(hi_C):.4f}  "
          f"d={d_vs_disordered:.2f}  p={p_disordered:.4f}")
    outcome = ("H1 SUPPORTED"
               if (d_vs_ordered > 0.8 and p_ordered < 0.05 and
                   d_vs_disordered > 0.8 and p_disordered < 0.05)
               else "inconclusive")
    print(f"  -> {outcome}")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555577")

    colour = "#4fc3f7"

    # Panel A: C vs eta
    ax = axes[0]
    ax.errorbar(etas, means, yerr=stds,
                fmt="o-", color=colour, capsize=5, linewidth=2,
                markersize=9, elinewidth=1.5, markeredgecolor="white",
                markeredgewidth=0.5, label="C (mean +/- 1 SD)")

    ax.axvline(peak_eta, color="#ffb74d", linestyle="--", linewidth=1.5,
               alpha=0.8, label=f"peak eta = {peak_eta:.2f}")

    # Mark approximate Vicsek critical point for these parameters
    ETA_C_APPROX = 2.0
    ax.axvline(ETA_C_APPROX, color="#ef9a9a", linestyle=":", linewidth=1.5,
               alpha=0.7, label=f"Vicsek eta_c ~ {ETA_C_APPROX:.1f} (approx)")

    ax.text(0.03, 0.97,
            "H1: C peaks at intermediate noise\n    (flocking edge-of-chaos)\n"
            "H0: C flat or monotone",
            transform=ax.transAxes, fontsize=9, va="top", color="white",
            bbox=dict(facecolor="#2d2d4e", edgecolor="#ffb74d",
                      alpha=0.9, boxstyle="round,pad=0.4"))

    ax.set_xlabel("Noise level  eta", fontsize=11)
    ax.set_ylabel("C  (complexity score)", fontsize=11)
    ax.set_title(
        "Boids / Vicsek Model\nC vs Noise Level  (eta)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, facecolor="#1e1e3f", labelcolor="white",
              edgecolor="#555577")
    ax.grid(True, alpha=0.2, color="white")

    # Panel B: sub-metric breakdown at each eta
    ax = axes[1]
    sub_labels = ["mean_H", "op_up", "op_down", "mi1", "tc_mean", "gzip_ratio"]
    sub_colours = ["#f48fb1", "#80cbc4", "#80deea", "#ffe082", "#c5e1a5", "#ce93d8"]

    for label, sc in zip(sub_labels, sub_colours):
        vals = [np.mean([r.get(label, 0.0) for r in results[e]]) for e in etas]
        ax.plot(etas, vals, "o-", color=sc, linewidth=1.8,
                markersize=6, label=label, alpha=0.9)

    ax.set_xlabel("Noise level  eta", fontsize=11)
    ax.set_ylabel("Raw sub-metric value", fontsize=11)
    ax.set_title("Sub-metric Breakdown vs Noise\n(shows which channels drive the C peak)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, facecolor="#1e1e3f",
              labelcolor="white", edgecolor="#555577")
    ax.grid(True, alpha=0.2, color="white")

    fig.suptitle(
        f"Vicsek Flocking  —  L={L}, N={N_AGENTS}, r={RADIUS}, v={SPEED}\n"
        f"{N_FRAMES} frames, {N_SEEDS} seeds per condition",
        fontsize=13, fontweight="bold", color="white", y=1.01)

    plt.tight_layout()
    png_path = os.path.join(_HERE, "boids_results.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot -> {png_path}")
    print(f"\nTotal elapsed: {time.time()-t0:.0f}s")
    print("Done.")
