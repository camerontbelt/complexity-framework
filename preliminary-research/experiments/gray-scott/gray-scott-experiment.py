"""
gray-scott-experiment.py
========================
Applies the agnostic complexity metric (C_a) to the Gray-Scott
reaction-diffusion system — a continuous 2D PDE substrate.

Gray-Scott dynamics:
  du/dt = Du·∇²u - u·v² + f·(1-u)
  dv/dt = Dv·∇²v + u·v² - (f+k)·v

Six parameter points span the known phase diagram from trivial to chaotic.
Ground truth classifications follow Pearson (1993) and the Gray-Scott explorer.

Key design choice: binarization threshold
-----------------------------------------
The metric requires a binary grid. For Gray-Scott, the v-field (inhibitor)
ranges from ~0 in the background to ~0.2-0.4 in spots/worms. A threshold
that captures 15-35% active cells keeps the spatial entropy gate
tanh(K·H)·tanh(K·(1-H)) in its productive intermediate range.
We use THRESHOLD = 0.10 as primary and report actual densities so the
reader can verify. A density diagnostic is printed for each parameter set.

H1: Parameter regimes producing dynamically rich patterns (self-replicating
    spots, interacting worms) score higher C_a than trivial or fully chaotic
    regimes. (d > 0.5, p < 0.05 for complex vs. non-complex)
H0: No systematic relationship between Gray-Scott behavioural class and C_a.
"""

import os
import sys
import csv as _csv
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display required
import matplotlib.pyplot as plt
from scipy import stats as sp

# =============================================================================
# Bootstrap — import compute_full_C from mnist-experiment.py
# =============================================================================
_HERE  = os.path.dirname(os.path.abspath(__file__))
_nn    = os.path.join(_HERE, "neural-network")
_spec  = importlib.util.spec_from_file_location(
             "mnist_exp", os.path.join(_nn, "mnist-experiment.py"))
_mod   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_full_C = _mod.compute_full_C

# =============================================================================
# Simulation constants
# =============================================================================
DU          = 0.16   # u diffusion coefficient
DV          = 0.08   # v diffusion coefficient
DT          = 1.0    # time step (stable for Du=0.16, Dv=0.08)
GRID        = 128    # spatial grid size (128×128 cells)
N_WARMUP    = 3000   # steps before measurement begins
SAMPLE_STEP = 20     # steps between measured frames
N_FRAMES    = 100    # number of frames measured (T for the metric)
N_SEEDS     = 3      # independent random initial conditions per parameter set
# =============================================================================
# Entity-level measurement
# =============================================================================
# Gray-Scott entities (spots, worms) span ~5-50 cells each.
# Feeding individual cells to the metric means the 3-cell spatial patch is
# blind to structure at entity scale — exactly the issue encountered on the
# first attempt.
#
# Fix: coarsen the binary grid to ENTITY_PATCH × ENTITY_PATCH super-cells.
# Each super-cell captures a region the size of one entity (or part of one).
# For GRID=128 and ENTITY_PATCH=8: 16×16 = 256 super-cells.
# A single spot of radius ~6-8 cells then maps to ~2-4 super-cells.
#
# Within the super-cell volume we still use an adaptive per-frame threshold:
# top ACTIVE_PCT % of super-cells by mean v → always 25 % active.
# Temporal dynamics of WHICH super-cells are active then reflect:
#   - static patterns    → same super-cells always active (high mi1, no decay)
#   - moving/dynamic     → different super-cells active (intermediate mi1, decay > 0)
#   - turbulent          → rapidly changing (low mi1)
# This is what we actually want to measure.
ENTITY_PATCH = 8      # spatial coarsening factor  (8×8 cells → 1 super-cell)
ACTIVE_PCT   = 25.0   # adaptive threshold: top 25 % of super-cells → active
THRESHOLD    = 0.10   # fixed cell-level threshold kept for density diagnostics

P_THRESH = 0.05
D_THRESH = 0.50

# =============================================================================
# Parameter sets spanning the Gray-Scott phase diagram
# (f, k) values follow Pearson 1993 / Munafo 1996 classifications.
# =============================================================================
PARAMETER_SETS = [
    #  name               f       k      description                expected class
    ("dead",            0.090,  0.059,  "v dies (no reaction)",    "trivial"),
    ("static_spots",    0.042,  0.065,  "stable Turing spots",     "ordered"),
    ("self_rep_spots",  0.030,  0.057,  "self-replicating spots",  "complex"),
    ("worm_complex",    0.055,  0.062,  "labyrinthine worms",      "complex"),
    ("solitons",        0.025,  0.060,  "moving solitons",         "complex"),
    ("chaotic",         0.026,  0.051,  "turbulent chaos",         "chaotic"),
]

# =============================================================================
# Gray-Scott simulation engine
# =============================================================================

def laplacian_2d(Z):
    """5-point Laplacian with periodic (wrap-around) boundary conditions."""
    return (np.roll(Z,  1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z,  1, axis=1) + np.roll(Z, -1, axis=1) - 4.0 * Z)


def gray_scott_init(n, seed=0):
    """
    Standard initial condition:
      u = 1 everywhere + small Gaussian noise
      v = 0 everywhere + small noise, except a centre square seeded with v=0.25
    """
    rng = np.random.RandomState(seed)
    u = np.ones((n, n)) + 0.02 * rng.randn(n, n)
    v = np.zeros((n, n)) + 0.02 * rng.randn(n, n)
    mid, sq = n // 2, n // 8      # 16-cell seed region for n=128
    u[mid-sq:mid+sq, mid-sq:mid+sq] = 0.50 + 0.02 * rng.randn(2*sq, 2*sq)
    v[mid-sq:mid+sq, mid-sq:mid+sq] = 0.25 + 0.02 * rng.randn(2*sq, 2*sq)
    return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)


def gray_scott_step(u, v, f, k, n_steps=1):
    """Euler integration for n_steps, clipping to [0,1] for stability."""
    for _ in range(n_steps):
        uvv  = u * v * v
        u_new = np.clip(u + DT * (DU * laplacian_2d(u) - uvv + f * (1.0 - u)), 0.0, 1.0)
        v_new = np.clip(v + DT * (DV * laplacian_2d(v) + uvv - (f + k) * v),   0.0, 1.0)
        u, v = u_new, v_new
    return u, v


def coarsen_to_entity_scale(v_field, patch=ENTITY_PATCH, active_pct=ACTIVE_PCT):
    """
    Coarsen the (GRID, GRID) v-field to entity-scale super-cells.

    Steps
    -----
    1. Divide the grid into non-overlapping (patch × patch) blocks.
    2. Compute the mean v in each block → (n_super, n_super) float array.
    3. Apply an adaptive per-frame threshold: top active_pct % → binary 1.
       This locks density to exactly active_pct / 100 in every frame,
       keeping mean_H constant across all parameter regimes.
       Discrimination comes purely from SPATIAL and TEMPORAL structure.

    Returns
    -------
    (1, 1, n_super²) binary float32 volume for compute_full_C.
    """
    n = v_field.shape[0]
    n_super = n // patch
    # Mean v in each (patch × patch) block
    blocks = v_field.reshape(n_super, patch, n_super, patch).mean(axis=(1, 3))  # (n_super, n_super)
    # Adaptive threshold: top active_pct % of super-cells
    cut    = np.percentile(blocks, 100.0 - active_pct)
    binary = (blocks > cut).astype(np.float32)   # (n_super, n_super)
    return binary.reshape(1, 1, -1)              # (1, 1, n_super²)


def run_gray_scott(f, k, seed=0, threshold=THRESHOLD):
    """
    Simulate Gray-Scott, extract T=N_FRAMES binary volumes.

    Returns
    -------
    volumes   : list of N_FRAMES arrays, each shape (1, 1, GRID*GRID)
    v_final   : (GRID, GRID) float array — final v field for visualisation
    densities : (N_FRAMES,) array of fraction of active cells per frame
    """
    u, v = gray_scott_init(GRID, seed)
    u, v = gray_scott_step(u, v, f, k, n_steps=N_WARMUP)

    volumes       = []
    densities_fix = []   # cell-level fixed threshold (diagnostic only)
    for _ in range(N_FRAMES):
        u, v = gray_scott_step(u, v, f, k, n_steps=SAMPLE_STEP)
        densities_fix.append(float((v > threshold).mean()))
        # Entity-level volume: coarsen to super-cell scale, adaptive threshold
        vol = coarsen_to_entity_scale(v)
        volumes.append(vol)
    n_super = GRID // ENTITY_PATCH
    return volumes, v, np.array(densities_fix), n_super


# =============================================================================
# Experiment
# =============================================================================

def run_experiment():
    print("\n" + "=" * 70)
    print("Gray-Scott Reaction-Diffusion — Complexity Experiment")
    print()
    print(f"  Grid: {GRID}×{GRID}   Warmup: {N_WARMUP} steps")
    print(f"  Measurement: {N_FRAMES} frames sampled every {SAMPLE_STEP} steps")
    n_super = GRID // ENTITY_PATCH
    print(f"  Entity-level measurement: {ENTITY_PATCH}x{ENTITY_PATCH} super-cells "
          f"-> {n_super}x{n_super} = {n_super*n_super} super-cells per frame")
    print(f"  Adaptive threshold: top {ACTIVE_PCT:.0f}% of super-cells active "
          f"(H constant across regimes; discrimination via spatial/temporal structure)")
    print(f"  Seeds per param set: {N_SEEDS}")
    print()
    print("  H1: Complex regime (self-replicating spots / worms) scores")
    print(f"      higher C_a than trivial or chaotic regimes.  "
          f"(d>{D_THRESH}, p<{P_THRESH})")
    print("  H0: No systematic C_a difference across behaviour classes.")
    print("=" * 70)

    rows       = []
    Ca_summary = {}   # name → {mean, std, Ca_list, v_final, densities}

    for name, f, k, desc, expected in PARAMETER_SETS:
        print(f"\n  [{name}]  f={f}, k={k}  ({desc})")
        Ca_list   = []
        all_dens  = []
        v_snap    = None

        for seed in range(N_SEEDS):
            vols, v_final, dens_fix, n_super = run_gray_scott(f, k, seed=seed)
            r = compute_full_C(vols)
            Ca_list.append(r["C_a"])
            all_dens.extend(dens_fix.tolist())
            if v_snap is None:
                v_snap = v_final
            rows.append({
                "name": name, "f": f, "k": k, "seed": seed,
                "expected_class": expected,
                "n_supercells":     n_super * n_super,
                "cell_density_v01": float(dens_fix.mean()),
                **r,
            })
            print(f"    seed {seed}: C_a={r['C_a']:.4f}  "
                  f"mi1={r['mi1']:.4f}  tc={r['tc_mean']:.4f}  "
                  f"H={r['mean_H']:.4f}  cell_dens={dens_fix.mean():.3f}")

        Ca_summary[name] = {
            "Ca_list":  Ca_list,
            "mean":     np.mean(Ca_list),
            "std":      np.std(Ca_list),
            "expected": expected,
            "f": f, "k": k,
            "v_final":  v_snap,
            "density":  np.mean(all_dens),
        }
        print(f"    -> mean C_a = {np.mean(Ca_list):.4f} ± {np.std(Ca_list):.4f}  "
              f"mean density = {np.mean(all_dens):.3f}")

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------
    complex_names  = [n for n, *_, ex in PARAMETER_SETS if ex == "complex"]
    other_names    = [n for n, *_, ex in PARAMETER_SETS if ex != "complex"]

    complex_Ca = [c for n in complex_names  for c in Ca_summary[n]["Ca_list"]]
    other_Ca   = [c for n in other_names    for c in Ca_summary[n]["Ca_list"]]

    d, p = None, None
    if complex_Ca and other_Ca:
        pooled_std = np.std(other_Ca + complex_Ca)
        d = (np.mean(complex_Ca) - np.mean(other_Ca)) / max(pooled_std, 1e-9)
        _, p = sp.mannwhitneyu(complex_Ca, other_Ca, alternative="greater")

    print("\n" + "-" * 70)
    print("Results by parameter set (sorted by C_a):")
    for nm, info in sorted(Ca_summary.items(), key=lambda x: -x[1]["mean"]):
        star = " <-- complex" if info["expected"] == "complex" else ""
        print(f"  {nm:20s}  [{info['expected']:8s}]  "
              f"C_a = {info['mean']:.4f} ± {info['std']:.4f}{star}")

    confirmed = False
    if d is not None:
        confirmed = d > D_THRESH and p < P_THRESH
        verdict = ("H1 CONFIRMED    [d>{:.1f}, p<{:.2f}]".format(D_THRESH, P_THRESH)
                   if confirmed else
                   "H0 NOT REJECTED [effect too small or not significant]")
        print(f"\n  Complex vs. Other: Cohen d = {d:.3f},  p = {p:.4f}")
        print(f"  RESULT: {verdict}")
    else:
        verdict = "insufficient data"

    # -------------------------------------------------------------------------
    # CSV output
    # -------------------------------------------------------------------------
    csv_path = os.path.join(_HERE, "gray_scott_results.csv")
    fields   = ["name", "f", "k", "seed", "expected_class",
                "n_supercells", "cell_density_v01",
                "C", "C_a",
                "wH_a", "wOPs_a", "wOPt_a", "wT_a", "wG_a",
                "mean_H", "std_H", "op_up", "op_down",
                "mi1", "decay", "tc_mean", "gzip_ratio"]
    with open(csv_path, "w", newline="") as f_csv:
        w = _csv.DictWriter(f_csv, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"\n  CSV -> {csv_path}")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 12))

    # -- Row 1: final v-field snapshots for each parameter set ----------------
    n_params = len(PARAMETER_SETS)
    for i, (name, f_val, k_val, desc, expected) in enumerate(PARAMETER_SETS):
        ax = fig.add_subplot(3, n_params, i + 1)
        v_img = Ca_summary[name]["v_final"]
        ax.imshow(v_img, cmap="inferno", vmin=0, vmax=0.4, origin="lower")
        cls_color = {"trivial": "grey", "ordered": "steelblue",
                     "complex": "green", "chaotic": "red"}[expected]
        ca_mean = Ca_summary[name]["mean"]
        ax.set_title(f"{name}\nf={f_val} k={k_val}\nC_a={ca_mean:.4f}",
                     fontsize=7.5, color=cls_color, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines["bottom"].set_color(cls_color)
        ax.spines["top"].set_color(cls_color)
        ax.spines["left"].set_color(cls_color)
        ax.spines["right"].set_color(cls_color)
        for sp_name in ["bottom", "top", "left", "right"]:
            ax.spines[sp_name].set_linewidth(2.5)

    # -- Row 2: C_a bar chart -------------------------------------------------
    ax_bar = fig.add_subplot(3, 1, 2)
    names_sorted  = sorted(Ca_summary.keys(), key=lambda n: -Ca_summary[n]["mean"])
    means_sorted  = [Ca_summary[n]["mean"] for n in names_sorted]
    stds_sorted   = [Ca_summary[n]["std"]  for n in names_sorted]
    cls_colors    = [{"trivial": "lightgrey", "ordered": "steelblue",
                      "complex": "mediumseagreen", "chaotic": "tomato"}
                     [Ca_summary[n]["expected"]] for n in names_sorted]
    xb = np.arange(len(names_sorted))
    ax_bar.bar(xb, means_sorted, color=cls_colors, edgecolor="k",
               linewidth=0.8, alpha=0.9)
    ax_bar.errorbar(xb, means_sorted, yerr=stds_sorted,
                    fmt="none", color="black", capsize=4)
    ax_bar.set_xticks(xb)
    ax_bar.set_xticklabels(names_sorted, rotation=15, ha="right", fontsize=9)
    ax_bar.set_ylabel("Agnostic C_a  (mean ± std)")
    ax_bar.set_title(f"C_a by Parameter Set\n{verdict}", fontsize=10,
                     fontweight="bold")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="lightgrey",      edgecolor="k", label="trivial"),
        Patch(facecolor="steelblue",      edgecolor="k", label="ordered"),
        Patch(facecolor="mediumseagreen", edgecolor="k", label="complex"),
        Patch(facecolor="tomato",         edgecolor="k", label="chaotic"),
    ]
    ax_bar.legend(handles=legend_elements, fontsize=8, loc="upper right")
    if d is not None:
        ax_bar.text(0.02, 0.95,
                    f"Complex vs. other: d={d:.3f}, p={p:.4f}",
                    transform=ax_bar.transAxes, va="top", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.85))

    # -- Row 3: sub-metric breakdown ------------------------------------------
    ax_sub = fig.add_subplot(3, 1, 3)
    sub_keys   = ["mean_H", "mi1", "tc_mean", "gzip_ratio", "op_down"]
    sub_labels = ["Spatial H", "Temp. MI (lag 1)", "Temp. compress.",
                  "Gzip ratio", "Opacity ↓"]
    xs = np.arange(len(sub_keys))
    width = 0.8 / n_params
    cmap_cls  = {"trivial": "lightgrey", "ordered": "steelblue",
                 "complex": "mediumseagreen", "chaotic": "tomato"}
    for i, (name, *_, expected) in enumerate(PARAMETER_SETS):
        vals = []
        for k_m in sub_keys:
            param_rows = [r for r in rows if r["name"] == name]
            vals.append(np.mean([r[k_m] for r in param_rows]))
        offset = (i - n_params / 2 + 0.5) * width
        ax_sub.bar(xs + offset, vals, width * 0.9,
                   label=name, color=cmap_cls[expected],
                   edgecolor="k", linewidth=0.5, alpha=0.85)
    ax_sub.set_xticks(xs)
    ax_sub.set_xticklabels(sub_labels, fontsize=9)
    ax_sub.set_ylabel("Metric value")
    ax_sub.set_title("Sub-metric breakdown by parameter set")
    ax_sub.legend(fontsize=7.5, ncol=n_params, loc="upper right")

    fig.suptitle("Gray-Scott Reaction-Diffusion — Agnostic Complexity (C_a)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    png_path = os.path.join(_HERE, "gray_scott_results.png")
    fig.savefig(png_path, dpi=150)
    print(f"  Plot -> {png_path}")
    plt.close(fig)

    return confirmed, Ca_summary, d, p


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print(f"Statistical thresholds: Cohen d > {D_THRESH},  p < {P_THRESH}")
    n_super = GRID // ENTITY_PATCH
    print(f"Entity scale: {ENTITY_PATCH}x{ENTITY_PATCH} -> {n_super}x{n_super}={n_super**2} super-cells  |  "
          f"Grid: {GRID}²  |  T={N_FRAMES} frames")

    confirmed, summary, d, p = run_experiment()

    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  H1 (complex > other): {'CONFIRMED' if confirmed else 'NOT REJECTED'}")
    if d is not None:
        print(f"  Cohen d = {d:.3f},  p = {p:.4f}")
    print()
    print("  Parameter set rankings:")
    for nm, info in sorted(summary.items(), key=lambda x: -x[1]["mean"]):
        print(f"    {nm:20s}  [{info['expected']:8s}]  "
              f"C_a = {info['mean']:.4f}  density = {info['density']:.3f}")
    print("=" * 70)
