"""
ca-multiscale.py
================
Applies multi-scale complexity analysis to the canonical early experiments:
  1. 1D Elementary Cellular Automata (Wolfram classes 1-4)
  2. 2D Conway's Game of Life (and variants)

Hypothesis
----------
The AUC of the C_a profile (sum of C_a across spatial pooling factors) is a
single, mathematically grounded complexity number that:

  (a) Produces the SAME ordering as the original single-scale C metric:
          Class 4 (complex) > Class 3 (chaotic) > Class 2 (periodic) > Class 1 (trivial)

  (b) Gives this ordering because:
      - Class 3 (pseudo-random): C_a is high at fine scale but drops as pooling
        averages out the randomness → LOW AUC despite high single-scale entropy.
      - Class 4 (edge-of-chaos): C_a stays elevated at entity scale because
        genuine multi-scale structure (gliders, interacting patterns) persists
        under coarsening → HIGH AUC.

AUC  =  sum of C_a(s) across s in POOL_FACTORS
      =  discrete integral of complexity over the coarsening axis
      =  total weight under the RG flow trajectory

This is the single number that should be used as the improved C metric for
all future experiments, replacing or augmenting the single-scale C_a.
"""

import os
import sys
import csv as _csv
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats as sp

# ===========================================================================
# Bootstrap — import compute_full_C from mnist-experiment.py
# ===========================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_nn   = os.path.join(os.path.dirname(_HERE), "neural-network")
_spec = importlib.util.spec_from_file_location(
            "mnist_exp", os.path.join(_nn, "mnist-experiment.py"))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_full_C = _mod.compute_full_C

# ===========================================================================
# Configuration
# ===========================================================================
ECA_N_CELLS   = 256
ECA_N_STEPS   = 500
ECA_N_SEEDS   = 3
GOL_GRID      = 64
GOL_N_STEPS   = 200
GOL_N_SEEDS   = 3
POOL_FACTORS  = [1, 2, 4, 8, 16]
ACTIVE_PCT    = 25.0     # adaptive threshold at coarse scales
MIN_W_CELLS   = 8        # skip pooling if n_supercells falls below this

# ECA rules spanning Wolfram Classes 1-4
# Each tuple: (rule_number, wolfram_class, description)
ECA_RULES = [
    (  0, 1, "Class 1 — all zeros"),
    (  8, 1, "Class 1 — sparse fixed"),
    ( 51, 2, "Class 2 — complement"),
    ( 23, 2, "Class 2 — periodic blocks"),
    ( 30, 3, "Class 3 — pseudo-random"),
    ( 90, 3, "Class 3 — Sierpinski fractal"),
    (110, 4, "Class 4 — universal computation"),
    ( 54, 4, "Class 4 — complex / gliders"),
]

# GoL-like rules (B/S notation → (birth_counts, survival_counts))
GOL_RULES = [
    ("GoL B3/S23",   frozenset([3]),    frozenset([2,3]),  4, "Conway's Game of Life"),
    ("GoL B36/S23",  frozenset([3,6]),  frozenset([2,3]),  4, "HighLife (gliders + replicators)"),
    ("GoL B1/S1",    frozenset([1]),    frozenset([1]),    2, "Class-2 periodic"),
    ("GoL B3/S12345",frozenset([3]),    frozenset([1,2,3,4,5]), 3, "Maze / Class-3 like"),
    ("GoL B2/S",     frozenset([2]),    frozenset([]),     1, "Class-1 (all die quickly)"),
]

CLASS_COLOR = {1: "dimgray", 2: "steelblue", 3: "tomato", 4: "mediumseagreen"}

# ===========================================================================
# ECA simulation
# ===========================================================================

def eca_step(cells, rule_bits):
    left  = np.roll(cells, 1)
    right = np.roll(cells, -1)
    idx   = (left * 4 + cells * 2 + right).astype(np.intp)
    return rule_bits[idx]


def run_eca(rule, n_cells=ECA_N_CELLS, n_steps=ECA_N_STEPS, seed=0):
    """Return (n_steps, n_cells) binary uint8 spacetime grid."""
    rng       = np.random.RandomState(seed)
    cells     = rng.randint(0, 2, n_cells).astype(np.uint8)
    rule_bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)
    grid      = np.empty((n_steps, n_cells), dtype=np.uint8)
    for t in range(n_steps):
        grid[t] = cells
        cells   = eca_step(cells, rule_bits)
    return grid


def eca_to_volumes(grid, pool_factor, active_pct=ACTIVE_PCT):
    """
    Convert (T, W) binary ECA spacetime grid to list of (1, 1, W') volumes.

    pool_factor=1 : pass raw binary cells unchanged.
    pool_factor>1 : mean-pool then apply adaptive threshold to maintain
                    consistent super-cell density across all rules.
    """
    T, W  = grid.shape
    n_sup = W // pool_factor
    if n_sup < MIN_W_CELLS:
        return None

    volumes = []
    if pool_factor == 1:
        for t in range(T):
            volumes.append(grid[t].astype(np.float32).reshape(1, 1, -1))
    else:
        for t in range(T):
            row = grid[t].astype(np.float32).reshape(n_sup, pool_factor).mean(axis=1)
            cut    = np.percentile(row, 100.0 - active_pct)
            binary = (row > cut).astype(np.float32)
            volumes.append(binary.reshape(1, 1, -1))
    return volumes


# ===========================================================================
# GoL simulation
# ===========================================================================

def gol_neighbors(grid):
    """Count live neighbours for each cell (periodic boundary)."""
    return sum(
        np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
        for dy in (-1, 0, 1) for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    )


def gol_step(grid, birth, survive):
    nb = gol_neighbors(grid)
    new = np.zeros_like(grid)
    for b in birth:
        new |= (grid == 0) & (nb == b)
    for s in survive:
        new |= (grid == 1) & (nb == s)
    return new.astype(np.uint8)


def run_gol(birth, survive, n=GOL_GRID, n_steps=GOL_N_STEPS,
            seed=0, density=0.35):
    """Return list of n_steps (n, n) binary uint8 frames."""
    rng    = np.random.RandomState(seed)
    grid   = (rng.rand(n, n) < density).astype(np.uint8)
    frames = []
    for _ in range(n_steps):
        grid = gol_step(grid, birth, survive)
        frames.append(grid.copy())
    return frames


def gol_to_volumes(frames, pool_factor, active_pct=ACTIVE_PCT):
    """
    Convert list of (N, N) binary frames to (1, 1, N') volumes.
    Same adaptive-threshold logic as Gray-Scott.
    """
    n     = frames[0].shape[0]
    n_sup = n // pool_factor
    if n_sup < MIN_W_CELLS:
        return None

    volumes = []
    if pool_factor == 1:
        for f in frames:
            volumes.append(f.astype(np.float32).reshape(1, 1, -1))
    else:
        for f in frames:
            blocks = f.astype(np.float32).reshape(
                n_sup, pool_factor, n_sup, pool_factor).mean(axis=(1, 3))
            cut    = np.percentile(blocks, 100.0 - active_pct)
            binary = (blocks > cut).astype(np.float32)
            volumes.append(binary.reshape(1, 1, -1))
    return volumes


# ===========================================================================
# Multi-scale metric
# ===========================================================================

def multiscale_Ca(vol_fn, pool_factors=POOL_FACTORS):
    """
    Given a volume-builder function vol_fn(pool_factor) -> volumes or None,
    compute C_a at each pool factor and return the profile dict + AUC.
    """
    profile = {}
    for pf in pool_factors:
        vols = vol_fn(pf)
        if vols is None:
            continue
        r = compute_full_C(vols)
        profile[pf] = r
    if not profile:
        return {}, 0.0
    auc = sum(profile[pf]["C_a"] for pf in profile)
    return profile, auc


def beta_from_profile(profile, pool_factors=POOL_FACTORS):
    """First differences of C_a in log2(pool_factor) space."""
    pfs  = [p for p in pool_factors if p in profile]
    vals = [profile[p]["C_a"] for p in pfs]
    return [vals[i+1] - vals[i] for i in range(len(vals)-1)], pfs


def entity_scale(beta_vals, pool_factors_used):
    """Return pool factor just before first negative beta, or None."""
    for i, b in enumerate(beta_vals):
        if b < 0:
            return pool_factors_used[i]
    return None


# ===========================================================================
# Run experiments
# ===========================================================================

def run_eca_experiment():
    print("\n" + "="*72)
    print("ECA Multi-Scale Experiment")
    print(f"  {ECA_N_CELLS} cells  x  {ECA_N_STEPS} steps  |  "
          f"{ECA_N_SEEDS} seeds  |  pool factors {POOL_FACTORS}")
    print("="*72)

    results = []
    csv_rows = []

    for rule, wclass, desc in ECA_RULES:
        print(f"\n  Rule {rule:3d}  [Class {wclass}]  {desc}")
        seed_aucs = []
        seed_profiles = []

        for seed in range(ECA_N_SEEDS):
            grid = run_eca(rule, seed=seed)

            def vol_fn(pf, g=grid):
                return eca_to_volumes(g, pf)

            profile, auc = multiscale_Ca(vol_fn)
            seed_aucs.append(auc)
            seed_profiles.append(profile)

            for pf, r in profile.items():
                csv_rows.append({
                    "experiment": "ECA",
                    "name": f"Rule_{rule}",
                    "class": wclass,
                    "seed": seed,
                    "pool_factor": pf,
                    "C_a": r["C_a"],
                    "mi1": r["mi1"],
                    "tc_mean": r["tc_mean"],
                    "gzip_ratio": r["gzip_ratio"],
                    "op_up": r["op_up"],
                    "op_down": r["op_down"],
                    "mean_H": r["mean_H"],
                })

            line = "    seed {}: ".format(seed) + "  ".join(
                "x{} {:.3f}".format(pf, profile[pf]["C_a"])
                for pf in POOL_FACTORS if pf in profile)
            print(line + f"  | AUC={auc:.3f}")

        mean_auc = float(np.mean(seed_aucs))
        std_auc  = float(np.std(seed_aucs))

        # Average profile across seeds
        avg_profile = {}
        for pf in POOL_FACTORS:
            vals = [p[pf]["C_a"] for p in seed_profiles if pf in p]
            if vals:
                avg_profile[pf] = np.mean(vals)

        betas, pfs_used = beta_from_profile(
            {pf: {"C_a": avg_profile[pf]} for pf in avg_profile})
        es = entity_scale(betas, pfs_used)

        print(f"    mean AUC = {mean_auc:.4f} +/- {std_auc:.4f}  "
              f"entity_scale={('x'+str(es)) if es else 'none'}")

        results.append({
            "name": f"Rule_{rule}",
            "rule": rule,
            "class": wclass,
            "desc": desc,
            "mean_auc": mean_auc,
            "std_auc":  std_auc,
            "avg_profile": avg_profile,
            "betas": betas,
            "entity_scale": es,
            "Ca_s1": avg_profile.get(1, 0.0),
        })

    return results, csv_rows


def run_gol_experiment():
    print("\n" + "="*72)
    print("GoL Multi-Scale Experiment")
    print(f"  {GOL_GRID}x{GOL_GRID} grid  x  {GOL_N_STEPS} steps  |  "
          f"{GOL_N_SEEDS} seeds  |  pool factors {POOL_FACTORS}")
    print("="*72)

    results  = []
    csv_rows = []

    for name, birth, survive, wclass, desc in GOL_RULES:
        print(f"\n  {name}  [Class {wclass}]  {desc}")
        seed_aucs    = []
        seed_profiles = []

        for seed in range(GOL_N_SEEDS):
            frames = run_gol(birth, survive, seed=seed)
            # Skip if grid goes dead early
            if sum(f.sum() for f in frames[-10:]) == 0:
                print(f"    seed {seed}: grid died — skipping")
                continue

            def vol_fn(pf, fr=frames):
                return gol_to_volumes(fr, pf)

            profile, auc = multiscale_Ca(vol_fn)
            seed_aucs.append(auc)
            seed_profiles.append(profile)

            for pf, r in profile.items():
                csv_rows.append({
                    "experiment": "GoL",
                    "name": name,
                    "class": wclass,
                    "seed": seed,
                    "pool_factor": pf,
                    "C_a": r["C_a"],
                    "mi1": r["mi1"],
                    "tc_mean": r["tc_mean"],
                    "gzip_ratio": r["gzip_ratio"],
                    "op_up": r["op_up"],
                    "op_down": r["op_down"],
                    "mean_H": r["mean_H"],
                })

            line = "    seed {}: ".format(seed) + "  ".join(
                "x{} {:.3f}".format(pf, profile[pf]["C_a"])
                for pf in POOL_FACTORS if pf in profile)
            print(line + f"  | AUC={auc:.3f}")

        if not seed_aucs:
            continue
        mean_auc = float(np.mean(seed_aucs))
        std_auc  = float(np.std(seed_aucs))

        avg_profile = {}
        for pf in POOL_FACTORS:
            vals = [p[pf]["C_a"] for p in seed_profiles if pf in p]
            if vals:
                avg_profile[pf] = np.mean(vals)

        betas, pfs_used = beta_from_profile(
            {pf: {"C_a": avg_profile[pf]} for pf in avg_profile})
        es = entity_scale(betas, pfs_used)

        print(f"    mean AUC = {mean_auc:.4f} +/- {std_auc:.4f}  "
              f"entity_scale={('x'+str(es)) if es else 'none'}")

        results.append({
            "name": name,
            "class": wclass,
            "desc": desc,
            "mean_auc": mean_auc,
            "std_auc":  std_auc,
            "avg_profile": avg_profile,
            "betas": betas,
            "entity_scale": es,
            "Ca_s1": avg_profile.get(1, 0.0),
        })

    return results, csv_rows


# ===========================================================================
# Visualisation
# ===========================================================================

def plot_experiment(eca_results, gol_results, output_path):
    """
    Layout
    ------
    Row 0 col 0-1: ECA multi-scale profiles (line chart per rule)
    Row 0 col 2:   ECA AUC bar chart
    Row 1 col 0-1: GoL multi-scale profiles
    Row 1 col 2:   GoL AUC bar chart
    """
    fig = plt.figure(figsize=(22, 14))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.30)

    def _draw_profiles(ax, results, title):
        for r in results:
            prof  = r["avg_profile"]
            color = CLASS_COLOR[r["class"]]
            x     = [p for p in POOL_FACTORS if p in prof]
            y     = [prof[p] for p in x]
            lbl   = "{} [C{}]".format(r["name"], r["class"])
            ax.plot(x, y, marker="o", markersize=7, linewidth=2.2,
                    color=color, label=lbl)
            # Shade beta<0 segments (entity scale detected)
            betas, pfs_u = beta_from_profile({p: {"C_a": prof[p]} for p in prof})
            for i, b in enumerate(betas):
                if b < 0:
                    pf_l, pf_r = pfs_u[i], pfs_u[i+1]
                    y0, y1 = prof[pf_l], prof[pf_r]
                    ax.fill_betweenx([min(y0,y1), max(y0,y1)],
                                     pf_l, pf_r, alpha=0.14, color=color)

        ax.set_xscale("log", base=2)
        ax.set_xticks(POOL_FACTORS)
        ax.set_xticklabels([f"x{p}" for p in POOL_FACTORS], fontsize=9)
        ax.set_xlabel("Spatial pooling factor  (-> coarser scale)", fontsize=9)
        ax.set_ylabel("C_a", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, loc="best", ncol=2)
        ax.grid(True, alpha=0.22)
        ax.set_ylim(bottom=0)
        ax.text(0.01, 0.97,
                "Shaded = beta<0 region (entity scale detected, complexity "
                "decreasing under coarsening).",
                transform=ax.transAxes, fontsize=7.5, va="top",
                bbox=dict(facecolor="lightyellow", edgecolor="goldenrod",
                          alpha=0.8, boxstyle="round,pad=0.3"))

    def _draw_bars(ax, results, title):
        # Sort by mean_auc descending
        srt = sorted(results, key=lambda r: -r["mean_auc"])
        x   = np.arange(len(srt))
        col = [CLASS_COLOR[r["class"]] for r in srt]
        bars = ax.bar(x, [r["mean_auc"] for r in srt], color=col,
                      edgecolor="k", linewidth=0.7, alpha=0.88)
        ax.errorbar(x, [r["mean_auc"] for r in srt],
                    yerr=[r["std_auc"] for r in srt],
                    fmt="none", color="black", capsize=3.5, linewidth=1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(
            ["{}\n[C{}]".format(r["name"], r["class"]) for r in srt],
            rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("AUC  (sum of C_a across pool factors)", fontsize=8.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.22)

        # Annotate entity scale
        for i, r in enumerate(srt):
            es = r["entity_scale"]
            lbl = f"peak x{es}" if es else "no peak"
            ax.text(i, r["mean_auc"] + r["std_auc"] + 0.001,
                    lbl, ha="center", va="bottom", fontsize=6.5,
                    color=CLASS_COLOR[r["class"]], fontweight="bold")

        from matplotlib.patches import Patch
        handles = [Patch(facecolor=CLASS_COLOR[c], edgecolor="k",
                         label=f"Class {c}") for c in sorted(CLASS_COLOR)]
        ax.legend(handles=handles, fontsize=7.5, loc="upper right")

    # --- ECA ---
    ax_eca_prof = fig.add_subplot(gs[0, :2])
    _draw_profiles(ax_eca_prof, eca_results,
                   "ECA Multi-Scale Profiles — Wolfram Classes 1-4")

    ax_eca_bar = fig.add_subplot(gs[0, 2])
    _draw_bars(ax_eca_bar, eca_results,
               "AUC ranking\n(Class 4 should beat Class 3)")

    # --- GoL ---
    ax_gol_prof = fig.add_subplot(gs[1, :2])
    _draw_profiles(ax_gol_prof, gol_results,
                   "Game-of-Life Variants — Multi-Scale Profiles")

    ax_gol_bar = fig.add_subplot(gs[1, 2])
    _draw_bars(ax_gol_bar, gol_results,
               "AUC ranking\n(B3/S23 GoL should rank highest)")

    fig.suptitle(
        "Multi-Scale Complexity (AUC) — ECA and Game of Life\n"
        "AUC = sum of C_a across spatial pooling factors  |  "
        "H1: Class 4 > Class 3 > Class 2 > Class 1 (same as original C metric)",
        fontsize=12, fontweight="bold", y=1.01)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot -> {output_path}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    eca_results, eca_csv = run_eca_experiment()
    gol_results, gol_csv = run_gol_experiment()

    # -- CSV --
    all_rows = eca_csv + gol_csv
    csv_path = os.path.join(_HERE, "ca_multiscale.csv")
    fields   = ["experiment", "name", "class", "seed", "pool_factor",
                "C_a", "mi1", "tc_mean", "gzip_ratio", "op_up", "op_down", "mean_H"]
    with open(csv_path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in all_rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"  CSV -> {csv_path}")

    # -- Plot --
    png_path = os.path.join(_HERE, "ca_multiscale.png")
    plot_experiment(eca_results, gol_results, png_path)

    # -- Final summary --
    print("\n" + "="*72)
    print("AUC RANKING SUMMARY")
    print("="*72)
    print("\n  ECA (sorted by AUC):")
    for r in sorted(eca_results, key=lambda x: -x["mean_auc"]):
        es = r["entity_scale"]
        print(f"    {r['name']:10s} [Class {r['class']}]  "
              f"AUC={r['mean_auc']:.4f}+/-{r['std_auc']:.4f}  "
              f"C_a(x1)={r['Ca_s1']:.4f}  "
              f"entity={'x'+str(es) if es else 'none':6s}  {r['desc']}")

    print("\n  GoL (sorted by AUC):")
    for r in sorted(gol_results, key=lambda x: -x["mean_auc"]):
        es = r["entity_scale"]
        print(f"    {r['name']:15s} [Class {r['class']}]  "
              f"AUC={r['mean_auc']:.4f}+/-{r['std_auc']:.4f}  "
              f"C_a(x1)={r['Ca_s1']:.4f}  "
              f"entity={'x'+str(es) if es else 'none':6s}  {r['desc']}")

    # -- Test hypothesis --
    print("\n  H1 test (Class 4 > Class 3 in AUC):")
    for results, label in [(eca_results, "ECA"), (gol_results, "GoL")]:
        c4 = [r["mean_auc"] for r in results if r["class"] == 4]
        c3 = [r["mean_auc"] for r in results if r["class"] == 3]
        if c4 and c3:
            h1 = min(c4) > max(c3)
            print(f"    {label}: Class4 min={min(c4):.4f}  Class3 max={max(c3):.4f}  "
                  f"-> H1 {'CONFIRMED' if h1 else 'NOT CONFIRMED'}")

    print("="*72)
    print("\nDone.")
