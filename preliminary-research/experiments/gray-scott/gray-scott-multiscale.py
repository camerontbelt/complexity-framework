"""
gray-scott-multiscale.py
========================
Tests the multi-scale complexity hypothesis on Gray-Scott reaction-diffusion.

Hypothesis: The spatial scale at which C_a peaks is characteristic of
the dynamical regime.  "Truly complex" (edge-of-chaos) systems should
maintain elevated C_a across *multiple* coarsening levels, while trivial
or narrowly ordered systems should peak at one scale or stay flat.

Method
------
For each of 6 parameter sets spanning the Gray-Scott phase diagram, run the
simulation, collect N_FRAMES snapshots of the v-field, then compute C_a at
each spatial pooling factor in POOL_FACTORS.  At each pooling level the grid
is coarsened to (GRID/p × GRID/p) super-cells and an adaptive 25th-percentile
threshold keeps density constant — so wH_a is identical across all scales and
the ONLY signal is spatial/temporal structure at that scale.

Hypotheses
----------
H1: Different dynamical classes (trivial / ordered / complex / chaotic) have
    C_a-vs-scale profiles that peak at measurably different pooling factors,
    with complex regimes showing the broadest elevated region.
H0: C_a profiles are flat or indistinguishable across all regimes.
"""

import os
import sys
import csv as _csv
import importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
# Experiment constants
# ===========================================================================
DU           = 0.16
DV           = 0.08
DT           = 1.0
GRID         = 128         # spatial grid (128 × 128 cells)
N_WARMUP     = 3000        # warm-up steps before measurement
SAMPLE_STEP  = 20          # GS steps between sampled frames
N_FRAMES     = 50          # measurement frames (T for the metric)
N_SEEDS      = 2           # random seeds per parameter set
ACTIVE_PCT   = 25.0        # adaptive threshold: top 25 % of super-cells → active
POOL_FACTORS = [1, 2, 4, 8, 16]   # spatial pooling factors to test

# ===========================================================================
# Parameter sets spanning the Gray-Scott phase diagram
# (f, k) values follow Pearson 1993 / Munafo 1996 classifications.
# ===========================================================================
PARAMETER_SETS = [
    #  name               f       k       description                 expected class
    ("dead",           0.090,  0.059,  "v dies — no reaction",      "trivial"),
    ("static_spots",   0.042,  0.065,  "stable Turing spots",       "ordered"),
    ("self_rep_spots", 0.030,  0.057,  "self-replicating spots",    "complex"),
    ("worm_complex",   0.055,  0.062,  "labyrinthine worms",        "complex"),
    ("solitons",       0.025,  0.060,  "moving solitons",           "complex"),
    ("chaotic",        0.026,  0.051,  "turbulent chaos",           "chaotic"),
]

CLS_COLOR = {
    "trivial": "dimgray",
    "ordered": "steelblue",
    "complex": "mediumseagreen",
    "chaotic": "tomato",
}

# ===========================================================================
# Gray-Scott simulation engine
# ===========================================================================

def laplacian_2d(Z):
    """5-point Laplacian with periodic boundary conditions."""
    return (np.roll(Z,  1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z,  1, axis=1) + np.roll(Z, -1, axis=1) - 4.0 * Z)


def gray_scott_init(n, seed=0):
    rng = np.random.RandomState(seed)
    u   = np.ones((n, n))  + 0.02 * rng.randn(n, n)
    v   = np.zeros((n, n)) + 0.02 * rng.randn(n, n)
    mid, sq = n // 2, n // 8
    u[mid-sq:mid+sq, mid-sq:mid+sq] = 0.50 + 0.02 * rng.randn(2*sq, 2*sq)
    v[mid-sq:mid+sq, mid-sq:mid+sq] = 0.25 + 0.02 * rng.randn(2*sq, 2*sq)
    return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)


def gray_scott_step(u, v, f, k, n_steps=1):
    for _ in range(n_steps):
        uvv   = u * v * v
        u_new = np.clip(u + DT*(DU*laplacian_2d(u) - uvv + f*(1.0-u)), 0.0, 1.0)
        v_new = np.clip(v + DT*(DV*laplacian_2d(v) + uvv - (f+k)*v),   0.0, 1.0)
        u, v  = u_new, v_new
    return u, v


def simulate(f, k, seed=0):
    """
    Run Gray-Scott warm-up then collect N_FRAMES raw v-field snapshots.

    Returns
    -------
    frames  : list of N_FRAMES  (GRID, GRID) float64 arrays
    v_final : (GRID, GRID) float64 — final state for visualisation
    """
    u, v = gray_scott_init(GRID, seed)
    u, v = gray_scott_step(u, v, f, k, n_steps=N_WARMUP)

    frames = []
    for _ in range(N_FRAMES):
        u, v = gray_scott_step(u, v, f, k, n_steps=SAMPLE_STEP)
        frames.append(v.copy())

    return frames, v


# ===========================================================================
# Multi-scale binarisation
# ===========================================================================

def frames_to_volumes(frames, pool_factor, active_pct=ACTIVE_PCT):
    """
    Coarsen each v-field by pool_factor then apply an adaptive threshold.

    Steps
    -----
    1. Mean-pool the (GRID, GRID) frame into (n_super, n_super) blocks.
    2. Apply per-frame adaptive threshold: top active_pct % → binary 1.
       This fixes density exactly at active_pct/100 every frame, so
       wH_a is constant across ALL pooling levels.  The ONLY things that
       change with scale are spatial/temporal structure.
    3. Return list of (1, 1, n_super²) float32 volumes for compute_full_C.

    Returns None if the coarsened grid is too small (n_super < 4).
    """
    n      = frames[0].shape[0]   # GRID
    n_s    = n // pool_factor      # n_super × n_super super-cells

    if n_s < 4:
        return None

    volumes = []
    for v in frames:
        if pool_factor == 1:
            blocks = v
        else:
            blocks = v.reshape(n_s, pool_factor, n_s, pool_factor).mean(axis=(1, 3))
        cut    = np.percentile(blocks, 100.0 - active_pct)
        binary = (blocks > cut).astype(np.float32)
        volumes.append(binary.reshape(1, 1, -1))   # (1, 1, n_s²)
    return volumes


# ===========================================================================
# Main experiment loop
# ===========================================================================

def run_experiment():
    print("\n" + "="*72)
    print("Gray-Scott Multi-Scale Complexity  — running")
    print(f"  Grid {GRID}²  |  warm-up {N_WARMUP}  |  {N_FRAMES} frames × {SAMPLE_STEP} steps")
    print(f"  Pool factors: {POOL_FACTORS}  |  {N_SEEDS} seeds  |  adaptive {ACTIVE_PCT:.0f}% threshold")
    print("="*72)

    results  = {}    # name → profile data
    csv_rows = []

    for name, f, k, desc, expected in PARAMETER_SETS:
        print(f"\n  [{name}]  f={f}  k={k}  ({desc})")

        Ca_by_pf   = {p: [] for p in POOL_FACTORS}
        subs_by_pf = {p: [] for p in POOL_FACTORS}
        v_snap     = None

        for seed in range(N_SEEDS):
            frames, v_final = simulate(f, k, seed=seed)
            if v_snap is None:
                v_snap = v_final

            for pf in POOL_FACTORS:
                vols = frames_to_volumes(frames, pf)
                if vols is None:
                    continue
                r = compute_full_C(vols)
                Ca_by_pf[pf].append(r["C_a"])
                subs_by_pf[pf].append(r)
                csv_rows.append({
                    "name": name, "f": f, "k": k, "seed": seed,
                    "expected_class": expected,
                    "pool_factor": pf,
                    "n_supercells": (GRID // pf) ** 2,
                    **{mk: r[mk] for mk in [
                        "C_a", "mi1", "decay", "tc_mean", "gzip_ratio",
                        "op_up", "op_down", "mean_H", "std_H",
                        "wH_a", "wOPs_a", "wOPt_a", "wT_a", "wG_a"]},
                })

            status = "  seed {}: ".format(seed) + "   ".join(
                "×{} {:.3f}".format(pf, Ca_by_pf[pf][-1])
                for pf in POOL_FACTORS if Ca_by_pf[pf]
            )
            print(status)

        Ca_profile  = {pf: {"mean": float(np.mean(Ca_by_pf[pf])),
                            "std":  float(np.std(Ca_by_pf[pf]))}
                       for pf in POOL_FACTORS if Ca_by_pf[pf]}
        sub_profile = {pf: {mk: float(np.mean([r[mk] for r in subs_by_pf[pf]]))
                            for mk in ["mi1", "tc_mean", "gzip_ratio", "op_down", "mean_H"]}
                       for pf in POOL_FACTORS if subs_by_pf[pf]}

        peak_pf = max(Ca_profile, key=lambda p: Ca_profile[p]["mean"])
        print(f"    peak C_a = {Ca_profile[peak_pf]['mean']:.4f}  at  ×{peak_pf}")

        results[name] = {
            "expected":    expected, "f": f, "k": k, "desc": desc,
            "v_snap":      v_snap,
            "Ca_profile":  Ca_profile,
            "sub_profile": sub_profile,
        }

    # ---- CSV output ----
    csv_path = os.path.join(_HERE, "gray_scott_multiscale.csv")
    fields   = [
        "name", "f", "k", "seed", "expected_class", "pool_factor", "n_supercells",
        "C_a", "mi1", "decay", "tc_mean", "gzip_ratio", "op_up", "op_down",
        "mean_H", "std_H", "wH_a", "wOPs_a", "wOPt_a", "wT_a", "wG_a",
    ]
    with open(csv_path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in csv_rows:
            w.writerow({k2: (round(v2, 6) if isinstance(v2, float) else v2)
                        for k2, v2 in row.items()})
    print(f"\n  CSV  ->  {csv_path}")

    return results


# ===========================================================================
# Visualisation
# ===========================================================================

def plot_results(results, output_path):
    """
    Layout
    ------
    Row 0 (full width): combined overview — all 6 regimes, C_a vs. scale.
    Rows 1–2 (2 × 3): individual regime panels, each with:
        • thick coloured line = C_a ± std
        • dashed purple = MI₁ (temporal correlation, scaled)
        • dashed orange = tc_mean (temporal compression, scaled)
        • star = peak scale
        • inset = v-field thumbnail
    """
    fig = plt.figure(figsize=(22, 19))
    gs  = GridSpec(3, 3, figure=fig,
                   height_ratios=[1.1, 1, 1],
                   hspace=0.45, wspace=0.32)

    ax_comb = fig.add_subplot(gs[0, :])

    # ---- Combined overview ----
    for name, f, k, desc, expected in PARAMETER_SETS:
        info  = results.get(name)
        if info is None:
            continue
        prof  = info["Ca_profile"]
        color = CLS_COLOR[expected]
        x     = [p for p in POOL_FACTORS if p in prof]
        y     = [prof[p]["mean"] for p in x]
        e     = [prof[p]["std"]  for p in x]
        x_lbl = [(GRID // p) ** 2 for p in x]

        ax_comb.plot(x, y, marker="o", markersize=8, linewidth=2.5,
                     color=color, label=f"{name}  [{expected}]")
        ax_comb.fill_between(x,
                             [m - s for m, s in zip(y, e)],
                             [m + s for m, s in zip(y, e)],
                             alpha=0.13, color=color)

    ax_comb.set_xscale("log", base=2)
    ax_comb.set_xticks(POOL_FACTORS)
    ax_comb.set_xticklabels(
        [f"×{p}\n({(GRID//p)**2:,} super-cells)" for p in POOL_FACTORS],
        fontsize=9)
    ax_comb.set_xlabel("Spatial pooling factor  (→ coarser scale,  fewer, larger super-cells)",
                       fontsize=10)
    ax_comb.set_ylabel("C_a  (agnostic complexity)", fontsize=10)
    ax_comb.set_title("Multi-Scale Complexity Profile — All Gray-Scott Regimes",
                      fontsize=13, fontweight="bold")
    ax_comb.legend(fontsize=9, ncol=3, loc="upper center",
                   bbox_to_anchor=(0.5, -0.18), framealpha=0.9)
    ax_comb.grid(True, alpha=0.25)
    ax_comb.text(
        0.01, 0.97,
        "Each line = one GS parameter set.  "
        "If complexity is scale-invariant, the line stays elevated across all ×-factors.  "
        "Scale-specific structure shows a clear peak.",
        transform=ax_comb.transAxes, fontsize=8.5, va="top",
        bbox=dict(facecolor="lightyellow", edgecolor="goldenrod",
                  alpha=0.85, boxstyle="round,pad=0.4"))

    # ---- Individual panels ----
    for i, (name, f, k, desc, expected) in enumerate(PARAMETER_SETS):
        row = 1 + i // 3
        col = i % 3
        ax  = fig.add_subplot(gs[row, col])
        info  = results.get(name)
        color = CLS_COLOR[expected]

        if info is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        prof     = info["Ca_profile"]
        sub_prof = info["sub_profile"]

        x = [p for p in POOL_FACTORS if p in prof]
        y = [prof[p]["mean"] for p in x]
        e = [prof[p]["std"]  for p in x]

        # Main C_a line
        ax.plot(x, y, marker="o", markersize=8, linewidth=2.5,
                color=color, label="C_a", zorder=3)
        ax.fill_between(x,
                        [m - s for m, s in zip(y, e)],
                        [m + s for m, s in zip(y, e)],
                        alpha=0.2, color=color)

        # Peak marker
        if y:
            pi   = y.index(max(y))
            ax.plot(x[pi], y[pi], "*", markersize=14, color=color, zorder=4,
                    markeredgecolor="k", markeredgewidth=0.8)
            ax.axvline(x[pi], color=color, linestyle=":", alpha=0.45, linewidth=1.2)

        # Sub-metrics (scaled to C_a range for overlay readability)
        ca_max = max(y) if y else 1e-9
        for mk, mc, ml in [("mi1",    "mediumpurple", "MI₁"),
                            ("tc_mean","darkorange",   "tc")]:
            sv = [sub_prof[p][mk] for p in x if p in sub_prof]
            if not sv:
                continue
            sv_max = max(sv) or 1e-9
            sv_s   = [v * ca_max / sv_max for v in sv]
            ax.plot(x, sv_s, linestyle="--", linewidth=1.3,
                    color=mc, alpha=0.75, label=f"{ml} (norm.)")

        ax.set_xscale("log", base=2)
        ax.set_xticks(POOL_FACTORS)
        ax.set_xticklabels([f"×{p}" for p in POOL_FACTORS], fontsize=8)
        ax.set_xlabel("Pool factor", fontsize=8)
        ax.set_ylabel("C_a", fontsize=8)
        ax.set_ylim(bottom=0)
        title_str = (f"{name}   f={f}, k={k}\n"
                     f"{desc}   [{expected}]")
        ax.set_title(title_str, fontsize=8.5, color=color, fontweight="bold", pad=4)
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.8)
        ax.grid(True, alpha=0.2)

        # v-field thumbnail (inset — upper left corner)
        v_snap = info.get("v_snap")
        if v_snap is not None:
            ax_in = inset_axes(ax, width="30%", height="38%", loc="upper left",
                               bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
                               bbox_transform=ax.transAxes,
                               borderpad=0.6)
            ax_in.imshow(v_snap, cmap="inferno", vmin=0.0, vmax=0.4,
                         origin="lower", interpolation="nearest")
            ax_in.set_xticks([])
            ax_in.set_yticks([])
            for sp_obj in ax_in.spines.values():
                sp_obj.set_edgecolor(color)
                sp_obj.set_linewidth(1.8)

    # Super-title
    fig.suptitle(
        "Gray-Scott Multi-Scale Complexity Analysis\n"
        f"Grid {GRID}²  ·  {N_FRAMES} frames × {SAMPLE_STEP} steps  ·  "
        f"{N_SEEDS} seeds  ·  adaptive {ACTIVE_PCT:.0f}% threshold  ·  "
        f"pool factors {POOL_FACTORS}",
        fontsize=12, fontweight="bold", y=1.005)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot ->  {output_path}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    results  = run_experiment()
    png_path = os.path.join(_HERE, "gray_scott_multiscale.png")
    plot_results(results, png_path)

    # Brief summary
    print("\n" + "="*72)
    print("MULTI-SCALE SUMMARY  (peak pooling factor per regime)")
    print("="*72)
    for name, f, k, desc, expected in PARAMETER_SETS:
        info = results.get(name)
        if info is None:
            continue
        prof    = info["Ca_profile"]
        peak_pf = max(prof, key=lambda p: prof[p]["mean"])
        peak_Ca = prof[peak_pf]["mean"]
        # Compute "breadth": number of factors within 80 % of peak
        broad   = sum(1 for pf in prof
                      if prof[pf]["mean"] >= 0.80 * peak_Ca)
        color   = expected
        print(f"  {name:18s} [{expected:8s}]  "
              f"peak x{peak_pf}  C_a={peak_Ca:.4f}  "
              f"breadth={broad}/{len(prof)} scales >= 80% of peak")
    print("="*72)
    print("\nDone.")
