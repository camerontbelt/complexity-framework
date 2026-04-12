"""
offset_analysis.py
==================
POST-HOC comparison of blind sweep peak locations against literature
critical temperatures.  This script is run AFTER blind_sweep.py has
finished and produced blind_sweep_peaks.csv.

The separation is deliberate: the measurement (blind_sweep.py) contains
no physics knowledge; the comparison (this file) brings it in after the fact.

Literature values
-----------------
  Ising (q=2):  T_c = 2 / ln(1 + sqrt(2))  ≈ 2.2692
  Potts q=3:    T_c = 1 / ln(1 + sqrt(3))   ≈ 0.9950
  Potts q=5:    T_c = 1 / ln(1 + sqrt(5))   ≈ 0.8515

  Ising and Potts q=3 are second-order (continuous) transitions.
  Potts q=5 is a FIRST-ORDER (discontinuous) transition in 2D.

Existing offset data (from lattice-experiments / fss-experiment)
----------------------------------------------------------------
  Potts q=4 (L=64):        +26.3%  (above T_c)
  Potts q=4 FSS L=32:      +43.8%
  Potts q=4 FSS L=64:      +26.3%
  Potts q=4 FSS L=128:     +41.8%
  Contact Process (L=64):  +20.0%
  Directed Percolation:    +21.8%
  Vicsek Flocking:         -50.0%  (below eta_c — opposite phase)

  "Above T_c" mean: +35.3%, SD: 9.1%

Questions this analysis answers
-------------------------------
1. Does the ~35% offset reproduce for Ising (q=2)?
2. Does it reproduce for Potts q=3?
3. Does it BREAK for Potts q=5 (first-order)?
   If so, the offset is tied to correlation-length divergence (second-order only).
   If not, it's more likely intrinsic to the metric geometry.
4. Is there a systematic L-dependence (finite-size scaling of the offset)?
"""

import os, csv, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Literature critical temperatures
# ---------------------------------------------------------------------------
LITERATURE = {
    "Ising (q=2)": {
        "T_c": 2.0 / np.log(1.0 + np.sqrt(2.0)),     # ≈ 2.2692
        "order": "second-order",
        "source": "Onsager (1944), exact: 2/ln(1+sqrt(2))",
    },
    "Potts q=3": {
        "T_c": 1.0 / np.log(1.0 + np.sqrt(3.0)),     # ≈ 0.9950
        "order": "second-order",
        "source": "Baxter (1973), exact: 1/ln(1+sqrt(3))",
    },
    "Potts q=5": {
        "T_c": 1.0 / np.log(1.0 + np.sqrt(5.0)),     # ≈ 0.8515
        "order": "first-order",
        "source": "Baxter (1973), exact: 1/ln(1+sqrt(5))",
    },
}

# Prior offset data (for comparison — all "above T_c" experiments)
PRIOR_OFFSETS = [
    {"experiment": "Potts q=4 (L=64)",     "rel_offset": +0.263},
    {"experiment": "Potts q=4 FSS L=32",   "rel_offset": +0.438},
    {"experiment": "Potts q=4 FSS L=64",   "rel_offset": +0.263},
    {"experiment": "Potts q=4 FSS L=128",  "rel_offset": +0.418},
    {"experiment": "Contact Process L=64",  "rel_offset": +0.200},
    {"experiment": "Directed Percolation",  "rel_offset": +0.218},
]

# ---------------------------------------------------------------------------
# Load blind sweep peaks
# ---------------------------------------------------------------------------

def load_peaks():
    path = os.path.join(_HERE, "blind_sweep_peaks.csv")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Run blind_sweep.py first.")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_results():
    path = os.path.join(_HERE, "blind_sweep_results.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def main():
    peaks = load_peaks()
    results = load_results()
    if not peaks:
        return

    DIVIDER = "=" * 78

    print(f"\n{DIVIDER}")
    print("  BLIND SWEEP — OFFSET ANALYSIS")
    print(f"  Comparing measured C peaks against literature T_c values")
    print(f"{DIVIDER}\n")

    records = []

    print(f"  {'Model':<22}{'L':>4}{'T_c (lit)':>10}{'C peak':>9}"
          f"{'Offset':>9}{'Rel%':>8}  {'Order'}")
    print("-" * 78)

    for p in peaks:
        model = p["model"]
        L     = int(p["L"])
        peak_T = float(p["peak_T"])

        if model not in LITERATURE:
            continue
        lit = LITERATURE[model]
        T_c = lit["T_c"]
        offset = peak_T - T_c
        rel = offset / T_c

        records.append({
            "model": model,
            "L": L,
            "T_c": T_c,
            "peak_T": peak_T,
            "offset": offset,
            "rel_offset": rel,
            "order": lit["order"],
        })

        sign = "+" if offset >= 0 else ""
        print(f"  {model:<22}{L:>4}{T_c:>10.4f}{peak_T:>9.4f}"
              f"{sign}{offset:>8.4f}{sign}{rel*100:>7.1f}%"
              f"  {lit['order']}")

    if not records:
        print("  No matching records found.")
        return

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print("  SUMMARY STATISTICS")
    print(f"{DIVIDER}\n")

    second = [r for r in records if r["order"] == "second-order"]
    first  = [r for r in records if r["order"] == "first-order"]

    if second:
        rel_s = [r["rel_offset"] for r in second]
        print(f"  Second-order transitions (n={len(second)}):")
        print(f"    Mean relative offset: {np.mean(rel_s)*100:+.1f}%")
        print(f"    Std relative offset:  {np.std(rel_s)*100:.1f}%")
        print(f"    Range:                {min(rel_s)*100:+.1f}% to {max(rel_s)*100:+.1f}%\n")

    if first:
        rel_f = [r["rel_offset"] for r in first]
        print(f"  First-order transitions (n={len(first)}):")
        print(f"    Mean relative offset: {np.mean(rel_f)*100:+.1f}%")
        print(f"    Std relative offset:  {np.std(rel_f)*100:.1f}%")
        print(f"    Range:                {min(rel_f)*100:+.1f}% to {max(rel_f)*100:+.1f}%\n")

    # Compare against prior data
    prior_rel = [p["rel_offset"] for p in PRIOR_OFFSETS]
    new_second_rel = [r["rel_offset"] for r in second] if second else []
    all_second = prior_rel + new_second_rel

    print(f"  Combined with prior data (second-order only):")
    print(f"    Prior experiments (n={len(prior_rel)}):   "
          f"mean={np.mean(prior_rel)*100:+.1f}%, std={np.std(prior_rel)*100:.1f}%")
    if new_second_rel:
        print(f"    New experiments (n={len(new_second_rel)}):    "
              f"mean={np.mean(new_second_rel)*100:+.1f}%, std={np.std(new_second_rel)*100:.1f}%")
        print(f"    All combined (n={len(all_second)}):     "
              f"mean={np.mean(all_second)*100:+.1f}%, std={np.std(all_second)*100:.1f}%")

    # -----------------------------------------------------------------------
    # Finite-size scaling of offset
    # -----------------------------------------------------------------------
    print(f"\n{DIVIDER}")
    print("  FINITE-SIZE SCALING OF OFFSET")
    print(f"{DIVIDER}\n")

    for model in LITERATURE:
        model_recs = [r for r in records if r["model"] == model]
        if len(model_recs) < 2:
            continue
        model_recs.sort(key=lambda r: r["L"])
        print(f"  {model}:")
        for r in model_recs:
            print(f"    L={r['L']:>4}  peak_T={r['peak_T']:.4f}"
                  f"  offset={r['rel_offset']*100:+.1f}%")
        # Check if offset decreases with L (FSS prediction for second-order)
        Ls = [r["L"] for r in model_recs]
        offsets = [r["rel_offset"] for r in model_recs]
        if len(Ls) >= 2:
            trend = "decreasing" if offsets[-1] < offsets[0] else "increasing/stable"
            print(f"    Trend with L: {trend}")
        print()

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    print(f"{DIVIDER}")
    print("  INTERPRETATION")
    print(f"{DIVIDER}\n")

    if first and second:
        mean_s = np.mean([r["rel_offset"] for r in second]) * 100
        mean_f = np.mean([r["rel_offset"] for r in first]) * 100
        diff = abs(mean_s - mean_f)

        if diff > 15:
            print(f"  The offset DIFFERS between second-order ({mean_s:+.1f}%) and")
            print(f"  first-order ({mean_f:+.1f}%) transitions (gap = {diff:.1f}%).")
            print(f"  This suggests the ~35% offset is tied to correlation-length")
            print(f"  divergence, which only occurs at second-order transitions.")
            print(f"  First-order transitions have finite correlation lengths at T_c,")
            print(f"  producing a different offset profile.\n")
        else:
            print(f"  The offset is SIMILAR for second-order ({mean_s:+.1f}%) and")
            print(f"  first-order ({mean_f:+.1f}%) transitions (gap = {diff:.1f}%).")
            print(f"  This suggests the offset is intrinsic to the metric geometry")
            print(f"  rather than tied to correlation-length divergence.\n")

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("#0d0d1a")

    # Panel 1: offset bar chart (this experiment)
    ax = axes[0]
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    labels = [f"{r['model']}\nL={r['L']}" for r in records]
    offsets = [r["rel_offset"] * 100 for r in records]
    colors = ["#ef9a9a" if r["order"] == "second-order" else "#80cbc4"
              for r in records]

    bars = ax.barh(range(len(records)), offsets, color=colors,
                   edgecolor="#888888", linewidth=0.5, height=0.65)
    ax.axvline(0, color="white", linewidth=1.2, alpha=0.6)

    # Mark the prior mean
    prior_mean = np.mean(prior_rel) * 100
    ax.axvline(prior_mean, color="#ef9a9a", linestyle="--", alpha=0.5,
               linewidth=1.2, label=f"Prior mean = +{prior_mean:.1f}%")

    ax.set_yticks(range(len(records)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("(C_peak - T_c) / T_c × 100%", fontsize=10)
    ax.set_title("Relative Offset: Blind Sweep\nRed=2nd order | Teal=1st order",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, facecolor="#1e1e3f", labelcolor="white",
              edgecolor="#444466")
    ax.grid(True, alpha=0.15, color="white", axis="x")

    for i, v in enumerate(offsets):
        sign = "+" if v >= 0 else ""
        ax.text(v + (1 if v >= 0 else -1), i,
                f"{sign}{v:.1f}%", va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8, color="white")

    # Panel 2: C vs T curves with T_c markers (L=64 only)
    ax = axes[1]
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    model_colors = {
        "Ising (q=2)": "#ef9a9a",
        "Potts q=3": "#80cbc4",
        "Potts q=5": "#ce93d8",
    }

    L_plot = 64
    for model in LITERATURE:
        model_rows = [r for r in results
                      if r["model"] == model and int(r["L"]) == L_plot]
        if not model_rows:
            continue
        by_T = {}
        for r in model_rows:
            T = float(r["T"])
            by_T.setdefault(T, []).append(float(r["C_a"]))

        # Normalise to [0, 1] for comparison across models
        Ts = sorted(by_T.keys())
        means = [np.mean(by_T[t]) for t in Ts]
        max_C = max(means) if max(means) > 0 else 1
        normed = [m / max_C for m in means]

        col = model_colors[model]
        ax.plot(Ts, normed, "o-", color=col, markersize=3,
                linewidth=1.2, label=model, alpha=0.85)

        # Mark T_c
        T_c = LITERATURE[model]["T_c"]
        ax.axvline(T_c, color=col, linestyle=":", alpha=0.6, linewidth=1)

    ax.set_xlabel("Temperature T", fontsize=10)
    ax.set_ylabel("C / C_max (normalised)", fontsize=10)
    ax.set_title(f"Normalised C vs T (L={L_plot})\nDotted = literature T_c",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, facecolor="#1e1e3f", labelcolor="white",
              edgecolor="#444466")
    ax.grid(True, alpha=0.15, color="white")

    # Panel 3: FSS — offset vs 1/L
    ax = axes[2]
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")

    for model in LITERATURE:
        model_recs = sorted(
            [r for r in records if r["model"] == model],
            key=lambda r: r["L"])
        if len(model_recs) < 2:
            continue
        inv_L = [1.0 / r["L"] for r in model_recs]
        rels  = [r["rel_offset"] * 100 for r in model_recs]
        col = model_colors[model]
        ax.plot(inv_L, rels, "o-", color=col, markersize=6,
                linewidth=1.5, label=model)

    ax.set_xlabel("1/L", fontsize=10)
    ax.set_ylabel("Relative offset (%)", fontsize=10)
    ax.set_title("Offset vs System Size\n(FSS: does offset → 0 as L → ∞?)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, facecolor="#1e1e3f", labelcolor="white",
              edgecolor="#444466")
    ax.grid(True, alpha=0.15, color="white")

    fig.suptitle(
        "Blind Sweep Offset Analysis\n"
        "Measured C peak vs literature critical temperatures",
        fontsize=13, fontweight="bold", color="white", y=1.02)

    plt.tight_layout()
    out_path = os.path.join(_HERE, "offset_analysis.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nPlot -> {out_path}")
    print("Done.")


if __name__ == "__main__":
    main()
