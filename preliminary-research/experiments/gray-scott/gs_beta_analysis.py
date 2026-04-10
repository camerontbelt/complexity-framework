"""
Compute and plot the discrete RG beta function from the multi-scale GS results.
beta(s->2s) = C_a(2s) - C_a(s)   [in log2-pooling-factor space]

A sign change in beta identifies the entity scale (where complexity peaks).
No sign change + low C_a(s=1)  -> trivial (dead)
No sign change + high C_a(s=1) -> turbulent / scale-free (chaotic)
Sign change present             -> entity scale exists (entity_scale = first sign-flip factor)
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(_HERE, "gray_scott_multiscale.csv")

rows = list(csv.DictReader(open(csv_path)))
POOL_FACTORS = [1, 2, 4, 8, 16]

# ---- Aggregate mean C_a per (name, pool_factor) ----
acc = defaultdict(list)
meta = {}
for r in rows:
    name = r["name"]
    pf   = int(r["pool_factor"])
    acc[(name, pf)].append(float(r["C_a"]))
    meta[name] = r["expected_class"]

ORDER = ["dead", "static_spots", "self_rep_spots", "worm_complex", "solitons", "chaotic"]
CLS_COLOR = {
    "trivial": "dimgray", "ordered": "steelblue",
    "complex": "mediumseagreen", "chaotic": "tomato",
}

profiles = {}
for name in ORDER:
    profiles[name] = {pf: np.mean(acc[(name, pf)]) for pf in POOL_FACTORS}

# ---- Compute beta (first differences in log2 space) ----
# beta_i = C_a(pf_{i+1}) - C_a(pf_i)   where pf are equally spaced in log2
beta = {}
for name in ORDER:
    p = profiles[name]
    vals = [p[pf] for pf in POOL_FACTORS]
    beta[name] = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
# pool factor transitions: 1->2, 2->4, 4->8, 8->16
beta_labels = ["x1->x2", "x2->x4", "x4->x8", "x8->x16"]
beta_x      = [1.5, 3.0, 6.0, 12.0]   # geometric midpoints for x-axis

# ---- Find entity scale (first sign change in beta) ----
def entity_scale(beta_vals):
    """Return pool factor JUST BEFORE the first negative beta, or None."""
    for i, b in enumerate(beta_vals):
        if b < 0:
            return POOL_FACTORS[i]   # the peak (beta was positive up to here)
    return None

entity_scales = {name: entity_scale(beta[name]) for name in ORDER}
Ca_s1 = {name: profiles[name][1] for name in ORDER}

# ---- Print summary ----
print()
print("=" * 70)
print("Discrete RG Beta Function Analysis  (beta = dC_a / d log2(s))")
print("=" * 70)
print(f"\n  {'Regime':18s}  {'class':8s}  {'C_a(s=1)':10s}  "
      f"{'beta_1->2':10s}  {'beta_2->4':10s}  {'beta_4->8':10s}  {'beta_8->16':10s}  "
      f"{'entity_scale':12s}")
print("-" * 105)
for name in ORDER:
    b = beta[name]
    es = entity_scales[name]
    ec = meta[name]
    print(f"  {name:18s}  {ec:8s}  {Ca_s1[name]:.4f}      "
          + "  ".join(f"{bi:+.3f}    " for bi in b)
          + f"  x{es}" if es else "  (none)  <- trivial or chaotic")

print()
print("Interpretation:")
print("  beta always positive + low  C_a(s=1) -> TRIVIAL  (dead)")
print("  beta always positive + high C_a(s=1) -> CHAOTIC  (scale-free)")
print("  beta has sign change                 -> ENTITY SCALE EXISTS")
print("  Entity scale = peak pool factor (where beta turns negative)")

# ---- 2D discriminant plot: C_a(s=1) vs. entity-scale tier ----
fig = plt.figure(figsize=(20, 12))
gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

# Panel 1: beta profiles (one line per regime)
ax1 = fig.add_subplot(gs[0, :2])
for name in ORDER:
    color  = CLS_COLOR[meta[name]]
    b_vals = beta[name]
    ax1.plot(range(len(beta_labels)), b_vals,
             marker="o", markersize=8, linewidth=2.3, color=color,
             label=f"{name} [{meta[name]}]")
    # Mark first sign change
    es = entity_scales[name]
    if es is not None:
        idx = POOL_FACTORS.index(es)   # index of the last positive beta
        ax1.plot(idx, b_vals[idx], "v", markersize=12, color=color,
                 markeredgecolor="k", markeredgewidth=0.8, zorder=5)

ax1.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.6)
ax1.fill_between(range(len(beta_labels)), 0, -0.5,
                 alpha=0.04, color="blue", label="beta < 0  (complexity decreasing)")
ax1.set_xticks(range(len(beta_labels)))
ax1.set_xticklabels(beta_labels, fontsize=10)
ax1.set_xlabel("Coarsening step (pooling factor transition)", fontsize=10)
ax1.set_ylabel("beta = C_a(2s) - C_a(s)", fontsize=10)
ax1.set_title(
    "Discrete RG Beta Function  beta(s->2s) = C_a(2s) - C_a(s)\n"
    "Downward triangle = first sign change (entity scale detection)",
    fontsize=11, fontweight="bold")
ax1.legend(fontsize=8.5, loc="upper right")
ax1.grid(True, alpha=0.22)

ax1.text(0.01, 0.04,
    "beta > 0 always: no entity scale (trivial or chaotic).\n"
    "beta sign change: entity scale identified at the peak pool factor.",
    transform=ax1.transAxes, fontsize=8.5, va="bottom",
    bbox=dict(facecolor="lightyellow", edgecolor="goldenrod",
              alpha=0.85, boxstyle="round,pad=0.4"))

# Panel 2: 2D discriminant scatter: (C_a at s=1) vs (entity scale)
ax2 = fig.add_subplot(gs[0, 2])
ys = {"none_low": -0.5, "none_high": 5.5, 4: 4, 8: 8}   # jitter positions

# Map entity scale to y-axis value (for scatter)
y_map = {None: None, 4: 4, 8: 8}   # None will be split by C_a(s=1)
jitter = np.random.RandomState(0).uniform(-0.2, 0.2, len(ORDER))

for i, name in enumerate(ORDER):
    color = CLS_COLOR[meta[name]]
    x_val = Ca_s1[name]
    es    = entity_scales[name]
    if es is not None:
        y_val = es
    else:
        # trivial or chaotic — put on x-axis at separate y levels
        y_val = -1.5 if x_val < 0.1 else 16.5
    ax2.scatter(x_val, y_val + jitter[i], s=180, color=color,
                edgecolors="k", linewidths=0.8, zorder=4)
    ax2.text(x_val + 0.005, y_val + jitter[i], name,
             fontsize=7.5, va="center", color=color, fontweight="bold")

ax2.axhline(0,  color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
ax2.axhline(13, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
ax2.axvline(0.1, color="gray", linewidth=1.0, linestyle="--", alpha=0.5)
ax2.set_yticks([-1.5, 4, 8, 16.5])
ax2.set_yticklabels(["no entity\n(trivial)", "x4\n(entity)", "x8\n(entity)", "no entity\n(chaotic)"],
                    fontsize=8.5)
ax2.set_xlabel("C_a at finest scale (s=1)", fontsize=9)
ax2.set_title("2D Discriminant\n(C_a at s=1  vs.  entity scale)", fontsize=10, fontweight="bold")
ax2.grid(True, alpha=0.2)

# Panel 3: C_a profiles with beta shading
ax3 = fig.add_subplot(gs[1, :])
for name in ORDER:
    color  = CLS_COLOR[meta[name]]
    p_vals = [profiles[name][pf] for pf in POOL_FACTORS]
    b_vals = beta[name]
    ax3.plot(POOL_FACTORS, p_vals, marker="o", markersize=7,
             linewidth=2.3, color=color, label=name)
    # Shade segments where beta < 0 (complexity decreasing under coarsening)
    for i, b in enumerate(b_vals):
        if b < 0:
            pf_left  = POOL_FACTORS[i]
            pf_right = POOL_FACTORS[i+1]
            y_lo = min(profiles[name][pf_left], profiles[name][pf_right])
            y_hi = max(profiles[name][pf_left], profiles[name][pf_right])
            ax3.fill_betweenx([y_lo, y_hi],
                              pf_left, pf_right,
                              alpha=0.15, color=color,
                              linewidth=0)

ax3.axhspan(0, 0.05, alpha=0.04, color="gray",
            label="Near-zero C_a (noise floor)")
ax3.set_xscale("log", base=2)
ax3.set_xticks(POOL_FACTORS)
ax3.set_xticklabels([f"x{p}" for p in POOL_FACTORS])
ax3.set_xlabel("Spatial pooling factor  (->  coarser scale)", fontsize=10)
ax3.set_ylabel("C_a", fontsize=10)
ax3.set_title(
    "C_a profiles with beta<0 segments shaded  "
    "(shaded region = complexity decreasing under coarsening = entity scale detected)",
    fontsize=10, fontweight="bold")
ax3.legend(fontsize=8.5, ncol=6, loc="upper left")
ax3.grid(True, alpha=0.22)

fig.suptitle(
    "Gray-Scott Multi-Scale Analysis: Discrete RG Beta Function\n"
    "beta(s->2s) = C_a(2s) - C_a(s)  |  sign change identifies entity scale  |  "
    "direction of profile separates trivial / entity-scale / scale-free regimes",
    fontsize=11, fontweight="bold", y=1.01)

out_path = os.path.join(_HERE, "gs_beta_analysis.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Plot -> {out_path}")
