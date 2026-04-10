"""
eca-scatter-beta.py
===================
Augments the 256-rule ECA scatter plot with the entity scale from the
discrete RG beta function.

Three panels:
  Left   — (C_a x1, AUC) coloured by Wolfram class  (same as before)
  Centre — (C_a x1, AUC) coloured by entity scale    (NEW)
           Entity scale = pool factor where beta first goes negative
           No sign change = no entity scale (trivial or scale-free)
  Right  — (entity scale, AUC) coloured by Wolfram class  (NEW)
           The most direct test: does adding the entity scale axis
           pull Class 4 away from Class 3?

Expected result
---------------
  Class 4 rules: entity scale at x4 or x8 (large entities = gliders)
  Class 3 rules: entity scale at x2 (local cell-pair correlations) or none
  If this holds, Class 4 will visually separate on the right panel.
"""

import os, sys, csv as _csv, importlib.util, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors

# ===========================================================================
# Bootstrap
# ===========================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_nn   = os.path.join(os.path.dirname(_HERE), "neural-network")
_spec = importlib.util.spec_from_file_location(
            "mnist_exp", os.path.join(_nn, "mnist-experiment.py"))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_full_C = _mod.compute_full_C

# ===========================================================================
# Config (identical to eca-full-scatter.py)
# ===========================================================================
N_CELLS      = 256
N_STEPS      = 300
N_SEEDS      = 1
ACTIVE_PCT   = 25.0
POOL_FACTORS = [1, 2, 4, 8]

# Wolfram class assignments (same as eca-full-scatter.py)
KNOWN_CLASSES = {
    0:1, 8:1, 32:1, 40:1, 128:1, 136:1, 160:1, 168:1, 255:1,
    4:2, 12:2, 13:2, 15:2, 19:2, 21:2, 25:2, 26:2, 27:2, 28:2,
    29:2, 31:2, 33:2, 34:2, 35:2, 36:2, 37:2, 38:2, 39:2, 42:2,
    43:2, 44:2, 46:2, 50:2, 51:2, 52:2, 53:2, 55:2, 56:2, 57:2,
    58:2, 59:2, 61:2, 62:2, 72:2, 73:2, 74:2, 76:2, 77:2, 78:2,
    88:2, 92:2, 94:2, 98:2, 104:2, 108:2, 130:2, 131:2, 132:2,
    133:2, 134:2, 138:2, 139:2, 140:2, 143:2, 152:2, 154:2, 156:2,
    162:2, 170:2, 172:2, 173:2, 178:2, 184:2, 194:2, 196:2, 200:2,
    201:2, 203:2, 204:2, 206:2, 226:2, 232:2, 234:2, 235:2, 238:2,
    18:3, 22:3, 30:3, 45:3, 60:3, 90:3, 105:3, 122:3, 126:3,
    146:3, 150:3, 182:3, 183:3,
    41:4, 54:4, 106:4, 110:4,
}

def mirror_rule(r):
    result = 0
    for i in range(8):
        j = ((i&1)<<2) | (i&2) | ((i>>2)&1)
        if (r >> i) & 1:
            result |= (1 << j)
    return result

def complement_rule(r):
    result = 0
    for i in range(8):
        j = 7 - i
        if not ((r >> i) & 1):
            result |= (1 << j)
    return result

def expand_classes(known):
    full = dict(known)
    for r, cls in list(known.items()):
        for equiv in [mirror_rule(r), complement_rule(r),
                      mirror_rule(complement_rule(r))]:
            if equiv not in full:
                full[equiv] = cls
    return full

WOLFRAM_CLASS = expand_classes(KNOWN_CLASSES)

CLASS_COLOR = {0:"#e0e0e0", 1:"#aaaaaa", 2:"steelblue",
               3:"tomato",  4:"mediumseagreen"}
CLASS_LABEL = {0:"Unknown", 1:"Class 1", 2:"Class 2",
               3:"Class 3", 4:"Class 4"}

# Entity-scale colour map: none=grey, x1=yellow, x2=orange, x4=teal, x8=purple
ENTITY_CMAP  = {None: "#cccccc", 1: "#f4d03f", 2: "#e67e22",
                4: "#27ae60",   8: "#8e44ad"}
ENTITY_LABEL = {None: "none (trivial/scale-free)", 1: "entity x1",
                2: "entity x2",  4: "entity x4",  8: "entity x8"}

# ===========================================================================
# ECA simulation
# ===========================================================================

def run_eca(rule, seed=0):
    rng       = np.random.RandomState(seed)
    cells     = rng.randint(0, 2, N_CELLS).astype(np.uint8)
    rule_bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)
    grid      = np.empty((N_STEPS, N_CELLS), dtype=np.uint8)
    for t in range(N_STEPS):
        grid[t] = cells
        left  = np.roll(cells, 1); right = np.roll(cells, -1)
        cells = rule_bits[(left*4+cells*2+right).astype(np.intp)]
    return grid

def grid_to_volumes(grid, pool_factor):
    T, W  = grid.shape
    n_sup = W // pool_factor
    if n_sup < 8:
        return None
    volumes = []
    if pool_factor == 1:
        for t in range(T):
            volumes.append(grid[t].astype(np.float32).reshape(1, 1, -1))
    else:
        for t in range(T):
            row    = grid[t].astype(np.float32).reshape(n_sup, pool_factor).mean(axis=1)
            cut    = np.percentile(row, 100.0 - ACTIVE_PCT)
            binary = (row > cut).astype(np.float32)
            volumes.append(binary.reshape(1, 1, -1))
    return volumes

# ===========================================================================
# Beta function + entity scale
# ===========================================================================

def compute_beta_entity(profile_ca):
    """
    profile_ca : dict {pool_factor: C_a_value}

    Returns
    -------
    betas        : list of first-differences in log2(pool_factor) space
    entity_scale : pool_factor just before first negative beta, or None
    """
    pfs  = sorted(p for p in POOL_FACTORS if p in profile_ca)
    vals = [profile_ca[p] for p in pfs]
    betas = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
    for i, b in enumerate(betas):
        if b < 0:
            return betas, pfs[i]   # entity_scale = factor at the peak
    return betas, None

# ===========================================================================
# Run all 256 rules
# ===========================================================================

def run_all():
    print(f"Running all 256 ECA rules  (N_CELLS={N_CELLS}, N_STEPS={N_STEPS})")
    records = []
    t0 = time.time()

    for rule in range(256):
        wclass = WOLFRAM_CLASS.get(rule, 0)
        aucs, ca_x1s, ca_peaks, entity_scales = [], [], [], []

        for seed in range(N_SEEDS):
            grid    = run_eca(rule, seed)
            profile = {}
            for pf in POOL_FACTORS:
                vols = grid_to_volumes(grid, pf)
                if vols is None:
                    continue
                r = compute_full_C(vols)
                profile[pf] = r["C_a"]

            if not profile:
                continue

            auc  = sum(profile.values())
            ca_x1 = profile.get(1, 0.0)
            ca_peak = max(profile.values())
            _, es = compute_beta_entity(profile)

            aucs.append(auc); ca_x1s.append(ca_x1)
            ca_peaks.append(ca_peak); entity_scales.append(es)

        if not aucs:
            continue

        # entity_scale: use mode across seeds (or the single seed value)
        from collections import Counter
        es_mode = Counter(entity_scales).most_common(1)[0][0]

        records.append({
            "rule":         rule,
            "class":        wclass,
            "auc":          float(np.mean(aucs)),
            "ca_x1":        float(np.mean(ca_x1s)),
            "ca_peak":      float(np.mean(ca_peaks)),
            "entity_scale": es_mode,
        })

        if rule % 32 == 0:
            print(f"  Rule {rule:3d}/255  {time.time()-t0:.0f}s  "
                  f"class={wclass}  auc={np.mean(aucs):.3f}  "
                  f"entity={es_mode}")

    print(f"Done in {time.time()-t0:.0f}s  ({len(records)} rules)\n")
    return records

# ===========================================================================
# Visualisation
# ===========================================================================

def confidence_ellipse(x, y, ax, n_std=1.5, **kw):
    if len(x) < 3:
        return
    cov        = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order      = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(np.abs(vals))
    ax.add_patch(Ellipse(xy=(np.mean(x), np.mean(y)),
                         width=w, height=h, angle=angle, **kw))


def plot_all(records, output_path):
    fig = plt.figure(figsize=(24, 9))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.32)

    # ----------------------------------------------------------------
    # Panel A — (C_a x1, AUC) coloured by Wolfram class
    # ----------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    for cls in [0, 1, 2, 3, 4]:
        pts = [(r["ca_x1"], r["auc"]) for r in records if r["class"] == cls]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax_a.scatter(xs, ys, c=CLASS_COLOR[cls],
                     s=50 if cls in (3,4) else 28,
                     alpha=0.85 if cls in (3,4) else 0.55,
                     edgecolors="k" if cls in (3,4) else "none",
                     linewidths=0.5, label=CLASS_LABEL[cls],
                     zorder=4 if cls in (3,4) else 2)
        if cls in (1,2,3,4) and len(xs) > 2:
            confidence_ellipse(np.array(xs), np.array(ys), ax_a,
                               edgecolor=CLASS_COLOR[cls], facecolor="none",
                               linewidth=1.5, linestyle="--", alpha=0.65, zorder=3)
    for r in records:
        if r["class"] in (3, 4):
            ax_a.annotate(str(r["rule"]),
                          xy=(r["ca_x1"], r["auc"]),
                          xytext=(3, 3), textcoords="offset points",
                          fontsize=6, color=CLASS_COLOR[r["class"]],
                          fontweight="bold")
    ax_a.set_xlabel("C_a at finest scale  (x1)", fontsize=10)
    ax_a.set_ylabel("AUC  (sum of C_a  across pool factors)", fontsize=10)
    ax_a.set_title("Coloured by Wolfram class\n(baseline view)", fontsize=10, fontweight="bold")
    ax_a.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_a.grid(True, alpha=0.2)

    # ----------------------------------------------------------------
    # Panel B — (C_a x1, AUC) coloured by ENTITY SCALE
    # ----------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])

    for es_val in [None, 1, 2, 4, 8]:
        pts = [(r["ca_x1"], r["auc"]) for r in records
               if r["entity_scale"] == es_val]
        if not pts:
            continue
        xs, ys = zip(*pts)
        is_focal = es_val in (4, 8)
        ax_b.scatter(xs, ys, c=ENTITY_CMAP[es_val],
                     s=55 if is_focal else 28,
                     alpha=0.90 if is_focal else 0.55,
                     edgecolors="k" if is_focal else "none",
                     linewidths=0.5,
                     label=ENTITY_LABEL[es_val],
                     zorder=4 if is_focal else 2)

    # Overlay Wolfram class markers for Class 3 and 4 only
    for r in records:
        if r["class"] in (3, 4):
            ax_b.annotate(str(r["rule"]),
                          xy=(r["ca_x1"], r["auc"]),
                          xytext=(3, 3), textcoords="offset points",
                          fontsize=6,
                          color=CLASS_COLOR[r["class"]],
                          fontweight="bold")

    ax_b.set_xlabel("C_a at finest scale  (x1)", fontsize=10)
    ax_b.set_ylabel("AUC  (sum of C_a  across pool factors)", fontsize=10)
    ax_b.set_title("Coloured by entity scale\n(pool factor where beta first goes negative)",
                   fontsize=10, fontweight="bold")
    ax_b.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_b.grid(True, alpha=0.2)

    # ----------------------------------------------------------------
    # Panel C — (entity scale, AUC) coloured by Wolfram class
    #           The direct test: does entity scale separate Class 3 vs 4?
    # ----------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])

    # Map entity scale to numeric x position with jitter
    scale_pos = {None: 0, 1: 1, 2: 2, 4: 3, 8: 4}
    rng_jit   = np.random.RandomState(42)

    for cls in [0, 1, 2, 3, 4]:
        pts = [(scale_pos[r["entity_scale"]], r["auc"])
               for r in records if r["class"] == cls]
        if not pts:
            continue
        xs_raw, ys = zip(*pts)
        xs = np.array(xs_raw, dtype=float)
        xs += rng_jit.uniform(-0.18, 0.18, len(xs))   # jitter

        ax_c.scatter(xs, ys, c=CLASS_COLOR[cls],
                     s=55 if cls in (3,4) else 30,
                     alpha=0.85 if cls in (3,4) else 0.50,
                     edgecolors="k" if cls in (3,4) else "none",
                     linewidths=0.5,
                     label=CLASS_LABEL[cls],
                     zorder=4 if cls in (3,4) else 2)

    for r in records:
        if r["class"] in (3, 4):
            jx = scale_pos[r["entity_scale"]] + rng_jit.uniform(-0.1, 0.1)
            ax_c.annotate(str(r["rule"]),
                          xy=(jx, r["auc"]),
                          xytext=(3, 3), textcoords="offset points",
                          fontsize=6,
                          color=CLASS_COLOR[r["class"]],
                          fontweight="bold")

    ax_c.set_xticks([0, 1, 2, 3, 4])
    ax_c.set_xticklabels(["none", "x1", "x2", "x4", "x8"], fontsize=9)
    ax_c.set_xlabel("Entity scale  (pool factor of first beta sign change)",
                    fontsize=10)
    ax_c.set_ylabel("AUC  (sum of C_a  across pool factors)", fontsize=10)
    ax_c.set_title("Entity scale  vs  AUC\n"
                   "Class 4 should cluster at x4 / x8",
                   fontsize=10, fontweight="bold")
    ax_c.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_c.grid(True, alpha=0.2)

    ax_c.text(0.02, 0.97,
              "H1: Class 4 concentrates at x4/x8\n"
              "    while Class 3 concentrates at x2/none.",
              transform=ax_c.transAxes, fontsize=8, va="top",
              bbox=dict(facecolor="lightyellow", edgecolor="goldenrod",
                        alpha=0.9, boxstyle="round,pad=0.4"))

    fig.suptitle(
        "256 ECA Rules — Multi-Scale Complexity + Beta Function Entity Scale\n"
        "Left: Wolfram class colour  |  Centre: entity scale colour  |  "
        "Right: entity scale vs AUC (the separation test)",
        fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {output_path}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    records  = run_all()

    # Save CSV
    csv_path = os.path.join(_HERE, "eca_scatter_beta.csv")
    fields   = ["rule", "class", "auc", "ca_x1", "ca_peak", "entity_scale"]
    with open(csv_path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else
                            (str(v) if v is not None else "none"))
                        for k, v in r.items()})
    print(f"  CSV -> {csv_path}")

    # Plot
    png_path = os.path.join(_HERE, "eca_scatter_beta.png")
    plot_all(records, png_path)

    # Summary
    print("\n" + "="*70)
    print("ENTITY SCALE DISTRIBUTION BY WOLFRAM CLASS")
    print("="*70)
    from collections import Counter
    for cls in [1, 2, 3, 4, 0]:
        pts  = [r for r in records if r["class"] == cls]
        if not pts:
            continue
        es_counts = Counter(r["entity_scale"] for r in pts)
        total     = len(pts)
        dist_str  = "  ".join(
            f"{'none' if k is None else ('x'+str(k))}:{v}/{total}"
            for k, v in sorted(es_counts.items(),
                                key=lambda x: (x[0] is None, x[0] or 0)))
        print(f"  {CLASS_LABEL[cls]:10s} ({total:3d}): {dist_str}")

    print("\nDone.")
