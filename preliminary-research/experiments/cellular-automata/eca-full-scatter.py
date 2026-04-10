"""
eca-full-scatter.py
===================
Run all 256 ECA rules through the multi-scale complexity pipeline and
plot every rule as a point in (C_a at x1, AUC) 2D space.

If the Wolfram classes form visual clusters, it confirms that the
(C_a_peak, AUC) pair is a principled 2D complexity descriptor.

Expected topology
-----------------
  Class 1  ->  bottom-left  (low single-scale C_a, low AUC)
  Class 2  ->  middle band   (low C_a at x1, elevated AUC from periodic structure)
  Class 3  ->  right cluster  (moderate-high C_a at x1, elevated AUC)
  Class 4  ->  top-right or distinct island  (elevated on both axes AND high AUC)

Speed settings: N_SEEDS=1, N_STEPS=300, POOL_FACTORS=[1, 2, 4, 8]
Estimated run: ~3-5 minutes on CPU.
"""

import os, sys, csv as _csv, importlib.util, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from scipy import stats as sp

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
# Configuration
# ===========================================================================
N_CELLS      = 256
N_STEPS      = 300
N_SEEDS      = 1
ACTIVE_PCT   = 25.0
POOL_FACTORS = [1, 2, 4, 8]

# Known Wolfram class assignments.
# Primary sources: Wolfram NKS (2002) + Culik & Yu (1988) + community consensus.
# Only the 88 minimal rules (under reflection/complement symmetry) have
# authoritative classifications; we label all 256 via equivalence mapping.
#
# Equivalence: for rule r, the four equivalent rules are:
#   r, mirror(r), complement(r), mirror(complement(r))
# where:
#   mirror(r)     = bit-reverse the 8-bit rule number
#   complement(r) = swap 0s and 1s in the rule table
#
# We assign the class of the canonical (smallest) rule in each group.

def mirror_rule(r):
    """Mirror symmetry: reverse the neighbourhood bits."""
    result = 0
    for i in range(8):
        left  = (i >> 2) & 1
        mid   = (i >> 1) & 1
        right =  i       & 1
        j     = (right << 2) | (mid << 1) | left
        if (r >> i) & 1:
            result |= (1 << j)
    return result

def complement_rule(r):
    """Complement symmetry: swap 0s and 1s in the state."""
    result = 0
    for i in range(8):
        ci = (7 - i) ^ 7          # complement neighbourhood: flip all bits
        j  = 7 - ci               # actually: complement means map i -> 7-i
        if (r >> i) & 1:
            result |= (1 << (7 - i))
    return result ^ 255            # also flip the output bits

# More explicit complement transform:
# complement means: replace each neighbourhood (abc) -> (1-a,1-b,1-c)
# and also flip the output: f(1-a,1-b,1-c) = 1 - f(a,b,c)
def complement_rule_correct(r):
    result = 0
    for i in range(8):
        j = 7 - i   # complement of neighbourhood i (flip all 3 bits)
        # output for neighbourhood j in new rule = 1 - output for i in old rule
        if not ((r >> i) & 1):
            result |= (1 << j)
    return result

# Canonical known classes (rule number -> Wolfram class)
KNOWN_CLASSES = {
    # Class 1: converge to uniform or fixed point
    0:   1,   8:   1,  32:  1,  40:  1,
    128: 1, 136:   1, 160:  1, 168:  1,
    255: 1,

    # Class 2: converge to periodic / stable structures
    4:   2,  12:  2,  13:  2,  15:  2,
    19:  2,  21:  2,  25:  2,  26:  2,
    27:  2,  28:  2,  29:  2,  31:  2,
    33:  2,  34:  2,  35:  2,  36:  2,
    37:  2,  38:  2,  39:  2,  42:  2,
    43:  2,  44:  2,  46:  2,  50:  2,
    51:  2,  52:  2,  53:  2,  55:  2,
    56:  2,  57:  2,  58:  2,  59:  2,
    61:  2,  62:  2,  72:  2,  73:  2,
    74:  2,  76:  2,  77:  2,  78:  2,
    88:  2,  92:  2,  94:  2,  98:  2,
    104: 2, 108:  2, 130:  2, 131:  2,
    132: 2, 133:  2, 134:  2, 138:  2,
    139: 2, 140:  2, 143:  2, 152:  2,
    154: 2, 156:  2, 162:  2, 170:  2,
    172: 2, 173:  2, 178:  2, 184:  2,
    194: 2, 196:  2, 200:  2, 201:  2,
    203: 2, 204:  2, 206:  2, 226:  2,
    232: 2, 234:  2, 235:  2, 238:  2,

    # Class 3: pseudo-random / chaotic
    18:  3,  22:  3,  30:  3,  45:  3,
    60:  3,  90:  3, 105:  3, 122:  3,
    126: 3, 146:  3, 150:  3, 182:  3,
    183: 3,

    # Class 4: complex / edge-of-chaos
    41:  4,  54:  4, 106:  4, 110:  4,
}

# Propagate known classes to all 4 equivalents
def expand_classes(known):
    full = dict(known)
    for r, cls in list(known.items()):
        for equiv in [mirror_rule(r),
                      complement_rule_correct(r),
                      mirror_rule(complement_rule_correct(r))]:
            if equiv not in full:
                full[equiv] = cls
    return full

WOLFRAM_CLASS = expand_classes(KNOWN_CLASSES)

CLASS_COLOR = {
    1: "#aaaaaa",           # grey
    2: "steelblue",         # blue
    3: "tomato",            # red
    4: "mediumseagreen",    # green
    0: "#dddddd",           # unknown — very light grey
}
CLASS_LABEL = {0: "Unknown", 1: "Class 1", 2: "Class 2", 3: "Class 3", 4: "Class 4"}

# ===========================================================================
# ECA simulation
# ===========================================================================

def eca_step(cells, rule_bits):
    left  = np.roll(cells, 1)
    right = np.roll(cells, -1)
    idx   = (left * 4 + cells * 2 + right).astype(np.intp)
    return rule_bits[idx]

def run_eca(rule, n_cells=N_CELLS, n_steps=N_STEPS, seed=0):
    rng       = np.random.RandomState(seed)
    cells     = rng.randint(0, 2, n_cells).astype(np.uint8)
    rule_bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)
    grid      = np.empty((n_steps, n_cells), dtype=np.uint8)
    for t in range(n_steps):
        grid[t] = cells
        cells   = eca_step(cells, rule_bits)
    return grid

def grid_to_volumes(grid, pool_factor, active_pct=ACTIVE_PCT):
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
            cut    = np.percentile(row, 100.0 - active_pct)
            binary = (row > cut).astype(np.float32)
            volumes.append(binary.reshape(1, 1, -1))
    return volumes

# ===========================================================================
# Run all 256 rules
# ===========================================================================

def run_all_rules():
    print(f"Running all 256 ECA rules  "
          f"(N_CELLS={N_CELLS}, N_STEPS={N_STEPS}, "
          f"N_SEEDS={N_SEEDS}, pool_factors={POOL_FACTORS})")
    print("Estimated time: 3-5 minutes\n")

    records = []
    t0 = time.time()

    for rule in range(256):
        wclass = WOLFRAM_CLASS.get(rule, 0)

        seed_aucs    = []
        seed_ca_x1   = []
        seed_ca_peak = []

        for seed in range(N_SEEDS):
            grid    = run_eca(rule, seed=seed)
            profile = {}
            for pf in POOL_FACTORS:
                vols = grid_to_volumes(grid, pf)
                if vols is None:
                    continue
                r = compute_full_C(vols)
                profile[pf] = r["C_a"]

            if not profile:
                continue

            auc      = sum(profile.values())
            ca_x1    = profile.get(1, 0.0)
            ca_peak  = max(profile.values())
            peak_pf  = max(profile, key=profile.get)

            seed_aucs.append(auc)
            seed_ca_x1.append(ca_x1)
            seed_ca_peak.append(ca_peak)

        if not seed_aucs:
            continue

        rec = {
            "rule":      rule,
            "class":     wclass,
            "auc":       float(np.mean(seed_aucs)),
            "ca_x1":     float(np.mean(seed_ca_x1)),
            "ca_peak":   float(np.mean(seed_ca_peak)),
        }
        records.append(rec)

        # Progress
        if rule % 32 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / max(rule, 1) * (256 - rule)
            print(f"  Rule {rule:3d}/255  elapsed={elapsed:.0f}s  "
                  f"ETA={eta:.0f}s  "
                  f"class={wclass}  auc={rec['auc']:.3f}  ca_x1={rec['ca_x1']:.3f}")

    elapsed = time.time() - t0
    print(f"\nAll rules done in {elapsed:.0f}s")
    return records

# ===========================================================================
# Save CSV
# ===========================================================================

def save_csv(records, path):
    fields = ["rule", "class", "auc", "ca_x1", "ca_peak"]
    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"  CSV -> {path}")

# ===========================================================================
# Visualisation
# ===========================================================================

def confidence_ellipse(x, y, ax, n_std=1.5, **kwargs):
    """Draw a covariance ellipse around a set of (x, y) points."""
    if len(x) < 3:
        return
    cov  = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(vals)
    ell   = Ellipse(xy=(np.mean(x), np.mean(y)),
                    width=w, height=h, angle=angle, **kwargs)
    ax.add_patch(ell)


def plot_scatter(records, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # ---------- Panel 1: (C_a at x1, AUC) ----------
    ax = axes[0]
    for cls in [0, 1, 2, 3, 4]:
        pts  = [(r["ca_x1"], r["auc"]) for r in records if r["class"] == cls]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys,
                   c=CLASS_COLOR[cls],
                   s=45 if cls in (3, 4) else 28,
                   alpha=0.82 if cls in (3, 4) else 0.55,
                   edgecolors="k" if cls in (3, 4) else "none",
                   linewidths=0.5,
                   label=CLASS_LABEL[cls],
                   zorder=4 if cls in (3, 4) else 2)
        # Confidence ellipse for known classes
        if cls in (1, 2, 3, 4) and len(xs) > 2:
            confidence_ellipse(np.array(xs), np.array(ys), ax,
                               n_std=1.5,
                               edgecolor=CLASS_COLOR[cls],
                               facecolor="none",
                               linewidth=1.5,
                               linestyle="--",
                               zorder=3,
                               alpha=0.7)

    # Label known Class 3 and Class 4 rules
    for r in records:
        if r["class"] in (3, 4):
            ax.annotate(str(r["rule"]),
                        xy=(r["ca_x1"], r["auc"]),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=6.5,
                        color=CLASS_COLOR[r["class"]],
                        fontweight="bold")

    ax.set_xlabel("C_a at finest scale  (x1,  256 cells)", fontsize=11)
    ax.set_ylabel("AUC  (sum of C_a across pool factors [1,2,4,8])", fontsize=11)
    ax.set_title("All 256 ECA Rules in (C_a x1, AUC) Space\n"
                 "Wolfram classes 1-4 labeled where known",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.22)

    ax.text(0.98, 0.02,
            "H1: Class 4 forms a distinct cluster / island\n"
            "    separate from Class 3 in 2D space.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(facecolor="lightyellow", edgecolor="goldenrod",
                      alpha=0.85, boxstyle="round,pad=0.4"))

    # ---------- Panel 2: (C_a_peak, AUC) ----------
    ax2 = axes[1]
    for cls in [0, 1, 2, 3, 4]:
        pts  = [(r["ca_peak"], r["auc"]) for r in records if r["class"] == cls]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax2.scatter(xs, ys,
                    c=CLASS_COLOR[cls],
                    s=45 if cls in (3, 4) else 28,
                    alpha=0.82 if cls in (3, 4) else 0.55,
                    edgecolors="k" if cls in (3, 4) else "none",
                    linewidths=0.5,
                    label=CLASS_LABEL[cls],
                    zorder=4 if cls in (3, 4) else 2)
        if cls in (1, 2, 3, 4) and len(xs) > 2:
            confidence_ellipse(np.array(xs), np.array(ys), ax2,
                               n_std=1.5,
                               edgecolor=CLASS_COLOR[cls],
                               facecolor="none",
                               linewidth=1.5,
                               linestyle="--",
                               zorder=3,
                               alpha=0.7)

    for r in records:
        if r["class"] in (3, 4):
            ax2.annotate(str(r["rule"]),
                         xy=(r["ca_peak"], r["auc"]),
                         xytext=(3, 3), textcoords="offset points",
                         fontsize=6.5,
                         color=CLASS_COLOR[r["class"]],
                         fontweight="bold")

    ax2.set_xlabel("C_a at peak scale  (max across all pool factors)", fontsize=11)
    ax2.set_ylabel("AUC  (sum of C_a across pool factors [1,2,4,8])", fontsize=11)
    ax2.set_title("All 256 ECA Rules in (C_a peak, AUC) Space\n"
                  "Wolfram classes 1-4 labeled where known",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax2.grid(True, alpha=0.22)

    fig.suptitle(
        "256 ECA Rules — Multi-Scale Complexity 2D Scatter\n"
        f"N_CELLS={N_CELLS}  N_STEPS={N_STEPS}  pool factors {POOL_FACTORS}  "
        f"N_SEEDS={N_SEEDS}  adaptive {ACTIVE_PCT:.0f}% threshold",
        fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {output_path}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    records  = run_all_rules()

    csv_path = os.path.join(_HERE, "eca_full_scatter.csv")
    save_csv(records, csv_path)

    png_path = os.path.join(_HERE, "eca_full_scatter.png")
    plot_scatter(records, png_path)

    # Summary by class
    print("\n" + "="*60)
    print("CLASS STATISTICS  (mean +/- std of AUC and C_a_x1)")
    print("="*60)
    for cls in [1, 2, 3, 4, 0]:
        pts = [r for r in records if r["class"] == cls]
        if not pts:
            continue
        aucs = [r["auc"]   for r in pts]
        cax1 = [r["ca_x1"] for r in pts]
        print(f"  {CLASS_LABEL[cls]:10s} ({len(pts):3d} rules)  "
              f"AUC={np.mean(aucs):.3f}+/-{np.std(aucs):.3f}  "
              f"Ca_x1={np.mean(cax1):.3f}+/-{np.std(cax1):.3f}")

    print("\nDone.")
