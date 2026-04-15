"""
eca-agnostic-rank.py
====================
Run all 256 ECA rules and rank by C_a_bp (agnostic composite with
bit-packed gzip). Computes metrics directly on the spatiotemporal grid
matching the v9 framework, then applies three gzip weight strategies:

  1. w_G (v8):    Gaussian at gzip_byte = 0.10 (CA-calibrated)
  2. w_G_a:       tanh gate on gzip_byte (agnostic, byte-encoded)
  3. w_G_a_bp:    tanh gate on gzip_bitpacked (agnostic, parameter-free)

The bit-packing correction (np.packbits for binary data) removes the
byte-encoding artifact (gzip ~ 1/8) and is derivable from first principles.
"""

import os, sys, csv, time, gzip, zlib
import importlib.util
import numpy as np
from collections import defaultdict

# Bootstrap agnostic weight functions from mnist-experiment.py
_HERE = os.path.dirname(os.path.abspath(__file__))
_nn   = os.path.join(os.path.dirname(_HERE), "neural-network")
_spec = importlib.util.spec_from_file_location(
            "mnist_exp", os.path.join(_nn, "mnist-experiment.py"))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Import agnostic weight functions
wh_agnostic   = _mod.wh_agnostic
wops_agnostic = _mod.wops_agnostic
wopt_agnostic = _mod.wopt_agnostic
wt_agnostic   = _mod.wt_agnostic
wg_agnostic   = _mod.wg_agnostic
wg_agnostic_bp = _mod.wg_agnostic_bp

# v8 weight functions for comparison
wh_weight   = _mod.wh_weight
wops_weight = _mod.wops_weight
wopt_weight = _mod.wopt_weight
wt_weight   = _mod.wt_weight
wg_weight   = _mod.wg_weight

# ---- Wolfram classes ----
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
        left = (i >> 2) & 1; mid = (i >> 1) & 1; right = i & 1
        j = (right << 2) | (mid << 1) | left
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
CLASS_LABEL = {0: "??", 1: "C1", 2: "C2", 3: "C3", 4: "C4"}

# ---- ECA runner (matching v9 framework) ----
N_CELLS = 150     # W=150 for direct comparability with existing data
N_STEPS = 300
BURNIN  = 50      # Discard transient
WINDOW  = 150     # Analysis window (v9 default)
DENSITY = 0.5

def run_eca(rule, n_cells=N_CELLS, n_steps=N_STEPS, seed=42):
    rng = np.random.RandomState(seed)
    cells = (rng.rand(n_cells) < DENSITY).astype(np.uint8)
    rule_bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)
    grid = np.empty((n_steps, n_cells), dtype=np.uint8)
    for t in range(n_steps):
        grid[t] = cells
        left  = np.roll(cells, 1)
        right = np.roll(cells, -1)
        cells = rule_bits[(left * 4 + cells * 2 + right).astype(np.intp)]
    return grid


# ---- Metric computation (matching v9 framework) ----

def compute_entropy(grid, burnin, window):
    """Per-cell binary entropy, mean and std across cells and timesteps."""
    post = grid[burnin:burnin + window].astype(np.float64)
    T, W = post.shape
    density = np.clip(post.mean(axis=0), 1e-12, 1 - 1e-12)  # per-cell density
    H_vals = -(density * np.log2(density) + (1 - density) * np.log2(1 - density))
    return float(H_vals.mean()), float(H_vals.std())


def compute_opacity_spatial(grid, burnin, window, n_bins=8):
    """H(global|local) and H(local|global) — matches v9."""
    post = grid[burnin:burnin + window]
    T, W = post.shape
    dens = post.mean(axis=1)
    gbins = np.clip((dens * n_bins).astype(int), 0, n_bins - 1)
    left = np.roll(post, 1, axis=1)
    right = np.roll(post, -1, axis=1)
    patch_int = (left * 4 + post * 2 + right).astype(np.int16)
    g_flat = np.repeat(gbins, W)
    p_flat = patch_int.ravel()
    joint = np.zeros((8, n_bins), dtype=np.int64)
    np.add.at(joint, (p_flat, g_flat), 1)
    if joint.sum() == 0:
        return 0.0, 0.0
    def _H(counts):
        c = counts[counts > 0].astype(float)
        p = c / c.sum()
        return float(-np.sum(p * np.log2(p)))
    H_joint = _H(joint.ravel())
    H_patch = _H(joint.sum(axis=1))
    H_glob  = _H(joint.sum(axis=0))
    op_up   = float(np.clip((H_joint - H_patch) / np.log2(n_bins), 0.0, 1.0))
    op_down = float(np.clip((H_joint - H_glob)  / np.log2(8),      0.0, 1.0))
    return op_up, op_down


def compute_opacity_temporal(grid, burnin, window, max_lag=10, stride=3):
    """MI(X_t; X_{t+lag}) / H(X_t) — matches v9."""
    T, W = grid.shape
    def _mi_at_lag(lag):
        joint = defaultdict(int)
        marg_t = defaultdict(int)
        marg_tl = defaultdict(int)
        for t in range(burnin, burnin + window - lag):
            for x in range(0, W, stride):
                a = int(grid[t, x])
                b = int(grid[t + lag, x])
                joint[(a, b)] += 1
                marg_t[a] += 1
                marg_tl[b] += 1
        total = sum(joint.values())
        if total == 0:
            return 0.0
        def _H(d):
            s = sum(d.values())
            return -sum((c/s) * np.log2(c/s) for c in d.values() if c > 0)
        MI = _H(marg_t) + _H(marg_tl) - _H(joint)
        Ht = _H(marg_t)
        return float(np.clip(MI / max(Ht, 1e-9), 0.0, 1.0))
    mi1 = _mi_at_lag(1)
    mi_k = _mi_at_lag(max_lag)
    decay = float(np.clip(mi1 - mi_k, 0.0, 1.0))
    return mi1, decay


def compute_tcomp(grid, burnin, window):
    """Mean temporal compression — matches v9."""
    post = grid[burnin:burnin + window]
    flips = np.sum(post[1:] != post[:-1], axis=0)
    return float(np.clip(1.0 - (1 + flips) / window, 0.0, 1.0).mean())


def compute_gzip_byte(grid, burnin, window):
    """Gzip ratio on byte-encoded grid — matches v9."""
    raw = grid[burnin:burnin + window].tobytes()
    return len(zlib.compress(raw, 6)) / len(raw)


def compute_gzip_bitpacked(grid, burnin, window):
    """Gzip ratio on bit-packed grid — 1 bit per binary cell.
    For binary data: np.packbits stores 8 cells per byte.
    The correction factor is derivable: bits_per_cell = ceil(log2(q))."""
    post = grid[burnin:burnin + window]
    packed = np.packbits(post.ravel())
    raw = packed.tobytes()
    compressed = gzip.compress(raw)
    return len(compressed) / max(len(raw), 1)


def compute_all_metrics(grid):
    """Compute all raw metrics on the spatiotemporal grid."""
    mean_H, std_H = compute_entropy(grid, BURNIN, WINDOW)
    op_up, op_down = compute_opacity_spatial(grid, BURNIN, WINDOW)
    mi1, decay = compute_opacity_temporal(grid, BURNIN, WINDOW)
    tc_mean = compute_tcomp(grid, BURNIN, WINDOW)
    gz_byte = compute_gzip_byte(grid, BURNIN, WINDOW)
    gz_bp   = compute_gzip_bitpacked(grid, BURNIN, WINDOW)
    return {
        "mean_H": mean_H, "std_H": std_H,
        "op_up": op_up, "op_down": op_down,
        "mi1": mi1, "decay": decay,
        "tc_mean": tc_mean,
        "gz_byte": gz_byte,
        "gz_bp": gz_bp,
        "gz_x8": gz_byte * 8.0,  # simple ×8 correction
    }


def compute_composites(m):
    """Compute all three composites from raw metrics."""
    # Common agnostic weights (same for all composites)
    wH_a    = wh_agnostic(m["mean_H"], m["std_H"])
    wOPs_a  = wops_agnostic(m["op_up"], m["op_down"])
    wOPt_a  = wopt_agnostic(m["mi1"], m["decay"])
    wT_a    = wt_agnostic(m["tc_mean"])

    # v8 weights
    wH    = wh_weight(m["mean_H"], m["std_H"])
    wOPs  = wops_weight(m["op_up"], m["op_down"])
    wOPt  = wopt_weight(m["mi1"], m["decay"])
    wT    = wt_weight(m["tc_mean"])

    # Three gzip weight strategies
    wG_v8    = wg_weight(m["gz_byte"])             # Gaussian at 0.10
    wG_a     = wg_agnostic(m["gz_byte"])           # tanh gate on byte gzip
    wG_a_bp  = wg_agnostic_bp(m["gz_bp"])          # tanh gate on bit-packed gzip
    wG_a_x8  = wg_agnostic_bp(m["gz_x8"])          # tanh gate on byte×8

    # Composites: C = wH × (wOPs + wOPt) × wT × wG
    C_v8   = wH   * (wOPs   + wOPt)   * wT   * wG_v8
    C_a    = wH_a * (wOPs_a + wOPt_a) * wT_a * wG_a
    C_a_bp = wH_a * (wOPs_a + wOPt_a) * wT_a * wG_a_bp
    C_a_x8 = wH_a * (wOPs_a + wOPt_a) * wT_a * wG_a_x8

    return {
        "C_v8": C_v8, "C_a": C_a, "C_a_bp": C_a_bp, "C_a_x8": C_a_x8,
        "wG_v8": wG_v8, "wG_a": wG_a, "wG_a_bp": wG_a_bp, "wG_a_x8": wG_a_x8,
        "wH_a": wH_a, "wOPs_a": wOPs_a, "wOPt_a": wOPt_a, "wT_a": wT_a,
    }


# ---- Main ----

def main():
    print("=" * 100)
    print("  ALL 256 ECA RULES — Agnostic C with Bit-Packed Gzip")
    print(f"  W={N_CELLS}  steps={N_STEPS}  burnin={BURNIN}  window={WINDOW}  "
          f"density={DENSITY}  seed=42")
    print("=" * 100)

    records = []
    t0 = time.time()

    for rule in range(256):
        wclass = WOLFRAM_CLASS.get(rule, 0)
        grid = run_eca(rule)
        m = compute_all_metrics(grid)
        c = compute_composites(m)

        rec = {"rule": rule, "class": wclass}
        rec.update(m)
        rec.update(c)
        records.append(rec)

        if rule % 32 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(rule, 1) * (256 - rule)
            print(f"  Rule {rule:3d}/255  elapsed={elapsed:.0f}s  ETA~{eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s\n")

    # ---- Rank by C_a_bp ----
    ranked_bp = sorted(records, key=lambda x: x["C_a_bp"], reverse=True)

    print("=" * 100)
    print("  TOP 30 RULES BY C_a_bp (Agnostic + Bit-Packed Gzip)")
    print("=" * 100)
    print(f"{'Rk':>3} {'Rule':>5} {'Cls':>3} {'C_a_bp':>8} {'C_a_x8':>8} "
          f"{'C_a':>8} {'C_v8':>8} {'gz_byte':>8} {'gz_bp':>7} {'gz_x8':>6} "
          f"{'wG_bp':>6} {'tcomp':>6}")
    print("-" * 100)
    for i, rec in enumerate(ranked_bp[:30]):
        print(f"{i+1:>3} {rec['rule']:>5} {CLASS_LABEL[rec['class']]:>3} "
              f"{rec['C_a_bp']:>8.4f} {rec['C_a_x8']:>8.4f} "
              f"{rec['C_a']:>8.4f} {rec['C_v8']:>8.4f} "
              f"{rec['gz_byte']:>8.5f} {rec['gz_bp']:>7.3f} "
              f"{rec['gz_x8']:>6.3f} {rec['wG_a_bp']:>6.3f} "
              f"{rec['tc_mean']:>6.3f}")

    # ---- Rank by C_a_x8 ----
    ranked_x8 = sorted(records, key=lambda x: x["C_a_x8"], reverse=True)

    print(f"\n{'=' * 100}")
    print("  TOP 30 RULES BY C_a_x8 (Agnostic + Byte Gzip × 8)")
    print("=" * 100)
    print(f"{'Rk':>3} {'Rule':>5} {'Cls':>3} {'C_a_x8':>8} {'C_a_bp':>8} "
          f"{'C_v8':>8} {'gz_byte':>8} {'gz_x8':>6} {'wG_x8':>6}")
    print("-" * 100)
    for i, rec in enumerate(ranked_x8[:30]):
        print(f"{i+1:>3} {rec['rule']:>5} {CLASS_LABEL[rec['class']]:>3} "
              f"{rec['C_a_x8']:>8.4f} {rec['C_a_bp']:>8.4f} "
              f"{rec['C_v8']:>8.4f} {rec['gz_byte']:>8.5f} "
              f"{rec['gz_x8']:>6.3f} {rec['wG_a_x8']:>6.3f}")

    # ---- Rank by C_v8 (for reference) ----
    ranked_v8 = sorted(records, key=lambda x: x["C_v8"], reverse=True)

    print(f"\n{'=' * 100}")
    print("  TOP 10 RULES BY C_v8 (Original Calibrated)")
    print("=" * 100)
    for i, rec in enumerate(ranked_v8[:10]):
        print(f"  {i+1:>2}. Rule {rec['rule']:>3} ({CLASS_LABEL[rec['class']]})  "
              f"C_v8={rec['C_v8']:.4f}  gz={rec['gz_byte']:.5f}")

    # ---- Class summary for all composites ----
    print(f"\n{'=' * 100}")
    print("  CLASS SUMMARY — ALL THREE GZIP APPROACHES")
    print("=" * 100)
    for cls in [4, 3, 2, 1]:
        recs = [r for r in records if r["class"] == cls]
        if not recs:
            continue
        print(f"\n  {CLASS_LABEL[cls]} ({len(recs)} rules):")
        for metric, label in [("C_v8", "C_v8 (Gaussian)"),
                              ("C_a", "C_a  (agnostic byte)"),
                              ("C_a_bp", "C_a_bp (bit-packed)"),
                              ("C_a_x8", "C_a_x8 (byte × 8)")]:
            vals = [r[metric] for r in recs]
            print(f"    {label:>25s}: mean={np.mean(vals):.4f}  "
                  f"min={min(vals):.4f}  max={max(vals):.4f}")
        # Raw gzip values
        print(f"    {'gz_byte':>25s}: mean={np.mean([r['gz_byte'] for r in recs]):.5f}")
        print(f"    {'gz_bp':>25s}: mean={np.mean([r['gz_bp'] for r in recs]):.3f}")
        print(f"    {'gz_x8':>25s}: mean={np.mean([r['gz_x8'] for r in recs]):.3f}")

    # ---- C4/C3 separation for all approaches ----
    print(f"\n{'=' * 100}")
    print("  C4/C3 SEPARATION COMPARISON")
    print("=" * 100)
    c4 = [r for r in records if r["class"] == 4]
    c3 = [r for r in records if r["class"] == 3]

    for metric, label in [("C_v8", "C_v8 (Gaussian)"),
                          ("C_a", "C_a  (agnostic byte)"),
                          ("C_a_bp", "C_a_bp (bit-packed)"),
                          ("C_a_x8", "C_a_x8 (byte × 8)")]:
        c4v = sorted([r[metric] for r in c4], reverse=True)
        c3v = sorted([r[metric] for r in c3], reverse=True)
        min_c4 = min(c4v)
        max_c3 = max(c3v)
        if max_c3 > 0:
            sep = min_c4 / max_c3
        elif min_c4 > 0:
            sep = float('inf')
        else:
            sep = 0
        # How many C4 in top N?
        ranked = sorted(records, key=lambda x: x[metric], reverse=True)
        c4_set = {r["rule"] for r in c4}
        top4 = {ranked[i]["rule"] for i in range(min(4, len(ranked)))}
        c4_in_top4 = len(c4_set & top4)

        print(f"  {label:>25s}: min(C4)={min_c4:.4f}  max(C3)={max_c3:.4f}  "
              f"sep={sep:.2f}x  C4_in_top4={c4_in_top4}/4")

    # ---- Gzip raw values: the key diagnostic ----
    print(f"\n{'=' * 100}")
    print("  GZIP RAW VALUES: C4 vs C3 (the key diagnostic)")
    print("=" * 100)
    print(f"\n  C4 rules:")
    for r in sorted(c4, key=lambda x: x["rule"]):
        print(f"    Rule {r['rule']:>3}: gz_byte={r['gz_byte']:.5f}  "
              f"gz_bp={r['gz_bp']:.3f}  gz_x8={r['gz_x8']:.3f}  "
              f"wG_bp={r['wG_a_bp']:.4f}  wG_x8={r['wG_a_x8']:.4f}")
    print(f"\n  C3 rules:")
    for r in sorted(c3, key=lambda x: x["rule"]):
        print(f"    Rule {r['rule']:>3}: gz_byte={r['gz_byte']:.5f}  "
              f"gz_bp={r['gz_bp']:.3f}  gz_x8={r['gz_x8']:.3f}  "
              f"wG_bp={r['wG_a_bp']:.4f}  wG_x8={r['wG_a_x8']:.4f}")

    # ---- Save CSV ----
    csv_path = os.path.join(_HERE, "eca_agnostic_rank.csv")
    fields = ["rank", "rule", "class", "C_a_bp", "C_a_x8", "C_a", "C_v8",
              "gz_byte", "gz_bp", "gz_x8", "wG_a_bp", "wG_a_x8",
              "mean_H", "std_H", "op_up", "op_down", "mi1", "decay", "tc_mean"]
    ranked = sorted(records, key=lambda x: x["C_a_bp"], reverse=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, rec in enumerate(ranked):
            row = {"rank": i + 1}
            row.update({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in rec.items() if k in fields})
            w.writerow(row)
    print(f"\n  CSV -> {csv_path}")


if __name__ == "__main__":
    main()
