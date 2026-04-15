"""
gzip_gate_analysis.py
=====================
Test Cameron's idea: multiply byte-encoded gzip ratio by 8 to recover
the "true" compression ratio, then apply a tanh gate.

Hypothesis: C4 rules have true_gzip < 1 (compressible), C3 rules have
true_gzip >= 1 (incompressible), so gate(true_gzip) naturally discriminates.
"""

import csv, os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

def gate(x, K=10):
    """Standard tanh gate: zero at 0 and 1, positive in between."""
    return np.tanh(K * x) * np.tanh(K * (1.0 - x))

def gate_clamped(x, K=10):
    """Gate clamped to non-negative (for x > 1)."""
    return max(0.0, gate(x, K))

def main():
    path = os.path.join(_HERE, "eca_raw_metrics.csv")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    print("=" * 85)
    print("  gzip × 8 ANALYSIS: Can tanh gates replace the Gaussian?")
    print("=" * 85)

    # Group by class
    by_class = {}
    for r in rows:
        cls = r["class"]
        rule = int(r["rule"])
        gzip_raw = float(r["gzip"])
        true_gzip = gzip_raw * 8.0
        by_class.setdefault(cls, []).append({
            "rule": rule,
            "gzip_raw": gzip_raw,
            "true_gzip": true_gzip,
        })

    # Summary by class
    print(f"\n{'Class':<6} {'Count':>5} {'gzip_raw':>10} {'gzip×8':>10} "
          f"{'min(×8)':>10} {'max(×8)':>10}")
    print("-" * 60)
    for cls in ["C4", "C3", "C2", "C1"]:
        entries = by_class.get(cls, [])
        if not entries:
            continue
        raws = [e["gzip_raw"] for e in entries]
        trues = [e["true_gzip"] for e in entries]
        print(f"{cls:<6} {len(entries):>5} {np.mean(raws):>10.5f} "
              f"{np.mean(trues):>10.4f} {min(trues):>10.4f} {max(trues):>10.4f}")

    # Detailed C4 and C3 comparison
    print(f"\n{'=' * 85}")
    print("  DETAILED: C4 vs C3 rules")
    print(f"{'=' * 85}")

    for cls in ["C4", "C3"]:
        entries = by_class.get(cls, [])
        if not entries:
            continue
        print(f"\n  {cls} rules:")
        print(f"  {'Rule':>6} {'gzip_raw':>10} {'gzip×8':>10} "
              f"{'gate(K=10)':>12} {'gate(K=5)':>10} {'gate(K=20)':>11}")
        for e in sorted(entries, key=lambda x: x["rule"]):
            x = e["true_gzip"]
            g10 = gate_clamped(x, K=10)
            g5  = gate_clamped(x, K=5)
            g20 = gate_clamped(x, K=20)
            marker = " <-- above 1!" if x >= 1.0 else ""
            print(f"  {e['rule']:>6} {e['gzip_raw']:>10.5f} {x:>10.4f} "
                  f"{g10:>12.6f} {g5:>10.6f} {g20:>11.6f}{marker}")

    # The key question: can the gate discriminate?
    print(f"\n{'=' * 85}")
    print("  DISCRIMINATION ANALYSIS")
    print(f"{'=' * 85}")

    for K in [5, 10, 20, 50]:
        c4_gates = [gate_clamped(e["true_gzip"], K) for e in by_class.get("C4", [])]
        c3_gates = [gate_clamped(e["true_gzip"], K) for e in by_class.get("C3", [])]
        c2_gates = [gate_clamped(e["true_gzip"], K) for e in by_class.get("C2", [])]
        c1_gates = [gate_clamped(e["true_gzip"], K) for e in by_class.get("C1", [])]

        print(f"\n  K = {K}:")
        for name, vals in [("C4", c4_gates), ("C3", c3_gates),
                           ("C2", c2_gates), ("C1", c1_gates)]:
            if vals:
                print(f"    {name}: mean={np.mean(vals):.4f}, "
                      f"min={min(vals):.4f}, max={max(vals):.4f}")
        if c4_gates and c3_gates:
            sep = np.mean(c4_gates) / max(np.mean(c3_gates), 1e-10)
            print(f"    C4/C3 separation: {sep:.1f}×")

    # Alternative: one-sided gate (only penalise incompressibility)
    print(f"\n{'=' * 85}")
    print("  ALTERNATIVE: One-sided gate — tanh(K*x) * softplus(1-x)")
    print("  Only penalises x >= 1 (incompressible), rewards all x < 1")
    print(f"{'=' * 85}")

    def one_sided(x, K=10):
        """Penalise x >= 1 but not x near 0 (let entropy gate handle that)."""
        return np.tanh(K * x) * max(0.0, np.tanh(K * (1.0 - x)))

    for cls in ["C4", "C3"]:
        entries = by_class.get(cls, [])
        if not entries:
            continue
        print(f"\n  {cls}:")
        for e in sorted(entries, key=lambda x: x["rule"]):
            x = e["true_gzip"]
            v = one_sided(x, K=10)
            print(f"    Rule {e['rule']:>3}: gzip×8={x:.4f}  one_sided={v:.4f}")

    # What about using the raw gzip with gate(8*x) centered differently?
    print(f"\n{'=' * 85}")
    print("  WHAT IF we bit-pack instead of ×8?")
    print("  From prior experiments, actual bit-packed ratios:")
    print("    C4 Rule 110: 0.781   C4 Rule 124: 0.678")
    print("    C3 Rule 30:  1.004   C1 Rule 0:   0.009")
    print(f"{'=' * 85}")

    bp_examples = [
        ("C4 Rule 110", 0.781),
        ("C4 Rule 124", 0.678),
        ("C3 Rule 30",  1.004),
        ("C1 Rule 0",   0.009),
    ]
    print(f"\n  {'System':<16} {'bit-packed':>12} {'gate(K=10)':>12}")
    for name, x in bp_examples:
        g = gate_clamped(x, K=10)
        print(f"  {name:<16} {x:>12.3f} {g:>12.4f}")

    print(f"\n  Bit-packed values are clearly better for tanh gates:")
    print(f"  C4 at 0.68-0.78 gives gate values ~0.97-0.99")
    print(f"  C3 at 1.0 gives gate value 0.0")
    print(f"  PERFECT discrimination.")
    print(f"\n  The ×8 approximation overcorrects because gzip achieves")
    print(f"  extra compression on byte-padded data (patterns in the padding).")
    print(f"  Actual bit-packing is the clean, parameter-free path.")


if __name__ == "__main__":
    main()
