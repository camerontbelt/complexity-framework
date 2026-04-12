"""
fss_analysis.py
===============
Finite-size scaling analysis of C from the blind sweep data.

Tests whether C follows standard FSS:
  T_peak(L) = T_c + A * L^(-1/nu)
  C_peak(L) ~ L^(gamma_C / nu)

If C follows FSS with known critical exponents, it means C is
coupled to the same diverging correlation length that governs
the phase transition — i.e., C is measuring something genuinely physical.

Also attempts data collapse: plotting C/C_peak vs (T-T_c)*L^(1/nu)
should produce a single universal curve if FSS holds.
"""

import os, csv, warnings
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))

# Known critical exponents (exact results for 2D square lattice)
MODELS = {
    "Ising (q=2)": {
        "T_c_exact": 2.0 / np.log(1 + np.sqrt(2)),     # 2.2692
        "nu": 1.0,                                       # exact
        "gamma": 7.0/4.0,                                # exact
        "order": "2nd",
    },
    "Potts q=3": {
        "T_c_exact": 1.0 / np.log(1 + np.sqrt(3)),     # 0.9950
        "nu": 5.0/6.0,                                   # exact
        "gamma": 13.0/9.0,                               # exact
        "order": "2nd",
    },
    "Potts q=5": {
        "T_c_exact": 1.0 / np.log(1 + np.sqrt(5)),     # 0.8515
        "nu": None,                                       # first-order: no divergence
        "gamma": None,
        "order": "1st",
    },
}

SIZES = [32, 64, 128]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_csv(name):
    path = os.path.join(_HERE, name)
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def get_curve(rows, model, L):
    subset = [r for r in rows if r["model"] == model and int(r["L"]) == L]
    by_T = {}
    for r in subset:
        T = float(r["T"])
        by_T.setdefault(T, []).append(float(r["C_a"]))
    Ts = sorted(by_T.keys())
    means = np.array([np.mean(by_T[t]) for t in Ts])
    stds  = np.array([np.std(by_T[t]) for t in Ts])
    return np.array(Ts), means, stds


# ---------------------------------------------------------------------------
# FSS fitting
# ---------------------------------------------------------------------------
def fss_free(L, T_c, A, inv_nu):
    """T_peak(L) = T_c + A * L^(-inv_nu)"""
    return T_c + A * L**(-inv_nu)


def main():
    rows  = load_csv("blind_sweep_results.csv")
    peaks = load_csv("blind_sweep_peaks.csv")
    if not rows or not peaks:
        return

    # Parse peaks
    peak_data = {}
    for p in peaks:
        model = p["model"]
        L = int(p["L"])
        peak_data.setdefault(model, {})[L] = {
            "T": float(p["peak_T"]),
            "C": float(p["peak_C"]),
        }

    DIVIDER = "=" * 78

    # ==================================================================
    # 1. PEAK LOCATION FSS
    # ==================================================================
    print(f"\n{DIVIDER}")
    print("  FINITE-SIZE SCALING: T_peak(L) = T_c + A * L^(-1/nu)")
    print(f"{DIVIDER}\n")

    fss_results = {}

    for model, info in MODELS.items():
        pd = peak_data.get(model, {})
        if len(pd) < 3:
            continue

        Ls = np.array(sorted(pd.keys()), dtype=float)
        T_peaks = np.array([pd[int(L)]["T"] for L in Ls])
        T_c_exact = info["T_c_exact"]

        print(f"  {model}  (T_c exact = {T_c_exact:.4f}, order = {info['order']})")
        print(f"  {'L':>6} {'T_peak':>10} {'T_peak - T_c':>14}")
        for L, Tp in zip(Ls, T_peaks):
            print(f"  {int(L):>6} {Tp:>10.4f} {Tp - T_c_exact:>+14.4f}")

        # Free 3-parameter fit
        try:
            popt, pcov = curve_fit(fss_free, Ls, T_peaks,
                                   p0=[T_c_exact, 1.0, 1.0],
                                   maxfev=10000)
            T_c_fit, A_fit, inv_nu_fit = popt
            perr = np.sqrt(np.diag(pcov))
            nu_fit = 1.0 / inv_nu_fit if abs(inv_nu_fit) > 0.01 else float("inf")

            print(f"\n  Free fit (3 params):")
            print(f"    T_c   = {T_c_fit:.4f} +/- {perr[0]:.4f}  "
                  f"(exact: {T_c_exact:.4f}, dev: {abs(T_c_fit - T_c_exact):.4f})")
            print(f"    A     = {A_fit:.4f}")
            print(f"    1/nu  = {inv_nu_fit:.4f}  (nu = {nu_fit:.3f})")
            if info["nu"] is not None:
                print(f"    Known nu = {info['nu']:.4f}  "
                      f"(fit deviation: {abs(nu_fit - info['nu']):.3f})")

            fss_results[model] = {
                "T_c_fit": T_c_fit, "A": A_fit, "inv_nu": inv_nu_fit,
                "nu_fit": nu_fit,
            }
        except Exception as e:
            print(f"\n  Free fit FAILED: {e}")

        # Fixed-nu fit (second-order only)
        if info["nu"] is not None:
            nu_known = info["nu"]
            def fss_fixed(L, T_c, A):
                return T_c + A * L**(-1.0/nu_known)
            try:
                popt2, pcov2 = curve_fit(fss_fixed, Ls, T_peaks,
                                         p0=[T_c_exact, 1.0], maxfev=10000)
                T_c_con, A_con = popt2
                perr2 = np.sqrt(np.diag(pcov2))
                print(f"\n  Fixed nu = {nu_known:.4f}:")
                print(f"    T_c   = {T_c_con:.4f} +/- {perr2[0]:.4f}")
                print(f"    Deviation from exact T_c: "
                      f"{abs(T_c_con - T_c_exact):.4f} "
                      f"({abs(T_c_con - T_c_exact)/T_c_exact*100:.1f}%)")
            except Exception as e:
                print(f"\n  Fixed-nu fit FAILED: {e}")

        # For first-order: pseudo-scaling with L^(-d) = L^(-2)
        if info["order"] == "1st":
            def fss_first_order(L, T_c, A):
                return T_c + A * L**(-2)
            try:
                popt3, pcov3 = curve_fit(fss_first_order, Ls, T_peaks,
                                         p0=[T_c_exact, 100.0], maxfev=10000)
                T_c_fo, A_fo = popt3
                perr3 = np.sqrt(np.diag(pcov3))
                print(f"\n  First-order pseudo-scaling (1/nu = d = 2):")
                print(f"    T_c   = {T_c_fo:.4f} +/- {perr3[0]:.4f}")
                print(f"    Deviation from exact T_c: "
                      f"{abs(T_c_fo - T_c_exact):.4f} "
                      f"({abs(T_c_fo - T_c_exact)/T_c_exact*100:.1f}%)")
            except Exception as e:
                print(f"\n  First-order fit FAILED: {e}")

        print()

    # ==================================================================
    # 2. PEAK HEIGHT SCALING
    # ==================================================================
    print(f"{DIVIDER}")
    print("  PEAK HEIGHT SCALING: C_peak(L) ~ L^alpha")
    print(f"{DIVIDER}\n")

    for model, info in MODELS.items():
        pd = peak_data.get(model, {})
        if len(pd) < 3:
            continue

        Ls = np.array(sorted(pd.keys()), dtype=float)
        C_peaks = np.array([pd[int(L)]["C"] for L in Ls])

        log_L = np.log(Ls)
        log_C = np.log(C_peaks)
        alpha, const = np.polyfit(log_L, log_C, 1)

        print(f"  {model}:")
        for L, Cp in zip(Ls, C_peaks):
            print(f"    L={int(L):>4}  C_peak = {Cp:.4f}")
        print(f"    Fit: C_peak ~ L^({alpha:.3f})")

        if info["gamma"] is not None and info["nu"] is not None:
            expected = info["gamma"] / info["nu"]
            print(f"    Expected gamma/nu for susceptibility: {expected:.3f}")
            print(f"    Ratio (measured/expected): {alpha / expected:.3f}")
        print()

    # ==================================================================
    # 3. MONOTONICITY CHECK — does peak shift TOWARD T_c with L?
    # ==================================================================
    print(f"{DIVIDER}")
    print("  KEY QUESTION: Does T_peak shift toward T_c as L increases?")
    print(f"{DIVIDER}\n")

    for model, info in MODELS.items():
        pd = peak_data.get(model, {})
        if len(pd) < 3:
            continue
        Ls = sorted(pd.keys())
        T_c = info["T_c_exact"]
        offsets = [abs(pd[L]["T"] - T_c) for L in Ls]

        converging = all(offsets[i] >= offsets[i+1] - 0.001 for i in range(len(offsets)-1))
        print(f"  {model}:")
        for L, off in zip(Ls, offsets):
            print(f"    L={L:>4}  |T_peak - T_c| = {off:.4f}")
        print(f"    Converging toward T_c? {'YES' if converging else 'NO'}")
        if converging:
            print(f"    This is CONSISTENT with FSS: the offset is a finite-size effect")
        else:
            print(f"    This VIOLATES standard FSS: offset is NOT decreasing with L")
        print()

    # ==================================================================
    # 4. DATA COLLAPSE PLOT
    # ==================================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor("#0d0d1a")

    colours = {32: "#80cbc4", 64: "#ef9a9a", 128: "#ce93d8"}

    for col_idx, (model, info) in enumerate(MODELS.items()):
        T_c = info["T_c_exact"]
        nu = info["nu"] if info["nu"] else 0.5

        # Top row: raw C vs T
        ax = axes[0, col_idx]
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

        for L in SIZES:
            Ts, means, stds = get_curve(rows, model, L)
            ax.errorbar(Ts, means, yerr=stds, fmt="o-", color=colours[L],
                        markersize=3, linewidth=1.2, capsize=2,
                        label=f"L={L}", alpha=0.85)

        ax.axvline(T_c, color="yellow", linestyle=":", alpha=0.7,
                   linewidth=1.5, label=f"T_c={T_c:.3f}")

        # Mark peaks
        pd = peak_data.get(model, {})
        for L in SIZES:
            if L in pd:
                ax.axvline(pd[L]["T"], color=colours[L], linestyle="--",
                           alpha=0.4, linewidth=1)

        ax.set_xlabel("T")
        ax.set_ylabel("C")
        ax.set_title(f"{model}\nRaw C vs T", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, facecolor="#1e1e3f", labelcolor="white",
                  edgecolor="#444466")
        ax.grid(True, alpha=0.15, color="white")

        # Bottom row: data collapse
        ax = axes[1, col_idx]
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

        for L in SIZES:
            Ts, means, stds = get_curve(rows, model, L)
            C_peak = pd.get(L, {}).get("C", 1.0)

            x_scaled = (Ts - T_c) * L**(1.0/nu)
            y_scaled = means / C_peak

            ax.plot(x_scaled, y_scaled, "o-", color=colours[L],
                    markersize=3, linewidth=1.2,
                    label=f"L={L}", alpha=0.85)

        ax.axvline(0, color="yellow", linestyle=":", alpha=0.5, linewidth=1)
        ax.set_xlabel(f"(T - T_c) * L^(1/nu)   [nu={nu:.3f}]")
        ax.set_ylabel("C / C_peak")
        nu_label = f"nu={nu:.3f}" if info["nu"] else "nu=0.5 (guess)"
        ax.set_title(f"{model}\nData Collapse ({nu_label})",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, facecolor="#1e1e3f", labelcolor="white",
                  edgecolor="#444466")
        ax.grid(True, alpha=0.15, color="white")

    fig.suptitle("Finite-Size Scaling Analysis of C",
                 fontsize=14, fontweight="bold", color="white", y=1.02)
    plt.tight_layout()
    out = os.path.join(_HERE, "fss_analysis.png")
    fig.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot -> {out}")
    print("Done.")


if __name__ == "__main__":
    main()
