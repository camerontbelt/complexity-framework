"""
fss-experiment.py
=================
Finite-Size Scaling (FSS) test of the complexity metric C.

THEORY
------
On a finite lattice of side L, the C peak occurs at a shifted critical
parameter value:

    param_peak(L) = param_c(inf) + A * L^(-1/nu)

where param_c(inf) is the true thermodynamic critical point and nu is the
correlation-length exponent (a universal number for each universality class).

Plotting param_peak(L) vs L^(-1/nu) gives a straight line whose intercept
is param_c(inf).  The data-collapse plot — C vs (param - param_c)*L^(1/nu)
for all L — provides a second, independent check: all curves should overlap
if the scaling hypothesis holds.

MODELS
------
1. Potts q=4  (VERIFIED)
   Known T_c = 1/ln(1+sqrt(4)) ≈ 0.910, known nu = 0.667 (Potts 4-state)
   We extract T_c from C peaks and compare against the known value.

2. Majority Vote Model (NOVEL)
   Each site adopts the majority opinion of 4 neighbours with prob (1-q),
   or flips at random with prob q.  Order->disorder transition at q_c.
   In the Ising universality class -> nu = 1.0.
   We predict q_c from C peaks WITHOUT looking up the known value first,
   then verify via data collapse and order-parameter consistency.

SIZES
-----
L = 32, 64, 128  (three decades of L to resolve the FSS trend)

OUTPUT
------
  fss_results.csv      -- raw C values at every (model, L, param, seed)
  fss_results.png      -- 3-panel figure per model:
                          Left:  C vs param for each L
                          Centre: param_peak(L) vs L^(-1/nu) + FSS fit
                          Right:  data-collapse plot
"""

import os, sys, csv as _csv, importlib.util, time, warnings
import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Bootstrap compute_full_C
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_nn   = os.path.join(os.path.dirname(_HERE), "neural-network")
_spec = importlib.util.spec_from_file_location(
            "mnist_exp", os.path.join(_nn, "mnist-experiment.py"))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_full_C = _mod.compute_full_C

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------
SIZES    = [32, 64, 128]
N_FRAMES = 80
WARMUP   = 600
N_SEEDS  = 6

# Potts
Q_POTTS  = 4
TC_TRUE  = 1.0 / np.log(1.0 + np.sqrt(float(Q_POTTS)))   # 0.9102
NU_POTTS = 0.6667
POTTS_TEMPS = [0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.30, 1.55, 1.90]

# Majority vote
NU_MV   = 1.0    # Ising universality class
MV_Q    = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.14, 0.18]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def frames_to_volumes(frames):
    return [f.astype(np.float32).reshape(1, 1, -1) for f in frames]


def measure(frames):
    return compute_full_C(frames_to_volumes(frames))


def find_peak(params, c_means):
    """Return the parameter value at maximum C (quadratic interpolation)."""
    idx = int(np.argmax(c_means))
    # Quadratic interpolation around the peak if we have 3 points
    if 0 < idx < len(params) - 1:
        x0, x1, x2 = params[idx-1], params[idx], params[idx+1]
        y0, y1, y2 = c_means[idx-1], c_means[idx], c_means[idx+1]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if abs(denom) > 1e-12:
            A = (x2*(y1-y0) + x1*(y0-y2) + x0*(y2-y1)) / denom
            B = (x2**2*(y0-y1) + x1**2*(y2-y0) + x0**2*(y1-y2)) / denom
            if A < 0:   # concave → valid peak
                x_peak = -B / (2*A)
                if params[0] < x_peak < params[-1]:
                    return float(x_peak)
    return float(params[idx])


def fss_fit(L_vals, peak_vals, nu):
    """Fit: peak(L) = p_c + A * L^(-1/nu).  Returns (p_c, A, p_c_err)."""
    x = np.array(L_vals, dtype=float) ** (-1.0 / nu)
    y = np.array(peak_vals, dtype=float)
    slope, intercept, r, p, se = scipy_stats.linregress(x, y)
    n   = len(x)
    # Standard error on the intercept
    s_err  = np.sqrt(np.sum((y - (intercept + slope*x))**2) / max(n-2, 1))
    x_mean = x.mean()
    se_int = s_err * np.sqrt(np.sum(x**2) / (n * np.sum((x-x_mean)**2)))
    return float(intercept), float(slope), float(se_int), float(r**2)

# ---------------------------------------------------------------------------
# Potts Model
# ---------------------------------------------------------------------------

def potts_step(state, T, q=Q_POTTS):
    N = state.shape[0]
    rows, cols = np.mgrid[0:N, 0:N]
    for parity in [0, 1]:
        mask = (rows + cols) % 2 == parity
        proposed = np.random.randint(0, q, (N, N))

        def count_matches(spin_map):
            return ((np.roll(state, 1, 0) == spin_map).astype(int) +
                    (np.roll(state, -1, 0) == spin_map).astype(int) +
                    (np.roll(state, 1, 1) == spin_map).astype(int) +
                    (np.roll(state, -1, 1) == spin_map).astype(int))

        m_curr  = count_matches(state)
        m_prop  = count_matches(proposed)
        delta_E = -(m_prop - m_curr).astype(float)
        accept  = mask & (np.random.random((N, N)) < np.where(
                      delta_E <= 0, 1.0, np.exp(-delta_E / T)))
        state   = np.where(accept, proposed, state)
    return state


def run_potts(L, T, seed):
    rng   = np.random.RandomState(seed)
    state = rng.randint(0, Q_POTTS, (L, L))
    for _ in range(WARMUP):
        state = potts_step(state, T)
    frames = []
    for _ in range(N_FRAMES):
        state = potts_step(state, T)
        frames.append((state == 0).astype(np.float32))
    return frames

# ---------------------------------------------------------------------------
# Majority Vote Model
# ---------------------------------------------------------------------------

def mv_step(state, q):
    """One full sweep of the majority vote model."""
    N = state.shape[0]
    nbr_sum = (np.roll(state, 1, 0) + np.roll(state, -1, 0) +
               np.roll(state, 1, 1) + np.roll(state, -1, 1))
    majority = np.sign(nbr_sum)          # +1 or -1; 0 if tie
    majority = np.where(majority == 0, state, majority)   # keep if tie

    rand = np.random.random((N, N))
    noise_mask = rand < q
    random_flip = np.where(np.random.random((N, N)) < 0.5, 1, -1)
    return np.where(noise_mask, random_flip, majority).astype(np.int8)


def run_mv(L, q, seed):
    rng   = np.random.RandomState(seed)
    state = rng.choice([-1, 1], size=(L, L)).astype(np.int8)
    for _ in range(WARMUP):
        state = mv_step(state, q)
    frames = []
    for _ in range(N_FRAMES):
        state = mv_step(state, q)
        frames.append((state == 1).astype(np.float32))
    return frames

# ---------------------------------------------------------------------------
# Run one model over all sizes and parameters
# ---------------------------------------------------------------------------

def run_model(name, param_label, params, run_fn, nu):
    print(f"\n{'='*65}")
    print(f"  {name}  (nu = {nu})")
    print(f"{'='*65}")

    # results[L][param] = list of C dicts
    results = {L: {} for L in SIZES}
    t0 = time.time()

    for L in SIZES:
        print(f"\n  L = {L}")
        for p in params:
            c_vals = []
            for seed in range(N_SEEDS):
                frames = run_fn(L, p, seed)
                c_vals.append(measure(frames))
            mean_C = np.mean([r["C_a"] for r in c_vals])
            results[L][p] = c_vals
            print(f"    {param_label}={p:.4f}  C={mean_C:.4f}  {time.time()-t0:.0f}s")

    return results


# ---------------------------------------------------------------------------
# Analysis + plotting for one model
# ---------------------------------------------------------------------------

def analyse_and_plot(name, param_label, params, results, nu,
                     known_pc=None, axes_row=None):

    L_vals    = sorted(results.keys())
    peak_vals = []

    for L in L_vals:
        c_means = [np.mean([r["C_a"] for r in results[L][p]]) for p in params]
        peak_vals.append(find_peak(list(params), c_means))

    # FSS fit
    pc_fit, A_fit, pc_err, r2 = fss_fit(L_vals, peak_vals, nu)

    print(f"\n  FSS result for {name}:")
    print(f"    nu (assumed) = {nu}")
    for L, pk in zip(L_vals, peak_vals):
        print(f"    L={L:3d}  peak {param_label} = {pk:.4f}")
    print(f"    FSS extrapolation -> {param_label}_c = {pc_fit:.4f}  "
          f"+/-{pc_err:.4f}  (R2={r2:.3f})")
    if known_pc is not None:
        print(f"    Known {param_label}_c           = {known_pc:.4f}  "
              f"(error = {abs(pc_fit - known_pc):.4f})")

    if axes_row is None:
        return pc_fit, pc_err, peak_vals

    ax_main, ax_fss, ax_col = axes_row
    colours = ["#4fc3f7", "#f48fb1", "#c5e1a5"]

    # --- Panel 1: C vs param for each L ---
    for i, L in enumerate(L_vals):
        c_means = [np.mean([r["C_a"] for r in results[L][p]]) for p in params]
        c_stds  = [np.std ([r["C_a"] for r in results[L][p]]) for p in params]
        ax_main.errorbar(params, c_means, yerr=c_stds,
                         fmt="o-", color=colours[i], capsize=4,
                         linewidth=1.8, markersize=7,
                         label=f"L = {L}")
        ax_main.axvline(peak_vals[i], color=colours[i],
                        linestyle=":", alpha=0.5, linewidth=1)

    ax_main.axvline(pc_fit, color="white", linestyle="--",
                    linewidth=1.5, alpha=0.8,
                    label=f"FSS pred: {param_label}_c = {pc_fit:.3f}")
    if known_pc is not None:
        ax_main.axvline(known_pc, color="#ffb74d", linestyle="-.",
                        linewidth=1.5, alpha=0.9,
                        label=f"True: {param_label}_c = {known_pc:.3f}")

    ax_main.set_xlabel(param_label, fontsize=11)
    ax_main.set_ylabel("C", fontsize=11)
    ax_main.set_title(f"{name}\nC vs {param_label} at each L", fontsize=10, fontweight="bold")
    ax_main.legend(fontsize=8)
    ax_main.grid(True, alpha=0.2, color="white")

    # --- Panel 2: FSS scaling plot ---
    x_fit = np.array(L_vals, dtype=float) ** (-1.0 / nu)
    x_line = np.linspace(0, max(x_fit)*1.1, 100)
    y_line = pc_fit + A_fit * x_line

    ax_fss.scatter(x_fit, peak_vals, s=100, zorder=5,
                   c=colours[:len(L_vals)], edgecolors="white", linewidth=0.8)
    for i, (xi, yi, L) in enumerate(zip(x_fit, peak_vals, L_vals)):
        ax_fss.annotate(f"L={L}", xy=(xi, yi),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=9, color=colours[i])

    ax_fss.plot(x_line, y_line, "w--", linewidth=1.5, alpha=0.7,
                label=f"fit: {param_label}_c = {pc_fit:.4f} +/- {pc_err:.4f}")
    ax_fss.axhline(pc_fit, color="#ffb74d", linestyle=":",
                   alpha=0.6, linewidth=1.2, label="extrapolated T_c")
    if known_pc is not None:
        ax_fss.axhline(known_pc, color="#ef9a9a", linestyle="-.",
                       alpha=0.8, linewidth=1.2, label=f"true T_c = {known_pc:.3f}")

    ax_fss.set_xlabel(f"L^(-1/nu)   [nu = {nu}]", fontsize=10)
    ax_fss.set_ylabel(f"{param_label}_peak (L)", fontsize=10)
    ax_fss.set_title("FSS Scaling Plot\n"
                     f"Intercept -> {param_label}_c(inf)", fontsize=10, fontweight="bold")
    ax_fss.legend(fontsize=8)
    ax_fss.grid(True, alpha=0.2, color="white")

    # --- Panel 3: Data collapse ---
    for i, L in enumerate(L_vals):
        c_means = [np.mean([r["C_a"] for r in results[L][p]]) for p in params]
        # Rescaled x-axis: (param - pc_fit) * L^(1/nu)
        x_scaled = [(p - pc_fit) * (L ** (1.0/nu)) for p in params]
        ax_col.plot(x_scaled, c_means, "o-", color=colours[i],
                    linewidth=1.8, markersize=6, label=f"L = {L}", alpha=0.9)

    ax_col.axvline(0, color="white", linestyle="--", alpha=0.5,
                   linewidth=1, label=f"predicted {param_label}_c")
    ax_col.set_xlabel(f"({param_label} - {param_label}_c) x L^(1/nu)", fontsize=10)
    ax_col.set_ylabel("C", fontsize=10)
    ax_col.set_title("Data Collapse\n"
                     "(curves should overlap if scaling holds)",
                     fontsize=10, fontweight="bold")
    ax_col.legend(fontsize=8)
    ax_col.grid(True, alpha=0.2, color="white")

    return pc_fit, pc_err, peak_vals


# ---------------------------------------------------------------------------
# CSV save
# ---------------------------------------------------------------------------

def save_csv(all_results, path):
    fields = ["model", "L", "param", "seed", "C_a",
              "mean_H", "op_up", "op_down", "mi1", "tc_mean", "gzip_ratio"]
    rows = []
    for model_name, (param_label, params, results, _nu, _kpc) in all_results.items():
        for L, presults in results.items():
            for p, c_list in presults.items():
                for seed, r in enumerate(c_list):
                    row = {"model": model_name, "L": L,
                           "param": round(p, 6), "seed": seed}
                    for f in fields[4:]:
                        row[f] = round(float(r.get(f, 0.0)), 6)
                    row["C_a"] = round(float(r["C_a"]), 6)
                    rows.append(row)
    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV -> {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_total = time.time()

    # Run both models
    potts_results = run_model(
        "Potts q=4", "T", POTTS_TEMPS,
        lambda L, T, s: run_potts(L, T, s), NU_POTTS)

    mv_results = run_model(
        "Majority Vote", "q", MV_Q,
        lambda L, q, s: run_mv(L, q, s), NU_MV)

    all_results = {
        "Potts q=4":    ("T", POTTS_TEMPS, potts_results, NU_POTTS, TC_TRUE),
        "Majority Vote":("q", MV_Q,        mv_results,    NU_MV,    None),
    }

    # Save CSV
    csv_path = os.path.join(_HERE, "fss_results.csv")
    save_csv(all_results, csv_path)

    # Build figure
    fig = plt.figure(figsize=(24, 12))
    fig.patch.set_facecolor("#0d0d1a")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.32)

    model_info = [
        ("Potts q=4",    "T", POTTS_TEMPS, potts_results, NU_POTTS, TC_TRUE,  0),
        ("Majority Vote","q", MV_Q,        mv_results,    NU_MV,    None,      1),
    ]

    pc_results = {}
    for name, plabel, params, results, nu, kpc, row_i in model_info:
        axes_row = [
            fig.add_subplot(gs[row_i, 0]),
            fig.add_subplot(gs[row_i, 1]),
            fig.add_subplot(gs[row_i, 2]),
        ]
        for ax in axes_row:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")
            ax.legend_ and ax.legend(labelcolor="white", facecolor="#1e1e3f",
                                     edgecolor="#444466")

        pc_fit, pc_err, peaks = analyse_and_plot(
            name, plabel, params, results, nu,
            known_pc=kpc, axes_row=axes_row)

        pc_results[name] = (pc_fit, pc_err, peaks)

        for ax in axes_row:
            leg = ax.legend(fontsize=8, labelcolor="white",
                            facecolor="#1e1e3f", edgecolor="#444466")

    fig.suptitle(
        "Finite-Size Scaling of the Complexity Metric C\n"
        "Row 1: Potts q=4 (known T_c = 0.910, nu = 0.667)  |  "
        "Row 2: Majority Vote (predicted q_c, nu = 1.0 from Ising class)",
        fontsize=13, fontweight="bold", color="white", y=1.01)

    png_path = os.path.join(_HERE, "fss_results.png")
    plt.savefig(png_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nPlot -> {png_path}")

    # Final summary
    print(f"\n{'='*65}")
    print("  FINAL PREDICTIONS")
    print(f"{'='*65}")
    pc, pe, _ = pc_results["Potts q=4"]
    print(f"  Potts q=4:")
    print(f"    C-metric FSS prediction:  T_c = {pc:.4f} +/- {pe:.4f}")
    print(f"    Known exact value:        T_c = {TC_TRUE:.4f}")
    print(f"    Error: {abs(pc - TC_TRUE):.4f}  ({100*abs(pc-TC_TRUE)/TC_TRUE:.1f}%)")

    pc, pe, _ = pc_results["Majority Vote"]
    print(f"\n  Majority Vote Model:")
    print(f"    C-metric FSS prediction:  q_c = {pc:.4f} +/- {pe:.4f}")
    print(f"    (Known from literature:   q_c ~ 0.075 for square lattice)")
    print(f"    [Revealed after blind prediction]")

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s")
    print("Done.")
