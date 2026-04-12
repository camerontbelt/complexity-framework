"""
blind_sweep.py
==============
Blind parameter sweep: find peak C for Ising (q=2), Potts q=3, and Potts q=5
across multiple system sizes.  NO critical-point values appear in this file.
The comparison against literature T_c is done in a separate analysis script.

Design philosophy
-----------------
This script is deliberately blind to known physics.  It sweeps temperature for
each model, records every sub-metric, finds the peak C via quadratic
interpolation, and writes CSVs.  At no point does it reference, use, or encode
any literature critical temperature.

Models
------
  Ising (q=2):   Two-spin Potts on 2D square lattice, Metropolis update.
  Potts q=3:     Three-spin Potts, second-order transition.
  Potts q=5:     Five-spin Potts, FIRST-ORDER transition.
                 This is the critical test: does the ~35% offset hold when
                 the correlation length does NOT diverge?

Output
------
  blind_sweep_results.csv  — all raw data per (model, L, T, seed)
  blind_sweep_peaks.csv    — peak C location per (model, L)
  blind_sweep.png          — C vs T for each model, with peak markers
"""

import os, sys, csv, time, warnings, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Bootstrap compute_full_C from mnist-experiment.py (agnostic weights)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_nn   = os.path.join(os.path.dirname(_HERE), "neural-network")
_spec = importlib.util.spec_from_file_location(
            "mnist_exp", os.path.join(_nn, "mnist-experiment.py"))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
compute_full_C = _mod.compute_full_C

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SIZES    = [32, 64, 128]      # lattice side lengths
N_FRAMES = 100                # frames captured per run
WARMUP   = 500                # equilibration steps before measurement
N_SEEDS  = 6                  # independent realisations per condition

# Temperature grids — deliberately wide, no physics encoded.
# We want 20+ points for smooth interpolation.
MODELS = {
    "Ising (q=2)": {
        "q": 2,
        # Sweep 1.0 to 4.0 — somewhere in here is a transition
        "T_values": np.concatenate([
            np.linspace(1.0, 1.8, 5),
            np.linspace(1.9, 2.8, 19),   # dense in the middle
            np.linspace(2.9, 4.0, 5),
        ]),
    },
    "Potts q=3": {
        "q": 3,
        # Sweep 0.3 to 2.5
        "T_values": np.concatenate([
            np.linspace(0.30, 0.70, 5),
            np.linspace(0.75, 1.60, 18),  # dense in the middle
            np.linspace(1.70, 2.50, 5),
        ]),
    },
    "Potts q=5": {
        "q": 5,
        # Sweep 0.3 to 2.0
        "T_values": np.concatenate([
            np.linspace(0.30, 0.55, 4),
            np.linspace(0.60, 1.40, 17),  # dense in the middle
            np.linspace(1.50, 2.00, 4),
        ]),
    },
}

# De-duplicate and sort T_values
for cfg in MODELS.values():
    cfg["T_values"] = np.unique(np.round(cfg["T_values"], 4))

# ---------------------------------------------------------------------------
# Potts simulation (generalised: q=2 is Ising)
# ---------------------------------------------------------------------------

def potts_step(state, T, q):
    """One checkerboard Metropolis sweep for q-state Potts on a 2D lattice."""
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

        m_curr = count_matches(state)
        m_prop = count_matches(proposed)
        delta_E = -(m_prop - m_curr).astype(float)

        accept_prob = np.where(delta_E <= 0, 1.0, np.exp(-delta_E / T))
        accept = mask & (np.random.random((N, N)) < accept_prob)
        state  = np.where(accept, proposed, state)
    return state


def run_potts(q, T, L, seed):
    """Run a q-state Potts simulation and return binary frames."""
    rng = np.random.RandomState(seed)
    state = rng.randint(0, q, (L, L))
    for _ in range(WARMUP):
        state = potts_step(state, T, q)
    frames = []
    for _ in range(N_FRAMES):
        state = potts_step(state, T, q)
        # Binarise: spin == 0 (density ~1/q disordered, ~1 ordered)
        frames.append((state == 0).astype(np.float32))
    return frames


def frames_to_volumes(frames):
    """List[np.array (L,L)] -> list of (1, 1, L^2) volumes."""
    return [f.reshape(1, 1, -1) for f in frames]


def measure(frames):
    """Run compute_full_C on a set of binary frames."""
    vols = frames_to_volumes(frames)
    return compute_full_C(vols)

# ---------------------------------------------------------------------------
# Peak finding via quadratic interpolation
# ---------------------------------------------------------------------------

def find_peak(T_vals, C_means):
    """Find refined peak location via quadratic interpolation."""
    idx = int(np.argmax(C_means))

    # Try quadratic interpolation if peak is interior
    if 0 < idx < len(T_vals) - 1:
        x0, x1, x2 = T_vals[idx-1], T_vals[idx], T_vals[idx+1]
        y0, y1, y2 = C_means[idx-1], C_means[idx], C_means[idx+1]
        denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
        if abs(denom) > 1e-12:
            A = (x2*(y1-y0) + x1*(y0-y2) + x0*(y2-y1)) / denom
            if A < 0:  # concave — genuine peak
                B = (x2**2*(y0-y1) + x1**2*(y2-y0) + x0**2*(y1-y2)) / denom
                x_peak = -B / (2*A)
                if T_vals[0] <= x_peak <= T_vals[-1]:
                    C_peak = np.interp(x_peak, T_vals, C_means)
                    return float(x_peak), float(C_peak)

    return float(T_vals[idx]), float(C_means[idx])

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    out_dir = _HERE
    results_path = os.path.join(out_dir, "blind_sweep_results.csv")
    peaks_path   = os.path.join(out_dir, "blind_sweep_peaks.csv")
    plot_path    = os.path.join(out_dir, "blind_sweep.png")

    all_rows = []
    peak_records = []

    total_sims = sum(
        len(cfg["T_values"]) * len(SIZES) * N_SEEDS
        for cfg in MODELS.values()
    )
    sim_count = 0
    t_start = time.time()

    for model_name, cfg in MODELS.items():
        q = cfg["q"]
        T_values = cfg["T_values"]
        print(f"\n{'='*60}")
        print(f"  {model_name}  (q={q})")
        print(f"{'='*60}")

        for L in SIZES:
            print(f"\n  L = {L}")
            by_T = {}

            for T in T_values:
                c_vals = []
                for seed in range(N_SEEDS):
                    sim_count += 1
                    frames = run_potts(q, T, L, seed)
                    res = measure(frames)
                    c_vals.append(res["C_a"])

                    row = {
                        "model": model_name,
                        "q": q,
                        "L": L,
                        "T": round(T, 4),
                        "seed": seed,
                        "C_a": res["C_a"],
                        "mean_H": res.get("mean_H", 0),
                        "std_H": res.get("std_H", 0),
                        "op_up": res.get("op_up", 0),
                        "op_down": res.get("op_down", 0),
                        "mi1": res.get("mi1", 0),
                        "decay": res.get("decay", 0),
                        "tc_mean": res.get("tc_mean", 0),
                        "gzip_ratio": res.get("gzip_ratio", 0),
                    }
                    all_rows.append(row)

                mean_C = np.mean(c_vals)
                std_C  = np.std(c_vals)
                by_T[T] = (mean_C, std_C)

                elapsed = time.time() - t_start
                rate = sim_count / elapsed if elapsed > 0 else 0
                eta  = (total_sims - sim_count) / rate if rate > 0 else 0
                print(f"    T={T:.4f}  C={mean_C:.4f} ± {std_C:.4f}"
                      f"  [{sim_count}/{total_sims}  ETA {eta/60:.1f}m]")

            # Find peak for this (model, L)
            Ts = sorted(by_T.keys())
            means = [by_T[t][0] for t in Ts]
            peak_T, peak_C = find_peak(np.array(Ts), np.array(means))

            peak_records.append({
                "model": model_name,
                "q": q,
                "L": L,
                "peak_T": round(peak_T, 5),
                "peak_C": round(peak_C, 5),
            })
            print(f"  >>> Peak for {model_name} L={L}: T={peak_T:.4f}, C={peak_C:.4f}")

    # -------------------------------------------------------------------
    # Write CSVs
    # -------------------------------------------------------------------
    fieldnames = ["model", "q", "L", "T", "seed", "C_a",
                  "mean_H", "std_H", "op_up", "op_down",
                  "mi1", "decay", "tc_mean", "gzip_ratio"]
    with open(results_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nResults -> {results_path}")

    peak_fields = ["model", "q", "L", "peak_T", "peak_C"]
    with open(peaks_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=peak_fields)
        w.writeheader()
        w.writerows(peak_records)
    print(f"Peaks   -> {peaks_path}")

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("#0d0d1a")

    colours = {32: "#80cbc4", 64: "#ef9a9a", 128: "#ce93d8"}

    for ax, (model_name, cfg) in zip(axes, MODELS.items()):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

        for L in SIZES:
            rows = [r for r in all_rows
                    if r["model"] == model_name and r["L"] == L]
            by_T = {}
            for r in rows:
                by_T.setdefault(r["T"], []).append(r["C_a"])

            Ts = sorted(by_T.keys())
            means = [np.mean(by_T[t]) for t in Ts]
            stds  = [np.std(by_T[t]) for t in Ts]

            ax.errorbar(Ts, means, yerr=stds, fmt="o-",
                        color=colours[L], markersize=3, linewidth=1.2,
                        capsize=2, label=f"L={L}", alpha=0.85)

            # Mark peak
            peak = [p for p in peak_records
                    if p["model"] == model_name and p["L"] == L][0]
            ax.axvline(peak["peak_T"], color=colours[L],
                       linestyle="--", alpha=0.5, linewidth=1)

        ax.set_xlabel("Temperature T", fontsize=11)
        ax.set_ylabel("C (agnostic)", fontsize=11)
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, facecolor="#1e1e3f", labelcolor="white",
                  edgecolor="#444466")
        ax.grid(True, alpha=0.15, color="white")

    fig.suptitle(
        "Blind Parameter Sweep — Peak C Location\n"
        f"Sizes {SIZES}, {N_FRAMES} frames, {N_SEEDS} seeds, {WARMUP} warmup",
        fontsize=13, fontweight="bold", color="white", y=1.02)

    plt.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Plot    -> {plot_path}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} minutes ({sim_count} simulations).")


if __name__ == "__main__":
    main()
