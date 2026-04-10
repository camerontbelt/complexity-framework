"""
lattice-experiments.py
======================
Three classical 2-D lattice models measured with the existing complexity
metric C (compute_full_C from mnist-experiment.py — the agnostic tanh weights).

Each frame of the simulation is flattened to a 1-D binary vector and fed to
compute_full_C exactly as the Gray-Scott / ECA experiments did.  No multi-scale
pooling layer is added; this is a straight "does C track the phase transition?"
test.

EXPERIMENTS
-----------
1. Noisy Voter Model
   - 2-D grid where each cell copies a random neighbour with prob (1-eps), or
     picks a random opinion with prob eps.  eps=0 → pure ordering (low C),
     eps=1 → pure noise (low C via different mechanism), intermediate → rich.
   H1: C peaks at an intermediate noise rate eps* that is neither ordered
       nor purely random.
   H0: C is flat or monotone across eps values.

2. Potts Model  (q = 4, Metropolis-Hastings)
   - kT_c = 1/ln(1+sqrt(4)) ≈ 0.910 on the 2-D square lattice.
   - Binarised as (spin == 0) — density ~1/q disordered, ~1 ordered.
   H1: C peaks near T_c; ordered and disordered phases score lower.
   H0: C does not discriminate temperature regimes.

3. Contact Process  (synchronous, recovery = 0.5)
   - Occupied sites (1) recover with prob r=0.5; empty sites (0) become
     occupied with prob 1-(1-lam/4)^n_occupied_neighbours.
   - Mean-field critical point: lam_c ≈ r = 0.5 (spatial fluctuations shift
     the real lam_c somewhat higher — we sweep around it empirically).
   H1: C peaks near the active/absorbing transition.
   H0: C does not track the transition.

OUTPUT
------
  lattice_results.csv   — all raw C values and sub-metrics per condition/seed
  lattice_results.png   — three-panel line plot  (C ± 1 SD vs parameter)
"""

import os, sys, csv as _csv, importlib.util, time, warnings
import numpy as np
from scipy import stats as scipy_stats
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
GRID     = 64        # lattice side length
N_FRAMES = 100       # frames captured per run (each = one complexity timestep)
WARMUP   = 400       # steps to discard (let system reach steady state)
N_SEEDS  = 6         # independent realisations per condition

# ---------------------------------------------------------------------------
# Shared helper: 2-D binary frame → volume list for compute_full_C
# ---------------------------------------------------------------------------
def frames_to_volumes(frames):
    """List[np.array (N,N)] → list of (1,1,N²) volumes."""
    return [f.astype(np.float32).reshape(1, 1, -1) for f in frames]

# ---------------------------------------------------------------------------
# EXPERIMENT 1 — Noisy Voter Model
# ---------------------------------------------------------------------------
VOTER_EPS = [0.0, 0.03, 0.07, 0.15, 0.25, 0.40]

def voter_step(state, eps):
    N  = state.shape[0]
    # Stack all four neighbours, pick one randomly per cell
    neighbours = np.stack([
        np.roll(state, 1, 0),   # up
        np.roll(state, -1, 0),  # down
        np.roll(state, 1, 1),   # left
        np.roll(state, -1, 1),  # right
    ], axis=0)                  # (4, N, N)
    dirs    = np.random.randint(0, 4, (N, N))
    chosen  = neighbours[dirs, np.arange(N)[:, None], np.arange(N)[None, :]]
    if eps > 0:
        noise_mask = np.random.random((N, N)) < eps
        rand_vals  = np.random.randint(0, 2, (N, N), dtype=state.dtype)
        return np.where(noise_mask, rand_vals, chosen)
    return chosen

def run_voter(eps, seed):
    rng   = np.random.RandomState(seed)
    state = rng.randint(0, 2, (GRID, GRID), dtype=np.int8)
    for _ in range(WARMUP):
        state = voter_step(state, eps)
    frames = []
    for _ in range(N_FRAMES):
        state = voter_step(state, eps)
        frames.append(state.astype(np.float32))
    return frames

# ---------------------------------------------------------------------------
# EXPERIMENT 2 — Potts Model  (q = 4, Metropolis-Hastings, J = 1)
# ---------------------------------------------------------------------------
Q_POTTS  = 4
TC_POTTS = 1.0 / np.log(1.0 + np.sqrt(float(Q_POTTS)))  # ≈ 0.910
# Sweep temperatures bracketing T_c
POTTS_T  = [0.40, 0.65, TC_POTTS, 1.15, 1.70, 2.50]

def potts_step(state, T):
    """One checkerboard sweep (two half-steps) for the Potts model."""
    N = state.shape[0]
    rows, cols = np.mgrid[0:N, 0:N]
    for parity in [0, 1]:
        mask = (rows + cols) % 2 == parity
        proposed = np.random.randint(0, Q_POTTS, (N, N))

        # Energy of current and proposed spin at each site
        nbrs = (np.roll(state, 1, 0) + np.roll(state, -1, 0) +
                np.roll(state, 1, 1) + np.roll(state, -1, 1))
        # delta_E negative = proposed matches more neighbours (favourable)
        # We count matches: neighbour_match = (nbr == spin) → 1 each
        def count_matches(spin_map):
            return ((np.roll(state, 1, 0) == spin_map).astype(int) +
                    (np.roll(state, -1, 0) == spin_map).astype(int) +
                    (np.roll(state, 1, 1) == spin_map).astype(int) +
                    (np.roll(state, -1, 1) == spin_map).astype(int))

        m_curr = count_matches(state)
        m_prop = count_matches(proposed)
        delta_E = -(m_prop - m_curr).astype(float)   # dE = -J*(m_prop-m_curr), J=1

        accept_prob = np.where(delta_E <= 0, 1.0,
                               np.exp(-delta_E / T))
        accept = mask & (np.random.random((N, N)) < accept_prob)
        state  = np.where(accept, proposed, state)
    return state

def run_potts(T, seed):
    rng   = np.random.RandomState(seed)
    state = rng.randint(0, Q_POTTS, (GRID, GRID))
    for _ in range(WARMUP):
        state = potts_step(state, T)
    frames = []
    for _ in range(N_FRAMES):
        state = potts_step(state, T)
        # Binarise: spin == 0  (density ~1/q disordered, ~1 if spin-0 ordered)
        frames.append((state == 0).astype(np.float32))
    return frames

# ---------------------------------------------------------------------------
# EXPERIMENT 3 — Contact Process  (synchronous, recovery = 0.5)
# ---------------------------------------------------------------------------
CP_RECOVERY = 0.5
# Sweep infection rates around the mean-field critical point (lam ~ recovery)
CP_LAMBDA = [0.25, 0.40, 0.55, 0.75, 1.10, 1.60]

def cp_step(state, lam, r=CP_RECOVERY):
    N = state.shape[0]
    occ_nbrs = (np.roll(state, 1, 0) + np.roll(state, -1, 0) +
                np.roll(state, 1, 1) + np.roll(state, -1, 1)).astype(float)
    rand = np.random.random((N, N))
    # Occupied → empty (recover)
    recover = (state == 1) & (rand < r)
    # Empty → occupied (infect); each occupied neighbour contributes lam/4
    p_infect = 1.0 - (1.0 - lam / 4.0) ** occ_nbrs
    infect   = (state == 0) & (rand < p_infect)
    new_state = state.copy()
    new_state[recover] = 0
    new_state[infect]  = 1
    return new_state

def run_cp(lam, seed):
    rng   = np.random.RandomState(seed)
    # Start with ~50 % occupied so subcritical cases have chance to die out
    state = (rng.random((GRID, GRID)) < 0.50).astype(np.int8)
    for _ in range(WARMUP):
        state = cp_step(state, lam)
    frames = []
    for _ in range(N_FRAMES):
        state = cp_step(state, lam)
        frames.append(state.astype(np.float32))
    return frames

# ---------------------------------------------------------------------------
# Run one condition and return the C dict
# ---------------------------------------------------------------------------
def measure(frames):
    vols = frames_to_volumes(frames)
    return compute_full_C(vols)

# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def cohens_d(a, b):
    a, b  = np.asarray(a), np.asarray(b)
    pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2.0)
    return float((a.mean() - b.mean()) / max(pooled_std, 1e-9))

def pvalue(a, b):
    _, p = scipy_stats.ttest_ind(a, b, equal_var=False)
    return float(p)

# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------
def run_experiment(name, param_label, params, run_fn):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    results = {}   # param → list of C dicts
    for p in params:
        c_vals = []
        t0 = time.time()
        for seed in range(N_SEEDS):
            frames = run_fn(p, seed)
            res    = measure(frames)
            c_vals.append(res)
        mean_density = np.mean([f.mean() for f in frames_to_volumes(frames)])
        print(f"  {param_label}={p:.4f}  "
              f"C={np.mean([r['C_a'] for r in c_vals]):.4f} "
              f"(+/-{np.std([r['C_a'] for r in c_vals]):.4f})  "
              f"density={mean_density:.3f}  "
              f"{time.time()-t0:.1f}s")
        results[p] = c_vals
    return results

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
def save_csv(all_results, path):
    fields = ["experiment", "param", "seed",
              "C_a", "mean_H", "std_H", "op_up", "op_down",
              "mi1", "decay", "tc_mean", "gzip_ratio"]
    rows = []
    for exp_name, (param_label, results) in all_results.items():
        for p, c_list in results.items():
            for seed, r in enumerate(c_list):
                row = {"experiment": exp_name, "param": round(p, 5), "seed": seed}
                for f in fields[3:]:
                    row[f] = round(float(r.get(f, 0.0)), 6)
                row["C_a"] = round(float(r["C_a"]), 6)
                rows.append(row)
    with open(path, "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV -> {path}")

# ---------------------------------------------------------------------------
# Statistics report
# ---------------------------------------------------------------------------
def report_stats(all_results):
    print(f"\n{'='*60}")
    print("  STATISTICAL SUMMARY")
    print(f"{'='*60}")
    for exp_name, (param_label, results) in all_results.items():
        params = sorted(results.keys())
        C_means = [np.mean([r["C_a"] for r in results[p]]) for p in params]
        peak_p  = params[int(np.argmax(C_means))]
        other_p = params[0] if params[0] != peak_p else params[-1]

        peak_C  = [r["C_a"] for r in results[peak_p]]
        other_C = [r["C_a"] for r in results[other_p]]
        d = cohens_d(peak_C, other_C)
        p = pvalue(peak_C, other_C)

        outcome = "H1 SUPPORTED" if (d > 0.8 and p < 0.05) else "inconclusive"
        print(f"\n  {exp_name}")
        print(f"    Peak C at {param_label}={peak_p:.4f}  "
              f"C={np.mean(peak_C):.4f} +/- {np.std(peak_C):.4f}")
        print(f"    vs {param_label}={other_p:.4f}  "
              f"C={np.mean(other_C):.4f} +/- {np.std(other_C):.4f}")
        print(f"    Cohen d={d:.2f}  p={p:.4f}  -> {outcome}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_results(all_results, output_path):
    fig = plt.figure(figsize=(21, 6))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.30)

    colours = {"Voter Model": "#2196F3",
               "Potts Model (q=4)": "#E91E63",
               "Contact Process": "#4CAF50"}

    panel_info = [
        ("Voter Model",         "Noise rate  eps"),
        ("Potts Model (q=4)",   f"Temperature  T  (T_c = {TC_POTTS:.3f})"),
        ("Contact Process",     f"Infection rate  lambda  (recovery={CP_RECOVERY})"),
    ]

    for idx, (exp_name, xlabel) in enumerate(panel_info):
        ax    = fig.add_subplot(gs[0, idx])
        _, results = all_results[exp_name]
        params     = sorted(results.keys())
        means      = [np.mean([r["C_a"] for r in results[p]]) for p in params]
        stds       = [np.std ([r["C_a"] for r in results[p]]) for p in params]

        colour = colours[exp_name]
        ax.errorbar(params, means, yerr=stds,
                    fmt="o-", color=colour, capsize=5, linewidth=2,
                    markersize=8, elinewidth=1.5, label="C (mean +/- 1 SD)")

        peak_idx = int(np.argmax(means))
        ax.axvline(params[peak_idx], color=colour, linestyle="--",
                   alpha=0.5, linewidth=1.2, label=f"peak at {params[peak_idx]:.3f}")

        # Mark the known critical point if applicable
        if "Potts" in exp_name:
            ax.axvline(TC_POTTS, color="black", linestyle=":", linewidth=1.5,
                       alpha=0.7, label=f"T_c = {TC_POTTS:.3f}")
        elif "Contact" in exp_name:
            ax.axvline(CP_RECOVERY, color="black", linestyle=":", linewidth=1.5,
                       alpha=0.7, label=f"MF lam_c ~ {CP_RECOVERY}")

        # Hypothesis text
        hyp = ("H1: C peaks at intermediate eps\nH0: C is flat/monotone"
               if "Voter" in exp_name else
               "H1: C peaks near T_c\nH0: C does not discriminate T"
               if "Potts" in exp_name else
               "H1: C peaks near absorbing transition\nH0: C is flat/monotone")
        ax.text(0.03, 0.97, hyp, transform=ax.transAxes,
                fontsize=8, va="top",
                bbox=dict(facecolor="lightyellow", edgecolor="goldenrod",
                          alpha=0.9, boxstyle="round,pad=0.4"))

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("C  (complexity score)", fontsize=10)
        ax.set_title(exp_name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Lattice Model Experiments  —  Complexity C vs Control Parameter\n"
        f"Grid {GRID}x{GRID}, {N_FRAMES} frames, {N_SEEDS} seeds per condition",
        fontsize=12, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot -> {output_path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t_start = time.time()

    voter_res  = run_experiment("Voter Model",       "eps", VOTER_EPS,
                                 lambda p, s: run_voter(p, s))
    potts_res  = run_experiment("Potts Model (q=4)", "T",   POTTS_T,
                                 lambda p, s: run_potts(p, s))
    cp_res     = run_experiment("Contact Process",   "lam", CP_LAMBDA,
                                 lambda p, s: run_cp(p, s))

    all_results = {
        "Voter Model":       ("eps", voter_res),
        "Potts Model (q=4)": ("T",   potts_res),
        "Contact Process":   ("lam", cp_res),
    }

    csv_path = os.path.join(_HERE, "lattice_results.csv")
    png_path = os.path.join(_HERE, "lattice_results.png")

    save_csv(all_results, csv_path)
    report_stats(all_results)
    plot_results(all_results, png_path)

    print(f"\nTotal elapsed: {time.time()-t_start:.0f}s")
    print("Done.")
