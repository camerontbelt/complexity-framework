# Frustrated Triangular Ising model: sweep J2 and temperature,
# compute full C, and generate a phase-style heatmap.
# ASCII console output only; saves CSV and PNG.

import numpy as np
import gzip
import csv
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================
L = 48
TEMPS = np.linspace(0.4, 4.0, 18)
J2_VALUES = np.linspace(0.0, 0.6, 13)
N_SWEEPS_BURN = 1500
N_SWEEPS_MEAS = 800
MEASURE_EVERY = 5
N_SEEDS = 6
J1 = -1.0
CSV_OUT = "triangular_j2_sweep.csv"
PNG_OUT = "triangular_j2_heatmap.png"

# =========================
# COMPLEXITY WEIGHTS
# =========================
K_ENTROPY = 50
MU_SIGMA = 0.012
SIGMA_SIGMA = 0.008
TCOMP_PEAKS = [(0.58, 0.08), (0.73, 0.08), (0.90, 0.05)]
GZIP_MU = 0.10
GZIP_SIGMA = 0.05
OP_UP_MU = 0.14
OP_UP_SIG = 0.10
OP_DOWN_MU = 0.97
OP_DOWN_SIG = 0.05
OP_TEMP_K = 10


def gauss(x, mu, sig):
    return np.exp(-((x - mu) ** 2) / (2 * sig ** 2))


def wh_weight(mean_H, std_H):
    gate1 = np.tanh(K_ENTROPY * mean_H)
    gate2 = np.tanh(K_ENTROPY * (1.0 - mean_H))
    gaussian = 1.0 + np.exp(-((std_H - MU_SIGMA) ** 2) / (2 * SIGMA_SIGMA ** 2))
    return float(gate1 * gate2 * gaussian)


def wops_weight(op_up, op_down):
    return float(gauss(op_up, OP_UP_MU, OP_UP_SIG) * gauss(op_down, OP_DOWN_MU, OP_DOWN_SIG))


def wopt_weight(mi1, decay):
    g1 = np.tanh(OP_TEMP_K * mi1)
    g2 = np.tanh(OP_TEMP_K * (1.0 - mi1))
    g3 = np.tanh(OP_TEMP_K * max(decay, 0.0))
    return float(g1 * g2 * g3)


def wt_weight(tc_mean):
    w = 0.0
    for mu, sig in TCOMP_PEAKS:
        w = max(w, gauss(tc_mean, mu, sig))
    return float(w)


def wg_weight(gzip_ratio):
    return float(gauss(gzip_ratio, GZIP_MU, GZIP_SIGMA))


# =========================
# MODEL
# =========================

def init_lattice(L):
    return np.random.choice([-1, 1], size=(L, L)).astype(np.int8)


def metropolis_sweep(spins, T, J2):
    L = spins.shape[0]
    for _ in range(L * L):
        i = np.random.randint(L)
        j = np.random.randint(L)
        s = spins[i, j]

        local_nn = (
            spins[(i + 1) % L, j] + spins[(i - 1) % L, j] +
            spins[i, (j + 1) % L] + spins[i, (j - 1) % L] +
            spins[(i + 1) % L, (j + 1) % L] + spins[(i - 1) % L, (j - 1) % L]
        )

        local_nnn = (
            spins[(i + 2) % L, j] + spins[(i - 2) % L, j] +
            spins[i, (j + 2) % L] + spins[i, (j - 2) % L] +
            spins[(i + 1) % L, (j - 1) % L] + spins[(i - 1) % L, (j + 1) % L]
        )

        dE = 2.0 * s * (J1 * local_nn + J2 * local_nnn)
        if dE <= 0 or np.random.rand() < np.exp(-dE / T):
            spins[i, j] *= -1


# =========================
# COMPLEXITY METRICS
# =========================

def entropy_binary(x):
    p = np.clip(np.mean(x), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def opacity_spatial(grid, n_bins=8):
    T, W = grid.shape
    dens = grid.mean(axis=1)
    gbins = np.clip((dens * n_bins).astype(int), 0, n_bins - 1)
    left = np.roll(grid, 1, axis=1)
    right = np.roll(grid, -1, axis=1)
    patch = (left * 4 + grid * 2 + right).astype(np.int16)
    joint = np.zeros((8, n_bins), dtype=np.int64)
    np.add.at(joint, (patch.ravel(), np.repeat(gbins, W)), 1)

    if joint.sum() == 0:
        return 0.0, 0.0

    def H(c):
        c = c[c > 0].astype(float)
        p = c / c.sum()
        return -np.sum(p * np.log2(p))

    Hj = H(joint.ravel())
    Hp = H(joint.sum(axis=1))
    Hg = H(joint.sum(axis=0))
    op_up = np.clip((Hj - Hp) / np.log2(n_bins), 0.0, 1.0)
    op_down = np.clip((Hj - Hg) / np.log2(8), 0.0, 1.0)
    return float(op_up), float(op_down)


def opacity_temporal(grid, max_lag=10):
    def mi_lag(lag):
        a = grid[:-lag].ravel()
        b = grid[lag:].ravel()
        joint = np.bincount(a * 2 + b, minlength=4).reshape(2, 2)
        total = joint.sum()
        if total == 0:
            return 0.0

        def H(c):
            p = c[c > 0] / total
            return -np.sum(p * np.log2(p))

        mi = H(joint.sum(axis=1)) + H(joint.sum(axis=0)) - H(joint.ravel())
        h = H(joint.sum(axis=1))
        return np.clip(mi / max(h, 1e-9), 0.0, 1.0)

    mi1 = mi_lag(1)
    mik = mi_lag(max_lag)
    return float(mi1), float(np.clip(mi1 - mik, 0.0, 1.0))


def compute_full_C(history):
    history = np.array(history, dtype=np.uint8)
    T = history.shape[0]
    grid = history.reshape(T, -1)

    H_rows = [entropy_binary(history[t]) for t in range(T)]
    mean_H = float(np.mean(H_rows))
    std_H = float(np.std(H_rows))

    op_up, op_down = opacity_spatial(grid)
    mi1, decay = opacity_temporal(grid)

    flips = np.sum(np.diff(grid, axis=0) != 0, axis=0)
    tc_mean = float(np.mean(np.clip(1.0 - (1.0 + flips) / T, 0.0, 1.0)))

    gzip_ratios = []
    for t in range(T):
        raw = history[t].tobytes()
        gzip_ratios.append(min(len(gzip.compress(raw)) / max(len(raw), 1), 1.0))
    gzip_ratio = float(np.mean(gzip_ratios))

    return float(
        wh_weight(mean_H, std_H) *
        (wops_weight(op_up, op_down) + wopt_weight(mi1, decay)) *
        wt_weight(tc_mean) *
        wg_weight(gzip_ratio)
    )


# =========================
# RUN
# =========================

def run_experiment():
    heatmap = np.zeros((len(J2_VALUES), len(TEMPS)))
    rows = []

    print("=" * 70)
    print("TRIANGULAR FRUSTRATED ISING: J2 x T SWEEP")
    print("=" * 70)

    for ji, J2 in enumerate(J2_VALUES):
        print(f"\nJ2 = {J2:.3f}")
        best_C = -1.0
        best_T = None

        for ti, T in enumerate(TEMPS):
            Cs = []
            print(f"  T = {T:.3f} ... ", end="", flush=True)

            for seed in range(N_SEEDS):
                np.random.seed(seed)
                spins = init_lattice(L)

                for _ in range(N_SWEEPS_BURN):
                    metropolis_sweep(spins, T, J2)

                history = []
                for sweep in range(N_SWEEPS_MEAS):
                    metropolis_sweep(spins, T, J2)
                    if sweep % MEASURE_EVERY == 0:
                        history.append((spins > 0).astype(np.uint8))

                C = compute_full_C(history)
                Cs.append(C)
                rows.append({"J2": J2, "T": T, "seed": seed, "C": C})

            mean_C = float(np.mean(Cs))
            std_C = float(np.std(Cs))
            heatmap[ji, ti] = mean_C

            if mean_C > best_C:
                best_C = mean_C
                best_T = T

            print(f"C = {mean_C:.4f} +/- {std_C:.4f}")

        print(f"  Peak for J2={J2:.3f}: T={best_T:.3f}, C={best_C:.4f}")

    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["J2", "T", "seed", "C"])
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        heatmap,
        aspect='auto',
        origin='lower',
        extent=[TEMPS[0], TEMPS[-1], J2_VALUES[0], J2_VALUES[-1]]
    )
    plt.colorbar(label='Mean Complexity C')
    plt.xlabel('Temperature T')
    plt.ylabel('J2 frustration')
    plt.title('Triangular frustrated Ising: complexity landscape')
    plt.tight_layout()
    plt.savefig(PNG_OUT, dpi=150)
    plt.show()

    print("\nSaved CSV:", CSV_OUT)
    print("Saved heatmap:", PNG_OUT)


if __name__ == "__main__":
    run_experiment()
