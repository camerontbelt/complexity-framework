import gzip
import zlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy import stats

# =========================
# Complexity weights — mirroring complexity_framework_v8.py exactly
# =========================
K_ENTROPY  = 50
MU_SIGMA   = 0.012
SIGMA_SIGMA = 0.008
TCOMP_PEAKS = [(0.58, 0.08), (0.73, 0.08), (0.90, 0.05)]
GZIP_MU, GZIP_SIGMA       = 0.10, 0.05
OP_UP_MU,   OP_UP_SIG     = 0.14, 0.10   # weight_opacity_spatial peaks
OP_DOWN_MU, OP_DOWN_SIG   = 0.97, 0.05
OP_TEMP_K                 = 10            # tanh gate sharpness

# w_H  ────────────────────────────────────────────────────────────────
def wh_weight(mean_H, std_H):
    gate1    = np.tanh(K_ENTROPY * mean_H)
    gate2    = np.tanh(K_ENTROPY * (1.0 - mean_H))
    gaussian = 1.0 + np.exp(-((std_H - MU_SIGMA)**2) / (2*SIGMA_SIGMA**2))
    return float(gate1 * gate2 * gaussian)

# w_OP_s  ─────────────────────────────────────────────────────────────
def wops_weight(op_up, op_down):
    """P1 spatial opacity: G(op_up, 0.14, 0.10) × G(op_down, 0.97, 0.05)"""
    w_up   = np.exp(-((op_up   - OP_UP_MU)  **2) / (2*OP_UP_SIG  **2))
    w_down = np.exp(-((op_down - OP_DOWN_MU)**2) / (2*OP_DOWN_SIG**2))
    return float(w_up * w_down)

# w_OP_t  ─────────────────────────────────────────────────────────────
def wopt_weight(mi1, decay):
    """P1 temporal opacity: tanh(k·MI₁)·tanh(k·(1−MI₁))·tanh(k·decay)"""
    g1 = np.tanh(OP_TEMP_K * float(mi1))
    g2 = np.tanh(OP_TEMP_K * (1.0 - float(mi1)))
    g3 = np.tanh(OP_TEMP_K * max(float(decay), 0.0))
    return float(g1 * g2 * g3)

# w_T  ────────────────────────────────────────────────────────────────
def wt_weight(tc_mean):
    w = 0.0
    for mu, sigma in TCOMP_PEAKS:
        w = max(w, np.exp(-((tc_mean - mu)**2) / (2*sigma**2)))
    return float(w)

# w_G  ────────────────────────────────────────────────────────────────
def wg_weight(gzip_ratio):
    return float(np.exp(-((gzip_ratio - GZIP_MU)**2) / (2*GZIP_SIGMA**2)))


# =========================
# Domain-agnostic weights
# =========================
# Replace CA-calibrated Gaussian peaks with tanh boundary gates that encode
# only logical necessary conditions — no substrate-specific attractor location.
#
# Every gate is zero at the information-theoretic trivial extremes and peaks
# wherever complex systems naturally sit, without specifying that location.
#
# Single sharpness constant K_GATE = 10 (same as OP_TEMP_K) throughout.
# The std_H bonus uses a softer K_STD = 3 so it discriminates within the
# observed CNN range (0.13 – 0.40) rather than saturating immediately.
# =========================
K_GATE = 10
K_STD  = 3    # softer gate for the std_H specialisation bonus

def wh_agnostic(mean_H, std_H):
    """Not dead (H->0), not pure noise (H->1); one-sided bonus for channel
    specialisation (higher std_H = more diverse feature detectors)."""
    gate_H  = np.tanh(K_GATE * float(mean_H)) * np.tanh(K_GATE * (1.0 - float(mean_H)))
    # std_H in [0, ~0.5] for H values in [0,1]; normalise to [0,1] then gate
    std_n   = float(np.clip(2.0 * float(std_H), 0.0, 1.0))
    bonus   = np.tanh(K_STD * std_n)   # monotone: more specialisation = more credit
    return float(gate_H * (1.0 + bonus))

def wops_agnostic(op_up, op_down):
    """Spatial opacity must be non-trivial in at least one direction.
    Additive (OR) rather than multiplicative (AND) so op_up=0 does not
    collapse the term when op_down carries real signal."""
    g_up   = np.tanh(K_GATE * float(op_up))   * np.tanh(K_GATE * (1.0 - float(op_up)))
    g_down = np.tanh(K_GATE * float(op_down))  * np.tanh(K_GATE * (1.0 - float(op_down)))
    return float(g_up + g_down)   # range [0, 2]

def wopt_agnostic(mi1, decay):
    """Unchanged — temporal opacity is already fully parameter-free."""
    g1 = np.tanh(K_GATE * float(mi1))
    g2 = np.tanh(K_GATE * (1.0 - float(mi1)))
    g3 = np.tanh(K_GATE * max(float(decay), 0.0))
    return float(g1 * g2 * g3)

def wt_agnostic(tc_mean):
    """Not frozen (tc->1), not chaotic (tc->0); peaks at intermediate tc."""
    return float(np.tanh(K_GATE * float(tc_mean)) * np.tanh(K_GATE * (1.0 - float(tc_mean))))

def wg_agnostic(gzip_ratio):
    """Not trivially compressible (gz->0), not incompressible random (gz->1);
    peaks at intermediate compressibility — no specific peak value assumed."""
    return float(np.tanh(K_GATE * float(gzip_ratio)) * np.tanh(K_GATE * (1.0 - float(gzip_ratio))))

def wg_agnostic_bp(gzip_ratio_bp):
    """Bit-packed gzip gate: tanh gate applied to bit-packed compression ratio,
    clamped to [0, 1]. Bit-packing removes the byte-encoding artifact (gzip ≈ 1/8)
    by storing 1 bit per binary cell instead of 8 bits. The correction factor
    (×8 for binary, ×8/log2(q) for q-state) is derivable from first principles.

    Clamped to non-negative because gzip ratios > 1.0 (incompressible + header
    overhead) make tanh(K*(1-x)) negative — the physical meaning is simply
    "no complexity signal", not "anti-complexity".

    After bit-packing:
      C4 rules: ~0.24-0.83 (genuine pattern structure → gate ≈ 0.93-1.0)
      C3 rules: ~1.0 (incompressible random → gate = 0.0)
      C1 rules: ~0.01 (trivially ordered → gate ≈ 0.10)
    """
    raw = np.tanh(K_GATE * float(gzip_ratio_bp)) * np.tanh(K_GATE * (1.0 - float(gzip_ratio_bp)))
    return float(max(0.0, raw))

# =========================
# Opacity metric helpers
# =========================

def _opacity_spatial(grid, n_bins=8):
    """
    Vectorised H(global | local) and H(local | global) on a (T, W) binary grid.
    Direct port of _opacity_both() from complexity_framework_v8.py.
    """
    T, W  = grid.shape
    dens  = grid.mean(axis=1)                                              # (T,)
    gbins = np.clip((dens * n_bins).astype(int), 0, n_bins - 1)           # (T,)
    left      = np.roll(grid, 1,  axis=1)
    right     = np.roll(grid, -1, axis=1)
    patch_int = (left * 4 + grid * 2 + right).astype(np.int16)            # (T, W)
    joint = np.zeros((8, n_bins), dtype=np.int64)
    np.add.at(joint, (patch_int.ravel(), np.repeat(gbins, W)), 1)
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


def _opacity_temporal(grid, max_lag=10, stride=100):
    """
    Vectorised I(X_t ; X_{t+lag}) / H(X_t) with decay, on a (T, W) binary grid.
    Matches _opacity_temporal() from complexity_framework_v8.py; uses numpy
    bincount instead of Python dicts for speed, and a spatial stride to keep
    runtime reasonable for large W.
    """
    sampled = grid[:, ::stride].astype(np.int8)  # (T, W//stride)

    def _mi_at_lag(lag):
        a = sampled[:-lag].ravel().astype(np.int64)
        b = sampled[lag: ].ravel().astype(np.int64)
        joint_flat = np.bincount(a * 2 + b, minlength=4).reshape(2, 2)
        total = float(joint_flat.sum())
        if total == 0:
            return 0.0
        def _H(c):
            p = c[c > 0].astype(float) / total
            return float(-np.sum(p * np.log2(p)))
        MI = _H(joint_flat.sum(axis=1)) + _H(joint_flat.sum(axis=0)) - _H(joint_flat.ravel())
        Ht = _H(joint_flat.sum(axis=1))
        return float(np.clip(MI / max(Ht, 1e-9), 0.0, 1.0))

    mi1   = _mi_at_lag(1)
    mi_k  = _mi_at_lag(max_lag)
    decay = float(np.clip(mi1 - mi_k, 0.0, 1.0))
    return mi1, decay

def compute_full_C(volumes):
    """
    Full composite C matching complexity_framework_v8.py:

        C = w_H × (w_OP_s + w_OP_t) × w_T × w_G

    volumes : list of np.array (C, H, W), continuous ReLU activations.

    Pipeline
    --------
    1. Binarise: (activation > 0) → binary 0/1 grid.
    2. Stack into (T, C*H*W) binary grid — test samples as time axis.
    3. Compute all six metrics (mean_H, std_H, op_up, op_down, mi1, decay,
       tc_mean, gzip_ratio) on that grid.
    4. Apply the five v8 weight functions and combine multiplicatively.
    """
    T = len(volumes)

    # 1. Binarise: neuron fired (>0) = 1, silent = 0
    bin_vols   = np.stack([(v > 0).astype(np.uint8) for v in volumes])  # (T, C, H, W)
    n_channels = bin_vols.shape[1]

    # 2. Flat (T, W) grid for opacity / tc metrics
    grid = bin_vols.reshape(T, -1)   # (T, C*H*W)

    # --- w_H : spatial entropy ---
    flat_s  = bin_vols.reshape(T, n_channels, -1).astype(np.float64)
    density = np.clip(flat_s.mean(axis=2), 1e-12, 1 - 1e-12)
    H_vals  = -(density * np.log2(density) + (1 - density) * np.log2(1 - density))
    mean_H  = float(H_vals.mean())
    std_H   = float(H_vals.std())
    wH      = wh_weight(mean_H, std_H)

    # --- w_OP_s : spatial opacity ---
    op_up, op_down = _opacity_spatial(grid)
    wOPs = wops_weight(op_up, op_down)

    # --- w_OP_t : temporal opacity ---
    mi1, decay = _opacity_temporal(grid)
    wOPt = wopt_weight(mi1, decay)

    # --- w_T : temporal compression ---
    flips   = np.sum(np.diff(grid.astype(np.int8), axis=0) != 0, axis=0)
    tc      = np.clip(1.0 - (1.0 + flips) / T, 0.0, 1.0)
    tc_mean = float(tc.mean())
    wT      = wt_weight(tc_mean)

    # --- w_G : gzip (per-sample binary tensor, then averaged) ---
    # Clip to [0,1]: small tensors (< ~80 bytes) can compress to larger than the
    # input due to gzip header overhead, which would make wG_a negative.
    gzip_ratios = [min(len(gzip.compress(bin_vols[t].tobytes())) /
                   max(len(bin_vols[t].tobytes()), 1), 1.0) for t in range(T)]
    gzip_ratio  = float(np.mean(gzip_ratios))
    wG = wg_weight(gzip_ratio)

    # --- w_G (bit-packed): pack binary data to 1 bit/cell before compressing ---
    # This removes the byte-encoding artifact (gzip ≈ 1/8) and recovers the true
    # compression ratio. The correction is derivable: for binary data, np.packbits
    # stores 8 cells per byte instead of 1. For q-state: ceil(log2(q)) bits/cell.
    packed_grid    = np.packbits(grid.ravel())
    packed_raw     = packed_grid.tobytes()
    packed_comp    = gzip.compress(packed_raw)
    gzip_ratio_bp  = min(len(packed_comp) / max(len(packed_raw), 1), 1.5)

    # Full composite — additive opacity channels, multiplicative otherwise
    C = wH * (wOPs + wOPt) * wT * wG

    # --- Agnostic composite (same raw metrics, parameter-free weights) ---
    wH_a    = wh_agnostic(mean_H, std_H)
    wOPs_a  = wops_agnostic(op_up, op_down)
    wOPt_a  = wopt_agnostic(mi1, decay)
    wT_a    = wt_agnostic(tc_mean)
    wG_a    = wg_agnostic(gzip_ratio)
    C_a     = wH_a * (wOPs_a + wOPt_a) * wT_a * wG_a

    # --- Agnostic composite with bit-packed gzip (fully parameter-free) ---
    wG_a_bp = wg_agnostic_bp(gzip_ratio_bp)
    C_a_bp  = wH_a * (wOPs_a + wOPt_a) * wT_a * wG_a_bp

    return {
        # v8 composite and weights
        "C": C,
        "wH": wH, "wOPs": wOPs, "wOPt": wOPt, "wT": wT, "wG": wG,
        # agnostic composite and weights
        "C_a": C_a,
        "wH_a": wH_a, "wOPs_a": wOPs_a, "wOPt_a": wOPt_a, "wT_a": wT_a, "wG_a": wG_a,
        # agnostic with bit-packed gzip
        "C_a_bp": C_a_bp,
        "wG_a_bp": wG_a_bp,
        # raw sub-component values
        "mean_H": mean_H, "std_H": std_H,
        "op_up": op_up, "op_down": op_down,
        "mi1": mi1, "decay": decay,
        "tc_mean": tc_mean,
        "gzip_ratio": gzip_ratio,
        "gzip_ratio_bp": gzip_ratio_bp,
    }

# =========================
# CNN Model
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.pool  = nn.MaxPool2d(2,2)
        
        # dummy forward to compute flattened size
        with torch.no_grad():
            dummy = torch.zeros(1,1,28,28)
            x = self.pool(torch.relu(self.conv2(torch.relu(self.conv1(dummy)))))
            self.flattened_size = x.numel()  # 16*14*14 = 3136

        self.fc1   = nn.Linear(self.flattened_size, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x  = torch.relu(self.conv1(x))
        # Keep pre-pool conv2 features for complexity measurement.
        # Post-pool (hidden) would inflate density to ~94% (MaxPool takes the max
        # of a 2×2 window, so even ~50% pre-pool activation becomes ~94% post-pool).
        # That locks tc ≈ 0.88 for ALL random seeds, collapsing discrimination.
        pre_pool = torch.relu(self.conv2(x))   # (batch, 16, 28, 28)
        x   = self.pool(pre_pool)
        hidden = x.view(x.size(0), -1)
        out = self.fc2(torch.relu(self.fc1(hidden)))
        return out, pre_pool
# =========================
# Data
# =========================
def get_mnist_subset(train_size=2000, test_size=200, batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_subset = Subset(train_dataset, range(train_size))
    test_subset  = Subset(test_dataset, range(test_size))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_subset, batch_size=test_size, shuffle=False)

    return train_loader, test_loader

# =========================
# Training
# =========================
def train_model(model, train_loader, device, epochs=5, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/total:.4f} | Acc: {correct/total:.3f}")

# =========================
# Extract 3D activation grids
# =========================
def extract_3d_activation_grids(model, test_loader, device):
    """Return pre-pool conv2 activations: shape (16, 28, 28) per sample."""
    model.eval()
    volumes = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            _, pre_pool = model(images)       # (batch, 16, 28, 28)
            for i in range(pre_pool.shape[0]):
                volumes.append(pre_pool[i].cpu().numpy())   # already (C, H, W)
    return volumes

# =========================
# Main experiment
# =========================
def run_experiment(n_random=300, csv_out="mnist_results.csv"):
    import csv as _csv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, test_loader = get_mnist_subset()

    # Train CNN
    print("\nTraining CNN...")
    torch.manual_seed(42)
    cnn = SimpleCNN().to(device)
    train_model(cnn, train_loader, device, epochs=5)

    # Extract activations
    print("\nExtracting 3D activation grids...")
    trained_volumes = extract_3d_activation_grids(cnn, test_loader, device)

    # Compute full C
    print("\nComputing full C for trained model...")
    trained_results = compute_full_C(trained_volumes)

    # Random controls — collect full component breakdown for every seed
    print(f"\nEvaluating {n_random} random controls...")
    all_rows   = []   # list of dicts, one per model
    random_Cs  = []
    random_C_as = []

    trained_row = {"model": "trained", "seed": 42, **trained_results}
    all_rows.append(trained_row)

    for seed in range(n_random):
        torch.manual_seed(seed)
        rand_model   = SimpleCNN().to(device)
        rand_volumes = extract_3d_activation_grids(rand_model, test_loader, device)
        res          = compute_full_C(rand_volumes)
        random_Cs.append(res["C"])
        random_C_as.append(res["C_a"])
        all_rows.append({"model": "random", "seed": seed, **res})
        if seed % 50 == 0:
            print(f"  {seed}/{n_random}  C={res['C']:.4f}  C_a={res['C_a']:.4f}  "
                  f"op_up={res['op_up']:.4f}  op_dn={res['op_down']:.4f}  "
                  f"mi1={res['mi1']:.4f}")

    # Write CSV
    if csv_out:
        fieldnames = ["model", "seed",
                      "C",   "wH",   "wOPs",   "wOPt",   "wT",   "wG",
                      "C_a", "wH_a", "wOPs_a", "wOPt_a", "wT_a", "wG_a",
                      "mean_H", "std_H",
                      "op_up", "op_down",
                      "mi1", "decay",
                      "tc_mean", "gzip_ratio"]
        with open(csv_out, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in all_rows:
                w.writerow({k: round(v, 6) if isinstance(v, float) else v
                            for k, v in row.items()})
        print(f"\nDiagnostics saved -> {csv_out}")

    random_mean   = np.mean(random_Cs)
    random_std    = np.std(random_Cs)
    cohen_d       = (trained_results["C"] - random_mean) / max(random_std, 1e-8)
    t_stat, p_val = stats.ttest_1samp(random_Cs, trained_results["C"])

    random_mean_a  = np.mean(random_C_as)
    random_std_a   = np.std(random_C_as)
    cohen_d_a      = (trained_results["C_a"] - random_mean_a) / max(random_std_a, 1e-8)
    _, p_val_a     = stats.ttest_1samp(random_C_as, trained_results["C_a"])

    # Quick summary of raw values for trained vs random mean
    all_keys = ["mean_H", "std_H", "op_up", "op_down", "mi1", "decay",
                "tc_mean", "gzip_ratio",
                "wH", "wOPs", "wOPt", "wT", "wG", "C",
                "wH_a", "wOPs_a", "wOPt_a", "wT_a", "wG_a", "C_a"]
    rand_mean_row = {k: np.mean([r[k] for r in all_rows if r["model"] == "random"])
                     for k in all_keys}
    print("\n-- Component summary (v8 weights) ------------------------------")
    fmt = "{:>12s}  trained={:>8.4f}  random_mean={:>8.4f}"
    for k in ["mean_H", "std_H", "op_up", "op_down", "mi1", "decay",
              "tc_mean", "gzip_ratio", "wH", "wOPs", "wOPt", "wT", "wG", "C"]:
        print(fmt.format(k, trained_results[k], rand_mean_row[k]))
    print("-- Component summary (agnostic weights) ------------------------")
    for k in ["wH_a", "wOPs_a", "wOPt_a", "wT_a", "wG_a", "C_a"]:
        print(fmt.format(k, trained_results[k], rand_mean_row[k]))
    print("---------------------------------------------------------------")

    random_rows = [r for r in all_rows if r["model"] == "random"]

    # =========================
    # Plot 1 — both composites side-by-side
    # =========================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CNN MNIST — Composite C comparison", fontsize=13, fontweight='bold')

    r = trained_results

    # --- 1a: v8 (CA-calibrated) ---
    ax1.hist(random_Cs, bins=30, alpha=0.7, label="Random CNNs")
    ax1.axvline(r["C"], color='r', linestyle='--', linewidth=2, label="Trained CNN")
    ax1.set_title("v8 (CA-calibrated) weights")
    ax1.set_xlabel("C"); ax1.set_ylabel("Frequency"); ax1.legend()
    ax1.text(0.62, 0.95,
             f"Trained  = {r['C']:.4f}\n"
             f"Rand mu  = {random_mean:.4f}\n"
             f"Rand std = {random_std:.4f}\n"
             f"Cohen d  = {cohen_d:.2f}\n"
             f"p-value  = {p_val:.4f}\n"
             f"-----------------\n"
             f"wH   = {r['wH']:.4f}\n"
             f"wOPs = {r['wOPs']:.4f}\n"
             f"wOPt = {r['wOPt']:.4f}\n"
             f"wT   = {r['wT']:.4f}\n"
             f"wG   = {r['wG']:.4f}",
             transform=ax1.transAxes, verticalalignment='top', fontsize=8,
             bbox=dict(facecolor='white', alpha=0.8))

    # --- 1b: agnostic weights ---
    ax2.hist(random_C_as, bins=30, alpha=0.7, color='mediumseagreen', label="Random CNNs")
    ax2.axvline(r["C_a"], color='r', linestyle='--', linewidth=2, label="Trained CNN")
    ax2.set_title("Agnostic (tanh boundary) weights")
    ax2.set_xlabel("C_a"); ax2.set_ylabel("Frequency"); ax2.legend()
    ax2.text(0.62, 0.95,
             f"Trained  = {r['C_a']:.4f}\n"
             f"Rand mu  = {random_mean_a:.4f}\n"
             f"Rand std = {random_std_a:.4f}\n"
             f"Cohen d  = {cohen_d_a:.2f}\n"
             f"p-value  = {p_val_a:.4f}\n"
             f"-----------------\n"
             f"wH_a   = {r['wH_a']:.4f}\n"
             f"wOPs_a = {r['wOPs_a']:.4f}\n"
             f"wOPt_a = {r['wOPt_a']:.4f}\n"
             f"wT_a   = {r['wT_a']:.4f}\n"
             f"wG_a   = {r['wG_a']:.4f}",
             transform=ax2.transAxes, verticalalignment='top', fontsize=8,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()

    # =========================
    # Plot 2 — per-metric attractor grid
    # Rows: raw metrics | weights
    # Trained value shown as red dashed line on every subplot.
    # =========================
    raw_metrics = [
        ("mean_H",     "Mean spatial entropy  H",        None),
        ("std_H",      "Std of H across channels",       None),
        ("op_up",      "Spatial opacity ↑  (H(glob|loc))", 0.14),
        ("op_down",    "Spatial opacity ↓  (H(loc|glob))", 0.97),
        ("mi1",        "Temporal MI at lag 1",           None),
        ("decay",      "Temporal MI decay  (MI₁ − MI₁₀)", None),
        ("tc_mean",    "Temporal compression  tc",       None),
        ("gzip_ratio", "Gzip ratio",                     0.10),
    ]
    weight_metrics_v8 = [
        ("wH",   "w_H  (v8)"),
        ("wOPs", "w_OP_s  (v8)"),
        ("wOPt", "w_OP_t  (v8)"),
        ("wT",   "w_T  (v8)"),
        ("wG",   "w_G  (v8)"),
    ]
    weight_metrics_a = [
        ("wH_a",   "w_H  (agnostic)"),
        ("wOPs_a", "w_OP_s  (agnostic)"),
        ("wOPt_a", "w_OP_t  (agnostic)"),
        ("wT_a",   "w_T  (agnostic)"),
        ("wG_a",   "w_G  (agnostic)"),
    ]
    weight_metrics = weight_metrics_v8 + weight_metrics_a

    n_raw = len(raw_metrics)
    n_wt  = len(weight_metrics)
    ncols = 5
    nrows = int(np.ceil((n_raw + n_wt) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.8, nrows * 2.8))
    axes = axes.ravel()
    fig.suptitle("Per-metric distributions — Random CNNs vs Trained CNN",
                 fontsize=12, fontweight='bold')

    trained_val = trained_results

    for idx, (key, label, peak) in enumerate(raw_metrics):
        ax = axes[idx]
        vals = [row[key] for row in random_rows]
        ax.hist(vals, bins=25, alpha=0.6, color='steelblue', label="Random")
        ax.axvline(trained_val[key], color='red', linestyle='--',
                   linewidth=1.8, label=f"Trained\n{trained_val[key]:.4f}")
        if peak is not None:
            ax.axvline(peak, color='green', linestyle=':', linewidth=1.4,
                       label=f"v8 peak\n{peak}")
        # Cohen's d for this raw metric
        rv = np.array(vals)
        d  = (trained_val[key] - rv.mean()) / max(rv.std(), 1e-9)
        ax.set_title(f"{label}\nd={d:.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc='upper right')

    for jdx, (key, label) in enumerate(weight_metrics):
        ax = axes[n_raw + jdx]
        vals = [row[key] for row in random_rows]
        color = 'darkorange' if jdx < len(weight_metrics_v8) else 'mediumseagreen'
        ax.hist(vals, bins=25, alpha=0.6, color=color, label="Random")
        ax.axvline(trained_val[key], color='red', linestyle='--',
                   linewidth=1.8, label=f"Trained\n{trained_val[key]:.4f}")
        rv = np.array(vals)
        d  = (trained_val[key] - rv.mean()) / max(rv.std(), 1e-9)
        ax.set_title(f"{label}\nd={d:.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc='upper right')

    # hide any unused axes
    for ax in axes[n_raw + n_wt:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()