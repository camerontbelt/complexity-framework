"""
targeted-experiments.py
=======================
Four experiments with pre-registered hypotheses, full CSV output, and
explicit statistical conclusions.

Rationale (from preliminary experiments):
  - Temporal metrics on independent test images are invalid: permutation control
    (Exp 3) showed mi1 and decay change with image ordering, confirming they
    measure dataset structure, not model dynamics.
  - GRU hidden states are genuinely causal: each step t causally follows t-1.
  - Per-channel C_a distribution is bimodal for trained CNNs (Exp 4), showing
    channel specialisation — a clean spatial signal.
  - Per-image spatial complexity avoids ALL temporal assumptions.

Experiments
-----------
  A  GRU trained vs random (n=100)     — does the metric distinguish trained
                                          from untrained on a causal substrate?
  B  GRU training dynamics             — does C_a track learning epoch by epoch?
  C  Channel Specialisation Index      — is the spread of per-channel C_a wider
                                          for a trained CNN than random CNNs?
  D  Per-image spatial complexity      — does training produce higher within-image
                                          channel diversity, with no temporal axis?

Each experiment:
  1. Prints H1 and H0
  2. Runs the experiment and writes a CSV
  3. Computes statistics (t-test / Spearman rho / Cohen's d / percentile)
  4. Prints a binary verdict: H1 CONFIRMED or H0 NOT REJECTED

Run:  python targeted-experiments.py
"""

import os
import csv as _csv
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy import stats as sp

# =============================================================================
# Bootstrap — load shared functions from mnist-experiment.py
# =============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
            "mnist_exp", os.path.join(_HERE, "mnist-experiment.py"))
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_full_C = _mod.compute_full_C   # gzip bug already patched in source
SimpleCNN      = _mod.SimpleCNN
train_model    = _mod.train_model      # (model, loader, device, lr, epochs)

# =============================================================================
# Experiment toggle
# =============================================================================
RUN = {"A": True, "B": True, "C": True, "D": True}

# Statistical thresholds (pre-registered)
P_THRESH = 0.05
D_THRESH = 0.50   # medium effect size (Cohen, 1988)

# =============================================================================
# Shared infrastructure
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_GATE, K_STD = 10, 3


def get_data(train_size=2000, test_size=200):
    t  = transforms.Compose([transforms.ToTensor()])
    tr = datasets.MNIST("./data", train=True,  download=True, transform=t)
    te = datasets.MNIST("./data", train=False, download=True, transform=t)
    return (DataLoader(Subset(tr, range(train_size)), batch_size=64, shuffle=True),
            DataLoader(Subset(te, range(test_size)),  batch_size=64, shuffle=False))


def make_cnn(seed=42):
    torch.manual_seed(seed)
    return SimpleCNN().to(DEVICE)


def train_cnn(seed=42, epochs=5):
    tr, te = get_data()
    m = make_cnn(seed)
    train_model(m, tr, DEVICE, epochs=epochs)
    return m, te


def extract_pre_pool(model, loader):
    model.eval(); vols = []
    with torch.no_grad():
        for imgs, _ in loader:
            _, pp = model(imgs.to(DEVICE))
            for i in range(pp.shape[0]):
                vols.append(pp[i].cpu().numpy())
    return vols


def save_csv(filename, rows, fields):
    path = os.path.join(_HERE, filename)
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f"  CSV -> {path}")


def verdict_d(label, d, p):
    """Conclusion based on Cohen's d and p-value."""
    confirmed = d > D_THRESH and p < P_THRESH
    v = ("H1 CONFIRMED    [d>{:.1f}, p<{:.2f}]".format(D_THRESH, P_THRESH)
         if confirmed else
         "H0 NOT REJECTED [effect too small or not significant]")
    print(f"\n  RESULT: {v}")
    print(f"  {label}: Cohen d={d:.3f}, p={p:.4f}")
    return confirmed, v


def verdict_rho(rho, p):
    """Conclusion based on Spearman's rho."""
    confirmed = rho > 0 and p < P_THRESH
    v = ("H1 CONFIRMED    [rho>0, p<{:.2f}]".format(P_THRESH)
         if confirmed else
         "H0 NOT REJECTED [no significant positive correlation]")
    print(f"\n  RESULT: {v}")
    print(f"  Spearman rho={rho:.3f}, p={p:.4f}")
    return confirmed, v


# =============================================================================
# GRU model (used by Exp A and B)
# =============================================================================
class SimpleGRU(nn.Module):
    """Row-by-row Sequential MNIST classifier.
    Input shape: (batch, seq_len=28, input_size=28)."""
    def __init__(self, hidden_size=64, n_classes=10):
        super().__init__()
        self.gru = nn.GRU(28, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        traj, _ = self.gru(x)
        return self.fc(traj[:, -1]), traj


def make_gru(seed=42):
    torch.manual_seed(seed)
    return SimpleGRU().to(DEVICE)


def train_gru_one_epoch(model, loader, opt, crit):
    model.train(); correct = total = 0
    for imgs, lbls in loader:
        x = imgs.squeeze(1).to(DEVICE); lbls = lbls.to(DEVICE)
        opt.zero_grad()
        logits, _ = model(x)
        loss = crit(logits, lbls); loss.backward(); opt.step()
        correct += (logits.argmax(1) == lbls).sum().item()
        total   += lbls.size(0)
    return correct / total


def extract_gru_volumes(model, loader):
    """
    Stack every hidden-state step from every image into one long sequence.
    T = n_images × seq_len = 200 × 28 = 5 600 genuine causal time steps.
    Each volume: (1, 1, 64) — C=1, H=1, W=hidden_size.
    """
    model.eval(); vols = []
    with torch.no_grad():
        for imgs, _ in loader:
            x = imgs.squeeze(1).to(DEVICE)
            _, traj = model(x)                      # (B, 28, 64)
            for b in range(traj.shape[0]):
                for t in range(traj.shape[1]):
                    h = traj[b, t].cpu().numpy()    # (64,)
                    vols.append(h[np.newaxis, np.newaxis, :])   # (1,1,64)
    return vols


# =============================================================================
# Experiment A — GRU Trained vs Random  (n = 100)
# =============================================================================
def exp_A(n_random=100, gru_epochs=10):
    print("\n" + "=" * 68)
    print("EXP A  GRU Trained vs Random  (n={})".format(n_random))
    print()
    print("  Substrate: GRU hidden state on Sequential MNIST")
    print("  Time axis: 28 row-steps × 200 images = 5 600 causal time steps")
    print()
    print("  H1: The trained GRU's hidden-state dynamics have significantly")
    print("      higher agnostic complexity (C_a) than untrained random GRUs.")
    print("      (Cohen d > {:.1f}, p < {:.2f})".format(D_THRESH, P_THRESH))
    print("  H0: Trained GRU C_a is drawn from the same distribution as")
    print("      random GRU C_a values.")
    print("=" * 68)

    tr, te = get_data()

    # --- Trained GRU ---
    gru = make_gru(42)
    opt = optim.Adam(gru.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    print(f"\n  Training GRU ({gru_epochs} epochs)...")
    for ep in range(gru_epochs):
        acc = train_gru_one_epoch(gru, tr, opt, crit)
        print(f"    epoch {ep+1:>2d}/{gru_epochs}  acc={acc:.3f}")

    t_vols = extract_gru_volumes(gru, te)
    t_res  = compute_full_C(t_vols)
    print(f"\n  Trained: C_a={t_res['C_a']:.4f}  mi1={t_res['mi1']:.4f}  "
          f"tc={t_res['tc_mean']:.4f}  mean_H={t_res['mean_H']:.4f}")

    # --- Random GRUs ---
    rows     = [{"model": "trained", "seed": 42, **t_res}]
    rand_Ca  = []
    print(f"\n  Evaluating {n_random} random GRUs...")
    for seed in range(n_random):
        rm  = make_gru(seed)
        rv  = extract_gru_volumes(rm, te)
        rr  = compute_full_C(rv)
        rand_Ca.append(rr["C_a"])
        rows.append({"model": "random", "seed": seed, **rr})
        if seed % 25 == 0:
            print(f"    {seed:>3d}/{n_random}  C_a={rr['C_a']:.4f}")

    # --- Statistics ---
    ra   = np.array(rand_Ca)
    d    = (t_res["C_a"] - ra.mean()) / max(ra.std(), 1e-9)
    _, p = sp.ttest_1samp(rand_Ca, t_res["C_a"])
    pct  = float(sp.percentileofscore(rand_Ca, t_res["C_a"]))

    print(f"\n  Trained C_a   : {t_res['C_a']:.4f}")
    print(f"  Random mean   : {ra.mean():.4f}  std={ra.std():.4f}")
    print(f"  Percentile    : {pct:.1f}th")
    confirmed, v = verdict_d("C_a (trained GRU vs random GRUs)", d, p)

    # --- CSV ---
    fields = ["model", "seed", "C", "C_a",
              "wH_a", "wOPs_a", "wOPt_a", "wT_a", "wG_a",
              "mean_H", "std_H", "op_up", "op_down",
              "mi1", "decay", "tc_mean", "gzip_ratio"]
    save_csv("expA_gru_vs_random.csv", rows, fields)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rand_Ca, bins=25, alpha=0.7, color="mediumseagreen",
            label=f"Random GRUs (n={n_random})")
    ax.axvline(t_res["C_a"], color="red", linestyle="--", linewidth=2,
               label=f"Trained GRU  ({t_res['C_a']:.4f})")
    ax.set_xlabel("C_a"); ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    ax.text(0.04, 0.95,
            f"Rand mean = {ra.mean():.4f}\nCohen d   = {d:.3f}\n"
            f"p-value   = {p:.4f}\nPercentile = {pct:.1f}th",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85))
    ax.set_title(f"Exp A: GRU Trained vs Random\n{v}", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(_HERE, "expA_gru_vs_random.png"), dpi=150)
    plt.show()
    return confirmed


# =============================================================================
# Experiment B — GRU Training Dynamics
# =============================================================================
def exp_B(gru_epochs=15, n_random_baselines=20):
    print("\n" + "=" * 68)
    print("EXP B  GRU Training Dynamics")
    print()
    print("  Substrate: GRU hidden state on Sequential MNIST")
    print("  Checkpoint: C_a measured after every training epoch")
    print()
    print("  H1: C_a correlates positively with training accuracy")
    print("      (Spearman rho > 0, p < {:.2f}).".format(P_THRESH))
    print("      Interpretation: the metric tracks the GRU learning to")
    print("      integrate temporal information across sequence steps.")
    print("  H0: C_a does not increase with training (no significant")
    print("      positive correlation).")
    print("=" * 68)

    tr, te = get_data()
    gru    = make_gru(42)
    opt    = optim.Adam(gru.parameters(), lr=1e-3)
    crit   = nn.CrossEntropyLoss()

    rows = []

    def _checkpoint(ep, acc):
        vols = extract_gru_volumes(gru, te)
        r    = compute_full_C(vols)
        row  = {"model": "trained", "seed": 42, "epoch": ep,
                "train_acc": round(acc, 4), **r}
        rows.append(row)
        print(f"  epoch {ep:>2d}  acc={acc:.3f}  C_a={r['C_a']:.4f}  "
              f"mi1={r['mi1']:.4f}  tc={r['tc_mean']:.4f}")
        return r

    _checkpoint(0, 0.0)   # random initialisation baseline
    for ep in range(1, gru_epochs + 1):
        acc = train_gru_one_epoch(gru, tr, opt, crit)
        _checkpoint(ep, acc)

    # --- Random baselines (horizontal reference band) ---
    rand_Ca = []
    print(f"\n  Computing {n_random_baselines} random GRU baselines...")
    for seed in range(n_random_baselines):
        rm  = make_gru(seed)
        rv  = extract_gru_volumes(rm, te)
        rr  = compute_full_C(rv)
        rand_Ca.append(rr["C_a"])
        rows.append({"model": "random", "seed": seed, "epoch": -1,
                     "train_acc": -1.0, **rr})

    # --- Statistics ---
    trained_rows = [r for r in rows if r["model"] == "trained"]
    epochs_arr   = np.array([r["epoch"]     for r in trained_rows])
    ca_arr       = np.array([r["C_a"]       for r in trained_rows])
    acc_arr      = np.array([r["train_acc"] for r in trained_rows])

    rho_ep,  p_ep  = sp.spearmanr(epochs_arr, ca_arr)
    rho_acc, p_acc = sp.spearmanr(acc_arr,    ca_arr)
    rand_mean = np.mean(rand_Ca); rand_std = np.std(rand_Ca)

    print(f"\n  Spearman rho(epoch, C_a)    = {rho_ep:.3f}   p={p_ep:.4f}")
    print(f"  Spearman rho(accuracy, C_a) = {rho_acc:.3f}   p={p_acc:.4f}")
    print(f"  Random baseline: mean C_a = {rand_mean:.4f}  std={rand_std:.4f}")

    # Primary test: rho(accuracy, C_a) — most theoretically meaningful
    confirmed, v = verdict_rho(rho_acc, p_acc)

    # --- CSV ---
    fields = ["model", "seed", "epoch", "train_acc", "C", "C_a",
              "wH_a", "wOPs_a", "wOPt_a", "wT_a", "wG_a",
              "mean_H", "std_H", "op_up", "op_down",
              "mi1", "decay", "tc_mean", "gzip_ratio"]
    save_csv("expB_gru_dynamics.csv", rows, fields)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    xs = [r["epoch"] for r in trained_rows]

    axes[0].plot(xs, ca_arr, "r-o", label=f"Trained (rho={rho_ep:.2f}, p={p_ep:.3f})",
                 zorder=5)
    axes[0].axhline(rand_mean, color="mediumseagreen", linestyle="--", alpha=0.8,
                    label=f"Random mean ({rand_mean:.4f})")
    axes[0].fill_between([0, gru_epochs],
                         rand_mean - rand_std, rand_mean + rand_std,
                         alpha=0.15, color="mediumseagreen", label="Random ±1sd")
    axes[0].set_xlabel("Epoch (0 = random init)"); axes[0].set_ylabel("C_a")
    axes[0].set_title("C_a vs Epoch"); axes[0].legend(fontsize=8)

    axes[1].scatter(acc_arr, ca_arr, color="red", zorder=5)
    axes[1].set_xlabel("Training accuracy"); axes[1].set_ylabel("C_a")
    axes[1].set_title(f"C_a vs Accuracy  (rho={rho_acc:.2f}, p={p_acc:.3f})")

    fig.suptitle(f"Exp B: GRU Training Dynamics\n{v}", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(_HERE, "expB_gru_dynamics.png"), dpi=150)
    plt.show()
    return confirmed


# =============================================================================
# Experiment C — Channel Specialisation Index
# =============================================================================
def _per_channel_Ca(model, loader, n_ch=16):
    vols = extract_pre_pool(model, loader)
    return [compute_full_C([v[ch:ch+1] for v in vols])["C_a"] for ch in range(n_ch)]


def exp_C(n_random=50):
    print("\n" + "=" * 68)
    print("EXP C  Channel Specialisation Index")
    print()
    print("  Substrate: CNN conv2 pre-pool activations (16 channels)")
    print("  Metric:    SI = std( per-channel C_a )")
    print()
    print("  H1: Trained CNN has higher SI than random CNNs — training drives")
    print("      channels into specialised roles (high C_a in some channels,")
    print("      low/silent in others) rather than uniform activation.")
    print("      (Cohen d > {:.1f}, p < {:.2f})".format(D_THRESH, P_THRESH))
    print("  H0: No significant difference in SI between trained and random.")
    print("=" * 68)

    model, te = train_cnn(42, epochs=5)

    print("\n  Computing trained model per-channel C_a...")
    t_scores = _per_channel_Ca(model, te)
    t_si     = float(np.std(t_scores))
    print(f"  Trained SI = {t_si:.4f}")
    print(f"  Per-channel: {[round(x,3) for x in t_scores]}")

    rows = [{"model": "trained", "seed": 42, "SI": t_si,
             **{f"ch_{i}": t_scores[i] for i in range(16)}}]

    rand_SI = []
    print(f"\n  Evaluating {n_random} random CNNs...")
    for seed in range(n_random):
        rm  = make_cnn(seed)
        rs  = _per_channel_Ca(rm, te)
        si  = float(np.std(rs))
        rand_SI.append(si)
        rows.append({"model": "random", "seed": seed, "SI": si,
                     **{f"ch_{i}": rs[i] for i in range(16)}})
        if seed % 10 == 0:
            print(f"    {seed:>2d}/{n_random}  SI={si:.4f}")

    # --- Statistics ---
    ra   = np.array(rand_SI)
    d    = (t_si - ra.mean()) / max(ra.std(), 1e-9)
    _, p = sp.ttest_1samp(rand_SI, t_si)
    pct  = float(sp.percentileofscore(rand_SI, t_si))

    print(f"\n  Trained SI  : {t_si:.4f}")
    print(f"  Random mean : {ra.mean():.4f}  std={ra.std():.4f}")
    print(f"  Percentile  : {pct:.1f}th")
    confirmed, v = verdict_d("SI (Specialisation Index)", d, p)

    # --- CSV ---
    fields = ["model", "seed", "SI"] + [f"ch_{i}" for i in range(16)]
    save_csv("expC_channel_specialisation.csv", rows, fields)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(rand_SI, bins=25, alpha=0.7, color="steelblue",
                 label=f"Random CNNs (n={n_random})")
    axes[0].axvline(t_si, color="red", linestyle="--", linewidth=2,
                    label=f"Trained ({t_si:.4f})")
    axes[0].set_xlabel("SI = std(per-channel C_a)"); axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=9)
    axes[0].text(0.04, 0.95,
                 f"Rand mean = {ra.mean():.4f}\nCohen d   = {d:.3f}\n"
                 f"p-value   = {p:.4f}\nPercentile = {pct:.1f}th",
                 transform=axes[0].transAxes, va="top", fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.85))
    axes[0].set_title("Channel Specialisation Index distribution")

    rand_all_ch = [rows[i][f"ch_{c}"]
                   for i in range(1, len(rows)) for c in range(16)]
    axes[1].hist(rand_all_ch, bins=30, alpha=0.55, color="steelblue",
                 density=True, label=f"Random ({len(rand_all_ch)} channels)")
    axes[1].hist(t_scores, bins=12, alpha=0.75, color="red",
                 density=True, label="Trained (16 channels)")
    axes[1].set_xlabel("Per-channel C_a"); axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=9)
    axes[1].set_title("Per-channel C_a distribution")

    fig.suptitle(f"Exp C: Channel Specialisation\n{v}", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(_HERE, "expC_channel_specialisation.png"), dpi=150)
    plt.show()
    return confirmed


# =============================================================================
# Experiment D — Per-Image Spatial Complexity  (no temporal axis)
# =============================================================================
def _spatial_score(volume):
    """
    Purely spatial complexity for one (C, H, W) activation volume.
    No time dimension — avoids all dataset-ordering artefacts.

    Returns mean_H, std_H, and spatial_C = wH_agnostic (boundary gate + diversity bonus).
    """
    v       = (volume > 0).astype(np.float64)
    density = np.clip(v.reshape(v.shape[0], -1).mean(axis=1), 1e-12, 1 - 1e-12)
    H_vals  = -(density * np.log2(density) + (1 - density) * np.log2(1 - density))
    mean_H  = float(H_vals.mean())
    std_H   = float(H_vals.std())
    gate    = np.tanh(K_GATE * mean_H) * np.tanh(K_GATE * (1 - mean_H))
    bonus   = np.tanh(K_STD * float(np.clip(2 * std_H, 0, 1)))
    return {"mean_H": mean_H, "std_H": std_H, "spatial_C": float(gate * (1 + bonus))}


def exp_D(n_random=50):
    print("\n" + "=" * 68)
    print("EXP D  Per-Image Spatial Complexity  (no temporal axis)")
    print()
    print("  Substrate: CNN conv2 pre-pool, scored image-by-image")
    print("  Metric:    spatial_C = wH_agnostic(mean_H, std_H) per image")
    print("  Key property: purely spatial, zero temporal assumptions.")
    print("  This directly tests channel-diversity (std_H) as a complexity")
    print("  signal without any sequential ordering of images.")
    print()
    print("  H1: Trained CNN produces higher spatial_C than random CNNs,")
    print("      driven by greater channel-entropy diversity (std_H).")
    print("      (Cohen d > {:.1f}, p < {:.2f})".format(D_THRESH, P_THRESH))
    print("  H0: No significant difference in spatial_C.")
    print("=" * 68)

    model, te = train_cnn(42, epochs=5)

    def _model_scores(m, loader):
        vols = extract_pre_pool(m, loader)
        return [_spatial_score(v) for v in vols]

    print("\n  Scoring trained model...")
    t_scores   = _model_scores(model, te)
    t_sc_vals  = [s["spatial_C"] for s in t_scores]
    t_stdH_vals = [s["std_H"]    for s in t_scores]

    rows = [{"model": "trained", "seed": 42, "image_idx": i, **t_scores[i]}
            for i in range(len(t_scores))]

    rand_mean_sc   = []
    rand_mean_stdH = []
    print(f"\n  Scoring {n_random} random CNNs...")
    for seed in range(n_random):
        rm = make_cnn(seed)
        rs = _model_scores(rm, te)
        rand_mean_sc.append(float(np.mean([s["spatial_C"] for s in rs])))
        rand_mean_stdH.append(float(np.mean([s["std_H"]    for s in rs])))
        for i, s in enumerate(rs):
            rows.append({"model": "random", "seed": seed, "image_idx": i, **s})
        if seed % 10 == 0:
            print(f"    {seed:>2d}/{n_random}  mean_sc={rand_mean_sc[-1]:.4f}")

    # --- Statistics — primary: spatial_C ---
    t_mean_sc = float(np.mean(t_sc_vals))
    ra_sc     = np.array(rand_mean_sc)
    d_sc      = (t_mean_sc - ra_sc.mean()) / max(ra_sc.std(), 1e-9)
    _, p_sc   = sp.ttest_1samp(rand_mean_sc, t_mean_sc)
    pct_sc    = float(sp.percentileofscore(rand_mean_sc, t_mean_sc))

    # Secondary: std_H (strongest known raw signal, Cohen d ~ +1.3 from prior work)
    t_mean_stdH = float(np.mean(t_stdH_vals))
    ra_stdH     = np.array(rand_mean_stdH)
    d_stdH      = (t_mean_stdH - ra_stdH.mean()) / max(ra_stdH.std(), 1e-9)
    _, p_stdH   = sp.ttest_1samp(rand_mean_stdH, t_mean_stdH)

    print(f"\n  Primary — spatial_C:")
    print(f"    Trained mean : {t_mean_sc:.4f}")
    print(f"    Random mean  : {ra_sc.mean():.4f}  std={ra_sc.std():.4f}")
    print(f"    Percentile   : {pct_sc:.1f}th")
    print(f"\n  Secondary — std_H:")
    print(f"    Trained mean : {t_mean_stdH:.4f}")
    print(f"    Random mean  : {ra_stdH.mean():.4f}  std={ra_stdH.std():.4f}")
    print(f"    Cohen d      : {d_stdH:.3f}   p={p_stdH:.4f}")

    confirmed, v = verdict_d("spatial_C (per-image wH_agnostic)", d_sc, p_sc)

    # --- CSV ---
    fields = ["model", "seed", "image_idx", "mean_H", "std_H", "spatial_C"]
    save_csv("expD_per_image_spatial.csv", rows, fields)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (ra, t_val, lab, col, xlabel) in zip(axes, [
        (ra_sc,    t_mean_sc,    f"d={d_sc:.2f}  p={p_sc:.4f}",
         "darkorange",  "mean spatial_C across test images"),
        (ra_stdH,  t_mean_stdH,  f"d={d_stdH:.2f}  p={p_stdH:.4f}",
         "mediumpurple", "mean std_H across test images"),
    ]):
        ax.hist(ra, bins=20, alpha=0.7, color=col,
                label=f"Random CNNs (n={n_random})")
        ax.axvline(t_val, color="red", linestyle="--", linewidth=2,
                   label=f"Trained ({t_val:.4f})")
        ax.set_xlabel(xlabel); ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.text(0.04, 0.95, lab, transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.85))

    axes[0].set_title("Primary metric: spatial_C")
    axes[1].set_title("Secondary metric: std_H (channel diversity)")
    fig.suptitle(f"Exp D: Per-Image Spatial Complexity\n{v}", fontsize=9, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(_HERE, "expD_per_image_spatial.png"), dpi=150)
    plt.show()
    return confirmed


# =============================================================================
# Main runner
# =============================================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Statistical thresholds: Cohen d > {D_THRESH}, p < {P_THRESH}")

    outcomes = {}
    if RUN["A"]: outcomes["A"] = exp_A()
    if RUN["B"]: outcomes["B"] = exp_B()
    if RUN["C"]: outcomes["C"] = exp_C()
    if RUN["D"]: outcomes["D"] = exp_D()

    labels = {
        "A": "GRU trained vs random  (C_a, n=100)            ",
        "B": "GRU training dynamics  (Spearman rho)           ",
        "C": "Channel Specialisation (SI = std per-ch C_a)    ",
        "D": "Per-image spatial C    (wH_agnostic, no time)   ",
    }
    print("\n\n" + "=" * 68)
    print("EXPERIMENT SUMMARY")
    print(f"Thresholds: Cohen d > {D_THRESH}, p < {P_THRESH}")
    print("=" * 68)
    for k, confirmed in outcomes.items():
        status = "H1 CONFIRMED" if confirmed else "H0 NOT REJECTED"
        print(f"  Exp {k}: {labels[k]}  [{status}]")
    print("=" * 68)
