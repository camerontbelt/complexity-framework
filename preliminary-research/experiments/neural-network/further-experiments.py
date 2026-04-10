"""
further-experiments.py
======================
Five experiments that probe when and where the complexity metric (v8 and
agnostic) captures learning in neural networks.

  Exp 1 — Training dynamics      : does C change monotonically as training progresses?
  Exp 2 — Layer-wise complexity  : how does C evolve across network depth?
  Exp 3 — Permutation control    : do temporal metrics depend on image ordering?
  Exp 4 — Per-channel analysis   : distribution of complexity across feature channels
  Exp 5 — GRU on Sequential MNIST: genuine causal temporal dynamics in an RNN

Toggle experiments with the RUN dict below, then run:
    python further-experiments.py
"""

import os
import sys
import importlib.util
import random as _random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Bootstrap — load shared functions from the sibling mnist-experiment.py
# =============================================================================
_HERE  = os.path.dirname(os.path.abspath(__file__))
_spec  = importlib.util.spec_from_file_location(
            "mnist_exp",
            os.path.join(_HERE, "mnist-experiment.py"))
_mod   = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_full_C = _mod.compute_full_C   # (volumes) -> dict
SimpleCNN      = _mod.SimpleCNN        # CNN architecture
train_model    = _mod.train_model      # (model, loader, device, lr, epochs)

# =============================================================================
# Toggle experiments
# =============================================================================
RUN = {
    1: True,   # Training dynamics
    2: True,   # Layer-wise complexity
    3: True,   # Permutation control
    4: True,   # Per-channel analysis
    5: True,   # GRU on Sequential MNIST
}

# =============================================================================
# Shared helpers
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_data(train_size=2000, test_size=200):
    t  = transforms.Compose([transforms.ToTensor()])
    tr = datasets.MNIST("./data", train=True,  download=True, transform=t)
    te = datasets.MNIST("./data", train=False, download=True, transform=t)
    return (DataLoader(Subset(tr, range(train_size)), batch_size=128, shuffle=True),
            DataLoader(Subset(te, range(test_size)),  batch_size=128, shuffle=False))

def make_cnn(seed=42):
    torch.manual_seed(seed)
    return SimpleCNN().to(DEVICE)

def train_cnn(seed=42, epochs=5):
    tr, te = get_data()
    m = make_cnn(seed)
    train_model(m, tr, DEVICE, epochs=epochs)
    return m, te

def extract_pre_pool(model, loader):
    """Pre-pool conv2 activations — one (16,28,28) volume per image."""
    model.eval(); vols = []
    with torch.no_grad():
        for imgs, _ in loader:
            _, pp = model(imgs.to(DEVICE))
            for i in range(pp.shape[0]):
                vols.append(pp[i].cpu().numpy())
    return vols

def extract_pre_pool_labelled(model, loader):
    """Same, but also returns the class label for each image."""
    model.eval(); vols, labs = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            _, pp = model(imgs.to(DEVICE))
            for i in range(pp.shape[0]):
                vols.append(pp[i].cpu().numpy())
                labs.append(lbls[i].item())
    return vols, labs

def extract_all_layers(model, loader):
    """Activations at every layer, per image.
    Conv layers returned as (C,H,W); FC layers as (1,1,N) for metric compatibility."""
    model.eval()
    out = {'conv1': [], 'conv2': [], 'fc1': [], 'fc2': []}
    with torch.no_grad():
        for imgs, _ in loader:
            x  = imgs.to(DEVICE)
            c1 = torch.relu(model.conv1(x))                   # (B,8,28,28)
            c2 = torch.relu(model.conv2(c1))                  # (B,16,28,28)
            h  = model.pool(c2).view(c2.shape[0], -1)
            f1 = torch.relu(model.fc1(h))                     # (B,128)
            f2 = model.fc2(f1)                                # (B,10) raw logits
            for b in range(x.shape[0]):
                out['conv1'].append(c1[b].cpu().numpy())       # (8,28,28)
                out['conv2'].append(c2[b].cpu().numpy())       # (16,28,28)
                out['fc1'].append(f1[b].cpu().numpy()[np.newaxis, np.newaxis, :])  # (1,1,128)
                out['fc2'].append(f2[b].cpu().numpy()[np.newaxis, np.newaxis, :])  # (1,1,10)
    return out

def _save(fig, name):
    path = os.path.join(_HERE, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved -> {path}")


# =============================================================================
# Experiment 1 — Training Dynamics
# =============================================================================
def exp1_training_dynamics(epochs=10, n_random=15):
    """
    Measure C and C_a at every training epoch (epoch 0 = random initialisation).
    Hypothesis: if the metric tracks learning, both should shift monotonically
    (or at least directionally) as the model improves.
    """
    print("\n" + "=" * 60)
    print("EXP 1 — Training Dynamics")
    print("=" * 60)

    tr, te = get_data()
    model  = make_cnn(42)
    opt    = optim.Adam(model.parameters(), lr=1e-3)
    crit   = nn.CrossEntropyLoss()

    hist = {'epoch': [], 'C': [], 'C_a': [], 'acc': []}

    def _checkpoint(ep, acc):
        vols = extract_pre_pool(model, te)
        r    = compute_full_C(vols)
        hist['epoch'].append(ep)
        hist['C'].append(r['C']); hist['C_a'].append(r['C_a'])
        hist['acc'].append(acc)
        print(f"  epoch {ep:>2d}  acc={acc:.3f}  C={r['C']:.4f}  C_a={r['C_a']:.4f}  "
              f"std_H={r['std_H']:.4f}")

    _checkpoint(0, 0.0)   # before any training

    for ep in range(1, epochs + 1):
        model.train(); correct = total = 0
        for imgs, lbls in tr:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            opt.zero_grad()
            out, _ = model(imgs)
            loss = crit(out, lbls); loss.backward(); opt.step()
            correct += (out.argmax(1) == lbls).sum().item(); total += lbls.size(0)
        _checkpoint(ep, correct / total)

    # Random baseline
    rand_C, rand_Ca = [], []
    for seed in range(n_random):
        rm = make_cnn(seed + 500)
        rv = extract_pre_pool(rm, te)
        rr = compute_full_C(rv)
        rand_C.append(rr['C']); rand_Ca.append(rr['C_a'])

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    xs = hist['epoch']

    for ax, key, rands, col, title in [
        (axes[0], 'C',   rand_C,  'steelblue',    'v8 C vs epoch'),
        (axes[1], 'C_a', rand_Ca, 'mediumseagreen','Agnostic C_a vs epoch'),
    ]:
        ax.plot(xs, hist[key], 'r-o', label='Training model', zorder=5)
        rmu, rsd = np.mean(rands), np.std(rands)
        ax.axhline(rmu, color=col, linestyle='--', alpha=0.8,
                   label=f'Random mean ({rmu:.4f})')
        ax.fill_between([0, epochs], rmu - rsd, rmu + rsd,
                        alpha=0.15, color=col, label='Random ±1σ')
        ax.set_xlabel('Epoch (0 = init)'); ax.set_ylabel(key)
        ax.set_title(title); ax.legend(fontsize=8)

    axes[2].plot(xs, hist['acc'], 'g-o')
    axes[2].set_title('Training accuracy'); axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')

    fig.suptitle('Experiment 1: Complexity vs Training Progress',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'exp1_training_dynamics.png')
    plt.show()
    return hist


# =============================================================================
# Experiment 2 — Layer-wise Complexity
# =============================================================================
def exp2_layer_wise(n_random=20):
    """
    Compute C separately at each layer (conv1, conv2, fc1, fc2).
    T = test images (200), W = neurons in that layer.
    Shows how representational complexity evolves with network depth.

    Note: fc1 and fc2 lack spatial 2D structure; std_H will be 0 (1 channel).
    The temporal metrics and mean_H are still meaningful at those layers.
    """
    print("\n" + "=" * 60)
    print("EXP 2 — Layer-wise Complexity")
    print("=" * 60)

    model, te = train_cnn()
    layers     = ['conv1', 'conv2', 'fc1', 'fc2']

    def _layer_C(m, loader):
        acts    = extract_all_layers(m, loader)
        results = {}
        for l in layers:
            results[l] = compute_full_C(acts[l])
        return results

    print("  Trained model:")
    trained_lc = _layer_C(model, te)
    for l in layers:
        r = trained_lc[l]
        print(f"    {l}: C={r['C']:.4f}  C_a={r['C_a']:.4f}  "
              f"mean_H={r['mean_H']:.4f}  std_H={r['std_H']:.4f}")

    rand_lc = {l: {'C': [], 'C_a': [], 'mean_H': [], 'std_H': []} for l in layers}
    for seed in range(n_random):
        rm = make_cnn(seed)
        rr = _layer_C(rm, te)
        for l in layers:
            for k in rand_lc[l]:
                rand_lc[l][k].append(rr[l][k])
        if seed % 5 == 0:
            print(f"  Random {seed}/{n_random}")

    # Plot
    metrics = [
        ('C',      'v8 C',               'steelblue'),
        ('C_a',    'Agnostic C_a',        'mediumseagreen'),
        ('mean_H', 'mean_H (entropy)',    'darkorange'),
        ('std_H',  'std_H (ch. diversity)','mediumpurple'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    x = np.arange(len(layers))

    for ax, (key, title, col) in zip(axes, metrics):
        t_vals   = [trained_lc[l][key] for l in layers]
        r_means  = [np.mean(rand_lc[l][key]) for l in layers]
        r_stds   = [np.std(rand_lc[l][key]) for l in layers]
        ax.errorbar(x, r_means, yerr=r_stds, fmt='o--',
                    color=col, capsize=4, label='Random mean ±1σ')
        ax.plot(x, t_vals, 'r-o', zorder=5, label='Trained')
        ax.set_xticks(x); ax.set_xticklabels(layers)
        ax.set_title(title); ax.set_ylabel(key); ax.legend(fontsize=8)
        ax.set_xlabel('Layer (input  ->  output)')

    fig.suptitle('Experiment 2: Layer-wise Complexity\n'
                 'T = test images (200), W = neurons in that layer',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'exp2_layer_wise.png')
    plt.show()
    return trained_lc, rand_lc


# =============================================================================
# Experiment 3 — Permutation Control
# =============================================================================
def exp3_permutation_control(n_random_orderings=5):
    """
    Compute C under different image orderings: original, class-sorted, reversed,
    and several random shuffles.
    Diagnostic question: if the temporal metrics (mi1, tc_mean) change significantly
    with ordering, the 'time' axis is measuring dataset structure, not model dynamics.
    """
    print("\n" + "=" * 60)
    print("EXP 3 — Permutation Control")
    print("=" * 60)

    model, te = train_cnn()
    vols, labs = extract_pre_pool_labelled(model, te)
    N = len(vols)

    orderings = {
        'original':    list(range(N)),
        'class_sorted':sorted(range(N), key=lambda i: labs[i]),
        'reverse_class':sorted(range(N), key=lambda i: -labs[i]),
    }
    for k in range(n_random_orderings):
        perm = list(range(N)); _random.shuffle(perm)
        orderings[f'random_{k}'] = perm

    results = {}
    for name, ordering in orderings.items():
        v_ord = [vols[i] for i in ordering]
        r     = compute_full_C(v_ord)
        results[name] = r
        print(f"  {name:22s}  mi1={r['mi1']:.4f}  decay={r['decay']:.4f}  "
              f"tc={r['tc_mean']:.4f}  C_a={r['C_a']:.4f}")

    names     = list(orderings.keys())
    x_labels  = ['original', 'class\nsorted', 'rev\nclass'] + \
                [f'rand {k}' for k in range(n_random_orderings)]
    colors    = ['red', 'royalblue', 'navy'] + ['gray'] * n_random_orderings
    metrics   = [
        ('mi1',      'Temporal MI at lag 1'),
        ('decay',    'Temporal MI decay'),
        ('tc_mean',  'Temporal compression tc'),
        ('C_a',      'Agnostic composite C_a'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (key, title) in zip(axes, metrics):
        vals = [results[n][key] for n in names]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=40, ha='right', fontsize=8)
        ax.set_title(title); ax.set_ylabel(key)

    fig.suptitle(
        'Experiment 3: Permutation Control\n'
        'If temporal metrics change with image ordering, the time axis is a dataset artefact',
        fontsize=10, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'exp3_permutation.png')
    plt.show()
    return results


# =============================================================================
# Experiment 4 — Per-Channel Analysis
# =============================================================================
def exp4_per_channel(n_random=15):
    """
    Compute complexity separately for each of the 16 conv2 channels.
    Prediction: trained model shows a wider, more bimodal channel distribution
    (some channels specialised for specific features, others quiet) compared to
    the roughly unimodal distribution from random CNNs.
    """
    print("\n" + "=" * 60)
    print("EXP 4 — Per-Channel Complexity  (16 channels × T=200 images)")
    print("=" * 60)

    model, te = train_cnn()
    n_ch      = 16

    def _per_ch(m, loader):
        vols     = extract_pre_pool(m, loader)
        ch_C, ch_Ca, ch_H = [], [], []
        for ch in range(n_ch):
            ch_vols = [v[ch:ch+1, :, :] for v in vols]   # keep (1,H,W) shape
            r = compute_full_C(ch_vols)
            ch_C.append(r['C']); ch_Ca.append(r['C_a']); ch_H.append(r['mean_H'])
        return ch_C, ch_Ca, ch_H

    print("  Trained model...")
    t_C, t_Ca, t_H = _per_ch(model, te)
    print(f"  Trained ch C_a: min={min(t_Ca):.3f}  max={max(t_Ca):.3f}  "
          f"std={np.std(t_Ca):.3f}")

    r_C, r_Ca, r_H = [], [], []
    for seed in range(n_random):
        rm = make_cnn(seed)
        c, ca, h = _per_ch(rm, te)
        r_C.extend(c); r_Ca.extend(ca); r_H.extend(h)
        if seed % 5 == 0:
            print(f"  Random {seed}/{n_random}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (t_vals, r_vals, title, xl) in zip(axes, [
        (t_C,  r_C,  'Per-channel v8 C',          'C'),
        (t_Ca, r_Ca, 'Per-channel Agnostic C_a',   'C_a'),
        (t_H,  r_H,  'Per-channel mean_H (entropy)','mean_H'),
    ]):
        ax.hist(r_vals, bins=30, alpha=0.55, color='steelblue', density=True,
                label=f'Random ({len(r_vals)} ch)')
        ax.hist(t_vals, bins=12, alpha=0.75, color='red', density=True,
                label=f'Trained (16 ch)')
        d = (np.mean(t_vals) - np.mean(r_vals)) / max(np.std(r_vals), 1e-9)
        ax.set_title(title); ax.set_xlabel(xl); ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.text(0.04, 0.95, f"Cohen d = {d:.2f}",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))

    fig.suptitle('Experiment 4: Per-channel Complexity Distribution\n'
                 'Wider/bimodal distribution for trained model = specialised channels',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'exp4_per_channel.png')
    plt.show()
    return t_Ca, r_Ca


# =============================================================================
# Experiment 5 — GRU on Sequential MNIST
# =============================================================================
class SimpleGRU(nn.Module):
    """
    Row-by-row Sequential MNIST classifier.
    Input: (batch, seq_len=28, input_size=28)  — one row of pixels per step.
    Hidden state: (batch, seq_len, hidden_size) — full trajectory extracted.
    """
    def __init__(self, input_size=28, hidden_size=64, n_classes=10):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, n_classes)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x: (B, 28, 28)
        traj, _ = self.gru(x)          # (B, 28, 64)
        logits  = self.fc(traj[:, -1]) # classify from last step
        return logits, traj


def _train_gru(model, loader, epochs=5):
    opt  = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        correct = total = 0
        for imgs, lbls in loader:
            x = imgs.squeeze(1).to(DEVICE)   # (B,28,28) — 28 rows of 28 px
            lbls = lbls.to(DEVICE)
            opt.zero_grad()
            logits, _ = model(x)
            loss = crit(logits, lbls); loss.backward(); opt.step()
            correct += (logits.argmax(1) == lbls).sum().item()
            total   += lbls.size(0)
        print(f"    GRU epoch {ep+1}/{epochs}  acc={correct/total:.3f}")


def _extract_rnn_volumes(model, loader):
    """
    Stack every hidden-state time step from every image into one long sequence.
    Returns T = n_images × seq_len volumes, each of shape (1, 1, hidden_size).
    This gives genuine causal temporal structure: each step t causally follows t-1
    within an image. The between-image boundary (3.6% of steps) adds minor noise.
    """
    model.eval(); vols = []
    with torch.no_grad():
        for imgs, _ in loader:
            x = imgs.squeeze(1).to(DEVICE)
            _, traj = model(x)            # (B, 28, 64)
            for b in range(traj.shape[0]):
                for t in range(traj.shape[1]):
                    h = traj[b, t].cpu().numpy()          # (64,)
                    vols.append(h[np.newaxis, np.newaxis, :])  # (1,1,64)
    return vols


def exp5_rnn(n_random=20):
    """
    Train a GRU to classify MNIST row-by-row and measure complexity on its
    hidden-state sequence.

    Why this is fundamentally different from the CNN experiment:
    - Time axis = actual causal sequence steps (row 0 -> row 1 -> ... -> row 27)
    - NOT independent test images treated as pseudo-time
    - T = 200 images × 28 rows = 5600 genuine temporal steps

    std_H will be 0 because there is only 1 'channel' (the 64-unit hidden state is
    treated as 64 spatial positions with C=1). All other metrics are fully meaningful.
    """
    print("\n" + "=" * 60)
    print("EXP 5 — GRU on Sequential MNIST")
    print("T = 200 images × 28 row-steps = 5600 genuine causal time steps")
    print("=" * 60)

    tr, te = get_data()

    torch.manual_seed(42)
    rnn = SimpleGRU().to(DEVICE)
    print("  Training GRU...")
    _train_gru(rnn, tr)

    print("  Extracting trained GRU hidden states...")
    t_vols = _extract_rnn_volumes(rnn, te)
    t_res  = compute_full_C(t_vols)
    print(f"  Trained: C={t_res['C']:.4f}  C_a={t_res['C_a']:.4f}  "
          f"mi1={t_res['mi1']:.4f}  tc={t_res['tc_mean']:.4f}  "
          f"mean_H={t_res['mean_H']:.4f}  gzip={t_res['gzip_ratio']:.4f}")

    rand_C, rand_Ca = [], []
    rand_rows = []
    for seed in range(n_random):
        torch.manual_seed(seed)
        rm = SimpleGRU().to(DEVICE)
        rv = _extract_rnn_volumes(rm, te)
        rr = compute_full_C(rv)
        rand_C.append(rr['C']); rand_Ca.append(rr['C_a'])
        rand_rows.append(rr)
        if seed % 5 == 0:
            print(f"  Random {seed}/{n_random}  C={rr['C']:.4f}  C_a={rr['C_a']:.4f}  "
                  f"mi1={rr['mi1']:.4f}  tc={rr['tc_mean']:.4f}")

    # Summary table
    all_keys = ['mi1', 'decay', 'tc_mean', 'mean_H', 'gzip_ratio', 'C', 'C_a']
    print("\n  -- GRU component summary --------------------------------")
    fmt = "  {:>12s}  trained={:>8.4f}  rand_mean={:>8.4f}  d={:>6.2f}"
    for k in all_keys:
        rv_arr = np.array([r[k] for r in rand_rows])
        d = (t_res[k] - rv_arr.mean()) / max(rv_arr.std(), 1e-9)
        print(fmt.format(k, t_res[k], rv_arr.mean(), d))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, rands, col, title in [
        (axes[0], 'C',   rand_C,  'steelblue',    'v8 C: GRU trained vs random'),
        (axes[1], 'C_a', rand_Ca, 'mediumseagreen','Agnostic C_a: GRU trained vs random'),
    ]:
        ax.hist(rands, bins=20, alpha=0.7, color=col, label='Random GRUs')
        ax.axvline(t_res[key], color='red', linestyle='--', linewidth=2,
                   label=f'Trained GRU ({t_res[key]:.4f})')
        rmu = np.mean(rands); rsd = np.std(rands)
        d   = (t_res[key] - rmu) / max(rsd, 1e-8)
        _, pv = stats.ttest_1samp(rands, t_res[key])
        ax.set_title(title); ax.set_xlabel(key); ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.text(0.05, 0.95,
                f"Rand mean = {rmu:.4f}\nCohen d = {d:.2f}\np = {pv:.4f}",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))

    fig.suptitle('Experiment 5: GRU Hidden State Complexity\n'
                 'T=5600 genuine causal sequence steps (28 rows × 200 images)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'exp5_rnn.png')
    plt.show()
    return t_res, rand_rows


# =============================================================================
# Main runner
# =============================================================================
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    results = {}
    if RUN[1]: results[1] = exp1_training_dynamics()
    if RUN[2]: results[2] = exp2_layer_wise()
    if RUN[3]: results[3] = exp3_permutation_control()
    if RUN[4]: results[4] = exp4_per_channel()
    if RUN[5]: results[5] = exp5_rnn()
    print("\n\nAll selected experiments complete.")
