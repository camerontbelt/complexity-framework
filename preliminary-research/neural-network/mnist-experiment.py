import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ==========================================================
# Complexity proxy (adapted from user's v9 proxy experiment)
# ==========================================================

def compute_spatial_entropy(grid):
    density = np.mean(grid)
    if density <= 0 or density >= 1:
        return 0.0
    return -density * np.log2(density) - (1 - density) * np.log2(1 - density)

K_ENTROPY = 50
MU_SIGMA = 0.012
SIGMA_SIGMA = 0.008
TCOMP_PEAKS = [(0.58, 0.08), (0.73, 0.08), (0.90, 0.05)]
GZIP_MU, GZIP_SIGMA = 0.10, 0.05


def wh_weight(Hs, sigma_H):
    gate1 = np.tanh(K_ENTROPY * Hs)
    gate2 = np.tanh(K_ENTROPY * (1 - Hs))
    gaussian = 1 + np.exp(-((sigma_H - MU_SIGMA) ** 2) / (2 * SIGMA_SIGMA ** 2))
    return gate1 * gate2 * gaussian


def wt_weight(tc_mean):
    w = 0.0
    for mu, sigma in TCOMP_PEAKS:
        w = max(w, np.exp(-((tc_mean - mu) ** 2) / (2 * sigma ** 2)))
    return w


def wg_weight(gzip_ratio):
    return np.exp(-((gzip_ratio - GZIP_MU) ** 2) / (2 * GZIP_SIGMA ** 2))


def compute_proxy_C(binary_grids):
    """
    binary_grids: list of (64,64) numpy arrays across samples.
    Treats sample index as ensemble axis (not time).
    """
    Hs = np.array([compute_spatial_entropy(g) for g in binary_grids])
    sigma_H = np.std(Hs)
    wH = wh_weight(np.mean(Hs), sigma_H)

    tc = []
    T = len(binary_grids)
    for i in range(64):
        for j in range(64):
            history = np.array([binary_grids[t][i, j] for t in range(T)])
            flips = np.sum(np.diff(history) != 0)
            tc.append(1 - (1 + flips) / T)

    tc_mean = np.mean(tc)
    wT = wt_weight(tc_mean)

    last_grid_bytes = binary_grids[-1].astype(np.uint8).tobytes()
    gzip_ratio = len(gzip.compress(last_grid_bytes)) / len(last_grid_bytes)
    wG = wg_weight(gzip_ratio)

    C = wH * wT * wG
    return {
        "C": float(C),
        "wH": float(wH),
        "wT": float(wT),
        "wG": float(wG),
        "mean_H": float(np.mean(Hs)),
        "sigma_H": float(sigma_H),
    }


# =========================
# Model
# =========================

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 4096)
        self.fc2 = nn.Linear(4096, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        hidden = torch.relu(self.fc1(x))
        out = self.fc2(hidden)
        return out, hidden


# =========================
# Data
# =========================

def get_mnist_subset(train_size=2000, test_size=200, batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_subset = Subset(train_dataset, range(train_size))
    test_subset = Subset(test_dataset, range(test_size))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=test_size, shuffle=False)

    return train_loader, test_loader


# =========================
# Training
# =========================

def train_model(model, train_loader, device, epochs=5, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

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

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {running_loss/total:.4f} | "
            f"Acc: {correct/total:.3f}"
        )


# =========================
# Activation extraction
# =========================

def extract_activation_grids(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.to(device)
        _, hidden = model(images)

    hidden_np = hidden.cpu().numpy()

    # Per-neuron median thresholding (better than fixed 0.5)
    neuron_medians = np.median(hidden_np, axis=0, keepdims=True)
    binary = (hidden_np > neuron_medians).astype(np.uint8)

    grids = [sample.reshape(64, 64) for sample in binary]
    return grids


# =========================
# Main experiment
# =========================

def run_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, test_loader = get_mnist_subset()

    # Trained model
    print("\nTraining model...")
    torch.manual_seed(42)
    trained_model = SimpleMLP().to(device)
    train_model(trained_model, train_loader, device, epochs=5)

    trained_grids = extract_activation_grids(trained_model, test_loader, device)
    trained_results = compute_proxy_C(trained_grids)

    # Random controls
    print("\nEvaluating random controls...")
    random_cs = []
    for seed in range(30):
        torch.manual_seed(seed)
        rand_model = SimpleMLP().to(device)
        rand_grids = extract_activation_grids(rand_model, test_loader, device)
        result = compute_proxy_C(rand_grids)
        random_cs.append(result["C"])

    random_mean = float(np.mean(random_cs))
    random_std = float(np.std(random_cs))
    cohen_d = (trained_results["C"] - random_mean) / max(random_std, 1e-8)

    print("\n=== MNIST Complexity Results ===")
    print(f"Trained C          : {trained_results['C']:.5f}")
    print(f"Random mean C      : {random_mean:.5f}")
    print(f"Random std         : {random_std:.5f}")
    print(f"Separation ratio   : {trained_results['C']/max(random_mean,1e-8):.2f}x")
    print(f"Cohen's d          : {cohen_d:.2f}")
    print(f"Trained wH / wT / wG: {trained_results['wH']:.4f}, {trained_results['wT']:.4f}, {trained_results['wG']:.4f}")


if __name__ == "__main__":
    run_experiment()
