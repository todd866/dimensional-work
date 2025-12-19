"""
Sim 3: Autoencoder Bottleneck Sweep (SGD-as-Langevin)

Demonstrates that forcing a representation into dimensions lower than the
intrinsic data dimensionality causes an explosion in "learning effort" (gradient work).

Interpretation: SGD is an overdamped Langevin process in parameter space.
The loss function L(theta) is an energy potential.
Cumulative gradient norm serves as a proxy for thermodynamic work.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)
rng = np.random.default_rng(2)


def make_data(n_samples=5000, in_dim=10, intrinsic_dim=2):
    """
    Generate synthetic data: Gaussian clusters in intrinsic_dim,
    projected linearly into in_dim with nonlinearity.
    """
    centers = rng.normal(size=(4, intrinsic_dim)) * 3.0
    X_intr = []
    for _ in range(n_samples):
        c = centers[rng.integers(0, 4)]
        x = c + rng.normal(scale=0.5, size=intrinsic_dim)
        X_intr.append(x)
    X_intr = np.stack(X_intr)
    W = rng.normal(size=(intrinsic_dim, in_dim))
    X = X_intr @ W
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return X.astype(np.float32)


class Autoencoder(nn.Module):
    def __init__(self, in_dim=10, bottleneck=3, hidden=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def train_autoencoder(X, bottleneck, n_epochs=50, batch_size=128, device="cpu"):
    """
    Train autoencoder and return reconstruction error + cumulative gradient norm.

    SGD-as-Langevin interpretation:
    - Loss L(theta) is an energy potential
    - Gradient descent updates approximate overdamped Langevin in theta-space
    - Cumulative gradient norm is a proxy for thermodynamic work
    """
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder(in_dim=X.shape[1], bottleneck=bottleneck).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    cumulative_grad_norm = 0.0
    for _ in range(n_epochs):
        for batch, in loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)
            loss = loss_fn(out, batch)
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm ** 2
            cumulative_grad_norm += np.sqrt(total_norm)
            opt.step()

    with torch.no_grad():
        X_t = torch.from_numpy(X).to(device)
        recon = model(X_t)
        recon_err = loss_fn(recon, X_t).item()
    return recon_err, cumulative_grad_norm


def main(save_fig=True):
    X = make_data()
    bottlenecks = [1, 2, 3, 4, 5, 6, 8, 10]
    recon_means = []
    recon_stds = []
    effort_means = []
    effort_stds = []
    n_seeds = 5

    for db in bottlenecks:
        errs = []
        effs = []
        print(f"Bottleneck = {db}")
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            err, effort = train_autoencoder(X, bottleneck=db)
            errs.append(err)
            effs.append(effort)
        errs = np.array(errs)
        effs = np.array(effs)
        recon_means.append(errs.mean())
        recon_stds.append(errs.std())
        effort_means.append(effs.mean())
        effort_stds.append(effs.std())
        print(f"  recon err: {errs.mean():.4f} +/- {errs.std():.4f}, "
              f"effort: {effs.mean():.1f} +/- {effs.std():.1f}")

    bottlenecks = np.array(bottlenecks)
    recon_means = np.array(recon_means)
    recon_stds = np.array(recon_stds)
    effort_means = np.array(effort_means)
    effort_stds = np.array(effort_stds)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    color1 = '#2c3e50'
    color2 = '#c0392b'

    ax1.errorbar(bottlenecks, recon_means, yerr=recon_stds,
                 marker="o", capsize=4, color=color1)
    ax2.errorbar(bottlenecks, effort_means, yerr=effort_stds,
                 marker="s", capsize=4, color=color2)

    # Mark intrinsic dimension
    ax1.axvline(x=2, color='gray', linestyle=':', alpha=0.7,
                label='Intrinsic dim')

    ax1.set_xlabel("Bottleneck dimension $d_b$")
    ax1.set_ylabel("Reconstruction MSE", color=color1)
    ax2.set_ylabel("Cumulative gradient norm (SGD work)", color=color2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("The Cost of Over-Compression")
    plt.tight_layout()

    if save_fig:
        plt.savefig('../figures/fig3_autoencoder.png', dpi=150)
        plt.savefig('../figures/fig3_autoencoder.pdf')
    plt.show()

    return bottlenecks, recon_means, effort_means


if __name__ == "__main__":
    main()
