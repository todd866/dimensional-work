#!/usr/bin/env python3
"""
Dimensional Landauer Bound - Complete Simulation Suite

All simulations for the paper:
"The Dimensional Landauer Bound: Why Information Flow Requires Geometric Work"

Usage:
    python simulations.py              # Run all simulations
    python simulations.py concept      # Just the concept figure
    python simulations.py curvature    # Sim 1: Brownian manifolds
    python simulations.py kuramoto     # Sim 2: Kuramoto oscillators
    python simulations.py autoencoder  # Sim 3: Autoencoder bottleneck
    python simulations.py ephaptic     # Sim 4: Ephaptic neural sheet
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Optional torch import for autoencoder
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Global RNG
rng = np.random.default_rng(42)

# Publication style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


# =============================================================================
# FIGURE 0: Concept Figure
# =============================================================================

def fig0_concept(save_fig=True):
    """
    Generate concept figure for Dimensional Landauer Bound paper.
    Panel (a): Classical Landauer - bit erasure
    Panel (b): Dimensional Landauer - projection to lower-dimensional manifold
    """
    fig = plt.figure(figsize=(7, 3.5))

    # Panel (a): Classical Landauer - bit erasure
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 1.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('(a) Classical Landauer', fontweight='bold', fontsize=11)

    # Draw two potential wells merging to one
    well_x = np.linspace(-0.3, 0.3, 50)
    well_y = 4 * well_x**2

    # State 0 well
    ax1.plot(well_x + 0.3, well_y + 0.1, 'b-', lw=2)
    ax1.fill_between(well_x + 0.3, 0, well_y + 0.1, alpha=0.3, color='blue')
    ax1.text(0.3, -0.15, '0', ha='center', fontsize=12, fontweight='bold')

    # State 1 well
    ax1.plot(well_x + 1.0, well_y + 0.1, 'r-', lw=2)
    ax1.fill_between(well_x + 1.0, 0, well_y + 0.1, alpha=0.3, color='red')
    ax1.text(1.0, -0.15, '1', ha='center', fontsize=12, fontweight='bold')

    # Arrow
    ax1.annotate('', xy=(1.9, 0.5), xytext=(1.4, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Final: single well (erased to 0)
    ax1.plot(well_x + 2.2, well_y + 0.1, 'b-', lw=2)
    ax1.fill_between(well_x + 2.2, 0, well_y + 0.1, alpha=0.5, color='blue')
    ax1.text(2.2, -0.15, '0', ha='center', fontsize=12, fontweight='bold')

    # Work annotation
    ax1.text(1.0, 1.2, r'$W \geq k_B T \ln 2$', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Heat dissipation arrow
    ax1.annotate('', xy=(2.2, 0.8), xytext=(2.2, 0.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='orange'))
    ax1.text(2.35, 0.65, 'Q', fontsize=10, color='orange')

    # Panel (b): Dimensional Landauer - projection
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('(b) Dimensional Landauer', fontweight='bold', fontsize=11)

    # Create high-dimensional point cloud
    n_points = 200
    theta = rng.uniform(0, 2*np.pi, n_points)
    r = rng.normal(1, 0.3, n_points)
    z = rng.normal(0, 0.4, n_points)
    x = r * np.cos(theta) + rng.normal(0, 0.15, n_points)
    y = r * np.sin(theta) + rng.normal(0, 0.15, n_points)

    # Plot high-D cloud
    ax2.scatter(x, y, z, c='steelblue', alpha=0.4, s=8, label=r'$D_{\rm in}$')

    # Create 1D manifold (curved line)
    t_manifold = np.linspace(0, 2*np.pi, 100)
    x_manifold = np.cos(t_manifold)
    y_manifold = np.sin(t_manifold)
    z_manifold = np.zeros_like(t_manifold)

    # Plot the manifold
    ax2.plot(x_manifold, y_manifold, z_manifold, 'r-', lw=3,
             label=r'$D_{\rm out}$ manifold')

    # Projection lines
    for i in range(0, n_points, 20):
        angle = np.arctan2(y[i], x[i])
        x_proj = np.cos(angle)
        y_proj = np.sin(angle)
        ax2.plot([x[i], x_proj], [y[i], y_proj], [z[i], 0],
                 'k--', alpha=0.3, lw=0.5)

    # Styling
    ax2.set_xlabel(r'$x_1$', labelpad=-8)
    ax2.set_ylabel(r'$x_2$', labelpad=-8)
    ax2.set_zlabel(r'$x_3$', labelpad=-8)
    ax2.set_xlim(-1.8, 1.8)
    ax2.set_ylim(-1.8, 1.8)
    ax2.set_zlim(-1.2, 1.2)
    ax2.view_init(elev=25, azim=45)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])

    ax2.text2D(0.5, 0.02, r'$W_{\rm dim} \propto k_B T \, C_\Phi$',
               transform=ax2.transAxes, ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax2.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    if save_fig:
        plt.savefig('../figures/fig0_concept.pdf', bbox_inches='tight', dpi=300)
        plt.savefig('../figures/fig0_concept.png', bbox_inches='tight', dpi=150)
        print("Saved fig0_concept.pdf and fig0_concept.png")
    plt.show()


# =============================================================================
# SIMULATION 1: Brownian Motion on Curved Manifolds
# =============================================================================

def f_curve(x1, A, B):
    """Manifold definition: x2 = A * sin(B * x1)"""
    return A * np.sin(B * x1)


def df_curve(x1, A, B):
    """Derivative of manifold"""
    return A * B * np.cos(B * x1) if (A != 0 and B != 0) else 0.0


def curvature(x1, A, B):
    """Signed curvature kappa = f''/(1 + f'^2)^(3/2)"""
    if A == 0 or B == 0:
        return 0.0
    f_prime = A * B * np.cos(B * x1)
    f_double_prime = -A * B**2 * np.sin(B * x1)
    return f_double_prime / (1 + f_prime**2)**1.5


def F_ctrl(x, A, B, k):
    """Control force from U = k/2 (x2 - f(x1))^2."""
    x1, x2 = x
    f_val = f_curve(x1, A, B)
    df_dx1 = df_curve(x1, A, B)
    g_val = x2 - f_val
    grad_g = np.array([-df_dx1, 1.0])
    return -k * g_val * grad_g


def theoretical_C_phi(A, B, k, D):
    """Theoretical contraction cost for confining to manifold."""
    kB_T = D
    sigma_perp_sq = kB_T / k
    C_dim = 0.5 * np.log(2 * np.pi * np.e * kB_T / k) if k > 0 else 0

    if A == 0 or B == 0:
        C_curv = 0.0
    else:
        x1_samples = np.linspace(0, 2*np.pi/B, 1000)
        kappa_sq = np.array([curvature(x, A, B)**2 for x in x1_samples])
        mean_kappa_sq = np.mean(kappa_sq)
        C_curv = 0.5 * mean_kappa_sq * sigma_perp_sq

    C_phi = max(0, C_dim) + C_curv
    return C_phi, C_dim, C_curv


def simulate_manifold_trial(A=0.0, B=0.0, T=5.0, dt=1e-3, D=0.1, k=10.0):
    """Single trial of 2D Brownian motion with confinement (Stratonovich)."""
    n_steps = int(T / dt)
    x = np.zeros(2)
    W = 0.0

    for _ in range(n_steps):
        dW = np.sqrt(2 * D * dt) * rng.normal(size=2)
        F_start = F_ctrl(x, A, B, k)
        x_new = x + F_start * dt + dW
        x_mid = 0.5 * (x + x_new)
        F_mid = F_ctrl(x_mid, A, B, k)
        dx = x_new - x
        W += np.dot(F_mid, dx)
        x = x_new

    return W / T


def sim1_curvature(save_fig=True):
    """Sim 1: Geometric cost of curvature."""
    print("=" * 60)
    print("SIMULATION 1: Brownian Motion on Curved Manifolds")
    print("=" * 60)

    T, dt, D, k, n_trials = 5.0, 1e-3, 0.1, 10.0, 10
    configs = [
        ("Linear", 0.0, 0.0),
        ("Mild curvature", 0.5, 1.0),
        ("High curvature", 1.0, 2.0),
    ]

    means, stds, labels, C_phis = [], [], [], []

    for label, A, B in configs:
        powers = [simulate_manifold_trial(A=A, B=B, T=T, dt=dt, D=D, k=k)
                  for _ in range(n_trials)]
        powers = np.array(powers)
        C_phi, _, _ = theoretical_C_phi(A, B, k, D)
        means.append(powers.mean())
        stds.append(powers.std())
        labels.append(label)
        C_phis.append(C_phi)
        print(f"{label}: mean power = {powers.mean():.4f} +/- {powers.std():.4f}")

    x = np.arange(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar(x, means, yerr=stds, capsize=5, color=['#bdc3c7', '#7f8c8d', '#2c3e50'])
    plt.xticks(x, labels, rotation=20)
    plt.ylabel(r"Mean control power $\langle W_{\mathrm{ctrl}}/T \rangle$")
    plt.title("Geometric Cost of Curvature")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_fig:
        plt.savefig('../figures/fig1_curvature.png', dpi=150)
        plt.savefig('../figures/fig1_curvature.pdf')
        print("Saved fig1_curvature.*")
    plt.show()
    return means, stds


# =============================================================================
# SIMULATION 2: Kuramoto Oscillators
# =============================================================================

def simulate_kuramoto_trial(N=64, K=0.5, T=50.0, dt=0.01, D=0.01, alpha=1.0):
    """Single trial of Kuramoto oscillators with control."""
    n_steps = int(T / dt)
    omegas = rng.normal(loc=0.0, scale=0.5, size=N)
    theta = 2 * np.pi * rng.random(N)

    W_ctrl = 0.0
    rs = []

    for step in range(n_steps):
        R = np.mean(np.exp(1j * theta))
        r = np.abs(R)
        psi = np.angle(R)
        rs.append(r)

        y = psi
        y_star = 0.0
        err = y - y_star
        u = -alpha * err * np.sin(psi - theta)

        diff = theta[:, None] - theta[None, :]
        coupling = (K / N) * np.sum(np.sin(-diff), axis=1)
        noise = np.sqrt(2 * D * dt) * rng.normal(size=N)
        dtheta = (omegas + coupling + u) * dt + noise
        theta_new = (theta + dtheta) % (2 * np.pi)

        W_ctrl += np.sum(u * dtheta)
        theta = theta_new

    mean_r = np.mean(rs[int(0.5 * n_steps):])
    return mean_r, W_ctrl / T


def sim2_kuramoto(save_fig=True):
    """Sim 2: Coherence reduces dimensional work."""
    print("=" * 60)
    print("SIMULATION 2: Kuramoto Oscillators")
    print("=" * 60)

    K_values = np.linspace(0.0, 2.0, 10)
    n_trials = 8
    mean_rs, std_rs, mean_P, std_P = [], [], [], []

    for K in K_values:
        results = [simulate_kuramoto_trial(K=K) for _ in range(n_trials)]
        rs = np.array([r[0] for r in results])
        Ps = np.array([r[1] for r in results])
        mean_rs.append(rs.mean())
        std_rs.append(rs.std())
        mean_P.append(Ps.mean())
        std_P.append(Ps.std())
        print(f"K={K:.2f}: r={rs.mean():.3f}, P={Ps.mean():.3e}")

    mean_rs, std_rs = np.array(mean_rs), np.array(std_rs)
    mean_P, std_P = np.array(mean_P), np.array(std_P)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    ax1.plot(K_values, mean_rs, marker="o", color='#27ae60')
    ax1.fill_between(K_values, mean_rs - std_rs, mean_rs + std_rs, alpha=0.2, color='#27ae60')
    ax2.plot(K_values, mean_P, marker="s", color='#2980b9')
    ax2.fill_between(K_values, mean_P - std_P, mean_P + std_P, alpha=0.2, color='#2980b9')

    ax1.set_xlabel("Coupling $K$")
    ax1.set_ylabel(r"Mean order parameter $\langle r \rangle$", color='#27ae60')
    ax2.set_ylabel(r"Mean control power $\langle W_{\mathrm{ctrl}}/T \rangle$", color='#2980b9')
    ax1.tick_params(axis='y', labelcolor='#27ae60')
    ax2.tick_params(axis='y', labelcolor='#2980b9')

    plt.title("Coherence Reduces Dimensional Work")
    plt.tight_layout()

    if save_fig:
        plt.savefig('../figures/fig2_kuramoto.png', dpi=150)
        plt.savefig('../figures/fig2_kuramoto.pdf')
        print("Saved fig2_kuramoto.*")
    plt.show()
    return K_values, mean_rs, mean_P


# =============================================================================
# SIMULATION 3: Autoencoder Bottleneck
# =============================================================================

def make_autoencoder_data(n_samples=5000, in_dim=10, intrinsic_dim=2):
    """Generate synthetic data with known intrinsic dimension."""
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


def sim3_autoencoder(save_fig=True):
    """Sim 3: Cost of over-compression in autoencoders."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available - skipping autoencoder simulation")
        return None

    print("=" * 60)
    print("SIMULATION 3: Autoencoder Bottleneck")
    print("=" * 60)

    class Autoencoder(nn.Module):
        def __init__(self, in_dim=10, bottleneck=3, hidden=32):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck))
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))

        def forward(self, x):
            return self.decoder(self.encoder(x))

    def train_autoencoder(X, bottleneck, n_epochs=50, batch_size=128):
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = Autoencoder(in_dim=X.shape[1], bottleneck=bottleneck)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        cumulative_grad_norm = 0.0
        for _ in range(n_epochs):
            for batch, in loader:
                opt.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                total_norm = sum(p.grad.data.norm(2).item()**2
                                 for p in model.parameters() if p.grad is not None)
                cumulative_grad_norm += np.sqrt(total_norm)
                opt.step()

        with torch.no_grad():
            recon_err = loss_fn(model(torch.from_numpy(X)), torch.from_numpy(X)).item()
        return recon_err, cumulative_grad_norm

    X = make_autoencoder_data()
    bottlenecks = [1, 2, 3, 4, 5, 6, 8, 10]
    recon_means, recon_stds, effort_means, effort_stds = [], [], [], []
    n_seeds = 5

    for db in bottlenecks:
        errs, effs = [], []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            err, effort = train_autoencoder(X, bottleneck=db)
            errs.append(err)
            effs.append(effort)
        errs, effs = np.array(errs), np.array(effs)
        recon_means.append(errs.mean())
        recon_stds.append(errs.std())
        effort_means.append(effs.mean())
        effort_stds.append(effs.std())
        print(f"d_b={db}: recon={errs.mean():.4f}, effort={effs.mean():.1f}")

    bottlenecks = np.array(bottlenecks)
    recon_means, effort_means = np.array(recon_means), np.array(effort_means)
    recon_stds, effort_stds = np.array(recon_stds), np.array(effort_stds)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    ax1.errorbar(bottlenecks, recon_means, yerr=recon_stds, marker="o", capsize=4, color='#2c3e50')
    ax2.errorbar(bottlenecks, effort_means, yerr=effort_stds, marker="s", capsize=4, color='#c0392b')
    ax1.axvline(x=2, color='gray', linestyle=':', alpha=0.7, label='Intrinsic dim')

    ax1.set_xlabel("Bottleneck dimension $d_b$")
    ax1.set_ylabel("Reconstruction MSE", color='#2c3e50')
    ax2.set_ylabel("Cumulative gradient norm (SGD work)", color='#c0392b')
    ax1.tick_params(axis='y', labelcolor='#2c3e50')
    ax2.tick_params(axis='y', labelcolor='#c0392b')

    plt.title("The Cost of Over-Compression")
    plt.tight_layout()

    if save_fig:
        plt.savefig('../figures/fig3_autoencoder.png', dpi=150)
        plt.savefig('../figures/fig3_autoencoder.pdf')
        print("Saved fig3_autoencoder.*")
    plt.show()
    return bottlenecks, recon_means, effort_means


# =============================================================================
# SIMULATION 4: Ephaptic Neural Sheet
# =============================================================================

def simulate_sheet_trial(N=64, T=50.0, dt=0.01, ge=0.0, mode="coherent",
                         alpha=0.5, tau_m=10.0):
    """Simulate neural sheet with optional ephaptic coupling."""
    n_steps = int(T / dt)
    V = rng.normal(scale=0.1, size=N)

    W_syn = np.zeros((N, N))
    for i in range(N):
        for j in [(i - 1) % N, (i + 1) % N]:
            W_syn[i, j] = 0.5

    def sigma(x):
        return 1.0 / (1.0 + np.exp(-x))

    def target_y(t):
        return 0.5 * np.sin(0.1 * t)

    V_hist = np.zeros((n_steps, N))
    W_ctrl = 0.0

    for step in range(n_steps):
        t = step * dt

        if mode == "coherent":
            E = V.mean()
        elif mode == "random":
            E = np.abs(V.mean()) * rng.choice([-1.0, 1.0])
        else:
            E = 0.0

        y = V.mean()
        y_star = target_y(t)
        I_ctrl = -alpha * (y - y_star)

        I_syn = W_syn @ sigma(V)
        I_eph = ge * E
        noise = np.sqrt(2 * dt) * rng.normal(size=N) * 0.05

        dV = dt / tau_m * (-V + I_syn + I_eph + I_ctrl) + noise
        W_ctrl += np.sum(I_ctrl * dV)
        V = V + dV
        V_hist[step] = V

    # Participation ratio
    V_centered = V_hist - V_hist.mean(0, keepdims=True)
    C = np.cov(V_centered.T)
    eigvals = np.maximum(np.linalg.eigvalsh(C), 0.0)
    pr = (eigvals.sum() ** 2) / np.sum(eigvals ** 2) if eigvals.sum() > 0 else 0.0

    return W_ctrl / T, pr


def sim4_ephaptic(save_fig=True):
    """Sim 4: Ephaptic coherence vs dimensional work."""
    print("=" * 60)
    print("SIMULATION 4: Ephaptic Neural Sheet")
    print("=" * 60)

    configs = [
        ("No ephaptic", 0.0, "none"),
        ("Random field", 0.5, "random"),
        ("Coherent ephaptic", 0.5, "coherent"),
    ]
    n_trials = 10
    results = []

    for label, ge, mode in configs:
        trials = [simulate_sheet_trial(ge=ge, mode=mode) for _ in range(n_trials)]
        powers = np.array([t[0] for t in trials])
        prs = np.array([t[1] for t in trials])
        results.append((label, powers.mean(), powers.std(), prs.mean(), prs.std()))
        print(f"{label}: P={powers.mean():.3e}, PR={prs.mean():.2f}")

    labels = [r[0] for r in results]
    P_mean = np.array([r[1] for r in results])
    P_std = np.array([r[2] for r in results])
    PR_mean = np.array([r[3] for r in results])
    PR_std = np.array([r[4] for r in results])

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    width = 0.35
    ax1.bar(x - width/2, P_mean, width, yerr=P_std, capsize=4, color='#c0392b', alpha=0.8)
    ax2.bar(x + width/2, PR_mean, width, yerr=PR_std, capsize=4, color='#2980b9', alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel(r"Mean control power", color='#c0392b')
    ax2.set_ylabel("Participation ratio", color='#2980b9')
    ax1.tick_params(axis='y', labelcolor='#c0392b')
    ax2.tick_params(axis='y', labelcolor='#2980b9')

    plt.title("Ephaptic Coherence vs Dimensional Work")
    fig.tight_layout()

    if save_fig:
        plt.savefig('../figures/fig4_ephaptic.png', dpi=150)
        plt.savefig('../figures/fig4_ephaptic.pdf')
        print("Saved fig4_ephaptic.*")
    plt.show()
    return results


# =============================================================================
# MAIN
# =============================================================================

def run_all():
    """Run all simulations."""
    print("=" * 60)
    print("DIMENSIONAL LANDAUER BOUND - COMPLETE SIMULATION SUITE")
    print("=" * 60)
    fig0_concept()
    sim1_curvature()
    sim2_kuramoto()
    sim3_autoencoder()
    sim4_ephaptic()
    print("\n" + "=" * 60)
    print("ALL SIMULATIONS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "concept":
            fig0_concept()
        elif cmd == "curvature":
            sim1_curvature()
        elif cmd == "kuramoto":
            sim2_kuramoto()
        elif cmd == "autoencoder":
            sim3_autoencoder()
        elif cmd == "ephaptic":
            sim4_ephaptic()
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python simulations.py [concept|curvature|kuramoto|autoencoder|ephaptic]")
    else:
        run_all()
