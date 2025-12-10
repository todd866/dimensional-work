"""
Sim 1: 2D Brownian Motion on Curved Manifolds (Stratonovich)

Demonstrates that maintaining a trajectory on a high-curvature manifold
requires more control work than a flat one, even if the logical path is identical.

Uses Stratonovich midpoint convention for thermodynamically correct work calculation.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


def f_curve(x1, A, B):
    """Manifold definition: x2 = A * sin(B * x1)"""
    return A * np.sin(B * x1)


def F_ctrl(x, A, B, k):
    """
    Control force from U = k/2 (x2 - f(x1))^2.
    Acts to confine the particle near the 1D manifold x2 = f(x1).
    """
    x1, x2 = x
    f_val = f_curve(x1, A, B)
    df_dx1 = A * B * np.cos(B * x1) if (A != 0 and B != 0) else 0.0
    g_val = x2 - f_val
    grad_g = np.array([-df_dx1, 1.0])
    return -k * g_val * grad_g


def simulate_manifold_trial(A=0.0, B=0.0, T=5.0, dt=1e-3, D=0.1, k=10.0):
    """
    Single trial of 2D Brownian motion with confinement to manifold.
    Work is computed with Stratonovich midpoint convention.
    """
    n_steps = int(T / dt)
    x = np.zeros(2)
    W = 0.0

    for _ in range(n_steps):
        dW = np.sqrt(2 * D * dt) * rng.normal(size=2)
        F_start = F_ctrl(x, A, B, k)
        # Tentative step (Euler-Maruyama)
        x_new = x + F_start * dt + dW
        # Stratonovich midpoint force
        x_mid = 0.5 * (x + x_new)
        F_mid = F_ctrl(x_mid, A, B, k)
        dx = x_new - x
        W += np.dot(F_mid, dx)
        x = x_new

    return W / T


def simulate_condition(A, B, n_trials=10, **kwargs):
    """Run multiple trials and return mean/std of power."""
    powers = []
    for _ in range(n_trials):
        P = simulate_manifold_trial(A=A, B=B, **kwargs)
        powers.append(P)
    powers = np.array(powers)
    return powers.mean(), powers.std()


def main(save_fig=True):
    T = 5.0
    dt = 1e-3
    D = 0.1
    k = 10.0
    n_trials = 10

    configs = [
        ("Linear", 0.0, 0.0),
        ("Mild curvature", 0.5, 1.0),
        ("High curvature", 1.0, 2.0),
    ]

    means = []
    stds = []
    labels = []

    for label, A, B in configs:
        m, s = simulate_condition(A, B, T=T, dt=dt, D=D, k=k, n_trials=n_trials)
        means.append(m)
        stds.append(s)
        labels.append(label)
        print(f"{label}: mean power = {m:.4f} +/- {s:.4f}")

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
    plt.show()

    return means, stds


if __name__ == "__main__":
    main()
