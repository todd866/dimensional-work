"""
Sim 1: 2D Brownian Motion on Curved Manifolds (Stratonovich) - WITH BOUND VALIDATION

Demonstrates that maintaining a trajectory on a high-curvature manifold
requires more control work than a flat one, even if the logical path is identical.

This version computes theoretical C_Phi and validates the Dimensional Landauer Bound.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


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
    """
    Control force from U = k/2 (x2 - f(x1))^2.
    Acts to confine the particle near the 1D manifold x2 = f(x1).
    """
    x1, x2 = x
    f_val = f_curve(x1, A, B)
    df_dx1 = df_curve(x1, A, B)
    g_val = x2 - f_val
    grad_g = np.array([-df_dx1, 1.0])
    return -k * g_val * grad_g


def theoretical_C_phi(A, B, k, D, T=1.0):
    """
    Theoretical contraction cost for confining to manifold x2 = A*sin(B*x1).

    C_Phi has two contributions:
    1. Dimensional reduction: suppressing 1 DOF costs ~ (1/2) ln(k*sigma^2 / k_B T)
    2. Curvature: curved manifolds cost more due to geometry

    For a harmonic confinement with stiffness k, the perpendicular variance is:
    sigma_perp^2 = k_B T / k = D / k (in our units where k_B T = D / mobility)

    The curvature contribution integrates <kappa^2> over the manifold.
    """
    # In our units, k_B T ~ D (diffusion coefficient sets the temperature scale)
    kB_T = D  # Effective temperature

    # Perpendicular confinement contribution
    # Variance perpendicular to manifold: sigma_perp^2 = k_B T / k
    sigma_perp_sq = kB_T / k

    # Base cost: confining one dimension
    # This is the KL divergence between 2D Gaussian and 1D Gaussian lifted to 2D
    C_dim = 0.5 * np.log(2 * np.pi * np.e * kB_T / k) if k > 0 else 0

    # Curvature contribution: <kappa^2> * sigma_perp^2
    # For f(x) = A*sin(Bx), kappa = -AB^2 sin(Bx) / (1 + A^2 B^2 cos^2(Bx))^(3/2)
    # Average over uniform x1 distribution
    if A == 0 or B == 0:
        C_curv = 0.0
    else:
        # Numerical integration of <kappa^2>
        x1_samples = np.linspace(0, 2*np.pi/B, 1000)
        kappa_sq = np.array([curvature(x, A, B)**2 for x in x1_samples])
        mean_kappa_sq = np.mean(kappa_sq)
        # Curvature correction scales as kappa^2 * sigma_perp^2
        C_curv = 0.5 * mean_kappa_sq * sigma_perp_sq

    # Total contraction cost (in nats, not bits)
    C_phi = max(0, C_dim) + C_curv

    return C_phi, C_dim, C_curv


def simulate_manifold_trial(A=0.0, B=0.0, T=5.0, dt=1e-3, D=0.1, k=10.0):
    """
    Single trial of 2D Brownian motion with confinement to manifold.
    Work is computed with Stratonovich midpoint convention.

    Returns: W/T (power), mean distance from manifold, mean |kappa|
    """
    n_steps = int(T / dt)
    x = np.zeros(2)
    W = 0.0

    distances = []
    kappas = []

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

        # Track distance from manifold and curvature
        dist = abs(x_new[1] - f_curve(x_new[0], A, B))
        distances.append(dist)
        kappas.append(abs(curvature(x_new[0], A, B)))

        x = x_new

    return W / T, np.mean(distances), np.mean(kappas)


def simulate_condition(A, B, n_trials=10, **kwargs):
    """Run multiple trials and return mean/std of power."""
    powers = []
    dists = []
    kappas = []
    for _ in range(n_trials):
        P, d, kap = simulate_manifold_trial(A=A, B=B, **kwargs)
        powers.append(P)
        dists.append(d)
        kappas.append(kap)
    powers = np.array(powers)
    return powers.mean(), powers.std(), np.mean(dists), np.mean(kappas)


def main(save_fig=True):
    T = 5.0
    dt = 1e-3
    D = 0.1
    k = 10.0
    n_trials = 20  # More trials for better statistics

    configs = [
        ("Linear", 0.0, 0.0),
        ("Mild curvature", 0.5, 1.0),
        ("High curvature", 1.0, 2.0),
    ]

    means = []
    stds = []
    labels = []
    C_phis = []

    print("=" * 70)
    print("DIMENSIONAL LANDAUER BOUND VALIDATION")
    print("=" * 70)
    print(f"Parameters: D={D}, k={k}, T={T}, dt={dt}")
    print("-" * 70)

    for label, A, B in configs:
        m, s, dist, kap = simulate_condition(A, B, T=T, dt=dt, D=D, k=k, n_trials=n_trials)
        C_phi, C_dim, C_curv = theoretical_C_phi(A, B, k, D)

        means.append(m)
        stds.append(s)
        labels.append(label)
        C_phis.append(C_phi)

        # The bound is W >= k_B T * C_phi
        # In our units k_B T ~ D, so bound is D * C_phi
        W_bound = D * C_phi

        print(f"\n{label} (A={A}, B={B}):")
        print(f"  Measured power:     {m:.4f} +/- {s:.4f}")
        print(f"  Theoretical C_phi:  {C_phi:.4f} (dim: {C_dim:.4f}, curv: {C_curv:.4f})")
        print(f"  Theoretical bound:  W >= {W_bound:.4f}")
        print(f"  Mean |kappa|:       {kap:.4f}")
        print(f"  Mean dist from M:   {dist:.4f}")

        if m >= W_bound - 2*s:
            print(f"  STATUS: BOUND SATISFIED (W = {m/W_bound:.2f}x bound)" if W_bound > 0 else "  STATUS: OK (flat manifold)")
        else:
            print(f"  STATUS: WARNING - measured below bound!")

    print("\n" + "=" * 70)

    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Left plot: measured vs theoretical
    ax1.bar(x - 0.2, means, 0.4, yerr=stds, capsize=5,
            color=['#3498db', '#3498db', '#3498db'], alpha=0.8, label='Measured W/T')
    ax1.bar(x + 0.2, [D * c for c in C_phis], 0.4,
            color=['#e74c3c', '#e74c3c', '#e74c3c'], alpha=0.8, label='Theoretical bound')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20)
    ax1.set_ylabel(r"Power $\langle W/T \rangle$")
    ax1.set_title("Measured Work vs Dimensional Landauer Bound")
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Right plot: original style
    ax2.bar(x, means, yerr=stds, capsize=5, color=['#bdc3c7', '#7f8c8d', '#2c3e50'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20)
    ax2.set_ylabel(r"Mean control power $\langle W_{\mathrm{ctrl}}/T \rangle$")
    ax2.set_title("Geometric Cost of Curvature")
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    if save_fig:
        plt.savefig('../figures/fig1_curvature_validated.png', dpi=150)
        plt.savefig('../figures/fig1_curvature_validated.pdf')
    plt.show()

    return means, stds, C_phis


if __name__ == "__main__":
    main()
