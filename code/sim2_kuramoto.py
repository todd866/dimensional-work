"""
Sim 2: Kuramoto Oscillators - Coherence vs Dimensional Work

Demonstrates that coherence (order parameter r) reduces the effective
dimensionality, lowering the work required to drive a low-dimensional readout.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)


def simulate_kuramoto_trial(N=64, K=0.5, T=50.0, dt=0.01, D=0.01, alpha=1.0):
    """
    Single trial of Kuramoto oscillators with control to track target phase.
    Returns mean order parameter and control power.
    """
    n_steps = int(T / dt)
    omegas = rng.normal(loc=0.0, scale=0.5, size=N)
    theta = 2 * np.pi * rng.random(N)

    def target_y(t):
        return 0.0

    W_ctrl = 0.0
    rs = []

    for step in range(n_steps):
        t = step * dt
        R = np.mean(np.exp(1j * theta))
        r = np.abs(R)
        psi = np.angle(R)
        rs.append(r)

        y = psi
        y_star = target_y(t)
        err = y - y_star

        # Control term ~ gradient of (y - y*)^2 w.r.t phases
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


def simulate_over_K(K_values, n_trials=5):
    """Run multiple trials for each K value."""
    mean_rs = []
    std_rs = []
    mean_P = []
    std_P = []

    for K in K_values:
        rs = []
        Ps = []
        for _ in range(n_trials):
            r, P = simulate_kuramoto_trial(K=K)
            rs.append(r)
            Ps.append(P)
        rs = np.array(rs)
        Ps = np.array(Ps)
        mean_rs.append(rs.mean())
        std_rs.append(rs.std())
        mean_P.append(Ps.mean())
        std_P.append(Ps.std())
        print(f"K={K:.2f}: r={rs.mean():.3f}+/-{rs.std():.3f}, P={Ps.mean():.3e}+/-{Ps.std():.3e}")

    return (np.array(mean_rs), np.array(std_rs),
            np.array(mean_P), np.array(std_P))


def main(save_fig=True):
    K_values = np.linspace(0.0, 2.0, 10)
    mean_rs, std_rs, mean_P, std_P = simulate_over_K(K_values, n_trials=8)

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    color1 = '#27ae60'
    color2 = '#2980b9'

    ax1.plot(K_values, mean_rs, marker="o", color=color1)
    ax1.fill_between(K_values, mean_rs - std_rs, mean_rs + std_rs,
                     alpha=0.2, color=color1)

    ax2.plot(K_values, mean_P, marker="s", color=color2)
    ax2.fill_between(K_values, mean_P - std_P, mean_P + std_P,
                     alpha=0.2, color=color2)

    ax1.set_xlabel("Coupling $K$")
    ax1.set_ylabel(r"Mean order parameter $\langle r \rangle$", color=color1)
    ax2.set_ylabel(r"Mean control power $\langle W_{\mathrm{ctrl}}/T \rangle$", color=color2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("Coherence Reduces Dimensional Work")
    plt.tight_layout()

    if save_fig:
        plt.savefig('../figures/fig2_kuramoto.png', dpi=150)
        plt.savefig('../figures/fig2_kuramoto.pdf')
    plt.show()

    return K_values, mean_rs, mean_P


if __name__ == "__main__":
    main()
