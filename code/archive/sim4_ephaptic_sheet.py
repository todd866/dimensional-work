"""
Sim 4: Ephaptic Neural Sheet (Energy-Matched Control)

Demonstrates that coherent ephaptic coupling reduces both effective
dimensionality (Participation Ratio) and control work, whereas
energy-matched random fields do not.

This proves that the benefit comes from geometric alignment (dimensionality
reduction), not just energy injection.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)


def simulate_sheet_trial(N=64, T=50.0, dt=0.01, ge=0.0, mode="coherent",
                         alpha=0.5, tau_m=10.0):
    """
    Simulate neural sheet with optional ephaptic coupling.

    mode:
        - "none": no ephaptic coupling
        - "coherent": E(t) = mean(V) (population-aligned field)
        - "random": E(t) = |mean(V)| * random_sign
          (same magnitude, scrambled direction - energy matched)

    This ensures the magnitude of energy injection is matched between
    coherent and random conditions; only the geometric structure differs.
    """
    n_steps = int(T / dt)
    V = rng.normal(scale=0.1, size=N)

    # Local synaptic weights: ring with nearest-neighbour excitation
    W_syn = np.zeros((N, N))
    for i in range(N):
        for j in [(i - 1) % N, (i + 1) % N]:
            W_syn[i, j] = 0.5

    def sigma(x):
        return 1.0 / (1.0 + np.exp(-x))

    def target_y(t):
        # Simple low-dimensional target signal
        return 0.5 * np.sin(0.1 * t)

    V_hist = np.zeros((n_steps, N))
    W_ctrl = 0.0
    y_errs = []

    for step in range(n_steps):
        t = step * dt

        # Coherent vs random vs none:
        if mode == "coherent":
            # Population-aligned ephaptic field
            E = V.mean()
        elif mode == "random":
            # Energy-matched but incoherent: same magnitude as coherent field,
            # but with randomized sign (scrambled "phase")
            mag = np.abs(V.mean())
            sign = rng.choice([-1.0, 1.0])
            E = mag * sign
        else:  # "none"
            E = 0.0

        # Low-dimensional readout: mean voltage
        y = V.mean()
        y_star = target_y(t)

        # Global control to keep y near y_star
        I_ctrl = -alpha * (y - y_star)

        # Synaptic + ephaptic + control + noise
        I_syn = W_syn @ sigma(V)
        I_eph = ge * E
        noise = np.sqrt(2 * dt) * rng.normal(size=N) * 0.05

        dV = dt / tau_m * (-V + I_syn + I_eph + I_ctrl) + noise
        V_new = V + dV

        # Control work: sum I_ctrl * dV over neurons
        W_ctrl += np.sum(I_ctrl * dV)

        V = V_new
        V_hist[step] = V
        y_errs.append((y - y_star) ** 2)

    # Effective dimensionality via participation ratio of covariance
    V_centered = V_hist - V_hist.mean(0, keepdims=True)
    C = np.cov(V_centered.T)
    eigvals = np.linalg.eigvalsh(C)
    eigvals = np.maximum(eigvals, 0.0)
    if eigvals.sum() > 0:
        pr = (eigvals.sum() ** 2) / np.sum(eigvals ** 2)
    else:
        pr = 0.0

    # Use late-time error for tracking quality
    return W_ctrl / T, pr, np.mean(y_errs[int(0.5 * n_steps):])


def simulate_conditions(n_trials=10):
    """Compare no field, random field, and coherent ephaptic field."""
    configs = [
        ("No ephaptic", 0.0, "none"),
        ("Random field", 0.5, "random"),
        ("Coherent ephaptic", 0.5, "coherent"),
    ]

    results = []
    for label, ge, mode in configs:
        powers = []
        prs = []
        errs = []
        for _ in range(n_trials):
            P, pr, err = simulate_sheet_trial(ge=ge, mode=mode)
            powers.append(P)
            prs.append(pr)
            errs.append(err)
        powers = np.array(powers)
        prs = np.array(prs)
        errs = np.array(errs)
        results.append(
            (label,
             powers.mean(), powers.std(),
             prs.mean(), prs.std(),
             errs.mean(), errs.std())
        )
    return results


def main(save_fig=True):
    results = simulate_conditions(n_trials=10)
    labels = [r[0] for r in results]
    P_mean = np.array([r[1] for r in results])
    P_std = np.array([r[2] for r in results])
    PR_mean = np.array([r[3] for r in results])
    PR_std = np.array([r[4] for r in results])

    x = np.arange(len(labels))

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    width = 0.35
    color1 = '#c0392b'
    color2 = '#2980b9'

    bars1 = ax1.bar(x - width/2, P_mean, width, yerr=P_std, capsize=4,
                    label="Control power", color=color1, alpha=0.8)
    bars2 = ax2.bar(x + width/2, PR_mean, width, yerr=PR_std, capsize=4,
                    label="Participation ratio", color=color2, alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_ylabel(r"Mean control power $\langle W_{\mathrm{ctrl}}/T \rangle$",
                   color=color1)
    ax2.set_ylabel("Participation ratio (effective dim)", color=color2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("Ephaptic Coherence vs Dimensional Work")
    fig.tight_layout()

    if save_fig:
        plt.savefig('../figures/fig4_ephaptic.png', dpi=150)
        plt.savefig('../figures/fig4_ephaptic.pdf')
    plt.show()

    print("\nResults summary:")
    for r in results:
        print(f"  {r[0]}: P={r[1]:.3e}+/-{r[2]:.3e}, "
              f"PR={r[3]:.2f}+/-{r[4]:.2f}, y_err={r[5]:.4f}+/-{r[6]:.4f}")

    return results


if __name__ == "__main__":
    main()
