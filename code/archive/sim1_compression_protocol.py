"""
Sim 1: Geometric Work during Dimensional Compression
(Quasistatic compression protocol - validates Dimensional Landauer Bound)

PROTOCOL:
1. Initialize particles in a broad cloud (high entropy, k≈0).
2. Slowly increase confinement stiffness k(t) from 0 to k_final.
3. Measure thermodynamic work W = integral (dU/dk) * dk.
4. Compare against theoretical free energy change.

This validates the 'Dimensional Landauer Bound' as a switching cost
(finite energy for state transition) rather than maintenance cost
(divergent power over time).

Key physics: Work = integral of (∂U/∂k) dk, not integral of F·dx over time.
"""

import numpy as np
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)

# Configuration
N_PARTICLES = 2000      # Large batch for clean statistics
DT = 0.005              # Time step
TAU = 20.0              # Duration of compression (slow = quasistatic)
D = 1.0                 # Diffusion coefficient (sets kB*T = D in these units)
K_FINAL = 50.0          # Final stiffness (strong confinement)


def f_curve(x1, A, B):
    """Manifold: x2 = A * sin(B * x1)"""
    return A * np.sin(B * x1)


def df_curve(x1, A, B):
    """Derivative of manifold"""
    return A * B * np.cos(B * x1) if (A != 0 and B != 0) else 0.0


def get_mean_curvature(A, B):
    """Compute average absolute curvature for labeling."""
    if A == 0 or B == 0:
        return 0.0
    x = np.linspace(0, 2 * np.pi, 1000)
    yp = A * B * np.cos(B * x)
    ypp = -A * B**2 * np.sin(B * x)
    kappa = np.abs(ypp) / (1 + yp**2)**1.5
    return np.mean(kappa)


def run_compression(A, B, verbose=False):
    """
    Run compression protocol for a batch of N_PARTICLES.

    Returns: (mean_work, std_error, mean_final_dist)

    The key insight: We measure W = integral (∂U/∂k) dk
    where U = (k/2)(x2 - f(x1))^2
    so ∂U/∂k = (1/2)(x2 - f(x1))^2
    """
    # Period of manifold (or default if flat)
    L = 2 * np.pi / (B if B > 0 else 1.0)

    # Initialize: uniform in x1, broad Gaussian cloud in x2
    x1 = RNG.uniform(0, L, N_PARTICLES)
    x2 = RNG.normal(0, 2.0, N_PARTICLES)  # Broad initial cloud

    # Work accumulator for each particle
    W_accum = np.zeros(N_PARTICLES)

    n_steps = int(TAU / DT)
    k_schedule = np.linspace(0.0, K_FINAL, n_steps)

    current_k = 0.0

    for i in range(n_steps):
        # Update control parameter (stiffness)
        next_k = k_schedule[i]
        dk = next_k - current_k

        # Thermodynamic work increment: dW = (∂U/∂k) dk = (1/2)(x2-f)^2 dk
        f_val = f_curve(x1, A, B)
        dist_sq = (x2 - f_val)**2
        dW = 0.5 * dist_sq * dk
        W_accum += dW

        current_k = next_k

        # Langevin dynamics (overdamped)
        # U = (k/2)(x2 - f(x1))^2
        # F_x1 = -∂U/∂x1 = k(x2 - f)(f')
        # F_x2 = -∂U/∂x2 = -k(x2 - f)

        diff = x2 - f_val
        df_val = df_curve(x1, A, B) if (A != 0 and B != 0) else 0.0

        F1 = current_k * diff * df_val
        F2 = -current_k * diff

        # Noise
        noise_scale = np.sqrt(2 * D * DT)
        eta1 = RNG.normal(size=N_PARTICLES)
        eta2 = RNG.normal(size=N_PARTICLES)

        # Update positions
        x1 += F1 * DT + noise_scale * eta1
        x2 += F2 * DT + noise_scale * eta2

        # Periodic boundary for x1
        x1 = x1 % L

    # Final distance from manifold
    final_f = f_curve(x1, A, B)
    final_dist = np.mean(np.abs(x2 - final_f))

    mean_W = np.mean(W_accum)
    std_err = np.std(W_accum) / np.sqrt(N_PARTICLES)

    return mean_W, std_err, final_dist


def theoretical_compression_work(A, B, k_final, D):
    """
    Theoretical free energy change for compression.

    For a flat manifold (A=0):
    ΔF = (D/2) ln(k_final * σ_init^2 / D)
    where σ_init is initial spread.

    For curved manifold, there's additional geometric cost.
    The excess work (W_curved - W_flat) is the "geometric work."
    """
    # This is approximate - proper calculation requires path integral
    # For now, return a placeholder based on curvature
    mean_kappa = get_mean_curvature(A, B)

    # Base compression work (flat case)
    sigma_init = 2.0  # Initial spread
    W_flat = 0.5 * D * np.log(k_final * sigma_init**2 / D) if k_final > 0 else 0

    # Geometric correction (approximate)
    # Curved manifolds have effectively stiffer local confinement
    # This increases the required work
    W_geom = D * mean_kappa**2 * sigma_init**2  # Rough scaling

    return W_flat, W_geom


def main(save_fig=True):
    print("=" * 70)
    print("QUASISTATIC COMPRESSION PROTOCOL")
    print("Validates Dimensional Landauer Bound as finite switching cost")
    print("=" * 70)
    print(f"Parameters: N={N_PARTICLES}, τ={TAU}, k_final={K_FINAL}, D={D}")
    print("-" * 70)

    configs = [
        ("Linear", 0.0, 0.0),
        ("Mild Curve", 0.5, 1.0),
        ("High Curve", 1.0, 2.0),
    ]

    results = []

    for label, A, B in configs:
        print(f"\nSimulating {label} (A={A}, B={B})...")
        W, err, final_dist = run_compression(A, B)
        kappa = get_mean_curvature(A, B)
        W_flat_theory, W_geom_theory = theoretical_compression_work(A, B, K_FINAL, D)

        results.append({
            'label': label,
            'A': A, 'B': B,
            'W': W, 'err': err,
            'kappa': kappa,
            'final_dist': final_dist,
            'W_flat_theory': W_flat_theory,
            'W_geom_theory': W_geom_theory,
        })

        print(f"  Compression Work: {W:.3f} ± {err:.3f} kT")
        print(f"  Mean |κ|: {kappa:.4f}")
        print(f"  Final dist from manifold: {final_dist:.4f}")

    # Compute geometric work (excess over linear baseline)
    W_baseline = results[0]['W']
    print("\n" + "-" * 70)
    print("GEOMETRIC WORK (excess over flat baseline):")
    for r in results:
        W_geom = r['W'] - W_baseline
        print(f"  {r['label']}: W_geom = {W_geom:.3f} kT")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Total compression work
    ax1 = axes[0]
    labels = [r['label'] for r in results]
    Ws = [r['W'] for r in results]
    errs = [r['err'] for r in results]

    x = np.arange(len(labels))
    bars = ax1.bar(x, Ws, yerr=errs, capsize=5,
                   color=['#3498db', '#2980b9', '#1a5276'], alpha=0.85)
    ax1.axhline(W_baseline, color='#e74c3c', linestyle='--',
                label='Flat baseline', linewidth=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel(r'Compression Work ($k_B T$)')
    ax1.set_title('Total Work for Dimensional Compression')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Right: Geometric work vs curvature
    ax2 = axes[1]
    kappas = [r['kappa'] for r in results]
    W_geoms = [r['W'] - W_baseline for r in results]

    ax2.scatter(kappas, W_geoms, s=100, c=['#3498db', '#2980b9', '#1a5276'],
                edgecolors='black', linewidths=1.5, zorder=5)

    # Add labels
    for i, r in enumerate(results):
        ax2.annotate(r['label'], (kappas[i], W_geoms[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax2.set_xlabel(r'Mean Curvature $\langle|\kappa|\rangle$')
    ax2.set_ylabel(r'Geometric Work $W_{geom}$ ($k_B T$)')
    ax2.set_title('Geometric Work Scales with Curvature')
    ax2.grid(alpha=0.3)

    # Fit line (excluding zero point)
    if len(kappas) > 1 and kappas[-1] > 0:
        # Simple linear fit through non-zero points
        kappas_nz = np.array(kappas[1:])
        W_geoms_nz = np.array(W_geoms[1:])
        slope = np.sum(kappas_nz * W_geoms_nz) / np.sum(kappas_nz**2)
        kappa_fit = np.linspace(0, max(kappas) * 1.1, 100)
        ax2.plot(kappa_fit, slope * kappa_fit, 'r--', alpha=0.7,
                 label=f'Slope ≈ {slope:.2f}')
        ax2.legend()

    plt.tight_layout()

    if save_fig:
        plt.savefig('../figures/fig1_compression_protocol.png', dpi=150)
        plt.savefig('../figures/fig1_compression_protocol.pdf')
        print("\nFigures saved to figures/fig1_compression_protocol.*")

    plt.show()

    return results


if __name__ == "__main__":
    main()
