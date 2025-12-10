#!/usr/bin/env python3
"""
Simulation: Dimensional Work for Surface Confinement

Demonstrates that relaxation toward a minimal surface (flat, A→0)
corresponds to minimization of dimensional work.

A Brownian particle in 3D is confined to a surface z = A*sin(x)*sin(y).
As the amplitude A decreases (less curvature), the control work decreases.
"""

import numpy as np
import matplotlib.pyplot as plt

# Style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def surface_z(x, y, A):
    """Surface height function z = A * sin(x) * sin(y)"""
    return A * np.sin(x) * np.sin(y)


def surface_gradient(x, y, A):
    """Gradient of surface: (dz/dx, dz/dy)"""
    dz_dx = A * np.cos(x) * np.sin(y)
    dz_dy = A * np.sin(x) * np.cos(y)
    return dz_dx, dz_dy


def mean_curvature(x, y, A):
    """
    Mean curvature H for z = A*sin(x)*sin(y).
    H = -div(n) / 2 where n is unit normal.
    For small A: H ≈ -A*(sin(x)*sin(y))
    """
    # Second derivatives
    d2z_dx2 = -A * np.sin(x) * np.sin(y)
    d2z_dy2 = -A * np.sin(x) * np.sin(y)
    dz_dx, dz_dy = surface_gradient(x, y, A)

    # Mean curvature formula for graph z=f(x,y)
    numer = (1 + dz_dy**2) * d2z_dx2 - 2*dz_dx*dz_dy*0 + (1 + dz_dx**2) * d2z_dy2
    denom = 2 * (1 + dz_dx**2 + dz_dy**2)**1.5
    H = numer / denom
    return H


def run_simulation(A, n_steps=10000, dt=1e-3, D=0.1, k=50.0, seed=None):
    """
    Run Langevin simulation of particle confined to surface z = A*sin(x)*sin(y).

    We measure the work required to maintain confinement, which is the
    integral of |F_conf|^2 * dt / gamma (power dissipated by control).

    Returns:
        work: Total control effort (power proxy)
        mean_H: Mean absolute curvature sampled
    """
    rng = np.random.default_rng(seed)

    # Initialize near origin on surface
    x, y = rng.uniform(-0.5, 0.5, 2)
    z = surface_z(x, y, A)

    total_effort = 0.0
    total_deviation_sq = 0.0
    curvatures = []

    sqrt_2D_dt = np.sqrt(2 * D * dt)

    for _ in range(n_steps):
        # Current position
        r_old = np.array([x, y, z])

        # Surface value and gradient at current position
        z_surf = surface_z(x, y, A)
        dz_dx, dz_dy = surface_gradient(x, y, A)

        # Confinement force: F = -k * (z - z_surf) * (-dz/dx, -dz/dy, 1)
        deviation = z - z_surf
        F_conf = -k * deviation * np.array([-dz_dx, -dz_dy, 1.0])

        # Control effort: |F|^2 * dt (power dissipation proxy)
        effort = np.dot(F_conf, F_conf) * dt
        total_effort += effort
        total_deviation_sq += deviation**2

        # Euler-Maruyama step
        noise = rng.standard_normal(3) * sqrt_2D_dt
        r_new = r_old + F_conf * dt + noise
        x, y, z = r_new

        # Track curvature
        H = mean_curvature(x, y, A)
        curvatures.append(np.abs(H))

    T = n_steps * dt
    effort_rate = total_effort / T
    mean_deviation_sq = total_deviation_sq / n_steps
    mean_H = np.mean(curvatures)

    return effort_rate, mean_H, mean_deviation_sq


def main():
    print("Surface Confinement: Dimensional Work vs Amplitude")
    print("=" * 55)

    # Amplitude sweep
    amplitudes = np.linspace(0.05, 1.5, 15)
    n_trials = 5

    work_means = []
    work_stds = []
    curvature_means = []

    for A in amplitudes:
        works = []
        curvs = []
        for trial in range(n_trials):
            w, h, _ = run_simulation(A, n_steps=8000, seed=42 + trial)
            works.append(w)
            curvs.append(h)

        work_means.append(np.mean(works))
        work_stds.append(np.std(works))
        curvature_means.append(np.mean(curvs))
        print(f"A = {A:.2f}: Control effort = {np.mean(works):.3f} ± {np.std(works):.3f}, "
              f"Mean |H| = {np.mean(curvs):.4f}")

    work_means = np.array(work_means)
    work_stds = np.array(work_stds)
    curvature_means = np.array(curvature_means)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # Panel 1: Work vs Amplitude
    ax1.errorbar(amplitudes, work_means, yerr=work_stds,
                 fmt='o-', color='steelblue', capsize=3, markersize=5)
    ax1.set_xlabel('Surface amplitude $A$')
    ax1.set_ylabel('Control effort $\\langle |F|^2 \\rangle$')
    ax1.set_title('(a) Dimensional work increases with curvature')
    ax1.set_xlim(0, 1.6)

    # Panel 2: Work vs Mean Curvature
    ax2.scatter(curvature_means, work_means, c=amplitudes, cmap='viridis',
                s=60, edgecolors='k', linewidths=0.5)
    ax2.set_xlabel('Mean curvature $\\langle |H| \\rangle$')
    ax2.set_ylabel('Control effort $\\langle |F|^2 \\rangle$')
    ax2.set_title('(b) Effort scales with geometric curvature')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=amplitudes.min(),
                                                  vmax=amplitudes.max()))
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label('Amplitude $A$')

    plt.tight_layout()
    plt.savefig('../figures/fig1_surface_work.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('../figures/fig1_surface_work.png', bbox_inches='tight', dpi=150)
    print("\nSaved fig1_surface_work.pdf and fig1_surface_work.png")

    # Additional figure: 3D surface visualization
    fig2 = plt.figure(figsize=(8, 3))

    for i, A in enumerate([0.1, 0.5, 1.2]):
        ax = fig2.add_subplot(1, 3, i+1, projection='3d')
        u = np.linspace(-np.pi, np.pi, 40)
        v = np.linspace(-np.pi, np.pi, 40)
        U, V = np.meshgrid(u, v)
        Z = A * np.sin(U) * np.sin(V)

        ax.plot_surface(U, V, Z, cmap='coolwarm', alpha=0.8,
                        linewidth=0, antialiased=True)
        ax.set_title(f'$A = {A}$')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.set_zlim(-1.5, 1.5)
        ax.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig('../figures/fig2_surfaces.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('../figures/fig2_surfaces.png', bbox_inches='tight', dpi=150)
    print("Saved fig2_surfaces.pdf and fig2_surfaces.png")


if __name__ == "__main__":
    main()
