"""
Generate figures for the Dimensional Landauer Bound paper.

Creates publication-quality figures for CSF submission.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Import simulation functions
from simulations import (
    curvature_simulation,
    kuramoto_simulation,
    simple_autoencoder_work,
    generate_manifold_data,
    ephaptic_simulation
)

# Create figures directory
FIGURES_DIR = './figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
})

np.random.seed(42)


def figure1_curvature():
    """
    Figure 1: Geometric Cost of Curvature

    Shows maintenance power scaling with manifold curvature.
    Main panel: bar plot of power for linear/mild/high curvature
    Insets: trajectory samples on each manifold
    """
    print("Generating Figure 1: Curvature scaling...")

    fig = plt.figure(figsize=(7, 5))
    gs = GridSpec(2, 3, height_ratios=[2, 1], hspace=0.35, wspace=0.3)

    # Run simulations
    results = {}
    curvature_types = ['linear', 'mild', 'high']
    curvature_labels = ['Linear\n(flat)', 'Mild\n(sinusoidal)', 'High\n(sinusoidal)']

    for ctype in curvature_types:
        results[ctype] = curvature_simulation(
            n_steps=15000, dt=0.005, k_confine=200.0, T=1.0, curvature_type=ctype
        )

    # Main panel: power vs curvature
    ax_main = fig.add_subplot(gs[0, :])

    powers = [results[ct]['power'] for ct in curvature_types]
    curvatures = [results[ct]['curvature_mean_sq'] for ct in curvature_types]

    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax_main.bar(range(3), powers, color=colors, edgecolor='black', linewidth=0.8)

    ax_main.set_xticks(range(3))
    ax_main.set_xticklabels(curvature_labels)
    ax_main.set_ylabel(r'Maintenance Power $P_{\mathrm{maint}}$ (arb. units)')
    ax_main.set_title('(A) Thermodynamic cost scales with manifold curvature', fontweight='bold', loc='left')

    # Add curvature annotations
    for i, (bar, kappa) in enumerate(zip(bars, curvatures)):
        height = bar.get_height()
        ax_main.annotate(f'$\\langle\\kappa^2\\rangle$ = {kappa:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax_main.set_ylim(0, max(powers) * 1.2)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    # Insets: trajectory samples
    for i, ctype in enumerate(curvature_types):
        ax_inset = fig.add_subplot(gs[1, i])

        traj = results[ctype]['trajectory']
        # Plot manifold
        x_range = np.linspace(traj[:, 0].min() - 0.5, traj[:, 0].max() + 0.5, 200)

        if ctype == 'linear':
            y_manifold = 0 * x_range
        elif ctype == 'mild':
            y_manifold = 0.2 * np.sin(x_range)
        else:  # high
            y_manifold = 0.8 * np.sin(2 * x_range)

        ax_inset.plot(x_range, y_manifold, 'k-', linewidth=2, label='Manifold')

        # Sample trajectory points
        n_show = min(500, len(traj))
        idx = np.linspace(0, len(traj)-1, n_show, dtype=int)
        ax_inset.scatter(traj[idx, 0], traj[idx, 1], c=colors[i], s=3, alpha=0.3)

        ax_inset.set_xlabel('$x_1$')
        if i == 0:
            ax_inset.set_ylabel('$x_2$')
        ax_inset.set_title(f'{curvature_types[i].capitalize()}', fontsize=9)
        ax_inset.set_aspect('equal')
        ax_inset.set_xlim(-3, 3)
        ax_inset.set_ylim(-1.5, 1.5)
        ax_inset.spines['top'].set_visible(False)
        ax_inset.spines['right'].set_visible(False)

    plt.savefig(f'{FIGURES_DIR}/fig1_curvature.pdf')
    plt.savefig(f'{FIGURES_DIR}/fig1_curvature.png')
    plt.close()
    print("  -> Saved fig1_curvature.pdf")


def draw_kuramoto_circle(ax, phases, title, color='#2E86AB'):
    """Draw a Kuramoto oscillator circle schematic."""
    # Draw unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1, alpha=0.3)

    # Draw oscillator arrows
    for phi in phases:
        dx = 0.3 * np.cos(phi)
        dy = 0.3 * np.sin(phi)
        x = 0.7 * np.cos(phi)
        y = 0.7 * np.sin(phi)
        ax.arrow(x, y, dx, dy, head_width=0.08, head_length=0.05,
                fc=color, ec=color, linewidth=1)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=8, pad=2)


def figure2_kuramoto():
    """
    Figure 2: Kuramoto Oscillators and Effective Dimension Collapse

    Shows how synchronization reduces the thermodynamic cost of projection.
    Top row: Schematic of incoherent vs coherent oscillators
    Bottom left: Coherence vs coupling
    Bottom right: Control power vs coherence
    """
    print("Generating Figure 2: Kuramoto oscillators...")

    fig = plt.figure(figsize=(7, 5))
    gs = GridSpec(2, 4, height_ratios=[1, 2], hspace=0.4, wspace=0.4)

    # Top row: Kuramoto circle schematics
    np.random.seed(123)

    # Incoherent (scattered phases)
    ax_incoh = fig.add_subplot(gs[0, 0:2])
    phases_incoh = np.random.uniform(0, 2*np.pi, 12)
    draw_kuramoto_circle(ax_incoh, phases_incoh, 'Incoherent ($r \\approx 0$)', color='#A23B72')

    # Coherent (clustered phases)
    ax_coh = fig.add_subplot(gs[0, 2:4])
    mean_phase = np.pi/4
    phases_coh = mean_phase + np.random.normal(0, 0.2, 12)
    draw_kuramoto_circle(ax_coh, phases_coh, 'Coherent ($r \\approx 1$)', color='#2ECC71')

    # Run simulations
    K_values = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    results = []

    for K in K_values:
        res = kuramoto_simulation(N=64, K=K, T_total=80.0, dt=0.01, T_temp=0.05)
        n_ss = len(res['coherence']) // 2
        results.append({
            'K': K,
            'coherence': np.mean(res['coherence'][n_ss:]),
            'participation_ratio': np.mean(res['participation_ratio'][n_ss:]),
            'control_power': np.mean(res['control_power'][n_ss:])
        })

    coherence = np.array([r['coherence'] for r in results])
    power = np.array([r['control_power'] for r in results])

    # Bottom left panel: coherence vs K
    ax1 = fig.add_subplot(gs[1, 0:2])
    ax1.plot(K_values, coherence, 'o-', color='#2E86AB', markersize=6, label='Order parameter $r$')
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(2.0, color='gray', linestyle=':', alpha=0.5, linewidth=1)  # Critical K
    ax1.set_xlabel('Coupling strength $K$')
    ax1.set_ylabel('Coherence $r$', color='#2E86AB')
    ax1.tick_params(axis='y', labelcolor='#2E86AB')
    ax1.set_xlim(0, 9)
    ax1.set_ylim(0, 1.05)
    ax1.set_title('(A) Phase transition at $K_c$', fontweight='bold', loc='left')

    # Add critical coupling annotation
    ax1.annotate('$K_c$', xy=(2.0, 0.05), fontsize=10, ha='center')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Bottom right panel: power vs coherence
    ax2 = fig.add_subplot(gs[1, 2:4])
    ax2.plot(coherence, power, 'o-', color='#F18F01', markersize=6)
    ax2.set_xlabel('Coherence $r$')
    ax2.set_ylabel('Control power $P_{\\mathrm{maint}}$', color='#F18F01')
    ax2.tick_params(axis='y', labelcolor='#F18F01')
    ax2.set_xlim(0, 1.05)
    ax2.set_title('(B) Coherence reduces control cost', fontweight='bold', loc='left')

    # Fit exponential decay
    from scipy.optimize import curve_fit
    try:
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        popt, _ = curve_fit(exp_decay, coherence, power, p0=[2, 3, 0.1], maxfev=5000)
        r_fit = np.linspace(0, 1, 100)
        ax2.plot(r_fit, exp_decay(r_fit, *popt), '--', color='gray', alpha=0.7,
                label=f'$P \\propto e^{{-{popt[1]:.1f}r}}$')
        ax2.legend(loc='upper right', frameon=False)
    except:
        pass

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig2_kuramoto.pdf')
    plt.savefig(f'{FIGURES_DIR}/fig2_kuramoto.png')
    plt.close()
    print("  -> Saved fig2_kuramoto.pdf")


def figure3_autoencoder():
    """
    Figure 3: Thermodynamic Divergence at Information Bottleneck

    Shows that work required to compress data below intrinsic dimension diverges.
    """
    print("Generating Figure 3: Autoencoder bottleneck...")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    # Generate data on 2D manifold in R^10
    np.random.seed(42)
    data, _ = generate_manifold_data(n_samples=500, d_intrinsic=2, d_ambient=10)

    bottleneck_dims = [1, 2, 3, 4, 5, 6, 7, 8]
    results = []

    for d_b in bottleneck_dims:
        np.random.seed(42 + d_b)  # Consistent initialization
        res = simple_autoencoder_work(data, d_b, n_epochs=300, lr=0.03)
        results.append(res)

    final_loss = np.array([r['final_loss'] for r in results])
    total_work = np.array([r['total_work'] for r in results])

    # Compute work efficiency: work per unit loss reduction
    # For d_b < d_intrinsic, this should be high (work spent with little gain)
    initial_loss = np.var(data)  # Approximate initial loss
    work_efficiency = total_work / (initial_loss - final_loss + 0.01)

    # Left: final loss vs bottleneck
    ax1 = axes[0]
    ax1.semilogy(bottleneck_dims, final_loss + 1e-6, 'o-', color='#2E86AB', markersize=8)
    ax1.axvline(2, color='#F18F01', linestyle='--', linewidth=2, alpha=0.7, label='$d_{\\mathrm{intrinsic}}=2$')
    ax1.fill_betweenx([1e-6, 10], 0, 2, alpha=0.15, color='#F18F01')
    ax1.set_xlabel('Bottleneck dimension $d_b$')
    ax1.set_ylabel('Reconstruction loss (log scale)')
    ax1.set_xlim(0.5, 8.5)
    ax1.set_ylim(1e-4, 1)
    ax1.legend(loc='upper right', frameon=False)
    ax1.set_title('(A) Loss diverges below $d_{\\mathrm{intrinsic}}$', fontweight='bold', loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: work vs bottleneck (normalized)
    ax2 = axes[1]
    # Show work-per-reconstruction-achieved
    effective_work = total_work * (final_loss + 0.001)  # Penalty for poor reconstruction
    ax2.bar(bottleneck_dims, effective_work, color='#A23B72', edgecolor='black', linewidth=0.8)
    ax2.axvline(2, color='#F18F01', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Bottleneck dimension $d_b$')
    ax2.set_ylabel('Effective work $W \\times L$')
    ax2.set_xlim(0.5, 8.5)
    ax2.set_title('(B) Thermodynamic cost', fontweight='bold', loc='left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add annotation
    ax2.annotate('Bottleneck\nsingularity', xy=(1, effective_work[0] * 0.85),
                fontsize=8, ha='center', va='top')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig3_autoencoder.pdf')
    plt.savefig(f'{FIGURES_DIR}/fig3_autoencoder.png')
    plt.close()
    print("  -> Saved fig3_autoencoder.pdf")


def figure4_ephaptic():
    """
    Figure 4: Field-Mediated Alignment

    Compares random vs coherent field coupling in reducing dimensionality.
    """
    print("Generating Figure 4: Ephaptic coupling...")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    # Run simulations for different field types
    field_types = ['none', 'random', 'coherent']
    field_labels = ['No field', 'Random\nfield', 'Coherent\nfield']
    colors = ['#2E86AB', '#A23B72', '#2ECC71']

    results = []
    for ftype in field_types:
        res = ephaptic_simulation(N=100, field_type=ftype, field_strength=0.5,
                                  T_total=80.0, dt=0.01, T_temp=0.1)
        n_ss = len(res['participation_ratio']) // 2
        results.append({
            'field_type': ftype,
            'participation_ratio': np.mean(res['participation_ratio'][n_ss:]),
            'control_work': np.mean(res['control_work'][n_ss:]),
            'pr_time': res['participation_ratio'],
            'cw_time': res['control_work']
        })

    # Left: participation ratio comparison
    ax1 = axes[0]
    pr_values = [r['participation_ratio'] for r in results]
    bars1 = ax1.bar(range(3), pr_values, color=colors, edgecolor='black', linewidth=0.8)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(field_labels)
    ax1.set_ylabel('Participation ratio $D_{\\mathrm{eff}}$')
    ax1.set_title('(A) Effective dimensionality', fontweight='bold', loc='left')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: control work comparison
    ax2 = axes[1]
    cw_values = [r['control_work'] for r in results]
    bars2 = ax2.bar(range(3), cw_values, color=colors, edgecolor='black', linewidth=0.8)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(field_labels)
    ax2.set_ylabel('Control work $W_{\\mathrm{ctrl}}$')
    ax2.set_title('(B) Thermodynamic cost', fontweight='bold', loc='left')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add percentage reduction annotation
    reduction = (cw_values[0] - cw_values[2]) / cw_values[0] * 100
    ax2.annotate(f'{reduction:.0f}% reduction', xy=(2, cw_values[2]),
                xytext=(2, cw_values[2] + 0.15),
                ha='center', fontsize=9,
                arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig4_ephaptic.pdf')
    plt.savefig(f'{FIGURES_DIR}/fig4_ephaptic.png')
    plt.close()
    print("  -> Saved fig4_ephaptic.pdf")


def generate_all_figures():
    """Generate all figures for the paper."""
    print("="*60)
    print("Generating figures for Dimensional Landauer Bound paper")
    print("="*60)

    figure1_curvature()
    figure2_kuramoto()
    figure3_autoencoder()
    figure4_ephaptic()

    print("="*60)
    print(f"All figures saved to {FIGURES_DIR}/")
    print("="*60)


if __name__ == '__main__':
    generate_all_figures()
