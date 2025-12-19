"""
Dimensional Landauer Bound: Simulation Code
============================================

Simulations validating the thermodynamic costs of dimensional reduction
in stochastic dynamics.

Author: Ian Todd
"""

import numpy as np
from scipy.integrate import odeint
from scipy.linalg import svd
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


# =============================================================================
# SIMULATION 1: Geometric Cost of Curvature
# =============================================================================

def curvature_simulation(n_steps=10000, dt=0.01, k_confine=100.0, T=1.0,
                         curvature_type='linear'):
    """
    Simulate Brownian motion confined to a 1D manifold in 2D space.

    The particle evolves under overdamped Langevin dynamics with a stiff
    harmonic potential enforcing projection onto the curve x2 = f(x1).

    Parameters
    ----------
    n_steps : int
        Number of simulation steps
    dt : float
        Time step
    k_confine : float
        Stiffness of confining potential
    T : float
        Temperature (k_B T units)
    curvature_type : str
        'linear', 'mild', or 'high' curvature manifold

    Returns
    -------
    dict with trajectory, work, power, curvature statistics
    """
    # Diffusion coefficient (Einstein relation: D = k_B T / gamma, set gamma=1)
    D = T
    noise_amplitude = np.sqrt(2 * D * dt)

    # Define manifold function and its derivatives
    if curvature_type == 'linear':
        f = lambda x: 0.0 * x
        df = lambda x: 0.0
        ddf = lambda x: 0.0
    elif curvature_type == 'mild':
        f = lambda x: 0.2 * np.sin(x)
        df = lambda x: 0.2 * np.cos(x)
        ddf = lambda x: -0.2 * np.sin(x)
    elif curvature_type == 'high':
        f = lambda x: 0.8 * np.sin(2*x)
        df = lambda x: 1.6 * np.cos(2*x)
        ddf = lambda x: -3.2 * np.sin(2*x)
    else:
        raise ValueError(f"Unknown curvature type: {curvature_type}")

    # Curvature: kappa = |f''| / (1 + f'^2)^(3/2)
    def curvature(x):
        return np.abs(ddf(x)) / (1 + df(x)**2)**1.5

    # Initialize particle on manifold
    x = np.zeros(2)
    x[0] = 0.0
    x[1] = f(x[0])

    # Storage
    trajectory = np.zeros((n_steps, 2))
    work_cumulative = 0.0
    work_history = np.zeros(n_steps)
    curvature_history = np.zeros(n_steps)

    for i in range(n_steps):
        trajectory[i] = x
        curvature_history[i] = curvature(x[0])

        # Control force: push particle back to manifold
        # F_control = -k * (x2 - f(x1)) * grad(x2 - f(x1))
        deviation = x[1] - f(x[0])
        F_control = np.array([k_confine * deviation * df(x[0]),
                              -k_confine * deviation])

        # Thermal noise
        noise = noise_amplitude * np.random.randn(2)

        # Overdamped Langevin: dx = F*dt + noise
        dx = F_control * dt + noise

        # Work done by control force: dW = F_control . dx
        # For maintenance power, we track |F_control|^2 * dt (steady state dissipation)
        work_increment = np.dot(F_control, F_control) * dt
        work_cumulative += work_increment
        work_history[i] = work_cumulative

        x = x + dx

    # Maintenance power (average dissipation rate)
    power = work_cumulative / (n_steps * dt)

    return {
        'trajectory': trajectory,
        'work_history': work_history,
        'power': power,
        'curvature_mean_sq': np.mean(curvature_history**2),
        'curvature_max': np.max(curvature_history),
        'curvature_type': curvature_type
    }


def run_curvature_experiment():
    """Run curvature scaling experiment for all manifold types."""
    results = {}
    for ctype in ['linear', 'mild', 'high']:
        results[ctype] = curvature_simulation(
            n_steps=20000, dt=0.005, k_confine=200.0, T=1.0,
            curvature_type=ctype
        )
        print(f"{ctype:8s}: power = {results[ctype]['power']:.2f}, "
              f"mean-squared curvature = {results[ctype]['curvature_mean_sq']:.3f}")
    return results


# =============================================================================
# SIMULATION 2: Kuramoto Oscillators and Effective Dimension Collapse
# =============================================================================

def kuramoto_simulation(N=64, K=1.0, T_total=50.0, dt=0.01, T_temp=0.1):
    """
    Simulate coupled Kuramoto oscillators with thermal noise.

    Parameters
    ----------
    N : int
        Number of oscillators
    K : float
        Coupling strength
    T_total : float
        Total simulation time
    dt : float
        Time step
    T_temp : float
        Temperature (noise level)

    Returns
    -------
    dict with coherence, participation ratio, control work
    """
    n_steps = int(T_total / dt)
    noise_amplitude = np.sqrt(2 * T_temp * dt)

    # Natural frequencies (drawn from Lorentzian with gamma=0.5)
    omega = np.random.standard_cauchy(N) * 0.5

    # Initialize phases uniformly
    theta = np.random.uniform(0, 2*np.pi, N)

    # Storage
    coherence_history = np.zeros(n_steps)
    participation_ratio_history = np.zeros(n_steps)
    control_power_history = np.zeros(n_steps)

    # For participation ratio, we need covariance of phases
    phase_buffer = []
    buffer_size = 100

    for step in range(n_steps):
        # Compute order parameter
        z = np.mean(np.exp(1j * theta))
        r = np.abs(z)
        psi = np.angle(z)
        coherence_history[step] = r

        # Coupling force: K * r * sin(psi - theta_i)
        coupling = K * r * np.sin(psi - theta)

        # Control work: force needed to maintain low-D projection
        # Proxy: variance of deviations from mean field
        phase_deviations = theta - psi
        phase_deviations = np.mod(phase_deviations + np.pi, 2*np.pi) - np.pi
        control_power = np.var(phase_deviations)
        control_power_history[step] = control_power

        # Phase covariance for participation ratio
        phase_buffer.append(np.cos(theta).copy())
        if len(phase_buffer) > buffer_size:
            phase_buffer.pop(0)

        if len(phase_buffer) == buffer_size:
            cov = np.cov(np.array(phase_buffer).T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if len(eigenvalues) > 0:
                pr = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
            else:
                pr = 1.0
            participation_ratio_history[step] = pr
        else:
            participation_ratio_history[step] = N  # Initial high value

        # Langevin update
        dtheta = (omega + coupling) * dt + noise_amplitude * np.random.randn(N)
        theta = theta + dtheta

    return {
        'coherence': coherence_history,
        'participation_ratio': participation_ratio_history,
        'control_power': control_power_history,
        'K': K,
        'N': N
    }


def run_kuramoto_experiment():
    """Run Kuramoto experiment across coupling strengths."""
    K_values = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]
    results = []

    for K in K_values:
        res = kuramoto_simulation(N=64, K=K, T_total=100.0, dt=0.01, T_temp=0.05)
        # Take steady-state averages (last 50%)
        n_ss = len(res['coherence']) // 2
        r_mean = np.mean(res['coherence'][n_ss:])
        pr_mean = np.mean(res['participation_ratio'][n_ss:])
        cp_mean = np.mean(res['control_power'][n_ss:])

        results.append({
            'K': K,
            'coherence': r_mean,
            'participation_ratio': pr_mean,
            'control_power': cp_mean
        })
        print(f"K={K:.1f}: r={r_mean:.3f}, PR={pr_mean:.1f}, power={cp_mean:.3f}")

    return results


# =============================================================================
# SIMULATION 3: Autoencoder Thermodynamic Divergence
# =============================================================================

def generate_manifold_data(n_samples=1000, d_intrinsic=2, d_ambient=10, noise=0.01):
    """
    Generate data lying on a low-dimensional manifold in high-D space.

    Creates a swiss roll-like manifold.
    """
    # Intrinsic coordinates
    t = np.random.uniform(0, 4*np.pi, n_samples)
    s = np.random.uniform(0, 1, n_samples)

    # Embed in d_intrinsic
    intrinsic = np.column_stack([t * np.cos(t), t * np.sin(t), s])[:, :d_intrinsic]

    # Random projection to ambient space
    proj = np.random.randn(d_intrinsic, d_ambient)
    proj, _ = np.linalg.qr(proj.T)
    proj = proj.T

    data = intrinsic @ proj + noise * np.random.randn(n_samples, d_ambient)

    return data, intrinsic


def simple_autoencoder_work(data, bottleneck_dim, n_epochs=100, lr=0.01):
    """
    Train a linear autoencoder and measure cumulative gradient work.

    This is a simplified proxy for SGD thermodynamics.
    """
    n_samples, d_in = data.shape

    # Normalize data for numerical stability
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data) + 1e-8
    data_norm = (data - data_mean) / data_std

    # Initialize encoder/decoder weights (Xavier initialization)
    W_enc = np.random.randn(d_in, bottleneck_dim) * np.sqrt(2.0 / (d_in + bottleneck_dim))
    W_dec = np.random.randn(bottleneck_dim, d_in) * np.sqrt(2.0 / (d_in + bottleneck_dim))

    work_history = []
    loss_history = []

    for epoch in range(n_epochs):
        # Forward pass
        encoded = data_norm @ W_enc
        reconstructed = encoded @ W_dec

        # Loss
        loss = np.mean((data_norm - reconstructed)**2)
        loss_history.append(loss)

        # Gradients (simplified batch gradient descent)
        error = reconstructed - data_norm
        grad_dec = encoded.T @ error / n_samples
        grad_enc = data_norm.T @ (error @ W_dec.T) / n_samples

        # Gradient clipping for stability
        grad_norm_enc = np.linalg.norm(grad_enc)
        grad_norm_dec = np.linalg.norm(grad_dec)
        max_grad = 10.0
        if grad_norm_enc > max_grad:
            grad_enc = grad_enc * max_grad / grad_norm_enc
        if grad_norm_dec > max_grad:
            grad_dec = grad_dec * max_grad / grad_norm_dec

        # Work = sum of squared gradient norms (proxy for SGD dissipation)
        work = np.sum(grad_enc**2) + np.sum(grad_dec**2)
        work_history.append(work)

        # Update
        W_enc -= lr * grad_enc
        W_dec -= lr * grad_dec

    return {
        'bottleneck_dim': bottleneck_dim,
        'final_loss': loss_history[-1],
        'total_work': np.sum(work_history),
        'work_history': work_history,
        'loss_history': loss_history
    }


def run_autoencoder_experiment():
    """Run autoencoder experiment with varying bottleneck dimensions."""
    # Generate data on 2D manifold in R^10
    data, _ = generate_manifold_data(n_samples=500, d_intrinsic=2, d_ambient=10)

    bottleneck_dims = [1, 2, 3, 4, 5, 6]
    results = []

    for d_b in bottleneck_dims:
        res = simple_autoencoder_work(data, d_b, n_epochs=200, lr=0.05)
        results.append(res)
        print(f"d_b={d_b}: loss={res['final_loss']:.4f}, work={res['total_work']:.2f}")

    return results


# =============================================================================
# SIMULATION 4: Ephaptic Field-Mediated Alignment
# =============================================================================

def ephaptic_simulation(N=100, field_type='coherent', field_strength=0.5,
                        T_total=20.0, dt=0.01, T_temp=0.1):
    """
    Simulate neural sheet with mean-field coupling (ephaptic).

    Parameters
    ----------
    N : int
        Number of neurons (1D sheet)
    field_type : str
        'none', 'random', or 'coherent'
    field_strength : float
        Strength of field coupling
    T_total : float
        Total simulation time
    dt : float
        Time step
    T_temp : float
        Temperature

    Returns
    -------
    dict with participation ratio, control work
    """
    n_steps = int(T_total / dt)
    noise_amplitude = np.sqrt(2 * T_temp * dt)

    # State: membrane potentials
    V = np.random.randn(N) * 0.1

    # Local connectivity (nearest neighbor)
    local_coupling = 0.3

    # Storage
    participation_ratio_history = np.zeros(n_steps)
    control_work_history = np.zeros(n_steps)

    V_buffer = []
    buffer_size = 50

    for step in range(n_steps):
        # Local diffusive coupling
        V_left = np.roll(V, 1)
        V_right = np.roll(V, -1)
        local_force = local_coupling * (V_left + V_right - 2*V)

        # Field coupling
        if field_type == 'none':
            field_force = np.zeros(N)
        elif field_type == 'random':
            field = np.random.randn()
            field_force = field_strength * field * np.ones(N)
        elif field_type == 'coherent':
            mean_V = np.mean(V)
            field_force = field_strength * (mean_V - V)
        else:
            field_force = np.zeros(N)

        # Control work: variance of deviations from mean field
        control_work = np.var(V)
        control_work_history[step] = control_work

        # Participation ratio from covariance
        V_buffer.append(V.copy())
        if len(V_buffer) > buffer_size:
            V_buffer.pop(0)

        if len(V_buffer) == buffer_size:
            cov = np.cov(np.array(V_buffer).T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            if len(eigenvalues) > 0:
                pr = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
            else:
                pr = 1.0
            participation_ratio_history[step] = pr
        else:
            participation_ratio_history[step] = N

        # Update
        dV = (local_force + field_force) * dt + noise_amplitude * np.random.randn(N)
        V = V + dV
        # Soft bounds
        V = np.clip(V, -5, 5)

    return {
        'participation_ratio': participation_ratio_history,
        'control_work': control_work_history,
        'field_type': field_type,
        'field_strength': field_strength
    }


def run_ephaptic_experiment():
    """Run ephaptic experiment with different field types."""
    field_types = ['none', 'random', 'coherent']
    results = []

    for ftype in field_types:
        res = ephaptic_simulation(N=100, field_type=ftype, field_strength=0.5,
                                  T_total=50.0, dt=0.01, T_temp=0.1)
        # Steady state averages
        n_ss = len(res['participation_ratio']) // 2
        pr_mean = np.mean(res['participation_ratio'][n_ss:])
        cw_mean = np.mean(res['control_work'][n_ss:])

        results.append({
            'field_type': ftype,
            'participation_ratio': pr_mean,
            'control_work': cw_mean
        })
        print(f"{ftype:10s}: PR={pr_mean:.1f}, work={cw_mean:.4f}")

    return results


# =============================================================================
# Utility: Compute Geometric Contraction Cost C_Phi
# =============================================================================

def compute_C_phi(jacobian_samples):
    """
    Compute the geometric contraction cost C_Phi from Jacobian samples.

    C_Phi = -1/2 * < ln det(J J^T) >

    Parameters
    ----------
    jacobian_samples : list of arrays
        Each array is a Jacobian matrix J of shape (d_out, d_in)

    Returns
    -------
    float : C_Phi value
    """
    log_dets = []
    for J in jacobian_samples:
        JJT = J @ J.T
        det = np.linalg.det(JJT)
        if det > 1e-15:
            log_dets.append(np.log(det))

    if len(log_dets) > 0:
        return -0.5 * np.mean(log_dets)
    else:
        return np.inf


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("="*60)
    print("Dimensional Landauer Bound: Validation Simulations")
    print("="*60)

    print("\n1. CURVATURE SCALING")
    print("-"*40)
    curvature_results = run_curvature_experiment()

    print("\n2. KURAMOTO OSCILLATORS")
    print("-"*40)
    kuramoto_results = run_kuramoto_experiment()

    print("\n3. AUTOENCODER BOTTLENECK")
    print("-"*40)
    autoencoder_results = run_autoencoder_experiment()

    print("\n4. EPHAPTIC COUPLING")
    print("-"*40)
    ephaptic_results = run_ephaptic_experiment()

    print("\n" + "="*60)
    print("All simulations complete.")
