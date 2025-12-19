# The Dimensional Landauer Bound

**Thermodynamic costs of dimensional reduction in stochastic dynamics**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

Landauer's principle establishes that erasing one bit costs at least $k_B T \ln 2$ of energy. But biological and physical systems rarely operate on discrete bits—they project high-dimensional dynamics onto lower-dimensional manifolds. This projection has its own thermodynamic cost.

We derive the **Dimensional Landauer Bound**:

$$W_{\min} \geq k_B T (\ln 2 \cdot \Delta I + C_\Phi)$$

where $C_\Phi$ is the geometric contraction cost governed by the Jacobian of the projection and the curvature of the target manifold.

## Key Results

- **Curvature costs energy**: Maintaining a low-dimensional manifold requires control power that scales with $\langle \kappa^2 \rangle$
- **Synchronization reduces cost**: Coherent oscillators spontaneously collapse effective dimension, reducing projection work
- **Bottleneck crossover**: Compressing below intrinsic dimension causes divergent training work in autoencoders
- **Structured noise is work**: Colored noise ($1/f^\alpha$) represents finite-dimensional confinement—filtering is thermodynamic work

## Running Simulations

```bash
cd code
python simulations.py           # Run all four experiments
python generate_figures.py      # Generate publication figures
```

## Paper

**The Dimensional Landauer Bound: Thermodynamic costs of dimensional reduction in stochastic dynamics**

Todd, I. (2025). *Chaos, Solitons & Fractals* (in preparation).

Companion paper: [Projection-Induced Discontinuities](https://github.com/todd866/projection-discontinuities)

## Citation

```bibtex
@article{todd2025dimensional,
  author  = {Todd, Ian},
  title   = {The Dimensional Landauer Bound: Thermodynamic costs of
             dimensional reduction in stochastic dynamics},
  journal = {Chaos, Solitons \& Fractals},
  year    = {2025},
  note    = {In preparation}
}
```

## License

MIT License
