# Thermodynamic Costs of Dimensional Reduction

**Extending Landauer's Principle to Geometric Compression**

> **Target:** Physical Review E
> **Status:** In preparation

## Abstract

Landauer's principle establishes a minimum energetic cost for logically irreversible operations ($k_B T \ln 2$ per bit erased). We derive a generalization showing that dimensional reduction imposes an additional thermodynamic cost:

$$W_{\min} \geq k_B T \ln 2 \cdot \Delta I + k_B T \cdot C_\Phi$$

where $C_\Phi$ is the **geometric contraction cost**—the information destroyed by projecting high-dimensional dynamics onto a lower-dimensional manifold, governed by the Jacobian of the projection.

## Key Result: Formation vs Maintenance

The Dimensional Landauer Bound distinguishes:

- **Formation cost** ($W_{\mathrm{dim}}$, Joules): One-time cost to create a low-dimensional representation
- **Maintenance power** ($P_{\mathrm{maint}}$, Watts): Continuous dissipation to sustain the representation against thermal noise

This framework:
- Connects to the Information Bottleneck (Tishby et al.) as its thermodynamic dual
- Explains why biological systems favour oscillatory, coherent dynamics
- Provides thermodynamic foundation for analog vs digital computing tradeoffs

## Repository Structure

```
dimensional-work/
├── manuscript_pre.tex/pdf      # Main manuscript
├── references.bib              # Bibliography
├── code/
│   └── simulations.py          # All simulations
├── figures/                    # Generated figures (PDF + PNG)
├── paper2/                     # "Dimensional Work of Surfaces" (PRE)
└── paper3/                     # "Dimensional Work of Black Holes" (PRD)
```

## Numerical Demonstrations

Run all simulations:
```bash
cd code
python simulations.py
```

Or run individual simulations:
```bash
python simulations.py concept      # Concept figure
python simulations.py curvature    # Brownian manifolds
python simulations.py kuramoto     # Coherence vs work
python simulations.py autoencoder  # Bottleneck divergence
python simulations.py ephaptic     # Neural sheet coupling
```

## Requirements

- Python 3.10+
- NumPy, SciPy, Matplotlib
- PyTorch (for autoencoder simulation)

## Citation

```bibtex
@article{todd2025dimensional,
  author  = {Todd, Ian},
  title   = {Thermodynamic costs of dimensional reduction in stochastic dynamics},
  journal = {Physical Review E},
  year    = {2025},
  note    = {In preparation}
}
```

## Related Work

- Todd I. (2025). The limits of falsifiability. *BioSystems*, 258, 105608.
- Todd I. (2025). Timing inaccessibility and the projection bound. *BioSystems*, 258, 105632.
- Tishby N. et al. (2000). The information bottleneck method.
- Seifert U. (2012). Stochastic thermodynamics. *Rep. Prog. Phys.*, 75, 126001.

## License

MIT License

## Contact

Ian Todd - itod2305@uni.sydney.edu.au
University of Sydney
