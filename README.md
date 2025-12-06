# The Dimensional Landauer Bound

**Formation Cost vs Maintenance Power in Thermodynamic Computing**

> **Journal:** Physical Review X
> **Status:** Under Review (Manuscript ID: es2025dec06_550)
> **Submitted:** December 6, 2025

## Abstract

Landauer's principle establishes a minimum energetic cost for logically irreversible operations ($k_B T \ln 2$ per bit erased). We derive a generalization showing that dimensional reduction imposes an additional thermodynamic cost:

$$W_{\min} \geq k_B T \ln 2 \cdot \Delta I + k_B T \cdot C_\Phi$$

where $C_\Phi = D_{\mathrm{KL}}(p(x) \| \Phi^\dagger p(y))$ is the **geometric contraction cost**---the information destroyed by projecting high-dimensional dynamics onto a lower-dimensional manifold.

## Key Result: Formation vs Maintenance

The Dimensional Landauer Bound distinguishes:

- **Formation cost** ($W_{\mathrm{dim}}$, Joules): One-time "deposit" to create a low-dimensional representation
- **Maintenance power** ($P_{\mathrm{maint}}$, Watts): Continuous "rent" to sustain the representation against thermal noise

This formation/maintenance distinction:
- Explains why biological systems favour oscillatory, coherent dynamics
- Provides thermodynamic foundation for analog vs digital computing tradeoffs
- Connects to surface tension (formation cost for interfaces) and black hole thermodynamics (Bekenstein-Hawking as formation cost, Hawking radiation as maintenance)

## Repository Structure

```
dimensional-landauer-bound/
├── manuscript_prx.tex/pdf      # Main manuscript
├── supplementary.tex/pdf       # Supplementary material with derivations
├── references.bib              # Bibliography
├── code/
│   ├── run_all_sims.py                  # Master script
│   ├── sim1_brownian_manifolds.py       # Curvature vs work
│   ├── sim1_brownian_manifolds_validated.py  # With validation
│   ├── sim1_compression_protocol.py     # Compression protocol demo
│   ├── sim2_kuramoto.py                 # Coherence vs work
│   ├── sim3_autoencoder.py              # Bottleneck vs effort
│   └── sim4_ephaptic_sheet.py           # Ephaptic coupling
└── figures/                    # Generated figures (PDF + PNG)
```

## Numerical Demonstrations

### 1. Curvature Increases Control Cost
2D Brownian motion constrained to curved 1D manifolds shows geometric work scales with curvature.

### 2. Coherence Reduces Dimensional Work
Kuramoto oscillators demonstrate that higher phase coherence reduces control work for the same readout fidelity.

### 3. The Thermodynamic Cost of Over-Compression
Autoencoder training (SGD-as-Langevin) shows effort explosion below intrinsic dimensionality.

### 4. Ephaptic Coupling as a Dimensional Strategy
Neural sheet with energy-matched random vs coherent fields proves that geometric alignment (not just energy) reduces work.

## Running Simulations

```bash
cd code
python run_all_sims.py
```

Generates all figures in `figures/`.

## Requirements

- Python 3.10+
- NumPy
- PyTorch
- Matplotlib

## Citation

```bibtex
@article{todd2025dimensional,
  author  = {Todd, Ian},
  title   = {The Dimensional Landauer Bound: Formation Cost vs Maintenance Power in Thermodynamic Computing},
  journal = {Physical Review X},
  year    = {2025},
  note    = {Under review, Manuscript ID: es2025dec06\_550}
}
```

## Related Work

- Todd I. (2025). The limits of falsifiability. *BioSystems*, 258, 105608.
- Todd I. (2025). Timing inaccessibility and the projection bound. *BioSystems*, 258, 105632.
- Landauer R. (1961). Irreversibility and heat generation. *IBM JRD*, 5(3), 183-191.
- Seifert U. (2012). Stochastic thermodynamics. *Rep. Prog. Phys.*, 75, 126001.

## License

MIT License

## Contact

Ian Todd - itod2305@uni.sydney.edu.au
University of Sydney
