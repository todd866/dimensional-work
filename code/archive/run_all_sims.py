#!/usr/bin/env python3
"""
Master script: Run all four simulations for the Dimensional Landauer Bound paper.

Generates all figures in ../figures/
"""

import os
import sys

# Ensure we're in the code directory for relative paths
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("THE DIMENSIONAL LANDAUER BOUND")
print("Simulation Suite")
print("=" * 60)

# Import and run each simulation
print("\n" + "=" * 60)
print("SIMULATION 1: Brownian Motion on Curved Manifolds")
print("=" * 60)
from sim1_brownian_manifolds import main as sim1_main
sim1_main(save_fig=True)

print("\n" + "=" * 60)
print("SIMULATION 2: Kuramoto Oscillators - Coherence vs Work")
print("=" * 60)
from sim2_kuramoto import main as sim2_main
sim2_main(save_fig=True)

print("\n" + "=" * 60)
print("SIMULATION 3: Autoencoder Bottleneck (SGD-as-Langevin)")
print("=" * 60)
from sim3_autoencoder import main as sim3_main
sim3_main(save_fig=True)

print("\n" + "=" * 60)
print("SIMULATION 4: Ephaptic Neural Sheet (Energy-Matched)")
print("=" * 60)
from sim4_ephaptic_sheet import main as sim4_main
sim4_main(save_fig=True)

print("\n" + "=" * 60)
print("ALL SIMULATIONS COMPLETE")
print("Figures saved to ../figures/")
print("=" * 60)
