#!/usr/bin/env python3
"""
Generate concept figure for Dimensional Landauer Bound paper.

Panel (a): Classical Landauer - bit erasure
Panel (b): Dimensional Landauer - projection to lower-dimensional manifold
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Style for publication
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

fig = plt.figure(figsize=(7, 3.5))

# Panel (a): Classical Landauer - bit erasure
ax1 = fig.add_subplot(121)
ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(-0.5, 1.5)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('(a) Classical Landauer', fontweight='bold', fontsize=11)

# Draw two potential wells merging to one
# Initial: two wells (0 and 1 states)
well_x = np.linspace(-0.3, 0.3, 50)
well_y = 4 * well_x**2

# State 0 well
ax1.plot(well_x + 0.3, well_y + 0.1, 'b-', lw=2)
ax1.fill_between(well_x + 0.3, 0, well_y + 0.1, alpha=0.3, color='blue')
ax1.text(0.3, -0.15, '0', ha='center', fontsize=12, fontweight='bold')

# State 1 well
ax1.plot(well_x + 1.0, well_y + 0.1, 'r-', lw=2)
ax1.fill_between(well_x + 1.0, 0, well_y + 0.1, alpha=0.3, color='red')
ax1.text(1.0, -0.15, '1', ha='center', fontsize=12, fontweight='bold')

# Arrow
ax1.annotate('', xy=(1.9, 0.5), xytext=(1.4, 0.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Final: single well (erased to 0)
ax1.plot(well_x + 2.2, well_y + 0.1, 'b-', lw=2)
ax1.fill_between(well_x + 2.2, 0, well_y + 0.1, alpha=0.5, color='blue')
ax1.text(2.2, -0.15, '0', ha='center', fontsize=12, fontweight='bold')

# Work annotation
ax1.text(1.0, 1.2, r'$W \geq k_B T \ln 2$', ha='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Heat dissipation arrow
ax1.annotate('', xy=(2.2, 0.8), xytext=(2.2, 0.5),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='orange'))
ax1.text(2.35, 0.65, 'Q', fontsize=10, color='orange')


# Panel (b): Dimensional Landauer - projection
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title('(b) Dimensional Landauer', fontweight='bold', fontsize=11)

# Create high-dimensional point cloud
np.random.seed(42)
n_points = 200
# Points distributed in 3D (representing high-D)
theta = np.random.uniform(0, 2*np.pi, n_points)
r = np.random.normal(1, 0.3, n_points)
z = np.random.normal(0, 0.4, n_points)
x = r * np.cos(theta) + np.random.normal(0, 0.15, n_points)
y = r * np.sin(theta) + np.random.normal(0, 0.15, n_points)

# Plot high-D cloud
ax2.scatter(x, y, z, c='steelblue', alpha=0.4, s=8, label=r'$D_{\rm in}$')

# Create 1D manifold (curved line)
t_manifold = np.linspace(0, 2*np.pi, 100)
x_manifold = np.cos(t_manifold)
y_manifold = np.sin(t_manifold)
z_manifold = np.zeros_like(t_manifold)

# Plot the manifold
ax2.plot(x_manifold, y_manifold, z_manifold, 'r-', lw=3,
         label=r'$D_{\rm out}$ manifold')

# Projection lines (a few examples)
for i in range(0, n_points, 20):
    # Project to nearest point on circle
    angle = np.arctan2(y[i], x[i])
    x_proj = np.cos(angle)
    y_proj = np.sin(angle)
    ax2.plot([x[i], x_proj], [y[i], y_proj], [z[i], 0],
             'k--', alpha=0.3, lw=0.5)

# Styling
ax2.set_xlabel(r'$x_1$', labelpad=-8)
ax2.set_ylabel(r'$x_2$', labelpad=-8)
ax2.set_zlabel(r'$x_3$', labelpad=-8)
ax2.set_xlim(-1.8, 1.8)
ax2.set_ylim(-1.8, 1.8)
ax2.set_zlim(-1.2, 1.2)
ax2.view_init(elev=25, azim=45)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_zticklabels([])

# Work annotation in 3D
ax2.text2D(0.5, 0.02, r'$W_{\rm dim} \propto k_B T \, C_\Phi$',
           transform=ax2.transAxes, ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax2.legend(loc='upper left', fontsize=8)

plt.tight_layout()
plt.savefig('../figures/fig0_concept.pdf', bbox_inches='tight', dpi=300)
plt.savefig('../figures/fig0_concept.png', bbox_inches='tight', dpi=150)
print("Saved fig0_concept.pdf and fig0_concept.png")
