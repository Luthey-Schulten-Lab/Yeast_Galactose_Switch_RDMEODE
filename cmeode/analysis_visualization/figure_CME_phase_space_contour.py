#!/usr/bin/env python
# coding: utf-8

"""
Phase Space Contour Plot for CME Data
This script creates a 2D contour plot showing mRNA (R2) vs protein (G2) dynamics
using all CME trajectories, with one trajectory overlaid to show the path.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_pub_figure import setup_publication_style
from jLM import CMEPostProcessing as PostProcessing
from scipy.stats import gaussian_kde

# Configuration
cme_traj_dir = "/data2/2024_Yeast_GS/my_current_code/my_cme_ode/output/03232025/"
cme_file = 'gal_cme_ode_gae11.1mM_11.1_gai0_rep50_delta1_time60.lm'
output_dir = "/data2/2024_Yeast_GS/my_current_code/rdme_ode_results/20251031_baseline_newcytoribono/figures_rdmecme_comparison/"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Setup publication style
colors = setup_publication_style(figure_size='medium', dpi=300)

print(f"Loading CME data from: {os.path.join(cme_traj_dir, cme_file)}")

# Load CME trajectory file
cme_traj = PostProcessing.openLMFile(os.path.join(cme_traj_dir, cme_file))

# Get species list
cme_species_list = PostProcessing.getSpecies(cme_traj)
num_trajectories = PostProcessing.getNumTrajectories(cme_traj)

print(f"Number of trajectories: {num_trajectories}")
print(f"Total species: {len(cme_species_list)}")

# Extract R2 (mRNA) and G2 (protein) trajectories
print("\nExtracting R2 and G2 trajectories...")
r2_trajectories = []
g2_trajectories = []
times = None
g2_species = ['G2', 'G2GAE', 'G2GAI']
# Get all trajectories for R2 and G2
for i in range(num_trajectories):
    # Get R2 trajectory
    r2_traj = PostProcessing.getTrajectory(cme_traj, i, 'R2')
    r2_trajectories.append(r2_traj)
    
    # Get G2 trajectory (sum of G2, G2GAE, G2GAI species at each time point)
    g2_traj = np.array(PostProcessing.getTrajectory(cme_traj, i, g2_species[0]))
    for species in g2_species[1:]:
        g2_traj += np.array(PostProcessing.getTrajectory(cme_traj, i, species))
    g2_trajectories.append(g2_traj.tolist())
    
    # Get times (only need to do this once)
    if times is None:
        # Get average data to extract time points
        _, _, times = PostProcessing.getAvgVarTrace(cme_traj, 'R2')

print(f"Successfully extracted {len(r2_trajectories)} R2 trajectories")
print(f"Successfully extracted {len(g2_trajectories)} G2 trajectories")
print(f"Time points per trajectory: {len(times)}")

# Flatten all trajectory points for contour plotting
print("\nCreating contour map from all trajectories...")
all_r2_points = []
all_g2_points = []

for r2_traj, g2_traj in zip(r2_trajectories, g2_trajectories):
    all_r2_points.extend(r2_traj)
    all_g2_points.extend(g2_traj)

all_r2_points = np.array(all_r2_points)
all_g2_points = np.array(all_g2_points)

print(f"Total data points for contour: {len(all_r2_points)}")
print(f"R2 range: [{all_r2_points.min():.1f}, {all_r2_points.max():.1f}]")
print(f"G2 range: [{all_g2_points.min():.1f}, {all_g2_points.max():.1f}]")

# Create the figure
fig, ax = plt.subplots(figsize=(10, 8))

# Create kernel density estimation (KDE) for smooth density contours
print("Computing kernel density estimation...")
# Stack the points into a 2D array for KDE
positions = np.vstack([all_r2_points, all_g2_points])

# Compute the KDE
kde = gaussian_kde(positions)

# Create a grid for evaluating the KDE
n_grid = 100  # Number of grid points in each dimension
r2_grid = np.linspace(all_r2_points.min(), all_r2_points.max(), n_grid)
g2_grid = np.linspace(all_g2_points.min(), all_g2_points.max(), n_grid)
R2_grid, G2_grid = np.meshgrid(r2_grid, g2_grid)

# Evaluate KDE on the grid
grid_positions = np.vstack([R2_grid.ravel(), G2_grid.ravel()])
density = kde(grid_positions)
density = density.reshape(R2_grid.shape)

print(f"Density range: [{density.min():.2e}, {density.max():.2e}]")

# Create contour plot based on density
contour = ax.contourf(R2_grid, G2_grid, density, levels=20, cmap='viridis', alpha=0.8)
cbar = plt.colorbar(contour, ax=ax, label='Probability Density')

# Calculate mean trajectory across all trajectories
print(f"\nCalculating mean trajectory across all {num_trajectories} trajectories...")

# Convert to arrays for easier mean calculation
r2_array = np.array(r2_trajectories)  # Shape: (num_trajectories, num_timepoints)
g2_array = np.array(g2_trajectories)  # Shape: (num_trajectories, num_timepoints)

# Calculate mean trajectory
r2_mean = np.mean(r2_array, axis=0)
g2_mean = np.mean(g2_array, axis=0)

# Find trajectories with minimum and maximum final G2 abundance
final_g2_values = g2_array[:, -1]  # Get final G2 value for each trajectory
idx_min_final = np.argmin(final_g2_values)  # Index of trajectory with lowest final G2
idx_max_final = np.argmax(final_g2_values)  # Index of trajectory with highest final G2

print(f"Trajectory with min final G2: {idx_min_final+1} (G2={final_g2_values[idx_min_final]:.1f})")
print(f"Trajectory with max final G2: {idx_max_final+1} (G2={final_g2_values[idx_max_final]:.1f})")

# Get the full trajectories with min and max final G2
r2_min_final = r2_array[idx_min_final, :]
g2_min_final = g2_array[idx_min_final, :]
r2_max_final = r2_array[idx_max_final, :]
g2_max_final = g2_array[idx_max_final, :]

# Plot the min and max final G2 trajectories with dashed lines
ax.plot(r2_min_final, g2_min_final, '--', color='white', linewidth=4.0, alpha=0.7, 
        label=f'Min Final G2 (Traj {idx_min_final+1})', zorder=9)
ax.plot(r2_max_final, g2_max_final, 'g--', linewidth=2.0, alpha=0.7, 
        label=f'Max Final G2 (Traj {idx_max_final+1})', zorder=9)

# Plot the mean trajectory with solid line
ax.plot(r2_mean, g2_mean, 'r-', linewidth=2.5, alpha=0.9, 
        label='Mean Trajectory', zorder=10)

# Add arrows to show direction of mean trajectory (sample every N points to avoid cluttering)
arrow_spacing = max(1, len(r2_mean) // 15)  # Show about 15 arrows
for i in range(0, len(r2_mean) - arrow_spacing, arrow_spacing):
    ax.annotate('', xy=(r2_mean[i+arrow_spacing], g2_mean[i+arrow_spacing]), 
               xytext=(r2_mean[i], g2_mean[i]),
               arrowprops=dict(arrowstyle='->', color='red', lw=2.5, alpha=0.8),
               zorder=11)

# Mark start and end points for mean trajectory
ax.plot(r2_mean[0], g2_mean[0], 'go', markersize=14, 
        label='Start (t=0)', zorder=12, markeredgecolor='darkgreen', markeredgewidth=2)
ax.plot(r2_mean[-1], g2_mean[-1], 'r*', markersize=18, 
        label=f'End (t={times[-1]:.1f} min)', zorder=12, markeredgecolor='darkred', markeredgewidth=1.5)

# Customize plot
ax.set_xlabel('mRNA R2 (counts)', fontsize=14, fontweight='bold')
ax.set_ylabel('Protein G2 (log(counts))', fontsize=14, fontweight='bold')
ax.set_title('CME: G2 Protein vs R2 mRNA Phase Space', fontsize=16, fontweight='bold', pad=20)
# The following uses matplotlib's 'log' scale, which means log base 10 (log10) by default.
ax.set_yscale('log')

# Add horizontal gridlines and annotate y-axis with actual final G2 values of the min and max G2 trajectories.
final_g2_min = final_g2_values[idx_min_final]
final_g2_max = final_g2_values[idx_max_final]

for val, label, color in [
    (final_g2_min, f"Min Final G2: {final_g2_min:.1f}", "white"),
    (final_g2_max, f"Max Final G2: {final_g2_max:.1f}", "green"),
    (g2_mean[-1], f"Mean Final G2: {g2_mean[-1]:.1f}", "red")]:
    ax.axhline(y=val, linestyle=':', linewidth=1.6, color=color, zorder=5, alpha=0.65)
    ax.text(0, val, f"{label}", 
            va='bottom' if color == "green" else 'top',
            ha='left',
            fontsize=12,
            fontweight="bold",
            color=color,
            bbox=dict(facecolor="black" if color == "white" else "white", 
                      edgecolor=color, alpha=0.65, boxstyle="round,pad=0.18"))

# (Optionally, to clarify in the plot that y is log10 scale)
ax.annotate("log$_{10}$ scale", xy=(0.97, 0.1), xycoords='axes fraction', fontsize=12, ha='right')
ax.legend(loc='best', framealpha=0.95, fontsize=11, edgecolor='black')
ax.grid(True, alpha=0.3, linestyle='--')

# Add some statistics as text
stats_text = f'N trajectories: {num_trajectories}\n'
stats_text += f'Time span: {times[0]:.1f} - {times[-1]:.1f} min\n'
stats_text += f'Points per traj: {len(times)}'
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Save figure
fig_path = os.path.join(output_dir, 'CME_G2_R2_phase_space_contour.png')
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nSaved phase space contour plot to: {fig_path}")



plt.close()

print("\n" + "="*70)
print("Phase space contour plot generation complete!")
print("="*70)

