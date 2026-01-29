#!/usr/bin/env python
# coding: utf-8

"""
Time-dependent plot of protein production rate normalized by mRNA for CME Data
This script calculates (protein_t+1 - protein_t) / (mRNA_R2) 
at each time step and plots it as a function of time.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_pub_figure import setup_publication_style
from jLM import CMEPostProcessing as PostProcessing

# Configuration
cme_traj_dir = "/data2/2024_Yeast_GS/my_current_code/my_cme_ode/output/03232025/"
cme_file = 'gal_cme_ode_gae11.1mM_11.1_gai0_rep50_delta1_time60.lm'
output_dir = "/data2/2024_Yeast_GS/my_current_code/rdme_ode_results/20251031_baseline_newcytoribono/figures_protein_production_rate_plots/"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Setup publication style
colors = setup_publication_style(dpi=300)

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

if num_trajectories == 0:
    print("Error: No trajectories were successfully loaded!")
    exit(1)

# Convert to arrays for easier calculation
r2_array = np.array(r2_trajectories)  # Shape: (num_trajectories, num_timepoints)
g2_array = np.array(g2_trajectories)  # Shape: (num_trajectories, num_timepoints)

# Calculate (protein_t+1 - protein_t) / (mRNA_R2) for each trajectory
print("\nCalculating protein production rate normalized by mRNA...")
production_rates = []  # List to store production rate trajectories

for i in range(num_trajectories):
    r2_traj = r2_array[i, :]
    g2_traj = g2_array[i, :]
    
    # Calculate delta_protein = protein_t+1 - protein_t
    delta_g2 = np.diff(g2_traj)  # Shape: (num_timepoints - 1,)
    
    # Get R2 values at time t (use all but last time point)
    r2_at_t = r2_traj[:-1]  # Shape: (num_timepoints - 1,)
    
    # Calculate production rate: delta_g2 / r2_at_t
    # Handle division by zero (set to NaN or 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.where(r2_at_t > 0, delta_g2 / r2_at_t, np.nan)
    
    production_rates.append(rate)

# Convert to array for easier mean calculation
production_rates_array = np.array(production_rates)  # Shape: (num_trajectories, num_timepoints - 1)

# Calculate mean production rate across all trajectories
production_rate_mean = np.nanmean(production_rates_array, axis=0)

# Time points for the production rate (use time points from t[0] to t[-2])
# since we have num_timepoints - 1 rate values
time_points_for_rate = times[:-1]

# Calculate overall mean value of the mean trajectory
overall_mean_value = np.nanmean(production_rate_mean)

print(f"\nProduction rate statistics:")
print(f"  Mean production rate range: [{np.nanmin(production_rate_mean):.4f}, {np.nanmax(production_rate_mean):.4f}]")
print(f"  Mean production rate (overall): {overall_mean_value:.4f}")

# Create the figure
fig, ax = plt.subplots(figsize=(12, 8))

# Plot mean trajectory with bold line
ax.plot(time_points_for_rate, production_rate_mean, 'r-', linewidth=3.0, alpha=0.9, 
        label='Mean Production Rate', zorder=10)

# Plot horizontal line showing overall mean value
ax.axhline(y=overall_mean_value, color='blue', linestyle='--', linewidth=2.5, alpha=0.8,
           label=f'Overall Mean: {overall_mean_value:.4f}', zorder=9)

# Customize plot
ax.set_xlabel('Time (min)', fontsize=18, fontweight='bold')
ax.set_ylabel('(G2_{t+1} - G2_t) / R2_t', fontsize=18, fontweight='bold')
# ax.set_title('CME: Protein Production Rate Normalized by mRNA', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='best', framealpha=0.95, fontsize=16, edgecolor='black')
ax.grid(True, alpha=0.3, linestyle='--')

# Add some statistics as text
stats_text = f'N trajectories: {num_trajectories}\n'
stats_text += f'Time span: {times[0]:.1f} - {times[-1]:.1f} min\n'
stats_text += f'Mean rate: {overall_mean_value:.4f}'
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        fontsize=16, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Save figure
fig_path = os.path.join(output_dir, 'CME_protein_production_rate.png')
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nSaved production rate plot to: {fig_path}")

plt.close()

print("\n" + "="*70)
print("CME protein production rate plot generation complete!")
print("="*70)

