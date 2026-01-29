#!/usr/bin/env python
# coding: utf-8

"""
Time-dependent plot of protein production rate normalized by mRNA
This script calculates (protein_t+1 - protein_t) / (mRNA_R2 + ribosomeR2) 
at each time step and plots it as a function of time.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_pub_figure import setup_publication_style
from jLM.RDME import File as RDMEFile
from traj_analysis_rdme import *
from tqdm import tqdm

# Configuration
rdme_traj_dir = "/data2/2024_Yeast_GS/my_current_code/rdme_ode_results/20251031_baseline_newcytoribono"
output_dir = os.path.join(rdme_traj_dir, 'figures_protein_production_rate_plots/')
traj_suff = "_ode.jsonl"

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Setup publication style
colors = setup_publication_style( dpi=300)

print(f"Loading RDME data from: {rdme_traj_dir}")

# Get list of RDME trajectory files
rdme_files = [f for f in os.listdir(rdme_traj_dir) if f.startswith('yeast') and f.endswith('.lm')]
rdme_files.sort()

print(f"Found {len(rdme_files)} RDME trajectory files")
print(f"Files: {rdme_files[:3]}..." if len(rdme_files) > 3 else f"Files: {rdme_files}")

# Species to extract
g2_species = 'G2'
r2_species = ['R2', 'ribosomeR2']

# Initialize storage for trajectories
r2_trajectories = []
g2_trajectories = []
times = None

# Process each RDME file
print("\nExtracting R2 and G2 trajectories from RDME files...")
for traj_file in tqdm(rdme_files, desc="Processing RDME files", unit="file"):
    try:
        # Load trajectory data
        traj, odeTraj, region_traj = get_traj(rdme_traj_dir, traj_file, traj_suff, region_suff='_region.jsonl')
        
        # Get data for plotting
        curr_rdmeTs, rdmeYs, curr_odeTs, odeYs, regionTs, regionYs = get_data_for_plot(traj, odeTraj, region_traj=region_traj, sparse_factor=1)
        
        # Store time points (only need to do this once)
        if times is None:
            times = curr_rdmeTs
        
        # Extract R2 trajectory (sum across all regions if needed)
        r2_traj = np.array(rdmeYs[r2_species[0]])
        for species in r2_species[1:]:
            if species in rdmeYs:
                r2_traj += np.array(rdmeYs[species])
        r2_trajectories.append(r2_traj.tolist())
        
        # Extract and sum G2 species trajectories
        g2_traj = None
        if g2_species in rdmeYs:
            g2_traj = np.array(rdmeYs[g2_species])
            g2_trajectories.append(g2_traj.tolist())
        else:
            print(f"Warning: {g2_species} not found in {traj_file}")
            continue
            
    except Exception as e:
        print(f"Error processing {traj_file}: {e}")
        continue

num_trajectories = len(r2_trajectories)
print(f"\nSuccessfully extracted {num_trajectories} R2 trajectories")
print(f"Successfully extracted {len(g2_trajectories)} G2 trajectories")
print(f"Time points per trajectory: {len(times)}")

if num_trajectories == 0:
    print("Error: No trajectories were successfully loaded!")
    exit(1)

# Convert to arrays for easier calculation
r2_array = np.array(r2_trajectories)  # Shape: (num_trajectories, num_timepoints)
g2_array = np.array(g2_trajectories)  # Shape: (num_trajectories, num_timepoints)

# Calculate (protein_t+1 - protein_t) / (mRNA_R2 + ribosomeR2) for each trajectory
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
ax.set_ylabel('(G2_{t+1} - G2_t) / (R2_t + ribosomeR2_t)', fontsize=18, fontweight='bold')
# ax.set_title('RDME: Protein Production Rate Normalized by mRNA', fontsize=16, fontweight='bold', pad=20)
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
fig_path = os.path.join(output_dir, 'RDME_protein_production_rate.png')
plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"\nSaved production rate plot to: {fig_path}")

plt.close()

print("\n" + "="*70)
print("RDME protein production rate plot generation complete!")
print("="*70)

