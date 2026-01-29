#!/usr/bin/env python
# coding: utf-8

# # CME-ODE Species and Totals Plotting
# This script extracts and plots CME-ODE simulation results for individual species and total species
# 11.1 mM file is in the output/03232025/ folder
# 5.55 mM file is in the output/25112025/ folder
import pickle
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('/data2/2024_Yeast_GS/my_current_code/rdme_ode')
from matplotlib_pub_figure import setup_publication_style
from tqdm import tqdm
import pandas as pd
import logging
from jLM import CMEPostProcessing as PostProcessing
import hashlib

# Configuration
cme_traj_dir = "/data2/2024_Yeast_GS/my_current_code/my_cme_ode/output/07122025/"
fig_dir = os.path.join(cme_traj_dir, 'figures_cme_species/')

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# Configure logging
log_file = os.path.join(fig_dir, 'run_log.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Setup publication style
colors = setup_publication_style(figure_size='medium', dpi=300)

logging.info(f"Loading CME-ODE data from: {cme_traj_dir}")

'''
================================================================================================
Load CME trajectory data
================================================================================================
'''

cme_files = ['gal_cme_ode_gae5.55_gia0_rep10_delta1_time420.lm']
logging.info(f"CME-ODE files: {cme_files}")

# Load CME data
cme_traj = PostProcessing.openLMFile(os.path.join(cme_traj_dir, cme_files[0]))

'''
================================================================================================
Pattern definitions for total species calculation
================================================================================================
'''
# Patterns to search for in the species name
total_species_pattern = {
    'G1': 'G1',
    'G2': 'G2',
    'G3': 'G3',
    'G4': 'G4',
    'G80': 'G80',
    'Grep': 'reporter',
    'R1': 'R1',
    'R2': 'R2',
    'R3': 'R3',
    'R4': 'R4',
    'R80': 'R80',
    'Rrep': 'reporter_rna',
}

'''
================================================================================================
Extract and organize CME species
================================================================================================
'''

cme_species_list = PostProcessing.getSpecies(cme_traj)

# Reorganize the species list
cme_species_list = sorted(cme_species_list, key=lambda x: (
    not x[0].startswith('DG'),                                # Sort DG species first
    not (x[0].startswith('R') or x[0] == 'reporter_rna'),     # Then R species and reporter_rna
    not (x[0].startswith('G') and not x[0].startswith('GA')   # Then G species (except GA)
        or x[0] == 'reporter'), 
    x[0].startswith('GA')                                     # GA species last
))

GA_species_list = ['GAI']
# Build general species list based on the name of cme_species_list except GAI 
general_species_list = [species for species in cme_species_list if species not in GA_species_list]

logging.info(f"Total number of species: {len(GA_species_list)} + {len(general_species_list)}")

'''
================================================================================================
Calculate statistics for CME species
================================================================================================
'''

avg_list_general = []
var_list_general = []
time_list_general = []

avg_list_GA = []
var_list_GA = []
time_list_GA = []

# Store raw trajectories for min/max calculation
raw_traj_general = []
raw_traj_GA = []

# Process general species
for species in tqdm(general_species_list, desc="Processing general species"):
    avg, var, times = PostProcessing.getAvgVarTrace(cme_traj, species)
    avg_list_general.append(avg)
    var_list_general.append(np.sqrt(var))
    time_list_general.append(times)
    
    # Get raw trajectories for min/max
    raw_trajs = []
    for i in range(PostProcessing.getNumTrajectories(cme_traj)):
        traj_data = PostProcessing.getTrajectory(cme_traj, i, species)
        raw_trajs.append(traj_data)
    raw_traj_general.append(raw_trajs)

# Process GA species (with unit conversion)
if len(GA_species_list) == 1:
    species = GA_species_list[0]
    avg, var, times = PostProcessing.getAvgVarTrace(cme_traj, species)
    count2concentration = 4.65e-8  # molecule/cell to mM
    avg_list_GA.append(avg * count2concentration)
    var_list_GA.append(np.sqrt(var) * count2concentration)
    time_list_GA.append(times)
    
    # Get raw trajectories for GA species
    raw_trajs = []
    for i in range(PostProcessing.getNumTrajectories(cme_traj)):
        traj_data = PostProcessing.getTrajectory(cme_traj, i, species) * count2concentration
        raw_trajs.append(traj_data)
    raw_traj_GA.append(raw_trajs)

'''
================================================================================================
Create CME results dataframes
================================================================================================
'''

cme_results = []
cme_total_results = []
cme_total_trajs = {}
all_raw_trajs = raw_traj_general + raw_traj_GA
all_species = general_species_list + GA_species_list
all_avgs = avg_list_general + avg_list_GA
all_stds = var_list_general + var_list_GA
all_times = time_list_general + time_list_GA

for species, avg, std, times, raw_trajs in zip(all_species, all_avgs, all_stds, all_times, all_raw_trajs):
    species_name = species[0] if isinstance(species, list) else species
    pattern_used = []
    
    # Identify which patterns this species belongs to
    for pattern_key, cme_pattern in total_species_pattern.items():
        if cme_pattern in species_name and 'D' + cme_pattern not in species_name:
            pattern_used.append(cme_pattern)
    
    # Calculate min/max across all trajectories
    trajectories_array = np.array(raw_trajs)
    min_vals = np.min(trajectories_array, axis=0)
    max_vals = np.max(trajectories_array, axis=0)
    
    # Store individual species results
    cme_results.append({
        'Species': species_name,
        'Time': ','.join(map(str, times)),
        'Average': ','.join(map(str, avg)),
        'Std': ','.join(map(str, std)),
        'Min': ','.join(map(str, min_vals)),
        'Max': ','.join(map(str, max_vals))
    })
    
    # Accumulate trajectories for total species calculation
    if len(pattern_used) > 0:
        for pattern in pattern_used:
            mult_factor = 1
            # Handle dimer species
            if pattern == 'G4':
                double_species = ['G4d']
            elif pattern == 'G80':
                double_species = ['G80d', 'G80Cd', 'G80G3i']
            else:
                double_species = []
            
            for double_sp in double_species:
                if double_sp in species_name:
                    mult_factor = 2
                    break
            
            if pattern not in cme_total_trajs.keys():
                cme_total_trajs[pattern] = trajectories_array * mult_factor
            else:
                cme_total_trajs[pattern] += trajectories_array * mult_factor

# Calculate total species statistics
for pattern in cme_total_trajs.keys():
    cme_total_traj = cme_total_trajs[pattern]
    cme_total_avg = np.mean(cme_total_traj, axis=0)
    cme_total_std = np.std(cme_total_traj, axis=0)
    cme_total_min = np.min(cme_total_traj, axis=0)
    cme_total_max = np.max(cme_total_traj, axis=0)
    cme_total_results.append({
        'Species': f"{pattern}_total",
        'Time': ','.join(map(str, times)),
        'Average': ','.join(map(str, cme_total_avg)),
        'Std': ','.join(map(str, cme_total_std)),
        'Min': ','.join(map(str, cme_total_min)),
        'Max': ','.join(map(str, cme_total_max))
    })

'''
================================================================================================
Save statistics to CSV files
================================================================================================
'''

cme_df = pd.DataFrame(cme_results)
cme_total_df = pd.DataFrame(cme_total_results)

# Combine both individual species and total species into a single CSV
cme_combined_df = pd.concat([cme_df, cme_total_df], ignore_index=True)
cme_csv_path = os.path.join(fig_dir, 'cme_species_statistics.csv')
cme_combined_df.to_csv(cme_csv_path, index=False)
logging.info(f"CME statistics (individual + total) saved to: {cme_csv_path}")

'''
================================================================================================
Plotting functions
================================================================================================
'''

def str_to_array(s):
    """Convert string of comma-separated values to numpy array"""
    return np.array([float(x) for x in s.split(',')])

'''
================================================================================================
Plot individual CME species
================================================================================================
'''

logging.info("Creating individual species plots...")

cme_species = set(cme_df['Species'].unique())

for species_name in tqdm(cme_species, desc="Plotting individual species"):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cme_species_rows = cme_df[cme_df['Species'] == species_name]
    
    if len(cme_species_rows) == 0:
        logging.warning(f"No data found for species: {species_name}")
        plt.close()
        continue
    
    data = cme_species_rows.iloc[0]
    
    time = str_to_array(data['Time'])
    avg = str_to_array(data['Average'])
    min_vals = str_to_array(data['Min'])
    max_vals = str_to_array(data['Max'])
    std = str_to_array(data['Std'])
    
    # Plot average with min/max shading
    ax.plot(time, avg, label=f'{species_name}', linestyle='-', color=colors[0], linewidth=2)
    ax.fill_between(time, min_vals, max_vals, alpha=0.2, color=colors[0], label='Min-Max range')
    
    # Customize plot
    ax.set_xlabel('Time (min)')
    if species_name == 'GAI':
        ax.set_ylabel('Concentration (mM)')
    elif "DG" in species_name:
        ax.set_ylabel('Probability')
    else:
        ax.set_ylabel('Counts')
    
    ax.set_title(f'{species_name}')
    ax.legend(framealpha=0.3, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Save figure
    fig_path = os.path.join(fig_dir, f'{species_name}_cme.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

logging.info(f"Individual species plots saved in: {fig_dir}")

'''
================================================================================================
Plot total species
================================================================================================
'''

logging.info("Creating total species plots...")

# cme_total_df is already available from the previous processing step
logging.info(f"Available total species: {cme_total_df['Species'].tolist()}")

for idx, row in tqdm(cme_total_df.iterrows(), desc="Plotting total species", total=len(cme_total_df)):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    species_name = row['Species']
    pattern = species_name.replace('_total', '')
    
    time = str_to_array(row['Time'])
    avg = str_to_array(row['Average'])
    min_vals = str_to_array(row['Min'])
    max_vals = str_to_array(row['Max'])
    std = str_to_array(row['Std'])
    
    # Plot total species
    ax.plot(time, avg, label=f'Total {pattern}', linestyle='-', color=colors[0], linewidth=2)
    ax.fill_between(time, min_vals, max_vals, alpha=0.2, color=colors[0], label='Min-Max range')
    
    # Customize plot
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Counts')
    ax.set_title(f'Total {pattern} Species')
    ax.legend(framealpha=0.3, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Save figure
    fig_path = os.path.join(fig_dir, f'{species_name}_plot.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

logging.info(f"Total species plots saved in: {fig_dir}")

'''
================================================================================================
Create combined plots for specific species groups
================================================================================================
'''

# Plot combined G2 species (G2 + G2GAE + G2GAI)
logging.info("Creating combined G2 species plot...")
fig, ax = plt.subplots(figsize=(10, 6))

g2_species = ['G2', 'G2GAE', 'G2GAI']
g2_species_found = []

for species_name in g2_species:
    species_data = cme_df[cme_df['Species'] == species_name]
    if len(species_data) > 0:
        data = species_data.iloc[0]
        g2_species_found.append(species_name)
        
        time = str_to_array(data['Time'])
        avg = str_to_array(data['Average'])
        
        ax.plot(time, avg, label=species_name, linestyle='-', linewidth=2)

# Also plot the total
g2_total_data = cme_total_df[cme_total_df['Species'] == 'G2_total']
if len(g2_total_data) > 0:
    data = g2_total_data.iloc[0]
    time = str_to_array(data['Time'])
    avg = str_to_array(data['Average'])
    ax.plot(time, avg, label='G2 Total', linestyle='--', linewidth=2, color='black')

ax.set_xlabel('Time (min)')
ax.set_ylabel('Counts')
ax.set_title('G2 Species')
ax.legend(framealpha=0.3, loc='best')
ax.grid(True, alpha=0.3)

fig_path = os.path.join(fig_dir, 'G2_combined_plot.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
logging.info(f"Combined G2 plot saved: {fig_path}")

# Plot combined GAI species (GAI + G1GAI + G3i + G2GAI)
logging.info("Creating combined GAI species plot...")
fig, ax = plt.subplots(figsize=(10, 6))

gai_species = ['GAI', 'G1GAI', 'G3i', 'G2GAI']
gai_species_found = []
count2concentration = 4.65e-8  # molecule/cell to mM

for species_name in gai_species:
    species_data = cme_df[cme_df['Species'] == species_name]
    if len(species_data) > 0:
        data = species_data.iloc[0]
        gai_species_found.append(species_name)
        
        time = str_to_array(data['Time'])
        avg = str_to_array(data['Average'])
        
        # Convert to mM if not GAI (which is already in mM)
        if species_name != 'GAI':
            avg = avg * count2concentration
        
        ax.plot(time, avg, label=species_name, linestyle='-', linewidth=2)

# Calculate and plot combined total
gai_combined_avg = None
for species_name in gai_species:
    species_data = cme_df[cme_df['Species'] == species_name]
    if len(species_data) > 0:
        data = species_data.iloc[0]
        time = str_to_array(data['Time'])
        avg = str_to_array(data['Average'])
        
        if species_name != 'GAI':
            avg = avg * count2concentration
        
        if gai_combined_avg is None:
            gai_combined_avg = avg
        else:
            gai_combined_avg += avg

if gai_combined_avg is not None:
    ax.plot(time, gai_combined_avg, label='GAI Total', linestyle='--', linewidth=2, color='black')

# Add horizontal line for GAE = 11.1mM
ax.axhline(y=11.1, color='gray', linestyle='--', label='GAE', linewidth=1.5)
ax.text(time[0] * 1.05, 10.8, '11.1 mM', color='gray', va='top', ha='left')

ax.set_xlabel('Time (min)')
ax.set_ylabel('Concentration (mM)')
ax.set_title('GAI Species')
ax.legend(framealpha=0.3, loc='best')
ax.grid(True, alpha=0.3)

fig_path = os.path.join(fig_dir, 'GAI_combined_plot.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
logging.info(f"Combined GAI plot saved: {fig_path}")

logging.info("=" * 80)
logging.info("Script completed successfully!")
logging.info(f"All plots saved in: {fig_dir}")
logging.info("=" * 80)

