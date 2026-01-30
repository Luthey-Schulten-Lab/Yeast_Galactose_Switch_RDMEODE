#!/usr/bin/env python
# coding: utf-8

# # Trajectory Comparison Script
# This code compares simulation results between two different trajectory directories
try:
    # This will work in IPython/Jupyter
    get_ipython().run_line_magic('run', 'env.ipynb')
except NameError:
    # This will run when executed as a regular Python script
    import sys
    import os
    print("Running as a standard Python script - importing modules directly")
    # Add the directory containing env.py to the path if needed
    # sys.path.append('/path/to/directory/containing/env')
    # Import any necessary modules directly
    # from env import *  # If you have env.py instead of env.ipynb

import pickle
import os
import numpy as np
from jLM.RDME import File as RDMEFile
import jLM
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_pub_figure import setup_publication_style
from traj_analysis_rdme import *
from tqdm import tqdm
import pandas as pd
import logging
import hashlib
from scipy import stats
from scipy.stats import ttest_ind, ks_2samp, chi2_contingency


def trajectories_to_array(trajectories):
    """
    Convert list of trajectories to numpy array, cutting to minimum length if ragged.
    
    Args:
        trajectories: list of 1D arrays/lists (possibly different lengths)
    
    Returns:
        numpy array of shape (n_trajectories, min_length)
    """
    if not trajectories:
        return np.array([])
    
    # Find minimum length
    min_len = min(len(t) for t in trajectories)
    
    # Cut all trajectories to minimum length
    truncated = [t[:min_len] for t in trajectories]
    
    return np.array(truncated)


'''
================================================================================================
This section is for loading trajectory, use cached data if available
================================================================================================
'''
def get_dir_cache_key(traj_dir, include_regions, region_suffix):
    """Create cache key for a single directory based only on number of trajs and traj file names"""
    abs_path = os.path.abspath(traj_dir)
    lm_files = [f for f in os.listdir(abs_path) if f.startswith('yeast') and f.endswith('.lm')]
    lm_files.sort()
    
    # Only use number of trajectories and trajectory file names for cache key
    num_trajs = len(lm_files)
    traj_names = tuple(lm_files)
    
    cache_data = (traj_names, num_trajs)
    return hashlib.md5(str(cache_data).encode()).hexdigest()

def save_dir_cache(traj_dir, data, fig_dir, include_regions, region_suffix):
    """Save directory data to cache"""
    cache_key = get_dir_cache_key(traj_dir, include_regions, region_suffix)
    cache_file = os.path.join(fig_dir, f'dir_cache_{cache_key}.pkl')
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        return True
    except:
        return False

def load_dir_cache(traj_dir, fig_dir, include_regions, region_suffix):
    """Load directory data from cache"""
    cache_key = get_dir_cache_key(traj_dir, include_regions, region_suffix)
    cache_file = os.path.join(fig_dir, f'dir_cache_{cache_key}.pkl')
    try:
        if os.path.exists(cache_file):
            logging.info(f"Found cache file: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            logging.info(f"Cache file does not exist: {cache_file}")
    except Exception as e:
        logging.info(f"Error loading cache file {cache_file}: {e}")
    return None

# Get user input for directories and comparison settings
def get_user_input():
    print("\n=== Trajectory Comparison Setup ===")
    traj_dir1 = input("Enter path to first trajectory directory: ")
    traj_dir2 = input("Enter path to second trajectory directory: ")
    
    # Ask if user wants to compare a third directory
    compare_third = input("Do you want to compare a third trajectory directory? (yes/no): ").lower() == 'yes'
    traj_dir3 = None
    if compare_third:
        traj_dir3 = input("Enter path to third trajectory directory: ")
    
    label1 = input("Enter label for first trajectory (will appear in legend): ")
    label2 = input("Enter label for second trajectory (will appear in legend): ")
    label3 = None
    if compare_third:
        label3 = input("Enter label for third trajectory (will appear in legend): ")
    
    colors = setup_publication_style(figure_size='medium')
    # Define colors for the plots (from the specified color scheme)
    color_dum = colors[0]
    # color_dum2 = colors[1]
    # color_dum3 = colors[2]
    # color_dum4 = colors[3]

    # Assign colors based on label1 content
    label1_lower = label1.lower()
    label2_lower = label2.lower()
    if "chromosome" in label1_lower:
        print(f"chromosome in label1")
        color1 = colors[1]
        color2 = colors[2]
        color3 = colors[3]
    elif "eff" in label2_lower:
        print(f"eff in label1")
        dummy_color = colors[1]
        dummy_color2 = colors[2]
        color1 = colors[3]
        color2 = colors[4]
        color3 = colors[5]
    elif "er" in label1_lower:
        print(f"ER in label1")
        dummy_color = colors[1]
        color1 = colors[2]
        color2 = colors[3]
        color3 = colors[4]
    else:
        # Default colors
        dummy_color = colors[0]
        dummy_color2 = colors[1]
        dummy_color3 = colors[2]
        color1 = colors[3]
        color2 = colors[4]
        color3 = colors[5]    
    
    include_regions = True
    region_suffix = "_region.jsonl"
    if include_regions:
        region_suffix = input(f"Enter region data file suffix (default: {region_suffix}): ") or region_suffix
    
    # Ask if user wants p-value plots
    draw_pvalues = input("Do you want to create p-value significance plots? (yes/no, default: yes): ").lower()
    if not draw_pvalues or draw_pvalues == 'yes' or draw_pvalues == 'y':
        draw_pvalues = True
    else:
        draw_pvalues = False
    
    compare_specific = input("Do you want to compare specific species between directories? (yes/no): ").lower()
    species_mapping = {}
    
    if compare_specific == 'yes':
        print("\nEnter species mapping (how species from dir1 correspond to dir2/dir3)")
        print("Format: species_dir1,species_dir2,species_dir3 (or press Enter to finish)")
        print("Note: For two-directory comparison, use: species_dir1,species_dir2")
        while True:
            mapping = input("Enter mapping (or press Enter to finish): ")
            if not mapping:
                break
            try:
                parts = mapping.split(',')
                if len(parts) >= 2:
                    sp1 = parts[0].strip()
                    sp2 = parts[1].strip()
                    species_mapping[sp1] = sp2
                    # If there's a third directory and third species is specified
                    if compare_third and len(parts) >= 3:
                        sp3 = parts[2].strip()
                        # Store mapping for third directory using a tuple
                        species_mapping[sp1] = (sp2, sp3)
            except ValueError:
                print("Invalid format. Please use: species_dir1,species_dir2 or species_dir1,species_dir2,species_dir3")
    
    while True:
        save_options = f"1 ({label1}), 2 ({label2})"
        if compare_third:
            save_options += f", 3 ({label3})"
        
        save_location = input(f"Save plots under directory: {save_options}? (1/2{'/3' if compare_third else ''}): ").strip()
        if save_location == '1':
            fig_dir = os.path.join(traj_dir1, 'trajectory_comparison/')
            break
        elif save_location == '2':
            fig_dir = os.path.join(traj_dir2, 'trajectory_comparison/')
            break
        elif compare_third and save_location == '3':
            fig_dir = os.path.join(traj_dir3, 'trajectory_comparison/')
            break
        else:
            print(f"Please enter either '1', '2'{' or 3' if compare_third else ''}")
        
    return traj_dir1, traj_dir2, traj_dir3, label1, label2, label3, color1, color2, color3, include_regions, region_suffix, species_mapping, fig_dir, draw_pvalues

# Get user input
traj_dir1, traj_dir2, traj_dir3, label1, label2, label3, color1, color2, color3, include_regions, region_suffix, species_mapping, fig_dir, draw_pvalues = get_user_input()

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# Initialize data structures
data1_species = {}
data1_species_region = {}
data1_species_total = {}
data1_species_total_traj = {}
data1_ode = {}
data2_species = {}
data2_species_region = {}
data2_species_total = {}
data2_species_total_traj = {}
data2_ode = {}
data3_species = {}
data3_species_region = {}
data3_species_total = {}
data3_species_total_traj = {}
data3_ode = {}
rdmeTs = None
odeTs = None
regionTs = None
NAV = None

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

logging.info(f"Comparing trajectories between:")
logging.info(f"Directory 1 ({label1}): {traj_dir1}")
logging.info(f"Directory 2 ({label2}): {traj_dir2}")
if traj_dir3:
    logging.info(f"Directory 3 ({label3}): {traj_dir3}")
logging.info(f"Include region-specific data: {include_regions}")
if include_regions:
    logging.info(f"Region data file suffix: {region_suffix}")
if species_mapping:
    logging.info("Species mapping:")
    for sp1, sp_map in species_mapping.items():
        if isinstance(sp_map, tuple):
            logging.info(f"  {sp1} -> {sp_map[0]} (dir2), {sp_map[1]} (dir3)")
        else:
            logging.info(f"  {sp1} -> {sp_map}")
'''
================================================================================================
This section is for processing the trajecotry data to get the average, min, max, and standard deviation
================================================================================================
'''
total_species_pattern = ['G1', 'G2', 'G3', 'G4', 'G80', 'Grep', 'R1', 'R2', 'R3', 'R4', 'R80', 'Rrep']

# Process directory 1 - try cache first
cached_data1 = load_dir_cache(traj_dir1, fig_dir, include_regions, region_suffix)
if cached_data1:
    logging.info(f"Using cached data for {label1}")
    data1_species, data1_species_region, data1_ode, rdmeTs, odeTs, regionTs, NAV = cached_data1
else:
    logging.info(f"Processing {label1} files")
    files1 = [f for f in os.listdir(traj_dir1) if f.startswith('yeast') and f.endswith('.lm')]
    if not files1:
        logging.error(f"No .lm files found in {traj_dir1}")
        print(f"Error: No .lm files found in {traj_dir1}")
        sys.exit(1)
    traj_suff = "_ode.jsonl"
    
    for traj_file in tqdm(files1, desc=f"Processing {label1} files", unit="file"):
        logging.info(f"Processing {label1} file: {traj_file}")
        region_traj = None
        if include_regions:
            traj, odeTraj, region_traj = get_traj(traj_dir1, traj_file, traj_suff, region_suff=region_suffix)
        else:
            traj, odeTraj, _ = get_traj(traj_dir1, traj_file, traj_suff)
        
        # Store NAV value from the first file (assume consistent across files)
        if NAV is None:
            NAV = 6.022e23 * (traj.reg.cytoplasm.volume + traj.reg.nucleoplasm.volume + traj.reg.plasmaMembrane.volume)
            logging.info(f"NAV value calculated: {NAV}")
        
        curr_rdmeTs, rdmeYs, curr_odeTs, odeYs, curr_regionTs, regionYs = get_data_for_plot(
            traj, odeTraj, region_traj=region_traj, sparse_factor=1)
        
        if rdmeTs is None:
            rdmeTs = curr_rdmeTs
            odeTs = curr_odeTs
            if curr_regionTs is not None:
                regionTs = curr_regionTs

        # Process overall species data
        for species, data in rdmeYs.items():
            if species not in data1_species:
                data1_species[species] = []
            data1_species[species].append(data)

        for species, data in odeYs.items():
            if species not in data1_ode:
                data1_ode[species] = []
            data1_ode[species].append(data)
        
        # Process region-specific data if available
        if regionYs is not None and region_traj is not None:
            regions = region_traj['regions']
            
            # Initialize the nested dictionary structure if needed
            for species, region_data in regionYs.items():
                if species not in data1_species_region:
                    data1_species_region[species] = {}
                
                # Initialize lists for each region if they don't exist
                for region in regions:
                    if region not in data1_species_region[species]:
                        data1_species_region[species][region] = []
                
                # Now append the data
                for i in range(len(regions)):
                    data1_species_region[species][regions[i]].append(regionYs[species][i])
    
    # Save directory 1 cache
    save_dir_cache(traj_dir1, (data1_species, data1_species_region, data1_ode, rdmeTs, odeTs, regionTs, NAV), 
                  fig_dir, include_regions, region_suffix)

# Process directory 2 - try cache first
cached_data2 = load_dir_cache(traj_dir2, fig_dir, include_regions, region_suffix)
if cached_data2:
    logging.info(f"Using cached data for {label2}")
    data2_species, data2_species_region, data2_ode, _, _, _, _ = cached_data2
else:
    logging.info(f"Processing {label2} files")
    files2 = [f for f in os.listdir(traj_dir2) if f.startswith('yeast') and f.endswith('.lm')]
    if not files2:
        logging.error(f"No .lm files found in {traj_dir2}")
        print(f"Error: No .lm files found in {traj_dir2}")
        sys.exit(1)
    traj_suff = "_ode.jsonl"
    
    for traj_file in tqdm(files2, desc=f"Processing {label2} files", unit="file"):
        logging.info(f"Processing {label2} file: {traj_file}")
        region_traj = None
        if include_regions:
            traj, odeTraj, region_traj = get_traj(traj_dir2, traj_file, traj_suff, region_suff=region_suffix)
        else:
            traj, odeTraj, _ = get_traj(traj_dir2, traj_file, traj_suff)
        
        curr_rdmeTs, rdmeYs, curr_odeTs, odeYs, curr_regionTs, regionYs = get_data_for_plot(
            traj, odeTraj, region_traj=region_traj, sparse_factor=1)

        for species, data in rdmeYs.items():
            if species not in data2_species:
                data2_species[species] = []
            data2_species[species].append(data)

        for species, data in odeYs.items():
            if species not in data2_ode:
                data2_ode[species] = []
            data2_ode[species].append(data)
        
        # Process region-specific data if available
        if regionYs is not None and region_traj is not None:
            regions = region_traj['regions']
            
            # Initialize the nested dictionary structure if needed
            for species, region_data in regionYs.items():
                if species not in data2_species_region:
                    data2_species_region[species] = {}
                
                # Initialize lists for each region if they don't exist
                for region in regions:
                    if region not in data2_species_region[species]:
                        data2_species_region[species][region] = []
                
                # Now append the data
                for i in range(len(regions)):
                    data2_species_region[species][regions[i]].append(regionYs[species][i])
    
    # Save directory 2 cache
    save_dir_cache(traj_dir2, (data2_species, data2_species_region, data2_ode, None, None, None, None), 
                  fig_dir, include_regions, region_suffix)

# Process directory 3 - try cache first
if traj_dir3:
    cached_data3 = load_dir_cache(traj_dir3, fig_dir, include_regions, region_suffix)
    if cached_data3:
        logging.info(f"Using cached data for {label3}")
        data3_species, data3_species_region, data3_ode, _, _, _, _ = cached_data3
    else:
        logging.info(f"Processing {label3} files")
        files3 = [f for f in os.listdir(traj_dir3) if f.startswith('yeast') and f.endswith('.lm')]
        if not files3:
            logging.error(f"No .lm files found in {traj_dir3}")
            print(f"Error: No .lm files found in {traj_dir3}")
            sys.exit(1)
        traj_suff = "_ode.jsonl"
        for traj_file in tqdm(files3, desc=f"Processing {label3} files", unit="file"):
            logging.info(f"Processing {label3} file: {traj_file}")
            region_traj = None
            if include_regions:
                traj, odeTraj, region_traj = get_traj(traj_dir3, traj_file, traj_suff, region_suff=region_suffix)
            else:
                traj, odeTraj, _ = get_traj(traj_dir3, traj_file, traj_suff)
            
            curr_rdmeTs, rdmeYs, curr_odeTs, odeYs, curr_regionTs, regionYs = get_data_for_plot(
                traj, odeTraj, region_traj=region_traj, sparse_factor=1)

            for species, data in rdmeYs.items():
                if species not in data3_species:
                    data3_species[species] = []
                data3_species[species].append(data)

            for species, data in odeYs.items():
                if species not in data3_ode:
                    data3_ode[species] = []
                data3_ode[species].append(data)
            
            # Process region-specific data if available
            if regionYs is not None and region_traj is not None:
                regions = region_traj['regions']
                
                # Initialize the nested dictionary structure if needed
                for species, region_data in regionYs.items():
                    if species not in data3_species_region:
                        data3_species_region[species] = {}
                    
                    # Initialize lists for each region if they don't exist
                    for region in regions:
                        if region not in data3_species_region[species]:
                            data3_species_region[species][region] = []
                    
                    # Now append the data
                    for i in range(len(regions)):
                        data3_species_region[species][regions[i]].append(regionYs[species][i])
        
        # Save directory 3 cache
        save_dir_cache(traj_dir3, (data3_species, data3_species_region, data3_ode, None, None, None, None), 
                      fig_dir, include_regions, region_suffix)

'''
================================================================================================
This section is for normalizing region data across all datasets
================================================================================================
'''
# Normalize region data - ensure all datasets have all regions for all species
# This fills missing regions with zero trajectories
if include_regions:
    logging.info("Normalizing region data across all datasets...")
    
    # Collect all unique species and regions across all datasets
    all_species_with_regions = set()
    all_regions_global = set()
    
    for data_region in [data1_species_region, data2_species_region] + ([data3_species_region] if traj_dir3 else []):
        for species, regions in data_region.items():
            all_species_with_regions.add(species)
            all_regions_global.update(regions.keys())
    
    logging.info(f"Found {len(all_species_with_regions)} species with regions: {sorted(all_species_with_regions)}")
    logging.info(f"Found {len(all_regions_global)} unique regions: {sorted(all_regions_global)}")
    
    # Helper function to fill missing regions with zeros
    def fill_missing_regions(species_region_dict, num_trajectories, time_points):
        """Fill missing regions with zero trajectories for all species"""
        for species in all_species_with_regions:
            if species not in species_region_dict:
                species_region_dict[species] = {}
            
            for region in all_regions_global:
                if region not in species_region_dict[species]:
                    # Create zero-filled trajectories
                    # Shape: list of num_trajectories arrays, each with time_points zeros
                    zero_trajectory = [np.zeros(time_points) for _ in range(num_trajectories)]
                    species_region_dict[species][region] = zero_trajectory
                    logging.info(f"  Filled missing region '{region}' for species '{species}' with zeros ({num_trajectories} trajectories)")
    
    # Get the number of trajectories for each dataset from the actual data
    # (files1, files2, files3 may not exist if cache was used)
    num_traj1 = 0
    num_traj2 = 0
    num_traj3 = 0
    
    # Find any species-region combination that has data to determine num trajectories
    if data1_species_region:
        for species, regions in data1_species_region.items():
            for region, trajectories in regions.items():
                if len(trajectories) > 0:
                    num_traj1 = len(trajectories)
                    break
            if num_traj1 > 0:
                break
    
    if data2_species_region:
        for species, regions in data2_species_region.items():
            for region, trajectories in regions.items():
                if len(trajectories) > 0:
                    num_traj2 = len(trajectories)
                    break
            if num_traj2 > 0:
                break
    
    if traj_dir3 and data3_species_region:
        for species, regions in data3_species_region.items():
            for region, trajectories in regions.items():
                if len(trajectories) > 0:
                    num_traj3 = len(trajectories)
                    break
            if num_traj3 > 0:
                break
    
    # Get the number of time points (from regionTs or rdmeTs)
    num_timepoints = len(regionTs) if regionTs is not None else len(rdmeTs)
    
    logging.info(f"Time points: {num_timepoints}")
    logging.info(f"Number of trajectories - {label1}: {num_traj1}, {label2}: {num_traj2}" + 
                 (f", {label3}: {num_traj3}" if traj_dir3 else ""))
    
    # Fill missing regions in all datasets
    logging.info(f"Filling missing regions for {label1}...")
    fill_missing_regions(data1_species_region, num_traj1, num_timepoints)
    
    logging.info(f"Filling missing regions for {label2}...")
    fill_missing_regions(data2_species_region, num_traj2, num_timepoints)
    
    if traj_dir3:
        logging.info(f"Filling missing regions for {label3}...")
        fill_missing_regions(data3_species_region, num_traj3, num_timepoints)
    
    logging.info("Region normalization complete! All datasets now have all regions.")

'''
================================================================================================
This section is for saving the statistics to CSV files
================================================================================================
'''
# Check if CSV files already exist
data1_csv_path = os.path.join(fig_dir, f'{label1}_species_statistics.csv')
data2_csv_path = os.path.join(fig_dir, f'{label2}_species_statistics.csv')
data3_csv_path = None
if traj_dir3:
    data3_csv_path = os.path.join(fig_dir, f'{label3}_species_statistics.csv')

csv_exists = os.path.exists(data1_csv_path) and os.path.exists(data2_csv_path)
if traj_dir3:
    csv_exists = csv_exists and os.path.exists(data3_csv_path)

# Check if CSV files have the correct format (Min/Max columns)
columns_correct = False
if csv_exists:
    try:
        temp_df = pd.read_csv(data1_csv_path)
        columns_correct = 'Min' in temp_df.columns and 'Max' in temp_df.columns
        if not columns_correct:
            logging.info("CSV files exist but have old format (Std instead of Min/Max), regenerating...")
    except Exception as e:
        logging.info(f"Error reading existing CSV: {e}, regenerating...")
        csv_exists = False

if csv_exists and columns_correct:
    logging.info("CSV files already exist with correct format, using cached data")
else:
    logging.info("Processing data and creating new CSV files")
    
    # Calculate and save dir1 statistics
    data1_results = []
    data1_results_region = []
    species_fit = dict()
    for pattern in total_species_pattern:
        species_fit[pattern] = []
    # Process overall species data for dir1
    for species, trajectories in data1_species.items():
        pattern_used = []  # Reset for each species to prevent accumulation bug

        trajectories_array = trajectories_to_array(trajectories)
        avg = np.mean(trajectories_array, axis=0)
        min_val = np.min(trajectories_array, axis=0)
        max_val = np.max(trajectories_array, axis=0)
        
        data1_results.append({
            'Species': f"RDME_{species}",
            'Time': ','.join(map(str, rdmeTs)),
            'Average': ','.join(map(str, avg)),
            'Min': ','.join(map(str, min_val)),
            'Max': ','.join(map(str, max_val))
        })
        
        # record the species that fit the pattern
        for pattern in total_species_pattern:
            dimer_pattern = pattern + 'd'
            if dimer_pattern in species:
                species_fit[pattern].append(species)
                pattern_used.append([pattern, 2])
            elif pattern in species and 'D'+pattern not in species and 'ODE' not in species:
                species_fit[pattern].append(species)
                pattern_used.append([pattern, 1])
        # Sanity check: warn if species matches multiple patterns (potential double-counting)
        if len(pattern_used) > 1:
            logging.warning(f"Dir1: Species '{species}' matches multiple patterns: {pattern_used}. This may cause overcounting.")
        if len(pattern_used) > 0:
            for pattern, multi_factor in pattern_used:
                if pattern not in data1_species_total_traj:
                    data1_species_total_traj[pattern] = trajectories_array * multi_factor
                else:
                    data1_species_total_traj[pattern] += trajectories_array * multi_factor
            
    # process the total trajectories
    for pattern in total_species_pattern:
        print(f"For traj 1 pattern {pattern}, we have species: {species_fit[pattern]}")
    # Add aggregated pattern totals to results (not per-species to avoid duplicates)
    for pattern, trajectories in data1_species_total_traj.items():
        avg = np.mean(trajectories, axis=0)
        min_val = np.min(trajectories, axis=0)
        max_val = np.max(trajectories, axis=0)
        # Sanity check: validate aggregated data
        if np.any(np.isnan(avg)) or np.any(np.isnan(min_val)) or np.any(np.isnan(max_val)):
            logging.error(f"Dir1: NaN values detected in aggregated pattern '{pattern}'!")
        max_avg = np.max(avg)
        if max_avg > 1e10:  # Arbitrary large threshold
            logging.warning(f"Dir1: Very large value detected in pattern '{pattern}': max_avg={max_avg:.2e}")
        logging.info(f"Dir1 pattern '{pattern}': max={max_avg:.2e}, contributing species: {species_fit[pattern]}")
        data1_results.append({
            'Species': f"RDME_{pattern}_total",
            'Time': ','.join(map(str, rdmeTs)),
            'Average': ','.join(map(str, avg)),
            'Min': ','.join(map(str, min_val)),
            'Max': ','.join(map(str, max_val))
        })

    # Process region-specific data for dir1
    for species, regions in data1_species_region.items():
        for region, trajectories in regions.items():
            trajectories_array = trajectories_to_array(trajectories)
            avg = np.mean(trajectories_array, axis=0)
            min_val = np.min(trajectories_array, axis=0)
            max_val = np.max(trajectories_array, axis=0)
            
            # Store region-specific results
            data1_results_region.append({
                'Species': species,
                'Region': region,
                'Time': ','.join(map(str, regionTs if regionTs is not None else rdmeTs)),
                'Average': ','.join(map(str, avg)),
                'Min': ','.join(map(str, min_val)),
                'Max': ','.join(map(str, max_val))
            })
            
            # Also store in main results with a special naming convention
            data1_results.append({
                'Species': f"RDME_{species}_{region}",
                'Time': ','.join(map(str, regionTs if regionTs is not None else rdmeTs)),
                'Average': ','.join(map(str, avg)),
                'Min': ','.join(map(str, min_val)),
                'Max': ','.join(map(str, max_val))
            })

    # Process ODE species data for dir1
    for species, trajectories in data1_ode.items():
        trajectories_array = trajectories_to_array(trajectories)
        avg = np.mean(trajectories_array, axis=0)
        min_val = np.min(trajectories_array, axis=0)
        max_val = np.max(trajectories_array, axis=0)
        
        data1_results.append({
            'Species': f"ODE_{species}",
            'Time': ','.join(map(str, odeTs)),
            'Average': ','.join(map(str, avg)),
            'Min': ','.join(map(str, min_val)),
            'Max': ','.join(map(str, max_val))
        })

    # Calculate and save dir2 statistics
    data2_results = []
    data2_results_region = []
    species_fit = dict()
    for pattern in total_species_pattern:
        species_fit[pattern] = []
    # Process overall species data for dir2
    for species, trajectories in data2_species.items():
        pattern_used = []  # Reset for each species to prevent accumulation bug
        
        trajectories_array = trajectories_to_array(trajectories)
        avg = np.mean(trajectories_array, axis=0)
        min_val = np.min(trajectories_array, axis=0)
        max_val = np.max(trajectories_array, axis=0)
        
        data2_results.append({
            'Species': f"RDME_{species}",
            'Time': ','.join(map(str, rdmeTs)),
            'Average': ','.join(map(str, avg)),
            'Min': ','.join(map(str, min_val)),
            'Max': ','.join(map(str, max_val))
        })
        for pattern in total_species_pattern:
            dimer_pattern = pattern + 'd'
            if dimer_pattern in species:
                species_fit[pattern].append(species)
                pattern_used.append([pattern, 2])
            elif pattern in species and 'D'+pattern not in species and 'ODE' not in species:
                species_fit[pattern].append(species)
                pattern_used.append([pattern, 1])
        # Sanity check: warn if species matches multiple patterns (potential double-counting)
        if len(pattern_used) > 1:
            logging.warning(f"Dir2: Species '{species}' matches multiple patterns: {pattern_used}. This may cause overcounting.")
        if len(pattern_used) > 0:
            for pattern, multi_factor in pattern_used:
                if pattern not in data2_species_total_traj.keys():
                    data2_species_total_traj[pattern] = trajectories_array * multi_factor
                else:
                    data2_species_total_traj[pattern] += trajectories_array * multi_factor
    # process the total trajectories
    for pattern in total_species_pattern:
        print(f"For traj 2 pattern {pattern}, we have species: {species_fit[pattern]}")
    for pattern, trajectories in data2_species_total_traj.items():
        avg = np.mean(trajectories, axis=0)
        min_val = np.min(trajectories, axis=0)
        max_val = np.max(trajectories, axis=0)
        # Sanity check: validate aggregated data
        if np.any(np.isnan(avg)) or np.any(np.isnan(min_val)) or np.any(np.isnan(max_val)):
            logging.error(f"Dir2: NaN values detected in aggregated pattern '{pattern}'!")
        max_avg = np.max(avg)
        if max_avg > 1e10:  # Arbitrary large threshold
            logging.warning(f"Dir2: Very large value detected in pattern '{pattern}': max_avg={max_avg:.2e}")
        logging.info(f"Dir2 pattern '{pattern}': max={max_avg:.2e}, contributing species: {species_fit[pattern]}")
        data2_results.append({
            'Species': f"RDME_{pattern}_total",
            'Time': ','.join(map(str, rdmeTs)),
            'Average': ','.join(map(str, avg)),
            'Min': ','.join(map(str, min_val)),
            'Max': ','.join(map(str, max_val))
        })
    # Process region-specific data for dir2
    for species, regions in data2_species_region.items():
        for region, trajectories in regions.items():
            trajectories_array = trajectories_to_array(trajectories)
            avg = np.mean(trajectories_array, axis=0)
            min_val = np.min(trajectories_array, axis=0)
            max_val = np.max(trajectories_array, axis=0)
            
            # Store region-specific results
            data2_results_region.append({
                'Species': species,
                'Region': region,
                'Time': ','.join(map(str, regionTs if regionTs is not None else rdmeTs)),
                'Average': ','.join(map(str, avg)),
                'Min': ','.join(map(str, min_val)),
                'Max': ','.join(map(str, max_val))
            })
            
            # Also store in main results with a special naming convention
            data2_results.append({
                'Species': f"RDME_{species}_{region}",
                'Time': ','.join(map(str, regionTs if regionTs is not None else rdmeTs)),
                'Average': ','.join(map(str, avg)),
                'Min': ','.join(map(str, min_val)),
                'Max': ','.join(map(str, max_val))
            })

    # Process ODE species data for dir2
    for species, trajectories in data2_ode.items():
        trajectories_array = trajectories_to_array(trajectories)
        avg = np.mean(trajectories_array, axis=0)
        min_val = np.min(trajectories_array, axis=0)
        max_val = np.max(trajectories_array, axis=0)
        
        data2_results.append({
            'Species': f"ODE_{species}",
            'Time': ','.join(map(str, odeTs)),
            'Average': ','.join(map(str, avg)),
            'Min': ','.join(map(str, min_val)),
            'Max': ','.join(map(str, max_val))
        })

    # Calculate and save dir3 statistics if it exists
    data3_results = []
    data3_results_region = []
    
    if traj_dir3:
        species_fit = dict()
        for pattern in total_species_pattern:
            species_fit[pattern] = []
        # Process overall species data for dir3
        for species, trajectories in data3_species.items():
            pattern_used = []  # Reset for each species to prevent accumulation bug
            trajectories_array = trajectories_to_array(trajectories)
            avg = np.mean(trajectories_array, axis=0)
            min_val = np.min(trajectories_array, axis=0)
            max_val = np.max(trajectories_array, axis=0)
            
            data3_results.append({
                'Species': f"RDME_{species}",
                'Time': ','.join(map(str, rdmeTs)),
                'Average': ','.join(map(str, avg)),
                'Min': ','.join(map(str, min_val)),
                'Max': ','.join(map(str, max_val))
            })
            
            for pattern in total_species_pattern:
                dimer_pattern = pattern + 'd'
                if dimer_pattern in species:
                    species_fit[pattern].append(species)
                    pattern_used.append([pattern, 2])
                elif pattern in species and 'D'+pattern not in species and 'ODE' not in species:
                    species_fit[pattern].append(species)
                    pattern_used.append([pattern, 1])
            # Sanity check: warn if species matches multiple patterns (potential double-counting)
            if len(pattern_used) > 1:
                logging.warning(f"Dir3: Species '{species}' matches multiple patterns: {pattern_used}. This may cause overcounting.")
            if len(pattern_used) > 0:
                for pattern, multi_factor in pattern_used:
                    if pattern not in data3_species_total_traj.keys():
                        data3_species_total_traj[pattern] = trajectories_array * multi_factor
                    else:
                        data3_species_total_traj[pattern] += trajectories_array * multi_factor
        # process the total trajectories
        for pattern in total_species_pattern:
            print(f"For traj 3 pattern {pattern}, we have species: {species_fit[pattern]}")
        for pattern, trajectories in data3_species_total_traj.items():
            avg = np.mean(trajectories, axis=0)
            min_val = np.min(trajectories, axis=0)
            max_val = np.max(trajectories, axis=0)
            # Sanity check: validate aggregated data
            if np.any(np.isnan(avg)) or np.any(np.isnan(min_val)) or np.any(np.isnan(max_val)):
                logging.error(f"Dir3: NaN values detected in aggregated pattern '{pattern}'!")
            max_avg = np.max(avg)
            if max_avg > 1e10:  # Arbitrary large threshold
                logging.warning(f"Dir3: Very large value detected in pattern '{pattern}': max_avg={max_avg:.2e}")
            logging.info(f"Dir3 pattern '{pattern}': max={max_avg:.2e}, contributing species: {species_fit[pattern]}")
            data3_results.append({
                'Species': f"RDME_{pattern}_total",
                'Time': ','.join(map(str, rdmeTs)),
                'Average': ','.join(map(str, avg)),
                'Min': ','.join(map(str, min_val)),
                'Max': ','.join(map(str, max_val))
            })

        # Process region-specific data for dir3
        for species, regions in data3_species_region.items():
            for region, trajectories in regions.items():
                trajectories_array = trajectories_to_array(trajectories)
                avg = np.mean(trajectories_array, axis=0)
                min_val = np.min(trajectories_array, axis=0)
                max_val = np.max(trajectories_array, axis=0)
                
                # Store region-specific results
                data3_results_region.append({
                    'Species': species,
                    'Region': region,
                    'Time': ','.join(map(str, regionTs if regionTs is not None else rdmeTs)),
                    'Average': ','.join(map(str, avg)),
                    'Min': ','.join(map(str, min_val)),
                    'Max': ','.join(map(str, max_val))
                })
                
                # Also store in main results with a special naming convention
                data3_results.append({
                    'Species': f"RDME_{species}_{region}",
                    'Time': ','.join(map(str, regionTs if regionTs is not None else rdmeTs)),
                    'Average': ','.join(map(str, avg)),
                    'Min': ','.join(map(str, min_val)),
                    'Max': ','.join(map(str, max_val))
                })

        # Process ODE species data for dir3
        for species, trajectories in data3_ode.items():
            trajectories_array = trajectories_to_array(trajectories)
            avg = np.mean(trajectories_array, axis=0)
            min_val = np.min(trajectories_array, axis=0)
            max_val = np.max(trajectories_array, axis=0)
            
            data3_results.append({
                'Species': f"ODE_{species}",
                'Time': ','.join(map(str, odeTs)),
                'Average': ','.join(map(str, avg)),
                'Min': ','.join(map(str, min_val)),
                'Max': ','.join(map(str, max_val))
            })

    # Save to CSV files
    data1_df = pd.DataFrame(data1_results)
    data2_df = pd.DataFrame(data2_results)
    
    data1_df.to_csv(data1_csv_path, index=False)
    data2_df.to_csv(data2_csv_path, index=False)
    
    logging.info(f"{label1} statistics saved to: {data1_csv_path}")
    logging.info(f"{label2} statistics saved to: {data2_csv_path}")


    # If region-specific data exists, save it separately
    if data1_results_region:
        data1_region_df = pd.DataFrame(data1_results_region)
        data1_region_csv_path = os.path.join(fig_dir, f'{label1}_region_statistics.csv')
        data1_region_df.to_csv(data1_region_csv_path, index=False)
        logging.info(f"{label1} region statistics saved to: {data1_region_csv_path}")
    
    if data2_results_region:
        data2_region_df = pd.DataFrame(data2_results_region)
        data2_region_csv_path = os.path.join(fig_dir, f'{label2}_region_statistics.csv')
        data2_region_df.to_csv(data2_region_csv_path, index=False)
        logging.info(f"{label2} region statistics saved to: {data2_region_csv_path}")
    
    # Save dir3 statistics to CSV if it exists
    if traj_dir3:
        data3_df = pd.DataFrame(data3_results)
        data3_df.to_csv(data3_csv_path, index=False)
        logging.info(f"{label3} statistics saved to: {data3_csv_path}")
    
        # If region-specific data exists, save it separately
        if data3_results_region:
            data3_region_df = pd.DataFrame(data3_results_region)
            data3_region_csv_path = os.path.join(fig_dir, f'{label3}_region_statistics.csv')
            data3_region_df.to_csv(data3_region_csv_path, index=False)
            logging.info(f"{label3} region statistics saved to: {data3_region_csv_path}")

'''
================================================================================================
This section is for loading the csvs, and create the plots 
================================================================================================
'''
# Read the saved statistics
data1_df = pd.read_csv(data1_csv_path)
data2_df = pd.read_csv(data2_csv_path)
# Load data3 statistics if it exists
if traj_dir3:
    data3_df = pd.read_csv(data3_csv_path)
    # logging.info(f"Available species in {label3}: {data3_df['Species'].tolist()}")

# Function to convert string of comma-separated values to numpy array
def str_to_array(s):
    return np.array([float(x) for x in s.split(',')])

def calculate_pvalue_timeseries(data1_trajectories, data2_trajectories, test_type='ttest', species_name=None):
    """
    Calculate p-values for each time point comparing two sets of trajectories.

    Parameters:
    data1_trajectories: list of arrays, each array is a trajectory from dataset 1
    data2_trajectories: list of arrays, each array is a trajectory from dataset 2
    test_type: 'ttest' for t-test, 'ks' for Kolmogorov-Smirnov test, 'chi2' for chi-square test
    species_name: name of the species to determine if chi-square test should be used for binary data

    Returns:
    p_values: array of p-values for each time point
    actual_test_type: the actual test type used (may differ from input due to automatic detection)
    """
    data1_array = trajectories_to_array(data1_trajectories)
    data2_array = trajectories_to_array(data2_trajectories)
    
    # Handle case where arrays might be empty or 1D
    if data1_array.ndim < 2 or data2_array.ndim < 2:
        return np.array([np.nan]), test_type
    
    # Use minimum time points from both arrays
    n_timepoints = min(data1_array.shape[1], data2_array.shape[1])
    data1_array = data1_array[:, :n_timepoints]
    data2_array = data2_array[:, :n_timepoints]
    p_values = np.zeros(n_timepoints)

    # Automatically determine if we should use chi-square test for binary RDME_DG species
    use_chi2 = (species_name is not None and species_name.startswith('RDME_DG'))
    actual_test_type = 'chi2' if use_chi2 else test_type
    
    for t in range(n_timepoints):
        values1 = data1_array[:, t]
        values2 = data2_array[:, t]
        
        # Check for problematic cases that cause NaN
        try:
            # Remove any NaN or infinite values
            values1 = values1[np.isfinite(values1)]
            values2 = values2[np.isfinite(values2)]
            
            # Check if we have enough data points
            if len(values1) < 2 or len(values2) < 2:
                p_values[t] = np.nan
                continue
            
            # Check for zero variance (all values identical)
            if np.var(values1) == 0 and np.var(values2) == 0:
                # If both groups have identical values
                if np.mean(values1) == np.mean(values2):
                    p_values[t] = 1.0  # No difference
                else:
                    p_values[t] = 0.0  # Perfect difference
                continue
            elif np.var(values1) == 0 or np.var(values2) == 0:
                # If only one group has zero variance, use a simple comparison
                if actual_test_type == 'ttest':
                    # Add tiny noise to break zero variance
                    epsilon = 1e-10
                    if np.var(values1) == 0:
                        values1 = values1 + np.random.normal(0, epsilon, len(values1))
                    if np.var(values2) == 0:
                        values2 = values2 + np.random.normal(0, epsilon, len(values2))

                    # Now perform t-test
                    _, p_val = ttest_ind(values1, values2, equal_var=False)
                    p_values[t] = p_val if np.isfinite(p_val) else 1.0
                else:
                    # For KS test, we can still compute it
                    _, p_val = ks_2samp(values1, values2)
                    p_values[t] = p_val if np.isfinite(p_val) else np.nan
                continue
            
            # Perform the statistical test
            if use_chi2:
                # For binary RDME_DG species, use chi-square test
                # Create contingency table for binary outcomes (0 or 1)
                # Count 0s and 1s in each group
                count1_0 = np.sum(values1 == 0)
                count1_1 = np.sum(values1 == 1)
                count2_0 = np.sum(values2 == 0)
                count2_1 = np.sum(values2 == 1)

                # Create 2x2 contingency table
                contingency_table = np.array([[count1_0, count1_1],
                                            [count2_0, count2_1]])

                # Check if contingency table is valid (no row/column sums to zero)
                if np.any(contingency_table.sum(axis=0) == 0) or np.any(contingency_table.sum(axis=1) == 0):
                    p_values[t] = np.nan
                    continue

                # Perform chi-square test
                chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
            elif actual_test_type == 'ttest':
                _, p_val = ttest_ind(values1, values2, equal_var=False)  # Use Welch's t-test
            elif actual_test_type == 'ks':
                _, p_val = ks_2samp(values1, values2)
            else:
                raise ValueError("actual_test_type must be 'ttest', 'ks', or 'chi2'")
            
            # Check if p-value is valid
            if np.isfinite(p_val):
                p_values[t] = p_val
            else:
                p_values[t] = np.nan
                
        except Exception as e:
            # If any error occurs, set to NaN and log it
            p_values[t] = np.nan
            if t < 5:  # Only log first few errors to avoid spam
                print(f"Warning: Error calculating p-value at timepoint {t}: {e}")
    
    return p_values, actual_test_type

def find_alpha_05_crossings(time, p_values):
    """
    Find time points where p-values cross α=0.05, merging close crossings.
    """
    crossings = []
    alpha = 0.05
    
    for i in range(1, len(p_values)):
        prev_val = p_values[i-1]
        curr_val = p_values[i]
        
        # Check if line crosses α=0.05 (in either direction)
        if (prev_val > alpha and curr_val < alpha) or (prev_val < alpha and curr_val > alpha):
            # Linear interpolation to find exact crossing time
            t_cross = time[i-1] + (time[i] - time[i-1]) * (alpha - prev_val) / (curr_val - prev_val)
            crossings.append(t_cross)
    
    # Merge crossings that are within 1 minute of each other
    if len(crossings) <= 1:
        return crossings
    
    merged_crossings = []
    current_group = [crossings[0]]
    
    for i in range(1, len(crossings)):
        # If current crossing is within 1 minute of the last one in the group
        if crossings[i] - current_group[-1] <= 1.0:
            current_group.append(crossings[i])
        else:
            # Finalize current group by taking the mean
            merged_crossings.append(sum(current_group) / len(current_group))
            current_group = [crossings[i]]
    
    # Don't forget the last group
    merged_crossings.append(sum(current_group) / len(current_group))
    
    return merged_crossings

def create_pvalue_plot(time, p_values, species_name, label1, label2, label3=None, fig_dir=None, 
                      significance_levels=[0.001, 0.01, 0.05], test_type='ttest'):
    """
    Create a separate p-value significance plot with α=0.05 crossing markers and specific timepoint annotations.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot p-values over time
    ax.plot(time, p_values, 'k-', linewidth=2, label='p-value')
    
    # Add horizontal lines for significance levels
    colors = ['red', 'orange', 'yellow']
    for i, sig_level in enumerate(significance_levels):
        ax.axhline(y=sig_level, color=colors[i], linestyle='--', alpha=0.7, 
                  label=f'p = {sig_level}')
    
    # Find and mark α=0.05 crossings
    crossings = find_alpha_05_crossings(time, p_values)
    
    for i, t_cross in enumerate(crossings):
        # Add vertical line at crossing time
        ax.axvline(x=t_cross, color='green', linestyle='--', alpha=0.8, linewidth=2)
        
        # Add text annotation on the plot
        y_pos = 0.05 * (2 ** (i % 3))  # Stagger heights for multiple crossings
        ax.annotate(f'Cross: {t_cross:.1f}min', 
                   xy=(t_cross, y_pos), 
                   xytext=(10, 20), 
                   textcoords='offset points',
                   ha='left', va='bottom',
                   fontsize=9, color='green', weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='green', lw=1))
    
    # Mark specific timepoints (10min, 30min, 60min) with p-values
    specific_times = [10, 30, 60]
    marker_colors = ['blue', 'purple', 'red']
    
    for spec_time, marker_color in zip(specific_times, marker_colors):
        # Find closest time index
        time_idx = np.argmin(np.abs(time - spec_time))
        actual_time = time[time_idx]
        p_val_at_time = p_values[time_idx]
        
        # Only mark if the time is within reasonable range
        if abs(actual_time - spec_time) <= 2:  # Within 2 minutes
            # Add marker point - made larger and more prominent
            ax.scatter(actual_time, p_val_at_time, color=marker_color, s=120, 
                      marker='o', edgecolor='black', linewidth=2, zorder=10, alpha=0.9)
            
            # Add vertical line at the timepoint for better visibility
            ax.axvline(x=actual_time, color=marker_color, linestyle=':', alpha=0.6, linewidth=2)
            
            # Add text annotation - made larger and more visible
            ax.annotate(f't={actual_time:.0f}min\np={p_val_at_time:.2e}', 
                       xy=(actual_time, p_val_at_time), 
                       xytext=(20, 25), 
                       textcoords='offset points',
                       ha='left', va='bottom',
                       fontsize=10, color=marker_color, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor=marker_color),
                       arrowprops=dict(arrowstyle='->', color=marker_color, lw=2))
    
    # Print crossing times and specific timepoint p-values to console
    if crossings:
        print(f"α=0.05 crossings for {species_name}: {[f'{t:.1f}min' for t in crossings]}")
    
    print(f"P-values at specific times for {species_name}:")
    for spec_time in specific_times:
        time_idx = np.argmin(np.abs(time - spec_time))
        actual_time = time[time_idx]
        p_val_at_time = p_values[time_idx]
        if abs(actual_time - spec_time) <= 2:
            print(f"  t={actual_time:.0f}min: p={p_val_at_time:.2e}")
    
    # Fill areas below significance levels
    for i, sig_level in enumerate(significance_levels):
        if i == 0:  # Most stringent level
            ax.fill_between(time, 0, sig_level, where=(p_values <= sig_level), 
                          color=colors[i], alpha=0.2, interpolate=True)
        else:
            prev_level = significance_levels[i-1]
            ax.fill_between(time, prev_level, sig_level, where=(p_values <= sig_level) & (p_values > prev_level),
                          color=colors[i], alpha=0.2, interpolate=True)
    
    # Set y-axis to log scale for better visualization
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1)
    
    # Labels and formatting
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('p-value (log scale)')
    
    # Create title based on comparison
    if label3:
        title = f'Statistical Significance: {species_name}\n({label1} vs {label2} vs {label3})'
    else:
        title = f'Statistical Significance: {species_name}\n({label1} vs {label2})'
    
    ax.set_title(title, fontsize=12)
    ax.legend(framealpha=0.3, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add text box with test information
    test_name = 'T-test' if test_type == 'ttest' else 'Kolmogorov-Smirnov test' if test_type == 'ks' else 'Chi-square test'
    ax.text(0.02, 0.98, f'Test: {test_name}', transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    if fig_dir:
        clean_species_name = species_name.replace(':', '_').replace('/', '_')
        filename = f'{clean_species_name}_pvalue_significance.png'
        fig_path = os.path.join(fig_dir, filename)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved p-value plot: {filename}")
    
    plt.close()

def create_three_way_pvalue_plot(time, p_values_12, p_values_13, p_values_23, species_name, 
                                label1, label2, label3, fig_dir=None, 
                                significance_levels=[0.001, 0.01, 0.05], test_type='ttest'):
    """
    Create a three-way p-value comparison plot for three datasets with α=0.05 crossing markers.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot p-values for each pairwise comparison
    ax.plot(time, p_values_12, label=f'{label1} vs {label2}', linewidth=2, color='#0072B2')
    ax.plot(time, p_values_13, label=f'{label1} vs {label3}', linewidth=2, color='#D55E00')  
    ax.plot(time, p_values_23, label=f'{label2} vs {label3}', linewidth=2, color='#009E73')
    
    # Add horizontal lines for significance levels
    line_colors = ['red', 'orange', 'yellow']
    for i, sig_level in enumerate(significance_levels):
        ax.axhline(y=sig_level, color=line_colors[i], linestyle='--', alpha=0.7, 
                  label=f'p = {sig_level}')
    
    # Find and mark α=0.05 crossings for each comparison
    comparison_labels = [f'{label1} vs {label2}', f'{label1} vs {label3}', f'{label2} vs {label3}']
    p_value_sets = [p_values_12, p_values_13, p_values_23]
    crossing_colors = ['#0072B2', '#D55E00', '#009E73']
    
    for p_values, comp_label, color in zip(p_value_sets, comparison_labels, crossing_colors):
        crossings = find_alpha_05_crossings(time, p_values)
        
        for t_cross in crossings:
            # Add vertical line at crossing time with comparison-specific color
            ax.axvline(x=t_cross, color=color, linestyle=':', alpha=0.6, linewidth=1.5)
            
            # Mark the time on x-axis with comparison-specific color
            ax.annotate(f'{t_cross:.1f}', 
                       xy=(t_cross, 0.05), 
                       xytext=(0, -15), 
                       textcoords='offset points',
                       ha='center', va='top',
                       fontsize=8, color=color, weight='bold')
        
        # Print crossing times to console
        if crossings:
            print(f"α=0.05 crossings for {comp_label}: {[f'{t:.1f}min' for t in crossings]}")
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1)
    
    # Labels and formatting
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('p-value (log scale)')
    ax.set_title(f'Pairwise Statistical Significance: {species_name}', fontsize=12)
    ax.legend(framealpha=0.3, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add test information
    test_name = 'T-test' if test_type == 'ttest' else 'Kolmogorov-Smirnov test'
    ax.text(0.02, 0.98, f'Test: {test_name}', transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    if fig_dir:
        clean_species_name = species_name.replace(':', '_').replace('/', '_')
        filename = f'{clean_species_name}_three_way_pvalue_significance.png'
        fig_path = os.path.join(fig_dir, filename)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved three-way p-value plot: {filename}")
    
    plt.close()

# Debug: logging.info available species
# logging.info(f"Available species in {label1}: {data1_df['Species'].tolist()}")
# logging.info(f"Available species in {label2}: {data2_df['Species'].tolist()}")

# Get unique species names directly from the CSV
# Handle species mapping if provided
# Modify species mapping handling to accommodate three directories
if species_mapping:
    # Create a list of comparable species tuples
    comparable_species = []
    for sp1 in data1_df['Species'].unique():
        # Check if this species has a mapping
        if sp1 in species_mapping:
            sp_map = species_mapping[sp1]
            if isinstance(sp_map, tuple) and traj_dir3:  # It's a mapping for three directories
                sp2, sp3 = sp_map
                # Check if mapped species exist in respective dataframes
                if sp2 in data2_df['Species'].unique() and sp3 in data3_df['Species'].unique():
                    comparable_species.append((sp1, sp2, sp3))
            elif sp_map in data2_df['Species'].unique():  # It's a mapping for two directories
                comparable_species.append((sp1, sp_map, None))
        # If no mapping, look for the same species name
        elif sp1 in data2_df['Species'].unique() and (not traj_dir3 or sp1 in data3_df['Species'].unique()):
            if traj_dir3:
                comparable_species.append((sp1, sp1, sp1))
            else:
                comparable_species.append((sp1, sp1, None))
    
    logging.info(f"\nComparable species groups: {comparable_species}")
else:
    # Without mapping, compare species with the same name
    if traj_dir3:
        unique_species = set(data1_df['Species']) & set(data2_df['Species']) & set(data3_df['Species'])
        comparable_species = [(sp, sp, sp) for sp in unique_species]
    else:
        unique_species = set(data1_df['Species']) & set(data2_df['Species'])
        comparable_species = [(sp, sp, None) for sp in unique_species]
    
    logging.info(f"\nCommon species: {unique_species}")

# After processing the data, create a list of all regions found
all_regions = set()
if include_regions:
    # Extract regions from data1_species_region and data2_species_region
    for species, regions in data1_species_region.items():
        all_regions.update(regions.keys())
    for species, regions in data2_species_region.items():
        all_regions.update(regions.keys())
    
    logging.info(f"Found regions: {sorted(list(all_regions))}")

# Plot settings - use publication style
colors = setup_publication_style(figure_size='medium', dpi=300)

# Create plots for each comparable species pair
for sp_group in comparable_species:
    if len(sp_group) == 3:
        sp1, sp2, sp3 = sp_group
    else:
        sp1, sp2 = sp_group
        sp3 = None
    
    # Skip region-specific entries by checking if any region name appears in the species name
    is_region_specific = False
    for region in all_regions:
        if f"_{region}" in sp1 or f"_{region}" in sp2 or (sp3 and f"_{region}" in sp3):
            is_region_specific = True
            break
    
    if is_region_specific:
        logging.info(f"Skipping region-specific entry: {sp1} vs {sp2}{f' vs {sp3}' if sp3 else ''}")
        continue
    
    fig, ax = plt.subplots()
    
    # Safely get data for first directory
    data1_species_data = data1_df[data1_df['Species'] == sp1]
    
    # Safely get data for second directory
    data2_species_data = data2_df[data2_df['Species'] == sp2]
    
    # Safely get data for third directory if applicable
    data3_species_data = pd.DataFrame()
    if sp3 and traj_dir3:
        data3_species_data = data3_df[data3_df['Species'] == sp3]
    
    if len(data1_species_data) == 0 or len(data2_species_data) == 0 or (sp3 and traj_dir3 and len(data3_species_data) == 0):
        logging.info(f"Skipping {sp1}/{sp2}{f'/{sp3}' if sp3 else ''} - data not found")
        plt.close(fig)
        continue
        
    data1_row = data1_species_data.iloc[0]
    data2_row = data2_species_data.iloc[0]
    data3_row = None
    if sp3 and traj_dir3 and len(data3_species_data) > 0:
        data3_row = data3_species_data.iloc[0]
    
    time = str_to_array(data1_row['Time'])
    data1_avg = str_to_array(data1_row['Average'])
    data1_min = str_to_array(data1_row['Min'])
    data1_max = str_to_array(data1_row['Max'])
    data2_avg = str_to_array(data2_row['Average'])
    data2_min = str_to_array(data2_row['Min'])
    data2_max = str_to_array(data2_row['Max'])
    data3_avg = None
    data3_min = None
    data3_max = None
    if data3_row is not None:
        data3_avg = str_to_array(data3_row['Average'])
        data3_min = str_to_array(data3_row['Min'])
        data3_max = str_to_array(data3_row['Max'])
    
    # Check if all trajectories are all zeros
    all_zeros = np.all(data1_avg == 0) and np.all(data2_avg == 0)
    if data3_avg is not None:
        all_zeros = all_zeros and np.all(data3_avg == 0)
        
    if all_zeros:
        logging.info(f"All trajectories are all zeros, skipping plot")
        plt.close(fig)
        continue
    
    # Get display name (remove prefix if needed)
    display_name1 = sp1.split('_', 1)[1] if '_' in sp1 else sp1
    display_name2 = sp2.split('_', 1)[1] if '_' in sp2 else sp2
    display_name3 = sp3.split('_', 1)[1] if sp3 and '_' in sp3 else sp3
    
    # Replace any subsequent underscores with colons
    display_name1 = display_name1.replace('_', ':')
    display_name2 = display_name2.replace('_', ':')
    if display_name3:
        display_name3 = display_name3.replace('_', ':')
    
    # Determine legend labels and filename
    if display_name1 == display_name2 and (not display_name3 or display_name1 == display_name3):
        legend_label1 = f'{label1}'
        legend_label2 = f'{label2}'
        legend_label3 = f'{label3}' if label3 else None
        plot_title = f'{display_name1} Comparison'
        output_filename = f'{sp1}_comparison.png'
    else:
        legend_label1 = f'{label1} - {display_name1}'
        legend_label2 = f'{label2} - {display_name2}'
        legend_label3 = f'{label3} - {display_name3}' if label3 and display_name3 else None
        if display_name3:
            plot_title = f'{display_name1} vs {display_name2} vs {display_name3} Comparison'
            output_filename = f'{sp1}_vs_{sp2}_vs_{sp3}_comparison.png'
        else:
            plot_title = f'{display_name1} vs {display_name2} Comparison'
            output_filename = f'{sp1}_vs_{sp2}_comparison.png'
    
     # Check if this is a gene species (contains DG) - if so, don't plot min/max
    is_gene_species = "DG" in sp1 or "DG" in sp2 or (sp3 and "DG" in sp3)
    
    # Plot dir1 data with specified color
    ax.plot(time, data1_avg, label=legend_label1, linestyle='-', color=color1)
    if not is_gene_species:
        ax.fill_between(time, data1_min, data1_max, alpha=0.1, color=color1)
    
    # Plot dir2 data with specified color
    ax.plot(time, data2_avg, label=legend_label2, linestyle='-', color=color2)
    if not is_gene_species:
        ax.fill_between(time, data2_min, data2_max, alpha=0.1, color=color2)
    
    # Plot dir3 data if it exists
    if data3_avg is not None and legend_label3:
        ax.plot(time, data3_avg, label=legend_label3, linestyle='-', color=color3)
        if not is_gene_species:
            ax.fill_between(time, data3_min, data3_max, alpha=0.1, color=color3)
    
    # Customize plot
    ax.set_xlabel('Time (min)')
    if sp1.startswith('RDME_DG') or sp2.startswith('RDME_DG') or (sp3 and sp3.startswith('RDME_DG')):
        ax.set_ylabel('Probability')
    else:
        ax.set_ylabel('Counts')
    # ax.set_title(plot_title)  # Uncomment if you want titles
    # ax.legend(framealpha=0.3, loc='best')
    ax.grid(False)
    
    # Save figure
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, output_filename)
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved plot: {output_filename}")
    plt.close()
    
    # Create p-value significance plots (if enabled)
    if draw_pvalues:
        # Get the raw trajectory data for statistical testing
        
        # Clean species names (remove prefixes)
        sp1_clean = sp1.replace('RDME_', '').replace('ODE_', '')
        sp2_clean = sp2.replace('RDME_', '').replace('ODE_', '')
        sp3_clean = sp3.replace('RDME_', '').replace('ODE_', '') if sp3 else None
        
        # Determine data sources based on species type
        data1_trajectories = None
        data2_trajectories = None
        data3_trajectories = None
        
        # Get trajectories for first two datasets
        if sp1.startswith('RDME_') and sp1_clean in data1_species:
            data1_trajectories = data1_species[sp1_clean]
        elif sp1.startswith('ODE_') and sp1_clean in data1_ode:
            data1_trajectories = data1_ode[sp1_clean]
        
        if sp2.startswith('RDME_') and sp2_clean in data2_species:
            data2_trajectories = data2_species[sp2_clean]
        elif sp2.startswith('ODE_') and sp2_clean in data2_ode:
            data2_trajectories = data2_ode[sp2_clean]
        
        # Get trajectories for third dataset if applicable
        if sp3 and traj_dir3:
            if sp3.startswith('RDME_') and sp3_clean in data3_species:
                data3_trajectories = data3_species[sp3_clean]
            elif sp3.startswith('ODE_') and sp3_clean in data3_ode:
                data3_trajectories = data3_ode[sp3_clean]
        
        # Only proceed if we have data for the required comparisons
        if data1_trajectories is not None and data2_trajectories is not None:
            # Create display name for p-value plots - preserve original species names with prefixes
            if sp1 == sp2 and (not sp3 or sp1 == sp3):
                # Same species comparison across different datasets
                pvalue_species_name = sp1  # Keep the original species name with prefix
            else:
                # Different species comparison
                if sp3:
                    pvalue_species_name = f'{sp1}_vs_{sp2}_vs_{sp3}'
                else:
                    pvalue_species_name = f'{sp1}_vs_{sp2}'

            # Calculate p-values using t-test
            p_values_ttest, actual_test_type = calculate_pvalue_timeseries(data1_trajectories, data2_trajectories, test_type='ttest', species_name=pvalue_species_name)

            # Create separate p-value plot directory
            pvalue_dir = os.path.join(fig_dir, 'pvalue_plots')
            os.makedirs(pvalue_dir, exist_ok=True)
            
            if data3_trajectories is not None and sp3 and traj_dir3:
                # Three-way comparison - create separate p-value plots
                p_values_13_ttest, actual_test_type_13 = calculate_pvalue_timeseries(data1_trajectories, data3_trajectories, test_type='ttest', species_name=pvalue_species_name)
                p_values_23_ttest, actual_test_type_23 = calculate_pvalue_timeseries(data2_trajectories, data3_trajectories, test_type='ttest', species_name=pvalue_species_name)
                
                # Create separate directories for each pairwise comparison
                pvalue_12_dir = os.path.join(pvalue_dir, 'p12')
                pvalue_13_dir = os.path.join(pvalue_dir, 'p13') 
                pvalue_23_dir = os.path.join(pvalue_dir, 'p23')
                os.makedirs(pvalue_12_dir, exist_ok=True)
                os.makedirs(pvalue_13_dir, exist_ok=True)
                os.makedirs(pvalue_23_dir, exist_ok=True)
                
                # Create three separate p-value plots
                create_pvalue_plot(time, p_values_ttest, pvalue_species_name, label1, label2,
                                 fig_dir=pvalue_12_dir, test_type=actual_test_type)
                create_pvalue_plot(time, p_values_13_ttest, pvalue_species_name, label1, label3,
                                 fig_dir=pvalue_13_dir, test_type=actual_test_type_13)
                create_pvalue_plot(time, p_values_23_ttest, pvalue_species_name, label2, label3,
                                 fig_dir=pvalue_23_dir, test_type=actual_test_type_23)
            else:
                # Two-way comparison
                create_pvalue_plot(time, p_values_ttest, pvalue_species_name, label1, label2,
                                 fig_dir=pvalue_dir, test_type=actual_test_type)
        else:
            logging.info(f"Skipping p-value plot for {sp1}/{sp2}{f'/{sp3}' if sp3 else ''} - trajectory data not found")

# ===== Create separate legend figures for 2x2 layouts =====
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Legend 1: Lines only (no shading)
fig_legend, ax_legend = plt.subplots(figsize=(6, 0.5))
ax_legend.set_axis_off()

legend_handles = [
    Line2D([0], [0], color=color1, linewidth=3, label=label1),
    Line2D([0], [0], color=color2, linewidth=3, label=label2),
]
if label3:
    legend_handles.append(Line2D([0], [0], color=color3, linewidth=3, label=label3))

ax_legend.legend(handles=legend_handles, 
                 loc='center', 
                 ncol=3 if label3 else 2, 
                 frameon=True, 
                 framealpha=0.8,
                 fontsize=12,
                 columnspacing=3.0,
                 handlelength=2.5)

plt.tight_layout()
legend_path = os.path.join(fig_dir, 'legend_separate.png')
plt.savefig(legend_path, dpi=300, bbox_inches='tight', transparent=True)
logging.info(f"Saved separate legend figure: {legend_path}")
plt.close()

# Legend 2: Lines with shading (min/max bands) - draw manually like figure_RDMECME_compare.py
from matplotlib.patches import Rectangle

fig_legend2, ax_legend2 = plt.subplots(figsize=(8, 0.8))
ax_legend2.set_axis_off()

# Build legend items list dynamically
legend_items = [
    {'color': color1, 'label': label1, 'x': 0.10},
    {'color': color2, 'label': label2, 'x': 0.45},
]
if label3:
    legend_items.append({'color': color3, 'label': label3, 'x': 0.75})

for item in legend_items:
    # Draw shaded rectangle
    rect = Rectangle((item['x'], 0.3), 0.08, 0.4, 
                     facecolor=item['color'], alpha=0.2, 
                     edgecolor=item['color'], linewidth=1.5,
                     transform=ax_legend2.transAxes)
    ax_legend2.add_patch(rect)
    # Draw line on top
    ax_legend2.plot([item['x'], item['x'] + 0.08], [0.5, 0.5], 
                   color=item['color'], linewidth=2.5, 
                   transform=ax_legend2.transAxes)
    # Add label
    ax_legend2.text(item['x'] + 0.10, 0.5, item['label'], 
                   fontsize=12, va='center', ha='left',
                   transform=ax_legend2.transAxes)

ax_legend2.set_xlim(0, 1)
ax_legend2.set_ylim(0, 1)

plt.tight_layout()
legend_path_shading = os.path.join(fig_dir, 'legend_separate_with_shading.png')
plt.savefig(legend_path_shading, dpi=300, bbox_inches='tight', transparent=True)
logging.info(f"Saved separate legend figure with shading: {legend_path_shading}")
plt.close()

# Ask if user wants to create region-specific comparison plots
region_data_available = include_regions and data1_species_region and data2_species_region
if traj_dir3:
    region_data_available = region_data_available and data3_species_region

if region_data_available:
    create_region_plots = input("\nDo you want to create region-specific comparison plots? (yes/no): ").lower() == 'yes'
    logging.info(f"Region plots won't show up if all trajectories are all 0.")
    if create_region_plots:
        # Create a directory for region plots
        region_plot_dir = os.path.join(fig_dir, 'region_plots')
        os.makedirs(region_plot_dir, exist_ok=True)
        
        # Load region-specific data
        data1_region_df = pd.read_csv(os.path.join(fig_dir, f'{label1}_region_statistics.csv'))
        data2_region_df = pd.read_csv(os.path.join(fig_dir, f'{label2}_region_statistics.csv'))
        data3_region_df = None
        if traj_dir3 and os.path.exists(os.path.join(fig_dir, f'{label3}_region_statistics.csv')):
            data3_region_df = pd.read_csv(os.path.join(fig_dir, f'{label3}_region_statistics.csv'))
        
        # Get unique species 
        species_set = set(data1_region_df['Species']) & set(data2_region_df['Species'])
        if data3_region_df is not None:
            species_set = species_set & set(data3_region_df['Species'])
        
        for species in species_set:
            # Get all regions for this species in both datasets
            data1_regions = set(data1_region_df[data1_region_df['Species'] == species]['Region'])
            data2_regions = set(data2_region_df[data2_region_df['Species'] == species]['Region'])
            data3_regions = set()
            if data3_region_df is not None:
                data3_regions = set(data3_region_df[data3_region_df['Species'] == species]['Region'])
            
            # Find common regions
            common_regions = data1_regions & data2_regions
            if data3_region_df is not None:
                common_regions = common_regions & data3_regions
            
            if not common_regions:
                logging.info(f"No common regions found for species {species}, skipping")
                continue
                
            # Create region-specific plots
            for region in common_regions:
                fig, ax = plt.subplots()
                
                # Get data for this species and region
                data1_region_data = data1_region_df[(data1_region_df['Species'] == species) & 
                                                   (data1_region_df['Region'] == region)]
                data2_region_data = data2_region_df[(data2_region_df['Species'] == species) & 
                                                   (data2_region_df['Region'] == region)]
                data3_region_data = None
                if data3_region_df is not None:
                    data3_region_data = data3_region_df[(data3_region_df['Species'] == species) & 
                                                       (data3_region_df['Region'] == region)]
                
                if len(data1_region_data) == 0 or len(data2_region_data) == 0 or (data3_region_data is not None and len(data3_region_data) == 0):
                    logging.info(f"Missing data for {species} in {region}, skipping")
                    plt.close(fig)
                    continue
                
                data1_row = data1_region_data.iloc[0]
                data2_row = data2_region_data.iloc[0]
                data3_row = None
                if data3_region_data is not None and len(data3_region_data) > 0:
                    data3_row = data3_region_data.iloc[0]
                
                time = str_to_array(data1_row['Time'])
                data1_avg = str_to_array(data1_row['Average'])
                data1_min = str_to_array(data1_row['Min'])
                data1_max = str_to_array(data1_row['Max'])
                data2_avg = str_to_array(data2_row['Average'])
                data2_min = str_to_array(data2_row['Min'])
                data2_max = str_to_array(data2_row['Max'])
                data3_avg = None
                data3_min = None
                data3_max = None
                if data3_row is not None:
                    data3_avg = str_to_array(data3_row['Average'])
                    data3_min = str_to_array(data3_row['Min'])
                    data3_max = str_to_array(data3_row['Max'])
                
                # Check if all trajectories are negligible
                all_zeros = np.all(data1_avg < 10e-6) and np.all(data2_avg < 10e-6)
                if data3_avg is not None:
                    all_zeros = all_zeros and np.all(data3_avg < 10e-6)
                    
                if all_zeros:
                    logging.info(f"All trajectories are all zeros for {species} in {region}, skipping plot")
                    plt.close(fig)
                    continue
                
                # Check if this is a gene species (contains DG) - if so, don't plot min/max
                is_gene_species_region = "DG" in species
                
                # Plot data with specific colors
                ax.plot(time, data1_avg, label=label1, linestyle='-', color=color1)
                if not is_gene_species_region:
                    ax.fill_between(time, data1_min, data1_max, alpha=0.1, color=color1)
                
                ax.plot(time, data2_avg, label=label2, linestyle='-', color=color2)
                if not is_gene_species_region:
                    ax.fill_between(time, data2_min, data2_max, alpha=0.1, color=color2)
                
                if data3_avg is not None:
                    ax.plot(time, data3_avg, label=label3, linestyle='-', color=color3)
                    if not is_gene_species_region:
                        ax.fill_between(time, data3_min, data3_max, alpha=0.1, color=color3)
                
                # Customize plot
                ax.set_xlabel('Time (min)')
                ax.set_ylabel('Counts')
                # ax.set_title(f'{species} in {region}')
                # Legend removed - using separate legend figure
                ax.grid(False)
                
                # Save figure
                plt.tight_layout()
                fig_path = os.path.join(region_plot_dir, f'{species}_{region}_comparison.png')
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                logging.info(f"Saved region plot: {species}_{region}_comparison.png")
                plt.close()
                
                # Create p-value plots for region-specific data (if enabled)
                if draw_pvalues and species in data1_species_region and region in data1_species_region[species] and \
                   species in data2_species_region and region in data2_species_region[species]:
                    
                    data1_region_trajectories = data1_species_region[species][region]
                    data2_region_trajectories = data2_species_region[species][region]
                    
                    # Calculate p-values
                    p_values_region_ttest, actual_test_type_region = calculate_pvalue_timeseries(data1_region_trajectories, data2_region_trajectories, test_type='ttest', species_name=species)
                    
                    # Create region p-value plot directory
                    region_pvalue_dir = os.path.join(fig_dir, 'region_pvalue_plots')
                    os.makedirs(region_pvalue_dir, exist_ok=True)
                    
                    region_species_name = f'{species}_{region}'
                    
                    if data3_region_df is not None and species in data3_species_region and region in data3_species_region[species]:
                        # Three-way region comparison
                        data3_region_trajectories = data3_species_region[species][region]
                        p_values_13_region_ttest, actual_test_type_region_13 = calculate_pvalue_timeseries(data1_region_trajectories, data3_region_trajectories, test_type='ttest', species_name=species)
                        p_values_23_region_ttest, actual_test_type_region_23 = calculate_pvalue_timeseries(data2_region_trajectories, data3_region_trajectories, test_type='ttest', species_name=species)
                        
                        # Create separate directories for each pairwise comparison
                        region_pvalue_12_dir = os.path.join(region_pvalue_dir, 'p12')
                        region_pvalue_13_dir = os.path.join(region_pvalue_dir, 'p13')
                        region_pvalue_23_dir = os.path.join(region_pvalue_dir, 'p23')
                        os.makedirs(region_pvalue_12_dir, exist_ok=True)
                        os.makedirs(region_pvalue_13_dir, exist_ok=True)
                        os.makedirs(region_pvalue_23_dir, exist_ok=True)
                        
                        # Create three separate region p-value plots
                        create_pvalue_plot(time, p_values_region_ttest, region_species_name, label1, label2,
                                         fig_dir=region_pvalue_12_dir, test_type=actual_test_type_region)
                        create_pvalue_plot(time, p_values_13_region_ttest, region_species_name, label1, label3,
                                         fig_dir=region_pvalue_13_dir, test_type=actual_test_type_region_13)
                        create_pvalue_plot(time, p_values_23_region_ttest, region_species_name, label2, label3,
                                         fig_dir=region_pvalue_23_dir, test_type=actual_test_type_region_23)
                    else:
                        # Two-way region comparison
                        create_pvalue_plot(time, p_values_region_ttest, region_species_name, label1, label2,
                                         fig_dir=region_pvalue_dir, test_type=actual_test_type_region)

        # DIY Custom Species-Region Plots
        create_diy_plots = input("\nDo you want to create custom species-region plots? (yes/no): ").lower() == 'yes'
        if create_diy_plots:
            print("\n=== DIY Species-Region Plots ===")
            print("Create plots comparing specific species in specific regions across trajectories.")
            print("Examples:")
            print("  - Compare GAL1 in cytoplasm across all trajectories")
            print("  - Compare GAL1 in cytoplasm+nucleoplasm (combined) across trajectories")
            print("  - Compare multiple species in different regions")
            print("Format: species1:region1+region2, species2:region3, ...")
            print("        Use + to combine multiple regions for the same species")
            
            # Get available species and regions from all trajectories
            all_species = set(data1_region_df['Species'])
            all_species.update(data2_region_df['Species'])
            if data3_region_df is not None:
                all_species.update(data3_region_df['Species'])
            
            # Collect regions from all trajectories (they might differ)
            all_available_regions = set(data1_region_df['Region'])
            all_available_regions.update(data2_region_df['Region'])
            if data3_region_df is not None:
                all_available_regions.update(data3_region_df['Region'])
            
            # Also collect regions from the original data structures for completeness
            if data1_species_region:
                for species, regions in data1_species_region.items():
                    all_available_regions.update(regions.keys())
            if data2_species_region:
                for species, regions in data2_species_region.items():
                    all_available_regions.update(regions.keys())
            if data3_species_region:
                for species, regions in data3_species_region.items():
                    all_available_regions.update(regions.keys())
            
            # Ask if user wants to see available species and regions
            show_lists = input("\nDo you want to see available species and regions? (yes/no): ").lower() == 'yes'
            if show_lists:
                print(f"\nAvailable Species ({len(all_species)}):")
                for i, species in enumerate(sorted(all_species), 1):
                    print(f"  {i:2d}. {species}")
                
                print(f"\nAvailable Regions ({len(all_available_regions)}):")
                for i, region in enumerate(sorted(all_available_regions), 1):
                    print(f"  {i:2d}. {region}")
                print()
            
            # Create DIY plots directory
            diy_plot_dir = os.path.join(fig_dir, 'diy_region_plots')
            os.makedirs(diy_plot_dir, exist_ok=True)
            
            while True:
                comparison = input("\nEnter species:region combinations (or 'done' to finish): ").strip()
                if comparison.lower() == 'done':
                    break
                    
                if not comparison:
                    continue
                
                try:
                    # Parse the comparison string
                    # Format: species1:region1+region2, species2:region3, ...
                    parts = [part.strip() for part in comparison.split(',')]
                    
                    valid_combinations = []
                    for part in parts:
                        if ':' not in part:
                            print(f"Invalid format in '{part}'. Use species:region format")
                            break
                        species, regions_str = part.split(':', 1)
                        species = species.strip()
                        regions = [r.strip() for r in regions_str.split('+')]
                        valid_combinations.append((species, regions))
                    
                    if not valid_combinations:
                        continue
                    
                    # Create separate plots for each species:region combination
                    for species, regions in valid_combinations:
                        fig, ax = plt.subplots()
                        plot_created = False
                        
                        # Plot data for each trajectory directory
                        for traj_idx, (region_df, label, color) in enumerate(zip(
                            [data1_region_df, data2_region_df] + ([data3_region_df] if data3_region_df is not None else []),
                            [label1, label2] + ([label3] if data3_region_df is not None else []),
                            [color1, color2] + ([color3] if data3_region_df is not None else [])
                        )):
                            if region_df is None:
                                continue
                            
                            # Combine data from multiple regions for this species
                            combined_avg = None
                            combined_min = None
                            combined_max = None
                            time = None
                            
                            for region in regions:
                                # Get data for this species and region
                                region_data = region_df[(region_df['Species'] == species) & 
                                                       (region_df['Region'] == region)]
                                
                                if len(region_data) > 0:
                                    row = region_data.iloc[0]
                                    curr_time = str_to_array(row['Time'])
                                    curr_avg = str_to_array(row['Average'])
                                    curr_min = str_to_array(row['Min'])
                                    curr_max = str_to_array(row['Max'])
                                    
                                    if combined_avg is None:
                                        time = curr_time
                                        combined_avg = curr_avg
                                        combined_min = curr_min
                                        combined_max = curr_max
                                    else:
                                        combined_avg += curr_avg
                                        combined_min += curr_min
                                        combined_max += curr_max
                            
                            # Plot combined data if available and meaningful
                            if combined_avg is not None and not np.all(combined_avg < 1e-6):
                                # Check if this is a gene species (contains DG) - if so, don't plot min/max
                                is_gene_species_diy = "DG" in species
                                
                                ax.plot(time, combined_avg, label=label, linestyle='-', color=color)
                                if not is_gene_species_diy:
                                    ax.fill_between(time, combined_min, combined_max, alpha=0.1, color=color)
                                plot_created = True
                        
                        # Only save plot if we have data to show
                        if plot_created:
                            # Create region string for filename and logging
                            region_str = "+".join(regions)
                            filename_region = "_".join(regions)
                            
                            # Customize plot
                            ax.set_xlabel('Time (min)')
                            ax.set_ylabel('Counts')
                            # ax.set_title(f'{species} in {region_str}')
                            # Legend removed - using separate legend figure
                            ax.grid(False)
                            
                            # Save figure
                            plt.tight_layout()
                            filename = f"{species}_{filename_region}_diy_comparison.png"
                            fig_path = os.path.join(diy_plot_dir, filename)
                            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                            logging.info(f"Saved DIY plot: {filename}")
                            plt.close()
                            
                            # Create p-value plots for DIY custom plots (if enabled)
                            if draw_pvalues:
                                # Try to get raw trajectory data for statistical testing
                                try:
                                    # Collect trajectory data from all specified regions for statistical testing
                                    traj_data_sets = []
                                    
                                    for traj_idx, (region_df, region_species_data) in enumerate(zip(
                                        [data1_region_df, data2_region_df] + ([data3_region_df] if data3_region_df is not None else []),
                                        [data1_species_region, data2_species_region] + ([data3_species_region] if data3_region_df is not None else [])
                                    )):
                                        if region_df is None or region_species_data is None:
                                            continue
                                            
                                        if species in region_species_data:
                                            # Combine trajectory data across regions
                                            combined_trajectories = None
                                            for region in regions:
                                                if region in region_species_data[species]:
                                                    region_trajectories = region_species_data[species][region]
                                                    if combined_trajectories is None:
                                                        # Explicitly cast to float64 to avoid dtype issues
                                                        combined_trajectories = [np.array(traj, dtype=np.float64) for traj in region_trajectories]
                                                    else:
                                                        # Add trajectories element-wise
                                                        for i in range(len(combined_trajectories)):
                                                            combined_trajectories[i] += np.array(region_trajectories[i], dtype=np.float64)
                                            
                                            if combined_trajectories is not None:
                                                traj_data_sets.append(combined_trajectories)
                                    
                                    # Create p-value plots if we have enough data
                                    if len(traj_data_sets) >= 2:
                                        diy_pvalue_dir = os.path.join(fig_dir, 'diy_pvalue_plots')
                                        os.makedirs(diy_pvalue_dir, exist_ok=True)
                                        
                                        diy_species_name = f"{species}_{filename_region}"
                                        
                                        if len(traj_data_sets) == 3:
                                            # Three-way comparison
                                            p_values_12_diy, actual_test_type_diy_12 = calculate_pvalue_timeseries(traj_data_sets[0], traj_data_sets[1], test_type='ttest', species_name=species)
                                            p_values_13_diy, actual_test_type_diy_13 = calculate_pvalue_timeseries(traj_data_sets[0], traj_data_sets[2], test_type='ttest', species_name=species)
                                            p_values_23_diy, actual_test_type_diy_23 = calculate_pvalue_timeseries(traj_data_sets[1], traj_data_sets[2], test_type='ttest', species_name=species)
                                            
                                            # Create separate directories for each pairwise comparison
                                            diy_pvalue_12_dir = os.path.join(diy_pvalue_dir, 'p12')
                                            diy_pvalue_13_dir = os.path.join(diy_pvalue_dir, 'p13')
                                            diy_pvalue_23_dir = os.path.join(diy_pvalue_dir, 'p23')
                                            os.makedirs(diy_pvalue_12_dir, exist_ok=True)
                                            os.makedirs(diy_pvalue_13_dir, exist_ok=True)
                                            os.makedirs(diy_pvalue_23_dir, exist_ok=True)
                                            
                                            # Create three separate DIY p-value plots
                                            create_pvalue_plot(time, p_values_12_diy, diy_species_name, label1, label2,
                                                             fig_dir=diy_pvalue_12_dir, test_type=actual_test_type_diy_12)
                                            create_pvalue_plot(time, p_values_13_diy, diy_species_name, label1, label3,
                                                             fig_dir=diy_pvalue_13_dir, test_type=actual_test_type_diy_13)
                                            create_pvalue_plot(time, p_values_23_diy, diy_species_name, label2, label3,
                                                             fig_dir=diy_pvalue_23_dir, test_type=actual_test_type_diy_23)
                                        else:
                                            # Two-way comparison
                                            p_values_diy, actual_test_type_diy = calculate_pvalue_timeseries(traj_data_sets[0], traj_data_sets[1], test_type='ttest', species_name=species)
                                            create_pvalue_plot(time, p_values_diy, diy_species_name, label1, label2,
                                                             fig_dir=diy_pvalue_dir, test_type=actual_test_type_diy)
                                            
                                except Exception as e:
                                    logging.info(f"Could not create p-value plot for DIY comparison {species}_{filename_region}: {e}")
                        else:
                            region_str = "+".join(regions)
                            logging.info(f"No meaningful data found for {species} in {region_str}")
                            plt.close(fig)
                    
                except Exception as e:
                    logging.info(f"Error creating DIY plot: {e}")
                    print(f"Error: {e}")

# Special case: G2 membrane totals
# Ask if user wants to create G2 total plot
create_g2_total = True

if create_g2_total:
    # Calculate G2 totals for first directory
    data1_g2_data = data1_df[data1_df['Species'].isin(['ODE_G2', 'ODE_G2GAE', 'ODE_G2GAI'])].copy()
    if len(data1_g2_data) > 0:
        time = str_to_array(data1_g2_data.iloc[0]['Time'])
        data1_total = np.zeros_like(str_to_array(data1_g2_data.iloc[0]['Average']))
        data1_total_min = np.zeros_like(data1_total)
        data1_total_max = np.zeros_like(data1_total)
        
        for _, row in data1_g2_data.iterrows():
            data1_total += str_to_array(row['Average'])
            data1_total_min += str_to_array(row['Min'])
            data1_total_max += str_to_array(row['Max'])

    # Calculate G2 totals for second directory
    data2_g2_data = data2_df[data2_df['Species'].isin(['ODE_G2', 'ODE_G2GAE', 'ODE_G2GAI'])].copy()
    if len(data2_g2_data) > 0:
        data2_total = np.zeros_like(str_to_array(data2_g2_data.iloc[0]['Average']))
        data2_total_min = np.zeros_like(data2_total)
        data2_total_max = np.zeros_like(data2_total)
        
        for _, row in data2_g2_data.iterrows():
            data2_total += str_to_array(row['Average'])
            data2_total_min += str_to_array(row['Min'])
            data2_total_max += str_to_array(row['Max'])
    # Calculate G2 totals for third directory if it exists
    data3_total = None
    data3_total_min = None
    data3_total_max = None
    if traj_dir3:
        data3_g2_data = data3_df[data3_df['Species'].isin(['ODE_G2', 'ODE_G2GAE', 'ODE_G2GAI'])].copy()
        if len(data3_g2_data) > 0:
            data3_total = np.zeros_like(str_to_array(data3_g2_data.iloc[0]['Average']))
            data3_total_min = np.zeros_like(data3_total)
            data3_total_max = np.zeros_like(data3_total)
            
            for _, row in data3_g2_data.iterrows():
                data3_total += str_to_array(row['Average'])
                data3_total_min += str_to_array(row['Min'])
                data3_total_max += str_to_array(row['Max'])

    # Create the plot
    plt.figure()
    plt.plot(time, data1_total, label=label1, linestyle='-', color=color1)
    plt.fill_between(time, data1_total_min, data1_total_max, alpha=0.1, color=color1)

    plt.plot(time, data2_total, label=label2, linestyle='-', color=color2)
    plt.fill_between(time, data2_total_min, data2_total_max, alpha=0.1, color=color2)
    
    # Add third directory data if it exists
    if data3_total is not None:
        plt.plot(time, data3_total, label=label3, linestyle='-', color=color3)
        plt.fill_between(time, data3_total_min, data3_total_max, alpha=0.1, color=color3)

    plt.xlabel('Time (min)')
    plt.ylabel('Counts')
    # plt.title('Total G2 Comparison (G2 + G2GAE + G2GAI)')
    plt.legend(framealpha=0.3, loc='best')
    plt.grid(False)

    # Save figure
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, 'G2_membrane_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved plot for G2 total")
    plt.close()
    
    # Create p-value plot for G2 total comparison (if enabled)
    if draw_pvalues:
        try:
            # Get raw trajectory data for G2 species to calculate combined totals
            g2_species_list = ['G2', 'G2GAE', 'G2GAI']
            
            # Calculate combined trajectories for each dataset
            data1_g2_combined = []
            data2_g2_combined = []
            data3_g2_combined = []
            
            # Get number of trajectories from first available species
            n_trajectories = 0
            for species in g2_species_list:
                if species in data1_ode:
                    n_trajectories = len(data1_ode[species])
                    break
            
            if n_trajectories > 0:
                # Initialize combined trajectories
                for i in range(n_trajectories):
                    data1_g2_combined.append(np.zeros_like(data1_ode[g2_species_list[0]][i]))
                    data2_g2_combined.append(np.zeros_like(data2_ode[g2_species_list[0]][i]))
                    if traj_dir3:
                        data3_g2_combined.append(np.zeros_like(data3_ode[g2_species_list[0]][i]))
                
                # Sum trajectories across G2 species
                for species in g2_species_list:
                    if species in data1_ode:
                        for i in range(n_trajectories):
                            data1_g2_combined[i] += data1_ode[species][i]
                    if species in data2_ode:
                        for i in range(n_trajectories):
                            data2_g2_combined[i] += data2_ode[species][i]
                    if traj_dir3 and species in data3_ode:
                        for i in range(n_trajectories):
                            data3_g2_combined[i] += data3_ode[species][i]
                
                # Create p-value plots
                special_pvalue_dir = os.path.join(fig_dir, 'special_pvalue_plots')
                os.makedirs(special_pvalue_dir, exist_ok=True)
                
                if traj_dir3 and data3_g2_combined:
                    # Three-way comparison
                    p_values_g2_12, actual_test_type_g2_12 = calculate_pvalue_timeseries(data1_g2_combined, data2_g2_combined, test_type='ttest', species_name='G2_membrane_total')
                    p_values_g2_13, actual_test_type_g2_13 = calculate_pvalue_timeseries(data1_g2_combined, data3_g2_combined, test_type='ttest', species_name='G2_membrane_total')
                    p_values_g2_23, actual_test_type_g2_23 = calculate_pvalue_timeseries(data2_g2_combined, data3_g2_combined, test_type='ttest', species_name='G2_membrane_total')
                    
                    # Create separate directories for each pairwise comparison
                    special_pvalue_12_dir = os.path.join(special_pvalue_dir, 'p12')
                    special_pvalue_13_dir = os.path.join(special_pvalue_dir, 'p13')
                    special_pvalue_23_dir = os.path.join(special_pvalue_dir, 'p23')
                    os.makedirs(special_pvalue_12_dir, exist_ok=True)
                    os.makedirs(special_pvalue_13_dir, exist_ok=True)
                    os.makedirs(special_pvalue_23_dir, exist_ok=True)
                    
                    # Create three separate G2_membrane_total p-value plots
                    create_pvalue_plot(time, p_values_g2_12, 'G2_membrane_total', label1, label2,
                                     fig_dir=special_pvalue_12_dir, test_type=actual_test_type_g2_12)
                    create_pvalue_plot(time, p_values_g2_13, 'G2_membrane_total', label1, label3,
                                     fig_dir=special_pvalue_13_dir, test_type=actual_test_type_g2_13)
                    create_pvalue_plot(time, p_values_g2_23, 'G2_membrane_total', label2, label3,
                                     fig_dir=special_pvalue_23_dir, test_type=actual_test_type_g2_23)
                else:
                    # Two-way comparison
                    p_values_g2, actual_test_type_g2 = calculate_pvalue_timeseries(data1_g2_combined, data2_g2_combined, test_type='ttest', species_name='G2_membrane_total')
                    create_pvalue_plot(time, p_values_g2, 'G2_membrane_total', label1, label2,
                                     fig_dir=special_pvalue_dir, test_type=actual_test_type_g2)
                
        except Exception as e:
            logging.info(f"Could not create p-value plot for G2 total: {e}")
    
'''==================================================
Special case: GAI total
=================================================='''
create_gai_total = True
# Modify GAI total plot to include third directory
if create_gai_total:
    # Ask if user wants to specify custom GAE value for horizontal line
    add_gae_line = input("Do you want to add a horizontal line for GAE reference value? (yes/no): ").lower()
    if add_gae_line == 'yes' or add_gae_line.strip() == '':
        add_gae_line = True
    gae_value = None
    
    if add_gae_line:
        try:
            gae_input = input("Enter GAE value in mM (default 11.1): ")
            if gae_input.strip() == '':
                gae_value = 11.1
            else:
                gae_value = float(gae_input)
        except ValueError:
            logging.info("Invalid GAE value, no reference line will be added")
            add_gae_line = False
    
    fig, ax = plt.subplots()

    # List of species to combine
    gai_species = ['GAI', 'G1GAI', 'G3i', 'G2GAI']
    
    # Allow user to customize the GAI species list
    customize_gai_species = input("Do you want to customize the GAI species list? (yes/no): ").lower() == 'yes'
    if customize_gai_species:
        print(f"Current GAI species list: {gai_species}")
        new_species_list = input("Enter comma-separated list of species to combine: ")
        if new_species_list:
            gai_species = [s.strip() for s in new_species_list.split(',')]
    
    # Initialize arrays for data from each directory
    data1_combined_avg = None
    data1_combined_min = None
    data1_combined_max = None
    data2_combined_avg = None
    data2_combined_min = None
    data2_combined_max = None
    data3_combined_avg = None
    data3_combined_min = None
    data3_combined_max = None
    time = None

    # For tracking which species are actually used
    data1_species_used = []
    data2_species_used = []
    data3_species_used = []

    # Combine data from first directory
    for species_name in gai_species:
        # Look for both ODE and RDME versions of the species
        matching_rows = data1_df[data1_df['Species'].str.contains(species_name)]
        
        if not matching_rows.empty:
            # Prefer ODE data if available
            data1_species_data = matching_rows[matching_rows['Species'].str.startswith('ODE')]
            if data1_species_data.empty:
                data1_species_data = matching_rows
                
            if len(data1_species_data) > 0:
                data1_row = data1_species_data.iloc[0]
                # Track which species are being used
                data1_species_used.append(data1_row['Species'])
                
                curr_avg = str_to_array(data1_row['Average']) / NAV * 1e3
                curr_min = str_to_array(data1_row['Min']) / NAV * 1e3
                curr_max = str_to_array(data1_row['Max']) / NAV * 1e3
                
                if data1_combined_avg is None:
                    time = str_to_array(data1_row['Time'])
                    data1_combined_avg = curr_avg
                    data1_combined_min = curr_min
                    data1_combined_max = curr_max
                else:
                    data1_combined_avg += curr_avg
                    data1_combined_min += curr_min
                    data1_combined_max += curr_max

    # Combine data from second directory
    for species_name in gai_species:
        # Look for both ODE and RDME versions of the species
        matching_rows = data2_df[data2_df['Species'].str.contains(species_name)]
        
        if not matching_rows.empty:
            # Prefer ODE data if available
            data2_species_data = matching_rows[matching_rows['Species'].str.startswith('ODE')]
            if data2_species_data.empty:
                data2_species_data = matching_rows
                
            if len(data2_species_data) > 0:
                data2_row = data2_species_data.iloc[0]
                # Track which species are being used
                data2_species_used.append(data2_row['Species'])
                
                curr_avg = str_to_array(data2_row['Average']) / NAV * 1e3
                curr_min = str_to_array(data2_row['Min']) / NAV * 1e3
                curr_max = str_to_array(data2_row['Max']) / NAV * 1e3
                
                if data2_combined_avg is None:
                    data2_combined_avg = curr_avg
                    data2_combined_min = curr_min
                    data2_combined_max = curr_max
                else:
                    data2_combined_avg += curr_avg
                    data2_combined_min += curr_min
                    data2_combined_max += curr_max

    # Combine data from third directory if it exists
    if traj_dir3:
        for species_name in gai_species:
            # Look for both ODE and RDME versions of the species
            matching_rows = data3_df[data3_df['Species'].str.contains(species_name)]
            
            if not matching_rows.empty:
                # Prefer ODE data if available
                data3_species_data = matching_rows[matching_rows['Species'].str.startswith('ODE')]
                if data3_species_data.empty:
                    data3_species_data = matching_rows
                    
                if len(data3_species_data) > 0:
                    data3_row = data3_species_data.iloc[0]
                    # Track which species are being used
                    data3_species_used.append(data3_row['Species'])
                    
                    curr_avg = str_to_array(data3_row['Average']) / NAV * 1e3
                    curr_min = str_to_array(data3_row['Min']) / NAV * 1e3
                    curr_max = str_to_array(data3_row['Max']) / NAV * 1e3
                    
                    if data3_combined_avg is None:
                        data3_combined_avg = curr_avg
                        data3_combined_min = curr_min
                        data3_combined_max = curr_max
                    else:
                        data3_combined_avg += curr_avg
                        data3_combined_min += curr_min
                        data3_combined_max += curr_max

    # Print which species were actually used
    logging.info(f"{label1} species used in GAI total: {data1_species_used}")
    logging.info(f"{label2} species used in GAI total: {data2_species_used}")
    if traj_dir3:
        logging.info(f"{label3} species used in GAI total: {data3_species_used}")


    # Plot first directory data if it exists
    if data1_combined_avg is not None and time is not None:
        ax.plot(time, data1_combined_avg, label=label1, linestyle='-', color=color1)
        ax.fill_between(time, data1_combined_min, data1_combined_max, alpha=0.1, color=color1)

    # Plot second directory data if it exists
    if data2_combined_avg is not None:
        ax.plot(time, data2_combined_avg, label=label2, linestyle='-', color=color2)
        ax.fill_between(time, data2_combined_min, data2_combined_max, alpha=0.1, color=color2)
    
    # Plot third directory data if it exists
    if data3_combined_avg is not None:
        ax.plot(time, data3_combined_avg, label=label3, linestyle='-', color=color3)
        ax.fill_between(time, data3_combined_min, data3_combined_max, alpha=0.1, color=color3)
                        
    # Add horizontal line for GAE reference value if requested
    if add_gae_line and gae_value is not None:
        ax.axhline(y=gae_value, color='gray', linestyle='--', linewidth=2, label='GAE')
        ax.text(time[0]*1.05, gae_value*0.97, f'{gae_value} mM', color='gray', va='top', ha='left')

    # Customize plot
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    # ax.set_title('Total GAI Species Comparison')
    # Legend removed - using separate legend figure
    ax.grid(False)

    # Save figure
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, 'GAI_total_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved plot for GAI total")
    plt.close()
    
    # Create p-value plot for GAI total comparison (if enabled)
    if draw_pvalues:
        try:
            # Calculate combined GAI trajectories for statistical testing
            data1_gai_combined = []
            data2_gai_combined = []
            data3_gai_combined = []
            
            # Get number of trajectories from first available GAI species
            n_trajectories = 0
            for species_name in gai_species:
                # Look for both ODE and RDME versions
                for data_dict in [data1_ode, data1_species]:
                    for species_key in data_dict.keys():
                        if species_name in species_key:
                            n_trajectories = len(data_dict[species_key])
                            break
                    if n_trajectories > 0:
                        break
                if n_trajectories > 0:
                    break
            
            if n_trajectories > 0:
                # Initialize combined trajectories with zeros
                for i in range(n_trajectories):
                    # Use first available trajectory as template for shape
                    template_traj = None
                    for species_name in gai_species:
                        for data_dict in [data1_ode, data1_species]:
                            for species_key in data_dict.keys():
                                if species_name in species_key and len(data_dict[species_key]) > 0:
                                    template_traj = data_dict[species_key][0]
                                    break
                            if template_traj is not None:
                                break
                        if template_traj is not None:
                            break
                    
                    if template_traj is not None:
                        data1_gai_combined.append(np.zeros_like(template_traj))
                        data2_gai_combined.append(np.zeros_like(template_traj))
                        if traj_dir3:
                            data3_gai_combined.append(np.zeros_like(template_traj))
                
                # Sum trajectories across GAI species for each dataset
                for species_name in gai_species:
                    # Dataset 1
                    for data_dict in [data1_ode, data1_species]:
                        for species_key in data_dict.keys():
                            if species_name in species_key and len(data_dict[species_key]) >= n_trajectories:
                                for i in range(n_trajectories):
                                    data1_gai_combined[i] += data_dict[species_key][i]
                                break
                    
                    # Dataset 2
                    for data_dict in [data2_ode, data2_species]:
                        for species_key in data_dict.keys():
                            if species_name in species_key and len(data_dict[species_key]) >= n_trajectories:
                                for i in range(n_trajectories):
                                    data2_gai_combined[i] += data_dict[species_key][i]
                                break
                    
                    # Dataset 3 (if exists)
                    if traj_dir3:
                        for data_dict in [data3_ode, data3_species]:
                            for species_key in data_dict.keys():
                                if species_name in species_key and len(data_dict[species_key]) >= n_trajectories:
                                    for i in range(n_trajectories):
                                        data3_gai_combined[i] += data_dict[species_key][i]
                                    break
                
                # Create p-value plots
                special_pvalue_dir = os.path.join(fig_dir, 'special_pvalue_plots')
                os.makedirs(special_pvalue_dir, exist_ok=True)
                
                if traj_dir3 and data3_gai_combined:
                    # Three-way comparison
                    p_values_gai_12, actual_test_type_gai_12 = calculate_pvalue_timeseries(data1_gai_combined, data2_gai_combined, test_type='ttest', species_name='GAI_total')
                    p_values_gai_13, actual_test_type_gai_13 = calculate_pvalue_timeseries(data1_gai_combined, data3_gai_combined, test_type='ttest', species_name='GAI_total')
                    p_values_gai_23, actual_test_type_gai_23 = calculate_pvalue_timeseries(data2_gai_combined, data3_gai_combined, test_type='ttest', species_name='GAI_total')
                    
                    # Create separate directories for each pairwise comparison
                    special_pvalue_12_dir = os.path.join(special_pvalue_dir, 'p12')
                    special_pvalue_13_dir = os.path.join(special_pvalue_dir, 'p13')
                    special_pvalue_23_dir = os.path.join(special_pvalue_dir, 'p23')
                    os.makedirs(special_pvalue_12_dir, exist_ok=True)
                    os.makedirs(special_pvalue_13_dir, exist_ok=True)
                    os.makedirs(special_pvalue_23_dir, exist_ok=True)
                    
                    # Create three separate GAI_total p-value plots
                    create_pvalue_plot(time, p_values_gai_12, 'GAI_total', label1, label2,
                                    fig_dir=special_pvalue_12_dir, test_type=actual_test_type_gai_12)
                    create_pvalue_plot(time, p_values_gai_13, 'GAI_total', label1, label3,
                                    fig_dir=special_pvalue_13_dir, test_type=actual_test_type_gai_13)
                    create_pvalue_plot(time, p_values_gai_23, 'GAI_total', label2, label3,
                                    fig_dir=special_pvalue_23_dir, test_type=actual_test_type_gai_23)
                else:
                    # Two-way comparison
                    p_values_gai, actual_test_type_gai = calculate_pvalue_timeseries(data1_gai_combined, data2_gai_combined, test_type='ttest', species_name='GAI_total')
                    create_pvalue_plot(time, p_values_gai, 'GAI_total', label1, label2,
                                    fig_dir=special_pvalue_dir, test_type=actual_test_type_gai)
                    
        except Exception as e:
            logging.info(f"Could not create p-value plot for GAI total: {e}")

logging.info(f"\nAll plots saved in: {fig_dir}")
logging.info(f"P-value significance plots saved in subdirectories:")
logging.info(f"  - Main species comparisons: {fig_dir}/pvalue_plots/")
logging.info(f"  - Region-specific comparisons: {fig_dir}/region_pvalue_plots/")
logging.info(f"  - DIY custom comparisons: {fig_dir}/diy_pvalue_plots/")
logging.info(f"  - Special comparisons (G2, GAI totals): {fig_dir}/special_pvalue_plots/")



logging.getLogger().handlers[0].flush()