#!/usr/bin/env python
# coding: utf-8

# # RDME/CME_compare
# This is the code used for extracting the data from RDME-ODE simulation results and comparing various trajectories of species with and without ER.

# In[ ]:


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
from pyLM import *
from pyLM.units import *

from jLM import CMEPostProcessing as PostProcessing
from scipy.stats import ttest_ind, ks_2samp
import hashlib
logging.getLogger('jLM').setLevel(logging.WARNING)
logging.getLogger('pyLM').setLevel(logging.WARNING)
cme_traj_dir = "/data2/2024_Yeast_GS/my_current_code/my_cme_ode/output/03232025/"
rdme_traj_dir = "/data2/2024_Yeast_GS/my_current_code/rdme_ode_results/20251031_baseline_newcytoribono"
fig_dir = os.path.join(rdme_traj_dir, 'figures_rdmecme_comparison/')


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

# Setup publication style (same as figure_comparison.py)
colors = setup_publication_style(figure_size='medium', dpi=300)

# Define custom colors for consistency
# CME-ODE: light green, RDME-ODE: orange (colors[1])
cme_color = '#2ca02c'  # Light green for CME-ODE
rdme_color = colors[1]  # Orange for RDME-ODE

logging.info(f"This is the file to compare between RDME-ODE and CME-ODE data: {rdme_traj_dir} and {cme_traj_dir}")

'''
================================================================================================
This section is for loading trajectory, use cached data if available
================================================================================================
'''
# default, get data required

# In[2]:

def get_cache_key(rdme_traj_dir, cme_traj_dir):
    """Generate a cache key based only on number of trajs and traj file names"""
    # Add RDME files info (only names and count)
    rdme_files = [f for f in os.listdir(rdme_traj_dir) if f.startswith('yeast') and f.endswith('.lm')]
    rdme_files.sort()
    rdme_data = (tuple(rdme_files), len(rdme_files))
    
    # Add CME files info (only names and count)
    cme_files = [f for f in os.listdir(cme_traj_dir) if f.endswith('.lm')]
    cme_files.sort()
    cme_data = (tuple(cme_files), len(cme_files))
    
    # Create hash using only file names and counts
    cache_data = (rdme_data, cme_data)
    return hashlib.md5(str(cache_data).encode()).hexdigest()

def save_cached_data(cache_file, rdme_species_data, rdme_species_region_data, rdme_ode_data, 
                     rdmeTs, odeTs, NAV):
    """Save processed trajectory data to cache (excluding cme_traj which can't be pickled)"""
    cache_data = {
        'rdme_species_data': rdme_species_data,
        'rdme_species_region_data': rdme_species_region_data, 
        'rdme_ode_data': rdme_ode_data,
        'rdmeTs': rdmeTs,
        'odeTs': odeTs,
        'NAV': NAV
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    logging.info(f"Saved cached data to: {cache_file}")

def load_cached_data(cache_file):
    """Load processed trajectory data from cache"""
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    logging.info(f"Loaded cached data from: {cache_file}")
    return (cache_data['rdme_species_data'], cache_data['rdme_species_region_data'], 
            cache_data['rdme_ode_data'], cache_data['rdmeTs'], cache_data['odeTs'],
            cache_data['NAV'])

# Check for cached data
cache_key = get_cache_key(rdme_traj_dir, cme_traj_dir)
cache_file = os.path.join(fig_dir, f'trajectory_cache_{cache_key}.pkl')

use_cache = os.path.exists(cache_file)
logging.info(f"Cache file: {cache_file}")
logging.info(f"Using cached data: {use_cache}")


if use_cache:
    # Load from cache
    rdme_species_data, rdme_species_region_data, rdme_ode_data, rdmeTs, odeTs, NAV = load_cached_data(cache_file)
    # Load CME data separately (can't be cached due to h5py objects)
    cme_files = ['gal_cme_ode_gae11.1mM_11.1_gai0_rep50_delta1_time60.lm']
    cme_traj = PostProcessing.openLMFile(os.path.join(cme_traj_dir + cme_files[0]))
else:
    # Process files normally
    rdme_files = [f for f in os.listdir(rdme_traj_dir) if f.startswith('yeast') and f.endswith('.lm')]
    cme_files = ['gal_cme_ode_gae11.1mM_11.1_gai0_rep50_delta1_time60.lm']
    traj_suff = "_ode.jsonl"

    logging.info(f"RDME-ODE files: {rdme_files}")
    logging.info(f"CME-ODE files: {cme_files}")

    # Initialize dictionaries to store data for each species
    rdme_species_data = {}
    rdme_species_region_data = {}
    rdme_ode_data = {}
    rdmeTs = None
    odeTs = None
    cmeTs = None

    # Process RDME files
    for traj_file in tqdm(rdme_files, desc="Processing RDME files", unit="file"):
        logging.info(f"Processing RDME file: {traj_file}")
        traj, odeTraj, region_traj = get_traj(rdme_traj_dir, traj_file, traj_suff,region_suff='_region.jsonl')

        curr_rdmeTs, rdmeYs, curr_odeTs, odeYs, regionTs, regionYs = get_data_for_plot(traj, odeTraj, region_traj=region_traj, sparse_factor=1)
        NAV = 6.022e23 * (traj.reg.cytoplasm.volume + traj.reg.nucleoplasm.volume + traj.reg.plasmaMembrane.volume)
        if rdmeTs is None:
            rdmeTs = curr_rdmeTs
            odeTs = curr_odeTs

        for species, data in rdmeYs.items():
            if species not in rdme_species_data:
                rdme_species_data[species] = []
            rdme_species_data[species].append(data)

        for species, data in odeYs.items():
            if species not in rdme_ode_data:
                rdme_ode_data[species] = []
            rdme_ode_data[species].append(data)
        
        # Process region data
        if regionYs is not None:
            regions = region_traj['regions']
         
             # Initialize the nested dictionary structure if needed
            for species, region_data in regionYs.items():
                if species not in rdme_species_region_data:
                    rdme_species_region_data[species] = {}
                
                # Initialize lists for each region if they don't exist
                for region in regions:
                    if region not in rdme_species_region_data[species]:
                        rdme_species_region_data[species][region] = []
                
                # Now append the data
                for i in range(len(regions)):
                    rdme_species_region_data[species][regions[i]].append(regionYs[species][i])
    
    # Load CME data
    cme_traj = PostProcessing.openLMFile(os.path.join(cme_traj_dir + cme_files[0]))
    
    # Save to cache (excluding cme_traj which can't be pickled)
    save_cached_data(cache_file, rdme_species_data, rdme_species_region_data, rdme_ode_data, 
                     rdmeTs, odeTs, NAV)
                
logging.info(f"the length of rdme_species_data: {len(rdme_species_data)}")
logging.info(f"the length of rdme_ode_data: {len(rdme_ode_data)}")
logging.info(f"the length of rdme_species_region_data: {len(rdme_species_region_data)}")


# In[3]:


print(rdme_species_data.keys())


# In[4]:

'''
================================================================================================
This section is for loading and calculating the statistics for RDME
================================================================================================
'''
rdme_results = []  # List to store overall species statistics (across all regions)
rdme_results_region = []  # List to store region-specific species statistics
rdme_total_results = []
rdme_total_results_traj = {}  # Dictionary to store total species statistics
rdme_serach_key = 0
cme_serach_key = 1
'''
pattern to search for in the species name in both RDME and CME
'''
total_species_pattern = {
    'G1': ('G1', 'G1'),
    'G2': ('G2', 'G2'),
    'G3': ('G3', 'G3'),
    'G4': ('G4', 'G4'),
    'G80': ('G80', 'G80'),
    'Grep': ('Grep', 'reporter'),
    'R1': ('R1', 'R1'),
    'R2': ('R2', 'R2'),
    'R3': ('R3', 'R3'),
    'R4': ('R4', 'R4'),
    'R80': ('R80', 'R80'),
    'Rrep': ('Rrep', 'reporter_rna'),
}
# Process overall species data (summed across all regions)
for species, trajectories in rdme_species_data.items():
    pattern_used = []
    for pattern in total_species_pattern.values():
        rdme_pattern = pattern[rdme_serach_key]
        if rdme_pattern in ['G4', 'G80']: # might be dimer
            dimer_pattern = rdme_pattern + 'd'
        else:
            dimer_pattern = None
        if dimer_pattern is not None and dimer_pattern in species:
            pattern_used.append(dimer_pattern)
        elif rdme_pattern in species and 'D'+rdme_pattern not in species:
            pattern_used.append(rdme_pattern)
    if len(pattern_used) > 1:
        logging.info(f"Multiple patterns found for species {species}: {pattern_used}")
    elif len(pattern_used) == 0:
        logging.info(f"No pattern found for species {species}")
    
        
    # you can have only one pattern either dimer or monoer

    trajectories_array = np.array(trajectories)  # Convert list of trajectories to numpy array
    avg = np.mean(trajectories_array, axis=0)    # Calculate mean across all trajectories for each timepoint
    std = np.std(trajectories_array, axis=0)     # Calculate standard deviation across trajectories
    min_vals = np.min(trajectories_array, axis=0)  # Calculate minimum values
    max_vals = np.max(trajectories_array, axis=0)  # Calculate maximum values
    
    # Store results as dictionary with comma-separated strings for time series data
    rdme_results.append({
        'Species': f"RDME_{species}",            # Prefix species name with RDME for identification
        'Time': ','.join(map(str, rdmeTs)),      # Convert time points to comma-separated string
        'Average': ','.join(map(str, avg)),      # Convert average values to comma-separated string
        'Std': ','.join(map(str, std)),          # Convert standard deviation values to comma-separated string
        'Min': ','.join(map(str, min_vals)),     # Convert minimum values to comma-separated string
        'Max': ','.join(map(str, max_vals))      # Convert maximum values to comma-separated string
    })
    for pattern in pattern_used:
        if 'd' in pattern:
            pattern = pattern.replace('d', '')
            mult_factor = 2
        else:
            mult_factor = 1
        
        if pattern not in rdme_total_results_traj.keys():
            rdme_total_results_traj[pattern] = trajectories_array * mult_factor
        else: # here we only need to add for avg, 
            rdme_total_results_traj[pattern] += trajectories_array * mult_factor

for pattern in rdme_total_results_traj.keys():
    rdme_total_traj = rdme_total_results_traj[pattern]
    rdme_total_avg = np.mean(rdme_total_traj, axis=0)
    rdme_total_std = np.std(rdme_total_traj, axis=0)
    rdme_total_min = np.min(rdme_total_traj, axis=0)
    rdme_total_max = np.max(rdme_total_traj, axis=0)
    rdme_total_results.append({
        'Species': f"RDME_{pattern}_total",
        'Time': ','.join(map(str, rdmeTs)),
        'Average': ','.join(map(str, rdme_total_avg)),
        'Std': ','.join(map(str, rdme_total_std)),
        'Min': ','.join(map(str, rdme_total_min)),
        'Max': ','.join(map(str, rdme_total_max))
    })

# Process region-specific species data
for species, regions in rdme_species_region_data.items():
    for region, trajectories in regions.items():
        trajectories_array = np.array(trajectories)  # Convert list of trajectories to numpy array
        avg = np.mean(trajectories_array, axis=0)    # Calculate mean across all trajectories for each timepoint
        std = np.std(trajectories_array, axis=0)     # Calculate standard deviation across trajectories
        min_vals = np.min(trajectories_array, axis=0)  # Calculate minimum values
        max_vals = np.max(trajectories_array, axis=0)  # Calculate maximum values
        
        # Store region-specific results as dictionary
        rdme_results_region.append({
            'Species': species,                       # Species name (without prefix)
            'Region': region,                         # Region name (e.g., cytoplasm, nucleus, etc.)
            'Time': ','.join(map(str,  rdmeTs)),  # Use region-specific time if available
            'Average': ','.join(map(str, avg)),       # Convert average values to comma-separated string
            'Std': ','.join(map(str, std)),           # Convert standard deviation values to comma-separated string
            'Min': ','.join(map(str, min_vals)),      # Convert minimum values to comma-separated string
            'Max': ','.join(map(str, max_vals))       # Convert maximum values to comma-separated string
        })
        # save a copy in rdme_results, with the species name as f"RDME_{species}_{region}"
        rdme_results.append({
        'Species': f"RDME_{species}_{region}",
        'Time': ','.join(map(str, rdmeTs)),
        'Average': ','.join(map(str, avg)),
        'Std': ','.join(map(str, std)),
        'Min': ','.join(map(str, min_vals)),
        'Max': ','.join(map(str, max_vals))
    })
        
for species, trajectories in rdme_ode_data.items():
    trajectories_array = np.array(trajectories)
    avg = np.mean(trajectories_array, axis=0)
    std = np.std(trajectories_array, axis=0)
    min_vals = np.min(trajectories_array, axis=0)
    max_vals = np.max(trajectories_array, axis=0)
    
    rdme_results.append({
        'Species': f"ODE_{species}",
        'Time': ','.join(map(str, odeTs)),
        'Average': ','.join(map(str, avg)),
        'Std': ','.join(map(str, std)),
        'Min': ','.join(map(str, min_vals)),
        'Max': ','.join(map(str, max_vals))
    })

logging.info(f"the length of rdme_results: {len(rdme_results)}")
logging.info(f"the length of rdme_results_region: {len(rdme_results_region)}")


# In[5]:


# Calculate and save CME statistics
# Note: cme_traj is already loaded from cache or during initial processing
cme_species_list = PostProcessing.getSpecies(cme_traj)


# Reorganize the species list based on the given criteria
# Reorganize the species list based on the given criteria
cme_species_list = sorted(cme_species_list, key=lambda x: (
    not x[0].startswith('DG'),                                # Sort DG species first
    not (x[0].startswith('R') or x[0] == 'reporter_rna'),     # Then R species and reporter_rna
    not (x[0].startswith('G') and not x[0].startswith('GA')   # Then G species (except GA)
        or x[0] == 'reporter'), 
    x[0].startswith('GA')                                     # GA species last
))
GA_species_list = ['GAI']
# build general species list based on the name of cme_species_list except GAI 
general_species_list = [species for species in cme_species_list if species not in GA_species_list]


# Compare species between RDME and CME models
# Extract species names from RDME data
rdme_species_names = set( species for species in rdme_species_data.keys())
# print(rdme_species_names)
# Extract species names from CME data
cme_species_names = set(species[0] if isinstance(species, list) else species 
                        for species in general_species_list)

# Find species in RDME but not in CME
rdme_only_species = rdme_species_names - cme_species_names
if rdme_only_species:
    logging.info(f"Species in RDME but not in CME: {sorted(rdme_only_species)}")

# Find species in CME but not in RDME
cme_only_species = cme_species_names - rdme_species_names
if cme_only_species:
    logging.info(f"Species in CME but not in RDME: {sorted(cme_only_species)}")

# Note: Species naming may differ between models. Manual mapping might be needed.
# Example mapping dictionary for differently named species
# species_mapping = {
#     # 'rdme_name': 'cme_name',
#     'G80d_G3i': 'G80G3i', 
#     'Grep': 'reporter', 
#     'Rrep': 'reporter_rna'
#     # Add mappings as needed based on the differences found
# }
logging.info("Note: Species may have different names in RDME vs CME models. Check for potential matches.")

# logging.info("CME species list:")
# logging.info(general_species_list)
# logging.info(GA_species_list)
# logging.info(f"total number of species: {len(GA_species_list)} + {len(general_species_list)}")
'''
================================================================================================
This is the sections to load and save the total statistics for CME
================================================================================================
'''

# In[6]:


avg_list_general = []
var_list_general = []
time_list_general = []

avg_list_GA = []
var_list_GA = []
time_list_GA = []

# Also store raw trajectories for min/max calculation
raw_traj_general = []
raw_traj_GA = []

for species in general_species_list:
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

# this unit conversion somehow not working for GAI
if len(GA_species_list) == 1:
    species = GA_species_list[0]
    avg, var, times = PostProcessing.getAvgVarTrace(cme_traj, species)
    count2concentration = 4.65e-8  #molecule/cell to mM
    avg_list_GA.append(avg*count2concentration)
    var_list_GA.append(np.sqrt(var)*count2concentration)
    time_list_GA.append(times)
    
    # Get raw trajectories for GA species
    raw_trajs = []
    for i in range(PostProcessing.getNumTrajectories(cme_traj)):
        traj_data = PostProcessing.getTrajectory(cme_traj, i, species) * count2concentration
        raw_trajs.append(traj_data)
    raw_traj_GA.append(raw_trajs)
else:
    for species in GA_species_list:
        avg, var, times = PostProcessing.getAvgVarTrace(cme_traj, species)
        count2concentration = 4.65e-8  #molecule/cell to mM
        avg_list_GA.append(avg*count2concentration)
        var_list_GA.append(np.sqrt(var)*count2concentration)
        time_list_GA.append(times)
        
        raw_trajs = []
        for i in range(PostProcessing.getNumTrajectories(cme_traj)):
            traj_data = PostProcessing.getTrajectory(cme_traj, i, species) * count2concentration
            raw_trajs.append(traj_data)
        raw_traj_GA.append(raw_trajs)

# Calculate min/max for CME results        
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
    for pattern in total_species_pattern.values():
        cme_pattern = pattern[cme_serach_key]
        if cme_pattern in species_name and 'D'+cme_pattern not in species_name:
            pattern_used.append(cme_pattern)
        
            
    # Calculate min/max across all trajectories
    trajectories_array = np.array(raw_trajs)
    
    min_vals = np.min(trajectories_array, axis=0)
    max_vals = np.max(trajectories_array, axis=0)
    
    cme_results.append({
        'Species': species_name,
        'Time': ','.join(map(str, times)),
        'Average': ','.join(map(str, avg)),
        'Std': ','.join(map(str, std)),
        'Min': ','.join(map(str, min_vals)),
        'Max': ','.join(map(str, max_vals))
    })
    if len(pattern_used) > 0:
        for pattern in pattern_used:
            mult_factor = 1
            if pattern == 'G4':
                double_species = ['G4d']
            elif pattern == 'G80':
                double_species = ['G80d', 'G80Cd', 'G80G3i']
            else:
                double_species = []
            for double_species in double_species:
                if double_species in species_name:
                    mult_factor = 2
                    break
            if pattern not in cme_total_trajs.keys():
                cme_total_trajs[pattern] = trajectories_array * mult_factor
            else:
                cme_total_trajs[pattern] += trajectories_array * mult_factor
        print(f"species {species_name} has pattern {pattern_used}")
# breakpoint()
# Calculate min/max for CME total results
for pattern in cme_total_trajs.keys():
    cme_total_traj = cme_total_trajs[pattern]
    cme_total_avg = np.mean(cme_total_traj, axis=0)
    cme_total_std = np.std(cme_total_traj, axis=0)
    cme_total_min = np.min(cme_total_traj, axis=0)
    cme_total_max = np.max(cme_total_traj, axis=0)
    cme_final_mean = np.mean(cme_total_avg[-1])
    print(f"final value for {pattern} is {cme_final_mean}")
    cme_total_results.append({
        'Species': f"{pattern}_total",
        'Time': ','.join(map(str, times)),
        'Average': ','.join(map(str, cme_total_avg)),
        'Std': ','.join(map(str, cme_total_std)),
        'Min': ','.join(map(str, cme_total_min)),
        'Max': ','.join(map(str, cme_total_max))
    })


# In[7]:

'''
================================================================================================
This section is for saving the statistics to CSV files
================================================================================================
'''
# Save to CSV files
rdme_df = pd.DataFrame(rdme_results)
rdme_region_df = pd.DataFrame(rdme_results_region)
cme_df = pd.DataFrame(cme_results)
rdme_total_df = pd.DataFrame(rdme_total_results)
cme_total_df = pd.DataFrame(cme_total_results)
rdme_csv_path = os.path.join(fig_dir, 'rdme_species_statistics.csv')
rdme_region_csv_path = os.path.join(fig_dir, 'rdme_region_statistics.csv')
cme_csv_path = os.path.join(fig_dir, 'cme_species_statistics.csv')
rdme_total_csv_path = os.path.join(fig_dir, 'rdme_total_statistics.csv')
cme_total_csv_path = os.path.join(fig_dir, 'cme_total_statistics.csv')
rdme_df.to_csv(rdme_csv_path, index=False)
rdme_region_df.to_csv(rdme_region_csv_path, index=False)
cme_df.to_csv(cme_csv_path, index=False)
rdme_total_df.to_csv(rdme_total_csv_path, index=False)
cme_total_df.to_csv(cme_total_csv_path, index=False)
logging.info(f"RDME statistics saved to: {rdme_csv_path}")
logging.info(f"RDME region statistics saved to: {rdme_region_csv_path}")
logging.info(f"CME statistics saved to: {cme_csv_path}")
logging.info(f"RDME total statistics saved to: {rdme_total_csv_path}")
logging.info(f"CME total statistics saved to: {cme_total_csv_path}")


# In[ ]:


def calculate_pvalue_timeseries(data1_trajectories, data2_trajectories, test_type='ttest'):
    """
    Calculate p-values for time series data comparison.
    
    Args:
        data1_trajectories: List of trajectories for dataset 1
        data2_trajectories: List of trajectories for dataset 2
        test_type: 'ttest' for t-test, 'ks' for Kolmogorov-Smirnov test
    
    Returns:
        Array of p-values for each timepoint
    """
    n_timepoints = len(data1_trajectories[0])
    p_values = np.zeros(n_timepoints)
    
    for t in range(n_timepoints):
        # Extract values at timepoint t from all trajectories
        values1 = [traj[t] for traj in data1_trajectories]
        values2 = [traj[t] for traj in data2_trajectories]
        
        if test_type == 'ttest':
            _, p_val = ttest_ind(values1, values2)
        elif test_type == 'ks':
            _, p_val = ks_2samp(values1, values2)
        else:
            raise ValueError("test_type must be 'ttest' or 'ks'")
        
        p_values[t] = p_val
    
    return p_values



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
    Create a p-value significance plot with vertical lines marking α=0.05 crossings.
    """
    # Use the same figure size as the main comparison script (10, 4)
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot p-values on log scale
    ax.semilogy(time, p_values, 'b-', linewidth=1.5, label='p-values')
    
    # Add significance threshold lines with same colors as main script
    colors_sig = ['red', 'orange', 'green']
    for i, alpha in enumerate(significance_levels):
        ax.axhline(y=alpha, color=colors_sig[i], linestyle='-', alpha=0.7,
                  label=f'α = {alpha}')
    
    # Find and mark α=0.05 crossings
    crossings = find_alpha_05_crossings(time, p_values)
    
    for t_cross in crossings:
        # Add vertical line at crossing time
        ax.axvline(x=t_cross, color='green', linestyle='--', alpha=0.8, linewidth=2)
        
        # Mark the time on x-axis
        ax.annotate(f'{t_cross:.1f}', 
                   xy=(t_cross, ax.get_ylim()[0]), 
                   xytext=(0, -20), 
                   textcoords='offset points',
                   ha='center', va='top',
                   fontsize=10, color='green', weight='bold',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1))
    
    # Print crossing times to console
    if crossings:
        print(f"α=0.05 crossings for {species_name}: {[f'{t:.1f}min' for t in crossings]}")
    
    # Customize plot with same style as main script
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('p-value (log scale)')
    test_name = 'T-test' if test_type == 'ttest' else 'Kolmogorov-Smirnov test'
    ax.set_title(f'{test_name}: {species_name} ({label1} vs {label2})')
    ax.legend(framealpha=0.3, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([1e-10, 1])
    
    # Save figure with same DPI as main script
    if fig_dir:
        fig_name = f'{species_name}_pvalue_{test_type}.png'
        fig_path = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved p-value plot: {fig_name}")
    
    plt.close()

# Create p-value plots directory
pvalue_dir = os.path.join(fig_dir, 'pvalue_plots')
os.makedirs(pvalue_dir, exist_ok=True)


# plot comparison graphs, this part can run separately

# In[8]:
'''
================================================================================================
This section is for comparing the species between RDME and CME for each species, non-region specific
================================================================================================
'''

# Read the saved statistics
rdme_df = pd.read_csv(os.path.join(fig_dir, 'rdme_species_statistics.csv'))
cme_df = pd.read_csv(os.path.join(fig_dir, 'cme_species_statistics.csv'))

# Function to convert string of comma-separated values to numpy array
def str_to_array(s):
    return np.array([float(x) for x in s.split(',')])

# Debug: Print available species
print("Available species in RDME:", rdme_df['Species'].tolist())
print("Available species in CME:", cme_df['Species'].tolist())

# plot based on CME species
cme_species = set(cme_df['Species'].unique())



# In[ ]:


for species_name in cme_species:
    species_mapping = {
        # 'cme_name': ['rdme_name1', 'rdme_name2'] - map to multiple RDME names for separate graphs
        'G80G3i': ['G80d_G3i'], 
        'reporter': ['Grep'], 
        'reporter_rna': ['Rrep', 'ribosomeRrep'],
        'R2': ['R2', 'ribosomeR2'],
        'R3': ['R3', 'ribosomeR3'],
        'R4': ['R4', 'ribosomeR4'],
        'R80': ['R80', 'ribosomeR80'],
        'R1': ['R1', 'ribosomeR1'],
        'G80': ['G80_nucleoplasm'],  # Create separate graphs for each mapping
        'G80C': ['G80_cytoplasm'],
        'G80d': ['G80d_nucleoplasm'],
        'G80Cd': ['G80d_cytoplasm'],
        'G2': ['G2', 'G2_plasmaMembrane', 'G2_cytoplasm'],  # Create separate graphs for each mapping
        'G4': ['G4_nucleoplasm', 'G4_cytoplasm'],
        'G4d': ['G4d_nucleoplasm', 'G4d_cytoplasm']
        # Add mappings as needed based on the differences found
    }
    
    # Get the list of RDME species to check for this CME species
    rdme_species_list = [species_name]  # Default to the same name
    if species_name in species_mapping:
        rdme_species_list = species_mapping[species_name]
    
    # Function to extract the actual species name after RDME_ or ODE_ prefix
    def extract_species_name(full_name):
        if full_name.startswith('RDME_') or full_name.startswith('ODE_'):
            return full_name.split('_', 1)[1]  # Split only on the first underscore
        return full_name
    
    # Create a separate plot for each RDME mapping
    for rdme_species_to_check in rdme_species_list:
        fig, ax = plt.subplots()
        
        # Match species after the RDME_ or ODE_ prefix
        matching_rows = rdme_df[rdme_df['Species'].apply(
            lambda x: extract_species_name(x) == rdme_species_to_check
        )]

        # If multiple rows are found, prioritize the one starting with 'RDME'
        if not matching_rows.empty:
            rdme_species_rows = matching_rows[matching_rows['Species'].str.startswith('RDME')]
            # If no match starts with 'RDME', default to the full list of matches
            if rdme_species_rows.empty:
                rdme_species_rows = matching_rows
        else:
            rdme_species_rows = pd.DataFrame()  # Empty DataFrame if no matches
        
        cme_species_rows = cme_df[cme_df['Species'] == species_name]
        
        if len(rdme_species_rows) == 0 or len(cme_species_rows) == 0:
            print(f"Skipping {species_name} -> {rdme_species_to_check} - data not found")
            plt.close()
            continue
            
        er_data = rdme_species_rows.iloc[0]
        noer_data = cme_species_rows.iloc[0]
        
        time = str_to_array(er_data['Time'])
        er_avg = str_to_array(er_data['Average'])
        er_min = str_to_array(er_data['Min'])
        er_max = str_to_array(er_data['Max'])
        noer_avg = str_to_array(noer_data['Average'])
        noer_min = str_to_array(noer_data['Min'])
        noer_max = str_to_array(noer_data['Max'])
        
        # Extract the part after underscore for display
        display_name = species_name
        if species_name == 'GAI':
            er_avg = er_avg / NAV * 1e3
            er_min = er_min / NAV * 1e3
            er_max = er_max / NAV * 1e3
        
        
        # Check if this is a gene species (starts with DG) - if so, don't plot min/max
        is_gene_species = "DG" in species_name
        
        ax.plot(time, noer_avg, label=f'CME-ODE', linestyle='-', color=cme_color)
        if not is_gene_species:
            ax.fill_between(time, noer_min, noer_max, alpha=0.2, color=cme_color)
        # Plot using publication style colors with min/max fill_between
        ax.plot(time, er_avg, label=f'RDME-ODE', linestyle='-', color=rdme_color)
        if not is_gene_species:
            ax.fill_between(time, er_min, er_max, alpha=0.2, color=rdme_color)
        # Customize plot
        ax.set_xlabel('Time (min)')
        if species_name == 'GAI':
            ax.set_ylabel('Concentration (mM)')
        elif "DG" in species_name:
            ax.set_ylabel('Probability')
        else:
            ax.set_ylabel('Counts')
        # ax.set_title(f'{display_name} Comparison')
        # Legend removed - using separate legend figure
        ax.grid(False)
        
        # Save figure with same DPI as main script (300)
        fig_path = os.path.join(fig_dir, f'{species_name}_vs_{rdme_species_to_check}_comparison.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot for {display_name} vs {rdme_species_to_check}")
        plt.close()
        
        # Calculate p-values and create p-value plots if raw trajectory data is available
        try:
            # Try to get raw trajectory data for p-value calculation
            rdme_species_key = rdme_species_to_check
            cme_species_key = species_name
            
            # Get RDME trajectories
            rdme_trajectories = None
            if rdme_species_key in rdme_species_data:
                rdme_trajectories = rdme_species_data[rdme_species_key]
            elif rdme_species_key in rdme_species_region_data:
                # For region-specific species, we need to handle differently
                parts = rdme_species_key.split('_')
                if len(parts) >= 2:
                    species_part = parts[0]
                    region_part = '_'.join(parts[1:])
                    if species_part in rdme_species_region_data and region_part in rdme_species_region_data[species_part]:
                        rdme_trajectories = rdme_species_region_data[species_part][region_part]
            elif rdme_species_key in rdme_ode_data:
                rdme_trajectories = rdme_ode_data[rdme_species_key]
                
            # Get CME trajectories from cached data
            cme_trajectories = None
            cme_species_key = species_name
            for species, raw_trajs in zip(all_species, all_raw_trajs):
                species_key = species[0] if isinstance(species, list) else species
                if species_key == cme_species_key:
                    cme_trajectories = raw_trajs
                    break
            
            if rdme_trajectories is not None and cme_trajectories is not None:
                # Calculate p-values
                p_values = calculate_pvalue_timeseries(rdme_trajectories, cme_trajectories, test_type='ttest')
                
                # Create p-value plot
                create_pvalue_plot(time, p_values, f'{species_name}_vs_{rdme_species_to_check}', 
                                 'RDME-ODE', 'CME-ODE', fig_dir=pvalue_dir, test_type='ttest')
                
        except Exception as e:
            print(f"Could not create p-value plot for {species_name} vs {rdme_species_to_check}: {e}")

print(f"\nPlots saved in: {fig_dir}")
print(f"P-value plots saved in: {pvalue_dir}")



'''
This section is for comparing the species between RDME and CME for each species, all regions combined
'''
# Create combined G2 species plot
fig, ax = plt.subplots()

# List of species to combine
g2_species = ['G2', 'G2GAE', 'G2GAI']

# Initialize arrays for RDME and CME data
rdme_combined_avg = None
rdme_combined_min = None
rdme_combined_max = None
rdme_pm_combined_avg = None
rdme_pm_combined_min = None
rdme_pm_combined_max = None
cme_combined_avg = None
cme_combined_min = None
cme_combined_max = None
time = None

# For tracking which species are actually used
rdme_species_used = []
rdme_pm_species_used = []
cme_species_used = []

# For p-value calculation - collect raw trajectory data
rdme_combined_trajectories = []
cme_combined_trajectories = []
# Load region-specific data from CSV
rdme_region_df = pd.read_csv(os.path.join(fig_dir, 'rdme_region_statistics.csv'))
# Combine RDME data (all regions - using RDME data)
for species_name in g2_species:
 
    g2_rows = rdme_region_df[(rdme_region_df['Species'] == species_name) ]
    
    if len(g2_rows) > 0:
        # Iterate through all regions for this species
        for idx in range(len(g2_rows)):
            rdme_g2_data = g2_rows.iloc[idx]
            # Track which species are being used
            rdme_species_used.append(f"{rdme_g2_data['Species']}_{rdme_g2_data['Region']}")
            
            curr_g2_avg = str_to_array(rdme_g2_data['Average'])
            curr_g2_min = str_to_array(rdme_g2_data['Min'])
            curr_g2_max = str_to_array(rdme_g2_data['Max'])

            if rdme_combined_avg is None:
                time = str_to_array(rdme_g2_data['Time'])
                rdme_combined_avg = curr_g2_avg
                rdme_combined_min = curr_g2_min
                rdme_combined_max = curr_g2_max
            else:
                rdme_combined_avg += curr_g2_avg
                rdme_combined_min += curr_g2_min  # Sum of minimums
                rdme_combined_max += curr_g2_max  # Sum of maximums



# Combine RDME plasma membrane data from region statistics
for species_name in g2_species:
    # Look for species with Region = plasmaMembrane
    pm_rows = rdme_region_df[(rdme_region_df['Species'] == species_name) & 
                            (rdme_region_df['Region'] == 'plasmaMembrane')]
    
    if len(pm_rows) > 0:
        er_pm_data = pm_rows.iloc[0]
        # Track which species are being used
        rdme_pm_species_used.append(f"{er_pm_data['Species']}_{er_pm_data['Region']}")
        
        curr_pm_avg = str_to_array(er_pm_data['Average'])
        curr_pm_min = str_to_array(er_pm_data['Min'])
        curr_pm_max = str_to_array(er_pm_data['Max'])
        
        if rdme_pm_combined_avg is None:
            rdme_pm_combined_avg = curr_pm_avg
            rdme_pm_combined_min = curr_pm_min
            rdme_pm_combined_max = curr_pm_max
        else:
            rdme_pm_combined_avg += curr_pm_avg
            rdme_pm_combined_min += curr_pm_min  # Sum of minimums
            rdme_pm_combined_max += curr_pm_max  # Sum of maximums

# Combine CME data
for species_name in g2_species:
    cme_species_data = cme_df[cme_df['Species'] == species_name]
    
    if len(cme_species_data) > 0:
        noer_data = cme_species_data.iloc[0]
        # Track which species are being used
        cme_species_used.append(noer_data['Species'])
        
        curr_avg = str_to_array(noer_data['Average'])
        curr_min = str_to_array(noer_data['Min'])
        curr_max = str_to_array(noer_data['Max'])
        
        # Get raw trajectory data for p-value calculation
        for species, raw_trajs in zip(all_species, all_raw_trajs):
            species_key = species[0] if isinstance(species, list) else species
            if species_key == species_name:
                if len(cme_combined_trajectories) == 0:
                    cme_combined_trajectories = [traj[:] for traj in raw_trajs]
                else:
                    for i, traj in enumerate(raw_trajs):
                        if i < len(cme_combined_trajectories):
                            cme_combined_trajectories[i] = [a + b for a, b in zip(cme_combined_trajectories[i], traj)]
                break
        
        if cme_combined_avg is None:
            cme_combined_avg = curr_avg
            cme_combined_min = curr_min
            cme_combined_max = curr_max
        else:
            cme_combined_avg += curr_avg
            cme_combined_min += curr_min  # Sum of minimums
            cme_combined_max += curr_max  # Sum of maximums

# Print which species were actually used
print("RDME species used in G2 total (all regions):", rdme_species_used)
print("RDME species used in G2 total (plasma membrane):", rdme_pm_species_used)
print("CME species used in G2 total:", cme_species_used)

# Plot using publication style colors with min/max fill_between
if rdme_combined_avg is not None and time is not None:
    ax.plot(time, rdme_combined_avg, label='RDME-ODE (all regions)', linestyle='-', color=colors[2])
    ax.fill_between(time, rdme_combined_min, rdme_combined_max, alpha=0.2, color=colors[2])

# Plot plasma membrane data if available
if rdme_pm_combined_avg is not None and time is not None:
    ax.plot(time, rdme_pm_combined_avg, label='RDME-ODE (plasma membrane)', linestyle='-', color=rdme_color)
    ax.fill_between(time, rdme_pm_combined_min, rdme_pm_combined_max, alpha=0.2, color=rdme_color)

if cme_combined_avg is not None and time is not None:
    ax.plot(time, cme_combined_avg, label='CME-ODE', linestyle='-', color=cme_color)
    ax.fill_between(time, cme_combined_min, cme_combined_max, alpha=0.2, color=cme_color)

# Customize plot with same style as main script
ax.set_xlabel('Time (min)')
ax.set_ylabel('Counts')
# Remove title to match main script style
# ax.set_title('Total G2 Species Comparison (G2 + G2GAE + G2GAI)')
# Legend removed - using separate legend figure
ax.grid(False)

# Save figure with same DPI as main script
plt.tight_layout()
fig_path = os.path.join(fig_dir, 'G2_total_plasma_membrane_comparison.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Saved combined G2 total plot")
plt.close()

# Calculate and plot p-values for G2 total comparison
try:
    if len(rdme_combined_trajectories) > 0 and len(cme_combined_trajectories) > 0 and time is not None:
        # Calculate p-values
        p_values = calculate_pvalue_timeseries(rdme_combined_trajectories, cme_combined_trajectories, test_type='ttest')
        
        # Create p-value plot
        create_pvalue_plot(time, p_values, 'G2_total', 'RDME-ODE', 'CME-ODE', fig_dir=pvalue_dir, test_type='ttest')
        print(f"Saved G2 total p-value plot")
except Exception as e:
    print(f"Could not create p-value plot for G2 total: {e}")

# Calculate and plot p-values for plasma membrane comparison with CME-ODE
try:
    if rdme_pm_combined_avg is not None and len(cme_combined_trajectories) > 0 and time is not None:
        # Collect plasma membrane trajectory data
        rdme_pm_combined_trajectories = []
        
        # Get raw trajectory data for plasma membrane species
        for species_name in g2_species:
            # Look for species with Region = plasmaMembrane
            if species_name in rdme_species_region_data and 'plasmaMembrane' in rdme_species_region_data[species_name]:
                species_trajectories = rdme_species_region_data[species_name]['plasmaMembrane']
                if len(rdme_pm_combined_trajectories) == 0:
                    rdme_pm_combined_trajectories = [traj[:] for traj in species_trajectories]
                else:
                    for i, traj in enumerate(species_trajectories):
                        if i < len(rdme_pm_combined_trajectories):
                            rdme_pm_combined_trajectories[i] = [a + b for a, b in zip(rdme_pm_combined_trajectories[i], traj)]
        
        if len(rdme_pm_combined_trajectories) > 0:
            # Calculate p-values comparing plasma membrane RDME-ODE with CME-ODE
            p_values_pm = calculate_pvalue_timeseries(rdme_pm_combined_trajectories, cme_combined_trajectories, test_type='ttest')
            
            # Create p-value plot for plasma membrane comparison
            create_pvalue_plot(time, p_values_pm, 'G2_plasmaMembrane_vs_CME', 'RDME-ODE (plasma membrane)', 'CME-ODE', fig_dir=pvalue_dir, test_type='ttest')
            print(f"Saved G2 plasma membrane vs CME-ODE p-value plot")
        else:
            print("No plasma membrane trajectory data found for p-value calculation")
            
except Exception as e:
    print(f"Could not create p-value plot for plasma membrane vs CME-ODE: {e}")

'''
================================================================================================
This section is for ploting GAI species comparison
================================================================================================
'''


# Create combined GAI species plot
fig, ax = plt.subplots()

# List of species to combine
gai_species = ['GAI', 'G1GAI', 'G3i', 'G2GAI']

# Initialize arrays for RDME and CME data
rdme_combined_avg = None
rdme_combined_min = None
rdme_combined_max = None
cme_combined_avg = None
cme_combined_min = None
cme_combined_max = None
time = None

# For tracking which species are actually used
rdme_species_used = []
cme_species_used = []

# For p-value calculation - collect raw trajectory data
rdme_combined_trajectories = []
cme_combined_trajectories = []

# Combine RDME data
for species_name in gai_species:
    # Match species after the RDME_ or ODE_ prefix
    matching_rows = rdme_df[rdme_df['Species'].apply(
        lambda x: extract_species_name(x) == species_name
    )]
  
    if not matching_rows.empty:
        
        
        rdme_species_data = matching_rows
            
        if len(rdme_species_data) > 0:
            er_data = rdme_species_data.iloc[0]
            # Track which species are being used
            rdme_species_used.append(er_data['Species'])
            
            curr_avg = str_to_array(er_data['Average'])
            curr_min = str_to_array(er_data['Min'])
            curr_max = str_to_array(er_data['Max'])
            # Convert counts to mM
            curr_avg = curr_avg / NAV * 1e3  # NAV*1e3 for RDME conversion
            curr_min = curr_min / NAV * 1e3
            curr_max = curr_max / NAV * 1e3
            
            # Get raw trajectory data for p-value calculation
            rdme_species_key = extract_species_name(er_data['Species'])
            # Try different data sources
            species_trajectories = None
            if rdme_species_key in rdme_species_data:
                species_trajectories = rdme_species_data[rdme_species_key]
            elif rdme_species_key in rdme_ode_data:
                species_trajectories = rdme_ode_data[rdme_species_key]
            
            if species_trajectories is not None:
                # Convert trajectories to mM
                converted_trajectories = [[val / NAV * 1e3 for val in traj] for traj in species_trajectories]
                if len(rdme_combined_trajectories) == 0:
                    rdme_combined_trajectories = [traj[:] for traj in converted_trajectories]
                else:
                    for i, traj in enumerate(converted_trajectories):
                        if i < len(rdme_combined_trajectories):
                            rdme_combined_trajectories[i] = [a + b for a, b in zip(rdme_combined_trajectories[i], traj)]
            
            if rdme_combined_avg is None:
                time = str_to_array(er_data['Time'])
                rdme_combined_avg = curr_avg
                rdme_combined_min = curr_min
                rdme_combined_max = curr_max
            else:
                rdme_combined_avg += curr_avg
                rdme_combined_min += curr_min  # Sum of minimums
                rdme_combined_max += curr_max  # Sum of maximums

# Combine CME data
for species_name in gai_species:
    cme_species_data = cme_df[cme_df['Species'] == species_name]
    
    if len(cme_species_data) > 0:
        noer_data = cme_species_data.iloc[0]
        # Track which species are being used
        cme_species_used.append(noer_data['Species'])
        
        curr_avg = str_to_array(noer_data['Average'])
        curr_min = str_to_array(noer_data['Min'])
        curr_max = str_to_array(noer_data['Max'])
        # Convert counts to mM if not already converted
        if species_name != 'GAI':
            count2concentration = 4.65e-8  # molecule/cell to mM
            curr_avg = curr_avg * count2concentration
            curr_min = curr_min * count2concentration
            curr_max = curr_max * count2concentration
        
        # Get raw trajectory data for p-value calculation
        for species, raw_trajs in zip(all_species, all_raw_trajs):
            species_key = species[0] if isinstance(species, list) else species
            if species_key == species_name:
                # Convert trajectories to mM if needed
                if species_name != 'GAI':
                    converted_trajectories = [[val * count2concentration for val in traj] for traj in raw_trajs]
                else:
                    converted_trajectories = raw_trajs
                    
                if len(cme_combined_trajectories) == 0:
                    cme_combined_trajectories = [traj[:] for traj in converted_trajectories]
                else:
                    for i, traj in enumerate(converted_trajectories):
                        if i < len(cme_combined_trajectories):
                            cme_combined_trajectories[i] = [a + b for a, b in zip(cme_combined_trajectories[i], traj)]
                break
        
        if cme_combined_avg is None:
            cme_time = str_to_array(noer_data['Time'])
            cme_combined_avg = curr_avg
            cme_combined_min = curr_min
            cme_combined_max = curr_max
        else:
            cme_combined_avg += curr_avg
            cme_combined_min += curr_min  # Sum of minimums
            cme_combined_max += curr_max  # Sum of maximums

# Print which species were actually used
print("RDME species used in GAI total:", rdme_species_used)
print("CME species used in GAI total:", cme_species_used)

if cme_combined_avg is not None:
    cme_time_to_use = cme_time if 'cme_time' in locals() else time
    if cme_time_to_use is not None:
        ax.plot(cme_time_to_use, cme_combined_avg, label='CME-ODE', linestyle='-', color=cme_color)
        ax.fill_between(cme_time_to_use, cme_combined_min, cme_combined_max, alpha=0.2, color=cme_color)
# Plot using publication style colors with min/max fill_between
if rdme_combined_avg is not None and time is not None:
    ax.plot(time, rdme_combined_avg, label='RDME-ODE', linestyle='-', color=rdme_color)
    ax.fill_between(time, rdme_combined_min, rdme_combined_max, alpha=0.2, color=rdme_color)



# Add horizontal line for GAE = 11.1mM using publication style color
if time is not None:
    ax.axhline(y=11.1, color='gray', linestyle='--', label='GAE')
    ax.text(time[0]*1.05, 10.8, '11.1 mM', color='gray',  va='top', ha='left')
# Customize plot
ax.set_xlabel('Time (min)')
ax.set_ylabel('Concentration (mM)')
# ax.set_title('Total GAI Species Comparison (GAI + G1GAI + G3i + G2GAI)')
# Legend removed - using separate legend figure
ax.grid(False)

# Save figure
# plt.tight_layout()
fig_path = os.path.join(fig_dir, 'GAI_total_comparison.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Saved combined GAI total plot")
plt.close()

# Calculate and plot p-values for GAI total comparison
try:
    if len(rdme_combined_trajectories) > 0 and len(cme_combined_trajectories) > 0 and time is not None:
        # Calculate p-values
        p_values = calculate_pvalue_timeseries(rdme_combined_trajectories, cme_combined_trajectories, test_type='ttest')

        # Create p-value plot
        create_pvalue_plot(time, p_values, 'GAI_total', 'RDME-ODE', 'CME-ODE', fig_dir=pvalue_dir, test_type='ttest')
        print(f"Saved GAI total p-value plot")
except Exception as e:
    print(f"Could not create p-value plot for GAI total: {e}")




# In[ ]:
'''
================================================================================================
This section creates comparison plots for total species from rdme_total_statistics.csv and cme_total_statistics.csv
================================================================================================
'''

# Read the total statistics CSVs
rdme_total_df = pd.read_csv(rdme_total_csv_path)
cme_total_df = pd.read_csv(cme_total_csv_path)

print("Available total species in RDME:", rdme_total_df['Species'].tolist())
print("Available total species in CME:", cme_total_df['Species'].tolist())

# Get unique species patterns (excluding the _total suffix and RDME_ prefix)
rdme_total_species = set()
for species in rdme_total_df['Species']:
    # Remove 'RDME_' prefix and '_total' suffix
    pattern = species.replace('RDME_', '').replace('_total', '')
    rdme_total_species.add(pattern)

cme_total_species = set()
for species in cme_total_df['Species']:
    # Remove '_total' suffix
    pattern = species.replace('_total', '')
    cme_total_species.add(pattern)

# Find common species patterns
common_patterns = rdme_total_species.intersection(cme_total_species)
print(f"\nCommon total species patterns: {sorted(common_patterns)}")

# Create comparison plots for each common pattern
for pattern in sorted(common_patterns):
    fig, ax = plt.subplots()

    # Get RDME total data
    rdme_species_name = f'RDME_{pattern}_total'
    rdme_rows = rdme_total_df[rdme_total_df['Species'] == rdme_species_name]

    # Get CME total data
    cme_species_name = f'{pattern}_total'
    cme_rows = cme_total_df[cme_total_df['Species'] == cme_species_name]

    if len(rdme_rows) == 0 or len(cme_rows) == 0:
        print(f"Skipping {pattern} - data not found")
        plt.close()
        continue

    rdme_data = rdme_rows.iloc[0]
    cme_data = cme_rows.iloc[0]

    # Extract time series data
    time = str_to_array(rdme_data['Time'])
    rdme_avg = str_to_array(rdme_data['Average'])
    rdme_min = str_to_array(rdme_data['Min'])
    rdme_max = str_to_array(rdme_data['Max'])

    cme_avg = str_to_array(cme_data['Average'])
    cme_min = str_to_array(cme_data['Min'])
    cme_max = str_to_array(cme_data['Max'])

    # Plot CME data
    ax.plot(time, cme_avg, label='CME-ODE', linestyle='-', color=cme_color)
    ax.fill_between(time, cme_min, cme_max, alpha=0.2, color=cme_color)

    # Plot RDME data
    ax.plot(time, rdme_avg, label='RDME-ODE', linestyle='-', color=rdme_color)
    ax.fill_between(time, rdme_min, rdme_max, alpha=0.2, color=rdme_color)

    # Customize plot
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Counts')
    # Legend removed - using separate legend figure
    ax.grid(False)

    # Save figure
    fig_path = os.path.join(fig_dir, f'{pattern}_total_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved total comparison plot for {pattern}")
    plt.close()

    # Try to create p-value plots if trajectory data is available
    try:
        # Get RDME trajectories for this total pattern
        rdme_total_trajectories = None
        if pattern in rdme_total_results_traj:
            rdme_total_trajectories = rdme_total_results_traj[pattern]

        # Get CME trajectories for this total pattern
        cme_total_trajectories = None
        if pattern in cme_total_trajs:
            cme_total_trajectories = cme_total_trajs[pattern]

        if rdme_total_trajectories is not None and cme_total_trajectories is not None:
            # Calculate p-values
            p_values = calculate_pvalue_timeseries(rdme_total_trajectories, cme_total_trajectories, test_type='ttest')

            # Create p-value plot
            create_pvalue_plot(time, p_values, f'{pattern}_total',
                             'RDME-ODE', 'CME-ODE', fig_dir=pvalue_dir, test_type='ttest')
            print(f"Saved p-value plot for {pattern}_total")
    except Exception as e:
        print(f"Could not create p-value plot for {pattern}_total: {e}")

print(f"\nTotal species comparison plots saved in: {fig_dir}")

'''
================================================================================================
This section creates a separate legend figure for use in 2x2 figure layouts
================================================================================================
'''

# Create a horizontal legend figure that can be placed at the bottom of a 2x2 layout
fig_legend, ax_legend = plt.subplots(figsize=(6, 0.5))
ax_legend.set_axis_off()

# Create dummy plot elements for the legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    (Line2D([0], [0], color=cme_color, linewidth=2), 
     Patch(facecolor=cme_color, alpha=0.2, edgecolor=cme_color),
     'CME-ODE'),
    (Line2D([0], [0], color=rdme_color, linewidth=2),
     Patch(facecolor=rdme_color, alpha=0.2, edgecolor=rdme_color),
     'RDME-ODE'),
]

# Create combined legend handles (line + patch)
handles = []
labels = []
for line, patch, label in legend_elements:
    handles.append((line, patch))
    labels.append(label)

# Use a simpler approach - just lines with markers
legend_handles = [
    Line2D([0], [0], color=cme_color, linewidth=3, label='CME-ODE'),
    Line2D([0], [0], color=rdme_color, linewidth=3, label='RDME-ODE'),
]

ax_legend.legend(handles=legend_handles, 
                 loc='center', 
                 ncol=2, 
                 frameon=True, 
                 framealpha=0.8,
                 fontsize=12,
                 columnspacing=3.0,
                 handlelength=2.5)

plt.tight_layout()
legend_path = os.path.join(fig_dir, 'legend_separate.png')
plt.savefig(legend_path, dpi=300, bbox_inches='tight', transparent=True)
print(f"\nSaved separate legend figure: {legend_path}")
plt.close()

# Also create a version with shading indicators
fig_legend2, ax_legend2 = plt.subplots(figsize=(8, 0.8))
ax_legend2.set_axis_off()

# Create custom legend with both line and shading representation
from matplotlib.patches import Rectangle

# Draw custom legend items manually
legend_items = [
    {'color': cme_color, 'label': 'CME-ODE', 'x': 0.15},
    {'color': rdme_color, 'label': 'RDME-ODE', 'x': 0.55},
]

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
legend_path2 = os.path.join(fig_dir, 'legend_separate_with_shading.png')
plt.savefig(legend_path2, dpi=300, bbox_inches='tight', transparent=True)
print(f"Saved separate legend figure with shading: {legend_path2}")
plt.close()
