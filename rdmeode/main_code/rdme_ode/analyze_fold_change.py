#!/usr/bin/env python
# coding: utf-8



import pickle
import os
import numpy as np
from jLM.RDME import File as RDMEFile
import jLM
import json
import logging
from pyLM import *
from pyLM.units import *
from jLM import CMEPostProcessing as PostProcessing
from scipy.stats import ttest_ind
import hashlib
import pandas as pd
from tqdm import tqdm
from traj_analysis_rdme import *

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('jLM').setLevel(logging.WARNING)
logging.getLogger('pyLM').setLevel(logging.WARNING)


# Directories
cme_traj_dir = "/data2/2024_Yeast_GS/my_current_code/my_cme_ode/output/03232025/"
rdme_traj_dir = "/data2/2024_Yeast_GS/my_current_code/rdme_ode_results/20251121_EFFCHROMO_newR2"
fig_dir = os.path.join(rdme_traj_dir, 'figures_foldchange_comparison/')
os.makedirs(fig_dir, exist_ok=True)
# Process files normally (Force Raw Processing)
rdme_files = [f for f in os.listdir(rdme_traj_dir) if f.startswith('yeast') and f.endswith('.lm')]
# Sort files to ensure deterministic order if needed
rdme_files.sort()

cme_files = ['gal_cme_ode_gae11.1mM_11.1_gai0_rep50_delta1_time60.lm']
traj_suff = "_ode.jsonl"

logging.info(f"RDME-ODE files: {len(rdme_files)} files found")
logging.info(f"CME-ODE files: {cme_files}")

# Initialize dictionaries to store data for each species
rdme_species_data = {}
rdme_species_region_data = {}
rdme_ode_data = {}
rdmeTs = None
odeTs = None
NAV = None

# Process RDME files
for traj_file in tqdm(rdme_files, desc="Processing RDME files", unit="file"):
    logging.info(f"Processing RDME file: {traj_file}")
    try:
        traj, odeTraj, region_traj = get_traj(rdme_traj_dir, traj_file, traj_suff, region_suff='_region.jsonl')

        curr_rdmeTs, rdmeYs, curr_odeTs, odeYs, regionTs, regionYs = get_data_for_plot(traj, odeTraj, region_traj=region_traj, sparse_factor=1)
        
        # specific to this dataset likely, but using logic from reference
        if NAV is None:
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
                    
    except Exception as e:
        logging.error(f"Failed to process {traj_file}: {e}")
        continue

# Load CME data
cme_files = ['gal_cme_ode_gae11.1mM_11.1_gai0_rep50_delta1_time60.lm']
cme_traj = PostProcessing.openLMFile(os.path.join(cme_traj_dir + cme_files[0]))

# Logic for aggregation exactly matching figure_RDMECME_compare.py
total_species_pattern = {
    'G1': ('G1', 'G1'),
    'G2': ('G2', 'G2'),
    'G3': ('G3', 'G3'),
    'G4': ('G4', 'G4'),
    'G80': ('G80', 'G80'),
}
rdme_serach_key = 0
cme_serach_key = 1

# --- PROCESS RDME DATA ---
rdme_total_results_traj = {}

for species, trajectories in rdme_species_data.items():
    pattern_used = []
    for pattern_key, pattern_vals in total_species_pattern.items():
        rdme_p = pattern_vals[rdme_serach_key]
        if rdme_p in ['G4', 'G80']:
            dimer_pattern = rdme_p + 'd'
        else:
            dimer_pattern = None
            
        if dimer_pattern is not None and dimer_pattern in species and '_total' not in species:
            pattern_used.append(dimer_pattern)
        elif rdme_p in species and 'D'+rdme_p not in species and '_total' not in species:
            pattern_used.append(rdme_p)

            
    trajectories_array = np.array(trajectories)
    for pattern in pattern_used:
        mult_factor = 1
        if 'd' in pattern:
            p_base = pattern.replace('d', '')
            mult_factor = 2
        else:
            p_base = pattern
            
        if p_base not in rdme_total_results_traj:
            rdme_total_results_traj[p_base] = trajectories_array * mult_factor
        else:
            rdme_total_results_traj[p_base] += trajectories_array * mult_factor

# --- PROCESS CME DATA ---
cme_total_trajs = {}
cme_species_list = PostProcessing.getSpecies(cme_traj)
cme_species_list = sorted(cme_species_list, key=lambda x: (
    not x[0].startswith('DG'),                                # Sort DG species first
    not (x[0].startswith('R') or x[0] == 'reporter_rna'),     # Then R species and reporter_rna
    not (x[0].startswith('G') and not x[0].startswith('GA')   # Then G species (except GA)
        or x[0] == 'reporter'), 
    x[0].startswith('GA')                                     # GA species last
))
print(cme_species_list)

GA_species_list = ['GAI']

# Replicate CME logic from figure_RDMECME_compare.py
for species_info in cme_species_list:
    species_name = species_info[0] if isinstance(species_info, list) else species_info
    
    # Validation and scaling
    avg_official, var_official, times_official = PostProcessing.getAvgVarTrace(cme_traj, species_name)
    num_cme_trajs = PostProcessing.getNumTrajectories(cme_traj)
    
    raw_trajs = []
    factor = 1
    if species_name in GA_species_list:
        factor = 4.65e-8  # molecule/cell to mM
        
    for i in range(num_cme_trajs):
        traj_data = PostProcessing.getTrajectory(cme_traj, i, species_name) * factor
        raw_trajs.append(traj_data)
    
    trajectories_array = np.array(raw_trajs)
    raw_mean = np.mean(trajectories_array, axis=0)
    
    # Check consistency (accounting for GA scaling)
    if not np.allclose(raw_mean, avg_official * factor, atol=1e-8):
        logging.warning(f"CME Validation Failed for {species_name}: raw mean != official average")
  
    
    pattern_used = []
    for pattern_key, pattern_vals in total_species_pattern.items():
        cme_p = pattern_vals[cme_serach_key]
        if cme_p in species_name and 'D'+cme_p not in species_name:
            pattern_used.append(cme_p)
        
  
    if len(pattern_used) > 0:
        for pattern in pattern_used:
            mult_factor = 1
            if pattern == 'G4':
                double_species_list = ['G4d']
            elif pattern == 'G80':
                double_species_list = ['G80d', 'G80Cd', 'G80G3i']
            else:
                double_species_list = []
                
            for d_species in double_species_list:
                if d_species in species_name:
                    mult_factor = 2
                    break
            
            if pattern not in cme_total_trajs:
                cme_total_trajs[pattern] = trajectories_array * mult_factor
            else:
                cme_total_trajs[pattern] += trajectories_array * mult_factor
        print(f"species {species_name} has pattern {pattern_used}") 


# --- ANALYZE FOLD CHANGE ---
results = []
per_traj_results = []

print("\n" + "="*120)
print(f"{'Species':<10} | {'Model':<10} | {'FC (Mean)':<12} | {'95% CI':<20} | {'p-value':<10} | {'Status':<10}")
print("-" * 120)

for species in total_species_pattern.keys():

    rdme_data = rdme_total_results_traj.get(species)
    cme_data = cme_total_trajs.get(species)
    
    if rdme_data is None or cme_data is None:
        print(f"Skipping {species} due to missing data.")
        continue
        
    rdme_ends = rdme_data[:, -1]
    
    cme_starts = cme_data[:, 0]
    cme_ends = cme_data[:, -1]
    
    # Use CME mean initial value as the baseline for both
    cme_start_mean = np.mean(cme_starts)
    cme_end_mean = np.mean(cme_ends)
    print(f"final value for {species} is {cme_end_mean}")
    epsilon = 1e-9
    rdme_fc = rdme_ends / (cme_start_mean + epsilon)
    cme_fc = cme_ends / (cme_start_mean + epsilon)
    
    # Collect per-trajectory data
    for i, fc in enumerate(rdme_fc):
        per_traj_results.append({'Species': species, 'Trajectory': i, 'Model': 'RDME', 'FoldChange': fc})
    for i, fc in enumerate(cme_fc):
        per_traj_results.append({'Species': species, 'Trajectory': i, 'Model': 'CME', 'FoldChange': fc})
    
    # Welch's t-test
    stat, p_val = ttest_ind(rdme_fc, cme_fc, equal_var=False)
    
    # Calculate 95% Confidence Intervals
    def get_ci95(data):
        if len(data) < 2: return (np.nan, np.nan)
        mean = np.mean(data)
        sem = np.std(data, ddof=1) / np.sqrt(len(data))
        return (mean - 1.96 * sem, mean + 1.96 * sem)
    
    rdme_ci = get_ci95(rdme_fc)
    cme_ci = get_ci95(cme_fc)
    
    # Discovery Threshold Check
    sig_status = "STRONG" if p_val < 0.001 else ("MARGINAL" if p_val < 0.05 else "NS")
    
    # Calculate Cohen's d
    n1, n2 = len(rdme_fc), len(cme_fc)
    s1, s2 = np.var(rdme_fc, ddof=1), np.var(cme_fc, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    d = (np.mean(rdme_fc) - np.mean(cme_fc)) / (pooled_std + epsilon)
    
    results.append({
        'Species': species,
        'RDME_FC': np.mean(rdme_fc),
        'RDME_CI95': rdme_ci,
        'CME_FC': np.mean(cme_fc),
        'CME_CI95': cme_ci,
        'p_value': p_val,
        'Significance': sig_status,
        'Cohens_d': d
    })
    
    # Print detailed metrics
    rdme_ci_str = f"[{rdme_ci[0]:.2f}, {rdme_ci[1]:.2f}]"
    cme_ci_str = f"[{cme_ci[0]:.2f}, {cme_ci[1]:.2f}]"
    
    print(f"{species:<10} | {'RDME':<10} | {np.mean(rdme_fc):<12.2f} | {rdme_ci_str:<20} | {'-':<10} | {'-':<10}")
    print(f"{'':<10} | {'CME':<10} | {np.mean(cme_fc):<12.2f} | {cme_ci_str:<20} | {p_val:<10.3e} | {sig_status:<10}")
    print("-" * 120)

# Save to CSVs
pd.DataFrame(results).to_csv(os.path.join(fig_dir, 'fold_change_significance_report.csv'), index=False)
pd.DataFrame(per_traj_results).to_csv(os.path.join(fig_dir, 'per_trajectory_fold_changes.csv'), index=False)

print(f"\nSignificance report saved to {os.path.join(fig_dir, 'fold_change_significance_report.csv')}")
print(f"Per-trajectory fold changes saved to {os.path.join(fig_dir, 'per_trajectory_fold_changes.csv')}")

