#!/usr/bin/env python
# coding: utf-8

# Ribosome Distance Statistical Significance Analysis
# This script analyzes statistical significance between ribosome distance datasets

import h5py
import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
from functools import partial
import logging
from scipy.spatial.distance import euclidean
from scipy import stats
from matplotlib_pub_figure import setup_publication_style
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Predefined parameters
RIBOSOME_SPECIES = ["ribosomeR1", "ribosomeR2", "ribosomeR3", "ribosomeR4", "ribosomeGrep", "ribosomeR80"]
NUCLEUS_CENTER = [131, 76, 110]  # Default nucleus center coordinates
SPACING = 28.8  # nm/cube lattice spacing

def get_idx(name, species_names):
    """Get species index from species names list"""
    try:
        return int(species_names.index(name) + 1)
    except ValueError:
        return None

def extract_all_ribosome_distances(file_path, ribosome_species=None, nucleus_center=None):
    """
    Extract ALL individual ribosome distances (not averaged) from a single .lm file
    
    Returns:
    --------
    dict: Dictionary with time points as keys and lists of all individual distances as values
    """
    if ribosome_species is None:
        ribosome_species = RIBOSOME_SPECIES
    if nucleus_center is None:
        nucleus_center = NUCLEUS_CENTER
    
    try:
        traj = h5py.File(file_path, 'r')
        
        # Get species names
        species_names_bin = traj['Parameters']['SpeciesNames'][:]
        species_names = [name[0] for name in species_names_bin]
        
        # Check if ribosome species exist
        valid_species = [spec for spec in ribosome_species if spec in species_names]
        if not valid_species:
            logging.warning(f"No valid ribosome species found in {file_path}")
            return {}
        
        # Get time points
        times = traj['Simulations']['0000001']['LatticeTimes'][:]
        all_distances_over_time = {}
        
        for t_idx, t in enumerate(times):
            t_int = int(t)
            t_str = f'{t_int:010d}'
            
            # Get lattice data
            try:
                lattice = np.array(traj['Simulations']['0000001']['Lattice'][t_str])
            except KeyError:
                continue
            
            all_distances = []
            
            # Process each valid ribosome species
            for spec in valid_species:
                spec_idx = get_idx(spec, species_names)
                if spec_idx is not None:
                    coords = np.argwhere(lattice == spec_idx)
                    if len(coords) > 0:
                        coords = coords[:, :3]  # Take only x, y, z coordinates
                        for coord in coords:
                            distance = euclidean(coord, nucleus_center)
                            all_distances.append(distance)
            
            # Store all individual distances for this time point
            if all_distances:
                all_distances_over_time[t_int] = all_distances
        
        traj.close()
        return all_distances_over_time
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return {}

def process_files_for_significance(file_paths, ribosome_species=None, nucleus_center=None, max_workers=6):
    """Process multiple .lm files to collect all individual distance measurements"""
    num_cores = min(multiprocessing.cpu_count(), max_workers)
    logging.info(f"Processing {len(file_paths)} files using {num_cores} CPU cores for significance analysis")
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        process_func = partial(
            extract_all_ribosome_distances, 
            ribosome_species=ribosome_species,
            nucleus_center=nucleus_center
        )
        results = pool.map(process_func, file_paths)
    
    return dict(zip(file_paths, results))

def combine_all_distances_by_time(file_results):
    """Combine all distance measurements across files, organized by time"""
    combined_data = {}
    
    for file_path, time_distances in file_results.items():
        for time, distances in time_distances.items():
            if time not in combined_data:
                combined_data[time] = []
            combined_data[time].extend(distances)
    
    return combined_data

def calculate_significance_by_time(data1, data2, data3=None):
    """
    Calculate statistical significance between datasets at each time point
    
    Returns:
    --------
    dict: Dictionary with comparison pairs as keys and time-series p-values as values
    """
    significance_results = {}
    
    # Get common time points
    common_times = set(data1.keys()) & set(data2.keys())
    if data3:
        common_times = common_times & set(data3.keys())
    
    common_times = sorted(list(common_times))
    
    # Compare data1 vs data2
    p_values_1v2 = {}
    effect_sizes_1v2 = {}
    
    for time in common_times:
        distances1 = np.array(data1[time]) * SPACING * 1e-3  # Convert to µm
        distances2 = np.array(data2[time]) * SPACING * 1e-3  # Convert to µm
        
        # Perform statistical tests
        # 1. Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(distances1, distances2, alternative='two-sided')
        p_values_1v2[time] = p_value
        
        # 2. Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(distances1) - 1) * np.var(distances1, ddof=1) + 
                             (len(distances2) - 1) * np.var(distances2, ddof=1)) / 
                            (len(distances1) + len(distances2) - 2))
        cohens_d = (np.mean(distances1) - np.mean(distances2)) / pooled_std
        effect_sizes_1v2[time] = cohens_d
    
    significance_results['1v2'] = {
        'p_values': p_values_1v2,
        'effect_sizes': effect_sizes_1v2,
        'times': common_times
    }
    
    # Compare data1 vs data3 if available
    if data3:
        common_times_1v3 = sorted(list(set(data1.keys()) & set(data3.keys())))
        p_values_1v3 = {}
        effect_sizes_1v3 = {}
        
        for time in common_times_1v3:
            distances1 = np.array(data1[time]) * SPACING * 1e-3
            distances3 = np.array(data3[time]) * SPACING * 1e-3
            
            statistic, p_value = stats.mannwhitneyu(distances1, distances3, alternative='two-sided')
            p_values_1v3[time] = p_value
            
            pooled_std = np.sqrt(((len(distances1) - 1) * np.var(distances1, ddof=1) + 
                                 (len(distances3) - 1) * np.var(distances3, ddof=1)) / 
                                (len(distances1) + len(distances3) - 2))
            cohens_d = (np.mean(distances1) - np.mean(distances3)) / pooled_std
            effect_sizes_1v3[time] = cohens_d
        
        significance_results['1v3'] = {
            'p_values': p_values_1v3,
            'effect_sizes': effect_sizes_1v3,
            'times': common_times_1v3
        }
        
        # Compare data2 vs data3
        common_times_2v3 = sorted(list(set(data2.keys()) & set(data3.keys())))
        p_values_2v3 = {}
        effect_sizes_2v3 = {}
        
        for time in common_times_2v3:
            distances2 = np.array(data2[time]) * SPACING * 1e-3
            distances3 = np.array(data3[time]) * SPACING * 1e-3
            
            statistic, p_value = stats.mannwhitneyu(distances2, distances3, alternative='two-sided')
            p_values_2v3[time] = p_value
            
            pooled_std = np.sqrt(((len(distances2) - 1) * np.var(distances2, ddof=1) + 
                                 (len(distances3) - 1) * np.var(distances3, ddof=1)) / 
                                (len(distances2) + len(distances3) - 2))
            cohens_d = (np.mean(distances2) - np.mean(distances3)) / pooled_std
            effect_sizes_2v3[time] = cohens_d
        
        significance_results['2v3'] = {
            'p_values': p_values_2v3,
            'effect_sizes': effect_sizes_2v3,
            'times': common_times_2v3
        }
    
    return significance_results

def create_significance_plots(significance_results, labels, fig_dir):
    """Create separate plots for p-values and effect sizes"""
    # Setup publication style
    colors = setup_publication_style(figure_size='medium', dpi=300)
    comparison_colors = {'1v2': '#E69F00', '1v3': '#56B4E9', '2v3': '#009E73'}
    
    # Create subplot figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: P-values over time
    for comparison, data in significance_results.items():
        times_min = [t/60 for t in data['times']]  # Convert to minutes
        p_values = [data['p_values'][t] for t in data['times']]
        
        # Create comparison label
        if comparison == '1v2':
            comp_label = f"{labels[0]} vs {labels[1]}"
        elif comparison == '1v3':
            comp_label = f"{labels[0]} vs {labels[2]}"
        elif comparison == '2v3':
            comp_label = f"{labels[1]} vs {labels[2]}"
        
        ax1.plot(times_min, p_values, 'o-', label=comp_label, 
                color=comparison_colors.get(comparison, 'black'), markersize=4)
    
    # Add significance thresholds
    ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    ax1.axhline(y=0.01, color='red', linestyle=':', alpha=0.7, label='p=0.01')
    ax1.axhline(y=0.001, color='red', linestyle='-', alpha=0.7, label='p=0.001')
    
    ax1.set_yscale('log')
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('P-value (log scale)')
    ax1.legend(framealpha=0.3, loc='best')
    ax1.grid(False)
    ax1.set_title('Statistical Significance (Mann-Whitney U test)')
    
    # Plot 2: Effect sizes over time  
    for comparison, data in significance_results.items():
        times_min = [t/60 for t in data['times']]  # Convert to minutes
        effect_sizes = [data['effect_sizes'][t] for t in data['times']]
        
        # Create comparison label
        if comparison == '1v2':
            comp_label = f"{labels[0]} vs {labels[1]}"
        elif comparison == '1v3':
            comp_label = f"{labels[0]} vs {labels[2]}"
        elif comparison == '2v3':
            comp_label = f"{labels[1]} vs {labels[2]}"
        
        ax2.plot(times_min, effect_sizes, 'o-', label=comp_label, 
                color=comparison_colors.get(comparison, 'black'), markersize=4)
    
    # Add effect size reference lines
    ax2.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Medium effect')
    ax2.axhline(y=0.8, color='gray', linestyle='-', alpha=0.5, label='Large effect')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel("Cohen's d (Effect Size)")
    ax2.legend(framealpha=0.3, loc='best')
    ax2.grid(False)
    ax2.set_title('Effect Size Analysis')
    
    # Save figure
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, 'ribosome_distance_significance.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved significance analysis plot: ribosome_distance_significance.png")
    plt.close()
    
    # Create summary statistics
    create_significance_summary(significance_results, labels, fig_dir)

def create_significance_summary(significance_results, labels, fig_dir):
    """Create a summary table of significant time points"""
    summary_data = []
    
    for comparison, data in significance_results.items():
        if comparison == '1v2':
            comp_name = f"{labels[0]} vs {labels[1]}"
        elif comparison == '1v3':
            comp_name = f"{labels[0]} vs {labels[2]}"
        elif comparison == '2v3':
            comp_name = f"{labels[1]} vs {labels[2]}"
        
        # Count significant time points
        p_values = list(data['p_values'].values())
        times = data['times']
        
        sig_001 = sum(1 for p in p_values if p < 0.001)
        sig_01 = sum(1 for p in p_values if p < 0.01)
        sig_05 = sum(1 for p in p_values if p < 0.05)
        total_points = len(p_values)
        
        # Average effect size
        avg_effect_size = np.mean(list(data['effect_sizes'].values()))
        
        summary_data.append({
            'Comparison': comp_name,
            'Total Time Points': total_points,
            'p < 0.001': f"{sig_001} ({sig_001/total_points*100:.1f}%)",
            'p < 0.01': f"{sig_01} ({sig_01/total_points*100:.1f}%)",
            'p < 0.05': f"{sig_05} ({sig_05/total_points*100:.1f}%)",
            'Avg Effect Size': f"{avg_effect_size:.3f}"
        })
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(fig_dir, 'significance_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"Saved significance summary: significance_summary.csv")
    
    # Print summary
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)

def get_user_input():
    """Get user input for directories and comparison settings"""
    print("\n=== Ribosome Distance Significance Analysis Setup ===")
    traj_dir1 = input("Enter path to first trajectory directory: ")
    traj_dir2 = input("Enter path to second trajectory directory: ")
    
    # Ask if user wants to compare a third directory
    compare_third = input("Do you want to compare a third trajectory directory? (yes/no): ").lower() == 'yes'
    traj_dir3 = None
    if compare_third:
        traj_dir3 = input("Enter path to third trajectory directory: ")
    
    label1 = input("Enter label for first trajectory: ")
    label2 = input("Enter label for second trajectory: ")
    label3 = None
    if compare_third:
        label3 = input("Enter label for third trajectory: ")
    
    # Ask where to save plots
    while True:
        save_options = f"1 ({label1}), 2 ({label2})"
        if compare_third:
            save_options += f", 3 ({label3})"
        
        save_location = input(f"Save plots under directory: {save_options}? (1/2{'/3' if compare_third else ''}): ").strip()
        if save_location == '1':
            fig_dir = os.path.join(traj_dir1, 'ribosome_distance_comparison/')
            break
        elif save_location == '2':
            fig_dir = os.path.join(traj_dir2, 'ribosome_distance_comparison/')
            break
        elif compare_third and save_location == '3':
            fig_dir = os.path.join(traj_dir3, 'ribosome_distance_comparison/')
            break
        else:
            print(f"Please enter either '1', '2'{' or 3' if compare_third else ''}")
    
    labels = [label1, label2]
    if label3:
        labels.append(label3)
    
    return traj_dir1, traj_dir2, traj_dir3, labels, fig_dir

def main():
    """Main function to run significance analysis"""
    # Get user input
    traj_dir1, traj_dir2, traj_dir3, labels, fig_dir = get_user_input()
    
    # Create output directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    # Configure logging
    log_file = os.path.join(fig_dir, 'significance_analysis_log.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Running significance analysis between:")
    logging.info(f"Directory 1 ({labels[0]}): {traj_dir1}")
    logging.info(f"Directory 2 ({labels[1]}): {traj_dir2}")
    if traj_dir3:
        logging.info(f"Directory 3 ({labels[2]}): {traj_dir3}")
    
    # Process each directory - collect ALL individual distances
    print("Processing directory 1...")
    files1 = [os.path.join(traj_dir1, f) for f in os.listdir(traj_dir1) if f.endswith('.lm')]
    results1 = process_files_for_significance(files1)
    data1 = combine_all_distances_by_time(results1)
    
    print("Processing directory 2...")
    files2 = [os.path.join(traj_dir2, f) for f in os.listdir(traj_dir2) if f.endswith('.lm')]
    results2 = process_files_for_significance(files2)
    data2 = combine_all_distances_by_time(results2)
    
    data3 = None
    if traj_dir3:
        print("Processing directory 3...")
        files3 = [os.path.join(traj_dir3, f) for f in os.listdir(traj_dir3) if f.endswith('.lm')]
        results3 = process_files_for_significance(files3)
        data3 = combine_all_distances_by_time(results3)
    
    if not data1 or not data2:
        logging.error("Could not process data from one or more directories")
        return
    
    print("Calculating statistical significance...")
    # Calculate significance
    significance_results = calculate_significance_by_time(data1, data2, data3)
    
    # Create significance plots
    create_significance_plots(significance_results, labels, fig_dir)
    
    logging.info(f"\nSignificance analysis completed. Results saved in: {fig_dir}")

if __name__ == "__main__":
    main()