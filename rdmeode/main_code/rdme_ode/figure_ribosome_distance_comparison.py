#!/usr/bin/env python
# coding: utf-8

# Ribosome Distance to Nucleus Comparison Script
# This code compares ribosome distances to nucleus center between different trajectory directories

import h5py
import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
from functools import partial
import logging
import glob
from scipy.spatial.distance import euclidean
from scipy.stats import ttest_ind, ks_2samp
from matplotlib_pub_figure import setup_publication_style
from tqdm import tqdm
import time
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Predefined parameters
RIBOSOME_SPECIES = ["ribosomeR1", "ribosomeR2", "ribosomeR3", "ribosomeR4", "ribosomeGrep", "ribosomeR80"]
# RIBOSOME_SPECIES = ["ribosomeR2"]
NUCLEUS_CENTER = [131, 76, 110]  # Default nucleus center coordinates
SPACING = 28.8  # nm/cube lattice spacing

def get_idx(name, species_names):
    """Get species index from species names list"""
    try:
        return int(species_names.index(name) + 1)
    except ValueError:
        return None

def extract_ribosome_coordinates(args):
    """
    Extract ribosome coordinates and calculate distances to nucleus center from a single .lm file
    
    Parameters:
    -----------
    args : tuple
        (file_path, ribosome_species, nucleus_center, progress_queue)
        
    Returns:
    --------
    dict: Dictionary with time points as keys and average distances as values
    """
    file_path, ribosome_species, nucleus_center, progress_queue = args
    if ribosome_species is None:
        ribosome_species = RIBOSOME_SPECIES
    if nucleus_center is None:
        nucleus_center = NUCLEUS_CENTER
    
    # Report start of processing
    filename = os.path.basename(file_path)
    if progress_queue:
        progress_queue.put(f"Loading: {filename}")
    
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
        total_frames = len(times)
        avg_distances_over_time = {}
        ribosome_counts_over_time = {}
        
        # Report total frames for this file
        if progress_queue:
            progress_queue.put(f"Frames: {filename} {total_frames}")
        
        for t_idx, t in enumerate(times):
            t_int = int(t)
            t_str = f'{t_int:010d}'
            
            # Get lattice data
            try:
                lattice = np.array(traj['Simulations']['0000001']['Lattice'][t_str])
            except KeyError:
                continue
            
            all_distances = []
            total_ribosomes = 0
            
            # Process each valid ribosome species
            for spec in valid_species:
                spec_idx = get_idx(spec, species_names)
                if spec_idx is not None:
                    coords = np.argwhere(lattice == spec_idx)
                    if len(coords) > 0:
                        coords = coords[:, :3]  # Take only x, y, z coordinates
                        total_ribosomes += len(coords)
                        for coord in coords:
                            distance = euclidean(coord, nucleus_center)
                            all_distances.append(distance)
            
            # Store counts and distances for this time point
            ribosome_counts_over_time[t_int] = total_ribosomes
            if all_distances:
                avg_distances_over_time[t_int] = np.mean(all_distances)
            
            # Report frame progress (every 360 frames = ~10% increments for 3601-frame files)
            if progress_queue and (t_idx + 1) % 360 == 0:
                progress_queue.put(f"Frame: {filename} {t_idx + 1}/{total_frames}")
        
        traj.close()
        
        # Report completion
        if progress_queue:
            progress_queue.put(f"Completed: {filename}")
        
        return {'distances': avg_distances_over_time, 'counts': ribosome_counts_over_time}
        
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        if progress_queue:
            progress_queue.put(f"Error: {filename}")
        return {'distances': {}, 'counts': {}}

def progress_monitor(progress_queue, total_files):
    """Monitor progress and display updates"""
    completed = 0
    loading = set()
    file_frames = {}  # Track total frames per file
    current_frames = {}  # Track current frame progress per file
    
    with tqdm(total=total_files, desc="Processing trajectory files", unit="file") as pbar:
        while completed < total_files:
            try:
                message = progress_queue.get(timeout=2)  # Increased timeout
                if message.startswith("Loading:"):
                    filename = message.split("Loading: ")[1]
                    loading.add(filename)
                    logging.info(f"Loading trajectory: {filename}")
                elif message.startswith("Frames:"):
                    parts = message.split(" ")
                    filename = parts[1]
                    total_frames = int(parts[2])
                    file_frames[filename] = total_frames
                    current_frames[filename] = 0
                    logging.info(f"Trajectory {filename}: {total_frames} frames to process")
                elif message.startswith("Frame:"):
                    parts = message.split(" ")
                    filename = parts[1]
                    frame_progress = parts[2]  # e.g., "50/100"
                    current_frame, total_frame = map(int, frame_progress.split("/"))
                    current_frames[filename] = current_frame
                    log_msg = f"Processing {filename}: frame {current_frame}/{total_frame}"
                    logging.info(log_msg)
                    # Force immediate output
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - INFO - {log_msg}", flush=True)
                elif message.startswith("Completed:"):
                    filename = message.split("Completed: ")[1]
                    if filename in loading:
                        loading.remove(filename)
                    completed += 1
                    pbar.update(1)
                    if filename in file_frames:
                        logging.info(f"Completed trajectory: {filename} ({file_frames[filename]} frames)")
                    else:
                        logging.info(f"Completed trajectory: {filename}")
                elif message.startswith("Error:"):
                    filename = message.split("Error: ")[1]
                    if filename in loading:
                        loading.remove(filename)
                    completed += 1
                    pbar.update(1)
                    logging.error(f"Error processing trajectory: {filename}")
            except:
                # Timeout or other error, continue
                continue

def process_files_parallel(file_paths, ribosome_species=None, nucleus_center=None, max_workers=128):
    """Process multiple .lm files in parallel with progress tracking"""
    num_cores = min(multiprocessing.cpu_count(), max_workers)
    logging.info(f"Processing {len(file_paths)} files using {num_cores} CPU cores")
    
    # Create progress queue for communication between processes
    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    
    # Start progress monitor in a separate process
    monitor_process = multiprocessing.Process(
        target=progress_monitor, 
        args=(progress_queue, len(file_paths))
    )
    monitor_process.start()
    
    # Prepare arguments for each worker
    args_list = [
        (file_path, ribosome_species, nucleus_center, progress_queue) 
        for file_path in file_paths
    ]
    
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(extract_ribosome_coordinates, args_list)
    
    # Wait for progress monitor to finish
    monitor_process.join()
    
    return dict(zip(file_paths, results))

def get_dir_cache_key(traj_dir):
    """Create a unique cache key for a trajectory directory"""
    # Get modification times of all .lm files in directory
    lm_files = [f for f in os.listdir(traj_dir) if f.endswith('.lm')]
    if not lm_files:
        return None
    
    # Create hash from directory path and file modification times
    hash_input = traj_dir
    for f in sorted(lm_files):
        file_path = os.path.join(traj_dir, f)
        mtime = os.path.getmtime(file_path)
        hash_input += f"{f}_{mtime}"
    
    return hashlib.md5(hash_input.encode()).hexdigest()[:16]

def save_data_to_csv(data, cache_dir, cache_key, label):
    """Save processed distance and count data to CSV files"""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save distance data
    distance_file = os.path.join(cache_dir, f"{cache_key}_{label}_distances.csv")
    distance_rows = []
    for time, stats in data['distances'].items():
        distance_rows.append({
            'time_seconds': time,
            'mean_distance': stats['mean'],
            'min_distance': stats['min'],
            'max_distance': stats['max'],
            'std_distance': stats['std']
        })
    
    if distance_rows:
        pd.DataFrame(distance_rows).to_csv(distance_file, index=False)
        logging.info(f"Saved distance data to {distance_file}")
    
    # Save count data
    count_file = os.path.join(cache_dir, f"{cache_key}_{label}_counts.csv")
    count_rows = []
    for time, stats in data['counts'].items():
        count_rows.append({
            'time_seconds': time,
            'mean_count': stats['mean'],
            'min_count': stats['min'],
            'max_count': stats['max'],
            'std_count': stats['std']
        })
    
    if count_rows:
        pd.DataFrame(count_rows).to_csv(count_file, index=False)
        logging.info(f"Saved count data to {count_file}")

def load_data_from_csv(cache_dir, cache_key, label):
    """Load processed distance and count data from CSV files"""
    distance_file = os.path.join(cache_dir, f"{cache_key}_{label}_distances.csv")
    count_file = os.path.join(cache_dir, f"{cache_key}_{label}_counts.csv")
    
    if not (os.path.exists(distance_file) and os.path.exists(count_file)):
        return None
    
    try:
        # Load distance data
        distance_df = pd.read_csv(distance_file)
        distances = {}
        for _, row in distance_df.iterrows():
            distances[int(row['time_seconds'])] = {
                'mean': row['mean_distance'],
                'min': row['min_distance'],
                'max': row['max_distance'],
                'std': row['std_distance']
            }
        
        # Load count data
        count_df = pd.read_csv(count_file)
        counts = {}
        for _, row in count_df.iterrows():
            counts[int(row['time_seconds'])] = {
                'mean': row['mean_count'],
                'min': row['min_count'],
                'max': row['max_count'],
                'std': row['std_count']
            }
        
        logging.info(f"Loaded cached data for {label}")
        return {'distances': distances, 'counts': counts}
    
    except Exception as e:
        logging.warning(f"Failed to load cached data for {label}: {e}")
        return None

def calculate_pvalue_timeseries(data1_trajectories, data2_trajectories, test_type='ttest'):
    """
    Calculate p-values for each time point comparing two sets of trajectories.
    
    Parameters:
    data1_trajectories: list of arrays, each array is a trajectory from dataset 1
    data2_trajectories: list of arrays, each array is a trajectory from dataset 2
    test_type: 'ttest' for t-test, 'ks' for Kolmogorov-Smirnov test
    
    Returns:
    p_values: array of p-values for each time point
    """
    data1_array = np.array(data1_trajectories)
    data2_array = np.array(data2_trajectories)
    
    # Use minimum number of timepoints to avoid index out of bounds
    n_timepoints = min(data1_array.shape[1], data2_array.shape[1])
    
    if data1_array.shape[1] != data2_array.shape[1]:
        logging.warning(f"Trajectories have different lengths: data1={data1_array.shape[1]}, data2={data2_array.shape[1]}. Using first {n_timepoints} timepoints.")
    
    p_values = np.zeros(n_timepoints)
    
    for t in range(n_timepoints):
        values1 = data1_array[:, t]
        values2 = data2_array[:, t]
        
        # Filter out NaN values independently for each dataset
        values1_valid = values1[~np.isnan(values1)]
        values2_valid = values2[~np.isnan(values2)]
        
        # Check if we have enough valid data points for comparison
        if len(values1_valid) < 2 or len(values2_valid) < 2:
            # Not enough valid data points for comparison
            p_values[t] = np.nan
            continue
        
        if test_type == 'ttest':
            _, p_val = ttest_ind(values1_valid, values2_valid)
        elif test_type == 'ks':
            _, p_val = ks_2samp(values1_valid, values2_valid)
        else:
            raise ValueError("test_type must be 'ttest' or 'ks'")
            
        p_values[t] = p_val
    
    return p_values

def create_pvalue_plot(time, p_values, comparison_name, label1, label2, label3=None, fig_dir=None, 
                      significance_levels=[0.001, 0.01, 0.05], test_type='ttest'):
    """
    Create a separate p-value significance plot.
    """
    setup_publication_style(figure_size='medium', dpi=300)
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Filter out NaN values for plotting
    valid_mask = ~np.isnan(p_values)
    if np.any(valid_mask):
        time_valid = np.array(time)[valid_mask]
        p_values_valid = p_values[valid_mask]
        
        # Plot p-values over time
        ax.plot(time_valid, p_values_valid, 'k-', linewidth=2, label='p-value')
        
        # Fill areas below significance levels (only for valid data)
        colors = ['red', 'orange', 'yellow']
        for i, sig_level in enumerate(significance_levels):
            if i == 0:  # Most stringent level
                mask = (p_values_valid <= sig_level)
                ax.fill_between(time_valid, 0, sig_level, where=mask, 
                              color=colors[i], alpha=0.2, interpolate=True)
            else:
                prev_level = significance_levels[i-1]
                mask = (p_values_valid <= sig_level) & (p_values_valid > prev_level)
                ax.fill_between(time_valid, prev_level, sig_level, where=mask,
                              color=colors[i], alpha=0.2, interpolate=True)
    else:
        logging.warning(f"No valid p-values to plot for {comparison_name}")
    
    # Add horizontal lines for significance levels
    colors = ['red', 'orange', 'yellow']
    for i, sig_level in enumerate(significance_levels):
        ax.axhline(y=sig_level, color=colors[i], linestyle='--', alpha=0.7, 
                   label=f'p = {sig_level}')
    
    # Set y-axis to log scale for better visualization
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1)
    
    # Labels and formatting
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('p-value (log scale)')
    
    # Create title based on comparison
    if label3:
        ax.set_title(f'Statistical Significance: {comparison_name} ({label1} vs {label2} vs {label3})', fontsize=12)
    else:
        ax.set_title(f'Statistical Significance: {comparison_name} ({label1} vs {label2})', fontsize=12)
    
    ax.legend(framealpha=0.3, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add text box with test information
    test_name = 'T-test' if test_type == 'ttest' else 'Kolmogorov-Smirnov test'
    ax.text(0.02, 0.98, f'Test: {test_name}', transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    if fig_dir:
        pvalue_dir = os.path.join(fig_dir, 'pvalue_plots')
        os.makedirs(pvalue_dir, exist_ok=True)
        filename = f'{comparison_name}_pvalue_significance.png'
        fig_path = os.path.join(pvalue_dir, filename)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved p-value plot: {filename}")
    
    plt.close()

def create_three_way_pvalue_plot(time, p_values_12, p_values_13, p_values_23, comparison_name, 
                                label1, label2, label3, fig_dir=None, 
                                significance_levels=[0.001, 0.01, 0.05], test_type='ttest'):
    """
    Create a three-way p-value comparison plot for three datasets.
    """
    setup_publication_style(figure_size='medium', dpi=300)
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
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1)
    
    # Labels and formatting
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('p-value (log scale)')
    ax.set_title(f'Pairwise Statistical Significance: {comparison_name}', fontsize=12)
    ax.legend(framealpha=0.3, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add test information
    test_name = 'T-test' if test_type == 'ttest' else 'Kolmogorov-Smirnov test'
    ax.text(0.02, 0.98, f'Test: {test_name}', transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    if fig_dir:
        pvalue_dir = os.path.join(fig_dir, 'pvalue_plots')
        os.makedirs(pvalue_dir, exist_ok=True)
        filename = f'{comparison_name}_three_way_pvalue_significance.png'
        fig_path = os.path.join(pvalue_dir, filename)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved three-way p-value plot: {filename}")
    
    plt.close()

def get_user_input():
    """Get user input for directories and comparison settings"""
    print("\n=== Ribosome Distance Comparison Setup ===")
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
    
    colors = setup_publication_style(figure_size='medium', dpi=300)
    color_dum = colors[0]
    color_dum2 = colors[1]
    color_dum3 = colors[2]
    # Define colors (from figure_comparison.py color scheme)
    color1 = colors[3]    # Blue for first trajectory
    color2 = colors[4]    # Vermillion/Orange-red for second trajectory  
    color3 = colors[5]    # Bluish green for third trajectory
    
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
    
    return traj_dir1, traj_dir2, traj_dir3, label1, label2, label3, color1, color2, color3, fig_dir

def process_directory_data(traj_dir, label, cache_dir=None, use_cache=True):
    """Process all .lm files in a directory and return distance and count data with caching"""
    logging.info(f"Processing {label} files from {traj_dir}")
    
    # Generate cache key
    cache_key = get_dir_cache_key(traj_dir) if cache_dir else None
    
    # Try to load from cache first
    if use_cache and cache_key and cache_dir:
        cached_data = load_data_from_csv(cache_dir, cache_key, label)
        if cached_data is not None:
            return cached_data, None  # Return None for raw trajectories when using cache
    
    # Find all .lm files
    files = [os.path.join(traj_dir, f) for f in os.listdir(traj_dir) if f.endswith('.lm')]
    if not files:
        logging.warning(f"No .lm files found in {traj_dir}")
        return {'distances': {}, 'counts': {}}, []
    
    # Process files in parallel
    file_results = process_files_parallel(files)
    
    # Combine distance results from all files
    combined_distances = {}
    combined_counts = {}
    raw_distance_trajectories = []
    raw_count_trajectories = []
    
    # First pass: collect all time points to ensure consistent indexing
    all_times = set()
    for file_data in file_results.values():
        all_times.update(file_data['distances'].keys())
    all_times = sorted(all_times)
    
    # Initialize raw trajectory storage
    num_files = len([f for f in file_results.values() if f['distances']])
    if num_files > 0:
        for _ in range(num_files):
            raw_distance_trajectories.append([np.nan] * len(all_times))
            raw_count_trajectories.append([np.nan] * len(all_times))
    
    file_idx = 0
    for file_data in file_results.values():
        if not file_data['distances']:  # Skip files with no data
            continue
            
        # Process distances
        for time_idx, time in enumerate(all_times):
            if time in file_data['distances']:
                distance = file_data['distances'][time]
                if time not in combined_distances:
                    combined_distances[time] = []
                combined_distances[time].append(distance)
                raw_distance_trajectories[file_idx][time_idx] = distance
        
        # Process counts
        for time_idx, time in enumerate(all_times):
            if time in file_data['counts']:
                count = file_data['counts'][time]
                if time not in combined_counts:
                    combined_counts[time] = []
                combined_counts[time].append(count)
                raw_count_trajectories[file_idx][time_idx] = count
                
        file_idx += 1
    
    # Calculate statistics for distances
    processed_distances = {}
    for time, distances in combined_distances.items():
        processed_distances[time] = {
            'mean': np.mean(distances),
            'min': np.min(distances), 
            'max': np.max(distances),
            'std': np.std(distances)
        }
    
    # Calculate statistics for counts
    processed_counts = {}
    for time, counts in combined_counts.items():
        processed_counts[time] = {
            'mean': np.mean(counts),
            'min': np.min(counts), 
            'max': np.max(counts),
            'std': np.std(counts)
        }
    
    processed_data = {'distances': processed_distances, 'counts': processed_counts}
    raw_trajectories = {
        'distances': raw_distance_trajectories,
        'counts': raw_count_trajectories,
        'times': all_times
    }
    
    # Save to cache if requested
    if cache_key and cache_dir:
        save_data_to_csv(processed_data, cache_dir, cache_key, label)
    
    return processed_data, raw_trajectories

def filter_minute_intervals(data_dict):
    """Convert time data to minute intervals and average within each minute"""
    minute_groups = {}
    
    for time_seconds, stats in data_dict.items():
        minute = int(time_seconds / 60)
        
        if minute not in minute_groups:
            minute_groups[minute] = {
                'means': [],
                'mins': [], 
                'maxs': [],
                'stds': []
            }
        
        minute_groups[minute]['means'].append(stats['mean'])
        minute_groups[minute]['mins'].append(stats['min'])
        minute_groups[minute]['maxs'].append(stats['max'])
        minute_groups[minute]['stds'].append(stats['std'])
    
    # Average within each minute
    minute_data = {}
    for minute, values in minute_groups.items():
        minute_data[minute] = {
            'mean': np.mean(values['means']),
            'min': np.mean(values['mins']),
            'max': np.mean(values['maxs']),
            'std': np.mean(values['stds'])
        }
    
    return minute_data

def create_distance_comparison_plot(data1, data2, data3, label1, label2, label3, 
                                   color1, color2, color3, fig_dir):
    """Create distance comparison plot using publication style"""
    # Setup publication style
    setup_publication_style(figure_size='medium', dpi=300)
    
    _, ax = plt.subplots()
    
    # Process data to minute intervals
    data1_minutes = filter_minute_intervals(data1['distances'])
    data2_minutes = filter_minute_intervals(data2['distances'])
    data3_minutes = filter_minute_intervals(data3['distances']) if data3 else {}
    
    # Convert to micrometers and plot
    def plot_data(data_minutes, label, color):
        if not data_minutes:
            return None
            
        times = sorted(data_minutes.keys())
        means = [data_minutes[t]['mean'] * SPACING * 1e-3 for t in times]  # Convert to µm
        mins = [data_minutes[t]['min'] * SPACING * 1e-3 for t in times]
        maxs = [data_minutes[t]['max'] * SPACING * 1e-3 for t in times]
        
        ax.plot(times, means, label=label, linestyle='-', color=color)
        ax.fill_between(times, mins, maxs, alpha=0.1, color=color)
        
        # Calculate overall mean for horizontal line
        overall_mean = sum(means) / len(means) if means else 0
        return overall_mean
    
    # Plot all datasets and collect overall means
    mean1 = plot_data(data1_minutes, label1, color1)
    mean2 = plot_data(data2_minutes, label2, color2)
    mean3 = plot_data(data3_minutes, label3, color3) if data3_minutes and label3 else None
    
    # Add horizontal mean lines
    if mean1 is not None:
        ax.axhline(y=mean1, color=color1, linestyle='--', alpha=0.7, linewidth=1)
    if mean2 is not None:
        ax.axhline(y=mean2, color=color2, linestyle='--', alpha=0.7, linewidth=1)
    if mean3 is not None:
        ax.axhline(y=mean3, color=color3, linestyle='--', alpha=0.7, linewidth=1)
    
    # Customize plot
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Distance to Nucleus Center (µm)')
    ax.legend(framealpha=0.3, loc='best')
    ax.grid(False)
    
    # Adjust layout first to get proper positions
    plt.tight_layout()
    
    # Position labels to the right of the axes to avoid overlap with y-axis label/ticks
    # Use axes coordinates for x: values > 1 place text outside the right edge
    label_x = 1.02  # Slightly to the right of the axes
    if mean1 is not None:
        ax.text(label_x, mean1, f'{mean1:.2f}', transform=ax.get_yaxis_transform(),
                color=color1, fontsize=8, verticalalignment='center', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7),
                clip_on=False)
    
    if mean2 is not None:
        ax.text(label_x, mean2, f'{mean2:.2f}', transform=ax.get_yaxis_transform(),
                color=color2, fontsize=8, verticalalignment='center', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7),
                clip_on=False)
    
    if mean3 is not None:
        ax.text(label_x, mean3, f'{mean3:.2f}', transform=ax.get_yaxis_transform(),
                color=color3, fontsize=8, verticalalignment='center', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7),
                clip_on=False)
    
    # Adjust subplot to make room for labels on the right
    fig = ax.figure
    fig.subplots_adjust(left=0.15, right=0.85)
    
    # Save figure
    fig_path = os.path.join(fig_dir, 'ribosome_distance_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved distance comparison plot: ribosome_distance_comparison.png")
    plt.close()

def create_ribosome_count_comparison_plot(data1, data2, data3, label1, label2, label3, 
                                        color1, color2, color3, fig_dir):
    """Create ribosome count comparison plot using publication style"""
    # Setup publication style
    setup_publication_style(figure_size='medium', dpi=300)
    
    _, ax = plt.subplots()
    
    # Process data to minute intervals
    data1_minutes = filter_minute_intervals(data1['counts'])
    data2_minutes = filter_minute_intervals(data2['counts'])
    data3_minutes = filter_minute_intervals(data3['counts']) if data3 else {}
    
    # Plot ribosome counts
    def plot_count_data(data_minutes, label, color):
        if not data_minutes:
            return
            
        times = sorted(data_minutes.keys())
        means = [data_minutes[t]['mean'] for t in times]
        mins = [data_minutes[t]['min'] for t in times]
        maxs = [data_minutes[t]['max'] for t in times]
        
        ax.plot(times, means, label=label, linestyle='-', color=color)
        ax.fill_between(times, mins, maxs, alpha=0.1, color=color)
    
    # Plot all datasets
    plot_count_data(data1_minutes, label1, color1)
    plot_count_data(data2_minutes, label2, color2)
    if data3_minutes and label3:
        plot_count_data(data3_minutes, label3, color3)
    
    # Customize plot
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Total Translating Ribosomes')
    ax.legend(framealpha=0.3, loc='best')
    ax.grid(False)
    
    # Save figure
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, 'ribosome_count_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved ribosome count comparison plot: ribosome_count_comparison.png")
    plt.close()

def main():
    """Main function to run the ribosome distance comparison"""
    # Get user input
    traj_dir1, traj_dir2, traj_dir3, label1, label2, label3, color1, color2, color3, fig_dir = get_user_input()
    
    # Create output directory
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    # Setup cache directory
    cache_dir = os.path.join(fig_dir, 'cache')
    
    # Ask user about using cache
    use_cache = input("Do you want to use cached data if available? (yes/no): ").lower() == 'yes'
    
    # Configure logging to file
    log_file = os.path.join(fig_dir, 'ribosome_distance_log.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Comparing ribosome distances between:")
    logging.info(f"Directory 1 ({label1}): {traj_dir1}")
    logging.info(f"Directory 2 ({label2}): {traj_dir2}")
    if traj_dir3:
        logging.info(f"Directory 3 ({label3}): {traj_dir3}")
    
    # Process each directory
    data1, raw1 = process_directory_data(traj_dir1, label1, cache_dir, use_cache)
    data2, raw2 = process_directory_data(traj_dir2, label2, cache_dir, use_cache)
    data3, raw3 = process_directory_data(traj_dir3, label3, cache_dir, use_cache) if traj_dir3 else ({'distances': {}, 'counts': {}}, None)
    
    if not data1['distances'] or not data2['distances']:
        logging.error("Could not process distance data from one or more directories")
        return
    
    # Create distance comparison plot
    create_distance_comparison_plot(data1, data2, data3, label1, label2, label3, 
                                   color1, color2, color3, fig_dir)
    
    # Create ribosome count comparison plot
    create_ribosome_count_comparison_plot(data1, data2, data3, label1, label2, label3, 
                                        color1, color2, color3, fig_dir)
    
    # Create p-value plots if we have raw trajectory data (not using cache)
    if raw1 is not None and raw2 is not None and len(raw1['distances']) > 0 and len(raw2['distances']) > 0:
        logging.info("Creating p-value significance plots...")
        
        # Calculate p-values for distances first to get the actual number of timepoints used
        p_values_dist = calculate_pvalue_timeseries(raw1['distances'], raw2['distances'], test_type='ttest')
        
        # Use the minimum length for time array
        min_timepoints = min(len(raw1['times']), len(raw2['times']), len(p_values_dist))
        times_minutes = [t / 60 for t in raw1['times'][:min_timepoints]]
        
        # Truncate p_values to match if needed
        if len(p_values_dist) > min_timepoints:
            p_values_dist = p_values_dist[:min_timepoints]
        
        # Calculate p-values for counts
        p_values_count = calculate_pvalue_timeseries(raw1['counts'], raw2['counts'], test_type='ttest')
        if len(p_values_count) > min_timepoints:
            p_values_count = p_values_count[:min_timepoints]
        
        if traj_dir3 and raw3 is not None and len(raw3['distances']) > 0:
            # Three-way comparison
            p_values_dist_13 = calculate_pvalue_timeseries(raw1['distances'], raw3['distances'], test_type='ttest')
            p_values_dist_23 = calculate_pvalue_timeseries(raw2['distances'], raw3['distances'], test_type='ttest')
            
            p_values_count_13 = calculate_pvalue_timeseries(raw1['counts'], raw3['counts'], test_type='ttest')
            p_values_count_23 = calculate_pvalue_timeseries(raw2['counts'], raw3['counts'], test_type='ttest')
            
            # Create three-way p-value plots
            create_three_way_pvalue_plot(times_minutes, p_values_dist, p_values_dist_13, p_values_dist_23,
                                        'ribosome_distance_to_nucleus', label1, label2, label3, 
                                        fig_dir=fig_dir, test_type='ttest')
            
            create_three_way_pvalue_plot(times_minutes, p_values_count, p_values_count_13, p_values_count_23,
                                        'ribosome_count', label1, label2, label3, 
                                        fig_dir=fig_dir, test_type='ttest')
        else:
            # Two-way comparison
            create_pvalue_plot(times_minutes, p_values_dist, 'ribosome_distance_to_nucleus', 
                             label1, label2, fig_dir=fig_dir, test_type='ttest')
            
            create_pvalue_plot(times_minutes, p_values_count, 'ribosome_count', 
                             label1, label2, fig_dir=fig_dir, test_type='ttest')
    
    elif use_cache:
        logging.info("P-value plots cannot be created when using cached data (raw trajectory data not available)")
        logging.info("To generate p-value plots, rerun without using cache or delete cached files")
    
    logging.info(f"\nComparison plots saved in: {fig_dir}")
    logging.info("Generated plots:")
    logging.info("- ribosome_distance_comparison.png (Distance to nucleus)")  
    logging.info("- ribosome_count_comparison.png (Total translating ribosomes)")
    if raw1 is not None and raw2 is not None:
        logging.info("- P-value significance plots saved in: pvalue_plots/")

if __name__ == "__main__":
    main()