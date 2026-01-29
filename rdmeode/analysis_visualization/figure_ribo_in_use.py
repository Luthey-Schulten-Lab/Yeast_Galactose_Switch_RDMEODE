#!/usr/bin/env python
# coding: utf-8

# Plot ribosome in use in one graph 

# In[1]:

from colorspace import qualitative_hcl, desaturate
import os 
from jLM.RDME import File as RDMEFile
import jLM
# traj_dir = "/data2/2024_Yeast_GS/my_current_code/rdme_ode_results/20251031_baseline_newcytoribono"
# traj_dir = "/data2/2024_Yeast_GS/my_current_code/rdme_ode_results/20251101_chromosome_newcytoribono"

traj_dir = "/data2/2024_Yeast_GS/my_current_code/rdme_ode_results/20251121_ER_newR2diff"
# traj_dir = "/data2/2024_Yeast_GS/my_current_code/rdme_ode_results/20251121_EFFCHROMO_newR2"
output_dir = os.path.join(traj_dir, 'ribo_in_translation/')


# In[2]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import hashlib
import fcntl
import errno
import time
from traj_analysis_rdme import *
from matplotlib_pub_figure import setup_publication_style

def is_file_locked(file_path):
    """
    Check if a file is locked by another process.
    
    Parameters:
    - file_path: path to the file to check
    
    Returns:
    - True if file is locked, False if available
    """
    try:
        # Try to open the file and acquire an exclusive lock
        with open(file_path, 'rb') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return False  # File is not locked
    except (IOError, OSError) as e:
        if e.errno == errno.EAGAIN or e.errno == errno.EACCES:
            return True  # File is locked
        # For other errors, assume file is accessible
        return False
    except Exception:
        # If we can't check (e.g., file doesn't exist), assume it's not locked
        return False

def wait_for_file_unlock(file_path, timeout=60, check_interval=2):
    """
    Wait for a file to become unlocked.
    
    Parameters:
    - file_path: path to the file
    - timeout: maximum time to wait in seconds
    - check_interval: how often to check in seconds
    
    Returns:
    - True if file became available, False if timeout
    """
    start_time = time.time()
    while is_file_locked(file_path):
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"Timeout waiting for {os.path.basename(file_path)} to unlock after {timeout}s")
            return False
        
        print(f"File {os.path.basename(file_path)} is locked, waiting... ({elapsed:.1f}s)")
        time.sleep(check_interval)
    
    return True

def is_file_ready(file_path, wait_timeout=30):
    """
    Check if a file is ready for reading (exists, not locked, not being written).
    
    Parameters:
    - file_path: path to check
    - wait_timeout: how long to wait for file to become ready
    
    Returns:
    - True if ready, False if not available
    """
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File does not exist: {os.path.basename(file_path)}")
        return False
    
    # Check if file is locked
    if is_file_locked(file_path):
        print(f"File is locked: {os.path.basename(file_path)}")
        if wait_timeout > 0:
            return wait_for_file_unlock(file_path, timeout=wait_timeout)
        else:
            return False
    
    return True

def plot_ribosomes_cached_with_lock_check(traj_dir, output_dir, normalize=False, force_reload=False, file_timeout=30):
    """
    Plot ribosome trajectories with CSV caching and file lock checking.
    
    Parameters:
    - traj_dir: directory with .lm files
    - output_dir: where to save plots and cache
    - normalize: convert to mM (default: False = counts)
    - force_reload: ignore cache and reprocess (default: False)
    - file_timeout: seconds to wait for locked files (0 = skip immediately)
    """
    
    # Setup cache
    os.makedirs(output_dir, exist_ok=True)
    cache_file = os.path.join(output_dir, f"ribo_cache_{'norm' if normalize else 'count'}.csv")
    
    # Check if we should use cache
    use_cache = (os.path.exists(cache_file) and not force_reload)
    
    if use_cache:
        print("Loading from cache...")
        df = pd.read_csv(cache_file)
        time_data = np.array([float(x) for x in df.columns[1:]])
        trajectories = df.iloc[:, 1:].values
        print(f"Loaded {len(trajectories)} trajectories from cache")
    else:
        print("Processing trajectory files...")
        # Get files
        lm_files = [f for f in os.listdir(traj_dir) if f.startswith('yeast') and f.endswith('.lm')]
        
        trajectories = []
        time_data = None
        skipped_files = []
        locked_files = []
        
        print(f"Found {len(lm_files)} .lm files")
        
        # First, check all files for lock status
        available_files = []
        for traj_file in lm_files:
            file_path = os.path.join(traj_dir, traj_file)
            ode_file_path = file_path + "_ode.jsonl"
            
            # Check both .lm and .jsonl files
            lm_ready = is_file_ready(file_path, wait_timeout=0)  # Quick check first
            ode_ready = is_file_ready(ode_file_path, wait_timeout=0)
            
            if lm_ready and ode_ready:
                available_files.append(traj_file)
            else:
                if not lm_ready:
                    locked_files.append(f"{traj_file} (.lm)")
                if not ode_ready:
                    locked_files.append(f"{traj_file} (.jsonl)")
        
        print(f"Available files: {len(available_files)}")
        if locked_files:
            print(f"Locked/unavailable files: {len(locked_files)}")
            for f in locked_files[:5]:  # Show first 5
                print(f"  - {f}")
            if len(locked_files) > 5:
                print(f"  ... and {len(locked_files) - 5} more")
        
        # Process available files
        for traj_file in tqdm(available_files, desc="Processing files"):
            try:
                file_path = os.path.join(traj_dir, traj_file)
                ode_file_path = file_path + "_ode.jsonl"
                
                # Double-check before processing (with optional wait)
                if not is_file_ready(file_path, wait_timeout=file_timeout):
                    skipped_files.append(f"{traj_file} (.lm locked)")
                    continue
                
                if not is_file_ready(ode_file_path, wait_timeout=file_timeout):
                    skipped_files.append(f"{traj_file} (.jsonl locked)")
                    continue
                
                # File is ready, proceed with processing
                traj, odeTraj, _ = get_traj(traj_dir, traj_file, "_ode.jsonl")
                curr_rdmeTs, rdmeYs, _, _, _, _ = get_data_for_plot(traj, odeTraj, sparse_factor=1)
                
                if time_data is None:
                    time_data = curr_rdmeTs
                
                # Find ribosome species (starting with 'ribo' but not just 'ribosome')
                ribo_species = [s for s in rdmeYs.keys() if s.lower().startswith('ribo') and s.lower() != 'ribosome']
                
                if ribo_species:
                    # Sum all ribosome species
                    ribo_sum = sum(rdmeYs[species] for species in ribo_species)
                    
                    # Normalize if requested
                    if normalize:
                        NAV = 6.022e23 * (traj.reg.cytoplasm.volume + traj.reg.nucleoplasm.volume + traj.reg.plasmaMembrane.volume)
                        ribo_sum = ribo_sum / NAV * 1e3  # Convert to mM
                    
                    trajectories.append(ribo_sum)
                    
            except Exception as e:
                print(f"Error processing {traj_file}: {e}")
                skipped_files.append(f"{traj_file} (processing error)")
        
        # Report results
        print(f"\nProcessing complete:")
        print(f"  Successfully processed: {len(trajectories)} files")
        print(f"  Skipped/failed: {len(skipped_files)} files")
        
        if skipped_files:
            print("Skipped files:")
            for f in skipped_files:
                print(f"  - {f}")
        
        # Save to cache
        if trajectories:
            trajectories = np.array(trajectories)
            df_cache = pd.DataFrame(trajectories, columns=[f't_{t:.1f}' for t in time_data])
            df_cache.to_csv(cache_file, index=False)
            print(f"Cached {len(trajectories)} trajectories to {cache_file}")
    
    # Plot results (same as before)
    if len(trajectories) > 0:
        colors = setup_publication_style(figure_size='medium')
        
        pal = qualitative_hcl(h = [0, 269], c = 52, l = 60)
        ## To get 5 colors simply call
        colors = pal.colors(5)
        trajectories = np.array(trajectories)
        
        # Individual trajectories + average
        fig, ax = plt.subplots()
        for i, traj in enumerate(trajectories):
            ax.plot(time_data, traj, color=colors[i % len(colors)], alpha=0.7, linewidth=1, label=f'Rep {i+1}')
        
        # Average
        avg = np.mean(trajectories, axis=0)
        ax.plot(time_data, avg, color='black', linewidth=2, label='Average')
        
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Concentration (mM)' if normalize else 'Count')
        ax.set_ylim(0, 120)
        # ax.set_title('Ribosomes in Translation')
        ax.legend(framealpha=0.3, loc='best')
        ax.grid(False)
        
        plt.tight_layout()
        fig_path = os.path.join(output_dir, 'ribosomes_plot.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plot saved: {fig_path}")
        return {'time': time_data, 'trajectories': trajectories, 'average': avg}
    else:
        print("No ribosome data found!")
        return None

# Usage examples:

# # Quick run - skip locked files immediately
# print("=== Quick run (skip locked files) ===")
# result = plot_ribosomes_cached_with_lock_check(traj_dir, output_dir, normalize=False, file_timeout=0)

# # Patient run - wait up to 30 seconds for locked files
# print("\n=== Patient run (wait for locked files) ===")
# result = plot_ribosomes_cached_with_lock_check(traj_dir, output_dir, normalize=False, file_timeout=30)

# Force reload with file checking
print("\n=== Force reload with file checking ===")
result = plot_ribosomes_cached_with_lock_check(traj_dir, output_dir, normalize=False, force_reload=True, file_timeout=10)

