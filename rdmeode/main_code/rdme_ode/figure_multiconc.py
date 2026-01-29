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
import gc  
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
import re 
def get_user_input():
    # Modified to handle multiple concentration folders
    print("Enter the path to the base directory containing concentration folders:")
    base_dir = input().strip()
    if base_dir == "":
        base_dir = "/data2/2024_Yeast_GS/my_current_code/rdme_ode_results/20250410_diffconc"
    # Automatically find all folders in the base directory
    conc_folders = [d for d in os.listdir(base_dir) 
                    if os.path.isdir(os.path.join(base_dir, d)) and d.endswith('mM')]
    
    
    # Sort folders to ensure consistent ordering (assumes concentration in folder name)
    try:
        # Try to sort by extracting numeric values from folder names
        conc_folders = sorted(conc_folders, 
                             key=lambda x: float(''.join(filter(lambda c: c.isdigit() or c == '.', x))) 
                             if any(c.isdigit() for c in x) else x)
    except:
        # Fall back to regular sorting if numeric sorting fails
        conc_folders = sorted(conc_folders)
    
    print(f"Found concentration folders: {', '.join(conc_folders)}")
    
    print("Use these folders? (yes/no, default: yes):")
    user_input = input().strip().lower()
    use_found_folders = user_input != "no"  # Default to yes if input is empty or anything other than "no"
    
    if not use_found_folders:
        print("Enter comma-separated list of concentration folder names:")
        conc_folders = [folder.strip() for folder in input().strip().split(',')]
    
    # Extract concentration values from folder names for labels if possible
    labels = []
    for folder in conc_folders:
        # Try to extract numeric value from folder name for the label
        numeric_part = ''.join(filter(lambda c: c.isdigit() or c == '.', folder))
        if numeric_part:
            try:
                conc_value = float(numeric_part)
                labels.append(f"{conc_value} mM")
            except ValueError:
                labels.append(folder)
        else:
            labels.append(folder)
    
    print(f"Generated labels: {', '.join(labels)}")
    print("Use these labels? (yes/no, default: yes):")
    user_input = input().strip().lower()
    use_generated_labels = user_input != "no" 
    
    if not use_generated_labels:
        print("Enter labels for each concentration folder (comma-separated, same order as folders):")
        labels = [label.strip() for label in input().strip().split(',')]
    
    print("Enter base color for gradient (e.g., 'blue', 'red', 'green', or leave blank for default):")
    base_color = input().strip() or 'blue'
    
    print("Do you want to include region-specific data? (yes/no):")
    include_regions = input().strip().lower() == 'yes' or True
    
    # Generate colors based on concentration levels
    if conc_folders and any('0mM' in folder for folder in conc_folders):
        # Custom color handling for when 0mM is present
        colors = []
        for folder in conc_folders:
            # Use regex to match exactly "0mM" at the end of the string, not preceded by any digit
            if re.search(r'(?<!\d)0mM$', folder):
                colors.append('#E41A1C')  # Red for 0mM
            else:
                # We'll add other colors later in a gradient
                colors.append(None)
        # print(f"colors: {colors}, len(colors): {len(colors)}")
        # Get indices of non-0mM folders that need gradient colors
        gradient_indices = [i for i, color in enumerate(colors) if color is None]
        # print(f"gradient_indices: {gradient_indices}")
        if gradient_indices:
            # Use Nature journal color palette 
            # Nature journals often use a specific set of colors for scientific figures
            nature_colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#E69F00', '#F0E442']
            # Blue, Green, Orange-red, Pink, Light blue, Orange, Yellow
            
            
            # For many concentrations, create a smooth gradient from orange to green
            import matplotlib.colors as mcolors
            
            # Create a gradient using primary Nature colors: orange to green
            # Using E69F00 (orange) to 009E73 (green)
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "nature_gradient", [nature_colors[5], nature_colors[1]], N=len(gradient_indices))
            
            # Generate colors evenly spaced along this gradient
            gradient_colors = [mcolors.to_hex(cmap(i/(len(gradient_indices)-1))) 
                                for i in range(len(gradient_indices))]
            # Assign gradient colors to the None positions in the colors list
            for idx, gradient_idx in enumerate(gradient_indices):
                colors[gradient_idx] = gradient_colors[idx]
    
    region_suffix = "_region.jsonl"
    if include_regions:
        region_suffix = input(f"Enter region data file suffix (default: {region_suffix}): ") or region_suffix
    
    print("Enter species mapping (format: species1:species2,species3:species4 or leave blank):")
    species_mapping_str = input().strip()
    species_mapping = {}
    if species_mapping_str:
        pairs = species_mapping_str.split(',')
        for pair in pairs:
            if ':' in pair:
                src, dst = pair.split(':')
                species_mapping[src.strip()] = dst.strip()
    
    fig_dir = os.path.join(base_dir, "result_figures")
    print(f"Output figures will be saved to: {fig_dir}")
    
    return base_dir, conc_folders, labels, colors, include_regions, region_suffix, species_mapping, fig_dir

def check_cached_csvs(fig_dir, labels):
    """
    Check if all CSV statistics files already exist for the given labels.
    
    Parameters:
    -----------
    fig_dir : str
        Directory where CSV files should be located
    labels : list
        List of labels for concentration folders
        
    Returns:
    --------
    bool
        True if all CSV files exist, False otherwise
    """
    for label in labels:
        csv_path = os.path.join(fig_dir, f'{label}_species_statistics.csv')
        if not os.path.exists(csv_path):
            return False
    return True

def generate_color_gradient(base_color, num_colors, end_color=None):
    """
    Generate a gradient of colors based on a base color.
    
    Parameters:
    -----------
    base_color : str
        Base color name (e.g., 'blue', 'red', 'green') or hex code
    num_colors : int
        Number of colors to generate
    end_color : str, optional
        End color for gradient. If provided, creates a gradient from base_color to end_color
        
    Returns:
    --------
    list
        List of color codes
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # Dictionary of presets for common colors
    color_presets = {
        'blue': ('#1E88E5', '#8AB4F8'),  # Dark blue to light blue
        'red': ('#D32F2F', '#FFCDD2'),    # Dark red to light red
        'green': ('#388E3C', '#C8E6C9'),  # Dark green to light green
        'purple': ('#7B1FA2', '#E1BEE7'), # Dark purple to light purple
        'orange': ('#F57C00', '#FFE0B2'), # Dark orange to light orange
        'teal': ('#00897B', '#B2DFDB')    # Dark teal to light teal
    }
    
    # If end_color is provided, use it directly
    if end_color:
        start_color = base_color  # Could be name or hex
        # If base_color is a color name and in presets, use the dark version
        if isinstance(base_color, str) and base_color.lower() in color_presets:
            start_color = color_presets[base_color.lower()][0]
        # Same for end_color - ensure it's a string before using .lower()
        if isinstance(end_color, str) and end_color.lower() in color_presets:
            end_color = color_presets[end_color.lower()][0]
    else:
        # Get color range based on base_color
        if isinstance(base_color, str) and base_color.lower() in color_presets:
            start_color, end_color = color_presets[base_color.lower()]
        else:
            # If base_color not in presets, use it directly with a lighter version
            start_color = base_color
            # Create a lighter version for end color if base_color is a hex code
            if isinstance(base_color, str) and base_color.startswith('#'):
                # Convert to RGB, lighten, then back to hex
                rgb = mcolors.hex2color(base_color)
                # Make it lighter (blend with white)
                lighter_rgb = tuple(0.3 + 0.7 * c for c in rgb)
                end_color = mcolors.rgb2hex(lighter_rgb)
            else:
                # Default to blue gradient if we can't determine a good end color
                start_color, end_color = color_presets['blue']
        
    # Create a colormap
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", [start_color, end_color])
    
    # Generate evenly spaced colors
    if num_colors == 1:
        return [mcolors.to_hex(cmap(0.5))]
    else:
        return [mcolors.to_hex(cmap(i/(num_colors-1))) for i in range(num_colors)]

# Main execution code after the function definitions
# Get user input
base_dir, conc_folders, labels, colors, include_regions, region_suffix, species_mapping, fig_dir = get_user_input()

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# Check if cached CSV files exist
use_cached = check_cached_csvs(fig_dir, labels)
if use_cached:
    logging.info("All CSV statistics files found. Using cached data.")
else:
    logging.info("Some CSV statistics files missing. Processing trajectory data.")

traj_suff = "_ode.jsonl"
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

logging.info(f"Comparing trajectories between {len(conc_folders)} concentration folders:")
for i, folder in enumerate(conc_folders):
    logging.info(f"Directory {i+1} ({labels[i]}): {os.path.join(base_dir, folder)}")

logging.info(f"Include region-specific data: {include_regions}")
if include_regions:
    logging.info(f"Region data file suffix: {region_suffix}")
if species_mapping:
    logging.info("Species mapping:")
    for sp1, sp_map in species_mapping.items():
            logging.info(f"  {sp1} -> {sp_map}")

# Initialize NAV value - needed for GAI calculations
NAV = None

# Only process trajectory data if cached CSV files don't exist
if not use_cached:
    # Initialize data storage structures
    all_traj_files = []  # List to store files for each folder
    all_data_species = []  # List to store species data for each folder
    all_data_species_region = []  # List to store region data for each folder
    all_data_ode = []  # List to store ODE data for each folder

    rdmeTs = None
    odeTs = None
    regionTs = None
    NAV = None  # Initialize avogadro constant * volume

    # Process each concentration folder
    for folder_idx, folder in enumerate(conc_folders):
        traj_dir = os.path.join(base_dir, folder)
        logging.info(f"Processing directory: {traj_dir}")
        
        # Get trajectory files
        files = [f for f in os.listdir(traj_dir) if f.startswith('yeast') and f.endswith('.lm')]
        all_traj_files.append(files)
        
        logging.info(f"{labels[folder_idx]} files: {files}")
        
        # Initialize data dictionaries for this folder
        data_species = {}
        data_species_region = {}
        data_ode = {}
        
        # Process files for this concentration folder
        for traj_file in tqdm(files, desc=f"Processing {labels[folder_idx]} files", unit="file"):
            logging.info(f"Processing {labels[folder_idx]} file: {traj_file}")
            region_traj = None
            if include_regions:
                traj, odeTraj, region_traj = get_traj(traj_dir, traj_file, traj_suff, region_suff=region_suffix)
            else:
                traj, odeTraj, _ = get_traj(traj_dir, traj_file, traj_suff)
    
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
                if species not in data_species:
                    data_species[species] = []
                data_species[species].append(data)

            for species, data in odeYs.items():
                if species not in data_ode:
                    data_ode[species] = []
                data_ode[species].append(data)
    
            # Process region-specific data if available
            if regionYs is not None and region_traj is not None:
                regions = region_traj['regions']
                print(f"regions: {regions} in region traj")
                
                # Process each species
                for species, region_data in regionYs.items():
                    # Initialize the nested dictionary structure if needed
                    if species not in data_species_region:
                        data_species_region[species] = {}
                    
                    # Initialize lists for each region if they don't exist
                    for region in regions:
                        if region not in data_species_region[species]:
                            data_species_region[species][region] = []
                    
                    # Now append the data for this species
                    for i in range(len(regions)):
                        data_species_region[species][regions[i]].append(regionYs[species][i])

        # Store the data dictionaries for this folder
        all_data_species.append(data_species)
        all_data_species_region.append(data_species_region)
        all_data_ode.append(data_ode)

    # Calculate and save statistics for each concentration folder
    all_results = []  # List to store results dataframes
    all_results_region = []  # List to store region results dataframes

    for folder_idx, (data_species, data_species_region, data_ode) in enumerate(zip(all_data_species, all_data_species_region, all_data_ode)):
        results = []
        results_region = []
        
        # Process overall species data
        for species, trajectories in data_species.items():
            trajectories_array = np.array(trajectories)
            avg = np.mean(trajectories_array, axis=0)
            std = np.std(trajectories_array, axis=0)
            
            results.append({
            'Species': f"RDME_{species}",
            'Time': ','.join(map(str, rdmeTs)),
            'Average': ','.join(map(str, avg)),
            'Std': ','.join(map(str, std))
            })
        print(f"data_species_region: {data_species_region}")
    
        # Process region-specific data
        for species, regions in data_species_region.items():
            for region, trajectories in regions.items():
                trajectories_array = np.array(trajectories)
                avg = np.mean(trajectories_array, axis=0)
                std = np.std(trajectories_array, axis=0)
                
                # Store region-specific results
                results_region.append({
                'Species': species,
                'Region': region,
                'Time': ','.join(map(str, regionTs if regionTs is not None else rdmeTs)),
                'Average': ','.join(map(str, avg)),
                'Std': ','.join(map(str, std))
                })
                
                # Also store in main results with a special naming convention
                results.append({
                'Species': f"RDME_{species}_{region}",
                'Time': ','.join(map(str, regionTs if regionTs is not None else rdmeTs)),
                'Average': ','.join(map(str, avg)),
                'Std': ','.join(map(str, std))
                })

        # Process ODE species data
        for species, trajectories in data_ode.items():
            trajectories_array = np.array(trajectories)
            avg = np.mean(trajectories_array, axis=0)
            std = np.std(trajectories_array, axis=0)
        
            results.append({
            'Species': f"ODE_{species}",
            'Time': ','.join(map(str, odeTs)),
            'Average': ','.join(map(str, avg)),
            'Std': ','.join(map(str, std))
            })

        # Convert to DataFrame and save
        results_df = pd.DataFrame(results)
        results_region_df = pd.DataFrame(results_region)
        
        csv_path = os.path.join(fig_dir, f'{labels[folder_idx]}_species_statistics.csv')
        results_df.to_csv(csv_path, index=False)
        logging.info(f"{labels[folder_idx]} statistics saved to: {csv_path}")
        
        if results_region:
            region_csv_path = os.path.join(fig_dir, f'{labels[folder_idx]}_region_statistics.csv')
            results_region_df.to_csv(region_csv_path, index=False)
            logging.info(f"{labels[folder_idx]} region statistics saved to: {region_csv_path}")
        
        all_results.append(results_df)
        all_results_region.append(results_region_df)

# If using cached data, we need to calculate NAV from a trajectory file
if use_cached and NAV is None:
    logging.info("Using cached data - calculating NAV from first available trajectory file")
    try:
        # Get the first trajectory file from any folder to calculate NAV
        for folder in conc_folders:
            traj_dir = os.path.join(base_dir, folder)
            files = [f for f in os.listdir(traj_dir) if f.startswith('yeast') and f.endswith('.lm')]
            if files:
                traj, _, _ = get_traj(traj_dir, files[0], traj_suff)
                NAV = 6.022e23 * (traj.reg.cytoplasm.volume + traj.reg.nucleoplasm.volume + traj.reg.plasmaMembrane.volume)
                logging.info(f"NAV value calculated from cached data: {NAV}")
                break
    except Exception as e:
        logging.warning(f"Could not calculate NAV from trajectory files: {e}")
        # Use a default NAV value (typical for yeast cell simulations)
        NAV = 6.022e23 * 1e-12  # Assuming ~1 picolitre total volume
        logging.info(f"Using default NAV value: {NAV}")

'''
Here we start to load the csvs, and create the plots 
'''
# Read the saved statistics
all_data_dfs = []
for i, label in enumerate(labels):
    df = pd.read_csv(os.path.join(fig_dir, f'{label}_species_statistics.csv'))
    all_data_dfs.append(df)
    logging.info(f"Available species in {label}: {df['Species'].tolist()}")

# Function to convert string of comma-separated values to numpy array
def str_to_array(s):
    return np.array([float(x) for x in s.split(',')])

# Identify species common to all concentration folders or mapped species
def get_comparable_species_groups(all_dfs, species_mapping=None):
    comparable_species = []
    
    # If no mapping is provided, find species common to all dfs
    if not species_mapping:
        # Get all unique species across all dfs
        all_species = set()
        for df in all_dfs:
            all_species.update(df['Species'].unique())
        
        # Check which species exist in all dfs
        for species in all_species:
            if all(species in df['Species'].values for df in all_dfs):
                comparable_species.append([species] * len(all_dfs))
    else:
        # With mapping, create species groups as specified
        # For simplicity, we'll only implement direct mappings here
        # This would need to be expanded for more complex mappings
        base_df = all_dfs[0]
        for species in base_df['Species'].unique():
            if species in species_mapping:
                mapped_species = species_mapping[species]
                # Check if the mapped species exists in the second df
                if mapped_species in all_dfs[1]['Species'].values:
                    comparable_species.append([species, mapped_species])
            # If no mapping, check if the same species exists in other dfs
            elif all(species in df['Species'].values for df in all_dfs[1:]):
                comparable_species.append([species] * len(all_dfs))
    
    return comparable_species

# Get comparable species groups
comparable_species_groups = get_comparable_species_groups(all_data_dfs, species_mapping)
logging.info(f"Comparable species groups: {comparable_species_groups}")

# Plot settings - use publication style
pub_colors = setup_publication_style(figure_size='medium', dpi=300)

# After your plot settings section, before the batched plotting begins, add a function to create a separate legend figure

def create_legend_figure(labels, colors, fig_dir):
    """
    Create a separate figure with just the legend in two rows
    
    Parameters:
    -----------
    labels : list
        List of labels for each concentration
    colors : list
        List of colors used for each concentration
    fig_dir : str
        Directory to save the legend figure
    """
    plt.figure(figsize=(10, 3))
    ax = plt.gca()
    
    # Calculate how many items per row
    items_per_row = int(np.ceil(len(labels) / 2))
    
    # Create proxy artists for the legend (invisible line objects)
    # First row
    first_row = []
    for i in range(min(items_per_row, len(labels))):
        # line = plt.Line2D([0], [0], color=colors[i], lw=1, linestyle=['-', '--', '-.', ':'][i % 4])
        line = plt.Line2D([0], [0], color=colors[i], lw=1)
        first_row.append((line, labels[i]))
    
    # Second row if needed
    second_row = []
    for i in range(items_per_row, len(labels)):
        # line = plt.Line2D([0], [0], color=colors[i], lw=1, linestyle=['-', '--', '-.', ':'][i % 4])
        line = plt.Line2D([0], [0], color=colors[i], lw=1)
        second_row.append((line, labels[i]))
    
    # Create the first legend for the first row
    if first_row:
        first_legend = ax.legend(*zip(*first_row), loc='upper center', 
                               bbox_to_anchor=(0.5, 0.8), ncol=items_per_row, framealpha=0.3)
        plt.gca().add_artist(first_legend)
    
    # Create the second legend for the second row
    if second_row:
        second_legend = ax.legend(*zip(*second_row), loc='upper center', 
                                bbox_to_anchor=(0.5, 0.2), ncol=len(second_row), framealpha=0.3)
    
    # Hide axis
    ax.set_axis_off()
    
    # Save the legend figure
    plt.tight_layout()
    legend_path = os.path.join(fig_dir, 'legend.png')
    plt.savefig(legend_path, dpi=600, bbox_inches='tight', transparent=True)
    plt.close()
    
    logging.info(f"Saved separate legend figure: {legend_path}")
    
    return legend_path

# Add this right after your plot settings section but before the batched plotting begins
# Create the separate legend figure
legend_path = create_legend_figure(labels, colors, fig_dir)

# Process plots in batches of 10
batch_size = 10
for i in range(0, len(comparable_species_groups), batch_size):
    batch = comparable_species_groups[i:i+batch_size]
    for sp_group in batch:
        fig, ax = plt.subplots()
        
        # Extract display name from the first species
        display_name = sp_group[0]
   
        
        # Generate plot title and filename
        plot_title = f'{display_name} Comparison'
        output_filename = f'{display_name}_comparison.png'
        
        # Plot data for each concentration folder
        for i, (species, df, label, color) in enumerate(zip(sp_group, all_data_dfs, labels, colors)):
            species_data = df[df['Species'] == species]
            
            if len(species_data) == 0:
                logging.info(f"Skipping {species} for {label} - data not found")
                continue
            
            data_row = species_data.iloc[0]
            time = str_to_array(data_row['Time'])
            avg = str_to_array(data_row['Average'])
            std = str_to_array(data_row['Std'])
            
        
            # Different line styles for different folders
            linestyle = ['-', '--', '-.', ':'][i % 4]
            
            # ax.plot(time, avg, label=label, linestyle=linestyle, color=color)
            ax.plot(time, avg, label=label, color=color, linewidth=1)
            # #ax.fill_between(time, avg - std, avg + std, alpha=0.2, color=color)
        
        # Customize plot
        ax.set_xlabel('Time (min)')
        if display_name.startswith('DG'):
            ax.set_ylabel('Probability')
        else:
            ax.set_ylabel('Counts')
        # ax.set_title(plot_title)  # Uncomment if you want titles
        # ax.legend(framealpha=0.3, loc='best')
        ax.grid(False)
        
        # Save figure
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, output_filename)
        plt.savefig(fig_path, dpi=600, bbox_inches='tight')
        logging.info(f"Saved plot: {output_filename}")
        plt.clf()  # Clear the current figure
    plt.close('all')  # Force memory cleanup between batches

# Create region-specific plots if requested
if include_regions:
    # Create a directory for region plots
    region_plot_dir = os.path.join(fig_dir, 'region_plots')
    os.makedirs(region_plot_dir, exist_ok=True)
    
    # Load region-specific data
    all_region_dfs = []
    for i, label in enumerate(labels):
        region_csv_path = os.path.join(fig_dir, f'{label}_region_statistics.csv')
        if os.path.exists(region_csv_path):
            df = pd.read_csv(region_csv_path)
            all_region_dfs.append(df)
        else:
            all_region_dfs.append(pd.DataFrame())
    
    # Find species and regions common to all concentration folders
    all_species_region = set()
    all_regions = set()
    
    for df in all_region_dfs:
        if not df.empty:
            all_species_region.update(df['Species'].unique())
            all_regions.update(df['Region'].unique())
    
    # Free up memory from data we no longer need (only if they exist)
    if not use_cached:
        del all_data_species
        del all_data_species_region
        del all_data_ode
    
    for species in all_species_region:
        # Check which regions have data for this species across all folders
        species_regions = set()
        for df in all_region_dfs:
            if not df.empty:
                species_df = df[df['Species'] == species]
                if not species_df.empty:
                    species_regions.update(species_df['Region'].unique())
        
        for region in species_regions:
            fig, ax = plt.subplots()
            
            # Plot data for each concentration folder
            for i, (df, label, color) in enumerate(zip(all_region_dfs, labels, colors)):
                if df.empty:
                    continue
                
                region_data = df[(df['Species'] == species) & (df['Region'] == region)]
                
                if len(region_data) == 0:
                    logging.info(f"Skipping {species} in {region} for {label} - data not found")
                    continue
                    
                data_row = region_data.iloc[0]
                time = str_to_array(data_row['Time'])
                avg = str_to_array(data_row['Average'])
                std = str_to_array(data_row['Std'])
                
                # Skip if all near zero
                # if np.all(avg < 10e-6):
                #     logging.info(f"All negligible values for {species} in {region} for {label}, skipping")
                #     continue
                
                # Different line styles for different folders
                linestyle = ['-', '--', '-.', ':'][i % 4]
                
                # ax.plot(time, avg, label=label, linestyle=linestyle, color=color)
                ax.plot(time, avg, label=label, color=color, linewidth=1)
                # #ax.fill_between(time, avg - std, avg + std, alpha=0.2, color=color)
            
            # Customize plot
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Counts')
            # ax.set_title(f'{species} in {region}')
            # ax.legend(framealpha=0.3, loc='best')
            ax.grid(False)
            
            # Save figure
            plt.tight_layout()
            fig_path = os.path.join(region_plot_dir, f'{species}_{region}_comparison.png')
            plt.savefig(fig_path, dpi=600, bbox_inches='tight')
            logging.info(f"Saved region plot: {species}_{region}_comparison.png")
            plt.clf()  # Clear the current figure
            plt.close('all')  # Close all figures
            
        # Free memory after processing each species
        gc.collect()  # Force garbage collection
    
    # Free up memory from region data when done
    del all_region_dfs
    del all_species_region
    del all_regions
    gc.collect()

# Special case: G2 membrane totals
create_g2_total = True

if create_g2_total:
    fig, ax = plt.subplots()
    
    for i, (df, label, color) in enumerate(zip(all_data_dfs, labels, colors)):
        # Calculate G2 totals
        g2_data = df[df['Species'].isin(['ODE_G2', 'ODE_G2GAE', 'ODE_G2GAI'])].copy()
        
        if len(g2_data) > 0:
            time = str_to_array(g2_data.iloc[0]['Time'])
            total = np.zeros_like(str_to_array(g2_data.iloc[0]['Average']))
            std_squared = np.zeros_like(total)
            
            for _, row in g2_data.iterrows():
                total += str_to_array(row['Average'])
                std_squared += str_to_array(row['Std'])**2
            
            total_std = np.sqrt(std_squared)
            
            # Different line styles for different folders
            linestyle = ['-', '--', '-.', ':'][i % 4]
            
            # ax.plot(time, total, label=label, linestyle=linestyle, color=color)
            ax.plot(time, total, label=label, color=color, linewidth=1)
            #ax.fill_between(time, total - total_std, total + total_std, alpha=0.2, color=color)
    
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Counts')
    # ax.set_title('Total G2 Comparison (G2 + G2GAE + G2GAI)')
    # ax.legend(framealpha=0.3, loc='upper right')
    ax.grid(False)

    # Save figure
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, 'G2_membrane_comparison.png')
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    logging.info(f"Saved plot for G2 total")
    plt.clf()  # Clear the current figure
    plt.close('all')  # Close all figures
    
# Special case: GAI total
create_gai_total = True

if create_gai_total:
    # Ask if user wants to specify custom GAE value for horizontal line
    add_gae_line = input("Do you want to add a horizontal line for GAE reference value? (yes/no): ").lower() == 'yes'
    gae_value = None
    
    if add_gae_line:
        try:
            gae_value = float(input("Enter GAE value in mM (e.g., 11.1): "))
        except ValueError:
            logging.info("Invalid GAE value, no reference line will be added")
            add_gae_line = False
    
    fig, ax = plt.subplots()

    # List of species to combine
    gai_species = ['GAI', 'G1GAI', 'G3i', 'G2GAI']
    
    # Allow user to customize the GAI species list
    customize_gai_species = True
    if customize_gai_species:
        print(f"Current GAI species list: {gai_species}")
        new_species_list = input("Enter comma-separated list of species to combine: ")
        if new_species_list:
            gai_species = [s.strip() for s in new_species_list.split(',')]
    
    for i, (df, label, color) in enumerate(zip(all_data_dfs, labels, colors)):
        # Initialize combined data for this folder
        combined_avg = None
        combined_var = None
        time = None
        species_used = []
        
        # Combine data for this folder
        for species_name in gai_species:
            # Look for both ODE and RDME versions of the species
            matching_rows = df[df['Species'].str.contains(species_name)]
            
            if not matching_rows.empty:
                # Prefer ODE data if available
                species_data = matching_rows[matching_rows['Species'].str.startswith('ODE')]
                if species_data.empty:
                    species_data = matching_rows
                    
                if len(species_data) > 0:
                    row = species_data.iloc[0]
                    # Track which species are being used
                    species_used.append(row['Species'])
                
                    curr_avg = str_to_array(row['Average']) / NAV * 1e3
                    curr_std = str_to_array(row['Std']) / NAV * 1e3
                    curr_var = curr_std ** 2  # Convert std to variance
                
                    if combined_avg is None:
                        time = str_to_array(row['Time'])
                        combined_avg = curr_avg
                        combined_var = curr_var
                    else:
                        combined_avg += curr_avg
                        combined_var += curr_var  # Variances add for independent variables
        
        # Log which species were used
        logging.info(f"{label} species used in GAI total: {species_used}")
        
        # Plot data if available
        if combined_avg is not None and time is not None:
            combined_std = np.sqrt(combined_var)
            
            # Different line styles for different folders
            linestyle = ['-', '--', '-.', ':'][i % 4]
            
            # ax.plot(time, combined_avg, label=label, linestyle=linestyle, color=color)
            ax.plot(time, combined_avg, label=label, color=color, linewidth=1)
            #ax.fill_between(time, combined_avg - combined_std, combined_avg + combined_std, alpha=0.2, color=color)
                        
    # Add horizontal line for GAE reference value if requested
    if add_gae_line and gae_value is not None:
        ax.axhline(y=gae_value, color='gray', linestyle='--', linewidth=1, label='GAE')
        ax.text(time[0]*1.05, gae_value*0.97, f'{gae_value} mM', color='gray', va='top', ha='left')

    # Customize plot
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Concentration (mM)')
    # ax.set_title('Total GAI Species Comparison')
    # ax.legend(framealpha=0.3, loc='upper right')
    ax.grid(False)

    # Save figure
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, 'GAI_total_comparison.png')
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    logging.info(f"Saved plot for GAI total")
    plt.clf()  # Clear the current figure
    plt.close('all')  # Close all figures

# Special case: Normalized GAI total
create_normalized_gai_total = True

if create_normalized_gai_total:
    fig, ax = plt.subplots()

    # List of species to combine
    gai_species = ['GAI', 'G1GAI', 'G3i', 'G2GAI']
    
    # Allow user to customize the GAI species list
    customize_gai_species = input("Do you want to customize the GAI species list for normalized plot? (yes/no): ").lower() == 'yes'
    if customize_gai_species:
        print(f"Current GAI species list: {gai_species}")
        new_species_list = input("Enter comma-separated list of species to combine: ")
        if new_species_list:
            gai_species = [s.strip() for s in new_species_list.split(',')]
    
    for i, (df, label, color, folder) in enumerate(zip(all_data_dfs, labels, colors, conc_folders)):
        # Extract GAE value from folder name
        gae_value = None
        try:
            # Extract numeric value from folder name
            numeric_part = ''.join(filter(lambda c: c.isdigit() or c == '.', folder))
            if numeric_part:
                gae_value = float(numeric_part)
                logging.info(f"Extracted GAE value for {folder}: {gae_value} mM")
            else:
                logging.warning(f"Could not extract GAE value from folder name: {folder}")
                continue
        except ValueError:
            logging.warning(f"Error extracting GAE value from folder name: {folder}")
            continue
            
        # Skip 0mM folder for normalization (can't divide by 0)
        if gae_value == 0:
            logging.info(f"Skipping {folder} for normalization (GAE value is 0)")
            continue
            
        # Initialize combined data for this folder
        combined_avg = None
        combined_var = None
        time = None
        species_used = []
        
        # Combine data for this folder
        for species_name in gai_species:
            # Look for both ODE and RDME versions of the species
            matching_rows = df[df['Species'].str.contains(species_name)]
            
            if not matching_rows.empty:
                # Prefer ODE data if available
                species_data = matching_rows[matching_rows['Species'].str.startswith('ODE')]
                if species_data.empty:
                    species_data = matching_rows
                    
                if len(species_data) > 0:
                    row = species_data.iloc[0]
                    # Track which species are being used
                    species_used.append(row['Species'])
                
                    curr_avg = str_to_array(row['Average']) / NAV * 1e3
                    curr_std = str_to_array(row['Std']) / NAV * 1e3
                    curr_var = curr_std ** 2  # Convert std to variance
                
                    if combined_avg is None:
                        time = str_to_array(row['Time'])
                        combined_avg = curr_avg
                        combined_var = curr_var
                    else:
                        combined_avg += curr_avg
                        combined_var += curr_var  # Variances add for independent variables
        
        # Log which species were used
        logging.info(f"{label} species used in normalized GAI total: {species_used}")
        
        # Plot normalized data if available
        if combined_avg is not None and time is not None and gae_value is not None:
            # Normalize by GAE value
            normalized_avg = combined_avg / gae_value
            
            # Properly scale the variance for the normalized values
            # For y = x/c where c is constant, Var(y) = Var(x)/c²
            normalized_var = combined_var / (gae_value ** 2)
            normalized_std = np.sqrt(normalized_var)
            
            # Different line styles for different folders
            linestyle = ['-', '--', '-.', ':'][i % 4]
            
            # ax.plot(time, normalized_avg, label=label, linestyle=linestyle, color=color)
            ax.plot(time, normalized_avg, label=label, color=color, linewidth=1)
            #ax.fill_between(time, normalized_avg - normalized_std, normalized_avg + normalized_std, alpha=0.2, color=color)
    
    # Add a horizontal line at y=1 to show the GAE reference
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='GAE reference')
    ax.text(time[0]*1.05, 1*0.97, 'External GAE', color='gray', va='top', ha='left')

    # Customize plot
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('GAI/GAE ratio')
    # ax.set_title('Normalized GAI Total (relative to external GAE)')
    # ax.legend(framealpha=0.3, loc='upper right')
    ax.grid(False)

    # Save figure
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, 'GAI_normalized_comparison.png')
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    logging.info(f"Saved plot for normalized GAI total")
    plt.clf()  # Clear the current figure
    plt.close('all')  # Close all figures
    
logging.info(f"\nAll plots saved in: {fig_dir}")
logging.getLogger().handlers[0].flush()