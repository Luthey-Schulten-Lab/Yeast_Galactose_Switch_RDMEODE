import pickle
import numpy as np
from jLM.RDME import File as RDMEFile
import jLM
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

# This file contains functions for analyzing and plotting trajectory data from RDME simulations

# Imports necessary libraries for data handling, plotting, and RDME file operations

def get_traj(traj_dir, traj_file, traj_suff=None, region_suff=None):
    # Reads trajectory data from either JSONL or PKL files
    # Returns both RDME trajectory object and ODE trajectory data
    traj_fname = os.path.join(traj_dir, traj_file)
    traj = RDMEFile(traj_fname)
    # Read the file based on its type
    if traj_suff is None:
        odeTraj = None
    elif traj_suff.endswith('.jsonl'):
        odeTraj = {'ts': [], 'ys': [], 'names': None}
        with open(traj_fname + traj_suff, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                odeTraj['ts'].append(data['time'])
                if odeTraj['names'] is None:
                    odeTraj['names'] = list(data['species'].keys())
                odeTraj['ys'].append([data['species'][name] for name in odeTraj['names']])
        
        # Convert lists to numpy arrays
        odeTraj['ts'] = np.array(odeTraj['ts'])
        odeTraj['ys'] = np.array(odeTraj['ys'])
    elif traj_suff.endswith('.pkl'):  # PKL file
        with open(traj_fname + traj_suff, 'rb') as f:
            odeTraj = pickle.load(f)
            # print(odeTraj)
    else:
        raise ValueError(f"Unsupported file type: {traj_suff}")

    # now load the region data
    
    if region_suff is None:
        region_traj = None
    elif region_suff.endswith('.jsonl'):
        region_traj = {'ts': [], 'region_trajs': {}, 'names': None, 'regions': None}
        with open(traj_fname + region_suff, 'r') as f:
            # structures initialization
            data = json.loads(f.readline().strip())
            region_traj['names'] = list(data.keys() - {'time'})
            region_traj['regions'] = list(data[region_traj['names'][0]].keys())
            # Initialize nested structure: species -> list of empty lists for each region
            region_traj['region_trajs'] = {
                species: [[] for _ in region_traj['regions']] 
                for species in region_traj['names']
            }
            # print("empty region_trajs:")
            # print(region_traj['region_trajs'])
            # reset the file pointer to the beginning
            f.seek(0)
            
            for line in f:
                data = json.loads(line.strip())
                region_traj['ts'].append(data['time'])
                for i, region in enumerate(region_traj['regions']):
                    for j, species in enumerate(region_traj['names']):
                        region_traj['region_trajs'][species][i].append(data[species][region])
        
    else:  
        raise ValueError(f"Unsupported file type: {region_suff}")
    return traj, odeTraj, region_traj

def get_NAV(traj_dir, traj_file):
    traj = RDMEFile(traj_dir+traj_file)
    return 6.022e23 * (traj.reg.cytoplasm.volume + traj.reg.nucleoplasm.volume + traj.reg.plasmaMembrane.volume)
def get_data_for_plot(traj, odeTraj, region_traj=None, sparse_factor = 1):
    # Processes trajectory data for plotting
    # Downscales data, converts time to minutes, and calculates molecule numbers
    # Returns time and trajectory data for both RDME and ODE simulations

    # Downscale the time and trajectory by a factor of sparse_factor, default is 1
    # output unit is counts(number of molecules)
    odeTs = odeTraj['ts'][::sparse_factor] / 60  # Convert to minutes and downscale
    NAV = 6.022e23 * (traj.reg.cytoplasm.volume + traj.reg.nucleoplasm.volume + traj.reg.plasmaMembrane.volume)
    
    odeYs = {n: odeTraj['ys'][::sparse_factor, i] * NAV for i, n in enumerate(odeTraj['names'])}  # Downscale
    # rdme data
    rdmeYs = dict()

    for sp in traj.speciesList:
        ts, ys = sp.getNumberTrajectory()
        rdmeYs[sp.name] = ys[::sparse_factor]  # Downscale RDME trajectories by a factor of 10
        rdmeTs = ts[::sparse_factor] / 60  # Convert to minutes and downscale
    # region data from rdme jsonl file
    if region_traj is not None:
        regionTs = np.array(region_traj['ts'][::sparse_factor]) / 60
        # iterate over each region to perform downscale
        regionYs = dict()
        for species in region_traj['names']:
            regionYs[species] = [
                np.array(region_data)[::sparse_factor] 
                for region_data in region_traj['region_trajs'][species]
            ]
    else:
        regionTs = None
        regionYs = None
    return rdmeTs, rdmeYs, odeTs, odeYs, regionTs, regionYs


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def plot_gene_states_duration(traj_file, fig_dir, rdmeTs, rdmeYs,dpi=300):
    # Creates a stacked area plot showing the duration of different gene states over time
    # Plots data for multiple genes and their states (Free, Activated, Repressed)
    # Saves the resulting plot as a PNG file

    # Define the genes and their states
    genes = [['DG1', 'DGrep'], ['DG2'], ['DG3', 'DG80']]
    states = ['', '_G4d', '_G4d_G80d']

    # Create a figure with subplots for each gene group
    fig, axs = plt.subplots(3, 2, figsize=(20, 10), sharex=True)
    fig.suptitle('Duration of Genes in Different States', fontsize=16, y=0.95)

    # Colors for different states
    colors = ['#B0BEC5', '#4CAF50', '#1E88E5']

    # Create a list to store legend handles
    legend_handles = []

    for row, gene_row in enumerate(genes):
        for col, gene in enumerate(gene_row):
            ax = axs[row, col]
            bottom = np.zeros_like(rdmeTs)
            
            for j, state in enumerate(states):
                species_name = f'{gene}{state}'
                if species_name in rdmeYs:
                    values = rdmeYs[species_name]
                    handle = ax.fill_between(rdmeTs, bottom, bottom + values, alpha=0.7, color=colors[j])
                    bottom += values
                    
                    # Only add to legend_handles for the first gene (to avoid duplicates)
                    if row == 0 and col == 0 and j == 0:
                        legend_handles = [handle]
                    elif row == 0 and col == 0:
                        legend_handles.append(handle)
            
            ax.set_ylabel('Count')
            ax.set_title(f'{gene} States')
            ax.grid(True, linestyle='--', alpha=0.3)

        # Hide unused subplot
        if len(gene_row) == 1:
            axs[row, 1].set_visible(False)

    # Set x-label for bottom subplots
    for ax in axs[-1, :]:
        ax.set_xlabel('Time (min)')

    # Add a single legend for all subplots
    fig.legend(legend_handles, ['Free', 'Activated', 'Repressed'], 
            loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)

    # Add the file_name as a string in the bottom right of the figure
    plt.gcf().text(0.95, 0.00, traj_file, fontsize=10, verticalalignment='bottom', horizontalalignment='right')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.92)  # Adjust these values to make room for the title and legend

    # Save the figure
    fig.savefig(os.path.join(fig_dir,  os.path.basename(traj_file) + '_gene_states_duration.png' ), dpi=dpi, bbox_inches='tight')


def plot_gene_states_duration_table(traj_file, fig_dir, rdmeTs, rdmeYs,dpi=300):
    genes = ['DG1', 'DGrep', 'DG2', 'DG3', 'DG80']
    states = ['', '_G4d', '_G4d_G80d']
    total_time = rdmeTs[-1] - rdmeTs[0]

    avg_times = {}
    for gene in genes:
        avg_times[gene] = {}
        for state in states:
            species_name = f'{gene}{state}'
            if species_name in rdmeYs:
                avg_time = np.trapz(rdmeYs[species_name], rdmeTs) / total_time
                avg_times[gene][state] = avg_time
            else:
                avg_times[gene][state] = 0

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')

    cell_text = [[f"{avg_times[gene][state]:.2f}" for state in states] for gene in genes]
    table = ax.table(cellText=cell_text, rowLabels=genes, colLabels=['Free', 'Activated', 'Repressed'], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title("Average Time Spent in Each State (%)", fontsize=16, pad=0)
    plt.gcf().text(0.95, 0.01, traj_file, fontsize=10, verticalalignment='bottom', horizontalalignment='right')

    plt.tight_layout()
    fig.savefig(os.path.join( fig_dir,  os.path.basename(traj_file)  + '_gene_states_duration_table.png'), dpi=dpi, bbox_inches='tight')
    #plt.show()


# Assuming rdmeYs, odeYs, rdmeTs, odeTs, and traj_fname are already defined
def plot_all_species(traj_file, traj_dir, fig_dir, rdmeTs, rdmeYs, odeTs, odeYs,dpi=300):
    traj_fname = traj_dir + traj_file
    # Prepare names and plot layout
    names = sorted(filter(lambda x: x[0], set(rdmeYs.keys()) | set(odeYs.keys())))
    nplots = len(names)
    ncols = 4
    nrows = int(np.ceil(nplots / ncols))

    # Arguments for the axes
    # axargs = dict(xlim=(min(odeTs.min(), rdmeTs.min()), max(odeTs.max(), rdmeTs.max())),
    #               xlabel='t/min', ylabel='Count', xticks=np.arange(0, 61, 10))
    x_min = min(odeTs.min(), rdmeTs.min())
    x_max = max(odeTs.max(), rdmeTs.max())
    axargs = dict(xlim=(x_min, x_max),
                xlabel='t/min', ylabel='Count',
                xticks=np.linspace(x_min, x_max, 7, endpoint=True))
    # Create figure and subplots
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15, 3 * nrows))

    # Adjust layout to make space for the suptitle
    fig.subplots_adjust(top=0.90)

    # Add a title to the figure without overlapping with subplots
    fig.suptitle(str(traj_fname), y=0.95)

    axs = axs.ravel()
    colors = sns.color_palette()

    # Plot data
    for ax, name in zip(axs, names):
        if name in rdmeYs:
            if name not in ['G1', 'G2']:
                rdmeHandle = ax.plot(rdmeTs, rdmeYs[name], c=colors[0], label="RDME")[0]
        if name in odeYs:
            odeHandle = ax.plot(odeTs, odeYs[name], c=colors[1], label="ODE")[0]
        ax.set(title=name, **axargs)

    # Remove empty subplots
    for i, ax in enumerate(axs):
        if i >= nplots:
            ax.remove()

    # Tight layout to adjust subplots spacing
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Add legend to the last subplot
    axs[nplots - 1].legend([rdmeHandle, odeHandle], ["RDME", "ODE"], bbox_to_anchor=(1.5, 1), loc=2, borderaxespad=0.)

    # Save the figure
    fig.savefig(os.path.join (fig_dir, traj_file + '_ll_species_counts' + '.png'), dpi=dpi)


# tailored species plot
def plot_tailored_species(traj, traj_file, traj_dir, fig_dir, rdmeTs, rdmeYs, odeTs, odeYs,dpi=300):
    # Define the species to plot in the specified order
    species_order = [
        ['G1', 'G1GAI', 'G1+G1GAI', 'Grep'],
        ['G2', 'G2GAE', 'G2GAI', 'G2GAE+G2GAI+G2'],
        ['G3', 'G3i'],
        ['G4', 'G4d', 'G80', 'G80d', 'G80d_G3i'],
        ['GAI', 'GAI+G3i'],
        ['ribosome_occupied']
    ]
    traj_fname = traj_dir + traj_file
    NAV = 6.022e23 * (traj.reg.cytoplasm.volume + traj.reg.nucleoplasm.volume + traj.reg.plasmaMembrane.volume)
    print("NAV: ", NAV)
    # Calculate the number of rows and columns
    nrows = len(species_order)
    ncols = max(len(row) for row in species_order)

    # Create figure and subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))

    # Adjust layout to make space for the suptitle
    fig.subplots_adjust(top=0.95)

    # Add a title to the figure without overlapping with subplots
    fig.suptitle(str(traj_fname), y=0.98)

    # Arguments for the axes
    x_min = min(odeTs.min(), rdmeTs.min())
    x_max = max(odeTs.max(), rdmeTs.max())
    axargs = dict(xlim=(x_min, x_max),
                xlabel='t/min', ylabel='Count',
                xticks=np.linspace(x_min, x_max, 7, endpoint=True))

    colors = sns.color_palette()

    # Plot data
    for row, species_row in enumerate(species_order):
        for col, species in enumerate(species_row):
            ax = axs[row, col]
            
            if row == nrows - 2:  # Last row
                ylabel = 'Concentration (mM)'
                divide_by_nav = True
            else:
                ylabel = 'Count'
                divide_by_nav = False
            
            ax.set_ylabel(ylabel)
            
            if '+' in species:
                # Sum the counts for combined species
                species_list = species.split('+')
                ode_sum = np.sum([odeYs.get(sp, np.zeros_like(odeTs)) for sp in species_list], axis=0)
                
                if divide_by_nav:
                    ode_sum = ode_sum / NAV * 1e3  # Convert to mM
                
                odeHandle = ax.plot(odeTs, ode_sum, c=colors[1], label="ODE")[0]
            else:
                if species in odeYs:
                    ode_data = odeYs[species]
                    if divide_by_nav:
                        ode_data = ode_data / NAV * 1e3  # Convert to mM
                    odeHandle = ax.plot(odeTs, ode_data, c=colors[1], label="ODE")[0]
            
            # Plot RDME data for rows 3 and 4 (index 2 and 3) and for 'Grep' in the first row
            if row in [2, 3] or (row == 0 and species == 'Grep'):
                if '+' in species:
                    rdme_sum = np.sum([rdmeYs.get(sp, np.zeros_like(rdmeTs)) for sp in species_list], axis=0)
                    rdmeHandle = ax.plot(rdmeTs, rdme_sum, c=colors[0], label="RDME")[0]
                elif species in rdmeYs:
                    rdmeHandle = ax.plot(rdmeTs, rdmeYs[species], c=colors[0], label="RDME")[0]
            #handle activated ribosome cases
            elif species == 'ribosome_occupied':
                ribosome_species = [sp for sp in rdmeYs.keys() if sp.startswith('ribosome') and sp != 'ribosome']
                ribosome_sum = np.sum([rdmeYs[sp] for sp in ribosome_species], axis=0)
                rdmeHandle = ax.plot(rdmeTs, ribosome_sum, c=colors[0], label="RDME")[0]
            ax.set(title=species, **axargs)
            ax.legend()

        # Remove empty subplots in the row
        for col in range(len(species_row), ncols):
            fig.delaxes(axs[row, col])

    # Tight layout to adjust subplots spacing
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Add legend to the figure
    fig.legend([odeHandle, rdmeHandle] if 'rdmeHandle' in locals() else [odeHandle], 
            ["ODE", "RDME"] if 'rdmeHandle' in locals() else ["ODE"], 
            loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0))

    # Save the figure
    fig.savefig(os.path.join(fig_dir, os.path.basename(traj_fname) + '_selected_species_counts_and_conc.png') , dpi=dpi, bbox_inches='tight')


# species specific plot
def plot_species_specific(interested_species, traj_file, traj_dir, fig_dir, rdmeTs, rdmeYs, odeTs, odeYs,dpi=300):
    from matplotlib.gridspec import GridSpec
    gene_name = 'DG' + interested_species[1:]
    mrna_name = 'R' + interested_species[1:]
    names = sorted(filter(lambda x: interested_species in x or mrna_name in x, set(rdmeYs.keys()) | set(odeYs.keys())))
    nplots = len(names) - 2


    ncols = 3
    nrows = int(np.ceil((nplots - 1)  / ncols))  # -1 because we're replacing 3 DG2 plots with 1
    colors = sns.color_palette()
    # Arguments for the axes
    x_min = min(odeTs.min(), rdmeTs.min())
    x_max = max(odeTs.max(), rdmeTs.max())
    axargs = dict(xlim=(x_min, x_max),
                xlabel='t/min', ylabel='Count',
                xticks=np.linspace(x_min, x_max, 7, endpoint=True))
    # Create figure and subplots
    fig = plt.figure(figsize=(15, 6 * nrows))
    gs = GridSpec(nrows + 1, ncols, figure=fig)  # +1 for the DG2 stacked area chart
    fig.subplots_adjust(top=0.95)

    # Add a title to the figure without overlapping with subplots
    fig.suptitle(f"{interested_species} related species - {traj_file}", y=0.98)

    # Create the stacked area chart for DG2
    ax = fig.add_subplot(gs[0, :])
    if gene_name == 'DG4':
        gene_states = ['']
        gene_colors = ['#B0BEC5']
        ode_DG = rdmeYs[gene_name]
        ax.fill_between(rdmeTs, 0, ode_DG, color=gene_colors[0], alpha=0.5, label='free')
        ax.set(title=interested_species, **axargs)
        ax.legend()
    else:
        gene_sates = ['', '_G4d', '_G4d_G80d']
        ode_DG = rdmeYs[gene_name]
        ode_DG_G4d = rdmeYs[gene_name + gene_sates[1]]
        ode_DG_G4d_G80d = rdmeYs[gene_name + gene_sates[2]]

        gene_colors = ['#B0BEC5', '#4CAF50', '#1E88E5']
        ax.fill_between(rdmeTs, 0, ode_DG, color=gene_colors[0], alpha=0.5, label='free')
        ax.fill_between(rdmeTs, ode_DG, ode_DG + ode_DG_G4d, color=gene_colors[1], alpha=0.5, label='activated')
        ax.fill_between(rdmeTs, ode_DG + ode_DG_G4d, ode_DG + ode_DG_G4d + ode_DG_G4d_G80d, color=gene_colors[2], alpha=0.5, label='repressed')
        ax.set(title=interested_species, **axargs)
        ax.legend()

    # Initialize handles for legend
    handles = []
    labels = []
   
    # Plot the rest of the species
    for i, name in enumerate(names):
        if gene_name not in name:
            print(name)
            ax = fig.add_subplot(gs[(i) // ncols , (i) % ncols])
            if name in rdmeYs:
                rdmeHandle = ax.plot(rdmeTs, rdmeYs[name], c=colors[0], label="RDME")[0]
                if "RDME" not in labels:
                    handles.append(rdmeHandle)
                    labels.append("RDME")
            if name in odeYs:
                odeHandle = ax.plot(odeTs, odeYs[name], c=colors[1], label="ODE")[0]
                if "ODE" not in labels:
                    handles.append(odeHandle)
                    labels.append("ODE")
            ax.set(title=name, **axargs)

    # Tight layout to adjust subplots spacing
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    colors = sns.color_palette()

    # Add legend to the figure only if we have handles
    if handles:
        fig.legend(handles, labels, 
                loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02))

    # Save the figure
    fig.savefig(os.path.join(fig_dir,  traj_file + f'_{interested_species}_species_counts.png'), dpi=dpi, bbox_inches='tight')



def plot_species_by_region(interested_species, interested_regions, traj_file, fig_dir, regionTs, regionYs, region_dict, species_dict,dpi=300):
    '''
    Plot a heatmap-style trajectory for a single species across different regions
    
    Parameters:
        species (str): Name of the species to plot
        traj_file (str): Name of the trajectory file (for title)
        fig_dir (str): Directory to save the figure
        regionTs (array): Time points
        regionYs (dict): Dictionary containing species trajectories by region
        region_dict (dict): Dictionary containing region information
        species_dict (dict): Dictionary containing species information
    '''
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create figure with transparent background
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                  gridspec_kw={'height_ratios': [3, 1]},
                                  facecolor='none')
    fig.patch.set_alpha(0.0)
    ax1.patch.set_alpha(0.0)
    ax2.patch.set_alpha(0.0)
    
    # Get data for the species
    region_data = regionYs[interested_species]
    
    
    # Plot line plot with different colors for each region
    colors = sns.color_palette("husl", len(interested_regions))
    for idx, (region_ts, color) in enumerate(zip(region_data, colors)):
        ax1.plot(regionTs, region_ts, label=interested_regions[idx], color=color, alpha=0.7)
    
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Count')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Create heatmap data
    heatmap_data = np.array(region_data)
    
    # Plot heatmap
    im = ax2.imshow(heatmap_data, 
                    aspect='auto', 
                    extent=[regionTs[0], regionTs[-1], -0.5, len(interested_regions)-0.5],
                     cmap='magma',  # or 'inferno', 'plasma', 'RdYlBu_r', 'coolwarm'
                    interpolation='nearest')
    
    # Customize heatmap
    ax2.set_yticks(range(len(interested_regions)))
    ax2.set_yticklabels(interested_regions)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Regions')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Count', rotation=270, labelpad=15)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure with transparency
    plt.savefig(os.path.join(fig_dir, f"{traj_file}_{interested_species}_region_distribution.png"), 
                dpi=dpi, 
                bbox_inches='tight',
                transparent=True,
                facecolor='none',
                edgecolor='none')
    plt.show()
    plt.close()

