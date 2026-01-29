#!/usr/bin/env python3

import os
import glob
import pickle
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_pub_figure import setup_publication_style
from scipy.stats import ttest_ind, ks_2samp, chi2_contingency
colors = setup_publication_style(figure_size='medium', dpi=300)
color_dum = colors[0]

# File paths
# file1 = "/data2/2024_Yeast_GS/my_current_code/rdme_ode_results/20251121_ER_newR2diff/trajectory_comparison/ER_species_statistics.csv"
# file2 = "/data2/2024_Yeast_GS/my_current_code/rdme_ode_results/20251121_ER_newR2diff/trajectory_comparison/ER eff ribo_species_statistics.csv"
# file1_label = "ER"
# file2_label = "ER eff ribo"
#ER file
color_dum2 = colors[1]
# color_dum3 = colors[2]
# color_dum4 = colors[3]
color1 = colors[2]    
color2 = colors[3]   
color3 = colors[4] 
file1 = "/data2/2024_Yeast_GS/my_current_code/rdme_ode_results/20251121_ER_newR2diff/trajectory_comparison/no ER_species_statistics.csv"
file2 = "/data2/2024_Yeast_GS/my_current_code/rdme_ode_results/20251121_ER_newR2diff/trajectory_comparison/with ER_species_statistics.csv"
file1_label = "no ER"
file2_label = "with ER"
#Eff file
# color_dum2 = colors[1]
# color_dum3 = colors[2]
# # color_dum4 = colors[3]
# color1 = colors[3]    
# color2 = colors[4]   
# color3 = colors[5] 
# file1 = "/data2/2024_Yeast_GS/my_current_code/rdme_ode_results/20251121_EFFCHROMO_newR2/trajectory_comparison/ER_species_statistics.csv"
# file2 = "/data2/2024_Yeast_GS/my_current_code/rdme_ode_results/20251121_EFFCHROMO_newR2/trajectory_comparison/ER eff ribo_species_statistics.csv"
# file1_label = "ER"
# file2_label = "ER eff ribo"
# Logging
out_dir = os.path.dirname(file1)
log_file = os.path.join(out_dir, "rdme_r2_ratio_comparison.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

# ===== p-value utilities (aligned with figure_comparison.py) =====

def calculate_pvalue_timeseries(data1_trajectories, data2_trajectories, test_type='ttest', species_name=None):
    data1_array = np.array(data1_trajectories)
    data2_array = np.array(data2_trajectories)

    n_timepoints = data1_array.shape[1]
    p_values = np.zeros(n_timepoints)

    use_chi2 = (species_name is not None and species_name.startswith('RDME_DG'))
    actual_test_type = 'chi2' if use_chi2 else test_type

    for t in range(n_timepoints):
        values1 = data1_array[:, t]
        values2 = data2_array[:, t]
        try:
            values1 = values1[np.isfinite(values1)]
            values2 = values2[np.isfinite(values2)]
            if len(values1) < 2 or len(values2) < 2:
                p_values[t] = np.nan
                continue

            if np.var(values1) == 0 and np.var(values2) == 0:
                p_values[t] = 1.0 if np.mean(values1) == np.mean(values2) else 0.0
                continue
            elif np.var(values1) == 0 or np.var(values2) == 0:
                if actual_test_type == 'ttest':
                    epsilon = 1e-10
                    if np.var(values1) == 0:
                        values1 = values1 + np.random.normal(0, epsilon, len(values1))
                    if np.var(values2) == 0:
                        values2 = values2 + np.random.normal(0, epsilon, len(values2))
                    _, p_val = ttest_ind(values1, values2, equal_var=False)
                    p_values[t] = p_val if np.isfinite(p_val) else 1.0
                else:
                    _, p_val = ks_2samp(values1, values2)
                    p_values[t] = p_val if np.isfinite(p_val) else np.nan
                continue

            if use_chi2:
                count1_0 = np.sum(values1 == 0)
                count1_1 = np.sum(values1 == 1)
                count2_0 = np.sum(values2 == 0)
                count2_1 = np.sum(values2 == 1)
                contingency_table = np.array([[count1_0, count1_1],
                                              [count2_0, count2_1]])
                if np.any(contingency_table.sum(axis=0) == 0) or np.any(contingency_table.sum(axis=1) == 0):
                    p_values[t] = np.nan
                    continue
                chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
            elif actual_test_type == 'ttest':
                _, p_val = ttest_ind(values1, values2, equal_var=False)
            elif actual_test_type == 'ks':
                _, p_val = ks_2samp(values1, values2)
            else:
                raise ValueError("actual_test_type must be 'ttest', 'ks', or 'chi2'")

            p_values[t] = p_val if np.isfinite(p_val) else np.nan
        except Exception:
            p_values[t] = np.nan

    return p_values, actual_test_type

def find_alpha_05_crossings(time, p_values):
    crossings = []
    alpha = 0.05
    for i in range(1, len(p_values)):
        prev_val = p_values[i-1]
        curr_val = p_values[i]
        if (prev_val > alpha and curr_val < alpha) or (prev_val < alpha and curr_val > alpha):
            t_cross = time[i-1] + (time[i] - time[i-1]) * (alpha - prev_val) / (curr_val - prev_val)
            crossings.append(t_cross)
    if len(crossings) <= 1:
        return crossings
    merged_crossings = []
    current_group = [crossings[0]]
    for i in range(1, len(crossings)):
        if crossings[i] - current_group[-1] <= 1.0:
            current_group.append(crossings[i])
        else:
            merged_crossings.append(sum(current_group) / len(current_group))
            current_group = [crossings[i]]
    merged_crossings.append(sum(current_group) / len(current_group))
    return merged_crossings

def create_pvalue_plot(time, p_values, species_name, label1, label2, label3=None, fig_dir=None,
                      significance_levels=[0.001, 0.01, 0.05], test_type='ttest'):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, p_values, 'k-', linewidth=2, label='p-value')
    colors_h = ['red', 'orange', 'yellow']
    for i, sig_level in enumerate(significance_levels):
        ax.axhline(y=sig_level, color=colors_h[i], linestyle='--', alpha=0.7, label=f'p = {sig_level}')
    crossings = find_alpha_05_crossings(time, p_values)
    for i, t_cross in enumerate(crossings):
        ax.axvline(x=t_cross, color='green', linestyle='--', alpha=0.8, linewidth=2)
        y_pos = 0.05 * (2 ** (i % 3))
        ax.annotate(f'Cross: {t_cross:.1f}min',
                    xy=(t_cross, y_pos),
                    xytext=(10, 20),
                    textcoords='offset points',
                    ha='left', va='bottom',
                    fontsize=9, color='green', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1))
    specific_times = [10, 30, 60]
    marker_colors = ['blue', 'purple', 'red']
    for spec_time, marker_color in zip(specific_times, marker_colors):
        time_idx = np.argmin(np.abs(time - spec_time))
        actual_time = time[time_idx]
        p_val_at_time = p_values[time_idx]
        if abs(actual_time - spec_time) <= 2:
            ax.scatter(actual_time, p_val_at_time, color=marker_color, s=120,
                       marker='o', edgecolor='black', linewidth=2, zorder=10, alpha=0.9)
            ax.axvline(x=actual_time, color=marker_color, linestyle=':', alpha=0.6, linewidth=2)
            ax.annotate(f't={actual_time:.0f}min\np={p_val_at_time:.2e}',
                        xy=(actual_time, p_val_at_time),
                        xytext=(20, 25),
                        textcoords='offset points',
                        ha='left', va='bottom',
                        fontsize=10, color=marker_color, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor=marker_color),
                        arrowprops=dict(arrowstyle='->', color=marker_color, lw=2))
    if crossings:
        print(f"α=0.05 crossings for {species_name}: {[f'{t:.1f}min' for t in crossings]}")
    print(f"P-values at specific times for {species_name}:")
    for spec_time in specific_times:
        time_idx = np.argmin(np.abs(time - spec_time))
        actual_time = time[time_idx]
        p_val_at_time = p_values[time_idx]
        if abs(actual_time - spec_time) <= 2:
            print(f"  t={actual_time:.0f}min: p={p_val_at_time:.2e}")
    for i, sig_level in enumerate(significance_levels):
        if i == 0:
            ax.fill_between(time, 0, sig_level, where=(p_values <= sig_level),
                            color=colors_h[i], alpha=0.2, interpolate=True)
        else:
            prev_level = significance_levels[i-1]
            ax.fill_between(time, prev_level, sig_level, where=(p_values <= sig_level) & (p_values > prev_level),
                            color=colors_h[i], alpha=0.2, interpolate=True)
    ax.set_yscale('log')
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('p-value (log scale)')
    title = f'Statistical Significance: {species_name}\n({label1} vs {label2})' if not label3 else \
            f'Statistical Significance: {species_name}\n({label1} vs {label2} vs {label3})'
    ax.set_title(title, fontsize=12)
    ax.legend(framealpha=0.3, loc='best')
    ax.grid(True, alpha=0.3)
    test_name = 'T-test' if test_type == 'ttest' else 'Kolmogorov-Smirnov test' if test_type == 'ks' else 'Chi-square test'
    ax.text(0.02, 0.98, f'Test: {test_name}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    if out_dir:
        clean_species_name = species_name.replace(':', '_').replace('/', '_')
        filename = f'{clean_species_name}_pvalue_significance.png'
        fig_path = os.path.join(out_dir, filename)
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved p-value plot: {filename}")
    plt.close()
# Read the CSV files
print("Loading data files...")
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Filter for RDME_R2 and RDME_ribosomeR2 species
print("Extracting RDME_R2 and RDME_ribosomeR2 data...")

# For file 1 (ER_species_statistics.csv)
r2_df1 = df1[df1['Species'] == 'RDME_R2']
ribosome_r2_df1 = df1[df1['Species'] == 'RDME_ribosomeR2']

# For file 2 (with ER_species_statistics.csv)
r2_df2 = df2[df2['Species'] == 'RDME_R2']
ribosome_r2_df2 = df2[df2['Species'] == 'RDME_ribosomeR2']

def parse_csv_data(species_data):
    """Parse the comma-separated values in Time and Average columns"""
    if len(species_data) == 0:
        return pd.DataFrame(columns=['Time', 'Average', 'Min', 'Max'])

    row = species_data.iloc[0]
    times = [float(x) for x in row['Time'].split(',')]
    averages = [float(x) for x in row['Average'].split(',')]
    mins = [float(x) for x in row['Min'].split(',')]
    maxs = [float(x) for x in row['Max'].split(',')]
    return pd.DataFrame({'Time': times, 'Average': averages, 'Min': mins, 'Max': maxs})

# Parse the data for each species and condition
print("Parsing time series data...")
r2_data_1 = parse_csv_data(r2_df1)
ribosome_r2_data_1 = parse_csv_data(ribosome_r2_df1)

r2_data_2 = parse_csv_data(r2_df2)
ribosome_r2_data_2 = parse_csv_data(ribosome_r2_df2)

def calculate_ratio(r2_data, ribosome_r2_data, label):
    """Calculate the ratio RDME_R2/(RDME_R2+RDME_ribosomeR2)"""
    if len(r2_data) == 0 or len(ribosome_r2_data) == 0:
        return pd.DataFrame(columns=['Time', 'ratio', 'ratio_min', 'ratio_max', 'condition'])

    merged = pd.merge(r2_data, ribosome_r2_data, on='Time', suffixes=('_R2', '_ribosomeR2'))
    merged['ratio'] = merged['Average_ribosomeR2'] / (merged['Average_R2'] + merged['Average_ribosomeR2'])

    # Calculate min and max ratios
    # For min ratio: use min R2 with max ribosomeR2
    # For max ratio: use max R2 with min ribosomeR2
    merged['ratio_min'] = merged['Min_ribosomeR2'] / (merged['Min_R2'] + merged['Min_ribosomeR2'])
    merged['ratio_max'] = merged['Max_ribosomeR2'] / (merged['Max_R2'] + merged['Max_ribosomeR2'])

    merged['condition'] = label
    return merged[['Time', 'ratio', 'ratio_min', 'ratio_max', 'condition']]

# Calculate ratios for both datasets
print("Calculating ratios...")
ratio_df1 = calculate_ratio(r2_data_1, ribosome_r2_data_1, file1_label)
ratio_df2 = calculate_ratio(r2_data_2, ribosome_r2_data_2, file2_label)

# Combine data for plotting
combined_data = pd.concat([ratio_df1, ratio_df2], ignore_index=True)

# Create the comparison plot
print("Creating plot...")
fig, ax = plt.subplots()

# Plot each condition
for i, condition in enumerate([file1_label, file2_label]):
    data = combined_data[combined_data['condition'] == condition]
    color = color1 if condition == file1_label else color2

    # Plot the average line
    line = ax.plot(data['Time'], data['ratio'], label=condition,  color=color)

    # Add fill_between for min and max
    # ax.fill_between(data['Time'], data['ratio_min'], data['ratio_max'],
                    # alpha=0.1, color=color)

ax.set_xlabel('Time (min)')
# ax.set_title('Comparison of RDME_R2 Ratio Between eff and with eff Conditions')
ax.legend(framealpha=0.3, loc='best')
ax.grid(False)
# ax.set_yscale('log')
# Set y-axis to show ratio as percentage
# ax.set_ylabel("log")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

# Add dotted horizontal mean lines and labels (styled like figure_ribosome_distance_comparison.py)
mean_lines = {}
for condition, color in [(file1_label, color1), (file2_label, color2)]:
    data = combined_data[combined_data['condition'] == condition]
    if len(data) > 0:
        mean_val = float(np.mean(data['ratio']))
        mean_lines[condition] = mean_val
        ax.axhline(y=mean_val, color=color, linestyle='--', alpha=0.7, linewidth=1)

plt.tight_layout()

# Place mean value labels to the right of the axes
label_x = 1.02
for condition, color in [(file1_label, color1), (file2_label, color2)]:
    if condition in mean_lines:
        mv = mean_lines[condition]
        ax.text(label_x, mv, f'{mv:.2%}', transform=ax.get_yaxis_transform(),
                color=color, fontsize=8, verticalalignment='center', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7),
                clip_on=False)

# Save the plot
output_file = os.path.join(out_dir, "rdme_r2_ratio_comparison.png")
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved as {output_file}")

# ===== Build p-value plot from per-trajectory caches =====
def load_ratio_trajectories_from_cache(cache_path, species_r2='R2', species_ribo='ribosomeR2'):
    with open(cache_path, 'rb') as f:
        data_species, data_species_region, data_ode, rdmeTs, odeTs, regionTs, NAV = pickle.load(f)
    print(f"data_species keys: {list(data_species.keys())}")

    

    r2_trajs = data_species[species_r2]
    ribo_trajs = data_species[species_ribo]
    print(f"r2_trajs length: {len(r2_trajs)}")
    print(f"ribo_trajs length: {len(ribo_trajs)}")
    if len(r2_trajs) == 0 or len(ribo_trajs) == 0:
        return None, None, None
    n = min(len(r2_trajs), len(ribo_trajs))
    ratio_trajs = []
    for i in range(n):
        denom = (np.array(r2_trajs[i]) + np.array(ribo_trajs[i]))
        ratio = np.divide(np.array(ribo_trajs[i]), denom, out=np.zeros_like(denom, dtype=float), where=(denom != 0))
        ratio_trajs.append(ratio)
    # Return rdmeTs as-is; it may be None in some caches
    return rdmeTs, ratio_trajs, (species_r2, species_ribo)

try:
    comp_dir = os.path.dirname(file1)
    cache_files = sorted(glob.glob(os.path.join(comp_dir, "dir_cache_*.pkl")))
    if len(cache_files) < 2:
        logging.info("Fewer than two cache files found; skipping p-value plot.")
    else:
        cache_files = sorted(cache_files, key=lambda p: os.path.getmtime(p), reverse=True)[:2]

        rdmeTs_1, ratios_1, used_sp = load_ratio_trajectories_from_cache(cache_files[0])
        rdmeTs_2, ratios_2, _ = load_ratio_trajectories_from_cache(cache_files[1])

        if ratios_1 is None or ratios_2 is None:
            logging.info("Required species not found in cache; skipping p-value plot.")
        else:
            # Target time grid from CSV (condition 1)
            time_csv = combined_data[combined_data['condition'] == file1_label]['Time'].values.astype(float)

            def align_trajs(ratio_trajs, t_src, t_dst):
                """Align a list of trajectories to t_dst.
                If t_src is provided, interpolate to t_dst.
                If t_src is None, align by normalized index (0..1) mapping.
                """
                if ratio_trajs is None or len(ratio_trajs) == 0:
                    return None
                n_src = len(ratio_trajs[0])
                n_dst = len(t_dst)
                # Treat non-sequence t_src (e.g., numpy scalar) as missing
                use_src = (t_src is not None)
                if use_src:
                    try:
                        t_src_arr = np.asarray(t_src, dtype=float)
                        # If scalar (ndim==0), treat as missing
                        if t_src_arr.ndim == 0:
                            use_src = False
                    except Exception:
                        use_src = False
                if use_src:
                    # If already aligned
                    if t_src_arr.shape[0] == n_dst and np.allclose(t_src_arr, t_dst, atol=1e-9):
                        return ratio_trajs
                    # Interpolate using source times
                    out = []
                    for tr in ratio_trajs:
                        out.append(np.interp(t_dst, t_src_arr, tr))
                    return out
                else:
                    # Align by normalized index mapping
                    x_src = np.linspace(0.0, 1.0, n_src)
                    x_dst = np.linspace(0.0, 1.0, n_dst)
                    out = []
                    for tr in ratio_trajs:
                        out.append(np.interp(x_dst, x_src, tr))
                    return out

            ratios_1_interp = align_trajs(ratios_1, rdmeTs_1, time_csv)
            ratios_2_interp = align_trajs(ratios_2, rdmeTs_2, time_csv)

            if ratios_1_interp is None or ratios_2_interp is None:
                logging.info("Could not align ratio trajectories; skipping p-value plot.")
            else:
                p_vals, actual_test = calculate_pvalue_timeseries(ratios_1_interp, ratios_2_interp, test_type='ttest', species_name='RDME_ratio_R2')
                create_pvalue_plot(time_csv, p_vals, 'RDME_R2_ratio (ribosomeR2 / (R2 + ribosomeR2))',
                                   file1_label, file2_label, fig_dir=comp_dir, test_type=actual_test)
except Exception as e:
    logging.info(f"Could not create p-value plot for R2 ratio: {e}")

# Print some summary statistics
print("\nSummary Statistics:")
print("ER condition:")
print(f"  Mean ratio: {ratio_df1['ratio'].mean():.4f}")
print(f"  Std ratio: {ratio_df1['ratio'].std():.4f}")
print(f"  Min ratio: {ratio_df1['ratio'].min():.4f}")
print(f"  Max ratio: {ratio_df1['ratio'].max():.4f}")

print("with ER condition:")
print(f"  Mean ratio: {ratio_df2['ratio'].mean():.4f}")
print(f"  Std ratio: {ratio_df2['ratio'].std():.4f}")
print(f"  Min ratio: {ratio_df2['ratio'].min():.4f}")
print(f"  Max ratio: {ratio_df2['ratio'].max():.4f}")

plt.show()