# Analysis and Visualization Scripts

This directory contains Python scripts for post-processing, statistical analysis, and visualization of RDME-ODE simulation results from the Yeast Galactose Switch model.

## Dependencies

```
numpy, pandas, matplotlib, scipy, h5py, tqdm
jLM (custom library for RDME file handling)
pyLM (for CME post-processing)
```

## Core Utilities

| Script                     | Description                                                                                                                                                                                  |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `traj_analysis_rdme.py`    | Core library providing trajectory loading functions (`get_traj`, `get_data_for_plot`) and species plotting routines (`plot_all_species`, `plot_tailored_species`, `plot_species_by_region`). |
| `matplotlib_pub_figure.py` | Publication-ready matplotlib styling with Dark2 color scheme, configurable DPI, and font settings.                                                                                           |
| `lm_functions.py`          | Low-level lattice manipulation utilities for particle operations (`deleteParticle`, `checkParticle`, `getParticlesInSite`).                                                                  |
| `getStats.py`              | Extracts particle statistics (counts and concentrations) by species and region from `.lm` files.                                                                                             |

## Comparison Scripts

| Script                                     | Description                                                                                                                                                                                                                      |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `figure_comparison.py`                     | Compares RDME trajectories between 2-3 directories. Generates species time-series plots with min/max shading, region-specific comparisons, and p-value significance plots (t-test/KS test). Supports caching for large datasets. |
| `figure_RDMECME_compare.py`                | Compares RDME-ODE and CME-ODE simulation results. Produces overlay plots for individual species, combined totals (G2, GAI), and generates separate legend figures.                                                               |
| `figure_rdme_ratio_comparison.py`          | Compares the ratio `ribosomeR2 / (R2 + ribosomeR2)` between conditions. Generates p-value plots for statistical significance.                                                                                                    |
| `figure_ribosome_distance_comparison.py`   | Compares ribosome distances to nucleus center and translating ribosome counts between conditions. Uses parallel processing for efficient HDF5 data extraction.                                                                   |
| `figure_ribosome_distance_significance.py` | Generates detailed p-value significance plots for ribosome distance comparisons.                                                                                                                                                 |

## Specialized Analyses

| Script                                   | Description                                                                                                                                                   |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `analyze_fold_change.py`                 | Calculates protein fold-change (final/initial) for RDME vs CME models. Reports Welch's t-test p-values, 95% confidence intervals, and Cohen's d effect sizes. |
| `query_species_abundance.py`             | Interactive command-line tool to query species abundances at specific time points from CSV statistics files.                                                  |
| `figure_ribo_in_use.py`                  | Plots translating ribosome trajectories with file-lock checking for concurrent access. Supports caching to CSV.                                               |
| `figure_RDME_phase_space_contour.py`     | Creates 2D phase space contour plots (mRNA R2 vs protein G2) using Gaussian KDE, with mean and boundary trajectories overlaid.                                |
| `figure_RDME_protein_production_rate.py` | Plots time-dependent protein production rate normalized by mRNA: `(G2_{t+1} - G2_t) / (R2_t + ribosomeR2_t)`.                                                 |

## Usage Patterns

### Basic Trajectory Comparison

```python
# In figure_comparison.py, modify the configuration section:
traj_dir1 = "/path/to/condition1"
traj_dir2 = "/path/to/condition2"
label1 = "Control"
label2 = "Treatment"
# Then run: python figure_comparison.py
```

### RDME vs CME Comparison

```python
# In figure_RDMECME_compare.py, modify:
rdme_traj_dir = "/path/to/rdme/results"
cme_traj_dir = "/path/to/cme/results"
fig_dir = "/output/directory"
# Then run: python figure_RDMECME_compare.py
```

### Querying Species Data

```bash
python query_species_abundance.py
# Follow interactive prompts to query specific species at time points
```

## Output Files

Scripts generate the following outputs in the specified `fig_dir`:

| Output Type                | Description                                         |
| -------------------------- | --------------------------------------------------- |
| `*_comparison.png`         | Species comparison plots                            |
| `*_species_statistics.csv` | Summary statistics (avg, std, min, max) per species |
| `pvalue_plots/`            | Directory containing p-value significance plots     |
| `region_pvalue_plots/`     | Region-specific p-value plots                       |
| `legend_separate.png`      | Standalone legend figure for multi-panel layouts    |
| `*.log`                    | Detailed processing logs                            |

## Caching

Several scripts implement caching to speed up repeated analyses:

```
dir_cache_*.pkl  # Cached trajectory data (figure_comparison.py)
trajectory_cache_*.pkl  # RDME/CME comparison cache (figure_RDMECME_compare.py)
*.csv  # Pre-computed statistics
```

Delete cache files to force reprocessing when source data changes.

## Notes

- All plots use publication-ready styling via `matplotlib_pub_figure.py`
- P-value calculations support both parametric (t-test) and non-parametric (KS test) methods
- Region-specific analyses require trajectory files with `_region.jsonl` suffix
- CME data is loaded via `jLM.CMEPostProcessing` from HDF5-based `.lm` files
