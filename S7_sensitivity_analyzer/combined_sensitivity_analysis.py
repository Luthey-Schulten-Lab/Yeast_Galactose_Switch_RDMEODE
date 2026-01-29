#!/usr/bin/env python3
"""
Combined Sensitivity Analysis for Galactose Switch ODE System

This script performs comprehensive parameter sensitivity analysis:
1. Generates individual parameter perturbation plots
2. Screens parameters for fold-change threshold qualification  
3. Creates heatmaps for qualifying parameter pairs

Author: Tianyu Wu, 2025
Combined from comprehensive_galactose_ode_system.py and sensitivity_screening_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from datetime import datetime
from itertools import combinations
from multiprocessing import Pool, cpu_count
from functools import partial
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rdme_ode.matplotlib_pub_figure import setup_publication_style

# Import the ODE system
from comprehensive_galactose_ode_system import (
    ComprehensiveGalactoseODESystem, 
    PERTURBATION_LOWER_BOUND,
    PERTURBATION_UPPER_BOUND,
    evaluate_parameter_sweep,
    evaluate_two_parameter_heatmap
)

class CombinedSensitivityAnalyzer:
    """
    Combined sensitivity analyzer that performs complete workflow:
    1. Individual parameter analysis
    2. Parameter screening and qualification
    3. Heatmap generation for qualifying pairs
    """
    
    def __init__(self, output_dir=None, regenerate_plots=False):
        self.ode_system = ComprehensiveGalactoseODESystem()
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"combined_sensitivity_results_{timestamp}"
        if not regenerate_plots:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Create subdirectories
            self.individual_dir = os.path.join(self.output_dir, "individual_plots")
            self.screening_dir = os.path.join(self.output_dir, "screening_results")
            self.heatmap_dir = os.path.join(self.output_dir, "qualifying_heatmaps")
            self.summary_dir = os.path.join(self.output_dir, "summary")
            
            for directory in [self.individual_dir, self.screening_dir, self.heatmap_dir, self.summary_dir]:
                os.makedirs(directory, exist_ok=True)
        
            print(f"Results will be saved to: {self.output_dir}/")
    
    def step1_individual_parameter_analysis(self, n_points=20, GAE_mM=11.1, GAI_mM=0, 
                                          use_parallel=True, n_processes=None):
        """
        Step 1: Generate individual parameter perturbation plots
        """
        print(f"\\n=== STEP 1: INDIVIDUAL PARAMETER ANALYSIS ===")
        
        if n_processes is None:
            n_processes = min(cpu_count(), len(self.ode_system.params))
        
        baseline_g2 = self.ode_system.get_G2_at_60min(GAE_mM, GAI_mM)
        print(f"Baseline G2: {baseline_g2:.1f} molecules/cell")
        print(f"Analyzing {len(self.ode_system.params)} parameters with {n_points} points each")
        print(f"Using {n_processes} CPU cores...")
        
        start_time = time.time()
        
        if use_parallel and len(self.ode_system.params) > 1:
            # Parallel processing
            param_items = list(self.ode_system.params.items())
            eval_func = partial(evaluate_parameter_sweep, 
                              ode_system=self.ode_system, n_points=n_points, 
                              GAE_mM=GAE_mM, GAI_mM=GAI_mM)
            
            with Pool(processes=n_processes) as pool:
                results = pool.map(eval_func, param_items)
        else:
            # Sequential processing
            results = []
            for param_name, param_val in self.ode_system.params.items():
                result = evaluate_parameter_sweep((param_name, param_val), 
                                                self.ode_system, n_points, GAE_mM, GAI_mM)
                results.append(result)
        
        elapsed_time = time.time() - start_time
        print(f"Parameter sweeps completed in {elapsed_time:.2f} seconds")
        
        # Generate plots
        print("Generating individual parameter plots...")
        self._plot_individual_parameters(results, baseline_g2)
        
        # Save data
        self._save_individual_data(results, baseline_g2)
        
        return results
    
    def step2_parameter_screening(self, individual_results, threshold=2.0):
        """
        Step 2: Screen parameters for fold-change qualification
        """
        print(f"\\n=== STEP 2: PARAMETER SCREENING ===")
        print(f"Fold-change threshold: {threshold}x")
        
        screening_results = {}
        qualifying_params = []
        
        for result_tuple in individual_results:
            param_name, result_data = result_tuple
            g2_values = result_data['g2_values']
            baseline_g2 = self.ode_system.get_G2_at_60min(11.1, 0)  # Use consistent baseline
            
            max_g2 = np.max(g2_values)
            min_g2 = np.min(g2_values)
            
            # Calculate maximum fold change
            max_fold_increase = max_g2 / baseline_g2 if baseline_g2 > 0 else 0
            max_fold_decrease = baseline_g2 / min_g2 if min_g2 > 0 else float('inf')
            max_fold_change = max(max_fold_increase, max_fold_decrease)
            
            qualifies = max_fold_change >= threshold
            
            screening_results[param_name] = {
                'max_fold_change': max_fold_change,
                'qualifies': qualifies,
                'max_g2': max_g2,
                'min_g2': min_g2,
                'baseline_g2': baseline_g2,
                'baseline_param_value': self.ode_system.params[param_name]
            }
            
            if qualifies:
                qualifying_params.append(param_name)
        
        # Summary
        n_qualifying = len(qualifying_params)
        total_params = len(screening_results)
        
        print(f"Total parameters tested: {total_params}")
        print(f"Parameters meeting {threshold}x threshold: {n_qualifying}")
        print(f"Qualification rate: {n_qualifying/total_params*100:.1f}%")
        
        if n_qualifying > 0:
            print(f"\\nTop 10 qualifying parameters:")
            print("-" * 60)
            sorted_results = sorted(screening_results.items(), 
                                  key=lambda x: x[1]['max_fold_change'], reverse=True)
            
            for i, (param_name, result) in enumerate(sorted_results[:10]):
                if result['qualifies']:
                    print(f"{i+1:2d}. {param_name:20s}: {result['max_fold_change']:.2f}x")
        
        # Save screening results
        self._save_screening_results(screening_results, threshold)
        
        return screening_results, qualifying_params
    
    def step3_qualifying_heatmaps(self, qualifying_params, n_points=12, GAE_mM=11.1, GAI_mM=0,
                                use_parallel=True, n_processes=None, max_pairs=None):
        """
        Step 3: Generate heatmaps for qualifying parameter pairs
        """
        print(f"\\n=== STEP 3: QUALIFYING PARAMETER HEATMAPS ===")
        
        if len(qualifying_params) < 2:
            print("Need at least 2 qualifying parameters for heatmaps")
            return {}
        
        # Generate all possible pairs
        all_pairs = list(combinations(qualifying_params, 2))
        n_total_pairs = len(all_pairs)
        
        if max_pairs is not None and max_pairs < n_total_pairs:
            print(f"Limiting to first {max_pairs} pairs out of {n_total_pairs} total")
            all_pairs = all_pairs[:max_pairs]
        
        print(f"Qualifying parameters: {len(qualifying_params)}")
        print(f"Parameter pairs to analyze: {len(all_pairs)}")
        print(f"Total simulations: {len(all_pairs) * n_points**2:,}")
        
        if n_processes is None:
            n_processes = min(cpu_count(), max(1, cpu_count() // 2))
        
        print(f"Using {n_processes} CPU cores...")
        
        start_time = time.time()
        
        if use_parallel and len(all_pairs) > 1:
            # Parallel processing
            param_info_list = []
            for param1_name, param2_name in all_pairs:
                param1_val = self.ode_system.params[param1_name]
                param2_val = self.ode_system.params[param2_name]
                param_info_list.append(((param1_name, param1_val), (param2_name, param2_val)))
            
            eval_func = partial(evaluate_two_parameter_heatmap,
                              ode_system=self.ode_system, n_points=n_points,
                              GAE_mM=GAE_mM, GAI_mM=GAI_mM)
            
            with Pool(processes=n_processes) as pool:
                parallel_results = pool.map(eval_func, param_info_list)
            
            # Convert to dictionary
            results = {}
            for result in parallel_results:
                pair_key = (result['param1_name'], result['param2_name'])
                results[pair_key] = result
        else:
            # Sequential processing
            results = {}
            from comprehensive_galactose_ode_system import TwoParameterAnalyzer
            analyzer = TwoParameterAnalyzer(self.ode_system)
            
            for i, (param1_name, param2_name) in enumerate(all_pairs):
                print(f"Processing pair {i+1}/{len(all_pairs)}: {param1_name} vs {param2_name}")
                result = analyzer.two_parameter_heatmap(param1_name, param2_name,
                                                      n_points=n_points, GAE_mM=GAE_mM,
                                                      GAI_mM=GAI_mM, save_plot=False)
                results[(param1_name, param2_name)] = result
        
        elapsed_time = time.time() - start_time
        print(f"Heatmap generation completed in {elapsed_time:.2f} seconds")
        
        # Generate heatmap plots
        self._plot_qualifying_heatmaps(results, GAE_mM, GAI_mM)
        
        # Generate summary
        self._generate_heatmap_summary(results, GAE_mM, GAI_mM)
        
        return results
    
    def _plot_individual_parameters(self, results, baseline_g2):
        """Generate individual parameter plots with dotted lines"""
        print("Creating individual parameter plots...")
        
        for i, result_tuple in enumerate(results):
            setup_publication_style(figure_size='medium')
            plt.figure()
            
            # Unpack the tuple from evaluate_parameter_sweep
            param_name, result_data = result_tuple
            param_values = result_data['param_range']
            g2_values = result_data['g2_values']
            baseline_val = result_data['baseline_param']
            
            # Plot with dotted line
            plt.plot(param_values/baseline_val, g2_values, '-', alpha=0.8, label='G2 response')
            
            # Highlight baseline
          
            plt.axhline(y=baseline_g2, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Baseline')
            plt.axvline(x=baseline_val, color='red', linestyle='--', alpha=0.8, linewidth=2)
            plt.xlabel(f'{param_name} (parameter value)')
            plt.ylabel('G2 molecules/cell')
            plt.title(f'Parameter Sensitivity: {param_name}')
            # plt.xscale('log')
            # plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Calculate fold change for title
            max_fold = np.max(g2_values) / baseline_g2
            min_fold = baseline_g2 / np.min(g2_values)
            max_fold_change = max(max_fold, min_fold)
            
            plt.title(f'Parameter Sensitivity: {param_name}\n'
                     f'Max Fold Change: {max_fold_change:.2f}x')
            
            plt.tight_layout()
            
            # Save plot
            safe_name = param_name.replace('/', '_').replace(':', '_')
            filename = f'param_{i+1:03d}_{safe_name}.png'
            filepath = os.path.join(self.individual_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Individual plots saved to: {self.individual_dir}/")
    
    def _save_individual_data(self, results, baseline_g2):
        """Save individual parameter data to CSV and complete sweep data"""
        # Save summary data
        summary_data = []
        for result_tuple in results:
            param_name, result_data = result_tuple
            g2_values = result_data['g2_values']
            max_fold = np.max(g2_values) / baseline_g2
            min_fold = baseline_g2 / np.min(g2_values)
            max_fold_change = max(max_fold, min_fold)
            
            summary_data.append({
                'Parameter': param_name,
                'Baseline_Value': result_data['baseline_param'],
                'Min_G2': np.min(g2_values),
                'Max_G2': np.max(g2_values),
                'Baseline_G2': baseline_g2,
                'Max_Fold_Change': max_fold_change
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Max_Fold_Change', ascending=False)
        
        summary_filepath = os.path.join(self.individual_dir, 'individual_parameter_data.csv')
        summary_df.to_csv(summary_filepath, index=False)
        print(f"Individual parameter summary saved to: {summary_filepath}")
        
        # Save complete sweep data for exact plot recreation
        complete_data = []
        for result_tuple in results:
            param_name, result_data = result_tuple
            param_range = result_data['param_range']
            g2_values = result_data['g2_values']
            baseline_param = result_data['baseline_param']
            
            # Create one row per parameter value
            for param_val, g2_val in zip(param_range, g2_values):
                complete_data.append({
                    'Parameter': param_name,
                    'Parameter_Value': param_val,
                    'G2_Value': g2_val,
                    'Baseline_Parameter_Value': baseline_param,
                    'Baseline_G2': baseline_g2
                })
        
        complete_df = pd.DataFrame(complete_data)
        complete_filepath = os.path.join(self.individual_dir, 'complete_parameter_sweep_data.csv')
        complete_df.to_csv(complete_filepath, index=False)
        print(f"Complete parameter sweep data saved to: {complete_filepath}")
    
    def _save_screening_results(self, screening_results, threshold):
        """Save screening results to CSV"""
        data = []
        for param_name, result in screening_results.items():
            data.append({
                'Parameter': param_name,
                'Baseline_Value': result['baseline_param_value'],
                'Baseline_G2': result['baseline_g2'],
                'Min_G2': result['min_g2'],
                'Max_G2': result['max_g2'],
                'Max_Fold_Change': result['max_fold_change'],
                'Qualifies': result['qualifies'],
                'Threshold': threshold
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Max_Fold_Change', ascending=False)
        
        # Save full results
        screening_file = os.path.join(self.screening_dir, 'parameter_screening_results.csv')
        df.to_csv(screening_file, index=False)
        
        # Save only qualifying parameters
        qualifying_df = df[df['Qualifies'] == True].copy()
        if len(qualifying_df) > 0:
            qualifying_file = os.path.join(self.screening_dir, 'qualifying_parameters.csv')
            qualifying_df.to_csv(qualifying_file, index=False)
        
        print(f"Screening results saved to: {self.screening_dir}/")
    
    def _plot_qualifying_heatmaps(self, results, GAE_mM, GAI_mM):
        """Generate individual heatmap plots"""
        print("Creating qualifying heatmap plots...")
        baseline_g2 = self.ode_system.get_G2_at_60min(GAE_mM, GAI_mM)
        
        for i, ((param1_name, param2_name), result) in enumerate(results.items()):
            setup_publication_style(figure_size='large')
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(result['g2_matrix'], aspect='auto', origin='lower',
                          extent=[result['param2_fold_range'][0], result['param2_fold_range'][-1],
                                 result['param1_fold_range'][0], result['param1_fold_range'][-1]],
                          cmap='viridis')
            
            # Add baseline crosshairs
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Baseline')
            ax.axvline(x=1, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            ax.set_xlabel(f'{param2_name} (fold change)', fontsize=12)
            ax.set_ylabel(f'{param1_name} (fold change)', fontsize=12)
            ax.set_title(f'G2 at 60 min: {param1_name} vs {param2_name}', fontsize=14, pad=15)
            
            # Position colorbar to maximize heatmap space
            cbar = plt.colorbar(im, ax=ax, label='G2 molecules/cell', shrink=0.8, pad=0.02)
            cbar.ax.tick_params(labelsize=10)
            
            # Add compact statistics box
            g2_min = np.min(result['g2_matrix'])
            g2_max = np.max(result['g2_matrix'])
            max_synergy = g2_max / baseline_g2
            
            stats_text = f'Range: {g2_min:.0f}-{g2_max:.0f}\\nBaseline: {baseline_g2:.0f}\\nMax: {max_synergy:.1f}x'
            
            # Smaller, compact text box in top-left corner
            props = dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8, 
                        edgecolor='navy', linewidth=0.5)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top', horizontalalignment='left',
                   bbox=props)
            
            # Legend in top-right with small size
            ax.legend(loc='upper right', fontsize=9, framealpha=0.8)
            
            # Maximize plot area
            plt.subplots_adjust(left=0.1, right=0.88, top=0.92, bottom=0.1)
            
            # Save plot
            safe_name1 = param1_name.replace('/', '_').replace(':', '_')
            safe_name2 = param2_name.replace('/', '_').replace(':', '_')
            filename = f'heatmap_{i+1:03d}_{safe_name1}_vs_{safe_name2}.png'
            filepath = os.path.join(self.heatmap_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        
        print(f"Heatmap plots saved to: {self.heatmap_dir}/")
    
    def _generate_heatmap_summary(self, results, GAE_mM, GAI_mM):
        """Generate heatmap summary analysis and save complete heatmap data"""
        baseline_g2 = self.ode_system.get_G2_at_60min(GAE_mM, GAI_mM)
        
        # Calculate statistics
        pair_effects = []
        for (param1_name, param2_name), result in results.items():
            g2_min = np.min(result['g2_matrix'])
            g2_max = np.max(result['g2_matrix'])
            max_synergy = g2_max / baseline_g2
            
            pair_effects.append({
                'param1': param1_name,
                'param2': param2_name,
                'g2_min': g2_min,
                'g2_max': g2_max,
                'max_synergy': max_synergy,
                'baseline_g2': baseline_g2
            })
        
        # Sort by synergy
        pair_effects.sort(key=lambda x: x['max_synergy'], reverse=True)
        
        # Save summary CSV
        results_df = pd.DataFrame(pair_effects)
        summary_file = os.path.join(self.summary_dir, 'heatmap_analysis.csv')
        results_df.to_csv(summary_file, index=False)
        
        # Save complete heatmap data for exact plot recreation
        self._save_complete_heatmap_data(results, baseline_g2)
        
        print(f"\\nTop 5 most synergistic parameter pairs:")
        print("-" * 60)
        for i, effect in enumerate(pair_effects[:5]):
            print(f"{i+1}. {effect['param1']} + {effect['param2']}: {effect['max_synergy']:.2f}x")
        
        print(f"Heatmap summary saved to: {summary_file}")
    
    def _save_complete_heatmap_data(self, results, baseline_g2):
        """Save complete heatmap matrices and parameter ranges for exact recreation"""
        import pickle
        import json
        
        # Save each heatmap's complete data
        for (param1_name, param2_name), result in results.items():
            # Create safe filename
            safe_name1 = param1_name.replace('/', '_').replace(':', '_')
            safe_name2 = param2_name.replace('/', '_').replace(':', '_')
            
            # Save as pickle for exact matrix recreation (includes numpy arrays)
            pickle_filename = f'heatmap_data_{safe_name1}_vs_{safe_name2}.pkl'
            pickle_filepath = os.path.join(self.summary_dir, pickle_filename)
            
            heatmap_data = {
                'param1_name': param1_name,
                'param2_name': param2_name,
                'g2_matrix': result['g2_matrix'],
                'param1_fold_range': result['param1_fold_range'],
                'param2_fold_range': result['param2_fold_range'],
                'baseline_g2': baseline_g2
            }
            
            with open(pickle_filepath, 'wb') as f:
                pickle.dump(heatmap_data, f)
        
        # Also save a JSON index file for easy access
        index_data = []
        for (param1_name, param2_name), result in results.items():
            safe_name1 = param1_name.replace('/', '_').replace(':', '_')
            safe_name2 = param2_name.replace('/', '_').replace(':', '_')
            
            index_data.append({
                'param1_name': param1_name,
                'param2_name': param2_name,
                'pickle_filename': f'heatmap_data_{safe_name1}_vs_{safe_name2}.pkl',
                'g2_min': float(np.min(result['g2_matrix'])),
                'g2_max': float(np.max(result['g2_matrix'])),
                'max_synergy': float(np.max(result['g2_matrix']) / baseline_g2),
                'matrix_shape': result['g2_matrix'].shape
            })
        
        index_file = os.path.join(self.summary_dir, 'heatmap_data_index.json')
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        print(f"Complete heatmap data saved to: {self.summary_dir}/heatmap_data_*.pkl")
        print(f"Heatmap data index saved to: {index_file}")
    
    def plot_from_cached_data(self, cached_dir=None, plot_types='all'):
        """
        Generate plots from cached analysis data without rerunning computations
        
        Parameters:
        -----------
        cached_dir : str, optional
            Path to directory containing cached results. If None, searches for most recent.
        plot_types : str or list, optional
            Types of plots to generate: 'all', 'individual', 'screening', 'heatmaps', 'summary'
        """
        print(f"\n=== PLOTTING FROM CACHED DATA ===")
        
        # Find cached directory if not provided
        if cached_dir is None:
            cached_dir = self._find_most_recent_cache()
            if cached_dir is None:
                print("No cached data found. Run full analysis first.")
                return
        
        if not os.path.exists(cached_dir):
            print(f"Cached directory not found: {cached_dir}")
            return
            
        print(f"Using cached data from: {cached_dir}")
        
        # Set up plot types
        if isinstance(plot_types, str):
            if plot_types == 'all':
                plot_types = ['individual', 'screening', 'heatmaps', 'summary']
            else:
                plot_types = [plot_types]
        
        # Create new timestamped directory for cached plots to avoid overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = f"cached_plots_{timestamp}"
        os.makedirs(plot_dir, exist_ok=True)
        
        # Create subdirectories matching original structure
        subdirs = ['individual_plots', 'screening_results', 'qualifying_heatmaps', 'summary']
        for subdir in subdirs:
            os.makedirs(os.path.join(plot_dir, subdir), exist_ok=True)
        
        print(f"Plots will be saved to new directory: {plot_dir}/")
        
        # Generate plots based on cached data
        if 'individual' in plot_types or plot_types == 'all':
            self._plot_cached_individual_summary(cached_dir, plot_dir)
            self._regenerate_individual_plots(cached_dir, plot_dir)
        
        if 'screening' in plot_types or plot_types == 'all':
            self._plot_cached_screening(cached_dir, plot_dir)
            
        if 'heatmaps' in plot_types or plot_types == 'all':
            self._plot_cached_heatmap_summary(cached_dir, plot_dir)
            self._regenerate_qualifying_heatmaps(cached_dir, plot_dir)
            
        if 'summary' in plot_types or plot_types == 'all':
            self._plot_cached_overview(cached_dir, plot_dir)
        
        print(f"\nCached plots generated in: {plot_dir}/")
    
    def _find_most_recent_cache(self):
        """Find the most recent cached results directory"""
        current_dir = os.getcwd()
        cache_dirs = []
        
        for item in os.listdir(current_dir):
            if item.startswith('combined_sensitivity_results_') and os.path.isdir(item):
                cache_dirs.append(item)
        
        if not cache_dirs:
            return None
        
        # Sort by timestamp in directory name
        cache_dirs.sort(reverse=True)
        return cache_dirs[0]
    
    def _plot_cached_individual_summary(self, cached_dir, plot_dir):
        """Generate summary plot from cached individual parameter data"""
        individual_data_file = os.path.join(cached_dir, 'individual_plots', 'individual_parameter_data.csv')
        
        if not os.path.exists(individual_data_file):
            print(f"Individual data file not found: {individual_data_file}")
            return
        
        print("Generating individual parameter summary plot...")
        
        # Read cached data
        df = pd.read_csv(individual_data_file)
        
        # Create summary plot
        setup_publication_style(figure_size='medium')
        fig, ax = plt.subplots()
        
        # Plot 1: Parameter sensitivity ranking (all parameters with fold change >= 2)
        qualifying_params = df[df['Max_Fold_Change'] >= 2.0].copy()
        y_pos = range(len(qualifying_params))
        
        ax.barh(y_pos, qualifying_params['Max_Fold_Change'], alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(qualifying_params['Parameter'], fontsize=8)
        ax.set_xlabel('Maximum Fold Change')
        ax.set_title(f'Parameters with Fold Change ≥ 2x ({len(qualifying_params)} total)')
        ax.axvline(x=2.0, color='red', linestyle='--', alpha=0.8, label='2x threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot in individual_plots subdirectory
        individual_plot_dir = os.path.join(plot_dir, 'individual_plots')
        plot_file = os.path.join(individual_plot_dir, 'individual_parameter_summary.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        # Plot 2: Distribution of fold changes
        fig, ax2 = plt.subplots()
        ax2.hist(df['Max_Fold_Change'], bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(x=2.0, color='red', linestyle='--', alpha=0.8, label='2x threshold')
        ax2.set_xlabel('Maximum Fold Change')
        ax2.set_ylabel('Number of Parameters')
        ax2.set_title('Distribution of Parameter Sensitivities')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot in individual_plots subdirectory
        individual_plot_dir = os.path.join(plot_dir, 'individual_plots')
        plot_file = os.path.join(individual_plot_dir, 'individual_parameter_histogram.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Individual parameter summary saved to: {plot_file}")
    
    def _plot_cached_screening(self, cached_dir, plot_dir):
        """Generate screening summary plot from cached data"""
        screening_file = os.path.join(cached_dir, 'screening_results', 'parameter_screening_results.csv')
        
        if not os.path.exists(screening_file):
            print(f"Screening data file not found: {screening_file}")
            return
        
        print("Generating parameter screening summary plot...")
        
        # Read cached data
        df = pd.read_csv(screening_file)
        
        # Create screening summary plot
        setup_publication_style(figure_size='large')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Qualifying vs non-qualifying
        qualifying_counts = df['Qualifies'].value_counts()
        ax1.pie(qualifying_counts.values, labels=['Non-qualifying', 'Qualifying'], 
                autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Parameter Qualification (2x threshold)\\n'
                     f'{qualifying_counts[True]} / {len(df)} parameters qualify')
        
        # Plot 2: Top qualifying parameters
        qualifying_df = df[df['Qualifies'] == True].head(10)
        if len(qualifying_df) > 0:
            y_pos = range(len(qualifying_df))
            ax2.barh(y_pos, qualifying_df['Max_Fold_Change'], alpha=0.7, color='green')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(qualifying_df['Parameter'], fontsize=8)
            ax2.set_xlabel('Maximum Fold Change')
            ax2.set_title('Top 10 Qualifying Parameters')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot in screening_results subdirectory
        screening_plot_dir = os.path.join(plot_dir, 'screening_results')
        plot_file = os.path.join(screening_plot_dir, 'screening_summary.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Screening summary saved to: {plot_file}")
    
    def _plot_cached_heatmap_summary(self, cached_dir, plot_dir):
        """Generate heatmap analysis summary from cached data"""
        heatmap_file = os.path.join(cached_dir, 'summary', 'heatmap_analysis.csv')
        
        if not os.path.exists(heatmap_file):
            print(f"Heatmap analysis file not found: {heatmap_file}")
            return
        
        print("Generating heatmap summary plot...")
        
        # Read cached data
        df = pd.read_csv(heatmap_file)
        
        # Create heatmap summary plot with maximum space usage
        setup_publication_style(figure_size='large')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Top synergistic pairs
        top_pairs = df.head(20)  # Show more pairs
        pair_labels = [f"{row['param1']}+{row['param2']}" for _, row in top_pairs.iterrows()]
        y_pos = range(len(top_pairs))
        
        bars = ax1.barh(y_pos, top_pairs['max_synergy'], alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(pair_labels, fontsize=7)
        ax1.set_xlabel('Maximum Synergy (fold change)', fontsize=12)
        ax1.set_title('Top 20 Most Synergistic Parameter Pairs', fontsize=14, pad=10)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars for top 5
        for i, (bar, value) in enumerate(zip(bars[:5], top_pairs['max_synergy'][:5])):
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{value:.1f}x', va='center', fontsize=8, fontweight='bold')
        
        # Plot 2: Distribution of synergies with better styling
        n, bins, patches = ax2.hist(df['max_synergy'], bins=25, alpha=0.8, 
                                   edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Maximum Synergy (fold change)', fontsize=12)
        ax2.set_ylabel('Number of Parameter Pairs', fontsize=12)
        ax2.set_title('Distribution of Parameter Pair Synergies', fontsize=14, pad=10)
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text box with compact formatting
        mean_synergy = df['max_synergy'].mean()
        median_synergy = df['max_synergy'].median()
        max_synergy = df['max_synergy'].max()
        
        stats_text = f'Mean: {mean_synergy:.1f}x\\nMedian: {median_synergy:.1f}x\\nMax: {max_synergy:.1f}x'
        
        # Smaller, more compact text box
        props = dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7, 
                    edgecolor='navy', linewidth=0.5)
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, 
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=props)
        
        # Maximize plot area
        plt.subplots_adjust(left=0.15, right=0.98, top=0.92, bottom=0.08, wspace=0.25)
        
        # Save plot in summary subdirectory
        summary_plot_dir = os.path.join(plot_dir, 'summary')
        plot_file = os.path.join(summary_plot_dir, 'heatmap_synergy_summary.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        
        print(f"Heatmap synergy summary saved to: {plot_file}")
    
    def _regenerate_individual_plots(self, cached_dir, plot_dir):
        """Regenerate exact individual parameter plots from complete cached data"""
        complete_data_file = os.path.join(cached_dir, 'individual_plots', 'complete_parameter_sweep_data.csv')
        
        if not os.path.exists(complete_data_file):
            print(f"Complete parameter sweep data file not found: {complete_data_file}")
            print("Using existing individual plots instead of regenerating...")
            return
        
        print("Regenerating exact individual parameter plots from complete cached data...")
        
        # Read complete sweep data
        df = pd.read_csv(complete_data_file)
        individual_plot_dir = os.path.join(plot_dir, 'individual_plots')
        
        # Group by parameter to recreate each plot
        param_groups = df.groupby('Parameter')
        
        for i, (param_name, param_data) in enumerate(param_groups):
            setup_publication_style(figure_size='medium')
            plt.figure()
            
            # Extract exact data
            param_values = param_data['Parameter_Value'].values
            g2_values = param_data['G2_Value'].values
            baseline_param = param_data['Baseline_Parameter_Value'].iloc[0]
            baseline_g2 = param_data['Baseline_G2'].iloc[0]
            
            # Calculate fold change for title
            max_fold = np.max(g2_values) / baseline_g2
            min_fold = baseline_g2 / np.min(g2_values)
            max_fold_change = max(max_fold, min_fold)
            
            # Plot exact data
            plt.plot(param_values/baseline_param, g2_values, '-', alpha=0.8, label='G2 response')
            
            # Highlight baseline
            plt.axhline(y=baseline_g2, color='red', linestyle='--', alpha=0.8, 
                       linewidth=2, label='Baseline')
            plt.axvline(x=baseline_param/baseline_param, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            plt.xlabel(f'{param_name} (fold change)')
            plt.ylabel('G2 molecules/cell')
            plt.title(f'Parameter Sensitivity: {param_name}\n'
                     f'Max Fold Change: {max_fold_change:.2f}x')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            
            # Save plot
            safe_name = param_name.replace('/', '_').replace(':', '_')
            filename = f'param_{i+1:03d}_{safe_name}_exact.png'
            filepath = os.path.join(individual_plot_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Exact individual plots regenerated and saved to: {individual_plot_dir}/")
    
    def _regenerate_qualifying_heatmaps(self, cached_dir, plot_dir):
        """Regenerate exact qualifying heatmaps from complete cached data"""
        import pickle
        import json
        
        heatmap_index_file = os.path.join(cached_dir, 'summary', 'heatmap_data_index.json')
        
        if not os.path.exists(heatmap_index_file):
            print(f"Heatmap data index file not found: {heatmap_index_file}")
            print("Using existing heatmap plots instead of regenerating...")
            return
        
        print("Regenerating exact qualifying heatmaps from complete cached data...")
        
        # Read heatmap index
        with open(heatmap_index_file, 'r') as f:
            heatmap_index = json.load(f)
        
        heatmap_plot_dir = os.path.join(plot_dir, 'qualifying_heatmaps')
        
        # Regenerate heatmaps using exact saved data
        for i, heatmap_info in enumerate(heatmap_index):
            pickle_filepath = os.path.join(cached_dir, 'summary', heatmap_info['pickle_filename'])
            
            if not os.path.exists(pickle_filepath):
                print(f"Heatmap data file not found: {pickle_filepath}")
                continue
            
            # Load exact heatmap data
            with open(pickle_filepath, 'rb') as f:
                heatmap_data = pickle.load(f)
            
            setup_publication_style(figure_size='medium')
            fig, ax = plt.subplots()
            
            # Extract exact data
            param1_name = heatmap_data['param1_name']
            param2_name = heatmap_data['param2_name']
            g2_matrix = heatmap_data['g2_matrix']
            param1_fold_range = heatmap_data['param1_fold_range']
            param2_fold_range = heatmap_data['param2_fold_range']
            baseline_g2 = heatmap_data['baseline_g2']
            
            # Create exact heatmap
            im = ax.imshow(g2_matrix, aspect='auto', origin='lower',
                          extent=[param2_fold_range[0], param2_fold_range[-1],
                                 param1_fold_range[0], param1_fold_range[-1]],
                          cmap='viridis')
            
            # Add baseline crosshairs
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Baseline')
            ax.axvline(x=1, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            ax.set_xlabel(f'{param2_name} (fold change)')
            ax.set_ylabel(f'{param1_name} (fold change)')
            ax.set_title(f'G2 at 60 min: {param1_name} vs {param2_name}')
            
            # Position colorbar to maximize heatmap space
            cbar = plt.colorbar(im, ax=ax, label='G2 molecules/cell', shrink=0.8, pad=0.02)
            cbar.ax.tick_params(labelsize=8)
            
            # Add compact statistics box
            g2_min = np.min(g2_matrix)
            g2_max = np.max(g2_matrix)
            max_synergy = g2_max / baseline_g2
            
            stats_text = f'Range: {g2_min:.0f}-{g2_max:.0f}\nBaseline: {baseline_g2:.0f}\nMax: {max_synergy:.1f}x'
            
            # Smaller, compact text box in top-left corner
            props = dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8, 
                        edgecolor='navy', linewidth=0.5)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top', horizontalalignment='left',
                   bbox=props)

            # Legend in top-right with small size
            ax.legend(loc='upper right', fontsize=8, framealpha=0.8)
            
            # Maximize plot area
            plt.subplots_adjust(left=0.1, right=0.88, top=0.92, bottom=0.1)
            
            # Save plot
            safe_name1 = param1_name.replace('/', '_').replace(':', '_')
            safe_name2 = param2_name.replace('/', '_').replace(':', '_')
            filename = f'heatmap_{i+1:03d}_{safe_name1}_vs_{safe_name2}_exact.png'
            filepath = os.path.join(heatmap_plot_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close()
        
        print(f"Exact qualifying heatmaps regenerated and saved to: {heatmap_plot_dir}/")
    
    def _plot_cached_overview(self, cached_dir, plot_dir):
        """Generate comprehensive overview plot from all cached data"""
        print("Generating comprehensive overview plot...")
        
        # Read all cached data files
        individual_file = os.path.join(cached_dir, 'individual_plots', 'individual_parameter_data.csv')
        screening_file = os.path.join(cached_dir, 'screening_results', 'parameter_screening_results.csv')
        heatmap_file = os.path.join(cached_dir, 'summary', 'heatmap_analysis.csv')
        
        if not all(os.path.exists(f) for f in [individual_file, screening_file]):
            print("Required data files not found for overview plot")
            return
        
        individual_df = pd.read_csv(individual_file)
        screening_df = pd.read_csv(screening_file)
        
        # Create comprehensive overview
        setup_publication_style(figure_size='large')
        fig = plt.figure(figsize=(18, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Top individual parameters
        ax1 = fig.add_subplot(gs[0, 0])
        top_individual = individual_df.head(10)
        y_pos = range(len(top_individual))
        ax1.barh(y_pos, top_individual['Max_Fold_Change'], alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_individual['Parameter'], fontsize=6)
        ax1.set_xlabel('Fold Change')
        ax1.set_title('Top 10 Individual Parameters')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Qualification pie chart
        ax2 = fig.add_subplot(gs[0, 1])
        qualifying_counts = screening_df['Qualifies'].value_counts()
        ax2.pie(qualifying_counts.values, labels=['Non-qualifying', 'Qualifying'], 
                autopct='%1.1f%%', startangle=90)
        ax2.set_title(f'Parameter Qualification\\n({qualifying_counts[True]}/{len(screening_df)} qualify)')
        
        # Plot 3: Fold change distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(individual_df['Max_Fold_Change'], bins=15, alpha=0.7, edgecolor='black')
        ax3.axvline(x=2.0, color='red', linestyle='--', alpha=0.8, label='2x threshold')
        ax3.set_xlabel('Fold Change')
        ax3.set_ylabel('Count')
        ax3.set_title('Sensitivity Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: G2 range for top parameters
        ax4 = fig.add_subplot(gs[1, :])
        top_params = individual_df.head(15)
        x_pos = range(len(top_params))
        baseline_g2 = top_params['Baseline_G2'].iloc[0]
        
        ax4.fill_between(x_pos, top_params['Min_G2'], top_params['Max_G2'], 
                        alpha=0.3, label='G2 Range')
        ax4.plot(x_pos, [baseline_g2]*len(x_pos), 'r--', linewidth=2, label='Baseline')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(top_params['Parameter'], rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('G2 molecules/cell')
        ax4.set_title('G2 Response Range for Top 15 Parameters')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Heatmap synergies (if available)
        if os.path.exists(heatmap_file):
            heatmap_df = pd.read_csv(heatmap_file)
            ax5 = fig.add_subplot(gs[2, :2])
            top_pairs = heatmap_df.head(10)
            pair_labels = [f"{row['param1']}+{row['param2']}" for _, row in top_pairs.iterrows()]
            y_pos = range(len(top_pairs))
            
            ax5.barh(y_pos, top_pairs['max_synergy'], alpha=0.7, color='orange')
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(pair_labels, fontsize=8)
            ax5.set_xlabel('Maximum Synergy')
            ax5.set_title('Top 10 Parameter Pair Synergies')
            ax5.grid(True, alpha=0.3)
            
            # Plot 6: Synergy distribution
            ax6 = fig.add_subplot(gs[2, 2])
            ax6.hist(heatmap_df['max_synergy'], bins=15, alpha=0.7, edgecolor='black', color='orange')
            ax6.set_xlabel('Synergy')
            ax6.set_ylabel('Count')
            ax6.set_title('Synergy Distribution')
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Galactose Switch Sensitivity Analysis Overview', fontsize=16, y=0.98)
        
        # Save plot in summary subdirectory
        summary_plot_dir = os.path.join(plot_dir, 'summary')
        plot_file = os.path.join(summary_plot_dir, 'analysis_overview.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis overview saved to: {plot_file}")
    
    def run_complete_analysis(self, threshold=2.0, individual_n_points=20, heatmap_n_points=12, 
                            GAE_mM=11.1, GAI_mM=0, use_parallel=True, max_heatmap_pairs=None):
        """
        Run the complete 3-step analysis pipeline
        """
        print(f"=== COMBINED SENSITIVITY ANALYSIS ===")
        print(f"Output directory: {self.output_dir}")
        print(f"Fold-change threshold: {threshold}x")
        
        # Step 1: Individual parameter analysis
        individual_results = self.step1_individual_parameter_analysis(
            n_points=individual_n_points, GAE_mM=GAE_mM, GAI_mM=GAI_mM, use_parallel=use_parallel)
        
        # Step 2: Parameter screening
        screening_results, qualifying_params = self.step2_parameter_screening(
            individual_results, threshold=threshold)
        
        # Step 3: Qualifying heatmaps
        if len(qualifying_params) >= 2:
            heatmap_results = self.step3_qualifying_heatmaps(
                qualifying_params, n_points=heatmap_n_points, GAE_mM=GAE_mM, GAI_mM=GAI_mM,
                use_parallel=use_parallel, max_pairs=max_heatmap_pairs)
        else:
            print(f"Only {len(qualifying_params)} parameters qualified - skipping heatmaps")
            heatmap_results = {}
        
        print(f"\\n=== ANALYSIS COMPLETE ===")
        print(f"All results saved to: {self.output_dir}/")
        
        return individual_results, screening_results, qualifying_params, heatmap_results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Combined sensitivity analysis")
    parser.add_argument('--threshold', type=float, default=2.0,
                       help='Fold-change threshold (default: 2.0)')
    parser.add_argument('--individual_points', type=int, default=20,
                       help='Points for individual plots (default: 20)')
    parser.add_argument('--heatmap_points', type=int, default=20,
                       help='Grid size for heatmaps (default: 20)')
    parser.add_argument('--max_heatmap_pairs', type=int, default=None,
                       help='Maximum heatmap pairs (default: all)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: auto-generated)')
    parser.add_argument('--plot_cached', action='store_true',
                       help='Generate plots from cached data instead of running analysis')
    parser.add_argument('--cached_dir', type=str, default=None,
                       help='Directory with cached results (default: most recent)')
    parser.add_argument('--plot_types', type=str, default='all',
                       help='Plot types: all, individual, screening, heatmaps, summary')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CombinedSensitivityAnalyzer(output_dir=args.output_dir, regenerate_plots=args.plot_cached)
    
    if args.plot_cached:
        # Generate plots from cached data
        analyzer.plot_from_cached_data(
            cached_dir=args.cached_dir,
            plot_types=args.plot_types
        )
        return None
    else:
        # Run complete analysis
        results = analyzer.run_complete_analysis(
            threshold=args.threshold,
            individual_n_points=args.individual_points,
            heatmap_n_points=args.heatmap_points,
            max_heatmap_pairs=args.max_heatmap_pairs
        )
    
    return results

def example_cached_plotting():
    """
    Example function showing how to use the enhanced cached data system for exact plotting
    """
    # Initialize analyzer
    analyzer = CombinedSensitivityAnalyzer()
    
    # Generate all plot types from most recent cached data (exact recreation)
    analyzer.plot_from_cached_data()
    
    # Or generate specific plot types
    # analyzer.plot_from_cached_data(plot_types=['individual', 'summary'])
    
    # Or use specific cached directory
    # analyzer.plot_from_cached_data(cached_dir='combined_sensitivity_results_20250818_150619')
    
    print("\\nData saved with enhanced caching:")
    print("- individual_plots/complete_parameter_sweep_data.csv: Full parameter sweep data")
    print("- summary/heatmap_data_*.pkl: Complete heatmap matrices")
    print("- summary/heatmap_data_index.json: Index of heatmap files")
    print("\\nCached plotting generates EXACT recreations of original plots!")

if __name__ == "__main__":
    main()