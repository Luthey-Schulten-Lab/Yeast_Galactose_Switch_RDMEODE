"""
Steady State Analysis for Galactose Switch ODE System
Author: Tianyu Wu, 2024

This script uses the comprehensive ODE model to compute steady state values
for all 37 species in the galactose regulatory network.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from comprehensive_galactose_ode_system import ComprehensiveGalactoseODESystem
import time

class SteadyStateAnalyzer:
    """
    Analyzer to compute and characterize steady state solutions
    """
    
    def __init__(self, ode_system=None):
        """
        Initialize analyzer with ODE system
        """
        if ode_system is None:
            self.ode_system = ComprehensiveGalactoseODESystem()
        else:
            self.ode_system = ode_system
    
    def check_steady_state(self, t, y, threshold=1e-6):
        """
        Check if the system has reached steady state by comparing
        the rate of change to a threshold
        
        Parameters:
        -----------
        t : array
            Time points
        y : array
            Species concentrations over time (shape: n_species x n_timepoints)
        threshold : float
            Maximum allowed relative change rate for steady state
            
        Returns:
        --------
        bool : Whether system is at steady state
        float : Maximum relative change rate
        """
        # Calculate rates of change at final time point
        # Use last 10% of simulation
        n_points = len(t)
        start_idx = int(0.9 * n_points)
        
        if start_idx >= n_points - 1:
            start_idx = max(0, n_points - 10)
        
        max_rel_change = 0
        
        for i in range(y.shape[0]):
            if y[i, -1] > 1e-10:  # Only check non-zero species
                # Calculate relative change rate
                dy = y[i, -1] - y[i, start_idx]
                dt = t[-1] - t[start_idx]
                
                if dt > 0:
                    rel_change_rate = abs(dy / (y[i, -1] * dt))
                    max_rel_change = max(max_rel_change, rel_change_rate)
        
        is_steady_state = max_rel_change < threshold
        
        return is_steady_state, max_rel_change
    
    def analyze_convergence_times(self, t, y, threshold=1e-5, window_size=100):
        """
        Analyze when each species reaches steady state
        
        Parameters:
        -----------
        t : array
            Time points
        y : array
            Species concentrations over time (shape: n_species x n_timepoints)
        threshold : float
            Convergence threshold - max relative change over window
        window_size : int
            Number of points to use for checking convergence
            
        Returns:
        --------
        dict : Dictionary mapping species names to convergence times
        """
        convergence_times = {}
        
        n_points = len(t)
        window_size = min(window_size, n_points // 10)
        
        for species_name, idx in self.ode_system.species_indices.items():
            convergence_time = None
            
            # Skip if species is essentially zero throughout
            if np.max(y[idx]) < 1e-10:
                convergence_times[species_name] = 0.0
                continue
            
            # Check convergence at each time point
            for i in range(window_size, n_points):
                # Calculate relative change over the past window
                start_idx = i - window_size
                current_val = y[idx, i]
                
                if current_val > 1e-10:
                    # Calculate max relative change in window
                    window_vals = y[idx, start_idx:i+1]
                    max_val = np.max(window_vals)
                    min_val = np.min(window_vals)
                    rel_change = abs(max_val - min_val) / current_val
                    
                    if rel_change < threshold:
                        convergence_time = t[i]
                        break
            
            if convergence_time is None:
                convergence_time = t[-1]  # Not fully converged
            
            convergence_times[species_name] = convergence_time
        
        return convergence_times
    
    def get_steady_state(self, GAE_mM=11.1, GAI_mM=0, t_max=1000, 
                        threshold=1e-6, max_iterations=5, params=None):
        """
        Compute steady state by running simulation until convergence
        
        Parameters:
        -----------
        GAE_mM : float
            External galactose concentration (mM)
        GAI_mM : float
            Initial internal galactose concentration (mM)
        t_max : float
            Maximum simulation time (minutes)
        threshold : float
            Convergence threshold for steady state detection
        max_iterations : int
            Maximum number of simulation extensions if not converged
        params : dict
            Optional parameter dictionary (uses baseline if None)
            
        Returns:
        --------
        dict : Steady state results including all species concentrations
        """
        print(f"Computing steady state (GAE={GAE_mM} mM, GAI={GAI_mM} mM)...")
        
        current_t_max = t_max
        iteration = 0
        converged = False
        
        while iteration < max_iterations and not converged:
            # Run simulation
            t, y = self.ode_system.simulate(GAE_mM, GAI_mM, current_t_max, params)
            
            # Check for steady state
            converged, max_change = self.check_steady_state(t, y, threshold)
            
            print(f"  Iteration {iteration + 1}: t_max={current_t_max:.0f} min, "
                  f"max_rel_change={max_change:.2e}, converged={converged}")
            
            if not converged:
                # Extend simulation time
                current_t_max *= 2
                iteration += 1
        
        if not converged:
            print(f"  WARNING: Did not fully converge after {max_iterations} iterations")
            print(f"  Final max relative change rate: {max_change:.2e}")
        else:
            print(f"  Converged to steady state!")
        
        # Extract steady state values (final time point)
        steady_state = {}
        for species_name, idx in self.ode_system.species_indices.items():
            steady_state[species_name] = y[idx, -1]
        
        # Add some useful derived quantities
        steady_state['G2_total'] = (steady_state['G2'] + 
                                    steady_state['G2GAI'] + 
                                    steady_state['G2GAE'])
        
        steady_state['G1_total'] = steady_state['G1'] + steady_state['G1GAI']
        
        steady_state['G4_total'] = steady_state['G4'] + 2 * steady_state['G4d']
        
        steady_state['G80_nuclear'] = (steady_state['G80'] + 
                                       2 * steady_state['G80d'])
        
        steady_state['G80_cytoplasmic'] = (steady_state['G80C'] + 
                                           2 * steady_state['G80Cd'] + 
                                           2 * steady_state['G80G3i'])
        
        steady_state['G80_total'] = (steady_state['G80_nuclear'] + 
                                     steady_state['G80_cytoplasmic'])
        
        steady_state['G3_total'] = (steady_state['G3'] + 
                                    steady_state['G3i'] + 
                                    2 * steady_state['G80G3i'])
        
        # Add GAI in mM
        steady_state['GAI_mM'] = steady_state['GAI'] * 4.65e-8
        
        # Metadata
        steady_state['GAE_mM'] = GAE_mM
        steady_state['simulation_time'] = t[-1]
        steady_state['converged'] = converged
        steady_state['max_rel_change'] = max_change
        
        return steady_state, t, y
    
    def steady_state_dose_response(self, GAE_range, GAI_mM=0, t_max=1000,
                                  save_results=True):
        """
        Compute steady state for a range of external galactose concentrations
        
        Parameters:
        -----------
        GAE_range : array-like
            Range of external galactose concentrations (mM)
        GAI_mM : float
            Initial internal galactose concentration (mM)
        t_max : float
            Maximum simulation time per concentration
        save_results : bool
            Whether to save results to CSV
            
        Returns:
        --------
        pd.DataFrame : Steady state results for all concentrations
        """
        print(f"Computing steady state dose response for {len(GAE_range)} GAE values...")
        start_time = time.time()
        
        results_list = []
        
        for i, GAE_mM in enumerate(GAE_range):
            print(f"\n[{i+1}/{len(GAE_range)}] GAE = {GAE_mM:.4f} mM")
            
            steady_state, _, _ = self.get_steady_state(GAE_mM, GAI_mM, t_max)
            results_list.append(steady_state)
        
        # Convert to DataFrame
        df = pd.DataFrame(results_list)
        
        elapsed_time = time.time() - start_time
        print(f"\nDose response analysis completed in {elapsed_time:.2f} seconds")
        
        if save_results:
            filename = 'steady_state_dose_response.csv'
            df.to_csv(filename, index=False)
            print(f"Results saved to '{filename}'")
        
        return df
    
    def visualize_steady_state(self, steady_state, save_plot=True):
        """
        Create visualization of steady state concentrations
        
        Parameters:
        -----------
        steady_state : dict
            Steady state results from get_steady_state()
        save_plot : bool
            Whether to save the plot
        """
        fig = plt.figure(figsize=(16, 12))
        
        # 1. RNA species
        ax1 = plt.subplot(3, 3, 1)
        rna_species = ['R1', 'R2', 'R3', 'R4', 'R80', 'reporter_rna']
        rna_values = [steady_state[s] for s in rna_species]
        ax1.bar(range(len(rna_species)), rna_values, color='skyblue')
        ax1.set_xticks(range(len(rna_species)))
        ax1.set_xticklabels(rna_species, rotation=45)
        ax1.set_ylabel('Molecules/cell')
        ax1.set_title('RNA Species at Steady State')
        ax1.grid(True, alpha=0.3)
        
        # 2. Major protein species
        ax2 = plt.subplot(3, 3, 2)
        protein_species = ['G1', 'G2', 'G3', 'G3i', 'G4', 'G4d', 'reporter']
        protein_values = [steady_state[s] for s in protein_species]
        ax2.bar(range(len(protein_species)), protein_values, color='lightcoral')
        ax2.set_xticks(range(len(protein_species)))
        ax2.set_xticklabels(protein_species, rotation=45)
        ax2.set_ylabel('Molecules/cell')
        ax2.set_title('Major Protein Species')
        ax2.grid(True, alpha=0.3)
        
        # 3. G80 species distribution
        ax3 = plt.subplot(3, 3, 3)
        g80_species = ['G80', 'G80C', 'G80d', 'G80Cd', 'G80G3i']
        g80_values = [steady_state[s] for s in g80_species]
        ax3.bar(range(len(g80_species)), g80_values, color='mediumpurple')
        ax3.set_xticks(range(len(g80_species)))
        ax3.set_xticklabels(g80_species, rotation=45)
        ax3.set_ylabel('Molecules/cell')
        ax3.set_title('G80 Species Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. G2 total breakdown
        ax4 = plt.subplot(3, 3, 4)
        g2_components = ['G2', 'G2GAI', 'G2GAE']
        g2_values = [steady_state[s] for s in g2_components]
        colors_g2 = ['#1f77b4', '#ff7f0e', '#2ca02c']
        wedges, texts, autotexts = ax4.pie(g2_values, labels=g2_components, autopct='%1.1f%%',
                                            colors=colors_g2, startangle=90)
        ax4.set_title(f'G2 Total: {steady_state["G2_total"]:.1f} molecules/cell')
        
        # 5. DNA promoter states - G2
        ax5 = plt.subplot(3, 3, 5)
        dg2_states = ['DG2', 'DG2_G4d', 'DG2_G4d_G80d']
        dg2_values = [steady_state[s] for s in dg2_states]
        ax5.bar(range(len(dg2_states)), dg2_values, color='gold')
        ax5.set_xticks(range(len(dg2_states)))
        ax5.set_xticklabels(dg2_states, rotation=45)
        ax5.set_ylabel('Copy number')
        ax5.set_title('G2 Promoter States')
        ax5.grid(True, alpha=0.3)
        
        # 6. DNA promoter states - G1
        ax6 = plt.subplot(3, 3, 6)
        dg1_states = ['DG1', 'DG1_G4d', 'DG1_G4d_G80d']
        dg1_values = [steady_state[s] for s in dg1_states]
        ax6.bar(range(len(dg1_states)), dg1_values, color='orange')
        ax6.set_xticks(range(len(dg1_states)))
        ax6.set_xticklabels(dg1_states, rotation=45)
        ax6.set_ylabel('Copy number')
        ax6.set_title('G1 Promoter States')
        ax6.grid(True, alpha=0.3)
        
        # 7. Galactose species
        ax7 = plt.subplot(3, 3, 7)
        gal_species = ['GAI', 'G1GAI', 'G2GAI', 'G2GAE']
        gal_values = [steady_state[s] for s in gal_species]
        ax7.bar(range(len(gal_species)), gal_values, color='green')
        ax7.set_xticks(range(len(gal_species)))
        ax7.set_xticklabels(gal_species, rotation=45)
        ax7.set_ylabel('Molecules/cell')
        ax7.set_title(f'Galactose Species (GAI={steady_state["GAI_mM"]:.2f} mM)')
        ax7.grid(True, alpha=0.3)
        
        # 8. Total protein pools
        ax8 = plt.subplot(3, 3, 8)
        total_pools = {
            'G1 total': steady_state['G1_total'],
            'G2 total': steady_state['G2_total'],
            'G3 total': steady_state['G3_total'],
            'G4 total': steady_state['G4_total'],
            'G80 total': steady_state['G80_total']
        }
        ax8.bar(range(len(total_pools)), list(total_pools.values()), color='teal')
        ax8.set_xticks(range(len(total_pools)))
        ax8.set_xticklabels(list(total_pools.keys()), rotation=45)
        ax8.set_ylabel('Molecules/cell')
        ax8.set_title('Total Protein Pools')
        ax8.grid(True, alpha=0.3)
        
        # 9. Summary text
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        summary_text = f"""
        STEADY STATE SUMMARY
        ═══════════════════════════════
        
        Conditions:
          GAE: {steady_state['GAE_mM']:.2f} mM
          Simulation time: {steady_state['simulation_time']:.1f} min
          Converged: {steady_state['converged']}
          Max rel. change: {steady_state['max_rel_change']:.2e}
        
        Key Results:
          G2 total: {steady_state['G2_total']:.1f} molecules
          G1 total: {steady_state['G1_total']:.1f} molecules
          G3 total: {steady_state['G3_total']:.1f} molecules
          G4 total: {steady_state['G4_total']:.1f} molecules
          G80 total: {steady_state['G80_total']:.1f} molecules
          
          GAI: {steady_state['GAI_mM']:.3f} mM
          Reporter: {steady_state['reporter']:.1f} molecules
        
        G80 Distribution:
          Nuclear: {steady_state['G80_nuclear']:.1f}
          Cytoplasmic: {steady_state['G80_cytoplasmic']:.1f}
        """
        
        ax9.text(0.1, 0.5, summary_text, fontsize=10, 
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Steady State Analysis (GAE={steady_state["GAE_mM"]:.2f} mM)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_plot:
            filename = f'steady_state_GAE_{steady_state["GAE_mM"]:.2f}mM.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to '{filename}'")
        
        plt.show()
    
    def visualize_dose_response(self, df, save_plot=True):
        """
        Visualize steady state dose response curves
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dose response data from steady_state_dose_response()
        save_plot : bool
            Whether to save the plot
        """
        fig = plt.figure(figsize=(16, 10))
        
        # 1. G2 total dose response
        ax1 = plt.subplot(2, 3, 1)
        ax1.semilogx(df['GAE_mM'], df['G2_total'], 'o-', linewidth=2, markersize=6)
        ax1.set_xlabel('External Galactose (mM)')
        ax1.set_ylabel('G2 Total (molecules/cell)')
        ax1.set_title('G2 Dose Response')
        ax1.grid(True, alpha=0.3)
        
        # 2. Multiple protein dose responses
        ax2 = plt.subplot(2, 3, 2)
        ax2.semilogx(df['GAE_mM'], df['G1_total'], 'o-', label='G1 total')
        ax2.semilogx(df['GAE_mM'], df['G2_total'], 's-', label='G2 total')
        ax2.semilogx(df['GAE_mM'], df['G3_total'], '^-', label='G3 total')
        ax2.set_xlabel('External Galactose (mM)')
        ax2.set_ylabel('Molecules/cell')
        ax2.set_title('Multiple Protein Dose Responses')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. RNA dose responses
        ax3 = plt.subplot(2, 3, 3)
        ax3.semilogx(df['GAE_mM'], df['R1'], 'o-', label='R1')
        ax3.semilogx(df['GAE_mM'], df['R2'], 's-', label='R2')
        ax3.semilogx(df['GAE_mM'], df['R3'], '^-', label='R3')
        ax3.semilogx(df['GAE_mM'], df['R4'], 'd-', label='R4')
        ax3.set_xlabel('External Galactose (mM)')
        ax3.set_ylabel('RNA (molecules/cell)')
        ax3.set_title('RNA Dose Responses')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Internal galactose (GAI)
        ax4 = plt.subplot(2, 3, 4)
        ax4.semilogx(df['GAE_mM'], df['GAI_mM'], 'o-', linewidth=2, markersize=6, color='purple')
        ax4.set_xlabel('External Galactose (mM)')
        ax4.set_ylabel('Internal Galactose (mM)')
        ax4.set_title('GAI vs GAE')
        ax4.grid(True, alpha=0.3)
        
        # 5. G80 distribution
        ax5 = plt.subplot(2, 3, 5)
        ax5.semilogx(df['GAE_mM'], df['G80_nuclear'], 'o-', label='Nuclear')
        ax5.semilogx(df['GAE_mM'], df['G80_cytoplasmic'], 's-', label='Cytoplasmic')
        ax5.semilogx(df['GAE_mM'], df['G80_total'], '^-', label='Total')
        ax5.set_xlabel('External Galactose (mM)')
        ax5.set_ylabel('G80 (molecules/cell)')
        ax5.set_title('G80 Localization vs GAE')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. G2 promoter occupancy
        ax6 = plt.subplot(2, 3, 6)
        ax6.semilogx(df['GAE_mM'], df['DG2'], 'o-', label='Free')
        ax6.semilogx(df['GAE_mM'], df['DG2_G4d'], 's-', label='G4d bound')
        ax6.semilogx(df['GAE_mM'], df['DG2_G4d_G80d'], '^-', label='G4d+G80d bound')
        ax6.set_xlabel('External Galactose (mM)')
        ax6.set_ylabel('Promoter State')
        ax6.set_title('G2 Promoter Occupancy')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Steady State Dose Response Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_plot:
            filename = 'steady_state_dose_response_plots.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Dose response plots saved to '{filename}'")
        
        plt.show()


def main():
    """
    Main function to compute steady state at GAE = 11.1 mM and save all species
    """
    print("=" * 70)
    print("STEADY STATE ANALYSIS FOR GALACTOSE REGULATORY NETWORK")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = SteadyStateAnalyzer()
    
    # Compute steady state at standard conditions
    print("\nComputing steady state at GAE = 11.1 mM...")
    steady_state, t, y = analyzer.get_steady_state(GAE_mM=11.1, GAI_mM=0)
    
    # Display key results
    print("\n" + "=" * 70)
    print("KEY STEADY STATE RESULTS")
    print("=" * 70)
    print(f"\nConditions:")
    print(f"  GAE: {steady_state['GAE_mM']:.2f} mM")
    print(f"  Simulation time: {steady_state['simulation_time']:.1f} min")
    print(f"  Converged: {steady_state['converged']}")
    print(f"  Max relative change: {steady_state['max_rel_change']:.2e}")
    
    print(f"\nRNA Species (molecules/cell):")
    print(f"  R1:  {steady_state['R1']:.4f}")
    print(f"  R2:  {steady_state['R2']:.4f}")
    print(f"  R3:  {steady_state['R3']:.4f}")
    print(f"  R4:  {steady_state['R4']:.4f}")
    print(f"  R80: {steady_state['R80']:.4f}")
    print(f"  reporter_rna: {steady_state['reporter_rna']:.4f}")
    
    print(f"\nProtein Species (molecules/cell):")
    print(f"  G1:  {steady_state['G1']:.2f}")
    print(f"  G2:  {steady_state['G2']:.2f}")
    print(f"  G3:  {steady_state['G3']:.2f}")
    print(f"  G3i: {steady_state['G3i']:.2f}")
    print(f"  G4:  {steady_state['G4']:.2f}")
    print(f"  G4d: {steady_state['G4d']:.2f}")
    print(f"  Reporter: {steady_state['reporter']:.2f}")
    
    print(f"\nG80 Species (molecules/cell):")
    print(f"  G80:    {steady_state['G80']:.2f}")
    print(f"  G80C:   {steady_state['G80C']:.2f}")
    print(f"  G80d:   {steady_state['G80d']:.2f}")
    print(f"  G80Cd:  {steady_state['G80Cd']:.2f}")
    print(f"  G80G3i: {steady_state['G80G3i']:.2f}")
    
    print(f"\nTotal Protein Pools (molecules/cell):")
    print(f"  G1 total:  {steady_state['G1_total']:.2f}")
    print(f"  G2 total:  {steady_state['G2_total']:.2f}")
    print(f"  G3 total:  {steady_state['G3_total']:.2f}")
    print(f"  G4 total:  {steady_state['G4_total']:.2f}")
    print(f"  G80 total: {steady_state['G80_total']:.2f}")
    print(f"  G80 nuclear:      {steady_state['G80_nuclear']:.2f}")
    print(f"  G80 cytoplasmic:  {steady_state['G80_cytoplasmic']:.2f}")
    
    print(f"\nGalactose Species:")
    print(f"  GAI (mM): {steady_state['GAI_mM']:.4f}")
    print(f"  GAI (molecules): {steady_state['GAI']:.2f}")
    print(f"  G1GAI: {steady_state['G1GAI']:.2f}")
    print(f"  G2GAI: {steady_state['G2GAI']:.2f}")
    print(f"  G2GAE: {steady_state['G2GAE']:.2f}")
    
    print(f"\nDNA Promoter States:")
    print(f"  DG1: {steady_state['DG1']:.6f}  DG1_G4d: {steady_state['DG1_G4d']:.6f}  DG1_G4d_G80d: {steady_state['DG1_G4d_G80d']:.6f}")
    print(f"  DG2: {steady_state['DG2']:.6f}  DG2_G4d: {steady_state['DG2_G4d']:.6f}  DG2_G4d_G80d: {steady_state['DG2_G4d_G80d']:.6f}")
    print(f"  DG3: {steady_state['DG3']:.6f}  DG3_G4d: {steady_state['DG3_G4d']:.6f}  DG3_G4d_G80d: {steady_state['DG3_G4d_G80d']:.6f}")
    print(f"  DGrep: {steady_state['DGrep']:.6f}  DGrep_G4d: {steady_state['DGrep_G4d']:.6f}  DGrep_G4d_G80d: {steady_state['DGrep_G4d_G80d']:.6f}")
    print(f"  DG80: {steady_state['DG80']:.6f}  DG80_G4d: {steady_state['DG80_G4d']:.6f}  DG80_G4d_G80d: {steady_state['DG80_G4d_G80d']:.6f}")
    
    # Save all species to CSV
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Convert to DataFrame with one row containing all species
    steady_state_df = pd.DataFrame([steady_state])
    
    # Reorder columns for better readability
    # RNA species first
    rna_cols = ['R1', 'R2', 'R3', 'R4', 'R80', 'reporter_rna']
    
    # Protein species
    protein_cols = ['G1', 'G2', 'G3', 'G3i', 'G4', 'G4d', 'G80', 'G80C', 
                    'G80d', 'G80Cd', 'G80G3i', 'reporter']
    
    # DNA species
    dna_cols = ['DG1', 'DG2', 'DG3', 'DGrep', 'DG80',
                'DG1_G4d', 'DG2_G4d', 'DG3_G4d', 'DGrep_G4d', 'DG80_G4d',
                'DG1_G4d_G80d', 'DG2_G4d_G80d', 'DG3_G4d_G80d', 
                'DGrep_G4d_G80d', 'DG80_G4d_G80d']
    
    # Galactose species
    gal_cols = ['GAI', 'GAI_mM', 'G1GAI', 'G2GAI', 'G2GAE']
    
    # Total pools
    total_cols = ['G1_total', 'G2_total', 'G3_total', 'G4_total', 'G80_total',
                  'G80_nuclear', 'G80_cytoplasmic']
    
    # Metadata
    meta_cols = ['GAE_mM', 'simulation_time', 'converged', 'max_rel_change']
    
    # Combine in order
    ordered_cols = meta_cols + rna_cols + protein_cols + dna_cols + gal_cols + total_cols
    
    # Make sure all columns exist
    ordered_cols = [col for col in ordered_cols if col in steady_state_df.columns]
    
    # Reorder
    steady_state_df = steady_state_df[ordered_cols]
    
    # Save to CSV
    output_file = 'steady_state_all_species.csv'
    steady_state_df.to_csv(output_file, index=False)
    print(f"\nAll species saved to: {output_file}")
    print(f"Total species saved: {len(ordered_cols)}")
    
    # Also save in transposed format for easier viewing
    output_file_transposed = 'steady_state_all_species_transposed.csv'
    steady_state_transposed = pd.DataFrame({
        'Species': ordered_cols,
        'Steady_State_Value': [steady_state[col] for col in ordered_cols]
    })
    steady_state_transposed.to_csv(output_file_transposed, index=False)
    print(f"Transposed format saved to: {output_file_transposed}")
    
    # Analyze convergence times for each species
    print("\n" + "=" * 70)
    print("ANALYZING CONVERGENCE TIMES")
    print("=" * 70)
    print("\nAnalyzing when each species reaches steady state...")
    
    convergence_times = analyzer.analyze_convergence_times(t, y, threshold=1e-5)
    
    # Create DataFrame with convergence times and steady state values
    convergence_df = pd.DataFrame({
        'Species': list(convergence_times.keys()),
        'Convergence_Time_min': list(convergence_times.values()),
        'Steady_State_Value': [steady_state.get(sp, 0) for sp in convergence_times.keys()]
    })
    
    # Sort by convergence time
    convergence_df = convergence_df.sort_values('Convergence_Time_min')
    
    # Save convergence times
    output_file_convergence = 'species_convergence_times.csv'
    convergence_df.to_csv(output_file_convergence, index=False)
    print(f"\nConvergence times saved to: {output_file_convergence}")
    
    # Display convergence time statistics
    print("\n" + "=" * 70)
    print("CONVERGENCE TIME SUMMARY")
    print("=" * 70)
    
    # Group by convergence speed
    fast_converge = convergence_df[convergence_df['Convergence_Time_min'] < 100]
    medium_converge = convergence_df[(convergence_df['Convergence_Time_min'] >= 100) & 
                                     (convergence_df['Convergence_Time_min'] < 1000)]
    slow_converge = convergence_df[convergence_df['Convergence_Time_min'] >= 1000]
    
    print(f"\nFast converging species (<100 min): {len(fast_converge)}")
    print(f"Medium converging species (100-1000 min): {len(medium_converge)}")
    print(f"Slow converging species (>1000 min): {len(slow_converge)}")
    
    print(f"\n10 Fastest converging species:")
    print("-" * 70)
    for idx, row in fast_converge.head(10).iterrows():
        print(f"  {row['Species']:<20s}: {row['Convergence_Time_min']:>8.2f} min  (value: {row['Steady_State_Value']:.4e})")
    
    print(f"\n10 Slowest converging species:")
    print("-" * 70)
    for idx, row in convergence_df.tail(10).iterrows():
        print(f"  {row['Species']:<20s}: {row['Convergence_Time_min']:>8.2f} min  (value: {row['Steady_State_Value']:.4e})")
    
    # RNA convergence times
    print(f"\nRNA species convergence times:")
    print("-" * 70)
    rna_species = ['R1', 'R2', 'R3', 'R4', 'R80', 'reporter_rna']
    for sp in rna_species:
        conv_time = convergence_times.get(sp, 0)
        ss_value = steady_state.get(sp, 0)
        print(f"  {sp:<15s}: {conv_time:>8.2f} min  (value: {ss_value:.4f})")
    
    # Key protein convergence times
    print(f"\nKey protein species convergence times:")
    print("-" * 70)
    key_proteins = ['G1', 'G2', 'G3', 'G3i', 'G4', 'G4d', 'G80', 'G80Cd', 'G80G3i', 'reporter']
    for sp in key_proteins:
        conv_time = convergence_times.get(sp, 0)
        ss_value = steady_state.get(sp, 0)
        print(f"  {sp:<15s}: {conv_time:>8.2f} min  (value: {ss_value:.4f})")
    
    # Create combined output with convergence times
    print("\n" + "=" * 70)
    print("CREATING COMBINED OUTPUT WITH CONVERGENCE TIMES")
    print("=" * 70)
    
    # Create comprehensive output
    comprehensive_output = []
    for species_name in analyzer.ode_system.species_indices.keys():
        comprehensive_output.append({
            'Species': species_name,
            'Steady_State_Value': steady_state.get(species_name, 0),
            'Convergence_Time_min': convergence_times.get(species_name, 0)
        })
    
    comprehensive_df = pd.DataFrame(comprehensive_output)
    comprehensive_df = comprehensive_df.sort_values('Convergence_Time_min')
    
    output_file_comprehensive = 'steady_state_with_convergence_times.csv'
    comprehensive_df.to_csv(output_file_comprehensive, index=False)
    print(f"\nComprehensive output saved to: {output_file_comprehensive}")
    
    print("\n" + "=" * 70)
    print("STEADY STATE ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  1. {output_file} - All species (wide format)")
    print(f"  2. {output_file_transposed} - All species (tall format)")
    print(f"  3. {output_file_convergence} - Convergence times sorted")
    print(f"  4. {output_file_comprehensive} - Combined steady state + convergence times")
    
    return analyzer, steady_state, convergence_times


if __name__ == "__main__":
    analyzer, steady_state, convergence_times = main()

