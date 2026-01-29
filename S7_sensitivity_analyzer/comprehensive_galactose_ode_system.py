"""
Comprehensive ODE System for Galactose Switch - Complete Model
Author: Tianyu Wu, 2024
Based on CME-ODE simulation code by David Bianchi

This code converts ALL reactions (both CME and ODE) into a single comprehensive ODE system
for complete parameter sensitivity analysis of G2 abundance.

Species List (42 total):
- RNA species: R1, R2, R3, R4, R80, reporter_rna  
- Protein species: G1, G2, G3, G3i, G4, G4d, G80, G80C, G80d, G80Cd, G80G3i, reporter
- DNA species: DG1, DG2, DG3, DGrep, DG80 (all in various binding states)
- Galactose species: GAI, G1GAI, G2GAI, G2GAE
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Global perturbation bounds - can be modified to change analysis range
PERTURBATION_LOWER_BOUND = 0.1  # Lower bound as fraction of baseline (e.g., 0.5 = 50% of baseline)
PERTURBATION_UPPER_BOUND = 10  # Upper bound as multiplier of baseline (e.g., 5.0 = 500% of baseline)

# Global analysis mode selection
ANALYSIS_MODE = "comprehensive_heatmap"  # Options: "single_param", "two_param_heatmap", "comprehensive_heatmap"

class ComprehensiveGalactoseODESystem:
    """
    Complete ODE implementation of the galactose regulatory network
    """
    
    def __init__(self):
        # Complete parameter set from all reaction modules
        self.params = {
            # ========== TRANSCRIPTION PARAMETERS ==========
            'kalpha1': 0.7379,      # min^-1, R1 transcription
            'kdr_gal1': 0.02236,    # min^-1, R1 degradation
            'kalpha2': 2.542,       # min^-1, R2 transcription  
            'kdr_gal2': 0.07702,    # min^-1, R2 degradation
            'kalpha3': 0.7465,      # min^-1, R3 transcription
            'kdr_gal3': 0.02666,    # min^-1, R3 degradation
            'kir_gal4': 0.009902,   # min^-1, R4 constitutive transcription
            'kdr_gal4': 0.02476,    # min^-1, R4 degradation
            'kalpha_rep': 1.1440,   # min^-1, reporter transcription
            'kdr_rep': 0.03466,     # min^-1, reporter RNA degradation
            'kalpha80': 0.6065,     # min^-1, R80 transcription
            'kdr_gal80': 0.02888,   # min^-1, R80 degradation
            
            # ========== TRANSLATION PARAMETERS ==========
            'kip_gal1': 1.9254,     # min^-1, G1 translation
            'kdp_gal1': 0.003851,   # min^-1, G1 protein degradation
            'kip_gal2': 13.4779,    # min^-1, G2 translation
            'kdp_gal2': 0.003851,   # min^-1, G2 protein degradation
            'kip_gal3': 55.4518,    # min^-1, G3 translation
            'kdp_gal3': 0.01155,    # min^-1, G3 protein degradation
            'kip_gal4': 10.7091,    # min^-1, G4 translation
            'kdp_gal4': 0.006931,   # min^-1, G4 protein degradation
            'kip_rep': 5.7762,      # min^-1, reporter translation
            'kdp_rep_prot': 0.01155,# min^-1, reporter protein degradation
            'kip_gal80': 3.6737,    # min^-1, G80 translation
            'kdp_gal80': 0.006931,  # min^-1, G80 protein degradation
            
            # ========== DNA-PROMOTER BINDING PARAMETERS ==========
            # Single binding site genes (G3, G80)
            'Kp': 0.0248,
            'Kq': 0.1885,
            'kf1': 0.1,             # molec^-1 min^-1
            'kf2': 0.1,             # molec^-1 min^-1
            
            # 4 binding site genes (G1, reporter)  
            'Kp4': 0.2600,
            'Kq4': 1.1721,
            'kf1_4': 0.1,           # molec^-1 min^-1
            'kf2_4': 0.1,           # molec^-1 min^-1
            
            # 5 binding site genes (G2)
            'Kp5': 0.0099,
            'Kq5': 0.7408,
            'kf1_5': 0.1,           # molec^-1 min^-1
            'kf2_5': 0.1,           # molec^-1 min^-1
            
            # ========== DIMERIZATION PARAMETERS ==========
            'Kfd': 100,             # molec^-1 min^-1, dimerization
            'Krd': 0.001,           # min^-1, dimer dissociation
            
            # ========== G80 TRANSPORT PARAMETERS ==========
            'Kf80': 500,            # min^-1, cytoplasm to nucleus
            'Kr80': 500,            # min^-1, nucleus to cytoplasm
            
            # ========== G3 ACTIVATION PARAMETERS ==========
            'Kfi': 7.45e-7,         # molec^-1 min^-1, G3 + GAI -> G3i
            'Kri': 890.0,           # min^-1, G3i -> G3 + GAI
            'Kfd3i80': 0.025716,    # molec^-1 min^-1, G80Cd + G3i -> G80G3i
            'Kdr3i80': 0.0159616,   # min^-1, G80G3i -> G80Cd + G3i
            
            # ========== GALACTOSE TRANSPORT PARAMETERS (from ode_func.py) ==========
            'k_TR_GAE_to_GAI': 4350,    # min^-1, G2GAE -> G2GAI
            'k_TR_GAI_to_GAE': 4350,    # min^-1, G2GAI -> G2GAE  
            'kr_TR': 2.3925e3,          # min^-1  
            'kf_TR': 3.1353e-4,         # molec^-1 min^-1
            'kf_GK': 4.0243e-4,         # molec^-1 min^-1
            'kr_GK': 1.8425e3,          # min^-1
            'kcat_GK': 3350,            # min^-1
        }
        
        # Calculate derived parameters
        self.params['kr1'] = self.params['kf1'] / self.params['Kp']
        self.params['kr2'] = self.params['kf2'] / self.params['Kq']
        self.params['kr1_4'] = self.params['kf1_4'] / self.params['Kp4']
        self.params['kr2_4'] = self.params['kf2_4'] / self.params['Kq4']
        self.params['kr1_5'] = self.params['kf1_5'] / self.params['Kp5']
        self.params['kr2_5'] = self.params['kf2_5'] / self.params['Kq5']
        
        # Species indices for easy reference
        self.species_indices = {
            # RNA species (0-5)
            'R1': 0, 'R2': 1, 'R3': 2, 'R4': 3, 'R80': 4, 'reporter_rna': 5,
            
            # Free protein species (6-13)
            'G1': 6, 'G2': 7, 'G3': 8, 'G3i': 9, 'G4': 10, 'G4d': 11, 
            'G80': 12, 'G80C': 13, 'G80d': 14, 'G80Cd': 15, 'G80G3i': 16, 'reporter': 17,
            
            # DNA species - free forms (18-22)
            'DG1': 18, 'DG2': 19, 'DG3': 20, 'DGrep': 21, 'DG80': 22,
            
            # DNA species - G4d bound forms (23-27)
            'DG1_G4d': 23, 'DG2_G4d': 24, 'DG3_G4d': 25, 'DGrep_G4d': 26, 'DG80_G4d': 27,
            
            # DNA species - G4d+G80d bound forms (28-32)
            'DG1_G4d_G80d': 28, 'DG2_G4d_G80d': 29, 'DG3_G4d_G80d': 30, 
            'DGrep_G4d_G80d': 31, 'DG80_G4d_G80d': 32,
            
            # Galactose-bound proteins (33-36)
            'GAI': 33, 'G1GAI': 34, 'G2GAI': 35, 'G2GAE': 36
        }
        
        # Total number of species
        self.n_species = 37
    
    def get_initial_conditions(self, GAI_mM=0):
        """
        Get initial conditions based on the CME-ODE code
        """
        # Convert from mM to molecules/cell
        gai_molec = GAI_mM / (4.65e-8)
        
        # Initialize all species to zero
        y0 = np.zeros(self.n_species)
        
        # Set initial conditions based on cme_ode_sim.py
        # RNA species start at 1
        y0[self.species_indices['R1']] = 1
        y0[self.species_indices['R2']] = 1
        y0[self.species_indices['R3']] = 1
        y0[self.species_indices['R4']] = 1
        y0[self.species_indices['R80']] = 1.18715948592467
        y0[self.species_indices['reporter_rna']] = 1
        
        # Protein species initial conditions
        y0[self.species_indices['G1']] = 132.318563460887
        y0[self.species_indices['G2']] = 1156.91017704601
        y0[self.species_indices['G3']] = 4341.70321120979
        y0[self.species_indices['G3i']] = 0
        y0[self.species_indices['G4']] = 308.921734355756
        y0[self.species_indices['G4d']] = 157.246650776274
        y0[self.species_indices['G80']] = 1
        y0[self.species_indices['G80C']] = 1
        y0[self.species_indices['G80d']] = 157.239961338382
        y0[self.species_indices['G80Cd']] = 0
        y0[self.species_indices['G80G3i']] = 0
        y0[self.species_indices['reporter']] = 132.317774287091
        
        # DNA species (all genes start as free)
        y0[self.species_indices['DG1']] = 1.0
        y0[self.species_indices['DG2']] = 1.0
        y0[self.species_indices['DG3']] = 1.0
        y0[self.species_indices['DGrep']] = 1.0
        y0[self.species_indices['DG80']] = 1.0
        
        # All other DNA states start at 0
        
        # Galactose species
        y0[self.species_indices['GAI']] = gai_molec
        y0[self.species_indices['G1GAI']] = 0
        y0[self.species_indices['G2GAI']] = 0
        y0[self.species_indices['G2GAE']] = 0
        
        return y0
    
    def ode_system(self, t, y, GAE, params=None):
        """
        Complete ODE system for the galactose regulatory network
        
        State vector y contains all 37 species
        """
        if params is None:
            params = self.params
            
        # Extract species concentrations
        species = {}
        for name, idx in self.species_indices.items():
            species[name] = y[idx]
        
        # Initialize derivatives
        dydt = np.zeros(self.n_species)
        
        # ========== TRANSCRIPTION REACTIONS ==========
        # R1 transcription and degradation
        dydt[self.species_indices['R1']] = (
            params['kalpha1'] * species['DG1_G4d'] - 
            params['kdr_gal1'] * species['R1']
        )
        
        # R2 transcription and degradation
        dydt[self.species_indices['R2']] = (
            params['kalpha2'] * species['DG2_G4d'] - 
            params['kdr_gal2'] * species['R2']
        )
        
        # R3 transcription and degradation
        dydt[self.species_indices['R3']] = (
            params['kalpha3'] * 0.571429 * species['DG3_G4d'] - 
            params['kdr_gal3'] * species['R3']
        )
        
        # R4 constitutive transcription and degradation
        dydt[self.species_indices['R4']] = (
            params['kir_gal4'] - 
            params['kdr_gal4'] * species['R4']
        )
        
        # R80 transcription and degradation
        dydt[self.species_indices['R80']] = (
            params['kalpha80'] * species['DG80_G4d'] - 
            params['kdr_gal80'] * species['R80']
        )
        
        # Reporter RNA transcription and degradation
        dydt[self.species_indices['reporter_rna']] = (
            params['kalpha_rep'] * species['DGrep_G4d'] - 
            params['kdr_rep'] * species['reporter_rna']
        )
        
        # ========== TRANSLATION REACTIONS ==========
        # G1 translation and degradation
        dydt[self.species_indices['G1']] = (
            params['kip_gal1'] * species['R1'] - 
            params['kdp_gal1'] * species['G1'] - 
            params['kf_GK'] * species['G1'] * species['GAI'] + 
            params['kr_GK'] * species['G1GAI'] + 
            params['kcat_GK'] * species['G1GAI']
        )
        
        # G2 translation and degradation + transport interactions
        dydt[self.species_indices['G2']] = (
            params['kip_gal2'] * species['R2'] - 
            params['kdp_gal2'] * species['G2'] - 
            params['kf_TR'] * species['G2'] * GAE + 
            params['kr_TR'] * species['G2GAE'] - 
            params['kf_TR'] * species['G2'] * species['GAI'] + 
            params['kr_TR'] * species['G2GAI']
        )
        
        # G3 translation, degradation, and activation
        dydt[self.species_indices['G3']] = (
            params['kip_gal3'] * species['R3'] - 
            params['kdp_gal3'] * species['G3'] - 
            params['Kfi'] * species['G3'] * species['GAI'] + 
            params['Kri'] * species['G3i']
        )
        
        # G3i (activated G3)
        dydt[self.species_indices['G3i']] = (
            params['Kfi'] * species['G3'] * species['GAI'] - 
            params['Kri'] * species['G3i'] - 
            params['kdp_gal3'] * species['G3i'] - 
            params['Kfd3i80'] * species['G80Cd'] * species['G3i'] + 
            params['Kdr3i80'] * species['G80G3i']
        )
        
        # G4 translation, degradation, and dimerization
        dydt[self.species_indices['G4']] = (
            params['kip_gal4'] * species['R4'] - 
            params['kdp_gal4'] * species['G4'] - 
            2 * params['Kfd'] * species['G4']**2 + 
            2 * params['Krd'] * species['G4d']
        )
        
        # G4d (G4 dimer)
        dydt[self.species_indices['G4d']] = (
            params['Kfd'] * species['G4']**2 - 
            params['Krd'] * species['G4d'] - 
            params['kdp_gal4'] * species['G4d'] - 
            # DNA binding reactions
            params['kf1_4'] * species['DG1'] * species['G4d'] + 
            params['kr1_4'] * species['DG1_G4d'] - 
            params['kf1_5'] * species['DG2'] * species['G4d'] + 
            params['kr1_5'] * species['DG2_G4d'] - 
            params['kf1'] * species['DG3'] * species['G4d'] + 
            params['kr1'] * species['DG3_G4d'] - 
            params['kf1_4'] * species['DGrep'] * species['G4d'] + 
            params['kr1_4'] * species['DGrep_G4d'] - 
            params['kf1'] * species['DG80'] * species['G4d'] + 
            params['kr1'] * species['DG80_G4d']
        )
        
        # G80 translation, degradation, and dimerization
        dydt[self.species_indices['G80']] = (
            params['kip_gal80'] * species['R80'] - 
            params['kdp_gal80'] * species['G80'] - 
            2 * params['Kfd'] * species['G80']**2 + 
            2 * params['Krd'] * species['G80d']
        )
        
        # G80C (cytoplasmic G80)
        dydt[self.species_indices['G80C']] = (
            params['kip_gal80'] * species['R80'] - 
            params['kdp_gal80'] * species['G80C'] - 
            2 * params['Kfd'] * species['G80C']**2 + 
            2 * params['Krd'] * species['G80Cd']
        )
        
        # G80d (nuclear G80 dimer)
        dydt[self.species_indices['G80d']] = (
            params['Kfd'] * species['G80']**2 - 
            params['Krd'] * species['G80d'] - 
            params['kdp_gal80'] * species['G80d'] + 
            params['Kf80'] * species['G80Cd'] - 
            params['Kr80'] * species['G80d'] - 
            # DNA repressor binding
            params['kf2_4'] * species['DG1_G4d'] * species['G80d'] + 
            params['kr2_4'] * species['DG1_G4d_G80d'] - 
            params['kf2_5'] * species['DG2_G4d'] * species['G80d'] + 
            params['kr2_5'] * species['DG2_G4d_G80d'] - 
            params['kf2'] * species['DG3_G4d'] * species['G80d'] + 
            params['kr2'] * species['DG3_G4d_G80d'] - 
            params['kf2_4'] * species['DGrep_G4d'] * species['G80d'] + 
            params['kr2_4'] * species['DGrep_G4d_G80d'] - 
            params['kf2'] * species['DG80_G4d'] * species['G80d'] + 
            params['kr2'] * species['DG80_G4d_G80d']
        )
        
        # G80Cd (cytoplasmic G80 dimer)
        dydt[self.species_indices['G80Cd']] = (
            params['Kfd'] * species['G80C']**2 - 
            params['Krd'] * species['G80Cd'] - 
            params['kdp_gal80'] * species['G80Cd'] - 
            params['Kf80'] * species['G80Cd'] + 
            params['Kr80'] * species['G80d'] - 
            params['Kfd3i80'] * species['G80Cd'] * species['G3i'] + 
            params['Kdr3i80'] * species['G80G3i']
        )
        
        # G80G3i (G80-G3i complex)
        dydt[self.species_indices['G80G3i']] = (
            params['Kfd3i80'] * species['G80Cd'] * species['G3i'] - 
            params['Kdr3i80'] * species['G80G3i'] - 
            0.5 * params['kdp_gal3'] * species['G80G3i']
        )
        
        # Reporter protein
        dydt[self.species_indices['reporter']] = (
            params['kip_rep'] * species['reporter_rna'] - 
            params['kdp_rep_prot'] * species['reporter']
        )
        
        # ========== DNA SPECIES ==========
        # DG1 (free)
        dydt[self.species_indices['DG1']] = (
            -params['kf1_4'] * species['DG1'] * species['G4d'] + 
            params['kr1_4'] * species['DG1_G4d']
        )
        
        # DG1_G4d
        dydt[self.species_indices['DG1_G4d']] = (
            params['kf1_4'] * species['DG1'] * species['G4d'] - 
            params['kr1_4'] * species['DG1_G4d'] - 
            params['kf2_4'] * species['DG1_G4d'] * species['G80d'] + 
            params['kr2_4'] * species['DG1_G4d_G80d']
        )
        
        # DG1_G4d_G80d
        dydt[self.species_indices['DG1_G4d_G80d']] = (
            params['kf2_4'] * species['DG1_G4d'] * species['G80d'] - 
            params['kr2_4'] * species['DG1_G4d_G80d']
        )
        
        # DG2 (free)
        dydt[self.species_indices['DG2']] = (
            -params['kf1_5'] * species['DG2'] * species['G4d'] + 
            params['kr1_5'] * species['DG2_G4d']
        )
        
        # DG2_G4d
        dydt[self.species_indices['DG2_G4d']] = (
            params['kf1_5'] * species['DG2'] * species['G4d'] - 
            params['kr1_5'] * species['DG2_G4d'] - 
            params['kf2_5'] * species['DG2_G4d'] * species['G80d'] + 
            params['kr2_5'] * species['DG2_G4d_G80d']
        )
        
        # DG2_G4d_G80d
        dydt[self.species_indices['DG2_G4d_G80d']] = (
            params['kf2_5'] * species['DG2_G4d'] * species['G80d'] - 
            params['kr2_5'] * species['DG2_G4d_G80d']
        )
        
        # DG3 (free)
        dydt[self.species_indices['DG3']] = (
            -params['kf1'] * species['DG3'] * species['G4d'] + 
            params['kr1'] * species['DG3_G4d']
        )
        
        # DG3_G4d
        dydt[self.species_indices['DG3_G4d']] = (
            params['kf1'] * species['DG3'] * species['G4d'] - 
            params['kr1'] * species['DG3_G4d'] - 
            params['kf2'] * species['DG3_G4d'] * species['G80d'] + 
            params['kr2'] * species['DG3_G4d_G80d']
        )
        
        # DG3_G4d_G80d
        dydt[self.species_indices['DG3_G4d_G80d']] = (
            params['kf2'] * species['DG3_G4d'] * species['G80d'] - 
            params['kr2'] * species['DG3_G4d_G80d']
        )
        
        # DGrep (free)
        dydt[self.species_indices['DGrep']] = (
            -params['kf1_4'] * species['DGrep'] * species['G4d'] + 
            params['kr1_4'] * species['DGrep_G4d']
        )
        
        # DGrep_G4d
        dydt[self.species_indices['DGrep_G4d']] = (
            params['kf1_4'] * species['DGrep'] * species['G4d'] - 
            params['kr1_4'] * species['DGrep_G4d'] - 
            params['kf2_4'] * species['DGrep_G4d'] * species['G80d'] + 
            params['kr2_4'] * species['DGrep_G4d_G80d']
        )
        
        # DGrep_G4d_G80d
        dydt[self.species_indices['DGrep_G4d_G80d']] = (
            params['kf2_4'] * species['DGrep_G4d'] * species['G80d'] - 
            params['kr2_4'] * species['DGrep_G4d_G80d']
        )
        
        # DG80 (free)
        dydt[self.species_indices['DG80']] = (
            -params['kf1'] * species['DG80'] * species['G4d'] + 
            params['kr1'] * species['DG80_G4d']
        )
        
        # DG80_G4d
        dydt[self.species_indices['DG80_G4d']] = (
            params['kf1'] * species['DG80'] * species['G4d'] - 
            params['kr1'] * species['DG80_G4d'] - 
            params['kf2'] * species['DG80_G4d'] * species['G80d'] + 
            params['kr2'] * species['DG80_G4d_G80d']
        )
        
        # DG80_G4d_G80d
        dydt[self.species_indices['DG80_G4d_G80d']] = (
            params['kf2'] * species['DG80_G4d'] * species['G80d'] - 
            params['kr2'] * species['DG80_G4d_G80d']
        )
        
        # ========== GALACTOSE TRANSPORT REACTIONS ==========
        # GAI
        dydt[self.species_indices['GAI']] = (
            params['kr_TR'] * species['G2GAI'] - 
            params['kf_TR'] * species['GAI'] * species['G2'] + 
            params['kr_GK'] * species['G1GAI'] - 
            params['kf_GK'] * species['G1'] * species['GAI'] + 
            params['kdp_gal1'] * species['G1GAI'] + 
            params['kdp_gal2'] * species['G2GAI'] + 
            params['Kri'] * species['G3i'] - 
            params['Kfi'] * species['G3'] * species['GAI'] + 
            params['kdp_gal3'] * species['G3i']
        )
        
        # G1GAI
        dydt[self.species_indices['G1GAI']] = (
            params['kf_GK'] * species['G1'] * species['GAI'] - 
            params['kr_GK'] * species['G1GAI'] - 
            params['kcat_GK'] * species['G1GAI'] - 
            params['kdp_gal1'] * species['G1GAI']
        )
        
        # G2GAI
        dydt[self.species_indices['G2GAI']] = (
            -params['k_TR_GAI_to_GAE'] * species['G2GAI'] + 
            params['kf_TR'] * species['GAI'] * species['G2'] - 
            params['kr_TR'] * species['G2GAI'] + 
            params['k_TR_GAE_to_GAI'] * species['G2GAE'] - 
            params['kdp_gal2'] * species['G2GAI']
        )
        
        # G2GAE
        dydt[self.species_indices['G2GAE']] = (
            params['k_TR_GAI_to_GAE'] * species['G2GAI'] - 
            params['k_TR_GAE_to_GAI'] * species['G2GAE'] - 
            params['kr_TR'] * species['G2GAE'] + 
            params['kf_TR'] * GAE * species['G2'] - 
            params['kdp_gal2'] * species['G2GAE']
        )
        
        return dydt
    
    def simulate(self, GAE_mM, GAI_mM, t_max, params=None):
        """
        Simulate the complete ODE system
        """
        # Convert GAE from mM to molecules/cell
        GAE = GAE_mM / (4.65e-8)
        
        # Initial conditions
        y0 = self.get_initial_conditions(GAI_mM)
        
        # Time points
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, 1000)
        
        # Solve ODE
        sol = solve_ivp(
            lambda t, y: self.ode_system(t, y, GAE, params),
            t_span, y0, t_eval=t_eval, 
            method='LSODA', rtol=1e-8, atol=1e-10
        )
        
        return sol.t, sol.y
    
    def get_G2_at_60min(self, GAE_mM=11.1, GAI_mM=0, params=None):
        """
        Get the G2 concentration at 60 minutes
        """
        t, y = self.simulate(GAE_mM, GAI_mM, t_max=60, params=params)
        # G2 total = G2 + G2GAI + G2GAE
        g2_idx = self.species_indices['G2']
        g2gai_idx = self.species_indices['G2GAI']
        g2gae_idx = self.species_indices['G2GAE']
        
        g2_total = y[g2_idx, -1] + y[g2gai_idx, -1] + y[g2gae_idx, -1]
        return g2_total

# Parallel processing helper functions
def evaluate_single_parameter_sensitivity(param_info, ode_system, GAE_mM, GAI_mM):
    """
    Helper function for parallel parameter sensitivity evaluation
    """
    param_name, param_val = param_info
    
    # Test parameter range using global bounds
    param_range = np.array([PERTURBATION_LOWER_BOUND * param_val, PERTURBATION_UPPER_BOUND * param_val])
    g2_range = []
    
    for test_val in param_range:
        params_test = ode_system.params.copy()
        params_test[param_name] = test_val
        g2_test = ode_system.get_G2_at_60min(GAE_mM, GAI_mM, params_test)
        g2_range.append(g2_test)
    
    # Calculate sensitivity as normalized slope
    if param_val != 0 and len(set(g2_range)) > 1:
        slope = (g2_range[1] - g2_range[0]) / (param_range[1] - param_range[0])
        G2_baseline = ode_system.get_G2_at_60min(GAE_mM, GAI_mM)
        if G2_baseline != 0:
            sensitivity = slope * (param_val / G2_baseline)
            return param_name, sensitivity
        else:
            return param_name, 0
    else:
        return param_name, 0

def evaluate_parameter_sweep(param_info, ode_system, n_points, GAE_mM, GAI_mM):
    """
    Helper function for parallel parameter sweep evaluation
    """
    param_name, param_val = param_info
    
    # Create parameter range using global bounds
    param_range = np.linspace(PERTURBATION_LOWER_BOUND * param_val, PERTURBATION_UPPER_BOUND * param_val, n_points)
    g2_values = []
    
    for test_val in param_range:
        params_test = ode_system.params.copy()
        params_test[param_name] = test_val
        g2_test = ode_system.get_G2_at_60min(GAE_mM, GAI_mM, params_test)
        g2_values.append(g2_test)
    
    return param_name, {
        'param_range': param_range,
        'param_fold_change': param_range / param_val,
        'g2_values': np.array(g2_values),
        'baseline_param': param_val
    }

class ComprehensiveSensitivityAnalyzer:
    """
    Parameter sensitivity analysis for the complete system
    """
    
    def __init__(self, ode_system):
        self.ode_system = ode_system
        
    def sensitivity_coefficients(self, GAE_mM=11.1, GAI_mM=0, use_parallel=True, n_processes=None):
        """
        Calculate sensitivity using parameter range defined by global bounds
        Returns the slope of G2 response over this parameter range
        """
        if n_processes is None:
            n_processes = min(cpu_count(), len(self.ode_system.params))
        
        print(f"Computing sensitivity coefficients using {n_processes} CPU cores...")
        start_time = time.time()
        
        if use_parallel and len(self.ode_system.params) > 1:
            # Parallel processing
            param_items = list(self.ode_system.params.items())
            
            # Create partial function with fixed arguments
            eval_func = partial(evaluate_single_parameter_sensitivity, 
                              ode_system=self.ode_system, GAE_mM=GAE_mM, GAI_mM=GAI_mM)
            
            # Use multiprocessing pool
            with Pool(processes=n_processes) as pool:
                results = pool.map(eval_func, param_items)
            
            # Convert results to dictionary
            sensitivities = dict(results)
            
        else:
            # Sequential processing (fallback)
            sensitivities = {}
            for param_name, param_val in self.ode_system.params.items():
                _, sensitivity = evaluate_single_parameter_sensitivity(
                    (param_name, param_val), self.ode_system, GAE_mM, GAI_mM)
                sensitivities[param_name] = sensitivity
        
        elapsed_time = time.time() - start_time
        print(f"Sensitivity analysis completed in {elapsed_time:.2f} seconds")
        
        return sensitivities
    
    def detailed_parameter_sweeps(self, top_params, n_points=20, GAE_mM=11.1, GAI_mM=0, use_parallel=True, n_processes=None):
        """
        Generate detailed parameter sweeps for the most sensitive parameters
        """
        if n_processes is None:
            n_processes = min(cpu_count(), len(top_params))
        
        print(f"Computing detailed parameter sweeps using {n_processes} CPU cores...")
        start_time = time.time()
        
        if use_parallel and len(top_params) > 1:
            # Parallel processing
            param_items = [(param_name, self.ode_system.params[param_name]) for param_name in top_params]
            
            # Create partial function with fixed arguments
            eval_func = partial(evaluate_parameter_sweep, 
                              ode_system=self.ode_system, n_points=n_points, 
                              GAE_mM=GAE_mM, GAI_mM=GAI_mM)
            
            # Use multiprocessing pool
            with Pool(processes=n_processes) as pool:
                results = pool.map(eval_func, param_items)
            
            # Convert results to dictionary
            sweep_results = dict(results)
            
        else:
            # Sequential processing (fallback)
            sweep_results = {}
            for param_name in top_params:
                param_val = self.ode_system.params[param_name]
                _, result = evaluate_parameter_sweep(
                    (param_name, param_val), self.ode_system, n_points, GAE_mM, GAI_mM)
                sweep_results[param_name] = result
        
        elapsed_time = time.time() - start_time
        print(f"Detailed parameter sweeps completed in {elapsed_time:.2f} seconds")
        
        return sweep_results
    
    def individual_parameter_plots(self, n_points=20, GAE_mM=11.1, GAI_mM=0, save_plots=True, use_parallel=True, n_processes=None):
        """
        Create individual plots for each parameter showing G2 response vs parameter value
        """
        if n_processes is None:
            n_processes = min(cpu_count(), len(self.ode_system.params))
        
        # Get baseline G2 value
        baseline_g2 = self.ode_system.get_G2_at_60min(GAE_mM, GAI_mM)
        
        # Create output directory for individual plots
        import os
        if save_plots:
            os.makedirs('individual_parameter_plots', exist_ok=True)
        
        print(f"Computing individual parameter data using {n_processes} CPU cores...")
        start_time = time.time()
        
        if use_parallel and len(self.ode_system.params) > 1:
            # Parallel processing for data generation
            param_items = list(self.ode_system.params.items())
            
            # Create partial function with fixed arguments
            eval_func = partial(evaluate_parameter_sweep, 
                              ode_system=self.ode_system, n_points=n_points, 
                              GAE_mM=GAE_mM, GAI_mM=GAI_mM)
            
            # Use multiprocessing pool
            with Pool(processes=n_processes) as pool:
                results = pool.map(eval_func, param_items)
            
            # Convert results to dictionary and add baseline info
            all_results = {}
            for param_name, result in results:
                all_results[param_name] = result
                all_results[param_name]['baseline_g2'] = baseline_g2
            
        else:
            # Sequential processing (fallback)
            all_results = {}
            for param_name, param_val in self.ode_system.params.items():
                _, result = evaluate_parameter_sweep(
                    (param_name, param_val), self.ode_system, n_points, GAE_mM, GAI_mM)
                all_results[param_name] = result
                all_results[param_name]['baseline_g2'] = baseline_g2
        
        elapsed_time = time.time() - start_time
        print(f"Parameter data computation completed in {elapsed_time:.2f} seconds")
        
        # Sequential plotting (plotting itself cannot be easily parallelized)
        if save_plots:
            print("Generating individual plots...")
            plot_start = time.time()
            
            for param_name, result in all_results.items():
                param_val = result['baseline_param']
                param_range = result['param_range']
                g2_values = result['g2_values']
                
                # Create individual plot
                plt.figure(figsize=(10, 8))
                
                # Plot G2 response
                plt.plot(param_range / param_val, g2_values, 'b-o', linewidth=2, markersize=6, label='G2 Total')
                
                # Add baseline reference lines
                plt.axhline(y=baseline_g2, color='red', linestyle='--', alpha=0.7, 
                           label=f'Baseline G2 ({baseline_g2:.1f})')
                plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, 
                           label='Baseline Parameter')
                
                # Calculate percent change from baseline
                percent_change = ((g2_values - baseline_g2) / baseline_g2) * 100
                
                # Add secondary y-axis for percent change
                ax1 = plt.gca()
                ax2 = ax1.twinx()
                ax2.plot(param_range / param_val, percent_change, 'g--', alpha=0.7, linewidth=1)
                ax2.set_ylabel('% Change from Baseline', color='g')
                ax2.tick_params(axis='y', labelcolor='g')
                
                # Formatting
                plt.xlabel(f'{param_name} (fold change from baseline)')
                ax1.set_ylabel('G2 Total at 60 min (molecules/cell)')
                plt.title(f'G2 Sensitivity to {param_name}\nBaseline: {param_val:.2e}')
                plt.grid(True, alpha=0.3)
                ax1.legend(loc='upper left')
                
                # Add text box with key info
                min_g2 = np.min(g2_values)
                max_g2 = np.max(g2_values)
                range_g2 = max_g2 - min_g2
                textstr = f'G2 Range: {min_g2:.1f} - {max_g2:.1f}\nΔG2: {range_g2:.1f} ({range_g2/baseline_g2*100:.1f}%)'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
                
                plt.tight_layout()
                plt.savefig(f'individual_parameter_plots/{param_name}_sensitivity.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
            
            plot_elapsed = time.time() - plot_start
            print(f"Individual plots generated in {plot_elapsed:.2f} seconds")
        
        return all_results
    
    def parameter_sweep(self, param_name, param_range, GAE_mM=11.1, GAI_mM=0):
        """
        Sweep a single parameter and measure G2 response at 60 minutes
        """
        g2_values = []
        
        for param_val in param_range:
            # Create modified parameter set
            params = self.ode_system.params.copy()
            params[param_name] = param_val
            
            # Get G2 at 60 minutes
            g2_60min = self.ode_system.get_G2_at_60min(GAE_mM, GAI_mM, params)
            g2_values.append(g2_60min)
            
        return np.array(g2_values)

def set_perturbation_bounds(lower_bound, upper_bound):
    """
    Set global perturbation bounds for sensitivity analysis.
    
    Parameters:
    -----------
    lower_bound : float
        Lower bound as fraction of baseline parameter value (e.g., 0.1 = 10% of baseline)
    upper_bound : float
        Upper bound as multiplier of baseline parameter value (e.g., 10.0 = 1000% of baseline)
    
    Examples:
    ---------
    set_perturbation_bounds(0.1, 10.0)  # Test 10% to 1000% of baseline values
    set_perturbation_bounds(0.8, 1.2)   # Test 80% to 120% of baseline values (narrow range)
    """
    global PERTURBATION_LOWER_BOUND, PERTURBATION_UPPER_BOUND
    
    if lower_bound <= 0:
        raise ValueError("Lower bound must be positive")
    if upper_bound <= lower_bound:
        raise ValueError("Upper bound must be greater than lower bound")
    
    PERTURBATION_LOWER_BOUND = lower_bound
    PERTURBATION_UPPER_BOUND = upper_bound
    
    print(f"Perturbation bounds set to {lower_bound}x - {upper_bound}x of baseline values")

def get_perturbation_bounds():
    """
    Get current global perturbation bounds.
    
    Returns:
    --------
    tuple : (lower_bound, upper_bound)
    """
    return (PERTURBATION_LOWER_BOUND, PERTURBATION_UPPER_BOUND)

def set_analysis_mode(mode):
    """
    Set global analysis mode.
    
    Parameters:
    -----------
    mode : str
        Analysis mode: "single_param", "two_param_heatmap", or "comprehensive_heatmap"
    """
    global ANALYSIS_MODE
    
    valid_modes = ["single_param", "two_param_heatmap", "comprehensive_heatmap"]
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}")
    
    ANALYSIS_MODE = mode
    print(f"Analysis mode set to: {mode}")

def get_analysis_mode():
    """
    Get current analysis mode.
    
    Returns:
    --------
    str : Current analysis mode
    """
    return ANALYSIS_MODE

def evaluate_two_parameter_heatmap(param_info, ode_system, n_points, GAE_mM, GAI_mM):
    """
    Helper function for parallel two-parameter heatmap evaluation
    """
    (param1_name, param1_val), (param2_name, param2_val) = param_info
    
    # Create parameter ranges
    param1_range = np.linspace(PERTURBATION_LOWER_BOUND * param1_val, PERTURBATION_UPPER_BOUND * param1_val, n_points)
    param2_range = np.linspace(PERTURBATION_LOWER_BOUND * param2_val, PERTURBATION_UPPER_BOUND * param2_val, n_points)
    
    g2_matrix = np.zeros((n_points, n_points))
    
    for i, p1_val in enumerate(param1_range):
        for j, p2_val in enumerate(param2_range):
            params_test = ode_system.params.copy()
            params_test[param1_name] = p1_val
            params_test[param2_name] = p2_val
            g2_test = ode_system.get_G2_at_60min(GAE_mM, GAI_mM, params_test)
            g2_matrix[i, j] = g2_test
    
    return {
        'param1_name': param1_name,
        'param2_name': param2_name,
        'param1_range': param1_range,
        'param2_range': param2_range,
        'param1_fold_range': param1_range / param1_val,
        'param2_fold_range': param2_range / param2_val,
        'g2_matrix': g2_matrix,
        'baseline_param1': param1_val,
        'baseline_param2': param2_val
    }

class TwoParameterAnalyzer:
    """
    Two-parameter heatmap analysis for the galactose system
    """
    
    def __init__(self, ode_system):
        self.ode_system = ode_system
    
    def two_parameter_heatmap(self, param1_name, param2_name, n_points=20, GAE_mM=11.1, GAI_mM=0, save_plot=True):
        """
        Generate a heatmap showing G2 response to simultaneous perturbation of two parameters
        
        Parameters:
        -----------
        param1_name : str
            Name of first parameter to vary
        param2_name : str
            Name of second parameter to vary
        n_points : int
            Number of points along each parameter axis (total evaluations = n_points^2)
        GAE_mM : float
            External galactose concentration
        GAI_mM : float
            Initial internal galactose concentration
        save_plot : bool
            Whether to save the heatmap plot
            
        Returns:
        --------
        dict : Dictionary containing parameter ranges and G2 response matrix
        """
        
        print(f"Computing two-parameter heatmap: {param1_name} vs {param2_name}")
        print(f"Grid size: {n_points}x{n_points} = {n_points**2} evaluations")
        
        param1_val = self.ode_system.params[param1_name]
        param2_val = self.ode_system.params[param2_name]
        
        # Generate parameter ranges
        param1_range = np.linspace(PERTURBATION_LOWER_BOUND * param1_val, PERTURBATION_UPPER_BOUND * param1_val, n_points)
        param2_range = np.linspace(PERTURBATION_LOWER_BOUND * param2_val, PERTURBATION_UPPER_BOUND * param2_val, n_points)
        
        # Compute G2 response matrix
        start_time = time.time()
        g2_matrix = np.zeros((n_points, n_points))
        
        for i, p1_val in enumerate(param1_range):
            for j, p2_val in enumerate(param2_range):
                params_test = self.ode_system.params.copy()
                params_test[param1_name] = p1_val
                params_test[param2_name] = p2_val
                g2_test = self.ode_system.get_G2_at_60min(GAE_mM, GAI_mM, params_test)
                g2_matrix[i, j] = g2_test
        
        elapsed_time = time.time() - start_time
        print(f"Heatmap computation completed in {elapsed_time:.2f} seconds")
        
        # Get baseline G2 value
        baseline_g2 = self.ode_system.get_G2_at_60min(GAE_mM, GAI_mM)
        
        # Create fold-change matrices for plotting
        param1_fold_range = param1_range / param1_val
        param2_fold_range = param2_range / param2_val
        
        if save_plot:
            # Create heatmap plot
            plt.figure(figsize=(12, 10))
            
            # Main heatmap
            plt.subplot(2, 2, (1, 2))
            im = plt.imshow(g2_matrix, aspect='auto', origin='lower', 
                           extent=[param2_fold_range[0], param2_fold_range[-1], 
                                  param1_fold_range[0], param1_fold_range[-1]],
                           cmap='viridis')
            
            # Add baseline crosshairs
            plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
            plt.axvline(x=1, color='red', linestyle='--', alpha=0.7, linewidth=2)
            
            plt.xlabel(f'{param2_name} (fold change)')
            plt.ylabel(f'{param1_name} (fold change)')
            plt.title(f'G2 Total at 60 min\n{param1_name} vs {param2_name}')
            plt.colorbar(im, label='G2 molecules/cell')
            plt.legend()
            
            # Cross-section along param1 (param2 at baseline)
            plt.subplot(2, 2, 3)
            baseline_idx2 = np.argmin(np.abs(param2_fold_range - 1.0))
            plt.plot(param1_fold_range, g2_matrix[:, baseline_idx2], 'b-o', markersize=4)
            plt.axhline(y=baseline_g2, color='red', linestyle='--', alpha=0.7, label=f'Baseline G2 ({baseline_g2:.1f})')
            plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)
            plt.xlabel(f'{param1_name} (fold change)')
            plt.ylabel('G2 Total')
            plt.title(f'{param1_name} sensitivity\n({param2_name} at baseline)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Cross-section along param2 (param1 at baseline)
            plt.subplot(2, 2, 4)
            baseline_idx1 = np.argmin(np.abs(param1_fold_range - 1.0))
            plt.plot(param2_fold_range, g2_matrix[baseline_idx1, :], 'g-o', markersize=4)
            plt.axhline(y=baseline_g2, color='red', linestyle='--', alpha=0.7, label=f'Baseline G2 ({baseline_g2:.1f})')
            plt.axvline(x=1, color='red', linestyle='--', alpha=0.7)
            plt.xlabel(f'{param2_name} (fold change)')
            plt.ylabel('G2 Total')
            plt.title(f'{param2_name} sensitivity\n({param1_name} at baseline)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            
            # Save plot
            filename = f'two_param_heatmap_{param1_name}_vs_{param2_name}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Heatmap saved as: {filename}")
        
        # Return results
        return {
            'param1_name': param1_name,
            'param2_name': param2_name,
            'param1_range': param1_range,
            'param2_range': param2_range,
            'param1_fold_range': param1_fold_range,
            'param2_fold_range': param2_fold_range,
            'g2_matrix': g2_matrix,
            'baseline_param1': param1_val,
            'baseline_param2': param2_val,
            'baseline_g2': baseline_g2
        }
    
    def compare_parameter_pairs(self, param_pairs, n_points=15, GAE_mM=11.1, GAI_mM=0):
        """
        Compare multiple parameter pairs in a grid of heatmaps
        
        Parameters:
        -----------
        param_pairs : list of tuples
            List of (param1_name, param2_name) pairs to analyze
        n_points : int
            Number of points along each parameter axis
        GAE_mM : float
            External galactose concentration
        GAI_mM : float
            Initial internal galactose concentration
        """
        n_pairs = len(param_pairs)
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_pairs == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        results = {}
        
        for i, (param1_name, param2_name) in enumerate(param_pairs):
            print(f"Processing pair {i+1}/{n_pairs}: {param1_name} vs {param2_name}")
            
            # Generate heatmap data
            result = self.two_parameter_heatmap(param1_name, param2_name, 
                                              n_points=n_points, GAE_mM=GAE_mM, 
                                              GAI_mM=GAI_mM, save_plot=False)
            results[(param1_name, param2_name)] = result
            
            # Plot in subplot
            ax = axes[i]
            im = ax.imshow(result['g2_matrix'], aspect='auto', origin='lower',
                          extent=[result['param2_fold_range'][0], result['param2_fold_range'][-1],
                                 result['param1_fold_range'][0], result['param1_fold_range'][-1]],
                          cmap='viridis')
            
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=1)
            ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            ax.set_xlabel(f'{param2_name} (fold)')
            ax.set_ylabel(f'{param1_name} (fold)')
            ax.set_title(f'{param1_name[:8]} vs {param2_name[:8]}')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Hide unused subplots
        for i in range(n_pairs, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('parameter_pairs_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return results

    def comprehensive_heatmap_analysis(self, n_points=10, GAE_mM=11.1, GAI_mM=0, 
                                     use_parallel=True, n_processes=None, max_pairs=None,
                                     save_individual_plots=True):
        """
        Generate heatmaps for ALL possible parameter combinations
        
        Parameters:
        -----------
        n_points : int
            Number of points along each parameter axis (default: 10 for speed)
        GAE_mM : float
            External galactose concentration
        GAI_mM : float
            Initial internal galactose concentration
        use_parallel : bool
            Whether to use parallel processing
        n_processes : int
            Number of processes to use (None = auto-detect)
        max_pairs : int
            Maximum number of parameter pairs to analyze (None = all pairs)
        save_individual_plots : bool
            Whether to save individual heatmap plots for each pair
        
        Returns:
        --------
        dict : Dictionary containing results for all parameter pairs
        """
        from itertools import combinations
        import os
        
        # Get all parameter names
        param_names = list(self.ode_system.params.keys())
        n_params = len(param_names)
        
        # Generate all possible parameter pairs
        all_pairs = list(combinations(param_names, 2))
        n_total_pairs = len(all_pairs)
        
        # Limit pairs if requested
        if max_pairs is not None and max_pairs < n_total_pairs:
            print(f"Limiting analysis to first {max_pairs} parameter pairs out of {n_total_pairs} total")
            all_pairs = all_pairs[:max_pairs]
        
        print(f"=== COMPREHENSIVE TWO-PARAMETER HEATMAP ANALYSIS ===")
        print(f"Total parameters: {n_params}")
        print(f"Parameter pairs to analyze: {len(all_pairs)}")
        print(f"Total simulations: {len(all_pairs) * n_points**2:,}")
        print(f"Grid size per heatmap: {n_points}x{n_points}")
        
        if use_parallel:
            if n_processes is None:
                n_processes = min(cpu_count(), max(1, cpu_count() // 2))  # Use half cores by default
            print(f"Using parallel processing with {n_processes} cores")
        
        # Create output directory for individual plots
        if save_individual_plots:
            output_dir = 'comprehensive_heatmaps'
            os.makedirs(output_dir, exist_ok=True)
            print(f"Individual heatmaps will be saved to: {output_dir}/")
        
        # Initialize results dictionary
        results = {}
        start_time = time.time()
        
        if use_parallel and len(all_pairs) > 1:
            # Parallel processing
            print("Starting parallel heatmap generation...")
            
            # Create parameter info tuples
            param_info_list = []
            for param1_name, param2_name in all_pairs:
                param1_val = self.ode_system.params[param1_name]
                param2_val = self.ode_system.params[param2_name]
                param_info_list.append(((param1_name, param1_val), (param2_name, param2_val)))
            
            # Create partial function with fixed arguments
            eval_func = partial(evaluate_two_parameter_heatmap,
                              ode_system=self.ode_system, n_points=n_points,
                              GAE_mM=GAE_mM, GAI_mM=GAI_mM)
            
            # Use multiprocessing pool
            with Pool(processes=n_processes) as pool:
                parallel_results = pool.map(eval_func, param_info_list)
            
            # Convert results to dictionary
            for result in parallel_results:
                pair_key = (result['param1_name'], result['param2_name'])
                results[pair_key] = result
        
        else:
            # Sequential processing
            print("Starting sequential heatmap generation...")
            for i, (param1_name, param2_name) in enumerate(all_pairs):
                print(f"Processing pair {i+1}/{len(all_pairs)}: {param1_name} vs {param2_name}")
                
                result = self.two_parameter_heatmap(param1_name, param2_name,
                                                  n_points=n_points, GAE_mM=GAE_mM,
                                                  GAI_mM=GAI_mM, save_plot=False)
                results[(param1_name, param2_name)] = result
        
        elapsed_time = time.time() - start_time
        print(f"Heatmap generation completed in {elapsed_time:.2f} seconds")
        
        # Save individual plots if requested
        if save_individual_plots:
            print("Saving individual heatmap plots...")
            plot_start_time = time.time()
            
            for i, ((param1_name, param2_name), result) in enumerate(results.items()):
                # Create individual heatmap plot
                plt.figure(figsize=(10, 8))
                
                im = plt.imshow(result['g2_matrix'], aspect='auto', origin='lower',
                              extent=[result['param2_fold_range'][0], result['param2_fold_range'][-1],
                                     result['param1_fold_range'][0], result['param1_fold_range'][-1]],
                              cmap='viridis')
                
                # Add baseline crosshairs
                plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Baseline')
                plt.axvline(x=1, color='red', linestyle='--', alpha=0.7, linewidth=2)
                
                plt.xlabel(f'{param2_name} (fold change)')
                plt.ylabel(f'{param1_name} (fold change)')
                plt.title(f'G2 Total at 60 min: {param1_name} vs {param2_name}')
                plt.colorbar(im, label='G2 molecules/cell')
                plt.legend()
                
                # Calculate some statistics
                g2_min = np.min(result['g2_matrix'])
                g2_max = np.max(result['g2_matrix'])
                g2_range = g2_max - g2_min
                baseline_g2 = result.get('baseline_g2', self.ode_system.get_G2_at_60min(GAE_mM, GAI_mM))
                
                # Add text box with statistics
                textstr = f'G2 Range: {g2_min:.0f} - {g2_max:.0f}\n'
                textstr += f'ΔG2: {g2_range:.0f} ({g2_range/baseline_g2*100:.1f}%)\n'
                textstr += f'Baseline: {baseline_g2:.0f}'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
                        verticalalignment='top', bbox=props)
                
                plt.tight_layout()
                
                # Save plot
                safe_name1 = param1_name.replace('/', '_').replace(':', '_')
                safe_name2 = param2_name.replace('/', '_').replace(':', '_')
                filename = f'{output_dir}/heatmap_{i+1:03d}_{safe_name1}_vs_{safe_name2}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
            
            plot_elapsed = time.time() - plot_start_time
            print(f"Individual plots saved in {plot_elapsed:.2f} seconds")
        
        # Generate summary statistics
        print("\n=== COMPREHENSIVE HEATMAP ANALYSIS SUMMARY ===")
        baseline_g2 = self.ode_system.get_G2_at_60min(GAE_mM, GAI_mM)
        
        # Calculate effect sizes for each parameter pair
        pair_effects = []
        for (param1_name, param2_name), result in results.items():
            g2_min = np.min(result['g2_matrix'])
            g2_max = np.max(result['g2_matrix'])
            g2_range = g2_max - g2_min
            percent_effect = (g2_range / baseline_g2) * 100
            
            pair_effects.append({
                'param1': param1_name,
                'param2': param2_name,
                'g2_min': g2_min,
                'g2_max': g2_max,
                'g2_range': g2_range,
                'percent_effect': percent_effect
            })
        
        # Sort by effect size
        pair_effects.sort(key=lambda x: x['percent_effect'], reverse=True)
        
        print(f"Baseline G2: {baseline_g2:.1f} molecules/cell")
        print(f"\nTop 20 most influential parameter pairs:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Parameter 1':<15} {'Parameter 2':<15} {'G2 Range':<10} {'% Effect':<8}")
        print("-" * 80)
        
        for i, effect in enumerate(pair_effects[:20]):
            print(f"{i+1:<4} {effect['param1']:<15} {effect['param2']:<15} "
                  f"{effect['g2_range']:<10.0f} {effect['percent_effect']:<8.1f}")
        
        # Save comprehensive results to CSV
        results_df = pd.DataFrame(pair_effects)
        results_df.to_csv('comprehensive_two_parameter_analysis.csv', index=False)
        print(f"\nDetailed results saved to 'comprehensive_two_parameter_analysis.csv'")
        
        # Save raw heatmap data (optional - can be large)
        print(f"Raw heatmap data contains {len(results)} parameter pairs with {n_points}x{n_points} matrices each")
        
        return results

def run_two_parameter_example():
    """
    Example function showing how to use the two-parameter heatmap analysis
    """
    print("=== Two-Parameter Heatmap Example ===")
    
    # Initialize system
    ode_sys = ComprehensiveGalactoseODESystem()
    analyzer = TwoParameterAnalyzer(ode_sys)
    
    # Example 1: Analyze G2 transcription vs translation
    print("Example 1: G2 transcription vs translation rates")
    result1 = analyzer.two_parameter_heatmap('kalpha2', 'kip_gal2', n_points=15)
    
    # Example 2: Analyze dimerization rates
    print("\nExample 2: Dimerization forward vs reverse rates")
    result2 = analyzer.two_parameter_heatmap('Kfd', 'Krd', n_points=15)
    
    # Example 3: Compare multiple pairs
    print("\nExample 3: Comparing multiple parameter pairs")
    param_pairs = [
        ('kalpha2', 'kip_gal2'),   # G2 synthesis
        ('Kfd', 'Krd'),            # Dimerization
        ('Kf80', 'Kr80'),          # G80 transport
        ('kalpha1', 'kip_gal1')    # G1 synthesis
    ]
    
    comparison_results = analyzer.compare_parameter_pairs(param_pairs, n_points=12)
    
    print("Two-parameter analysis complete!")
    return result1, result2, comparison_results

def run_comprehensive_heatmap_analysis():
    """
    Run comprehensive two-parameter heatmap analysis for ALL parameter combinations
    """
    print("=== COMPREHENSIVE TWO-PARAMETER HEATMAP ANALYSIS ===")
    
    # Initialize system
    ode_sys = ComprehensiveGalactoseODESystem()
    analyzer = TwoParameterAnalyzer(ode_sys)
    
    # Calculate total number of parameter pairs
    n_params = len(ode_sys.params)
    total_pairs = n_params * (n_params - 1) // 2
    
    print(f"System has {n_params} parameters")
    print(f"Total possible parameter pairs: {total_pairs}")
    
    # Ask user for confirmation for large analyses
    if total_pairs > 100:
        print(f"\nWARNING: This will generate {total_pairs} heatmaps!")
        print("This may take considerable time and disk space.")
        print("Consider using max_pairs parameter to limit the analysis.")
        
        # For demonstration, limit to first 50 pairs by default
        max_pairs = 50
        print(f"Limiting analysis to first {max_pairs} parameter pairs for demonstration...")
    else:
        max_pairs = None
    
    # Run comprehensive analysis
    # Use smaller grid (10x10) for speed, increase n_points for higher resolution
    results = analyzer.comprehensive_heatmap_analysis(
        n_points=10,           # Grid resolution (10x10 = 100 simulations per pair)
        GAE_mM=11.1,          # Standard galactose concentration
        GAI_mM=0,             # No initial internal galactose
        use_parallel=True,     # Use parallel processing
        n_processes=None,      # Auto-detect CPU cores
        max_pairs=max_pairs,   # Limit number of pairs
        save_individual_plots=True  # Save individual heatmap plots
    )
    
    print(f"\nAnalysis complete! Generated {len(results)} heatmaps.")
    print("Check 'comprehensive_heatmaps/' directory for individual plots.")
    print("Check 'comprehensive_two_parameter_analysis.csv' for summary results.")
    
    return results

def main():
    """
    Main analysis function for comprehensive sensitivity analysis
    """
    # Example: Uncomment to change analysis mode
    # set_analysis_mode("two_param_heatmap")  # Switch to two-parameter heatmap analysis
    # set_perturbation_bounds(0.1, 10.0)     # Optional: change parameter bounds
    
    print("=== Comprehensive Galactose Switch ODE Analysis ===")
    print("Complete system with all reactions (transcription, translation, regulation, transport)")
    print(f"Total species: 37")
    print(f"Total parameters: {len(ComprehensiveGalactoseODESystem().params)}")
    print(f"Available CPU cores: {cpu_count()}")
    print(f"Current analysis mode: {ANALYSIS_MODE}")
    print(f"Perturbation bounds: {PERTURBATION_LOWER_BOUND}x to {PERTURBATION_UPPER_BOUND}x of baseline values")
    print("Using parallel processing for parameter sensitivity analysis")
    
    # Initialize system
    ode_sys = ComprehensiveGalactoseODESystem()
    
    # Choose analysis based on mode
    if ANALYSIS_MODE == "comprehensive_heatmap":
        # Run comprehensive two-parameter heatmap analysis for ALL parameter pairs
        print("\n=== COMPREHENSIVE TWO-PARAMETER HEATMAP ANALYSIS ===")
        results = run_comprehensive_heatmap_analysis()
        return results  # Exit here for comprehensive heatmap mode
        
    elif ANALYSIS_MODE == "two_param_heatmap":
        # Run two-parameter heatmap analysis
        print("\n=== TWO-PARAMETER HEATMAP ANALYSIS ===")
        analyzer = TwoParameterAnalyzer(ode_sys)
        
        # Example: Single heatmap for two most sensitive parameters
        # First, do a quick sensitivity analysis to find top parameters
        sensitivity_analyzer = ComprehensiveSensitivityAnalyzer(ode_sys)
        print("Finding most sensitive parameters...")
        sensitivities = sensitivity_analyzer.sensitivity_coefficients()
        sorted_sens = sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Get top 2 parameters
        top_params = [x[0] for x in sorted_sens[:2]]
        print(f"Top 2 sensitive parameters: {top_params}")
        
        # Generate heatmap for top 2 parameters
        result = analyzer.two_parameter_heatmap(top_params[0], top_params[1], n_points=20)
        
        # Example: Compare multiple parameter pairs
        print("\nComparing multiple parameter pairs...")
        param_pairs = [
            (top_params[0], top_params[1]),
            ('kalpha2', 'kip_gal2'),  # G2 transcription vs translation
            ('Kfd', 'Krd'),           # Dimerization forward vs reverse
            ('Kf80', 'Kr80')          # G80 transport rates
        ]
        
        # Filter pairs to ensure all parameters exist
        valid_pairs = []
        for p1, p2 in param_pairs:
            if p1 in ode_sys.params and p2 in ode_sys.params:
                valid_pairs.append((p1, p2))
        
        if len(valid_pairs) > 1:
            comparison_results = analyzer.compare_parameter_pairs(valid_pairs, n_points=15)
            print(f"Generated comparison heatmaps for {len(valid_pairs)} parameter pairs")
        
        return  # Exit here for two-parameter mode
    
    # Default: single-parameter analysis
    print("\n=== SINGLE-PARAMETER SENSITIVITY ANALYSIS ===")
    analyzer = ComprehensiveSensitivityAnalyzer(ode_sys)
    
    # 1. Baseline simulation
    print("\n1. Running baseline simulation...")
    t, y = ode_sys.simulate(GAE_mM=11.1, GAI_mM=0, t_max=60)
    
    # Plot key species
    plt.figure(figsize=(15, 10))
    
    # Plot G2 total
    g2_idx = ode_sys.species_indices['G2']
    g2gai_idx = ode_sys.species_indices['G2GAI']
    g2gae_idx = ode_sys.species_indices['G2GAE']
    g2_total = y[g2_idx] + y[g2gai_idx] + y[g2gae_idx]
    
    plt.subplot(2, 3, 1)
    plt.plot(t, g2_total, 'r-', linewidth=2, label='G2 Total')
    plt.plot(t, y[g2_idx], 'r--', alpha=0.7, label='G2 free')
    plt.plot(t, y[g2gai_idx], 'g--', alpha=0.7, label='G2GAI')
    plt.plot(t, y[g2gae_idx], 'b--', alpha=0.7, label='G2GAE')
    plt.xlabel('Time (min)')
    plt.ylabel('Molecules/cell')
    plt.title('G2 Species')
    plt.legend()
    
    # Plot other key proteins
    plt.subplot(2, 3, 2)
    plt.plot(t, y[ode_sys.species_indices['G1']], label='G1')
    plt.plot(t, y[ode_sys.species_indices['G3']], label='G3')
    plt.plot(t, y[ode_sys.species_indices['G3i']], label='G3i')
    plt.plot(t, y[ode_sys.species_indices['G4d']], label='G4d')
    plt.xlabel('Time (min)')
    plt.ylabel('Molecules/cell')
    plt.title('Key Proteins')
    plt.legend()
    
    # Plot RNAs
    plt.subplot(2, 3, 3)
    plt.plot(t, y[ode_sys.species_indices['R1']], label='R1')
    plt.plot(t, y[ode_sys.species_indices['R2']], label='R2')
    plt.plot(t, y[ode_sys.species_indices['R3']], label='R3')
    plt.plot(t, y[ode_sys.species_indices['R4']], label='R4')
    plt.xlabel('Time (min)')
    plt.ylabel('Molecules/cell')
    plt.title('mRNAs')
    plt.legend()
    
    # Plot GAI
    plt.subplot(2, 3, 4)
    gai_idx = ode_sys.species_indices['GAI']
    plt.plot(t, y[gai_idx] * 4.65e-8, 'purple', linewidth=2)  # Convert to mM
    plt.axhline(y=11.1, color='red', linestyle='--', alpha=0.7, label='GAE (11.1 mM)')
    plt.xlabel('Time (min)')
    plt.ylabel('Concentration (mM)')
    plt.title('Internal Galactose (GAI)')
    plt.legend()
    
    # 2. Sensitivity analysis using 0.5x to 5x parameter range
    print(f"2. Computing sensitivity coefficients ({PERTURBATION_LOWER_BOUND}x to {PERTURBATION_UPPER_BOUND}x range)...")
    sensitivities = analyzer.sensitivity_coefficients()
    
    # Plot top sensitivities
    plt.subplot(2, 3, 5)
    # Sort by absolute sensitivity
    sorted_sens = sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
    top_10 = sorted_sens[:10]
    
    params = [x[0] for x in top_10]
    sens_vals = [x[1] for x in top_10]
    
    colors = ['red' if s < 0 else 'blue' for s in sens_vals]
    plt.barh(range(len(params)), sens_vals, color=colors)
    plt.yticks(range(len(params)), params)
    plt.xlabel('Sensitivity Coefficient')
    plt.title(f'Top 10 G2 Sensitivities ({PERTURBATION_LOWER_BOUND}x-{PERTURBATION_UPPER_BOUND}x)')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # 3. Dose-response analysis
    # print("3. Dose-response analysis...")
    # GAE_range = np.logspace(-2, 1, 20)  # 0.01 to 10 mM
    # g2_responses = []
    
    # for GAE_mM in GAE_range:
    #     g2_60min = ode_sys.get_G2_at_60min(GAE_mM, 0)
    #     g2_responses.append(g2_60min)
    
    # plt.subplot(2, 3, 6)
    # plt.semilogx(GAE_range, g2_responses, 'o-')
    # plt.xlabel('External Galactose (mM)')
    # plt.ylabel('G2 Total at 60 min')
    # plt.title('G2 Dose Response')
    # plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_galactose_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Detailed parameter sweeps for top sensitive parameters
    print("3. Generating detailed parameter sweeps...")
    top_5_params = [x[0] for x in sorted_sens[:5]]
    sweep_results = analyzer.detailed_parameter_sweeps(top_5_params)
    
    # Plot detailed parameter sweeps
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (param_name, results) in enumerate(sweep_results.items()):
        if i < 5:  # Plot first 5 parameters
            ax = axes[i]
            ax.plot(results['param_fold_change'], results['g2_values'], 'o-', linewidth=2)
            ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
            ax.set_xlabel(f'{param_name} (fold change)')
            ax.set_ylabel('G2 Total at 60 min')
            ax.set_title(f'{param_name}')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    # Hide unused subplot
    if len(sweep_results) < 6:
        axes[5].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('parameter_sweep_details.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Generate individual parameter plots
    print("4. Generating individual parameter plots...")
    print("   This will create ~60 individual plots in 'individual_parameter_plots/' directory")
    individual_results = analyzer.individual_parameter_plots()
    print(f"   Created {len(individual_results)} individual parameter plots")
    
    # 5. Summary report
    print("\n=== COMPREHENSIVE SENSITIVITY ANALYSIS SUMMARY ===")
    baseline_g2 = ode_sys.get_G2_at_60min(11.1, 0)
    print(f"Baseline G2 total at 60 min (GAE=11.1mM, GAI=0): {baseline_g2:.2f} molecules/cell")
    print(f"\nTop 15 Parameter Sensitivity Coefficients ({PERTURBATION_LOWER_BOUND}x to {PERTURBATION_UPPER_BOUND}x range):")
    print("-" * 60)
    
    sorted_sens = sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
    
    for i, (param, sens) in enumerate(sorted_sens[:15]):
        direction = "↑" if sens > 0 else "↓"
        print(f"{i+1:2d}. {param:20s}: {sens:8.3f} {direction}")
    
    print(f"\nMost sensitive parameter: {sorted_sens[0][0]} (|S| = {abs(sorted_sens[0][1]):.3f})")
    
    # Save detailed results
    results_df = pd.DataFrame([
        {'Parameter': param, 'Sensitivity': sens, 'Baseline_Value': ode_sys.params.get(param, 'N/A')}
        for param, sens in sensitivities.items()
    ])
    results_df['Abs_Sensitivity'] = results_df['Sensitivity'].abs()
    results_df = results_df.sort_values('Abs_Sensitivity', ascending=False)
    results_df.to_csv('comprehensive_g2_sensitivity_analysis_wide_range.csv', index=False)
    print("\nDetailed results saved to 'comprehensive_g2_sensitivity_analysis_wide_range.csv'")
    
    # Save parameter sweep data
    sweep_data = []
    for param_name, results in sweep_results.items():
        for i, (fold_change, g2_val) in enumerate(zip(results['param_fold_change'], results['g2_values'])):
            sweep_data.append({
                'Parameter': param_name,
                'Fold_Change': fold_change,
                'Parameter_Value': results['param_range'][i],
                'G2_Total_60min': g2_val
            })
    
    sweep_df = pd.DataFrame(sweep_data)
    sweep_df.to_csv('parameter_sweep_data.csv', index=False)
    print("Parameter sweep data saved to 'parameter_sweep_data.csv'")
    
    # Save individual parameter analysis summary
    individual_summary = []
    for param_name, results in individual_results.items():
        min_g2 = np.min(results['g2_values'])
        max_g2 = np.max(results['g2_values'])
        range_g2 = max_g2 - min_g2
        percent_range = (range_g2 / results['baseline_g2']) * 100
        
        individual_summary.append({
            'Parameter': param_name,
            'Baseline_Value': results['baseline_param'],
            'Baseline_G2': results['baseline_g2'],
            'Min_G2': min_g2,
            'Max_G2': max_g2,
            'G2_Range': range_g2,
            'Percent_Range': percent_range
        })
    
    individual_df = pd.DataFrame(individual_summary)
    individual_df['Abs_Percent_Range'] = individual_df['Percent_Range'].abs()
    individual_df = individual_df.sort_values('Abs_Percent_Range', ascending=False)
    individual_df.to_csv('individual_parameter_summary.csv', index=False)
    print("Individual parameter analysis summary saved to 'individual_parameter_summary.csv'")
    
    # Parameter classification
    print("\n=== PARAMETER CLASSIFICATION ===")
    transcription_params = [p for p in sensitivities.keys() if 'kalpha' in p or 'kdr_' in p or 'kir_' in p]
    translation_params = [p for p in sensitivities.keys() if 'kip_' in p or 'kdp_' in p]
    binding_params = [p for p in sensitivities.keys() if 'kf' in p or 'kr' in p or 'Kp' in p or 'Kq' in p]
    transport_params = [p for p in sensitivities.keys() if 'TR' in p or 'GK' in p or 'k_' in p]
    
    categories = {
        'Transcription': transcription_params,
        'Translation': translation_params, 
        'DNA Binding': binding_params,
        'Transport': transport_params
    }
    
    for category, param_list in categories.items():
        if param_list:
            avg_sens = np.mean([abs(sensitivities[p]) for p in param_list])
            max_sens = max([abs(sensitivities[p]) for p in param_list])
            print(f"{category:15s}: {len(param_list):2d} params, avg |S| = {avg_sens:.3f}, max |S| = {max_sens:.3f}")

if __name__ == "__main__":
    # To run comprehensive heatmap analysis, use:
    # set_analysis_mode("comprehensive_heatmap")
    main()

# Quick test function for comprehensive analysis
def test_comprehensive_heatmaps():
    """
    Test function to run comprehensive heatmap analysis with a small subset
    """
    print("=== TESTING COMPREHENSIVE HEATMAP ANALYSIS ===")
    
    # Initialize system
    ode_sys = ComprehensiveGalactoseODESystem()
    analyzer = TwoParameterAnalyzer(ode_sys)
    
    # Run with limited parameters for testing (first 10 pairs)
    results = analyzer.comprehensive_heatmap_analysis(
        n_points=8,            # Small grid for speed
        max_pairs=10,          # Limit to 10 pairs for testing
        save_individual_plots=True,
        use_parallel=True
    )
    
    print(f"Test completed! Generated {len(results)} test heatmaps.")
    return results