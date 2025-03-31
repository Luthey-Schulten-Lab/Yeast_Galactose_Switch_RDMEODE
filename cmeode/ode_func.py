"""
Author: Tianyu Wu
Original Author: David Bianchi
Original Date: 4/13/17
A function to implement the changing concentrations of each species as an ODE

Based on OS_Degradation_Reversible_MM_GalODEModel.m Matlab file

receives: the rate constants (floats) and species counts (ints) for all of the species involved
in reactions with the species of interest

returns: An array (numpy array) of the dx_dt (rates) values for the species of interest
"""

import numpy as np
"""
The rate equations to be solved by the ODE solver.

@param counts An array holding the values of all of the "ODE species"
@param current_time The current simulation time(mins)
@param rates An array holding the rates to be changed *** UNNECESSARY ***
@param GAE The number of extracellular galactose molecules
"""

def dxdt(counts, current_time, rates, GAE):
    # transport 
    k_TR = 4350 # min^-1
    kr_TR = 2.3925e3 # min^-1
    kf_TR = 3.1353e-4 # molec^-1 min^-1
    kdp_gal2 = 0.003851 # min^-1 
    # enzymatic rate constants
    kf_GK = 4.0243e-4 # molec^-1 min^-1
    kr_GK = 1.8425e3 #  min^-1
    kcat_GK = 3350 # min^-1
    kdp_gal1 = 0.003851 # min^-1
    # protein decay(dilution rate)
    kdp_gal1 = 0.003851 # min^-1
    kdp_gal2 = 0.003851 # min^-1
    
    # get the particle count values
    GAI = counts[0]
    G2GAI = counts[1]
    G2GAE = counts[2]
    G1GAI = counts[3]
    G1 = counts[4]
    G2 = counts[5]
    
    #get the next step counts
     # GAI
    dGAI_dt = kr_TR*G2GAI - kf_TR*GAI*G2 + kr_GK*G1GAI - kf_GK*G1*GAI + kdp_gal1*G1GAI + kdp_gal2*G2GAI

    # G2GAI
    dG2GAI_dt = -k_TR*G2GAI + kf_TR*GAI*G2 - kr_TR*G2GAI + k_TR*G2GAE # f[1] G2GAI

    # G2GAE
    dG2GAE_dt = k_TR*G2GAI - k_TR*G2GAE - kr_TR*G2GAE + kf_TR*GAE*G2 # f[2] G2GAE

    #print("dG2GAE: ", dG2GAE_dt, flush=True)

    # G1GAI
    dG1GAI_dt = kf_GK*G1*GAI - kr_GK*G1GAI - kcat_GK*G1GAI # f[3] G1GAI
    
    # G1
    dG1_dt = - kf_GK*G1*GAI + kr_GK*G1GAI + kcat_GK*G1GAI # f[4] G1

    # G2
    dG2_dt = -kf_TR*G2*GAE + kr_TR*G2GAE - kf_TR*G2*GAI + kr_TR*G2GAI # f[5] G2
     
    # A list of all the derivatives that were solved for
    dx_dt = [dGAI_dt, dG2GAI_dt, dG2GAE_dt, dG1GAI_dt, dG1_dt, dG2_dt]

    # Convert to a numpy array for ease of use
    dx_dt_array = np.asarray(dx_dt) 

    # Returns the solution to the differential equations
    return (dx_dt_array)