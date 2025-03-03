'''
CME-ODE Simulation of the Galactose Switch in Baker's Yeast, knock out the regulation to match the Ramsey's paper.
In short, they change the promoter for DG3 and DG80 to CYC1 promoter. Which can allow Gal3 and Gal80 to be transcribed 
and translated without the need of Gal4.
Author: Tianyu Wu, 12/23/2024

'''
'''
Part1: Package Import
'''
from pyLM import *
from pyLM.units import *

import logging
LMLogger.setLMLoggerLevel(logging.INFO)
# This method call sets the logging level for LMLogger to INFO. 
# The logging level controls the severity of messages that the logger 
# will handle. In this case, it means that the logger will handle 
# messages with a severity of INFO or higher (i.e., INFO, WARNING, 
# ERROR, CRITICAL).

import sys
import os
from contextlib import redirect_stdout
# Custom Post-processing Imports
# This library may be useful for plotting
import numpy as np
import scipy
import scipy.optimize
import copy
'''
Part2: Allows user to input system conditions
'''
import argparse
ap = argparse.ArgumentParser()
# The concentration of external galactose (mM)
ap.add_argument('-gex', '--GAE', required = True, help='The concentration of external galactose (mM)')

# The concentration of internal galactose (mM)
ap.add_argument('-gic', '--GAI', required = True, help='The concentration of internal galactose (mM)')

# The number of replicates to be run
ap.add_argument('-rep', '--replicates', required = True, help='The number of replicates to be run')

# The CME write step to hdf5 and communication timestep between CME-ODE
ap.add_argument('-delt', '--delt', required = True, help='communication timestep between CME-ODE, in seconds')

# The total simulation time
ap.add_argument('-t', '--simTime', type=int, default=750, help='Simulation time (default: 750 min)')
ap.add_argument('-tag', '--tag', type=str, default='', help='annotation tag for results.')
ap.add_argument('-q', '--q', type=float, default=1.0, help='q for mutant type enhancement')
ap.add_argument('-fmut', '--fmut', type=float, default=0.04, help='f_mut, original 0.04~0.06')
ap.add_argument('-g80m', '--g80multiple', type=float, default=8.0, help='Multiple for G80-containing species counts')
args = ap.parse_args()

import datetime

save_path = "output/" + datetime.datetime.now().strftime("%m%d%Y") + "/"
output_file = "gal_cme_ode_gae_nonreg" + str(args.tag) + '_'+ str(args.GAE) + "_gia" + str(args.GAI) + "_rep" + str(args.replicates) + "_delta" + str(args.delt)+"_time" + str(args.simTime) + ".lm"
log_file =  "log_cme_ode_gae_nonreg" + str(args.tag) +'_' + str(args.GAE) + "_gia" + str(args.GAI) + "_rep" + str(args.replicates) + "_delta" + str(args.delt) + "_time" + str(args.simTime) + ".log"

print("pid for this program is:", os.getpid())

'''=============================================================
                    Main Code starts from here
=============================================================='''

'''=============================================================
part3: CME-ODE solver creation
=============================================================='''

# import lm
from scipy.integrate import odeint 
import ode_func as ode_solver
import lm
class CMEODESolver(lm.GillespieDSolver):
    """
        Initialize the ODE hook solver

        @param self The object pointer
        @param counts Array containing the species counts of the ODE species
        @param delt The communication timestep between the hook and main LM simulation
        @param rxnsICareAbout List of reactions whose propensities are updated due to ODE changes
        @param ks List of the rate constants of the above reactions
        @param odestep The maximum stepsize given to the Adaptive Timestepping ODE solver
        @param speciesCount The instance of SpeciesCount Class used to pass species counts 
    """
    def initializeSolver(self, counts, delt, rxnsICareAbout, ks, gae, ode_step, speciesCount):

        # Save the initial conditions, for restarting the solver upon a new replicate
        self.ic = (counts,delt,rxnsICareAbout,ks,gae,ode_step,speciesCount)

        # Set the initial conditions
        self.restart()
    
    """
    Get the same initial conditions for a new replicate
    @param self The object pointer
    """
    def restart(self):
        # Set the previous time to be 0, we are starting the simulation
        self.oldtime = 0
        # deep copy of all initial conditions
        # create a new objects, independent of the original
        self.counts = copy.deepcopy(self.ic[0])             # counts of the species in the ode
        self.delt = copy.deepcopy(self.ic[1])               # communication timestep between ODE and CME
        self.rxnsICareAbout = copy.deepcopy(self.ic[2])     # reactions whose propensities are updated due to ODE changes
        self.ks = copy.deepcopy(self.ic[3])                 # rate constants of the above reactions
        self.gae = copy.deepcopy(self.ic[4])                # extracellular galactose concentration
        self.odestep = copy.deepcopy(self.ic[5])            # maximum stepsize given to the Adaptive Timestepping ODE solver
        self.species = copy.deepcopy(self.ic[6])            # instance of SpeciesCount Class used to pass species counts
    """
    hook Simulation method defined here will be called at every frame write time.
    The return value is either 0 or 1, which will indicate if we changed the state or not and need the lattice to be copied back to the GPU
    (In the case of the RDME) before continuing. If you do not return 1, your changes will not be reflected.
    @param self The object pointer
    @param time The current simulation time (min)
    """
    def hookSimulation(self, time):
        # notify if a new sim start
        if (time == 0.0):
            print("New Replicate", flush=True)
            self.restart()
            return 0
        # We are at an CME-ODE communication timestep
        else:
            # Update the pointer to the current species counts
            # at simulation time = time
            self.species.update(self)
            ## NOT NECESSARY: This argument is unused in this case
            rates = np.zeros(len(self.rxnsICareAbout))
            # ODE should be concerned with things that alter galactose concentrations only
            self.counts[0] = self.species['GAI']
            self.counts[1] = self.species['G2GAI']
            self.counts[2] = self.species['G2GAE']
            self.counts[3] = self.species['G1GAI']
            self.counts[4] = self.species['G1']
            self.counts[5] = self.species['G2']
            
            '''scipy.odeint
            ODE solver starts here
            time in unit of minutes
            '''
            if time < 100:    
                stepsize = self.odestep/10
                # ODE SOLVER LSODA
                sol = odeint(ode_solver.dxdt, self.counts, np.linspace(time,time+self.delt,int(np.ceil(self.delt/stepsize))+1), args=(rates, self.gae), hmax=stepsize)
            else:
                # ODE SOLVER LSODA
                sol = odeint(ode_solver.dxdt, self.counts, np.linspace(time,time+self.delt,int(np.ceil(self.delt/self.odestep))+1), args=(rates, self.gae), hmax=self.odestep)
            
            # Update the species counts
            self.counts = sol[-1]
            
            ############################################
            #species shared between CME and ODE should follow the counts from CME
            ############################################ 

            # Need to conserve particle number across regimes
            # ODE should not change protein counts, only the distribution of protein/inducer states
            totalG2 = self.species['G2GAI'] + self.species['G2GAE'] + self.species['G2']
            self.species['G2GAI'] = round(self.counts[1])
            self.species['G2GAE'] = round(self.counts[2])
            self.species['G2'] = round(totalG2 - self.species['G2GAI'] - self.species['G2GAE'])
            # G1
            totalG1 = self.species['G1GAI'] + self.species['G1']
            self.species['G1GAI'] = round(self.counts[3])
            self.species['G1'] = round(totalG1 - self.species['G1GAI'])
            
            # GAI
            self.species['GAI'] = round(self.counts[0])
            
            # Set the simulation time when we "stepped in" to the ODE solver to be the 
            # previous time
            self.oldtime = time
            
            return 1


# Instantiate the CME ODE solver'''
odeHookSolver = CMEODESolver()

'''=============================================================
part4: species add, simulation time, output file, log file
=============================================================='''

# Get the species involved in CME solved reactions
# and those involved in ODE solved reactions
ode_species = ['GAI','G2GAI','G2GAE','G1GAI','G1','G2']
# this part also removed the G4d and G4d_G80d for DG3 and DG80
cme_species = ['R1', 'R2', 'R3', 'R4', 'reporter_rna', 'R80', 'G1', 'G2', 'G3', 'G3i', 'G4', 'G4d', 'reporter', 'G80', 'G80C', 'G80d', 'G80Cd', 'G80G3i', 'GAI', 'DG1', 'DG1_G4d', 'DG1_G4d_G80d', 'DG2', 'DG2_G4d', 'DG2_G4d_G80d', 'DG3', 'DGrep', 'DGrep_G4d', 'DGrep_G4d_G80d', 'DG80', 'G2GAI', 'G2GAE', 'G1GAI']


# Total simulation time
simTime = int(args.simTime) # min

# communication timestep between ODE and CME (Converted to minutes)
delt = float(args.delt)/60.0 # min


''' create a CME-ODE simulation object'''

sim=CME.CMESimulation()


# create a species count object: to access the species
import species_counts
mySpecies = species_counts.SpeciesCounts(sim)

# Define the species involved in CME solved reactions
sim.defineSpecies(cme_species)

# the reactions involved in the ODE
Frxns = []

# initial rate constant
init_ks = []

# Convert the external galactose concentration from mM to Molecule/unit cell
Gae = float(args.GAE)/(4.65e-8) # units in molec

'''=============================================================
Part5: Add reactions for CME
=============================================================='''
from cme_rxns import transcription, translation, regulators_promoters, dimerization, g80region_swap, g3_rxns
print(f"q is {args.q}, f_mut is {args.fmut}")
transcription.getTranscriptionReactions_non_reg(sim, mutant_q=args.q, f_mut=args.fmut)
translation.getTranslationReactions(sim)
regulators_promoters.getDNAPromoterReactions_non_reg(sim)
dimerization.getDimerizationReactions(sim)
g80region_swap.getG80TransportReactions(sim)

################################################################
# This set of reactions contains a reaction whose propensity
# we would like to update as the G3i concentration changes
# in the CME. 
# We must pass in the containers to hold these reactions
# and their reaction rate constants
################################################################
g3_rxns.getG3Reactions(sim,Frxns,init_ks)


'''=============================================================
Part 6:Set the initial conditions for the ODE solver
@param sim The simulation object
@param cme_species The list of "CME species", added to the simulation object sequentially
@param ode_species The list of "ODE species", added to the simulation object sequentially
@param gai The concentration of intracellular galactose (mM)
=============================================================='''
def multiple_g80_species_counts(species_list, count_list, multiple):
    """
    Multiplies the counts for all species containing 'G80' in their name by the given multiple.
    
    Args:
        species_list: List of species names
        count_list: List of corresponding counts
        multiple: Factor to multiply G80-containing species counts by
        
    Returns:
        Modified count list with multiplied values for G80-containing species
    """
    # Create a copy of the count list to avoid modifying the original
    modified_counts = count_list.copy()
    
    # Iterate through the species list
    for i, species in enumerate(species_list):
        # Check if 'G80' is the start of the species name
        if species.startswith('G80'):
            # Multiply the count for this species
            modified_counts[i] = modified_counts[i] * multiple
            print(f"Multiplied count for {species}: {count_list[i]} → {modified_counts[i]} (×{multiple})")
    
    return modified_counts
def print_species_counts(species_name, original_count, rounded_count):
    """
    Prints the species name, original count, and rounded count.
    
    Args:
        species_name: Name of the species
        original_count: Original count before rounding
        rounded_count: Count after rounding
    """
    print(f"Species: {species_name}, Original count: {original_count}, Rounded count: {rounded_count}")


def setInitialCounts(sim,cme_species,gai):
        # gae can be added the as the input 
        # Convert from mM to molecules/cell
        # gae_molec = float(gae)/(4.65e-8)
        gai_molec = float(gai)/(4.65e-8)

        # The CME species counts
        cme_count_list = [0.26, 0.33, 0.9, 0.4, 0.26, 1.18, 132.318563460887, 1156.91017704601, 4341.70321120979, 0, 0.15, 308.921734355756, 132.317774287091, 0.11, 0.11, 157.246650776274, 157.239961338382, 0, 64516129.03225806, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1, 0.0, 0.0, 1.0, 1, 0.0, 0.0, 0.0]
        # cme_count_list = [0.26, 0.33, 0.9, 0.4, 0.26, 1.18, 132.318563460887, 1156.91017704601, 4341.70321120979, 0, 0.15, 308.921734355756, 132.317774287091, 0.11, 0.11, 157.246650776274, 157.239961338382, 0, 238709677.41935483, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        # Double the counts for G80-containing species
        cme_count_list = multiple_g80_species_counts(cme_species, cme_count_list, args.g80multiple)
        # The ODE species counts
        ode_count_list = [gai_molec,0,0,0,132.318563460887,1156.91017704601]

        # Add the particles for each species to the simulation (the cell) in integer form
        # for i in range(len(cme_species)):
        #     sim.addParticles(species=cme_species[i],count=int(round((cme_count_list[i]),1))) # All counts must be in integer form
        for i in range(len(cme_species)):
            original_count = cme_count_list[i]
            rounded_count = int(round(original_count))
            print_species_counts(cme_species[i], original_count, rounded_count)
            sim.addParticles(species=cme_species[i], count=rounded_count) 
        # Return the list of initial ODE species counts to be used to initialize the hook solver
        return ode_count_list
ode_counts = setInitialCounts(sim,cme_species,args.GAI)

try:
    ## Set the interval at which communication will occur
    sim.setHookInterval(delt)

    ## Set the write interval to the output file. (1 minute)
    sim.setWriteInterval(delt)

## Hooking is connected to I/O
except AttributeError:

    ## Set the write interval to the output file. (the communication timestep)
    sim.setWriteInterval(delt)


# Total simulation time
# Set the total simulation time
sim.setSimulationTime(args.simTime)
# Set the name of the output file
# my_lm_file = str(args.outputFile)
my_lm_file = save_path + output_file
# Save the initial system conditions to the output file


# check if the directory exists, if not create it
if not os.path.exists(save_path):
    os.makedirs(save_path)
# Save the initial system conditions to the output file
# check if the file exists, if exits delete it
if os.path.exists(my_lm_file):
    os.remove(my_lm_file)
'''save the file'''
sim.save(my_lm_file)


# Supposed UPPER Limit on the Adaptive timestep used by the ODE solver
odestep = 0.0001


'''=============================================================
Part 7: Run the simulation
=============================================================='''

# Log the run output
# with open(str(args.logfile), 'w') as f, redirect_stdout(f):
with open(str(save_path + log_file), 'w') as f, redirect_stdout(f):

    # Initialize the ODE solver that will communicate with the CME solver through LM
    odeHookSolver.initializeSolver(ode_counts, delt, Frxns, np.asarray(init_ks), Gae, odestep, mySpecies)
    
    # Run the solver so that it is stepped into at every CME writestep (sim.setWriteInterval)
    sim.runSolver(filename=my_lm_file,solver=odeHookSolver,replicates=int(args.replicates))

    # Close the simulation
    f.close()

# get all outputs to the output file
sys.stdout.flush()