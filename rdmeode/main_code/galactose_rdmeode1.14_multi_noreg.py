#!/usr/bin/env python
# coding: utf-8
'''build by Tianyu Wu v 1.10  : 
1. fix several species error that might cause the error in simulation 
1.1 DNA regulation use G4d, G80d for promoters

1.3 change the G3 activation to be 0 when GAI is 0
assocRt = max(0,self.g3actRc*ys1[self.odeSpIndex("GAI")])
no signifcant change in the simulation, just a guarantee
1.4 modified based on the 1.10 version of the rdmeode code, enable the multi-gpu simulation
  this version also support the single gpu simulation
1.5 modify the siteType to be Byte for multi-gpu simulation; Int for single gpu simulation
1.6 add timer to the code to record the time taken for the simulation
1.7 add timer to track the time spent in the hook function
1.10.8 from 1.10 version, a trajectory file for region specific counts
1.10.9 add tag for the output file name to distinguish different simulations
1.10.10 error fix on G4d G80d de_binding reaction
in v 1.12
1.12 change the custom ODE solver to LSODA 
1.12.2 add fixed gene location option
1.12.3 fix gene location to be in the nucleoplasm and add a check if to make sure they actually inside
1.12.4 reminder: the default location extract from the trajectory file is in the format of [z, y, x], 

1.12.5 change to non-regulation case

in v 1.13
1.13.1 change the region obstacle to 3 separate regions, add cell wall in as well. 
1.13.2 change the GAL3 transcription q factor from 0.571429 to 1
1.13.3 add f_mut as a parameter
in v1.14
1.14.1 support checkpoint start
1.14.2 support memory usage check 
1.14.3 support ribo_dummy added 
'''
# write a timer 
import time
import signal
import sys
import os
import psutil
start_time = time.time()

IF_DGX = False # parameter to for quick conversion to use on DGX
version = "1.14.3:wmtng"
f_mut = 0.06
# In[1]:
import argparse
parser = argparse.ArgumentParser(description='Accept three parameters: output index, simtime(min) and external galactose concentration(mM)')
parser.add_argument('-id', '--index',  type=int, required=True, help='index of the output lm files')
parser.add_argument('-t', '--simtime',  type=float, default=60, help='simulation time')
parser.add_argument('-g', '--galactose',  type=float, default=11.1, help='external galactose concentration')
parser.add_argument('-gpus', '--gpus',  type=int, default=1, help='available gpus to use(default 1, use single gpu)')
parser.add_argument('-tag', '--tag',  type=str, default='', help='tag for the output folder')
parser.add_argument('-geo', '--geometry',  type=str, default='yeast-lattice.2.pkl.xz', help='geometry file name, default is yeast-lattice.2.pkl.xz')
parser.add_argument('-mt', '--max_time', type=float, default=1000, help='Maximum allowed simulation time in hours')
parser.add_argument('-geloc', '--gene_location', type=str, default='random', help='location of the genes, default is random')
parser.add_argument('-ckpt', '--checkpoint', type=str, default='', help='checkpoint file name, default is empty')

# get the args
args = parser.parse_args()
output_order = args.index
simtime = args.simtime
externalGal_input = args.galactose
gpus = args.gpus
output_tag = args.tag
geometry_file = args.geometry
gene_location = args.gene_location
checkpoint_file = args.checkpoint
#get date in format yyyymmdd
import datetime
date = datetime.datetime.now().strftime("%Y%m%d")
output_dir = "simulation_results_mt_id_" + str(output_order)
if IF_DGX == True:
   
    output_folder = "workspace/yeast" + version + "_multi_" + str(date) + "_" + str(output_order) + "_t"+ str(simtime) + "min"+ "GAE" + str(externalGal_input)+ "mM" + output_tag +"_gpu"+ str(gpus) +".lm"
else:
    
    output_folder = "yeast" + version  + str(date) + "_" + str(output_order) + "_t"+ str(simtime) + "min"+ "GAE" + str(externalGal_input)+ "mM" + output_tag +"_gpu"+ str(gpus) +".lm"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_folder = os.path.join(output_dir, output_folder)

print("output_folder: ", output_folder)
print("simtime: ", simtime)
print("geometry_file: ", geometry_file)
import pickle, lzma
import numpy as np
import scipy.integrate as spint
from jLM.Solvers import ConstBoundaryConc, makeSolver
from lm import MGPUMpdRdmeSolver, MpdRdmeSolver,IntMpdRdmeSolver
from jLM.RDME import Sim as RDMESim
from jLM.RDME import File as RDMEFile
from jLM.RegionBuilder import RegionBuilder
import jLM


# # Initialization and Spatial Geometry
def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024} MB")

# In[35]:

if IF_DGX == True:
    latticeData = pickle.load(lzma.open("workspace/"+geometry_file, "rb"))
else:
    latticeData = pickle.load(lzma.open(geometry_file, "rb")) # 1.10 use the original lattice instead of .2

siteMap = {n:i for i,n in enumerate(latticeData['names'])}
def boolLattice(x):
    return latticeData['lattice'] == siteMap[x]

extracellular = boolLattice("extracellular") 
cellWall = boolLattice("cellWall")
nuclearEnvelope = boolLattice("nuclearEnvelope")
mitochondria = boolLattice("mitochondria")
vacuole = boolLattice("vacuole")
ribosomes = boolLattice("ribosomes")
membrane = boolLattice("plasmaMembrane")
nucleus = boolLattice("nucleoplasm") | boolLattice("nuclearPores")
cytoplasm = boolLattice("cytoplasm")
'''support the dummy ribosome add'''
if 'ribosome_dummy' in latticeData['names']:
    print('dummy ribosomes region detected!Treated as obstacles.')
    ribosome_dummy = boolLattice("ribosome_dummy")
    
'''read decimation'''
decimation = latticeData['decimation']


# In[36]:
if gpus == 1: 
    siteType = "Int"
else:
    siteType = "Byte"

sim = RDMESim("Galactose switch, RDME/ODE hybrid",
              output_folder,
              latticeData['lattice'].shape,
              latticeData['latticeSpacing'],
              "extracellular",siteType)

print("the shape of the lattice is: ", latticeData['lattice'].shape)
# In[37]:


B = RegionBuilder(sim)

'''support the dummy ribosome add'''
if 'ribosome_dummy' in latticeData['names']:
        B.compose((sim.region('extracellular'), extracellular),
            (sim.region('cellWall'), cellWall),
            (sim.region('nuclearEnvelope'), nuclearEnvelope),
            (sim.region('mitochondria'), mitochondria),
            (sim.region('vacuole'), vacuole),
            (sim.region('plasmaMembrane'), membrane),
            (sim.region('cytoplasm'), cytoplasm),
            (sim.region('nucleoplasm'), nucleus),
            (sim.region('ribosomes'), ribosomes),
            (sim.region('ribodummy'), ribosome_dummy)
            )
else:
    B.compose((sim.region('extracellular'), extracellular),
            (sim.region('cellWall'), cellWall),
            (sim.region('nuclearEnvelope'), nuclearEnvelope),
            (sim.region('mitochondria'), mitochondria),
            (sim.region('vacuole'), vacuole),
            (sim.region('plasmaMembrane'), membrane),
            (sim.region('cytoplasm'), cytoplasm),
            (sim.region('nucleoplasm'), nucleus),
            (sim.region('ribosomes'), ribosomes))


# abbreviation for object access

# In[38]:


sp = sim.sp   # species object access
reg = sim.reg # region object access
rc = sim.rc   # rate constant object access
dc = sim.dc   # diffusion constant object access


# # Simulation parameters

# In[39]:


sim.simulationTime = simtime * 60 # seconds
sim.timestep = 50e-6 # seconds
# mimic the old way to define the interval
hook_interval = 1.0 # seconds
write_interval = 1.0 # seconds
sim.latticeWriteInterval= int(write_interval/sim.timestep) # .05s
sim.speciesWriteInterval= int(write_interval/sim.timestep) # .05s
sim.hookInterval= int(hook_interval/sim.timestep) #.05s

# initial conditions for external galactose 
externalGal = externalGal_input * 1e-3 # M   
# Number of ribosomes in total
nRibosomes = np.sum(sim.siteLattice == reg.ribosomes.idx)
# diffusion coefficients for mRNA
# mRNADiffusion = 0.5e-12 # m^2/s
mRNADiffusion = 0.05e-12 # number from sif 



# # Species Definitions

# ## Reporter GFP

# In[40]:


with sim.construct():
    sim.species('DGrep', texRepr='D_{rep}', annotation="Reporter gene (inactive)")
    sim.species('DGrep_G4d', texRepr='D_{rep}{:}G_{4D}', annotation="Reporter gene activated")
    sim.species('DGrep_G4d_G80d', texRepr='D_{rep}{:}G_{4D}{:}G_{80D}', annotation="Reporter gene repressed")
    sim.species('Rrep', texRepr='R_{rep}', annotation="Reporter mRNA")
    sim.species('Grep', texRepr='G_{rep}', annotation="Reporter GFP")
    
    


# ## GAL1 (G1) metabolism/ degradation process

# In[41]:


with sim.construct():
    sim.species('DG1', texRepr='D_{G1}', annotation="Galactose metabolism gene (inactive)")
    sim.species('DG1_G4d', texRepr='D_{G1}{:}G_{4D}', annotation="Galactose metabolism gene activated")
    sim.species('DG1_G4d_G80d', texRepr='D_{G1}{:}G_{4D}{:}G_{80D}', annotation="Galactose metabolism gene repressed")
    sim.species('R1', texRepr='R_{1}', annotation="Galactose metabolism mRNA")
    sim.species('G1', texRepr='G_{1}', annotation="Galactose metabolism protein")


# ## GAL2(G2) transporter

# In[42]:


with sim.construct():
    sim.species('DG2', texRepr='D_{G2}', annotation="Galactose transport gene (inactive)")
    sim.species('DG2_G4d', texRepr='D_{G2}{:}G_{4D}', annotation="Galactose transport gene activated")
    sim.species('DG2_G4d_G80d', texRepr='D_{G2}{:}G_{4D}{:}G_{80D}', annotation="Galactose transport gene repressed")
    sim.species('R2', texRepr='R_{2}', annotation="Galactose transport mRNA")
    sim.species('G2', texRepr='G_{2}', annotation="Galactose transport protein")


# ## Gal3(regulatory protein)

# In[43]:


# with sim.construct():
#     sim.species('DG3', texRepr='D_{G3}', annotation="Gal3 gene (inactive)")
#     sim.species('DG3_G4d', texRepr='D_{G3}{:}G_{4D}', annotation="Gal3 gene activated")
#     sim.species('DG3_G4d_G80d', texRepr='D_{G3}{:}G_{4D}{:}G_{80D}', annotation="Gal3 gene repressed")
#     sim.species('R3', texRepr='R_{3}', annotation="Gal3 mRNA")
#     sim.species('G3', texRepr='G_{3}', annotation="Gal3 protein")
#     sim.species('G3i', texRepr='G_{3i}', annotation="activated Gal3 bound to galactose ")
# no reg case
with sim.construct():
    sim.species('DG3', texRepr='D_{G3}', annotation="Gal3 gene with CYC1 bind")
    # sim.species('DG3_G4d', texRepr='D_{G3}{:}G_{4D}', annotation="Gal3 gene activated")
    # sim.species('DG3_G4d_G80d', texRepr='D_{G3}{:}G_{4D}{:}G_{80D}', annotation="Gal3 gene repressed")
    sim.species('R3', texRepr='R_{3}', annotation="Gal3 mRNA")
    sim.species('G3', texRepr='G_{3}', annotation="Gal3 protein")
    sim.species('G3i', texRepr='G_{3i}', annotation="activated Gal3 bound to galactose ")



# ## GAL4(G4)

# In[44]:


with sim.construct():
    sim.species('DG4', texRepr='D_{G4}', annotation="Gal4 gene (inactive)")
    sim.species('R4', texRepr='R_{4}', annotation="Gal4 mRNA")
    sim.species('G4', texRepr='G_{4}', annotation="Gal4 protein")
    sim.species('G4d', texRepr='G_{4D}', annotation="Gal4 dimer")


# ## GAL80 (G80)

# In[45]:


# with sim.construct():
#     sim.species('DG80', texRepr='D_{G80}', annotation="Gal80 gene (inactive)")
#     sim.species('DG80_G4d', texRepr='D_{G80}{:}G_{4D}', annotation="Gal80 gene activated")
#     sim.species('DG80_G4d_G80d', texRepr='D_{G80}{:}G_{4D}{:}G_{80D}', annotation="Gal80 gene repressed")
#     sim.species('R80', texRepr='R_{80}', annotation="Gal80 mRNA")
#     sim.species('G80', texRepr='G_{80}', annotation="Gal80 protein")
#     sim.species('G80d', texRepr='G_{80D}', annotation="Gal80 dimer")
#     sim.species('G80d_G3i', texRepr='G_{80D}{:}G_{3i}', annotation="Gal80 dimer bound to activated Gal3")
# no reg case
with sim.construct():
    sim.species('DG80', texRepr='D_{G80}', annotation="Gal80 gene (with CYC1 bind)")
    # sim.species('DG80_G4d', texRepr='D_{G80}{:}G_{4D}', annotation="Gal80 gene activated")
    # sim.species('DG80_G4d_G80d', texRepr='D_{G80}{:}G_{4D}{:}G_{80D}', annotation="Gal80 gene repressed")
    sim.species('R80', texRepr='R_{80}', annotation="Gal80 mRNA")
    sim.species('G80', texRepr='G_{80}', annotation="Gal80 protein")
    sim.species('G80d', texRepr='G_{80D}', annotation="Gal80 dimer")
    sim.species('G80d_G3i', texRepr='G_{80D}{:}G_{3i}', annotation="Gal80 dimer bound to activated Gal3")


# ## Ribosomes Particles

# In[46]:


with sim.construct():
    sim.species('ribosome', texRepr='Ribosome', annotation="Ribosome (inactive)")
    sim.species('ribosomeR1', texRepr='Ribosome{:}R_{1}', annotation="Ribosome bound to Gal1 mRNA")
    sim.species('ribosomeR2', texRepr='Ribosome{:}R_{2}', annotation="Ribosome bound to Gal2 mRNA")
    sim.species('ribosomeR3', texRepr='Ribosome{:}R_{3}', annotation="Ribosome bound to Gal3 mRNA")
    sim.species('ribosomeR4', texRepr='Ribosome{:}R_{4}', annotation="Ribosome bound to Gal4 mRNA")
    sim.species('ribosomeR80', texRepr='Ribosome{:}R_{80}', annotation="Ribosome bound to Gal80 mRNA")
    sim.species('ribosomeGrep', texRepr='Ribosome{:}G_{rep}', annotation="Ribosome bound to reporter mRNA")


# # 3. Reactions

# In[47]:


cellVol = 3.57e-14 # L, cell size from Ramsey paper SI, haploid yeast
nav = cellVol*6.022e23  # volume times Avogadro's number
invMin2invSec = 1/60.0 # s^-1/min^-1
conv2ndOrder = invMin2invSec*nav # convert to s^-1*M^-1
conv1stOrder = invMin2invSec # convert to s^-1


# ## Dimerization

# In[48]:


with sim.construct():
    sim.rateConst('fd',100 * conv2ndOrder, order= 2, annotation ="Gal4p/Gal80p dimer formation")
    sim.rateConst('rd', 0.001 * conv1stOrder, order= 1, annotation ="Gal4p/Gal80p dimer dissociation")
    sim.reaction([sp.G4, sp.G4], [sp.G4d], rc.fd, annotation ="Gal4p/Gal80p dimer formation", regions=[reg.cytoplasm, reg.nucleoplasm])
    sim.reaction([sp.G4d], [sp.G4, sp.G4], rc.rd, annotation ="Gal4p/Gal80p dimer dissociation", regions=[reg.cytoplasm, reg.nucleoplasm])
    sim.reaction([sp.G80, sp.G80], [sp.G80d], rc.fd, annotation ="Gal80p/Gal80p dimer formation", regions=[reg.cytoplasm, reg.nucleoplasm])
    sim.reaction([sp.G80d], [sp.G80, sp.G80], rc.rd, annotation ="Gal80p/Gal80p dimer dissociation", regions=[reg.cytoplasm, reg.nucleoplasm])


# ## DNA regulation

# In[49]:


with sim.construct():
    Kp4 = 0.2600 # 4 binding sites
    Kq4 = 1.1721 
    
    kf1_4 = 0.1
    kf2_4 = 0.1
    kr1_4 = kf1_4/Kp4
    kr2_4 = kf2_4/Kq4
    
    Kp5 = 0.0099 # 5 binding sites
    Kq5 = 0.7408 
    
    kf1_5 = 0.1
    kf2_5 = 0.1
    kr1_5 = kf1_5/Kp5
    kr2_5 = kf2_5/Kq5
    
    
    Kp = 0.0248 # 1 binding site
    Kq = 0.1885 
    kf1 = 0.1
    kr1 = kf1/Kp
    kf2 = 0.1
    kr2 = kf2/Kq

    #convBinding = invMin2invSec*sim.siteNAV

    sim.rateConst("f1", kf1*conv2ndOrder, order=2, annotation="Gene/Gal4p binding [1 site]")
    sim.rateConst("r1", kr1/100*conv1stOrder, order=1, annotation="Gene/Gal4p dissociation [1 site]")
    sim.rateConst("f2", kf2/100*conv2ndOrder, order=2, annotation="Gene/Gal80p binding [1 site]")
    sim.rateConst("r2", kr2*conv1stOrder, order=1, annotation="Gene/Gal80p dissociation [1 site]")
    
    sim.rateConst("f1_4", kf1_4*conv2ndOrder, order=2, annotation="Gene/Gal4p binding [4 sites]")
    sim.rateConst("r1_4", kr1_4/100*conv1stOrder, order=1, annotation="Gene/Gal4p dissociation [4 sites]")
    sim.rateConst("f2_4", kf2_4/100*conv2ndOrder, order=2, annotation="Gene/Gal80p binding [4 sites]")
    sim.rateConst("r2_4", kr2_4*conv1stOrder, order=1, annotation="Gene/Gal80p dissociation [4 sites]")
    
    sim.rateConst("f1_5", kf1_5*conv2ndOrder, order=2, annotation="Gene/Gal4p binding [5 sites]")
    sim.rateConst("r1_5", kr1_5/100*conv1stOrder, order=1, annotation="Gene/Gal4p dissociation [5 sites]")
    sim.rateConst("f2_5", kf2_5/100*conv2ndOrder, order=2, annotation="Gene/Gal80p binding [5 sites]")
    sim.rateConst("r2_5", kr2_5*conv1stOrder, order=1, annotation="Gene/Gal80p dissociation [5 sites]")
    
    ### details
    # G1, Grep has 4 sites
    # G2 has 5 sites
    # G3, G80 have 1 site
    
    ### no regulation for G3 and G80
    dnas            = [sp.DG1           , sp.DG2       , sp.DGrep]
    dna_gal4        = [sp.DG1_G4d       , sp.DG2_G4d   , sp.DGrep_G4d]
    dna_gal4_gal80  = [sp.DG1_G4d_G80d, sp.DG2_G4d_G80d, sp.DGrep_G4d_G80d]
    f_g4s           = [rc.f1_4          , rc.f1_5      , rc.f1_4]
    r_g4s           = [rc.r1_4          , rc.r1_5      , rc.r1_4]
    f_g4g80s        = [rc.f2_4          , rc.f2_5      , rc.f2_4]
    r_g4g80s        = [rc.r2_4          , rc.r2_5      , rc.r2_4]
    
    for dna, dna_gal4, dna_gal4_gal80, f_g4, r_g4, f_g4g80, r_g4g80 in zip(dnas, dna_gal4, dna_gal4_gal80, f_g4s, r_g4s, f_g4g80s, r_g4g80s):
        sim.reaction([dna, sp.G4d], [dna_gal4], f_g4, annotation="Gal4p binding to gene", regions=reg.nucleoplasm)
        sim.reaction([dna_gal4], [dna, sp.G4d], r_g4, annotation="Gal4p dissociation from gene", regions=reg.nucleoplasm)
        # this is modified by reducing the binding rate by 10^2 times, to increase the time it exits in state DG_G4d for transcription.
        sim.reaction([dna_gal4, sp.G80d], [dna_gal4_gal80], f_g4g80, annotation="Gal80p binding to gene", regions=reg.nucleoplasm)
        sim.reaction([dna_gal4_gal80], [dna_gal4, sp.G80d], r_g4g80, annotation="Gal80p dissociation from gene", regions=reg.nucleoplasm)
    


# ## G3 activation

# In[50]:


with sim.construct():
    sim.rateConst("fi", 7.45e-7*conv2ndOrder, order=1, annotation="Gal3p activation ") # needs to be multiplied by [GAI] in mol/L,  modified by set reactions rate in hook 
    sim.rateConst("ri", 890.0*conv1stOrder, order=1, annotation="Gal3p deactivation")
    sim.rateConst("fd3i80", 0.025716*conv2ndOrder, order=2, annotation="Gal3p*/Gal80 association")
    sim.rateConst("dr3i80", 0.0159616*conv1stOrder, order=1, annotation="Gal3p*/Gal80 disassociation")
    sim.rateConst("dp_gal3", 0.01155*conv1stOrder, order=1, annotation="GAL3 degradation")
    sim.rateConst("dp_gal3gal80", 0.5*rc.dp_gal3.value, order=1, annotation="Gal3p*:Gal80 degradation")
    
    sim.reaction(sp.G3, sp.G3i, rc.fi, annotation="Gal3p activation", regions=reg.cytoplasm)
    sim.reaction(sp.G3i, sp.G3, rc.ri, annotation="Gal3p deactivation", regions=reg.cytoplasm)
    ## Gal3p* Gal80p
    sim.reaction([sp.G3i, sp.G80d], [sp.G80d_G3i], rc.fd3i80, annotation="Gal3p*/Gal80 association", regions=reg.cytoplasm)
    sim.reaction([sp.G80d_G3i], [sp.G3i, sp.G80d], rc.dr3i80, annotation="Gal3p*/Gal80 disassociation", regions=reg.cytoplasm)
    ## Gal3p+galactose degradation, should left the internal galactose, but neglected
    sim.reaction(sp.G3i, [], rc.dp_gal3, annotation="GAL3 degradation", regions=reg.cytoplasm)
    sim.reaction(sp.G80d_G3i, [], rc.dp_gal3gal80, annotation="Gal3p*:Gal80 degradation", regions=reg.cytoplasm)


# ## Transcription

# In[51]:

# mutant is added here.
with sim.construct():
    sim.rateConst("alpha1", 0.7379*conv1stOrder, order=1, annotation='GAL1 transcription')
    sim.rateConst("alpha2", 2.542*conv1stOrder, order=1, annotation='GAL2 transcription')
    sim.rateConst("alpha3", 1.0*0.7465*f_mut*conv1stOrder, order=1, annotation='GAL3 transcription')
    sim.rateConst("ir_gal4", 0.009902*conv1stOrder, order=1, annotation='GAL4 transcription')
    sim.rateConst("alpha_rep", 1.1440*conv1stOrder, order=1, annotation='GFP transcription')
    sim.rateConst("alpha80", 0.6065*f_mut*conv1stOrder, order=1, annotation='GAL80 transcription')

    sim.rateConst("dr_gal1", 0.02236*conv1stOrder, order=1, annotation='GAL1 mRNA degradation')
    sim.rateConst("dr_gal2", 0.07702*conv1stOrder, order=1, annotation='GAL2 mRNA degradation')
    sim.rateConst("dr_gal3", 0.02666*conv1stOrder, order=1, annotation='GAL3 mRNA degradation')
    sim.rateConst("dr_gal4", 0.02476*conv1stOrder, order=1, annotation='GAL4 mRNA degradation')
    sim.rateConst("dr_rep", 0.03466*conv1stOrder, order=1, annotation='GFP mRNA degradation')
    sim.rateConst("dr_gal80", 0.02888*conv1stOrder, order=1, annotation='GAL80 mRNA degradation')
    
    transcription_rates = [rc.alpha1,   rc.alpha2,  rc.alpha3,  rc.ir_gal4, rc.alpha_rep,   rc.alpha80]
    decay_rates         = [rc.dr_gal1,  rc.dr_gal2, rc.dr_gal3, rc.dr_gal4, rc.dr_rep,      rc.dr_gal80]
    genes               = [sp.DG1_G4d,  sp.DG2_G4d, sp.DG3,     sp.DG4,     sp.DGrep_G4d,   sp.DG80]
    mrnas               = [sp.R1,       sp.R2,      sp.R3,      sp.R4,      sp.Rrep,        sp.R80]
    
    for trans_rate, decay_rate, gene, mrna in zip(transcription_rates, decay_rates, genes, mrnas):
        sim.reaction([gene], [gene, mrna], trans_rate, regions=reg.nucleoplasm)
        sim.reaction([mrna], [], decay_rate, regions=[reg.nucleoplasm, reg.cytoplasm, reg.ribosomes])


# transcription -2, dummy related transcription and degradation.

# ## Translation

# Ribosome/mRNA association rate. For the diffusive propensity to be equal to the reaction propensity, we need
# $$ k = 2000DN_A\lambda.$$
# We will choose the reaction propensity to be 0.2 of the diffsive propensity so that immediate reassociation happens 1 of 5 dissociation events.

# In[52]:


with sim.construct():
    #tlInitDet = 100e6 # 10.1016/j.molcel.2006.02.014 [eco]
    tlInitDet = 0.2 * 2000 * mRNADiffusion * sim.NA * sim.latticeSpacing
    sim.rateConst("rib_assoc", tlInitDet, order=2, annotation='mRNA/Ribosome association rate') 
    
    # translation rates
    sim.rateConst("ip_gal1", 1.9254*conv1stOrder, order=1, annotation='GAL1 translation')
    sim.rateConst("ip_gal2", 13.4779*conv1stOrder, order=1, annotation="GAL2 translation")
    sim.rateConst("ip_gal3", 55.4518*conv1stOrder, order=1, annotation="GAL3 translation")
    sim.rateConst("ip_gal4", 10.7091*conv1stOrder, order=1, annotation="GAL4 translation")
    sim.rateConst("ip_rep", 5.7762*conv1stOrder, order=1, annotation="GFP translation")
    sim.rateConst("ip_gal80", 3.6737*conv1stOrder, order=1, annotation="GAL80 translation")
    # protein degradation rates
    sim.rateConst("dp_gal1", 0.003851*conv1stOrder, order=1, annotation='GAL1 degradation')
    sim.rateConst("dp_gal2", 0.003851*conv1stOrder, order=1, annotation="GAL2 degradation")
    sim.rateConst("dp_gal3", 0.01155*conv1stOrder, order=1, annotation="GAL3 degradation")
    sim.rateConst("dp_gal4", 0.006931*conv1stOrder, order=1, annotation="GAL4 degradation")
    sim.rateConst("dp_rep", 0.01155*conv1stOrder, order=1, annotation="GFP degradation")
    sim.rateConst("dp_gal80", 0.006931*conv1stOrder, order=1, annotation="GAL80 degradation")
    #rates and species
    ktls =                 [rc.ip_gal1,     rc.ip_gal2,     rc.ip_gal3,     rc.ip_gal4,     rc.ip_rep,      rc.ip_gal80]
    dcys =                 [rc.dp_gal1,     rc.dp_gal2,     rc.dp_gal3,     rc.dp_gal4,     rc.dp_rep,      rc.dp_gal80] 
    mdcys =                [rc.dr_gal1,     rc.dr_gal2,     rc.dr_gal3,     rc.dr_gal4,     rc.dr_rep,      rc.dr_gal80]
    translatingRibosomes = [sp.ribosomeR1, sp.ribosomeR2, sp.ribosomeR3, sp.ribosomeR4, sp.ribosomeGrep, sp.ribosomeR80]
    prots =                [sp.G1,         sp.G2,         sp.G3,         sp.G4,         sp.Grep,         sp.G80]
    
    for mrna, translatingRibosomes, protein, ktl, dcy, mdcy in zip(mrnas, translatingRibosomes, prots, ktls, dcys, mdcys):
        #association
        sim.reaction([sp.ribosome, mrna], [translatingRibosomes], rc.rib_assoc, regions=reg.ribosomes)
        #translation
        sim.reaction([translatingRibosomes], [sp.ribosome, mrna, protein], ktl, regions=reg.ribosomes)
        #degradation in association form
        sim.reaction([translatingRibosomes], [sp.ribosome], mdcy, regions=reg.ribosomes)
    
    


# -3, dummy rna association, translation and degradation

# ## Protein degradation(out of ribosome)

# In[53]:


with sim.construct():
    deg_compartments =         [[reg.ribosomes, reg.cytoplasm], 
                                [reg.ribosomes, reg.cytoplasm, reg.plasmaMembrane], 
                                [reg.ribosomes, reg.cytoplasm], 
                                [reg.ribosomes, reg.cytoplasm, reg.nucleoplasm], 
                                [reg.ribosomes, reg.cytoplasm], 
                                [reg.ribosomes, reg.cytoplasm, reg.nucleoplasm]]
    for protein, decay_rate, region in zip(prots, dcys, deg_compartments):
        sim.reaction([protein], [], decay_rate, regions=region)


# In[54]:


# sim.showReactions()


# # Initial conditions

# In[55]:

if IF_DGX == True:
    initMolec = pickle.load(open("/workspace/cme_species_counts.pkl", "rb"))
else:
    initMolec = pickle.load(open("cme_species_counts.pkl", "rb"))
# change molecular/unit volume to actual molecules
volScale = np.sum(B.convexHull(sim.siteLattice==reg.plasmaMembrane.idx))*sim.siteV/cellVol
def initMolecules(x):
    # convert molecules/unit volume to molecues
    counts = int(round(initMolec[x]*volScale))
    return counts
if checkpoint_file == "":
    # place all genes randomly in the nucleoplasm
    if gene_location == "random":
        print("gene location random")
        print("This is a non regulation model, DG3 and DG80 are not regulated, Mutant Ramsey example")
        # initial states of DG1, DG2, DGrep
        # non regulation , DG3 and DG80 are not regulated
        for b in ["DG1", "DG2", "DGrep"]:
            ops = [b+x for x in ["", "_G4d", "_G4d_G80d"]]
            spName = max(ops, key=lambda x:initMolec[x])
            print("{} in state {}".format(b, spName))
            sim.species(spName).placeNumberInto(reg.nucleoplasm, 1)
        
        sp.DG4.placeNumberInto(reg.nucleoplasm, 1)
        print("{} in state {}".format("Gene4", "DG4"))
        sp.DG3.placeNumberInto(reg.nucleoplasm, 1)
        print("{} in state {}".format("Gene3", "DG3"))
        sp.DG80.placeNumberInto(reg.nucleoplasm, 1)
        print("{} in state {}".format("Gene80", "DG80"))
    else:
        print("gene location fixed")
        # dna_gal4_gal80  = [sp.DG1_G4d_G80d, sp.DG2_G4d_G80d , sp.DG3_G4d_G80d   , sp.DG80_G4d_G80d  , sp.DGrep_G4d_G80d]
        # check if x, y, z are in the nucleoplasm
        
        # Define gene locations
        gene_locations_0823_traj4 = {
            "DGrep_G4d_G80d": [117, 86, 133],
            "DG1_G4d_G80d": [132, 90, 90],
            "DG2_G4d_G80d": [116, 73, 132],
            "DG3_G4d_G80d": [111, 58, 100],
            "DG80_G4d_G80d": [132, 94, 115],
            "DG4": [126, 61, 115]
        }

        gene_locations_0823_traj3 = {
            "DGrep_G4d_G80d": [137, 64, 137],
            "DG1_G4d_G80d": [128, 71, 123],
            "DG2_G4d_G80d": [144, 74, 120],
            "DG3_G4d_G80d": [119, 82, 110],
            "DG80_G4d_G80d": [135, 90, 85],
            "DG4": [109, 90, 120]
        }
        # Get nucleoplasm coordinates
        nucleoplasm_coords = set(map(tuple, np.argwhere(sim.siteLattice == reg.nucleoplasm.idx)))
        if gene_location == "0823_traj4":
            # Check each gene location
            for gene, loc in gene_locations_0823_traj4.items():
                if tuple(loc) in nucleoplasm_coords:
                    print(f"{gene} is inside the nucleoplasm")
                    # Place the gene at the specified location
                    sim.placeNumber(sp=getattr(sp, gene), x=loc[0], y=loc[1], z=loc[2], n=1)
                else:
                    print(f"{gene} is not inside the nucleoplasm")
                    print(tuple(loc))
        elif gene_location == "0823_traj3":
            # Check each gene location
            for gene, loc in gene_locations_0823_traj3.items():
                if tuple(loc) in nucleoplasm_coords:
                    print(f"{gene} is inside the nucleoplasm")
                    # Place the gene at the specified location
                    sim.placeNumber(sp=getattr(sp, gene), x=loc[0], y=loc[1], z=loc[2], n=1)
                else:
                    print(f"{gene} is not inside the nucleoplasm")
                    print(tuple(loc))
        else:
            print(f"gene location not recognized: {gene_location}")

    # place proteins/ metabolites
    sp.G1.placeNumberInto(reg.cytoplasm, initMolecules("G1"))
    print("G1 in cytoplasm: {}".format(initMolecules("G1")))
    sp.G2.placeNumberInto(reg.plasmaMembrane, initMolecules("G2"))
    print("G2 in plasma membrane: {}".format(initMolecules("G2")))
    sp.G3.placeNumberInto(reg.cytoplasm, initMolecules("G3"))
    print("G3 in cytoplasm: {}".format(initMolecules("G3")))
    sp.G4d.placeNumberInto(reg.nucleoplasm, initMolecules("G4d"))
    print("G4d in nucleoplasm: {}".format(initMolecules("G4d")))
    sp.Grep.placeNumberInto(reg.cytoplasm, initMolecules("Grep"))
    print("Grep in cytoplasm: {}".format(initMolecules("Grep")))
    # place mRNA for G3 and G80 
    # modifications = {'reporter_rna': 0.26,
    #              'R1': 0.26,
    #              'R2': 0.33,
    #              'R3': 0.90,
    #              'R4': 0.40,
    #              'R80': 1.18,
    #              'G80C': 0.11,
    #              'G80': 0.11,
    #              'G4': 0.15
    #              }
    sp.R3.placeNumberInto(reg.nucleoplasm, 1)
    sp.R80.placeNumberInto(reg.nucleoplasm, 1)
    print("R3 in nucleoplasm: {}".format(initMolecules("R3")))
    print("R80 in nucleoplasm: {}".format(initMolecules("R80")))
    # place G80
    # place as suggested by 2006 paper
    sp.G80.placeNumberInto(reg.cytoplasm, initMolecules("G80C")*4)
    sp.G80.placeNumberInto(reg.nucleoplasm, initMolecules("G80")*4)
    sp.G80d.placeNumberInto(reg.cytoplasm, initMolecules("G80Cd")*4)
    sp.G80d.placeNumberInto(reg.nucleoplasm, initMolecules("G80d")*4)
    print("G80 in cytoplasm: {}, in nucleoplasm: {}".format(initMolecules("G80C")*4, initMolecules("G80")*4))
    print("G80d in cytoplasm: {}, in nucleoplasm: {}".format(initMolecules("G80Cd")*4, initMolecules("G80d")*4))

    # place ribosome particles in ribosomes region

    for x, y, z in np.argwhere(sim.siteLattice == reg.ribosomes.idx):
        sp.ribosome.placeParticle(x, y, z, 1)

    print("ribosomes number:", np.sum(sim.siteLattice == reg.ribosomes.idx))
else:
    print(f"using checkpoint file:{checkpoint_file}")
    # start from the last checkpoint file
    if os.path.exists(checkpoint_file):
        print(f"start from the last checkpoint file{checkpoint_file}")
        try:
            sim.copyParticleLattice(checkpoint_file, replicate=1, frame=-1)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint file {checkpoint_file}. Error: {str(e)}")
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}. Cannot restart simulation.")

# # Diffusion Coefficients

# In[56]:


with sim.construct():
    sim.transitionRate(None, None, None, sim.diffusionZero)


# ## DNA
# Fix DNA in the location:

# In[57]:


with sim.construct():
    for sps in sim.speciesList.matchRegex("D.*"):
        sps.diffusionRate(None, sim.diffusionZero)


# ## mRNA 

# In[58]:


with sim.construct():
    sim.diffusionConst("mrna", mRNADiffusion, texRepr=r'D_{mRNA}', annotation='Generic mRNA')

    for mrna in sim.speciesList.matchRegex("R.*"):
        sim.transitionRate(mrna, reg.nucleoplasm, reg.cytoplasm, dc.mrna)
        sim.transitionRate(mrna, reg.cytoplasm, reg.nucleoplasm, sim.diffusionZero)
        sim.transitionRate(mrna, reg.nucleoplasm, reg.nucleoplasm, dc.mrna)
        sim.transitionRate(mrna, reg.cytoplasm, reg.cytoplasm, dc.mrna)
        sim.transitionRate(mrna, reg.ribosomes, reg.ribosomes, dc.mrna)
        sim.transitionRate(mrna, reg.ribosomes, reg.cytoplasm, dc.mrna)
        sim.transitionRate(mrna, reg.cytoplasm, reg.ribosomes, dc.mrna)


# ## Ribosome occlusion
# after proteins get translated, it can only diffuse out of ribosomes, cant get back.

# In[59]:


with sim.construct():
    sim.diffusionConst("prot", 1e-12, texRepr=r'D_{prot}', annotation='Generic protein')
    for sps in [sp.G1, sp.G2, sp.G3, sp.G3i, sp.G4, sp.G4d, sp.G80, sp.G80d, sp.G80d_G3i, sp.Grep]:
        sim.transitionRate(sps, reg.cytoplasm, reg.cytoplasm, dc.prot)
        sim.transitionRate(sps, reg.ribosomes, reg.cytoplasm, sim.diffusionFast)
        sim.transitionRate(sps, reg.cytoplasm, reg.ribosomes, sim.diffusionZero)


# ## Transcription Factors(G4, G80)
# 
# allow them to diffuse into the nucleoplasm, and diffuse in the nucleoplasm

# In[60]:


with sim.construct():
    for sps in [sp.G4, sp.G4d, sp.G80, sp.G80d]:
        sim.transitionRate(sps, reg.nucleoplasm, reg.nucleoplasm, dc.prot)
        sim.transitionRate(sps, reg.nucleoplasm, reg.cytoplasm, dc.prot)
        sim.transitionRate(sps, reg.cytoplasm, reg.nucleoplasm, dc.prot)

print("fast diffsuion rate is ", sim.diffusionFast.value)
# sys.exit(0)
# ## cytoplasmic protein
# 
# prevent them diffuse into nucleoplasm

# In[61]:


with sim.construct():
    for sps in [sp.G1, sp.G2, sp.G3, sp.G3i, sp.G80d_G3i, sp.Grep]:
        sim.transitionRate(sps, reg.cytoplasm, reg.nucleoplasm, sim.diffusionZero) 


# ## Transporter

# once diffuse into the membrane, can not get out.

# In[62]:


with sim.construct():
    sim.transitionRate(sp.G2, reg.cytoplasm, reg.plasmaMembrane, dc.prot)
    sim.transitionRate(sp.G2, reg.plasmaMembrane, reg.cytoplasm, sim.diffusionZero)
    sim.diffusionConst("mem", 0.01e-12, texRepr=r'D_{mem}', annotation='Generic protein on membrane')

    # sp.G2.diffusionRate(reg.plasmaMembrane, dc.mem)
    sim.transitionRate(sp.G2, reg.plasmaMembrane, reg.plasmaMembrane, dc.mem)


# ## Ribosomes
# 
# not moving, stay in the corresponding cube

# In[63]:


with sim.construct():
    for sps in sim.speciesList.matchRegex("ribosome.*"):
        sim.transitionRate(sps, None, None, sim.diffusionZero)


# # RDME-ODE hybrid

# In[64]:


# import pyximport
# pyximport.install(setup_args={ "include_dirs":np.get_include()})
# import sys
# if IF_DGX == True:
#     sys.path.append("/workspace")
# from ode import RHS
import json
import scipy.integrate as spi
# In[65]:
# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
class OdeRdmeHybridSolver:
    """Hybrid solver for Galactose switch
    
    Transport reactions and galactose metabolism handled by ODE, rest by RDME. The
    ODE system is coupled to the RDME by updating the protein counts in the ODE 
    each communication step. New proteins are added to the state unbound to
    galactose. If the number of proteins decreases, both bound and unbound 
    states are decreased by the same proportion and the galactose is
    added back as internal or external galactose. The RDME is coupled to the 
    ODE system through the internal galactose concentration. The rate of
    G3 -> G3i is updated with the internal galactose concentration each
    communication step."""
    def __init__(self, lmFile, initialExternalGalactose):
        super(OdeRdmeHybridSolver, self).__init__()
        self.GAE = initialExternalGalactose #M
        self.odeYs = None
        self.lastOdeEval = 0
        self.odeEvals = []
        self.odeSpNames = ['GAI', 'G1', 'G1GAI', 'G2GAI', 'G2GAE', 'G2']
        if isinstance(lmFile, (RDMEFile, RDMESim)):
            self.rdme = lmFile
        else:
            self.rdme = RDME.File(lmFile)
        self.cellVol = self.rdme.reg.cytoplasm.volume + self.rdme.reg.nucleoplasm.volume + self.rdme.reg.plasmaMembrane.volume
        self.NAV = 6.022e23*self.cellVol
        # self.buildOdeSys(
        #     #  unit is /s or /s/M
        #         [[["G1","GAI"],  ["G1GAI"],     1.442e5],
        #         [["G1GAI"],     ["G1", "GAI"], 30.708],
        #         [["G1GAI"],     ["G1"],        55.833],
        #         [["G2GAI"],     ["G2GAE"],     72.5],
        #         [["G2GAE"],     ["G2GAI"],     72.5],
        #         [["G2GAE"],     ["G2"], 39.875],
        #         [["G2"],        ["G2GAE"],     1.123e5*self.GAE],
        #         [["G2", "GAI"], ["G2GAI"],     1.123e5],
        #         [["G2GAI"],     ["G2", "GAI"], 39.875]])
        
                  
        self.g3actRidx = self.rdme.reaction(self.rdme.sp.G3, self.rdme.sp.G3i, self.rdme.rc.fi).idx
        self.g3actRc = self.rdme.rc.fi._toLM()
        # Open the output file once during initialization
        self.save_cts_by_region_file = output_folder + "_region.jsonl"
        self.save_cts_by_region_handle = open(self.save_cts_by_region_file, "w")  # Open in write mode to start fresh
        self.hook_time = 0  # Initialize timer for hookSimulation
        
        # Add a new file handle for ODE data
        self.save_ode_data_file = output_folder + "_ode.jsonl"
        self.save_ode_data_handle = open(self.save_ode_data_file, "w")  # Open in write mode to start fresh
    
    def copyInitialConditions(self, cts):
        if checkpoint_file == "":
            # initialization will count all G1 and G2 in ODE
            y = np.zeros(len(self.odeSpNames))
            y[self.odeSpIndex("GAI")] = 0
            y[self.odeSpIndex("G1")] = cts['countBySpecies'][self.rdme.sp.G1]/self.NAV 
            y[self.odeSpIndex("G1GAI")] = 0
            y[self.odeSpIndex("G2")] = cts['countBySpecies'][self.rdme.sp.G2]/self.NAV
            y[self.odeSpIndex("G2GAE")] = 0
            y[self.odeSpIndex("G2GAI")] = 0
            print(f"G2 concentration: ", y[self.odeSpIndex("G2")] , "G1 concentration: ",y[self.odeSpIndex("G1")] )
        else:
            print(f"using checkoutpoint:{checkpoint_file}")
            checkpoint_ode = checkpoint_file + "_ode.jsonl"
           
            # Get last frame from ODE JSONL
            with open(checkpoint_ode, 'r') as f:
                last_line = None
                for line in f:
                    last_line = line
                
                if last_line is None:
                    raise RuntimeError(f"ODE checkpoint file {checkpoint_ode} is empty")
                
                last_ode_state = json.loads(last_line.strip())
                if 'species' not in last_ode_state:
                    raise RuntimeError(f"Invalid ODE state format in {checkpoint_ode}")
                
                # Initialize y from the last ODE state
                y = np.zeros(len(self.odeSpNames))
                for i, name in enumerate(self.odeSpNames):
                    y[self.odeSpIndex(name)] = last_ode_state['species'][name]
                
                print(f"Initialized ODE state from time {last_ode_state['time']}")
        
        self.boundGal = self.rdmeGal(cts)
        return y
    
    def rdmeGal(self, cts):
        # get internal galactose counts from RDME and change to concentration in ODE
        return (cts['countBySpecies'][self.rdme.sp.G3i] + cts['countBySpecies'][self.rdme.sp.G80d_G3i])/self.NAV

    def rdme2odeConc(self, y0, cts):
        y = y0.copy()
        # update G1 in ODE
        g1ode = y0[self.odeSpIndex("G1")]
        g1gaiode = y0[self.odeSpIndex("G1GAI")]
        g1rdme = cts['countBySpecies'][self.rdme.sp.G1]/self.NAV
        #update criteria
        change = g1rdme-g1ode-g1gaiode
        
        if change > 0:
            y[self.odeSpIndex("G1")] = g1ode + change
        else:
            fracChange = g1rdme/(g1ode+g1gaiode)
            y[self.odeSpIndex("G1")] = g1ode*fracChange
            y[self.odeSpIndex("G1GAI")] = g1gaiode*fracChange
            y[self.odeSpIndex("GAI")] += g1gaiode*(1-fracChange)
            
        # update G2 in ODE
        g2ode = y0[self.odeSpIndex("G2")]
        g2gaiode = y0[self.odeSpIndex("G2GAI")]
        g2gaeode = y0[self.odeSpIndex("G2GAE")]
        g2rdme = cts['countBySpeciesRegion'][self.rdme.sp.G2][self.rdme.reg.plasmaMembrane]/self.NAV
        
        
        change = g2rdme-g2ode-g2gaiode-g2gaeode
        
        if change >= 0:
            y[self.odeSpIndex("G2")] = g2ode + change
        else:
            fracChange = g2rdme/(g2ode+g2gaiode+g2gaeode)
            y[self.odeSpIndex("G2")] = g2ode*fracChange
            y[self.odeSpIndex("G2GAI")] = g2gaiode*fracChange
            y[self.odeSpIndex("GAI")] += g2gaiode*(1-fracChange)
            y[self.odeSpIndex("G2GAE")] = g2gaeode*fracChange
            # y[self.odeSpIndex("G2")] = int(g2ode*fracChange)
            # y[self.odeSpIndex("G2GAI")] = int(g2gaiode*fracChange)
            # y[self.odeSpIndex("GAI")] += int(g2gaiode*(1-fracChange))
            # y[self.odeSpIndex("G2GAE")] = int(g2gaeode*fracChange)
        # update internal galactose in ODE
        g0 = self.boundGal
        g1 = self.rdmeGal(cts)
        y[self.odeSpIndex("GAI")] += g1-g0
        self.boundGal = g1
            
        return y
  
                   
    def hookSimulation(self, t, lattice):
        print_memory_usage()
        start_time_hook = time.time()  # Start timer for this hook call
        
        # this is for the hook simulation 
        cts = self.rdme.particleStatistics(particleLattice=lattice.getParticleLatticeView(),
                                           siteLattice=lattice.getSiteLatticeView())
        if self.odeYs is None:
            ys0 = self.copyInitialConditions(cts)
        else:
            ys0 = self.rdme2odeConc(self.odeYs, cts)
                   
        dt = t-self.lastOdeEval
        if dt>0:
            ys1 = self.stepOde(dt, ys0)
        else:
            ys1 = ys0
            
        self.odeEvals.append((t,ys1))
        self.odeYs = ys1
        self.lastOdeEval = t
            
        assocRt = max(0,self.g3actRc*ys1[self.odeSpIndex("GAI")])
        # print(self.rdme.reactionList[self.g3actRidx].rate.value)
        self.setReactionRate(self.g3actRidx, assocRt)
        self.save_rdme_cts_by_region(t, cts)
        self.save_ode_data(t, ys1)  # New method call to save ODE data
        # print(self.rdme.reactionList[self.g3actRidx].rate.value)
        self.print_ode_evals(t,assocRt,cts)
        
        end_time_hook = time.time()
        self.hook_time += end_time_hook - start_time_hook  # Add time spent in this hook call
        # Check if maximum simulation time has been reached
        
        if args.max_time is not None and (end_time_hook - start_time) >= args.max_time * 3600:  # Convert hours to seconds
            print(f"Maximum simulation time of {args.max_time} hours reached. Stopping simulation.")
            return 3  # Return 3 to stop the simulation
        return 0
    def print_ode_evals(self,t,assocRt,cts):
        print("="*80)
        print("t=",t)
        print("ODE")
        for i,n in enumerate(self.odeSpNames):
            print("  {:<16s}{:16.5g}".format(n,self.odeYs[i]))
        print("RDME")
        for n in self.rdme.speciesList:
            print("  {:<16s}{:16d}".format(n.name,cts['countBySpecies'][n]))
        print("new rate g3 activation: {:.3g}".format(assocRt))
        print("-"*80)
        return 
    def odeSpIndex(self, sp):
        return self.odeSpNames.index(sp)
    
    def buildOdeSys(self, rxns):
        odeSpNames = set()
        for react, prod, _ in rxns:
            odeSpNames |= set(prod) | set(react)
        nsp = len(odeSpNames)
        nrxn = len(rxns)
        stoch = np.zeros((nrxn, nsp), dtype=np.int32)
        dep = np.zeros((nrxn, nsp), dtype=np.int32)
        ks = np.zeros(nrxn, dtype=np.float64)
        self.odeSpNames = sorted(odeSpNames)
        for i, (react, prod, rate) in enumerate(rxns):
            ks[i] = rate
            for r in react:
                dep[i, self.odeSpIndex(r)] += 1
                stoch[i, self.odeSpIndex(r)] -= 1
            for p in prod:
                stoch[i, self.odeSpIndex(p)] += 1
        self.odeStoch = stoch
        self.odeDep = dep
        self.odeKs = ks
    def ode_model(self,conc, ts,  GAE):
        '''
        System:
                [[["G1","GAI"],  ["G1GAI"],     1.442e5],
                [["G1GAI"],     ["G1", "GAI"], 30.708],
                [["G1GAI"],     ["G1"],        55.833],
                [["G2GAI"],     ["G2GAE"],     72.5],
                [["G2GAE"],     ["G2GAI"],     72.5],
                [["G2GAE"],     ["G2"], 39.875],
                [["G2"],        ["G2GAE"],     1.123e5*GAE],
                [["G2", "GAI"], ["G2GAI"],     1.123e5],
                [["G2GAI"],     ["G2", "GAI"], 39.875]]
        Input:
                conc: a list of species concentration(M) in initial time step
                GAE: the extracellular galactose concentration in mM
                ts: the time steps to be evaluated
        Output:
                dydt: a list of derivatives of the species concentration(M)
                species_list: a list of species names
        '''
        NA = 6.02214076e23 # Avogadro's number
        # galactokinease G1
        kf_GK = 1.442e5 # M^-1 s^-1 
        kr_GK = 30.708 # s^-1
        # metabolites
        kcat_GK = 55.833 # s^-1
        # Transporter G2
        kcat_TR = 72.5 # s^-1
        kr_TR = 39.875 # s^-1
        kf_TR = 1.123e5 # M^-1 s^-1 
        # G2 --> G2GAE
        kf_TR_gae = 1.123e5* GAE # s^-1
        GAI = conc[self.odeSpIndex("GAI")]
        G2GAI = conc[self.odeSpIndex("G2GAI")]
        G2GAE = conc[self.odeSpIndex("G2GAE")]
        G1GAI = conc[self.odeSpIndex("G1GAI")]
        G1 = conc[self.odeSpIndex("G1")]
        G2 = conc[self.odeSpIndex("G2")]
    
        # GAI
        dGAI_dt = kr_TR*G2GAI - kf_TR*GAI*G2 + kr_GK*G1GAI - kf_GK*G1*GAI  #+ kdp_gal1*G1GAI + kdp_gal2*G2GAI
        # G1
        dG1_dt =  kr_GK*G1GAI - kf_GK*G1*GAI + kcat_GK*G1GAI
        # G1GAI
        dG1GAI_dt = kf_GK*G1*GAI - kr_GK*G1GAI - kcat_GK*G1GAI
        # G2
        dG2_dt = kr_TR*G2GAI - kf_TR*G2*GAI + kr_TR*G2GAE - kf_TR_gae*G2
        # G2GAE
        dG2GAE_dt = kf_TR_gae*G2 - kr_TR*G2GAE - kcat_TR*G2GAE + kcat_TR*G2GAI
        # G2GAI
        dG2GAI_dt = kf_TR*G2*GAI - kr_TR*G2GAI - kcat_TR*G2GAI + kcat_TR*G2GAE
       
        dx_dt = [0] * len(self.odeSpNames)  # Initialize with zeros

        dx_dt[self.odeSpIndex("GAI")] = dGAI_dt
        dx_dt[self.odeSpIndex("G1")] = dG1_dt
        dx_dt[self.odeSpIndex("G1GAI")] = dG1GAI_dt
        dx_dt[self.odeSpIndex("G2")] = dG2_dt
        dx_dt[self.odeSpIndex("G2GAI")] = dG2GAI_dt
        dx_dt[self.odeSpIndex("G2GAE")] = dG2GAE_dt
        dx_dt_array = np.asarray(dx_dt)
        return (dx_dt_array)
    
    
    def stepOde(self, dt, ys0):
        # rhs = RHS(self.odeStoch, self.odeDep, self.odeKs)
        # ts = np.linspace(0,t, 10)
        # ys = spint.odeint(rhs.evaluate, ys0, ts)
        # return ys[-1,:]
        odestep = 0.001
        ts = np.linspace(0,dt, int(np.ceil(dt/odestep))+1)
        # default is LSODA
        ys = spi.odeint(self.ode_model, ys0, ts, args=(self.GAE,), hmax=odestep)
        return ys[-1]
    # def stepOde(self, t, ys0):
    #     rhs = RHS(self.odeStoch, self.odeDep, self.odeKs)
    #     ts = np.linspace(0,t, 10)
    #     ys = spint.odeint(rhs.evaluate, ys0, ts)
    #     return ys[-1,:]

    def save_rdme_cts_by_region(self, t, stats):

     
        # Initialize a dictionary to store counts by species and region for this time step
        counts_by_region = {'time': float(t)}  # Convert t to float if it's numpy.float64

        # Iterate through all species and regions
        for species in self.rdme.speciesList:
            counts_by_region[species.name] = {}
            for region in self.rdme.regionList:
                count = stats['countBySpeciesRegion'][species][region]
                counts_by_region[species.name][region.name] = int(count)  # Convert to int

        

        # Write the current time step data to file
        json.dump(counts_by_region, self.save_cts_by_region_handle, cls=NumpyEncoder)
        self.save_cts_by_region_handle.write('\n')  # Add a newline for readability
        self.save_cts_by_region_handle.flush()  # Ensure data is written to disk

        print(f"Data for time {t} appended to {self.save_cts_by_region_file}")

        return counts_by_region

    def save_ode_data(self, t, ys):
        ode_data = {
            'time': float(t),
            'species': {name: float(value) for name, value in zip(self.odeSpNames, ys)}
        }

        json.dump(ode_data, self.save_ode_data_handle, cls=NumpyEncoder)
        self.save_ode_data_handle.write('\n')  # Add a newline for readability
        self.save_ode_data_handle.flush()  # Ensure data is written to disk

        print(f"ODE data for time {t} appended to {self.save_ode_data_file}")
#signal handler
def signal_handler(signum, frame):
    print("Interrupt received, stopping simulation...")
    if 'solver' in globals():
        if hasattr(solver, 'save_cts_by_region_handle'):
            solver.save_cts_by_region_handle.close()
        if hasattr(solver, 'save_ode_data_handle'):
            solver.save_ode_data_handle.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# In[66]:

if gpus == 1:
    Solver = makeSolver(IntMpdRdmeSolver, OdeRdmeHybridSolver)
else:
    Solver = makeSolver(MGPUMpdRdmeSolver, OdeRdmeHybridSolver)

solver = Solver(sim, externalGal)
sim.finalize()
try:
    if gpus == 1:
        traj = sim.run(solver=solver, cudaDevices=[0])
    else:
        gpu_list = list(range(gpus))
        print("using gpus: ", gpu_list)
        traj = sim.run(solver=solver, cudaDevices=gpu_list)
except Exception as e:
    print(f"An error occurred: {e}")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    import traceback
    print("Traceback:")
    traceback.print_exc()
    traj = None
finally:
    # Ensure output files are closed
    if hasattr(solver, 'save_cts_by_region_handle'):
        solver.save_cts_by_region_handle.close()
        print(f"Closed output file: {solver.save_cts_by_region_file}")
    if hasattr(solver, 'save_ode_data_handle'):
        solver.save_ode_data_handle.close()
        print(f"Closed output file: {solver.save_ode_data_file}")

if traj is not None:
    print(f"Total time spent in hookSimulation: {solver.hook_time} seconds")
else:
    print("Trajectory is not defined due to an error in simulation.")
# end timer
end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time} seconds")

