#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
v1.12 ER 
allow G2 to Diffuse through the ER
in v 1.13
1.13.1 change the region obstacle to 3 separate regions, add cell wall in as well. 
'''
# write a timer 
import time
import signal
import sys
start_time = time.time()

IF_DGX = False # parameter to for quick conversion to use on DGX
version = "1.12er_mt"
# In[1]:
import argparse
parser = argparse.ArgumentParser(description='Accept three parameters: output index, simtime(min) and external galactose concentration(mM)')
parser.add_argument('-id', '--index',  type=int, required=True, help='index of the output lm files')
parser.add_argument('-t', '--simtime',  type=float, default=60, help='simulation time')
parser.add_argument('-g', '--galactose',  type=float, default=11.1, help='external galactose concentration')
parser.add_argument('-gpus', '--gpus',  type=int, default=1, help='available gpus to use(default 1, use single gpu)')
parser.add_argument('-tag', '--tag',  type=str, default='', help='tag for the output folder')
parser.add_argument('-geo', '--geometry',  type=str, default='lattice_ER_tunnels_data.pkl.xz', help='geometry file name, default is yeast-lattice.2.pkl.xz')
parser.add_argument('-mt', '--max_time', type=float, default=1000, help='Maximum allowed simulation time in hours')
parser.add_argument('-geloc', '--gene_location', type=str, default='random', help='location of the genes, default is random')
# get the args
args = parser.parse_args()
output_order = args.index
simtime = args.simtime
externalGal_input = args.galactose
gpus = args.gpus
output_tag = args.tag
geometry_file = args.geometry
gene_location = args.gene_location
#get date in format yyyymmdd
import datetime
date = datetime.datetime.now().strftime("%Y%m%d")
if IF_DGX == True:
    if gpus == 1:
        output_folder = "workspace/yeastER" + version + "_" + str(date) + "_" + str(output_order) + "_t"+ str(simtime) + "min"+ "GAE" + str(externalGal_input)+ "mM" + output_tag +".lm"
    else:
        output_folder = "workspace/yeastER" + version + "_multi_" + str(date) + "_" + str(output_order) + "_t"+ str(simtime) + "min"+ "GAE" + str(externalGal_input)+ "mM" + output_tag +"_gpu"+ str(gpus) +".lm"
else:
    if gpus == 1:
        output_folder = "yeastER" + version + "_" + str(date) + "_" + str(output_order) + "_t"+ str(simtime) + "min"+ "GAE" + str(externalGal_input)+ "mM" + output_tag +".lm"
    else:
        output_folder = "yeastER" + version + "_multi_" + str(date) + "_" + str(output_order) + "_t"+ str(simtime) + "min"+ "GAE" + str(externalGal_input)+ "mM" + output_tag +"_gpu"+ str(gpus) +".lm"

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


# In[3]:


if IF_DGX == True:
    latticeData = pickle.load(lzma.open("workspace/"+geometry_file, "rb"))
else:
    latticeData = pickle.load(lzma.open(geometry_file, "rb")) 

siteMap = {n:i for i,n in enumerate(latticeData['names'])}
def boolLattice(x):
    return latticeData['lattice'] == siteMap[x]

extracellular = boolLattice("extracellular") 
cellWall = boolLattice("cellWall")
nuclearEnvelope = boolLattice("nuclearEnvelope")
mitochondria = boolLattice("mitochondria")
vacuole = boolLattice("vacuole")
cytoRibosomes = boolLattice("cytoRibosomes")
membrane = boolLattice("plasmaMembrane")
nucleus = boolLattice("nucleoplasm") | boolLattice("nuclearPores")
cytoplasm = boolLattice("cytoplasm")
endoplasmicReticulum = boolLattice("cecER") | boolLattice("tubER")
pmaER =boolLattice("pmaER")
erRibosomes = boolLattice("pmaRibosomes") | boolLattice("cecRibosomes") | boolLattice("tubRibosomes")
# resolution decrease scale from cryoET
decimation = latticeData['decimation']


# In[4]:


if gpus == 1: 
    siteType = "Int"
else:
    siteType = "Byte"

sim = RDMESim("Galactose switch ER, RDME/ODE hybrid",
              output_folder,
              latticeData['lattice'].shape,
              latticeData['latticeSpacing'],
              "extracellular",siteType)


# In[5]:


B = RegionBuilder(sim)
B.compose((sim.region('extracellular'), extracellular),
          (sim.region('cellWall'), cellWall),
          (sim.region('nuclearEnvelope'), nuclearEnvelope),
          (sim.region('mitochondria'), mitochondria),
          (sim.region('vacuole'), vacuole),
          (sim.region('plasmaMembrane'), membrane),
          (sim.region('cytoplasm'), cytoplasm),
          (sim.region('nucleoplasm'), nucleus),
          (sim.region('cytoRibosomes'), cytoRibosomes),
          (sim.region('endoplasmicReticulum'), endoplasmicReticulum),
          (sim.region('pmaER'), pmaER),
          (sim.region('erRibosomes'), erRibosomes))

sp = sim.sp   # species object access
reg = sim.reg # region object access
rc = sim.rc   # rate constant object access
dc = sim.dc   # diffusion constant object accessb


# In[6]:


# define all necessary sim parameters
sim.simulationTime = simtime * 60 # seconds
sim.timestep = 50e-6 # seconds
# mimic the old way to define the interval
interval = 1.0 # seconds
sim.latticeWriteInterval= int(interval/sim.timestep) # .05s
sim.speciesWriteInterval= int(interval/sim.timestep) # .05s
sim.hookInterval= int(interval/sim.timestep) #.05s

# initial conditions for external galactose 
externalGal = externalGal_input * 1e-3 # M   
# Number of ribosomes in total
ncytoRibosomes = np.sum(sim.siteLattice == reg.cytoRibosomes.idx)
nERribosomes = np.sum(sim.siteLattice == reg.erRibosomes.idx)
nRibosomes = ncytoRibosomes + nERribosomes
# diffusion coefficients for mRNA
# mRNADiffusion = 0.5e-12 # m^2/s
mRNADiffusion = 0.05e-12 # number from sif 
# proteinERDiffusion = 1e-12 # number agree with most exp papers


# In[7]:


# here we define all species needed 
with sim.construct():
    sim.species('DGrep', texRepr='D_{rep}', annotation="Reporter gene (inactive)")
    sim.species('DGrep_G4d', texRepr='D_{rep}{:}G_{4D}', annotation="Reporter gene activated")
    sim.species('DGrep_G4d_G80d', texRepr='D_{rep}{:}G_{4D}{:}G_{80D}', annotation="Reporter gene repressed")
    sim.species('Rrep', texRepr='R_{rep}', annotation="Reporter mRNA")
    sim.species('Grep', texRepr='G_{rep}', annotation="Reporter GFP")


# In[8]:


with sim.construct():
    sim.species('DG1', texRepr='D_{G1}', annotation="Galactose metabolism gene (inactive)")
    sim.species('DG1_G4d', texRepr='D_{G1}{:}G_{4D}', annotation="Galactose metabolism gene activated")
    sim.species('DG1_G4d_G80d', texRepr='D_{G1}{:}G_{4D}{:}G_{80D}', annotation="Galactose metabolism gene repressed")
    sim.species('R1', texRepr='R_{1}', annotation="Galactose metabolism mRNA")
    sim.species('G1', texRepr='G_{1}', annotation="Galactose metabolism protein")


# In[9]:


with sim.construct():
    sim.species('DG2', texRepr='D_{G2}', annotation="Galactose transport gene (inactive)")
    sim.species('DG2_G4d', texRepr='D_{G2}{:}G_{4D}', annotation="Galactose transport gene activated")
    sim.species('DG2_G4d_G80d', texRepr='D_{G2}{:}G_{4D}{:}G_{80D}', annotation="Galactose transport gene repressed")
    sim.species('R2', texRepr='R_{2}', annotation="Galactose transport mRNA")
    sim.species('G2', texRepr='G_{2}', annotation="Galactose transport protein")


# In[10]:


with sim.construct():
    sim.species('DG3', texRepr='D_{G3}', annotation="Gal3 gene (inactive)")
    sim.species('DG3_G4d', texRepr='D_{G3}{:}G_{4D}', annotation="Gal3 gene activated")
    sim.species('DG3_G4d_G80d', texRepr='D_{G3}{:}G_{4D}{:}G_{80D}', annotation="Gal3 gene repressed")
    sim.species('R3', texRepr='R_{3}', annotation="Gal3 mRNA")
    sim.species('G3', texRepr='G_{3}', annotation="Gal3 protein")
    sim.species('G3i', texRepr='G_{3i}', annotation="activated Gal3 bound to galactose ")


# In[11]:


with sim.construct():
    sim.species('DG4', texRepr='D_{G4}', annotation="Gal4 gene (inactive)")
    sim.species('R4', texRepr='R_{4}', annotation="Gal4 mRNA")
    sim.species('G4', texRepr='G_{4}', annotation="Gal4 protein")
    sim.species('G4d', texRepr='G_{4D}', annotation="Gal4 dimer")


# In[12]:


with sim.construct():
    sim.species('DG80', texRepr='D_{G80}', annotation="Gal80 gene (inactive)")
    sim.species('DG80_G4d', texRepr='D_{G80}{:}G_{4D}', annotation="Gal80 gene activated")
    sim.species('DG80_G4d_G80d', texRepr='D_{G80}{:}G_{4D}{:}G_{80D}', annotation="Gal80 gene repressed")
    sim.species('R80', texRepr='R_{80}', annotation="Gal80 mRNA")
    sim.species('G80', texRepr='G_{80}', annotation="Gal80 protein")
    sim.species('G80d', texRepr='G_{80D}', annotation="Gal80 dimer")
    sim.species('G80d_G3i', texRepr='G_{80D}{:}G_{3i}', annotation="Gal80 dimer bound to activated Gal3")


# In[13]:


with sim.construct():
    sim.species('ribosome', texRepr='Ribosome', annotation="Ribosome (inactive)")
    sim.species('ribosomeR1', texRepr='Ribosome{:}R_{1}', annotation="Ribosome bound to Gal1 mRNA")
    sim.species('ribosomeR2', texRepr='Ribosome{:}R_{2}', annotation="Ribosome bound to Gal2 mRNA")
    sim.species('ribosomeR3', texRepr='Ribosome{:}R_{3}', annotation="Ribosome bound to Gal3 mRNA")
    sim.species('ribosomeR4', texRepr='Ribosome{:}R_{4}', annotation="Ribosome bound to Gal4 mRNA")
    sim.species('ribosomeR80', texRepr='Ribosome{:}R_{80}', annotation="Ribosome bound to Gal80 mRNA")
    sim.species('ribosomeGrep', texRepr='Ribosome{:}G_{rep}', annotation="Ribosome bound to reporter mRNA")


# In[14]:


# reactions difined here
cellVol = 3.57e-14 # L, cell size from Ramsey paper SI, haploid yeast
nav = cellVol*6.022e23  # volume times Avogadro's number
invMin2invSec = 1/60.0 # s^-1/min^-1
conv2ndOrder = invMin2invSec*nav # convert to s^-1*M^-1
conv1stOrder = invMin2invSec # convert to s^-1


# In[15]:


with sim.construct():
    sim.rateConst('fd',100 * conv2ndOrder, order= 2, annotation ="Gal4p/Gal80p dimer formation")
    sim.rateConst('rd', 0.001 * conv1stOrder, order= 1, annotation ="Gal4p/Gal80p dimer dissociation")
    sim.reaction([sp.G4, sp.G4], [sp.G4d], rc.fd, annotation ="Gal4p/Gal80p dimer formation", regions=[reg.cytoplasm, reg.nucleoplasm])
    sim.reaction([sp.G4d], [sp.G4, sp.G4], rc.rd, annotation ="Gal4p/Gal80p dimer dissociation", regions=[reg.cytoplasm, reg.nucleoplasm])
    sim.reaction([sp.G80, sp.G80], [sp.G80d], rc.fd, annotation ="Gal80p/Gal80p dimer formation", regions=[reg.cytoplasm, reg.nucleoplasm])
    sim.reaction([sp.G80d], [sp.G80, sp.G80], rc.rd, annotation ="Gal80p/Gal80p dimer dissociation", regions=[reg.cytoplasm, reg.nucleoplasm])


# In[16]:


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
    
    dnas            = [sp.DG1           , sp.DG2        , sp.DG3            ,  sp.DG80          , sp.DGrep]
    # mrnas           = [sp.R1            , sp.R2         , sp.R3,  sp.R80, sp.Rrep]
    dna_gal4        = [sp.DG1_G4d       , sp.DG2_G4d    , sp.DG3_G4d        ,  sp.DG80_G4d      , sp.DGrep_G4d]
    dna_gal4_gal80  = [sp.DG1_G4d_G80d, sp.DG2_G4d_G80d , sp.DG3_G4d_G80d   , sp.DG80_G4d_G80d  , sp.DGrep_G4d_G80d]
    f_g4s           = [rc.f1_4          , rc.f1_5       , rc.f1             , rc.f1             , rc.f1_4]
    r_g4s           = [rc.r1_4          , rc.r1_5       , rc.r1             , rc.r1             , rc.r1_4]
    f_g4g80s        = [rc.f2_4          , rc.f2_5       , rc.f2             , rc.f2             , rc.f2_4]
    r_g4g80s        = [rc.r2_4          , rc.r2_5       , rc.r2             , rc.r2             , rc.r2_4]
    
    for dna, dna_gal4, dna_gal4_gal80, f_g4, r_g4, f_g4g80, r_g4g80 in zip(dnas, dna_gal4, dna_gal4_gal80, f_g4s, r_g4s, f_g4g80s, r_g4g80s):
        sim.reaction([dna, sp.G4d], [dna_gal4], f_g4, annotation="Gal4p binding to gene", regions=reg.nucleoplasm)
        sim.reaction([dna_gal4], [dna, sp.G4d], r_g4, annotation="Gal4p dissociation from gene", regions=reg.nucleoplasm)
        # this is modified by reducing the binding rate by 10^2 times, to increase the time it exits in state DG_G4d for transcription.
        sim.reaction([dna_gal4, sp.G80d], [dna_gal4_gal80], f_g4g80, annotation="Gal80p binding to gene", regions=reg.nucleoplasm)
        sim.reaction([dna_gal4_gal80], [dna_gal4, sp.G80d], r_g4g80, annotation="Gal80p dissociation from gene", regions=reg.nucleoplasm)
    


# In[17]:


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


# In[18]:


# ## Transcription
with sim.construct():
    sim.rateConst("alpha1", 0.7379*conv1stOrder, order=1, annotation='GAL1 transcription')
    sim.rateConst("alpha2", 2.542*conv1stOrder, order=1, annotation='GAL2 transcription')
    sim.rateConst("alpha3", 0.571429*0.7465*conv1stOrder, order=1, annotation='GAL3 transcription')
    sim.rateConst("ir_gal4", 0.009902*conv1stOrder, order=1, annotation='GAL4 transcription')
    sim.rateConst("alpha_rep", 1.1440*conv1stOrder, order=1, annotation='GFP transcription')
    sim.rateConst("alpha80", 0.6065*conv1stOrder, order=1, annotation='GAL80 transcription')

    sim.rateConst("dr_gal1", 0.02236*conv1stOrder, order=1, annotation='GAL1 mRNA degradation')
    sim.rateConst("dr_gal2", 0.07702*conv1stOrder, order=1, annotation='GAL2 mRNA degradation')
    sim.rateConst("dr_gal3", 0.02666*conv1stOrder, order=1, annotation='GAL3 mRNA degradation')
    sim.rateConst("dr_gal4", 0.02476*conv1stOrder, order=1, annotation='GAL4 mRNA degradation')
    sim.rateConst("dr_rep", 0.03466*conv1stOrder, order=1, annotation='GFP mRNA degradation')
    sim.rateConst("dr_gal80", 0.02888*conv1stOrder, order=1, annotation='GAL80 mRNA degradation')
    
    transcription_rates = [rc.alpha1,   rc.alpha2,  rc.alpha3,  rc.ir_gal4, rc.alpha_rep,   rc.alpha80]
    decay_rates         = [rc.dr_gal1,  rc.dr_gal2, rc.dr_gal3, rc.dr_gal4, rc.dr_rep,      rc.dr_gal80]
    genes               = [sp.DG1_G4d,  sp.DG2_G4d, sp.DG3_G4d, sp.DG4,     sp.DGrep_G4d,   sp.DG80_G4d]
    mrnas               = [sp.R1,       sp.R2,      sp.R3,      sp.R4,      sp.Rrep,        sp.R80]
    
    for trans_rate, decay_rate, gene, mrna in zip(transcription_rates, decay_rates, genes, mrnas):
        sim.reaction([gene], [gene, mrna], trans_rate, regions=reg.nucleoplasm)
        sim.reaction([mrna], [], decay_rate, regions=[reg.nucleoplasm, reg.cytoplasm, reg.cytoRibosomes, reg.erRibosomes])


# In[19]:


# translation 
## we need to consider both cytoplasm and ER ribosomes
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
        sim.reaction([sp.ribosome, mrna], [translatingRibosomes], rc.rib_assoc, regions=[reg.cytoplasm, reg.erRibosomes])
        #translation
        sim.reaction([translatingRibosomes], [sp.ribosome, mrna, protein], ktl, regions=[reg.cytoplasm, reg.erRibosomes])
        #degradation in association form
        sim.reaction([translatingRibosomes], [sp.ribosome], mdcy, regions=[reg.cytoplasm, reg.erRibosomes])
    


# In[20]:


## protein degradation
with sim.construct():
    deg_compartments =         [[reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm],  # G1
                                [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm, reg.plasmaMembrane],  # G2
                                [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm],  # G3
                                [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm, reg.nucleoplasm], 
                                [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm], 
                                [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm, reg.nucleoplasm]]
    for protein, decay_rate, region in zip(prots, dcys, deg_compartments):
        sim.reaction([protein], [], decay_rate, regions=region)


# ## inital counts

# In[21]:


if IF_DGX == True:
    initMolec = pickle.load(open("/workspace/ysZeroGAE.pkl", "rb"))
else:
    initMolec = pickle.load(open("ysZeroGAE.pkl", "rb"))
# change molecular/unit volume to actual molecules
volScale = np.sum(B.convexHull(sim.siteLattice==reg.plasmaMembrane.idx))*sim.siteV/cellVol
def initMolecules(x):
    # convert molecules/unit volume to molecues
    counts = int(initMolec[x]*volScale)
    return counts


# In[22]:


if gene_location == "random":
    print("gene location random")
    for b in ["DG1", "DG2", "DG3", "DG80", "DGrep"]:
        ops = [b+x for x in ["", "_G4d", "_G4d_G80d"]]
        spName = max(ops, key=lambda x:initMolec[x])
        print("{} in state {}".format(b, spName))
        sim.species(spName).placeNumberInto(reg.nucleoplasm, 1)
    # change the dafult way of placing all genes into activated state
    # sp.DG1_G4d.placeNumberInto(reg.nucleoplasm, 1)
    # print("{} in state {}".format("Gene1", "DG1_G4d"))
    # sp.DG2_G4d.placeNumberInto(reg.nucleoplasm, 1)
    # print("{} in state {}".format("Gene2", "DG2_G4d"))  
    # sp.DG3_G4d.placeNumberInto(reg.nucleoplasm, 1)
    # print("{} in state {}".format("Gene3", "DG3_G4d"))
    # sp.DG80_G4d.placeNumberInto(reg.nucleoplasm, 1)
    # print("{} in state {}".format("Gene80", "DG80_G4d"))
    # sp.DGrep_G4d.placeNumberInto(reg.nucleoplasm, 1)
    # print("{} in state {}".format("GeneRep", "DGrep_G4d"))
    # this is the original code 
    sp.DG4.placeNumberInto(reg.nucleoplasm, 1)
    print("{} in state {}".format("Gene4", "DG4"))
else:
    print("gene location fixed")
    # dna_gal4_gal80  = [sp.DG1_G4d_G80d, sp.DG2_G4d_G80d , sp.DG3_G4d_G80d   , sp.DG80_G4d_G80d  , sp.DGrep_G4d_G80d]
    sim.placeNumber(sp=sp.DGrep_G4d_G80d, x=133, y=86, z=117, n=1)
    sim.placeNumber(sp=sp.DG1_G4d_G80d, x=90, y=90, z=132, n=1)
    sim.placeNumber(sp=sp.DG2_G4d_G80d, x=132, y=73, z=116, n=1)
    sim.placeNumber(sp=sp.DG3_G4d_G80d, x=100, y=58, z=111, n=1)
    sim.placeNumber(sp=sp.DG80_G4d_G80d, x=115, y=94, z=132, n=1)
    print("{} in state {}".format("GeneRep", "DGrep_G4d_G80d"))
    print("{} in state {}".format("Gene1", "DG1_G4d_G80d"))
    print("{} in state {}".format("Gene2", "DG2_G4d_G80d"))
    print("{} in state {}".format("Gene3", "DG3_G4d_G80d"))
    print("{} in state {}".format("Gene80", "DG80_G4d_G80d"))
    sim.placeNumber(sp=sp.DG4, x=126, y=61, z=115, n=1)
    print("{} in state {}".format("Gene4", "DG4"))


# In[23]:


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

# place G80
# G80 is placed in the cytoplasm and nucleoplasm, separated by relative volume
cscl = reg.cytoplasm.volume/(reg.cytoplasm.volume+reg.nucleoplasm.volume)
totM = initMolecules("G80C") + initMolecules("G80")
totD = initMolecules("G80Cd") + initMolecules("G80d")
sp.G80.placeNumberInto(reg.cytoplasm, int(cscl*totM))
sp.G80.placeNumberInto(reg.nucleoplasm, int((1-cscl)*totM))
sp.G80d.placeNumberInto(reg.cytoplasm, int(cscl*totD))
sp.G80d.placeNumberInto(reg.nucleoplasm, int((1-cscl)*totD))
print("G80 in cytoplasm: {}, in nucleoplasm: {}".format(int(cscl*totM), int((1-cscl)*totM)))
print("G80d in cytoplasm: {}, in nucleoplasm: {}".format(int(cscl*totD), int((1-cscl)*totD)))


# In[24]:


for x, y, z in np.argwhere(sim.siteLattice == reg.cytoRibosomes.idx):
    sp.ribosome.placeParticle(x, y, z, 1)

for x, y, z in np.argwhere(sim.siteLattice == reg.erRibosomes.idx):
    sp.ribosome.placeParticle(x, y, z, 1)

print("ribosomes number:", np.sum(sim.siteLattice == reg.cytoRibosomes.idx) + np.sum(sim.siteLattice == reg.erRibosomes.idx))



# ## diffusion

# In[28]:


reg


# In[26]:


sim.transitionRate(None, None, None, sim.diffusionZero)


# In[27]:


#genes
for sps in sim.speciesList.matchRegex("D.*"):
        sps.diffusionRate(None, sim.diffusionZero)


# In[29]:


#mRNAs
# 1. we allow all mRNAs to diffuse into cytoplasm, cytoRibosomes and get translated
#  this included R2 

sim.diffusionConst("mrna", mRNADiffusion, texRepr=r'D_{mRNA}', annotation='Generic mRNA')

for mrna in sim.speciesList.matchRegex("R.*"):
    #m RNA out of nucleoplasm
    sim.transitionRate(mrna, reg.nucleoplasm, reg.cytoplasm, dc.mrna)
    sim.transitionRate(mrna, reg.cytoplasm, reg.nucleoplasm, sim.diffusionZero)
    sim.transitionRate(mrna, reg.nucleoplasm, reg.nucleoplasm, dc.mrna)
    sim.transitionRate(mrna, reg.cytoplasm, reg.cytoplasm, dc.mrna)
    # in ribosomes
    sim.transitionRate(mrna, reg.cytoRibosomes, reg.cytoRibosomes, dc.mrna)
    sim.transitionRate(mrna, reg.cytoRibosomes, reg.cytoplasm, dc.mrna)
    sim.transitionRate(mrna, reg.cytoplasm, reg.cytoRibosomes, dc.mrna)
    
# 2. we only allow R2 to diffuse into erRibosomes
sim.transitionRate(sp.R2, reg.cytoplasm, reg.erRibosomes, dc.mrna)
sim.transitionRate(sp.R2, reg.erRibosomes, reg.cytoplasm, dc.mrna)
sim.transitionRate(sp.R2, reg.erRibosomes, reg.erRibosomes, dc.mrna)
# dont allow R2 to diffuse into ER




# In[31]:


## Proteins 
# if G2 get translated from cytoRibosomes, it will be in cytoplasm
sim.diffusionConst("prot", 1e-12, texRepr=r'D_{prot}', annotation='Generic protein')
sim.diffusionConst("ribo", 3e-13, texRepr=r'D_{ribosome}', annotation='Generic ribosome')
for sps in [sp.G1, sp.G2, sp.G3, sp.G3i, sp.G4, sp.G4d, sp.G80, sp.G80d, sp.G80d_G3i, sp.Grep]:
    # allow diffusion in the cytoplasm
    sim.transitionRate(sps, reg.cytoplasm, reg.cytoplasm, dc.prot)
    # we allow proteins generated in cytoRibosomes to diffuse into both cytoplasm and ER
    sim.transitionRate(sps, reg.cytoRibosomes, reg.cytoplasm, sim.diffusionFast)
    
    sim.transitionRate(sps, reg.cytoplasm, reg.cytoRibosomes, sim.diffusionZero)


# we only allow R2 to diffuse through ER membrane as membrane protein
sim.transitionRate(sp.G2, reg.endoplasmicReticulum, reg.endoplasmicReticulum, sim.diffusionFast)
# mimic the diffusion of R2-ribosome complex to rought ER
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3694300/ 
sim.transitionRate(sp.G2, reg.cytoRibosomes, reg.endoplasmicReticulum, dc.ribo)

# erRibosomes, only allow protein to diffuse into ER
sim.transitionRate(sp.G2, reg.erRibosomes, reg.endoplasmicReticulum, sim.diffusionFast)
sim.transitionRate(sp.G2, reg.endoplasmicReticulum, reg.erRibosomes, sim.diffusionZero)
sim.transitionRate(sp.G2, reg.endoplasmicReticulum, reg.cytoRibosomes, sim.diffusionZero)
# since we separate the pmaER and rest of ER, we need to allow G2 diffuse across ERs
# and further allow it to get diffuse out of ER
sim.transitionRate(sp.G2, reg.endoplasmicReticulum, reg.pmaER, sim.diffusionFast)
sim.transitionRate(sp.G2, reg.pmaER, reg.endoplasmicReticulum, sim.diffusionZero)
sim.transitionRate(sp.G2, reg.pmaER, reg.cytoplasm, sim.diffusionFast)
# since we dont have vesicle to carry G2 from pmaER to membrane, we just allow it to diffuse
# back and forth right now, in case it get out of in the side of pmaER far from plasma membrane
sim.transitionRate(sp.G2, reg.cytoplasm, reg.pmaER, sim.diffusionFast)



# In[32]:


# TF
for sps in [sp.G4, sp.G4d, sp.G80, sp.G80d]:
    sim.transitionRate(sps, reg.nucleoplasm, reg.nucleoplasm, dc.prot)
    sim.transitionRate(sps, reg.nucleoplasm, reg.cytoplasm, dc.prot)
    sim.transitionRate(sps, reg.cytoplasm, reg.nucleoplasm, dc.prot)


# In[33]:


# non-TF  protein, dont allow them diffuse into nucleoplasm
for sps in [sp.G1, sp.G2, sp.G3, sp.G3i, sp.G80d_G3i, sp.Grep]:
    sim.transitionRate(sps, reg.cytoplasm, reg.nucleoplasm, sim.diffusionZero) 


# In[34]:


# membrane protein
sim.transitionRate(sp.G2, reg.cytoplasm, reg.plasmaMembrane, dc.prot)
sim.transitionRate(sp.G2, reg.pmaER, reg.plasmaMembrane, dc.prot)
sim.transitionRate(sp.G2, reg.plasmaMembrane, reg.cytoplasm, sim.diffusionZero)
sim.diffusionConst("mem", 0.01e-12, texRepr=r'D_{mem}', annotation='Generic protein on membrane')

# sp.G2.diffusionRate(reg.plasmaMembrane, dc.mem)
sim.transitionRate(sp.G2, reg.plasmaMembrane, reg.plasmaMembrane, dc.mem)


# In[35]:


for sps in sim.speciesList.matchRegex("ribosome.*"):
    sim.transitionRate(sps, None, None, sim.diffusionZero)


# ## RMDE-ODE hybrid

# In[ ]:


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
        # initialization will count all G1 and G2 in ODE
        y = np.zeros(len(self.odeSpNames))
        y[self.odeSpIndex("GAI")] = 0
        y[self.odeSpIndex("G1")] = cts['countBySpecies'][self.rdme.sp.G1]/self.NAV 
        y[self.odeSpIndex("G1GAI")] = 0
        y[self.odeSpIndex("G2")] = cts['countBySpecies'][self.rdme.sp.G2]/self.NAV
        y[self.odeSpIndex("G2GAE")] = 0
        y[self.odeSpIndex("G2GAI")] = 0
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

