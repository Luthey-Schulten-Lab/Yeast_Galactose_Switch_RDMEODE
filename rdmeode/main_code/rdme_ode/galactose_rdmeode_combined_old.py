#!/usr/bin/env python
# coding: utf-8
'''Combined simulation file by merging galactose_rdmeode1.15_multi.py and galactoseER_rdmeode1.13.py
Version: 1.16_combined
Features:
- Optional chromosome support (--enable-chromosome)
- Optional ER support (--enable-er) 
- Gene location options (random, center, edge, chromosome)
- Checkpoint support
- Multi-GPU support
- Memory monitoring
- Flexible geometry handling
'''

import time
import signal
import sys
import os
start_time = time.time()

IF_DGX = False
version = "1.16_combined"

import argparse
parser = argparse.ArgumentParser(description='Combined RDME/ODE simulation with optional chromosome and ER support')
parser.add_argument('-id', '--index',  type=int, required=True, help='index of the output lm files')
parser.add_argument('-t', '--simtime',  type=float, default=60, help='simulation time')
parser.add_argument('-g', '--galactose',  type=float, default=11.1, help='external galactose concentration')
parser.add_argument('-gpus', '--gpus',  type=int, default=1, help='available gpus to use(default 1, use single gpu)')
parser.add_argument('-tag', '--tag',  type=str, default='', help='tag for the output folder')
parser.add_argument('-geo', '--geometry',  type=str, default='yeast-lattice.2.pkl.xz', help='geometry file name')
parser.add_argument('-mt', '--max_time', type=float, default=1000, help='Maximum allowed simulation time in hours')
parser.add_argument('-geloc', '--gene_location', type=str, default='random', help='location of the genes (random, center, edge, chromosome)')
parser.add_argument('-ckpt', '--checkpoint', type=str, default='', help='checkpoint file name, default is empty')
parser.add_argument('--enable-chromosome', action='store_true', help='Enable chromosome regions and related functionality')
parser.add_argument('--enable-er', action='store_true', help='Enable ER regions and related functionality')
parser.add_argument('--enable-effective-ribosome', action='store_true', help='Enable effective ribosome case (includes both ribosome and ribosome_dummy)')

args = parser.parse_args()
output_order = args.index
simtime = args.simtime
externalGal_input = args.galactose
gpus = args.gpus
output_tag = args.tag
geometry_file = args.geometry
gene_location = args.gene_location
checkpoint_file = args.checkpoint
enable_chromosome = args.enable_chromosome
enable_er = args.enable_er
enable_effective_ribosome = args.enable_effective_ribosome

# Auto-detect features from geometry file if not explicitly set
if not enable_chromosome and not enable_er:
    if 'ER' in geometry_file or 'er' in geometry_file.lower():
        enable_er = True
        print("Auto-detected ER support from geometry filename")
    if 'chromosome' in geometry_file.lower():
        enable_chromosome = True
        print("Auto-detected chromosome support from geometry filename")
if enable_chromosome:
    gene_location = "chromosome"
if enable_er:
    if "lattice_ER_tunnels" not in geometry_file:
        raise ValueError("ER support requires lattice_ER_tunnels... geometry file")
print(f"Chromosome support: {enable_chromosome}")
print(f"ER support: {enable_er}")
print(f"Effective ribosome support: {enable_effective_ribosome}")

import datetime
date = datetime.datetime.now().strftime("%Y%m%d")

# Handle checkpoint directory
if checkpoint_file:
    output_dir = os.path.dirname(checkpoint_file)
    print(f"Using directory from checkpoint file: {output_dir}")
else:
    output_dir = "simulation_results_id_" + str(output_order)

# Create output folder name
feature_suffix = ""
if enable_er:
    feature_suffix += "_ER"
if enable_chromosome:
    feature_suffix += "_CHROMO"
if enable_effective_ribosome:
    feature_suffix += "_EFFRIBO"
if IF_DGX:
    dir_dgx = "workspace/"
else:
    dir_dgx = ""

if IF_DGX:
    base_name = f"workspace/yeast{version}_{date}_{output_order}_t{simtime}min_GAE{externalGal_input}mM{feature_suffix}{output_tag}"
    if gpus > 1:
        base_name += f"_gpu{gpus}"
    output_folder = base_name + ".lm"
    if not os.path.exists(os.path.join("workspace/", output_dir)):
        os.makedirs(os.path.join("workspace/", output_dir))
else:
    base_name = f"yeast{version}_{date}_{output_order}_t{simtime}min_GAE{externalGal_input}mM{feature_suffix}{output_tag}"
    if gpus > 1:
        base_name += f"_gpu{gpus}"
    output_folder = base_name + ".lm"
    output_folder = os.path.join(output_dir, output_folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

print("output_folder: ", output_folder)
print("simtime: ", simtime)
print("geometry_file: ", geometry_file)

import pickle, lzma
import numpy as np
import scipy.integrate as spint
from jLM.Solvers import ConstBoundaryConc, makeSolver
from lm import MGPUMpdRdmeSolver, MpdRdmeSolver, IntMpdRdmeSolver
from jLM.RDME import Sim as RDMESim
from jLM.RDME import File as RDMEFile
from jLM.RegionBuilder import RegionBuilder
import jLM
import psutil

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024} MB")

# Load lattice data
latticeData = pickle.load(lzma.open(dir_dgx + geometry_file, "rb"))


siteMap = {n: i for i, n in enumerate(latticeData['names'])}
def boolLattice(x):
    return latticeData['lattice'] == siteMap[x]

# Define common regions
extracellular = boolLattice("extracellular")
cellWall = boolLattice("cellWall")
nuclearEnvelope = boolLattice("nuclearEnvelope") 
mitochondria = boolLattice("mitochondria")
vacuole = boolLattice("vacuole")
membrane = boolLattice("plasmaMembrane")
nucleus = boolLattice("nucleoplasm") | boolLattice("nuclearPores")
cytoplasm = boolLattice("cytoplasm")

# Conditional region definitions based on enabled features
if enable_chromosome:

    chromo_gene = np.load(dir_dgx + "gene_masks.npy").astype(bool)
    chromo_dummy = np.load(dir_dgx + "dummy_chromosome.npy").astype(bool)

if enable_er:
    # Handle effective ribosome case for ER
    if enable_effective_ribosome:
        print('Loading effective ribosome regions for ER case')
        effective_cytoRibosomes = np.load(dir_dgx + "effective_cyto_ribosomes_ER_Marie.npy").astype(bool)
        dummy_cytoRibosomoe = np.load(dir_dgx + "dummy_cyto_ribosomes_ER_Marie.npy").astype(bool)
        effective_erRibosomes = np.load(dir_dgx + "effective_er_ribosomes_ER_Marie.npy").astype(bool)
        dummy_erRibosomes = np.load(dir_dgx + "dummy_er_ribosomes_ER_Marie.npy").astype(bool)
    else: 
        erRibosomes = boolLattice("pmaRibosomes") | boolLattice("cecRibosomes") | boolLattice("tubRibosomes")
        cytoRibosomes = boolLattice("cytoRibosomes")
    # enable ER then, to overlap
  
    pmaER = boolLattice("pmaER")
    endoplasmicReticulum = boolLattice("cecER") | boolLattice("tubER")
    
else:
    if enable_effective_ribosome:
        print('Loading effective ribosome regions for no ER case')
        ribosome_dummy = np.load(dir_dgx + "dummy_ribosomes_noER.npy").astype(bool)
        ribosome = np.load(dir_dgx + "effective_ribosomes_noER.npy").astype(bool)
    else:
        ribosomes = boolLattice("ribosomes")
        
    


decimation = latticeData['decimation']

# Simulation setup
if gpus == 1:
    siteType = "Int"
else:
    siteType = "Byte"

sim_title = "Galactose switch"
if enable_er:
    sim_title += " ER"
if enable_chromosome:
    sim_title += " chromosome"
if enable_effective_ribosome:
    sim_title += " effective-ribosome"
sim_title += ", RDME/ODE hybrid"

sim = RDMESim(sim_title, output_folder, latticeData['lattice'].shape, 
              latticeData['latticeSpacing'], "extracellular", siteType)

print("the shape of the lattice is: ", latticeData['lattice'].shape)

# Region composition
B = RegionBuilder(sim)

# Base regions
regions_to_compose = [
    (sim.region('extracellular'), extracellular),
    (sim.region('cellWall'), cellWall),
    (sim.region('nuclearEnvelope'), nuclearEnvelope),
    (sim.region('mitochondria'), mitochondria),
    (sim.region('vacuole'), vacuole),
    (sim.region('plasmaMembrane'), membrane),
    (sim.region('cytoplasm'), cytoplasm),
    
]

# Add ribosome regions based on features
if enable_er:
    
    if enable_effective_ribosome:
        regions_to_compose.extend([
            (sim.region('dum_cytoRibosomes'), dummy_cytoRibosomoe),
            (sim.region('dum_erRibosomes'), dummy_erRibosomes),
            (sim.region('cytoRibosomes'), effective_cytoRibosomes),
            (sim.region('erRibosomes'), effective_erRibosomes),
        ])
    else: 
        regions_to_compose.extend([
        (sim.region('cytoRibosomes'), cytoRibosomes),
        (sim.region('erRibosomes'), erRibosomes),
        ])
    regions_to_compose.extend([
        (sim.region('endoplasmicReticulum'), endoplasmicReticulum),
        (sim.region('pmaER'), pmaER)
    ])
else:
    regions_to_compose.append((sim.region('ribosomes'), ribosomes))
    # Add dummy ribosomes if present and not using effective ribosome
    if enable_effective_ribosome:
        regions_to_compose.append((sim.region('ribodummy'), ribosome_dummy))

# Add chromosome regions if enabled
if enable_chromosome:
    nucleus_no_chromo = nucleus & ~(chromo_gene | chromo_dummy)
    regions_to_compose.extend([
        (sim.region('chromosome'), chromo_gene),
        (sim.region('chromo_dummy'), chromo_dummy),
        (sim.region('nucleoplasm'), nucleus_no_chromo),
    ])
else:
    regions_to_compose.append((sim.region('nucleoplasm'), nucleus))
    
B.compose(*regions_to_compose)

# Object access shortcuts
sp = sim.sp
reg = sim.reg
rc = sim.rc
dc = sim.dc

# Simulation parameters
sim.simulationTime = simtime * 60  # seconds
sim.timestep = 50e-6  # seconds
hook_interval = 1  # seconds
write_interval = 1  # seconds
sim.latticeWriteInterval = int(write_interval/sim.timestep)
sim.speciesWriteInterval = int(write_interval/sim.timestep)
sim.hookInterval = int(hook_interval/sim.timestep)

# Initial conditions
externalGal = externalGal_input * 1e-3  # M

# Calculate ribosome numbers
if enable_er:
    ncytoRibosomes = np.sum(sim.siteLattice == reg.cytoRibosomes.idx)
    nERribosomes = np.sum(sim.siteLattice == reg.erRibosomes.idx)
    nRibosomes = ncytoRibosomes + nERribosomes
else:
    nRibosomes = np.sum(sim.siteLattice == reg.ribosomes.idx)

mRNADiffusion = 0.05e-12  # m^2/s

# Species Definitions
with sim.construct():
    # Reporter GFP
    sim.species('DGrep', texRepr='D_{rep}', annotation="Reporter gene (inactive)")
    sim.species('DGrep_G4d', texRepr='D_{rep}{:}G_{4D}', annotation="Reporter gene activated")
    sim.species('DGrep_G4d_G80d', texRepr='D_{rep}{:}G_{4D}{:}G_{80D}', annotation="Reporter gene repressed")
    sim.species('Rrep', texRepr='R_{rep}', annotation="Reporter mRNA")
    sim.species('Grep', texRepr='G_{rep}', annotation="Reporter GFP")

    # GAL1 (G1)
    sim.species('DG1', texRepr='D_{G1}', annotation="Galactose metabolism gene (inactive)")
    sim.species('DG1_G4d', texRepr='D_{G1}{:}G_{4D}', annotation="Galactose metabolism gene activated")
    sim.species('DG1_G4d_G80d', texRepr='D_{G1}{:}G_{4D}{:}G_{80D}', annotation="Galactose metabolism gene repressed")
    sim.species('R1', texRepr='R_{1}', annotation="Galactose metabolism mRNA")
    sim.species('G1', texRepr='G_{1}', annotation="Galactose metabolism protein")

    # GAL2 (G2)
    sim.species('DG2', texRepr='D_{G2}', annotation="Galactose transport gene (inactive)")
    sim.species('DG2_G4d', texRepr='D_{G2}{:}G_{4D}', annotation="Galactose transport gene activated")
    sim.species('DG2_G4d_G80d', texRepr='D_{G2}{:}G_{4D}{:}G_{80D}', annotation="Galactose transport gene repressed")
    sim.species('R2', texRepr='R_{2}', annotation="Galactose transport mRNA")
    sim.species('G2', texRepr='G_{2}', annotation="Galactose transport protein")

    # GAL3 (G3)
    sim.species('DG3', texRepr='D_{G3}', annotation="Gal3 gene (inactive)")
    sim.species('DG3_G4d', texRepr='D_{G3}{:}G_{4D}', annotation="Gal3 gene activated")
    sim.species('DG3_G4d_G80d', texRepr='D_{G3}{:}G_{4D}{:}G_{80D}', annotation="Gal3 gene repressed")
    sim.species('R3', texRepr='R_{3}', annotation="Gal3 mRNA")
    sim.species('G3', texRepr='G_{3}', annotation="Gal3 protein")
    sim.species('G3i', texRepr='G_{3i}', annotation="activated Gal3 bound to galactose")

    # GAL4 (G4)
    sim.species('DG4', texRepr='D_{G4}', annotation="Gal4 gene (inactive)")
    sim.species('R4', texRepr='R_{4}', annotation="Gal4 mRNA")
    sim.species('G4', texRepr='G_{4}', annotation="Gal4 protein")
    sim.species('G4d', texRepr='G_{4D}', annotation="Gal4 dimer")

    # GAL80 (G80)
    sim.species('DG80', texRepr='D_{G80}', annotation="Gal80 gene (inactive)")
    sim.species('DG80_G4d', texRepr='D_{G80}{:}G_{4D}', annotation="Gal80 gene activated")
    sim.species('DG80_G4d_G80d', texRepr='D_{G80}{:}G_{4D}{:}G_{80D}', annotation="Gal80 gene repressed")
    sim.species('R80', texRepr='R_{80}', annotation="Gal80 mRNA")
    sim.species('G80', texRepr='G_{80}', annotation="Gal80 protein")
    sim.species('G80d', texRepr='G_{80D}', annotation="Gal80 dimer")
    sim.species('G80d_G3i', texRepr='G_{80D}{:}G_{3i}', annotation="Gal80 dimer bound to activated Gal3")

    # Ribosomes
    sim.species('ribosome', texRepr='Ribosome', annotation="Ribosome (inactive)")
    sim.species('ribosomeR1', texRepr='Ribosome{:}R_{1}', annotation="Ribosome bound to Gal1 mRNA")
    sim.species('ribosomeR2', texRepr='Ribosome{:}R_{2}', annotation="Ribosome bound to Gal2 mRNA")
    sim.species('ribosomeR3', texRepr='Ribosome{:}R_{3}', annotation="Ribosome bound to Gal3 mRNA")
    sim.species('ribosomeR4', texRepr='Ribosome{:}R_{4}', annotation="Ribosome bound to Gal4 mRNA")
    sim.species('ribosomeR80', texRepr='Ribosome{:}R_{80}', annotation="Ribosome bound to Gal80 mRNA")
    sim.species('ribosomeGrep', texRepr='Ribosome{:}G_{rep}', annotation="Ribosome bound to reporter mRNA")

# Reactions
cellVol = 3.57e-14  # L, cell size from Ramsey paper SI, haploid yeast
nav = cellVol * 6.022e23
invMin2invSec = 1/60.0
conv2ndOrder = invMin2invSec * nav
conv1stOrder = invMin2invSec

# Dimerization
with sim.construct():
    sim.rateConst('fd', 100 * conv2ndOrder, order=2, annotation="Gal4p/Gal80p dimer formation")
    sim.rateConst('rd', 0.001 * conv1stOrder, order=1, annotation="Gal4p/Gal80p dimer dissociation")
    sim.reaction([sp.G4, sp.G4], [sp.G4d], rc.fd, annotation="Gal4p/Gal80p dimer formation", regions=[reg.cytoplasm, reg.nucleoplasm])
    sim.reaction([sp.G4d], [sp.G4, sp.G4], rc.rd, annotation="Gal4p/Gal80p dimer dissociation", regions=[reg.cytoplasm, reg.nucleoplasm])
    sim.reaction([sp.G80, sp.G80], [sp.G80d], rc.fd, annotation="Gal80p/Gal80p dimer formation", regions=[reg.cytoplasm, reg.nucleoplasm])
    sim.reaction([sp.G80d], [sp.G80, sp.G80], rc.rd, annotation="Gal80p/Gal80p dimer dissociation", regions=[reg.cytoplasm, reg.nucleoplasm])

# DNA regulation
with sim.construct():
    Kp4 = 0.2600  # 4 binding sites
    Kq4 = 1.1721
    kf1_4 = 0.1
    kf2_4 = 0.1
    kr1_4 = kf1_4/Kp4
    kr2_4 = kf2_4/Kq4

    Kp5 = 0.0099  # 5 binding sites
    Kq5 = 0.7408
    kf1_5 = 0.1
    kf2_5 = 0.1
    kr1_5 = kf1_5/Kp5
    kr2_5 = kf2_5/Kq5

    Kp = 0.0248  # 1 binding site
    Kq = 0.1885
    kf1 = 0.1
    kr1 = kf1/Kp
    kf2 = 0.1
    kr2 = kf2/Kq

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

    # G1, Grep has 4 sites; G2 has 5 sites; G3, G80 have 1 site
    dnas = [sp.DG1, sp.DG2, sp.DG3, sp.DG80, sp.DGrep]
    dna_gal4 = [sp.DG1_G4d, sp.DG2_G4d, sp.DG3_G4d, sp.DG80_G4d, sp.DGrep_G4d]
    dna_gal4_gal80 = [sp.DG1_G4d_G80d, sp.DG2_G4d_G80d, sp.DG3_G4d_G80d, sp.DG80_G4d_G80d, sp.DGrep_G4d_G80d]
    f_g4s = [rc.f1_4, rc.f1_5, rc.f1, rc.f1, rc.f1_4]
    r_g4s = [rc.r1_4, rc.r1_5, rc.r1, rc.r1, rc.r1_4]
    f_g4g80s = [rc.f2_4, rc.f2_5, rc.f2, rc.f2, rc.f2_4]
    r_g4g80s = [rc.r2_4, rc.r2_5, rc.r2, rc.r2, rc.r2_4]

    for dna, dna_gal4, dna_gal4_gal80, f_g4, r_g4, f_g4g80, r_g4g80 in zip(dnas, dna_gal4, dna_gal4_gal80, f_g4s, r_g4s, f_g4g80s, r_g4g80s):
        sim.reaction([dna, sp.G4d], [dna_gal4], f_g4, annotation="Gal4p binding to gene", regions=[reg.nucleoplasm, reg.chromosome])
        sim.reaction([dna_gal4], [dna, sp.G4d], r_g4, annotation="Gal4p dissociation from gene", regions=[reg.nucleoplasm, reg.chromosome])
        sim.reaction([dna_gal4, sp.G80d], [dna_gal4_gal80], f_g4g80, annotation="Gal80p binding to gene", regions=[reg.nucleoplasm, reg.chromosome])
        sim.reaction([dna_gal4_gal80], [dna_gal4, sp.G80d], r_g4g80, annotation="Gal80p dissociation from gene", regions=[reg.nucleoplasm, reg.chromosome])

# G3 activation
with sim.construct():
    sim.rateConst("fi", 7.45e-7*conv2ndOrder, order=1, annotation="Gal3p activation")
    sim.rateConst("ri", 890.0*conv1stOrder, order=1, annotation="Gal3p deactivation")
    sim.rateConst("fd3i80", 0.025716*conv2ndOrder, order=2, annotation="Gal3p*/Gal80 association")
    sim.rateConst("dr3i80", 0.0159616*conv1stOrder, order=1, annotation="Gal3p*/Gal80 disassociation")
    sim.rateConst("dp_gal3", 0.01155*conv1stOrder, order=1, annotation="GAL3 degradation")
    sim.rateConst("dp_gal3gal80", 0.5*rc.dp_gal3.value, order=1, annotation="Gal3p*:Gal80 degradation")

    sim.reaction(sp.G3, sp.G3i, rc.fi, annotation="Gal3p activation", regions=reg.cytoplasm)
    sim.reaction(sp.G3i, sp.G3, rc.ri, annotation="Gal3p deactivation", regions=reg.cytoplasm)
    sim.reaction([sp.G3i, sp.G80d], [sp.G80d_G3i], rc.fd3i80, annotation="Gal3p*/Gal80 association", regions=reg.cytoplasm)
    sim.reaction([sp.G80d_G3i], [sp.G3i, sp.G80d], rc.dr3i80, annotation="Gal3p*/Gal80 disassociation", regions=reg.cytoplasm)
    sim.reaction(sp.G3i, [], rc.dp_gal3, annotation="GAL3 degradation", regions=reg.cytoplasm)
    sim.reaction(sp.G80d_G3i, [], rc.dp_gal3gal80, annotation="Gal3p*:Gal80 degradation", regions=reg.cytoplasm)

# Transcription
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

    transcription_rates = [rc.alpha1, rc.alpha2, rc.alpha3, rc.ir_gal4, rc.alpha_rep, rc.alpha80]
    decay_rates = [rc.dr_gal1, rc.dr_gal2, rc.dr_gal3, rc.dr_gal4, rc.dr_rep, rc.dr_gal80]
    genes = [sp.DG1_G4d, sp.DG2_G4d, sp.DG3_G4d, sp.DG4, sp.DGrep_G4d, sp.DG80_G4d]
    mrnas = [sp.R1, sp.R2, sp.R3, sp.R4, sp.Rrep, sp.R80]

    # Define regions for mRNA degradation based on enabled features
    if enable_er:
        mrna_regions = [reg.nucleoplasm, reg.cytoplasm, reg.cytoRibosomes, reg.erRibosomes]
    else:
        mrna_regions = [reg.nucleoplasm, reg.cytoplasm, reg.ribosomes]

    for trans_rate, decay_rate, gene, mrna in zip(transcription_rates, decay_rates, genes, mrnas):
        sim.reaction([gene], [gene, mrna], trans_rate, regions=[reg.nucleoplasm, reg.chromosome])
        sim.reaction([mrna], [], decay_rate, regions=mrna_regions)

# Translation
with sim.construct():
    tlInitDet = 0.2 * 2000 * mRNADiffusion * sim.NA * sim.latticeSpacing
    sim.rateConst("rib_assoc", tlInitDet, order=2, annotation='mRNA/Ribosome association rate')

    sim.rateConst("ip_gal1", 1.9254*conv1stOrder, order=1, annotation='GAL1 translation')
    sim.rateConst("ip_gal2", 13.4779*conv1stOrder, order=1, annotation="GAL2 translation")
    sim.rateConst("ip_gal3", 55.4518*conv1stOrder, order=1, annotation="GAL3 translation")
    sim.rateConst("ip_gal4", 10.7091*conv1stOrder, order=1, annotation="GAL4 translation")
    sim.rateConst("ip_rep", 5.7762*conv1stOrder, order=1, annotation="GFP translation")
    sim.rateConst("ip_gal80", 3.6737*conv1stOrder, order=1, annotation="GAL80 translation")

    sim.rateConst("dp_gal1", 0.003851*conv1stOrder, order=1, annotation='GAL1 degradation')
    sim.rateConst("dp_gal2", 0.003851*conv1stOrder, order=1, annotation="GAL2 degradation")
    sim.rateConst("dp_gal3", 0.01155*conv1stOrder, order=1, annotation="GAL3 degradation")
    sim.rateConst("dp_gal4", 0.006931*conv1stOrder, order=1, annotation="GAL4 degradation")
    sim.rateConst("dp_rep", 0.01155*conv1stOrder, order=1, annotation="GFP degradation")
    sim.rateConst("dp_gal80", 0.006931*conv1stOrder, order=1, annotation="GAL80 degradation")

    ktls = [rc.ip_gal1, rc.ip_gal2, rc.ip_gal3, rc.ip_gal4, rc.ip_rep, rc.ip_gal80]
    dcys = [rc.dp_gal1, rc.dp_gal2, rc.dp_gal3, rc.dp_gal4, rc.dp_rep, rc.dp_gal80]
    mdcys = [rc.dr_gal1, rc.dr_gal2, rc.dr_gal3, rc.dr_gal4, rc.dr_rep, rc.dr_gal80]
    translatingRibosomes = [sp.ribosomeR1, sp.ribosomeR2, sp.ribosomeR3, sp.ribosomeR4, sp.ribosomeGrep, sp.ribosomeR80]
    prots = [sp.G1, sp.G2, sp.G3, sp.G4, sp.Grep, sp.G80]

    # Define ribosome regions based on enabled features
    if enable_er:
        ribosome_regions = [reg.cytoRibosomes, reg.erRibosomes]
    else:
        ribosome_regions = [reg.ribosomes]

    for mrna, translatingRibosomes, protein, ktl, dcy, mdcy in zip(mrnas, translatingRibosomes, prots, ktls, dcys, mdcys):
        sim.reaction([sp.ribosome, mrna], [translatingRibosomes], rc.rib_assoc, regions=ribosome_regions)
        sim.reaction([translatingRibosomes], [sp.ribosome, mrna, protein], ktl, regions=ribosome_regions)
        sim.reaction([translatingRibosomes], [sp.ribosome], mdcy, regions=ribosome_regions)

# Protein degradation
with sim.construct():
    # Define degradation compartments based on enabled features
    if enable_er:
        deg_compartments = [
            [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm],  # G1
            [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm, reg.plasmaMembrane],  # G2
            [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm],  # G3
            [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm, reg.nucleoplasm],  # G4
            [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm],  # Grep
            [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm, reg.nucleoplasm]   # G80
        ]
    else:
        deg_compartments = [
            [reg.ribosomes, reg.cytoplasm],  # G1
            [reg.ribosomes, reg.cytoplasm, reg.plasmaMembrane],  # G2
            [reg.ribosomes, reg.cytoplasm],  # G3
            [reg.ribosomes, reg.cytoplasm, reg.nucleoplasm],  # G4
            [reg.ribosomes, reg.cytoplasm],  # Grep
            [reg.ribosomes, reg.cytoplasm, reg.nucleoplasm]   # G80
        ]

    for protein, decay_rate, region in zip(prots, dcys, deg_compartments):
        sim.reaction([protein], [], decay_rate, regions=region)

# Initial conditions
if checkpoint_file == "":
    if IF_DGX:
        
        initMolec = pickle.load(open("/workspace/cme_species_counts.pkl", "rb"))
    else:
        
        initMolec = pickle.load(open("cme_species_counts.pkl", "rb"))

    volScale = np.sum(B.convexHull(sim.siteLattice==reg.plasmaMembrane.idx)) * sim.siteV / cellVol

    def initMolecules(x):
        if enable_er:
            counts = int(initMolec[x] * volScale)
        else:
            counts = int(round(initMolec[x] * volScale))
        return counts

    # Gene placement based on location option
    if gene_location == "random":
        print("gene location random")
        for b in ["DG1", "DG2", "DG3", "DG80", "DGrep"]:
            ops = [b+x for x in ["", "_G4d", "_G4d_G80d"]]
            spName = max(ops, key=lambda x:initMolec[x])
            print("{} in state {}".format(b, spName))
            sim.species(spName).placeNumberInto(reg.nucleoplasm, 1)
        sp.DG4.placeNumberInto(reg.nucleoplasm, 1)
        print("{} in state {}".format("Gene4", "DG4"))

    elif gene_location == "chromosome" and enable_chromosome:
        print("gene location chromosome")
        gene_coordinates = {
            'DG1': [116,87,112],
            'DG2': [120,73,116],
            'DG3': [120,90,103],
            'DG4': [137,91,123],
            'DG80': [124,100,111],
            'DGrep': [147,77,102],
        }
        genes = ["DG1", "DG2", "DG3", "DG80", "DGrep", "DG4"]
        for i, gene in enumerate(genes):
            if gene == "DG4":
                sp.DG4.placeParticle(*gene_coordinates['DG4'], 1)
                print("{} in state {} at position {}".format("Gene4", "DG4", gene_coordinates['DG4']))
            else:
                ops = [gene+x for x in ["", "_G4d", "_G4d_G80d"]]
                spName = max(ops, key=lambda x:initMolec[x])
                sim.species(spName).placeParticle(*gene_coordinates[gene], 1)
                print("{} in state {} at position {}".format(gene, spName, gene_coordinates[gene]))

    elif gene_location == "center":
        print("gene location center")
        nucleoplasm_indices = np.argwhere(sim.siteLattice == reg.nucleoplasm.idx)
        center = np.mean(nucleoplasm_indices, axis=0).astype(int)
        print(f"Center of nucleoplasm: {center}")
        
        if sim.siteLattice[tuple(center)] != reg.nucleoplasm.idx:
            distances = np.sum((nucleoplasm_indices - center)**2, axis=1)
            closest_idx = np.argmin(distances)
            center = nucleoplasm_indices[closest_idx]
            print(f"Adjusted center to ensure it's in nucleoplasm: {center}")

        genes = ["DG1", "DG2", "DG3", "DG80", "DGrep", "DG4"]
        positions = []
        x, y, z = center
        positions.append((x, y, z))

        for i in range(1, len(genes)):
            for offset in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                test_pos = (center[0] + offset[0], center[1] + offset[1], center[2] + offset[2])
                if (sim.siteLattice[test_pos] == reg.nucleoplasm.idx and 
                    test_pos not in positions):
                    positions.append(test_pos)
                    break
            
            if len(positions) <= i:
                for idx in nucleoplasm_indices:
                    pos = tuple(idx)
                    if pos not in positions:
                        positions.append(pos)
                        break

        for i, gene in enumerate(genes):
            if gene == "DG4":
                sp.DG4.placeParticle(*positions[i], 1)
                print("{} in state {} at position {}".format("Gene4", "DG4", positions[i]))
            else:
                ops = [gene+x for x in ["", "_G4d", "_G4d_G80d"]]
                spName = max(ops, key=lambda x:initMolec[x])
                sim.species(spName).placeParticle(*positions[i], 1)
                print("{} in state {} at position {}".format(gene, spName, positions[i]))

    elif gene_location == "edge":
        print("gene location edge")
        nucleoplasm_only_indices = np.argwhere(boolLattice("nucleoplasm"))
        nuclear_pores_indices = np.argwhere(boolLattice("nuclearPores"))
        
        nucleoplasm_set = {tuple(idx) for idx in nucleoplasm_only_indices}
        nuclear_pores_set = {tuple(idx) for idx in nuclear_pores_indices}
        
        edge_voxels = []
        neighbor_offsets = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        
        for idx in nucleoplasm_only_indices:
            x, y, z = idx
            if tuple(idx) in nuclear_pores_set:
                continue
                
            for dx, dy, dz in neighbor_offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                if (nx < 0 or ny < 0 or nz < 0 or 
                    nx >= latticeData['lattice'].shape[0] or 
                    ny >= latticeData['lattice'].shape[1] or 
                    nz >= latticeData['lattice'].shape[2]):
                    continue
                    
                if tuple((nx, ny, nz)) not in nucleoplasm_set:
                    edge_voxels.append((x, y, z))
                    break
        
        print(f"Found {len(edge_voxels)} edge voxels in nucleoplasm")
        
        if len(edge_voxels) < 6:
            print("Warning: Not enough edge voxels found, using some interior voxels")
            interior_voxels = [tuple(idx) for idx in nucleoplasm_only_indices 
                              if tuple(idx) not in nuclear_pores_set 
                              and tuple(idx) not in edge_voxels]
            
            additional_needed = 6 - len(edge_voxels)
            if len(interior_voxels) >= additional_needed:
                selected_interior = np.random.choice(len(interior_voxels), additional_needed, replace=False)
                for idx in selected_interior:
                    edge_voxels.append(interior_voxels[idx])

        if len(edge_voxels) >= 6:
            selected_indices = np.array(edge_voxels)[np.random.choice(len(edge_voxels), 6, replace=False)]
        else:
            selected_indices = np.array(edge_voxels)
            print(f"Warning: Only {len(selected_indices)} valid positions found for genes")

        genes = ["DG1", "DG2", "DG3", "DG80", "DGrep", "DG4"]
        for i, gene in enumerate(genes[:len(selected_indices)]):
            pos = tuple(selected_indices[i])
            if gene == "DG4":
                sp.DG4.placeParticle(*pos, 1)
                print("{} in state {} at position {}".format("Gene4", "DG4", pos))
            else:
                ops = [gene+x for x in ["", "_G4d", "_G4d_G80d"]]
                spName = max(ops, key=lambda x:initMolec[x])
                sim.species(spName).placeParticle(*pos, 1)
                print("{} in state {} at position {}".format(gene, spName, pos))

        if len(selected_indices) < 6:
            remaining_genes = genes[len(selected_indices):]
            print(f"Placing remaining genes {remaining_genes} randomly in nucleoplasm")
            for gene in remaining_genes:
                if gene == "DG4":
                    sp.DG4.placeNumberInto(reg.nucleoplasm, 1)
                    print("{} in state {} placed randomly".format("Gene4", "DG4"))
                else:
                    ops = [gene+x for x in ["", "_G4d", "_G4d_G80d"]]
                    spName = max(ops, key=lambda x:initMolec[x])
                    sim.species(spName).placeNumberInto(reg.nucleoplasm, 1)
                    print("{} in state {} placed randomly".format(gene, spName))

    else:
        print(f"gene location not recognized: {gene_location}")
        raise ValueError(f"gene location not recognized: {gene_location}")

    # Place proteins/metabolites
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

    # Place mRNAs
 
    sp.R3.placeNumberInto(reg.nucleoplasm, initMolecules("R3"))
    sp.R80.placeNumberInto(reg.nucleoplasm, initMolecules("R80"))
    print("R3 in nucleoplasm: {}".format(initMolecules("R3")))
    print("R80 in nucleoplasm: {}".format(initMolecules("R80")))

    # Place G80
 
    cscl = reg.cytoplasm.volume/(reg.cytoplasm.volume+reg.nucleoplasm.volume)
    totM = initMolecules("G80C") + initMolecules("G80")
    totD = initMolecules("G80Cd") + initMolecules("G80d")
    sp.G80.placeNumberInto(reg.cytoplasm, int(cscl*totM))
    sp.G80.placeNumberInto(reg.nucleoplasm, int((1-cscl)*totM))
    sp.G80d.placeNumberInto(reg.cytoplasm, int(cscl*totD))
    sp.G80d.placeNumberInto(reg.nucleoplasm, int((1-cscl)*totD))
    print("G80 in cytoplasm: {}, in nucleoplasm: {}".format(int(cscl*totM), int((1-cscl)*totM)))
    print("G80d in cytoplasm: {}, in nucleoplasm: {}".format(int(cscl*totD), int((1-cscl)*totD)))
  
    # Place ribosomes
    if enable_er:
        for x, y, z in np.argwhere(sim.siteLattice == reg.cytoRibosomes.idx):
            sp.ribosome.placeParticle(x, y, z, 1)
        for x, y, z in np.argwhere(sim.siteLattice == reg.erRibosomes.idx):
            sp.ribosome.placeParticle(x, y, z, 1)
        print("ribosomes number:", np.sum(sim.siteLattice == reg.cytoRibosomes.idx) + np.sum(sim.siteLattice == reg.erRibosomes.idx))
        print(f"cytoRibosomes: {np.sum(sim.siteLattice == reg.cytoRibosomes.idx)}, erRibosomes: {np.sum(sim.siteLattice == reg.erRibosomes.idx)}")
    else:
        for x, y, z in np.argwhere(sim.siteLattice == reg.ribosomes.idx):
            sp.ribosome.placeParticle(x, y, z, 1)
        print("ribosomes number:", np.sum(sim.siteLattice == reg.ribosomes.idx))

else:
    print(f"using checkpoint file:{checkpoint_file}")
    if os.path.exists(checkpoint_file):
        print(f"start from the last checkpoint file{checkpoint_file}")
        try:
            sim.copyParticleLattice(checkpoint_file, replicate=1, frame=-1)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint file {checkpoint_file}. Error: {str(e)}")
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}. Cannot restart simulation.")

# Diffusion Coefficients
with sim.construct():
    sim.transitionRate(None, None, None, sim.diffusionZero)

# DNA - Fix in location
with sim.construct():
    for sps in sim.speciesList.matchRegex("D.*"):
        sps.diffusionRate(None, sim.diffusionZero)

# mRNA diffusion
with sim.construct():
    sim.diffusionConst("mrna", mRNADiffusion, texRepr=r'D_{mRNA}', annotation='Generic mRNA')

    if enable_er:
        # Special handling for ER version
        for mrna in sim.speciesList.matchRegex("R.*"):
            if mrna.name != "R2":
                sim.transitionRate(mrna, reg.nucleoplasm, reg.cytoplasm, dc.mrna)
                sim.transitionRate(mrna, reg.cytoplasm, reg.nucleoplasm, sim.diffusionZero)
                sim.transitionRate(mrna, reg.nucleoplasm, reg.nucleoplasm, dc.mrna)
                sim.transitionRate(mrna, reg.cytoplasm, reg.cytoplasm, dc.mrna)
                sim.transitionRate(mrna, reg.cytoRibosomes, reg.cytoRibosomes, dc.mrna)
                sim.transitionRate(mrna, reg.cytoRibosomes, reg.cytoplasm, dc.mrna)
                sim.transitionRate(mrna, reg.cytoplasm, reg.cytoRibosomes, dc.mrna)

        # R2 special handling for ER
        sim.transitionRate(sp.R2, reg.nucleoplasm, reg.cytoplasm, dc.mrna)
        sim.transitionRate(sp.R2, reg.cytoplasm, reg.nucleoplasm, sim.diffusionZero)
        sim.transitionRate(sp.R2, reg.nucleoplasm, reg.nucleoplasm, dc.mrna)
        sim.transitionRate(sp.R2, reg.cytoplasm, reg.erRibosomes, dc.mrna)
        sim.transitionRate(sp.R2, reg.erRibosomes, reg.cytoplasm, dc.mrna)
        sim.transitionRate(sp.R2, reg.erRibosomes, reg.erRibosomes, dc.mrna)
    else:
        # Standard mRNA diffusion
        for mrna in sim.speciesList.matchRegex("R.*"):
            sim.transitionRate(mrna, reg.nucleoplasm, reg.cytoplasm, dc.mrna)
            sim.transitionRate(mrna, reg.cytoplasm, reg.nucleoplasm, sim.diffusionZero)
            sim.transitionRate(mrna, reg.nucleoplasm, reg.nucleoplasm, dc.mrna)
            sim.transitionRate(mrna, reg.cytoplasm, reg.cytoplasm, dc.mrna)
            sim.transitionRate(mrna, reg.ribosomes, reg.ribosomes, dc.mrna)
            sim.transitionRate(mrna, reg.ribosomes, reg.cytoplasm, dc.mrna)
            sim.transitionRate(mrna, reg.cytoplasm, reg.ribosomes, dc.mrna)

    # Chromosome-specific mRNA diffusion
    if enable_chromosome:
        for mrna in sim.speciesList.matchRegex("R.*"):
            sim.transitionRate(mrna, reg.chromosome, reg.chromosome, sim.diffusionZero)
            sim.transitionRate(mrna, reg.chromosome, reg.nucleoplasm, sim.diffusionFast)

# Protein diffusion
with sim.construct():
    sim.diffusionConst("prot", 1e-12, texRepr=r'D_{prot}', annotation='Generic protein')
    # general ribosome diffusion
    sim.diffusionConst("ribo", 3e-13, texRepr=r'D_{ribosome}', annotation='Generic ribosome')
    if enable_er:
        
        # General protein diffusion for ER version
        for sps in [sp.G1, sp.G3, sp.G3i, sp.G4, sp.G4d, sp.G80, sp.G80d, sp.G80d_G3i, sp.Grep]:
            sim.transitionRate(sps, reg.cytoplasm, reg.cytoplasm, dc.prot)
            sim.transitionRate(sps, reg.cytoRibosomes, reg.cytoplasm, sim.diffusionFast)
            sim.transitionRate(sps, reg.cytoplasm, reg.cytoRibosomes, sim.diffusionZero)

        # G2 special ER handling
        sim.transitionRate(sp.G2, reg.erRibosomes, reg.endoplasmicReticulum, sim.diffusionFast)
        sim.transitionRate(sp.G2, reg.endoplasmicReticulum, reg.endoplasmicReticulum, dc.prot)

        sim.transitionRate(sp.G2, reg.endoplasmicReticulum, reg.erRibosomes, sim.diffusionZero)
        sim.transitionRate(sp.G2, reg.endoplasmicReticulum, reg.pmaER, dc.prot)
        sim.transitionRate(sp.G2, reg.pmaER, reg.endoplasmicReticulum, sim.diffusionZero)
        sim.transitionRate(sp.G2, reg.pmaER, reg.cytoplasm, sim.diffusionFast)
        sim.transitionRate(sp.G2, reg.pmaER, reg.pmaER, dc.prot)
        sim.transitionRate(sp.G2, reg.cytoplasm, reg.pmaER, sim.diffusionFast) # no diffusion back
    else:
        # Standard protein diffusion
        for sps in [sp.G1, sp.G2, sp.G3, sp.G3i, sp.G4, sp.G4d, sp.G80, sp.G80d, sp.G80d_G3i, sp.Grep]:
            sim.transitionRate(sps, reg.cytoplasm, reg.cytoplasm, dc.prot)
            sim.transitionRate(sps, reg.ribosomes, reg.cytoplasm, sim.diffusionFast)
            sim.transitionRate(sps, reg.cytoplasm, reg.ribosomes, sim.diffusionZero)

# Transcription factors
with sim.construct():
    for sps in [sp.G4, sp.G4d, sp.G80, sp.G80d]:
        sim.transitionRate(sps, reg.nucleoplasm, reg.nucleoplasm, dc.prot)
        sim.transitionRate(sps, reg.nucleoplasm, reg.cytoplasm, dc.prot)
        sim.transitionRate(sps, reg.cytoplasm, reg.nucleoplasm, dc.prot)
        
        if enable_chromosome:
            sim.transitionRate(sps, reg.chromosome, reg.chromosome, sim.diffusionZero)
            sim.transitionRate(sps, reg.chromosome, reg.nucleoplasm, sim.diffusionFast)
            sim.transitionRate(sps, reg.nucleoplasm, reg.chromosome, sim.diffusionFast)

# Cytoplasmic proteins - prevent nuclear diffusion  
with sim.construct():
    for sps in [sp.G1, sp.G2, sp.G3, sp.G3i, sp.G80d_G3i, sp.Grep]:
        sim.transitionRate(sps, reg.cytoplasm, reg.nucleoplasm, sim.diffusionZero)

# Membrane transporter
with sim.construct():
    sim.transitionRate(sp.G2, reg.cytoplasm, reg.plasmaMembrane, dc.prot)
    if enable_er:
        sim.transitionRate(sp.G2, reg.pmaER, reg.plasmaMembrane, dc.prot)
    sim.transitionRate(sp.G2, reg.plasmaMembrane, reg.cytoplasm, sim.diffusionZero)
    sim.diffusionConst("mem", 0.01e-12, texRepr=r'D_{mem}', annotation='Generic protein on membrane')
    sim.transitionRate(sp.G2, reg.plasmaMembrane, reg.plasmaMembrane, dc.mem)

# Ribosomes - fixed
with sim.construct():
    for sps in sim.speciesList.matchRegex("ribosome.*"):
        sim.transitionRate(sps, None, None, sim.diffusionZero)

# Import ODE hybrid solver components
import json
import scipy.integrate as spi

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
            self.rdme = RDMEFile(lmFile)
        self.cellVol = self.rdme.reg.cytoplasm.volume + self.rdme.reg.nucleoplasm.volume + self.rdme.reg.plasmaMembrane.volume
        self.NAV = 6.022e23*self.cellVol
        
        self.g3actRidx = self.rdme.reaction(self.rdme.sp.G3, self.rdme.sp.G3i, self.rdme.rc.fi).idx
        self.g3actRc = self.rdme.rc.fi._toLM()
        
        # Open output files
        self.save_cts_by_region_file = output_folder + "_region.jsonl"
        self.save_cts_by_region_handle = open(self.save_cts_by_region_file, "w")
        self.hook_time = 0
        
        self.save_ode_data_file = output_folder + "_ode.jsonl"
        self.save_ode_data_handle = open(self.save_ode_data_file, "w")
    
    def copyInitialConditions(self, cts):
        if checkpoint_file == "":
            y = np.zeros(len(self.odeSpNames))
            y[self.odeSpIndex("GAI")] = 0
            y[self.odeSpIndex("G1")] = cts['countBySpecies'][self.rdme.sp.G1]/self.NAV 
            y[self.odeSpIndex("G1GAI")] = 0
            y[self.odeSpIndex("G2")] = cts['countBySpecies'][self.rdme.sp.G2]/self.NAV
            y[self.odeSpIndex("G2GAE")] = 0
            y[self.odeSpIndex("G2GAI")] = 0
        else:
            print(f"using checkpoint:{checkpoint_file}")
            checkpoint_ode = checkpoint_file + "_ode.jsonl"
           
            with open(checkpoint_ode, 'r') as f:
                last_line = None
                for line in f:
                    last_line = line
                
                if last_line is None:
                    raise RuntimeError(f"ODE checkpoint file {checkpoint_ode} is empty")
                
                last_ode_state = json.loads(last_line.strip())
                if 'species' not in last_ode_state:
                    raise RuntimeError(f"Invalid ODE state format in {checkpoint_ode}")
                
                y = np.zeros(len(self.odeSpNames))
                for i, name in enumerate(self.odeSpNames):
                    y[self.odeSpIndex(name)] = last_ode_state['species'][name]
                
                print(f"Initialized ODE state from time {last_ode_state['time']}")
        
        self.boundGal = self.rdmeGal(cts)
        return y
    
    def rdmeGal(self, cts):
        return (cts['countBySpecies'][self.rdme.sp.G3i] + cts['countBySpecies'][self.rdme.sp.G80d_G3i])/self.NAV

    def rdme2odeConc(self, y0, cts):
        y = y0.copy()
        
        # Update G1 in ODE
        g1ode = y0[self.odeSpIndex("G1")]
        g1gaiode = y0[self.odeSpIndex("G1GAI")]
        g1rdme = cts['countBySpecies'][self.rdme.sp.G1]/self.NAV
        change = g1rdme-g1ode-g1gaiode
        
        if change > 0:
            y[self.odeSpIndex("G1")] = g1ode + change
        else:
            fracChange = g1rdme/(g1ode+g1gaiode) if (g1ode+g1gaiode) > 0 else 0
            y[self.odeSpIndex("G1")] = g1ode*fracChange
            y[self.odeSpIndex("G1GAI")] = g1gaiode*fracChange
            y[self.odeSpIndex("GAI")] += g1gaiode*(1-fracChange)
            
        # Update G2 in ODE
        g2ode = y0[self.odeSpIndex("G2")]
        g2gaiode = y0[self.odeSpIndex("G2GAI")]
        g2gaeode = y0[self.odeSpIndex("G2GAE")]
        g2rdme = cts['countBySpeciesRegion'][self.rdme.sp.G2][self.rdme.reg.plasmaMembrane]/self.NAV
        
        change = g2rdme-g2ode-g2gaiode-g2gaeode
        
        if change >= 0:
            y[self.odeSpIndex("G2")] = g2ode + change
        else:
            total = g2ode+g2gaiode+g2gaeode
            fracChange = g2rdme/total if total > 0 else 0
            y[self.odeSpIndex("G2")] = g2ode*fracChange
            y[self.odeSpIndex("G2GAI")] = g2gaiode*fracChange
            y[self.odeSpIndex("GAI")] += g2gaiode*(1-fracChange)
            y[self.odeSpIndex("G2GAE")] = g2gaeode*fracChange

        # Update internal galactose in ODE
        g0 = self.boundGal
        g1 = self.rdmeGal(cts)
        y[self.odeSpIndex("GAI")] += g1-g0
        self.boundGal = g1
            
        return y
                   
    def hookSimulation(self, t, lattice):
        print_memory_usage()
        start_time_hook = time.time()
        
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
        self.setReactionRate(self.g3actRidx, assocRt)
        self.save_rdme_cts_by_region(t, cts)
        self.save_ode_data(t, ys1)
        self.print_ode_evals(t,assocRt,cts)
        
        end_time_hook = time.time()
        self.hook_time += end_time_hook - start_time_hook
        
        if args.max_time is not None and (end_time_hook - start_time) >= args.max_time * 3600:
            print(f"Maximum simulation time of {args.max_time} hours reached. Stopping simulation.")
            return 3
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
    
    def ode_model(self,conc, ts, GAE):
        NA = 6.02214076e23
        kf_GK = 1.442e5  # M^-1 s^-1 
        kr_GK = 30.708   # s^-1
        kcat_GK = 55.833 # s^-1
        kcat_TR = 72.5   # s^-1
        kr_TR = 39.875   # s^-1
        kf_TR = 1.123e5  # M^-1 s^-1 
        kf_TR_gae = 1.123e5* GAE # s^-1
        
        GAI = conc[self.odeSpIndex("GAI")]
        G2GAI = conc[self.odeSpIndex("G2GAI")]
        G2GAE = conc[self.odeSpIndex("G2GAE")]
        G1GAI = conc[self.odeSpIndex("G1GAI")]
        G1 = conc[self.odeSpIndex("G1")]
        G2 = conc[self.odeSpIndex("G2")]
    
        # GAI
        dGAI_dt = kr_TR*G2GAI - kf_TR*GAI*G2 + kr_GK*G1GAI - kf_GK*G1*GAI
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
       
        dx_dt = [0] * len(self.odeSpNames)
        dx_dt[self.odeSpIndex("GAI")] = dGAI_dt
        dx_dt[self.odeSpIndex("G1")] = dG1_dt
        dx_dt[self.odeSpIndex("G1GAI")] = dG1GAI_dt
        dx_dt[self.odeSpIndex("G2")] = dG2_dt
        dx_dt[self.odeSpIndex("G2GAI")] = dG2GAI_dt
        dx_dt[self.odeSpIndex("G2GAE")] = dG2GAE_dt
        dx_dt_array = np.asarray(dx_dt)
        return (dx_dt_array)
    
    def stepOde(self, dt, ys0):
        odestep = 0.001
        ts = np.linspace(0,dt, int(np.ceil(dt/odestep))+1)
        ys = spi.odeint(self.ode_model, ys0, ts, args=(self.GAE,), hmax=odestep)
        return ys[-1]

    def save_rdme_cts_by_region(self, t, stats):
        counts_by_region = {'time': float(t)}

        for species in self.rdme.speciesList:
            counts_by_region[species.name] = {}
            for region in self.rdme.regionList:
                count = stats['countBySpeciesRegion'][species][region]
                counts_by_region[species.name][region.name] = int(count)

        json.dump(counts_by_region, self.save_cts_by_region_handle, cls=NumpyEncoder)
        self.save_cts_by_region_handle.write('\n')
        self.save_cts_by_region_handle.flush()

        print(f"Data for time {t} appended to {self.save_cts_by_region_file}")
        return counts_by_region

    def save_ode_data(self, t, ys):
        ode_data = {
            'time': float(t),
            'species': {name: float(value) for name, value in zip(self.odeSpNames, ys)}
        }

        json.dump(ode_data, self.save_ode_data_handle, cls=NumpyEncoder)
        self.save_ode_data_handle.write('\n')
        self.save_ode_data_handle.flush()

        print(f"ODE data for time {t} appended to {self.save_ode_data_file}")

# Signal handler
def signal_handler(signum, frame):
    print("Interrupt received, stopping simulation...")
    if 'solver' in globals():
        if hasattr(solver, 'save_cts_by_region_handle'):
            solver.save_cts_by_region_handle.close()
        if hasattr(solver, 'save_ode_data_handle'):
            solver.save_ode_data_handle.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Solver setup and execution
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

end_time = time.time()
total_time = end_time - start_time
print(f"Total time taken: {total_time} seconds")