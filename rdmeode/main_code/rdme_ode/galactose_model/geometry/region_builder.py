"""Build spatial regions for RDME simulation

This module constructs all spatial regions including:
- Basic cellular compartments (cytoplasm, nucleus, etc.)
- Optional chromosome regions
- Optional ER regions
- Ribosome regions (effective or standard)
- Species definitions
- Reactions
- Initial conditions
- Diffusion coefficients
"""

import numpy as np
import os
import pickle
from jLM.RDME import Sim as RDMESim
from jLM.RegionBuilder import RegionBuilder
from .lattice_loader import get_bool_lattice

region_dir = "./geometry/"

def build_regions(lattice_data, output_folder, args):
    """Build all spatial regions for the simulation

    Args:
        lattice_data (dict): Loaded lattice data
        args: Parsed command-line arguments

    Returns:
        RDMESim: Configured simulation object with all regions defined
    """
    # Extract arguments
    enable_chromosome = args.enable_chromosome
    enable_er = args.enable_er
    enable_effective_ribosome = args.enable_effective_ribosome
    enable_rna_tracking = args.enable_rna_tracking
    er_num = args.er_num
    gene_location = args.gene_location
    checkpoint_file = args.checkpoint
    simtime = args.simtime
    externalGal_input = args.galactose
    output_order = args.index
    gpus = args.gpus
    if_dgx = args.if_dgx
    enable_ribosome_movement = args.enable_ribosome_movement
    ribosome_movement_mode = args.ribosome_movement_mode 
    # ribosome_movement_move_fraction = args.ribosome_movement_move_fraction
    # ribosome_movement_move_interval = args.ribosome_movement_move_interval
    dir_dgx = "workspace/" if if_dgx else ""
    dir_prefix = os.path.join(dir_dgx, region_dir)
    # Get boolean lattice function
    bool_lattice_func = get_bool_lattice(lattice_data)

    # Create closure to access lattice_data
    boolLattice = bool_lattice_func

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
    
        chromo_gene = np.load(dir_prefix + "gene_masks.npy").astype(bool)
        chromo_dummy = np.load(dir_prefix + "dummy_chromosome.npy").astype(bool)
    if enable_er:
        # Handle effective ribosome case for ER
        if enable_effective_ribosome:
            print('Loading effective ribosome regions for ER case')
            effective_cytoRibosomes = np.load(dir_prefix + "effective_cyto_ribosomes_ER_Marie_4152.npy").astype(bool)
            dummy_cytoRibosomoe = np.load(dir_prefix + "dummy_cyto_ribosomes_ER_Marie_4152.npy").astype(bool)
            effective_erRibosomes = np.load(dir_prefix + "effective_er_ribosomes_ER_Marie_2151.npy").astype(bool)
            dummy_erRibosomes = np.load(dir_prefix + "dummy_er_ribosomes_ER_Marie_2151.npy").astype(bool)
        else: 
            # erRibosomes = boolLattice("pmaRibosomes") | boolLattice("cecRibosomes") | boolLattice("tubRibosomes")
            pmaRibosomes = boolLattice("pmaRibosomes")
            cecRibosomes = boolLattice("cecRibosomes")
            tubRibosomes = boolLattice("tubRibosomes")
            cytoRibosomes = boolLattice("cytoRibosomes")
        # enable ER then, to overlap
        ER_geo_dir = "ER_geometry"
        pmaER = np.load(os.path.join(dir_prefix, ER_geo_dir, "pmER_fixed_geometry.npy")).astype(bool)
        tub_dir = "combined_tubes"
        if er_num == 4:
            tubER = np.load(os.path.join(dir_prefix, ER_geo_dir, "tubER_fixed_geometry.npy")).astype(bool)
            print("tubER is loaded from tubER_fixed_geometry.npy")
       
        elif er_num == 3:
            tubER = np.load(os.path.join(dir_prefix, tub_dir, "combined_tubes_3.npy")).astype(bool) | np.load(os.path.join(dir_prefix, tub_dir, "combined_tubes_2.npy")).astype(bool) | np.load(os.path.join(dir_prefix, tub_dir, "combined_tubes_1.npy")).astype(bool)
            print("tubER is loaded from combined_tubes_3.npy, combined_tubes_2.npy, combined_tubes_1.npy")
        elif er_num == 2:
            tubER = np.load(os.path.join(dir_prefix, tub_dir, "combined_tubes_2.npy")).astype(bool) | np.load(os.path.join(dir_prefix, tub_dir, "combined_tubes_1.npy")).astype(bool)
            print("tubER is loaded from combined_tubes_2.npy, combined_tubes_1.npy")
        elif er_num == 1:
            tubER = np.load(os.path.join(dir_prefix, tub_dir, "combined_tubes_1.npy")).astype(bool)
            print("tubER is loaded from combined_tubes_1.npy")
        else: 
            print("ER tunnels number should be 1, 2, 3 or 4, you have entered: ", er_num)
        endoplasmicReticulum = np.load(os.path.join(dir_prefix, ER_geo_dir, "cecER_fixed_geometry.npy")).astype(bool) | tubER
        
    else:
        if enable_effective_ribosome:
            print('Loading effective ribosome regions for no ER case')
            ribosome_dummy = np.load(dir_prefix + "dummy_ribosomes_noER.npy").astype(bool)
            ribosome = np.load(dir_prefix + "effective_ribosomes_noER.npy").astype(bool)
        else:
            ribosomes = boolLattice("ribosomes")
            
        
    
    
    # decimation = lattice_data['decimation']
    
    # Simulation setup
    if gpus == 1:
        siteType = "Int"
    else:
        siteType = "Byte"
    
    sim_title = "Galactose switch"
    # if enable_er:
    #     sim_title += " ER"
    # if enable_chromosome:
    #     sim_title += " chromosome"
    # if enable_effective_ribosome:
    #     sim_title += " effective-ribosome"
    # sim_title += ", RDME/ODE hybrid"
    
    sim = RDMESim(sim_title, output_folder, lattice_data['lattice'].shape, 
                  lattice_data['latticeSpacing'], "extracellular", siteType)
    
    print("the shape of the lattice is: ", lattice_data['lattice'].shape)
    
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
            # (sim.region('erRibosomes'), erRibosomes),
            (sim.region('pmaRibosomes'), pmaRibosomes),
            (sim.region('cecRibosomes'), cecRibosomes),
            (sim.region('tubRibosomes'), tubRibosomes),
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
        # nERribosomes = np.sum(sim.siteLattice == reg.erRibosomes.idx)
        npmaRibosomes = np.sum(sim.siteLattice == reg.pmaRibosomes.idx)
        ncecRibosomes = np.sum(sim.siteLattice == reg.cecRibosomes.idx)
        ntubRibosomes = np.sum(sim.siteLattice == reg.tubRibosomes.idx)
        nERribosomes = npmaRibosomes + ncecRibosomes + ntubRibosomes
        nRibosomes = ncytoRibosomes + nERribosomes
    else:
        nRibosomes = np.sum(sim.siteLattice == reg.ribosomes.idx)
    
    mRNADiffusion = 0.05e-12  # m^2/s
    
    # Species Definitions
    
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
    if enable_rna_tracking:
        sim.species('tracker_R1', texRepr='Tracker{:}R_{1}', annotation="Tracking particle for Gal1 mRNA")
        sim.species('tracker_R2', texRepr='Tracker{:}R_{2}', annotation="Tracking particle for Gal2 mRNA")
        sim.species('tracker_R3', texRepr='Tracker{:}R_{3}', annotation="Tracking particle for Gal3 mRNA")
        sim.species('tracker_R4', texRepr='Tracker{:}R_{4}', annotation="Tracking particle for Gal4 mRNA")
        sim.species('tracker_R80', texRepr='Tracker{:}R_{80}', annotation="Tracking particle for Gal80 mRNA")
        sim.species('tracker_Rrep', texRepr='Tracker{:}R_{rep}', annotation="Tracking particle for reporter mRNA")
    
    # Reactions
    cellVol = 3.57e-14  # L, cell size from Ramsey paper SI, haploid yeast
    nav = cellVol * 6.022e23
    invMin2invSec = 1/60.0
    conv2ndOrder = invMin2invSec * nav
    conv1stOrder = invMin2invSec
    
    # Dimerization
    
    sim.rateConst('fd', 100 * conv2ndOrder, order=2, annotation="Gal4p/Gal80p dimer formation")
    sim.rateConst('rd', 0.001 * conv1stOrder, order=1, annotation="Gal4p/Gal80p dimer dissociation")
    sim.reaction([sp.G4, sp.G4], [sp.G4d], rc.fd, annotation="Gal4p/Gal80p dimer formation", regions=[reg.cytoplasm, reg.nucleoplasm])
    sim.reaction([sp.G4d], [sp.G4, sp.G4], rc.rd, annotation="Gal4p/Gal80p dimer dissociation", regions=[reg.cytoplasm, reg.nucleoplasm])
    sim.reaction([sp.G80, sp.G80], [sp.G80d], rc.fd, annotation="Gal80p/Gal80p dimer formation", regions=[reg.cytoplasm, reg.nucleoplasm])
    sim.reaction([sp.G80d], [sp.G80, sp.G80], rc.rd, annotation="Gal80p/Gal80p dimer dissociation", regions=[reg.cytoplasm, reg.nucleoplasm])
    
    # DNA regulation
    
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
    if enable_chromosome:
        chromosome_regions = [reg.nucleoplasm, reg.chromosome]
    else:
        chromosome_regions = [reg.nucleoplasm]
    for dna, dna_gal4, dna_gal4_gal80, f_g4, r_g4, f_g4g80, r_g4g80 in zip(dnas, dna_gal4, dna_gal4_gal80, f_g4s, r_g4s, f_g4g80s, r_g4g80s):
        sim.reaction([dna, sp.G4d], [dna_gal4], f_g4, annotation="Gal4p binding to gene", regions=chromosome_regions)
        sim.reaction([dna_gal4], [dna, sp.G4d], r_g4, annotation="Gal4p dissociation from gene", regions=chromosome_regions)
        sim.reaction([dna_gal4, sp.G80d], [dna_gal4_gal80], f_g4g80, annotation="Gal80p binding to gene", regions=chromosome_regions)
        sim.reaction([dna_gal4_gal80], [dna_gal4, sp.G80d], r_g4g80, annotation="Gal80p dissociation from gene", regions=chromosome_regions)
    
    # G3 activation
    
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
    if enable_rna_tracking:
        mrna_trackers = [sp.tracker_R1, sp.tracker_R2, sp.tracker_R3, sp.tracker_R4, sp.tracker_R80, sp.tracker_Rrep]
    # Define regions for mRNA degradation based on enabled features
    if enable_er:
        # mrna_regions = [reg.nucleoplasm, reg.cytoplasm, reg.cytoRibosomes, reg.erRibosomes]
        mrna_regions = [reg.nucleoplasm, reg.cytoplasm, reg.cytoRibosomes, reg.pmaRibosomes, reg.cecRibosomes, reg.tubRibosomes]
    else:
        mrna_regions = [reg.nucleoplasm, reg.cytoplasm, reg.ribosomes]
    
    if enable_chromosome:
        chromosome_regions = [reg.nucleoplasm, reg.chromosome]
    else:
        chromosome_regions = [reg.nucleoplasm]
    if enable_rna_tracking:
        for trans_rate, decay_rate, gene, mrna, mrna_tracker in zip(transcription_rates, decay_rates, genes, mrnas, mrna_trackers):
            sim.reaction([gene], [gene, mrna, mrna_tracker], trans_rate, regions=chromosome_regions)
            sim.reaction([mrna], [], decay_rate, regions=mrna_regions)
    else:
        for trans_rate, decay_rate, gene, mrna in zip(transcription_rates, decay_rates, genes, mrnas):
            sim.reaction([gene], [gene, mrna], trans_rate, regions=chromosome_regions)
            sim.reaction([mrna], [], decay_rate, regions=mrna_regions)
    
    # Translation
    
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
        # ribosome_regions = [reg.cytoRibosomes, reg.erRibosomes]
        ribosome_regions = [reg.cytoRibosomes, reg.pmaRibosomes, reg.cecRibosomes, reg.tubRibosomes]
    else:
        ribosome_regions = [reg.ribosomes]
    
    for mrna, translatingRibosomes, protein, ktl, dcy, mdcy in zip(mrnas, translatingRibosomes, prots, ktls, dcys, mdcys):
        sim.reaction([sp.ribosome, mrna], [translatingRibosomes], rc.rib_assoc, regions=ribosome_regions)
        sim.reaction([translatingRibosomes], [sp.ribosome, mrna, protein], ktl, regions=ribosome_regions)
        sim.reaction([translatingRibosomes], [sp.ribosome], mdcy, regions=ribosome_regions)
    
    # Protein degradation
    
    # Define degradation compartments based on enabled features
    if enable_er:
        # deg_compartments = [
        #     [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm],  # G1
        #     [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm, reg.plasmaMembrane],  # G2
        #     [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm],  # G3
        #     [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm, reg.nucleoplasm],  # G4
        #     [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm],  # Grep
        #     [reg.cytoRibosomes, reg.erRibosomes, reg.cytoplasm, reg.nucleoplasm]   # G80
        # ]
        deg_compartments = [
            [reg.cytoRibosomes, reg.pmaRibosomes, reg.cecRibosomes, reg.tubRibosomes, reg.cytoplasm],  # G1
            [reg.cytoRibosomes, reg.pmaRibosomes, reg.cecRibosomes, reg.tubRibosomes, reg.cytoplasm, reg.plasmaMembrane],  # G2
            [reg.cytoRibosomes, reg.pmaRibosomes, reg.cecRibosomes, reg.tubRibosomes, reg.cytoplasm],  # G3
            [reg.cytoRibosomes, reg.pmaRibosomes, reg.cecRibosomes, reg.tubRibosomes, reg.cytoplasm, reg.nucleoplasm],  # G4
            [reg.cytoRibosomes, reg.pmaRibosomes, reg.cecRibosomes, reg.tubRibosomes, reg.cytoplasm],  # Grep
            [reg.cytoRibosomes, reg.pmaRibosomes, reg.cecRibosomes, reg.tubRibosomes, reg.cytoplasm, reg.nucleoplasm]   # G80
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
        if if_dgx:
            
            initMolec = pickle.load(open("/workspace/cme_species_counts.pkl", "rb"))
        else:
            
            initMolec = pickle.load(open("./init_species/cme_species_counts.pkl", "rb"))
    
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
                        nx >= lattice_data['lattice'].shape[0] or 
                        ny >= lattice_data['lattice'].shape[1] or 
                        nz >= lattice_data['lattice'].shape[2]):
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
    
            for x, y, z in np.argwhere(sim.siteLattice == reg.pmaRibosomes.idx):
                sp.ribosome.placeParticle(x, y, z, 1)
            for x, y, z in np.argwhere(sim.siteLattice == reg.cecRibosomes.idx):
                sp.ribosome.placeParticle(x, y, z, 1)
            for x, y, z in np.argwhere(sim.siteLattice == reg.tubRibosomes.idx):
                sp.ribosome.placeParticle(x, y, z, 1)
            print("ribosomes number:", np.sum(sim.siteLattice == reg.cytoRibosomes.idx) + nERribosomes)
            print(f"cytoRibosomes: {np.sum(sim.siteLattice == reg.cytoRibosomes.idx)}, pmaRibosomes: {np.sum(sim.siteLattice == reg.pmaRibosomes.idx)}, cecRibosomes: {np.sum(sim.siteLattice == reg.cecRibosomes.idx)}, tubRibosomes: {np.sum(sim.siteLattice == reg.tubRibosomes.idx)}")
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
    
    sim.transitionRate(None, None, None, sim.diffusionZero)
    
    # DNA - Fix in location
    
    for sps in sim.speciesList.matchRegex("D.*"):
        sps.diffusionRate(None, sim.diffusionZero)
    
    # mRNA diffusion
    
    
    
    # mRNA diffusion
    
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
        sim.transitionRate(sp.R2, reg.cytoplasm, reg.cytoplasm, dc.mrna)
        sim.transitionRate(sp.R2, reg.nucleoplasm, reg.nucleoplasm, dc.mrna)
        # sim.transitionRate(sp.R2, reg.cytoplasm, reg.erRibosomes, dc.mrna)
        # sim.transitionRate(sp.R2, reg.erRibosomes, reg.cytoplasm, dc.mrna)
        # sim.transitionRate(sp.R2, reg.erRibosomes, reg.erRibosomes, dc.mrna)
        sim.transitionRate(sp.R2, reg.cytoplasm, reg.pmaRibosomes, dc.mrna)
        sim.transitionRate(sp.R2, reg.pmaRibosomes, reg.cytoplasm, dc.mrna)
        sim.transitionRate(sp.R2, reg.pmaRibosomes, reg.pmaRibosomes, dc.mrna)
        sim.transitionRate(sp.R2, reg.cytoplasm, reg.cecRibosomes, dc.mrna)
        sim.transitionRate(sp.R2, reg.cecRibosomes, reg.cytoplasm, dc.mrna)
        sim.transitionRate(sp.R2, reg.cecRibosomes, reg.cecRibosomes, dc.mrna)
        sim.transitionRate(sp.R2, reg.cytoplasm, reg.tubRibosomes, dc.mrna)
        sim.transitionRate(sp.R2, reg.tubRibosomes, reg.cytoplasm, dc.mrna)
        sim.transitionRate(sp.R2, reg.tubRibosomes, reg.tubRibosomes, dc.mrna)
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
            sim.transitionRate(mrna, reg.chromosome, reg.chromosome, dc.mrna)
            sim.transitionRate(mrna, reg.chromosome, reg.nucleoplasm, dc.mrna)
    
    
    if enable_rna_tracking:
        if enable_er:
            # Special handling for ER version
            for mrna in sim.speciesList.matchRegex("tracker_R.*"):
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
            sim.transitionRate(sp.R2, reg.cytoplasm, reg.cytoplasm, dc.mrna)
            sim.transitionRate(sp.R2, reg.nucleoplasm, reg.nucleoplasm, dc.mrna)
            # sim.transitionRate(sp.R2, reg.cytoplasm, reg.erRibosomes, dc.mrna)
            # sim.transitionRate(sp.R2, reg.erRibosomes, reg.cytoplasm, dc.mrna)
            # sim.transitionRate(sp.R2, reg.erRibosomes, reg.erRibosomes, dc.mrna)
            sim.transitionRate(sp.R2, reg.cytoplasm, reg.pmaRibosomes, dc.mrna)
            sim.transitionRate(sp.R2, reg.pmaRibosomes, reg.cytoplasm, dc.mrna)
            sim.transitionRate(sp.R2, reg.pmaRibosomes, reg.pmaRibosomes, dc.mrna)
            sim.transitionRate(sp.R2, reg.cytoplasm, reg.cecRibosomes, dc.mrna)
            sim.transitionRate(sp.R2, reg.cecRibosomes, reg.cytoplasm, dc.mrna)
            sim.transitionRate(sp.R2, reg.cecRibosomes, reg.cecRibosomes, dc.mrna)
            sim.transitionRate(sp.R2, reg.cytoplasm, reg.tubRibosomes, dc.mrna)
            sim.transitionRate(sp.R2, reg.tubRibosomes, reg.cytoplasm, dc.mrna)
            sim.transitionRate(sp.R2, reg.tubRibosomes, reg.tubRibosomes, dc.mrna)
        else:
            # Standard mRNA diffusion
            for mrna in sim.speciesList.matchRegex("tracker_R.*"):
                sim.transitionRate(mrna, reg.nucleoplasm, reg.cytoplasm, dc.mrna)
                sim.transitionRate(mrna, reg.cytoplasm, reg.nucleoplasm, sim.diffusionZero)
                sim.transitionRate(mrna, reg.nucleoplasm, reg.nucleoplasm, dc.mrna)
                sim.transitionRate(mrna, reg.cytoplasm, reg.cytoplasm, dc.mrna)
                sim.transitionRate(mrna, reg.ribosomes, reg.ribosomes, dc.mrna)
                sim.transitionRate(mrna, reg.ribosomes, reg.cytoplasm, dc.mrna)
                sim.transitionRate(mrna, reg.cytoplasm, reg.ribosomes, dc.mrna)
    
        # Chromosome-specific mRNA diffusion
        if enable_chromosome:
            for mrna in sim.speciesList.matchRegex("tracker_R.*"):
                sim.transitionRate(mrna, reg.chromosome, reg.chromosome, dc.mrna)
                sim.transitionRate(mrna, reg.chromosome, reg.nucleoplasm, dc.mrna)
    # Protein diffusion
    
    sim.diffusionConst("prot", 1e-12, texRepr=r'D_{prot}', annotation='Generic protein')
    # general ribosome diffusion
    sim.diffusionConst("ribo", 3e-13, texRepr=r'D_{ribosome}', annotation='Generic ribosome')
    if enable_er:
        
        # General protein diffusion for ER version
        for sps in [sp.G1, sp.G3, sp.G3i, sp.G4, sp.G4d, sp.G80, sp.G80d, sp.G80d_G3i, sp.Grep]:
            sim.transitionRate(sps, reg.cytoplasm, reg.cytoplasm, dc.prot)
            # sim.transitionRate(sps, reg.cytoRibosomes, reg.cytoRibosomes, sim.diffusionFast)
            sim.transitionRate(sps, reg.cytoRibosomes, reg.cytoplasm, sim.diffusionFast)
            sim.transitionRate(sps, reg.cytoplasm, reg.cytoRibosomes, sim.diffusionZero)
    
        # G2 special ER handling
        # sim.transitionRate(sp.G2, reg.erRibosomes, reg.endoplasmicReticulum, sim.diffusionFast)
        sim.transitionRate(sp.G2, reg.pmaRibosomes, reg.endoplasmicReticulum, sim.diffusionFast)
        sim.transitionRate(sp.G2, reg.cecRibosomes, reg.endoplasmicReticulum, sim.diffusionFast)
        sim.transitionRate(sp.G2, reg.tubRibosomes, reg.endoplasmicReticulum, sim.diffusionFast)
        sim.transitionRate(sp.G2, reg.endoplasmicReticulum, reg.endoplasmicReticulum, dc.prot)
    
        # sim.transitionRate(sp.G2, reg.endoplasmicReticulum, reg.erRibosomes, sim.diffusionZero)
        sim.transitionRate(sp.G2, reg.endoplasmicReticulum, reg.pmaRibosomes, sim.diffusionZero)
        sim.transitionRate(sp.G2, reg.endoplasmicReticulum, reg.cecRibosomes, sim.diffusionZero)
        sim.transitionRate(sp.G2, reg.endoplasmicReticulum, reg.tubRibosomes, sim.diffusionZero)
        sim.transitionRate(sp.G2, reg.endoplasmicReticulum, reg.pmaER, dc.prot)
        sim.transitionRate(sp.G2, reg.pmaER, reg.endoplasmicReticulum, sim.diffusionZero)
        sim.transitionRate(sp.G2, reg.pmaER, reg.cytoplasm, sim.diffusionFast)
        sim.transitionRate(sp.G2, reg.pmaER, reg.pmaER, dc.prot)
        sim.transitionRate(sp.G2, reg.cytoplasm, reg.pmaER, sim.diffusionFast) # no diffusion back
    else:
        # Standard protein diffusion
        for sps in [sp.G1, sp.G2, sp.G3, sp.G3i, sp.G4, sp.G4d, sp.G80, sp.G80d, sp.G80d_G3i, sp.Grep]:
            sim.transitionRate(sps, reg.cytoplasm, reg.cytoplasm, dc.prot)
            # sim.transitionRate(sps, reg.ribosomes, reg.ribosomes, sim.diffusionFast)
            sim.transitionRate(sps, reg.ribosomes, reg.cytoplasm, sim.diffusionFast)
            sim.transitionRate(sps, reg.cytoplasm, reg.ribosomes, sim.diffusionZero)
    
    # Transcription factors
    
    for sps in [sp.G4, sp.G4d, sp.G80, sp.G80d]:
        sim.transitionRate(sps, reg.nucleoplasm, reg.nucleoplasm, dc.prot)
        sim.transitionRate(sps, reg.nucleoplasm, reg.cytoplasm, dc.prot)
        sim.transitionRate(sps, reg.cytoplasm, reg.nucleoplasm, dc.prot)
        
        if enable_chromosome:
            sim.transitionRate(sps, reg.chromosome, reg.chromosome, dc.prot)
            sim.transitionRate(sps, reg.chromosome, reg.nucleoplasm, dc.prot)
            sim.transitionRate(sps, reg.nucleoplasm, reg.chromosome, dc.prot)
    
    # Cytoplasmic proteins - prevent nuclear diffusion  
    
    for sps in [sp.G1, sp.G2, sp.G3, sp.G3i, sp.G80d_G3i, sp.Grep]:
        sim.transitionRate(sps, reg.cytoplasm, reg.nucleoplasm, sim.diffusionZero)
    
    # Membrane transporter
    
    sim.transitionRate(sp.G2, reg.cytoplasm, reg.plasmaMembrane, dc.prot)
    if enable_er:
        sim.transitionRate(sp.G2, reg.pmaER, reg.plasmaMembrane, dc.prot)
    sim.transitionRate(sp.G2, reg.plasmaMembrane, reg.cytoplasm, sim.diffusionZero)
    sim.diffusionConst("mem", 0.01e-12, texRepr=r'D_{mem}', annotation='Generic protein on membrane')
    sim.transitionRate(sp.G2, reg.plasmaMembrane, reg.plasmaMembrane, dc.mem)
    
    # Ribosomes - movement configuration
    if enable_ribosome_movement and ribosome_movement_mode == 'diffusion':
        # Enable limited diffusion within ribosome regions only
        # This is the fastest option as it's handled by the solver
        sim.diffusionConst("ribo_move", ribosome_diffusion_rate, texRepr=r'D_{ribosome_move}', annotation='Ribosome movement diffusion')
        
        # Get all ribosome regions
        if enable_er:
            if enable_effective_ribosome:
                ribo_regions_list = [reg.cytoRibosomes, reg.erRibosomes]
            else:
                ribo_regions_list = [reg.cytoRibosomes, reg.pmaRibosomes, reg.cecRibosomes, reg.tubRibosomes]
        else:
            ribo_regions_list = [reg.ribosomes]
        
        # Allow diffusion only within ribosome regions
        for sps in sim.speciesList.matchRegex("ribosome.*"):
            for ribo_reg in ribo_regions_list:
                # Allow movement within the same ribosome region
                sim.transitionRate(sps, ribo_reg, ribo_reg, dc.ribo_move)
            # Prevent movement outside ribosome regions
            sim.transitionRate(sps, None, None, sim.diffusionZero)
            # Explicitly block cross-region movement
            for ribo_reg1 in ribo_regions_list:
                for ribo_reg2 in ribo_regions_list:
                    if ribo_reg1 != ribo_reg2:
                        sim.transitionRate(sps, ribo_reg1, ribo_reg2, sim.diffusionZero)
    else:
        # Ribosomes - fixed (default behavior)
        for sps in sim.speciesList.matchRegex("ribosome.*"):
            sim.transitionRate(sps, None, None, sim.diffusionZero)
    
    # Import ODE hybrid solver components
    # import json
    # import scipy.integrate as spi
    

    return sim
