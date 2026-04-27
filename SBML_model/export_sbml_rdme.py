#!/usr/bin/env python
import libsbml
import sys

def check(value, message):
    if value is None:
        print(f"Error: {message} returned None")
        sys.exit(1)
    if isinstance(value, int) and value != libsbml.LIBSBML_OPERATION_SUCCESS:
        print(f"Error: {message} failed. Code: {value}")
        sys.exit(1)

def export_rdme_ode_sbml(filename="galactose_rdme_ode.xml"):
    try:
        document = libsbml.SBMLDocument(3, 1)
    except ValueError:
        print("Could not create SBMLDocument object")
        sys.exit(1)

    model = document.createModel()
    check(model, "createModel")
    check(model.setId("yeast_galactose_rdme_ode"), "setId on Model")
    check(model.setName("Galactose Switch RDME-ODE Model"), "setName on Model")

    def add_compartment(m, cid, name, size, npy_file):
        c = m.createCompartment()
        c.setId(cid)
        c.setName(name)
        c.setSpatialDimensions(3)
        c.setConstant(True)
        c.setSize(size)
        
        if npy_file:
            annotation = f"""<annotation>
                <geometry_link xmlns="http://www.simulationobjects.com/yeastgs">
                    <file path="{npy_file}"/>
                </geometry_link>
            </annotation>"""
            c.setAnnotation(annotation)
        return c

    add_compartment(model, "extracellular", "Extracellular Space", 1.0, "yeast-lattice.2.pkl.xz")
    add_compartment(model, "cellWall", "Cell Wall", 0.05, "yeast-lattice.2.pkl.xz")
    add_compartment(model, "nuclearEnvelope", "Nuclear Envelope", 0.01, "yeast-lattice.2.pkl.xz")
    add_compartment(model, "mitochondria", "Mitochondria", 0.1, "yeast-lattice.2.pkl.xz")
    add_compartment(model, "vacuole", "Vacuoles", 0.2, "yeast-lattice.2.pkl.xz")
    add_compartment(model, "plasmaMembrane", "Plasma Membrane", 0.01, "workspace/plasmMembrane_connected.npy")
    add_compartment(model, "nucleoplasm", "Nucleoplasm", 0.1, "yeast-lattice.2.pkl.xz")
    add_compartment(model, "cytoplasm", "Cytoplasm", 0.5, "yeast-lattice.2.pkl.xz")
    add_compartment(model, "endoplasmicReticulum", "Endoplasmic Reticulum", 0.05, "workspace/ER_geometry/cecER_fixed_geometry.npy")
    add_compartment(model, "pmaER", "Plasma Membrane Associated ER", 0.02, "workspace/ER_geometry/pmER_fixed_geometry.npy")
    add_compartment(model, "chromosome", "Chromosome", 0.01, "workspace/gene_masks.npy")
    add_compartment(model, "cytoRibosomes", "Cytosolic Ribosomes", 0.05, "workspace/effective_cyto_ribosomes_ER_Marie_4152.npy")
    add_compartment(model, "pmaRibosomes", "PMA Ribosomes", 0.01, "workspace/effective_pma_ribosomes_ER_Marie_4256.npy")

    cellVol = 3.57e-14
    nav = cellVol * 6.022e23
    invMin2invSec = 1/60.0
    conv2ndOrder = invMin2invSec * nav
    conv1stOrder = invMin2invSec

    def add_parameter(m, pid, val):
        p = m.createParameter()
        p.setId(pid)
        p.setValue(val)
        p.setConstant(True)

    def add_species(m, sid, comp, init_amt, diff_coeff=0.0):
        s = m.createSpecies()
        s.setId(sid)
        s.setCompartment(comp)
        s.setInitialAmount(init_amt)
        s.setHasOnlySubstanceUnits(True)
        s.setBoundaryCondition(False)
        s.setConstant(False)
        
        # Attach Diffusion coefficient to annotation
        annotation = f"""<annotation>
            <spatial_properties xmlns="http://www.simulationobjects.com/yeastgs">
                <diffusion_coefficient value="{diff_coeff}" unit="m2/s"/>
            </spatial_properties>
        </annotation>"""
        s.setAnnotation(annotation)
        return s

    def add_reaction(m, rid, reactants, products, param_id, sim_type="RDME", comp="cytoplasm", spaces="cytoplasm"):
        r = m.createReaction()
        r.setId(rid)
        r.setReversible(False)
        r.setCompartment(comp)
        
        annot = f"""<annotation>
            <simulation_engine xmlns="http://www.simulationobjects.com/yeastgs" type="{sim_type}">
                <reaction_space valid_compartments="{spaces}"/>
            </simulation_engine>
        </annotation>"""
        r.setAnnotation(annot)
        
        formula = param_id
        for spe, stoich in reactants:
            s_ref = r.createReactant()
            s_ref.setSpecies(spe)
            s_ref.setStoichiometry(stoich)
            s_ref.setConstant(True)
            for _ in range(stoich):
                formula += f" * {spe}"
        for spe, stoich in products:
            p_ref = r.createProduct()
            p_ref.setSpecies(spe)
            p_ref.setStoichiometry(stoich)
            p_ref.setConstant(True)
            
        kinetics = r.createKineticLaw()
        math_ast = libsbml.parseL3Formula(formula)
        kinetics.setMath(math_ast)
        return r

    params = [
        ("fd", 100 * conv2ndOrder), ("rd", 0.001 * conv1stOrder), ("f1_4", 0.1 * conv2ndOrder),
        ("r1_4", (0.1/0.2600)/100 * conv1stOrder), ("f2_4", 0.1/100 * conv2ndOrder),
        ("r2_4", (0.1/1.1721) * conv1stOrder), ("f1_5", 0.1 * conv2ndOrder),
        ("r1_5", (0.1/0.0099)/100 * conv1stOrder), ("f2_5", 0.1/100 * conv2ndOrder),
        ("r2_5", (0.1/0.7408) * conv1stOrder), ("f1", 0.1 * conv2ndOrder),
        ("r1", (0.1/0.0248)/100 * conv1stOrder), ("f2", 0.1/100 * conv2ndOrder),
        ("r2", (0.1/0.1885) * conv1stOrder), ("fi", 7.45e-7 * conv2ndOrder),
        ("ri", 890.0 * conv1stOrder), ("fd3i80", 0.025716 * conv2ndOrder),
        ("dr3i80", 0.0159616 * conv1stOrder), ("dp_gal3", 0.01155 * conv1stOrder),
        ("dp_gal3gal80", 0.01155 * conv1stOrder * 0.5), ("alpha1", 0.7379 * conv1stOrder),
        ("alpha2", 2.542 * conv1stOrder), ("alpha3", 0.571429 * 0.7465 * conv1stOrder),
        ("ir_gal4", 0.009902 * conv1stOrder), ("alpha_rep", 1.1440 * conv1stOrder),
        ("alpha80", 0.6065 * conv1stOrder), ("dr_gal1", 0.02236 * conv1stOrder),
        ("dr_gal2", 0.07702 * conv1stOrder), ("dr_gal3", 0.02666 * conv1stOrder),
        ("dr_gal4", 0.02476 * conv1stOrder), ("dr_rep", 0.03466 * conv1stOrder),
        ("dr_gal80", 0.02888 * conv1stOrder), ("ip_gal1", 1.9254 * conv1stOrder),
        ("ip_gal2", 13.4779 * conv1stOrder), ("ip_gal3", 55.4518 * conv1stOrder),
        ("ip_gal4", 10.7091 * conv1stOrder), ("ip_rep", 5.7762 * conv1stOrder),
        ("ip_gal80", 3.6737 * conv1stOrder), ("dp_gal1", 0.003851 * conv1stOrder),
        ("dp_gal2", 0.003851 * conv1stOrder), ("dp_gal4", 0.006931 * conv1stOrder),
        ("dp_rep", 0.01155 * conv1stOrder), ("dp_gal80", 0.006931 * conv1stOrder),
        ("k_TR", 4350), ("kr_TR", 2.3925e3), ("kf_TR", 3.1353e-4),
        ("kf_GK", 4.0243e-4), ("kr_GK", 1.8425e3), ("kcat_GK", 3350)
    ]
    for pid, val in params:
        add_parameter(model, pid, val)

    mRNADiffusion = 0.05e-12
    protDiffusion = 1e-12
    riboDiffusion = 3e-13
    
    # ODE species
    add_species(model, "GAE", "extracellular", 100000, diff_coeff=1e-11)
    add_species(model, "GAI", "cytoplasm", 1, diff_coeff=1e-11)

    # General Proteins
    for sp, cnt, c in [("G1", 132, "cytoplasm"), ("G2", 1157, "plasmaMembrane"), ("G3", 4342, "cytoplasm"),
                       ("G3i", 0, "cytoplasm"), ("G4", 1, "nucleoplasm"), ("G4d", 309, "nucleoplasm"),
                       ("G80", 70, "cytoplasm"), ("G80d", 200, "nucleoplasm"), ("G80G3i", 0, "cytoplasm"),
                       ("G2GAE", 0, "plasmaMembrane"), ("G2GAI", 0, "plasmaMembrane"), ("G1GAI", 0, "cytoplasm")]:
        add_species(model, sp, c, cnt, diff_coeff=protDiffusion)

    # mRNAs
    for sp in ['R1','R2','R3','R4','R80']:
        add_species(model, sp, "nucleoplasm", 1, diff_coeff=mRNADiffusion)

    add_species(model, "ribosome", "cytoRibosomes", 20000, diff_coeff=riboDiffusion)

    # Genes and States (No diffusion: 0.0)
    for gn in ['DG1','DG1_G4d','DG1_G4d_G80d','DG2','DG2_G4d','DG2_G4d_G80d',
               'DG3','DG3_G4d','DG3_G4d_G80d','DG80','DG80_G4d','DG80_G4d_G80d']:
        add_species(model, gn, "chromosome", 1 if '_' not in gn else 0, diff_coeff=0.0)
        
    # Example Transport and Enzymes (ODE)
    add_reaction(model, "ode_G2_GAE_bind", [("GAE", 1), ("G2", 1)], [("G2GAE", 1)], "kf_TR", sim_type="ODE", comp="plasmaMembrane", spaces="extracellular,plasmaMembrane")
    add_reaction(model, "ode_G2GAE_unbind", [("G2GAE", 1)], [("GAE", 1), ("G2", 1)], "kr_TR", sim_type="ODE", comp="plasmaMembrane", spaces="extracellular,plasmaMembrane")
    add_reaction(model, "ode_G1_metabolize", [("G1GAI", 1)], [("G1", 1)], "kcat_GK", sim_type="ODE", comp="cytoplasm", spaces="cytoplasm")

    # Example Stochastic Reactions (RDME)
    add_reaction(model, "g4_dimerize", [("G4", 2)], [("G4d", 1)], "fd", sim_type="RDME", comp="nucleoplasm", spaces="nucleoplasm")
    add_reaction(model, "g1_g4_bind", [("DG1", 1), ("G4d", 1)], [("DG1_G4d", 1)], "f1_4", sim_type="RDME", comp="chromosome", spaces="chromosome")
    add_reaction(model, "trans_g1", [("DG1_G4d", 1)], [("DG1_G4d", 1), ("R1", 1)], "alpha1", sim_type="RDME", comp="chromosome", spaces="chromosome")
    add_reaction(model, "transl_g1", [("R1", 1)], [("R1", 1), ("G1", 1)], "ip_gal1", sim_type="RDME", comp="cytoplasm", spaces="cytoRibosomes,pmaRibosomes")

    libsbml.writeSBMLToFile(document, filename)

if __name__ == "__main__":
    export_rdme_ode_sbml()
