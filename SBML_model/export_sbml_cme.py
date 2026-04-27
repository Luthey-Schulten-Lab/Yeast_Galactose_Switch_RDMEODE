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

def export_cme_ode_sbml(filename="galactose_cme_ode.xml"):
    try:
        document = libsbml.SBMLDocument(3, 1)
    except ValueError:
        print("Could not create SBMLDocument object")
        sys.exit(1)

    model = document.createModel()
    check(model, "createModel")
    model.setId("yeast_galactose_cme_ode")
    model.setName("Galactose Switch CME-ODE Model")

    # Define Compartments
    comp_cyto = model.createCompartment()
    comp_cyto.setId("cytoplasm")
    comp_cyto.setConstant(True)
    comp_cyto.setSize(3.57e-14)

    comp_nucl = model.createCompartment()
    comp_nucl.setId("nucleoplasm")
    comp_nucl.setConstant(True)
    comp_nucl.setSize(3.57e-14 * 0.1)

    # Coeffs
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

    def add_species(m, sid, comp, init_amt):
        s = m.createSpecies()
        s.setId(sid)
        s.setCompartment(comp)
        s.setInitialAmount(init_amt)
        s.setHasOnlySubstanceUnits(True)
        s.setBoundaryCondition(False)
        s.setConstant(False)

    def add_reaction(m, rid, reactants, products, param_id, sim_type="CME", comp="cytoplasm"):
        r = m.createReaction()
        r.setId(rid)
        r.setReversible(False)
        r.setCompartment(comp)
        
        # Add annotation to differentiate CME vs ODE
        annot = f"""<annotation>
            <simulation_engine xmlns="http://www.simulationobjects.com/yeastgs" type="{sim_type}"/>
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
        
        # --- ODE Transport & Enzymatic Parameters ---
        ("k_TR", 4350), ("kr_TR", 2.3925e3), ("kf_TR", 3.1353e-4),
        ("kf_GK", 4.0243e-4), ("kr_GK", 1.8425e3), ("kcat_GK", 3350)
    ]
    for pid, val in params:
        add_parameter(model, pid, val)

    cme_count_list = [1,1,1,1,1,1,132,1157,4342,0,1,309,132,1,1,157,157,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0]
    cme_species_names = ['R1','R2','R3','R4','reporter_rna','R80','G1','G2','G3','G3i','G4','G4d','reporter','G80','G80C','G80d','G80Cd','G80G3i','GAI','DG1','DG1_G4d','DG1_G4d_G80d','DG2','DG2_G4d','DG2_G4d_G80d','DG3','DG3_G4d','DG3_G4d_G80d','DGrep','DGrep_G4d','DGrep_G4d_G80d','DG80','DG80_G4d','DG80_G4d_G80d','G2GAI','G2GAE','G1GAI']
    
    # ODE Extra Species
    ext_species = ['GAE']
    add_species(model, "GAE", "cytoplasm", 100000)

    for count, name in zip(cme_count_list, cme_species_names):
        comp = "nucleoplasm" if "DG" in name else "cytoplasm"
        add_species(model, name, comp, int(count))

    # --- CME Reactions ---
    add_reaction(model, "g4_dimerize", [("G4", 2)], [("G4d", 1)], "fd", sim_type="CME")
    add_reaction(model, "g4_dedimerize", [("G4d", 1)], [("G4", 2)], "rd", sim_type="CME")
    add_reaction(model, "g1_g4_bind", [("DG1", 1), ("G4d", 1)], [("DG1_G4d", 1)], "f1_4", sim_type="CME", comp="nucleoplasm")
    add_reaction(model, "trans_g1", [("DG1_G4d", 1)], [("DG1_G4d", 1), ("R1", 1)], "alpha1", sim_type="CME", comp="nucleoplasm")
    add_reaction(model, "transl_g1", [("R1", 1)], [("R1", 1), ("G1", 1)], "ip_gal1", sim_type="CME")

    # --- ODE Reactions ---
    # G2 Transport
    add_reaction(model, "ode_G2_GAI_bind", [("GAI", 1), ("G2", 1)], [("G2GAI", 1)], "kf_TR", sim_type="ODE")
    add_reaction(model, "ode_G2GAI_unbind", [("G2GAI", 1)], [("GAI", 1), ("G2", 1)], "kr_TR", sim_type="ODE")
    
    add_reaction(model, "ode_G2_GAE_bind", [("GAE", 1), ("G2", 1)], [("G2GAE", 1)], "kf_TR", sim_type="ODE")
    add_reaction(model, "ode_G2GAE_unbind", [("G2GAE", 1)], [("GAE", 1), ("G2", 1)], "kr_TR", sim_type="ODE")
    
    add_reaction(model, "ode_G2GAE_to_G2GAI", [("G2GAE", 1)], [("G2GAI", 1)], "k_TR", sim_type="ODE")
    add_reaction(model, "ode_G2GAI_to_G2GAE", [("G2GAI", 1)], [("G2GAE", 1)], "k_TR", sim_type="ODE")
    
    # G1 Enzymatic Metabolism
    add_reaction(model, "ode_G1_GAI_bind", [("GAI", 1), ("G1", 1)], [("G1GAI", 1)], "kf_GK", sim_type="ODE")
    add_reaction(model, "ode_G1GAI_unbind", [("G1GAI", 1)], [("GAI", 1), ("G1", 1)], "kr_GK", sim_type="ODE")
    add_reaction(model, "ode_G1_metabolize", [("G1GAI", 1)], [("G1", 1)], "kcat_GK", sim_type="ODE") # GAI consumed

    libsbml.writeSBMLToFile(document, filename)

if __name__ == "__main__":
    export_cme_ode_sbml()
