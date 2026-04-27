import jLM
import jLM.CME
import jLM.SBMLReader

# Create empty CME Simulation
sim = jLM.CME.CMESimulation()

print("Attempting to load SBML into CMESimulation...")
try:
    jLM.SBMLReader.readSBMLtoCME(sim, "galactose_cme_ode.xml")
    print("Successfully loaded the SBML file!")
    print(f"Number of Species in SIM: {len(sim.species)}")
except Exception as e:
    print(f"Error reading SBML: {e}")
