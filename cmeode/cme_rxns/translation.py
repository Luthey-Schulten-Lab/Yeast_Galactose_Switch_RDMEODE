"""
Gal operon Translation 

In this case degradation refers not only to degradation but also to 
the dilution that will occur as a yeast cell divides over a cell cycle.

@param sim The simulation object to which the reactions will be added
"""
def getTranslationReactions(sim):

        # Translation of G1
        kip_gal1 = 1.9254 # min^-1
        sim.addReaction(reactant='R1', product=('G1','R1'), rate=kip_gal1)

        # Degradation of G1
        kdp_gal1 = 0.003851 # min^-1
        sim.addReaction(reactant='G1', product='', rate=kdp_gal1)
        sim.addReaction(reactant='G1GAI', product='GAI', rate=kdp_gal1)

        # Translation of G2
        kip_gal2 = 13.4779 # min^-1
        sim.addReaction(reactant='R2', product=('G2','R2'),rate=kip_gal2)

        # Degradation of G2
        kdp_gal2 = 0.003851 # min^-1
        sim.addReaction(reactant='G2', product='', rate=kdp_gal2)
        sim.addReaction(reactant='G2GAE', product='', rate=kdp_gal2)
        sim.addReaction(reactant='G2GAI', product='GAI', rate=kdp_gal2)
    
        # Translation of G3
        kip_gal3 = 55.4518 # min^-1
        sim.addReaction(reactant='R3', product=('G3','R3'), rate=kip_gal3)

        # Degradation of G3
        kdp_gal3 = 0.01155 # min^-1
        sim.addReaction(reactant='G3', product='', rate=kdp_gal3)
        sim.addReaction(reactant='G3i',product='GAI',rate=kdp_gal3)

        # Translation of G4
        kip_gal4 = 10.7091 # min^-1
        sim.addReaction(reactant='R4', product=('G4','R4'), rate=kip_gal4)

        # Degradation of G4
        kdp_gal4 = 0.006931 # min^-1
        sim.addReaction(reactant='G4', product='', rate=kdp_gal4)
        
        # Translation of reporter
        kip_rep = 5.7762 # min^-1
        sim.addReaction(reactant='reporter_rna', product=('reporter','reporter_rna'), rate=kip_rep)

        # Degradation of reporter
        kdp_rep = 0.01155 # min ^-1
        sim.addReaction(reactant='reporter', product = '', rate=kdp_rep)

        # Translation of G80
        kip_gal80 = 3.6737 # min^-1
        sim.addReaction(reactant='R80', product=('G80','R80'), rate=kip_gal80)

        # Degradation of G80
        kdp_gal80 = 0.006931 # min^-1
        sim.addReaction(reactant='G80', product='', rate=kdp_gal80)