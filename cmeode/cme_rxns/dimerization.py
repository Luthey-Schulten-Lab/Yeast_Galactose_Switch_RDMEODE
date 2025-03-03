"""
The Dimerization reactions

@param sim The simulation object the reactions will be added to
"""

def getDimerizationReactions(sim):
    # Kfd = 0.001 # molec^-1 min^-1
    # Krd = 100 # min^-1
    # original results 
    Kfd = 100 # molec^-1 min^-1
    Krd = 0.001 # min^-1
    kdp_gal4 = 0.006931 # min^-1
    kdp_gal80 = 0.006931 # min^-1

    # G4 + G4 -> G4d (Dimerization of G4)
    sim.addReaction(reactant=('G4','G4'),product='G4d',rate=Kfd)

    # G4d -> G4 + G4 (Unbinding of G4 molecules)
    sim.addReaction(reactant='G4d',product=('G4','G4'),rate=Krd)

    # G4d -> Null (Degradation of G4d)
    sim.addReaction(reactant='G4d',product='',rate=kdp_gal4)

    # G80 + G80 -> G80d (Dimerization of G80)
    sim.addReaction(reactant=('G80','G80'),product='G80d',rate=Kfd)

    # G80d -> G80 + G80 (Unbinding of G80 molecules)
    sim.addReaction(reactant='G80d',product=('G80','G80'),rate=Krd)

    # G80d -> Null (Degradation of G80d)
    sim.addReaction(reactant='G80d',product='',rate=kdp_gal80)

    # G80C + G80C -> G80Cd (Dimerization of G80 in the cytoplasm)
    sim.addReaction(reactant=('G80C','G80C'),product='G80Cd',rate=Kfd)

    # G80Cd -> G80C + G80C
    sim.addReaction(reactant='G80Cd',product=('G80C','G80C'),rate=Krd)

    # G80Cd -> Null (Degradation of G80Cd)
    sim.addReaction(reactant='G80Cd',product='',rate=kdp_gal80)