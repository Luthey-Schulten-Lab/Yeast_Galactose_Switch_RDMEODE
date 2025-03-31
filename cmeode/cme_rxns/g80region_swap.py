"""
G80 region diffusion swap reactions: cytoplasm <=> nucleoplasm

symmetric rate constants

@param sim The simulation object to which the reactions will be added
"""
def getG80TransportReactions(sim):

    Kf80 = 500 # min^-1
    Kr80 = 500 # min^-1

    # G80Cd -> G80d (Transport of G80 from the cytoplasm into the nucleus)
    sim.addReaction(reactant='G80Cd',product='G80d',rate=Kf80)

    # G80d -> G80Cd (Transport of G80 from the nucleus into the cytoplasm)
    sim.addReaction(reactant='G80d',product='G80Cd',rate=Kr80)
