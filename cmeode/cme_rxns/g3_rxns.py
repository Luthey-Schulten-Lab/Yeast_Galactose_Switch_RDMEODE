"""
G3 activation

@param sim The simulation object to which the reactions will be added
@param frxns The reactions whose propensities will be updated by ODE solutions
@param ks The rate constants of the frxns, should be stored
"""

def getG3Reactions(sim,frxns,ks):
    Kfi = 7.45e-7 # molec^-1 min^-1
    Kri = 890.0 # min^-1
    Kfd3i80 = 0.025716 # molec^-1 min^-1
    Kdr3i80 = 0.0159616 # min^-1
    kdp_gal3 = 0.01155 # min^-1

    # Note that since we're defining a GAI species in the CME representation and updating it
    # in the hook, we don't need to modify the reaction rate.
    sim.addReaction(reactant='G3i', product=('GAI','G3'), rate=Kri)
    sim.addReaction(reactant=('GAI','G3'), product='G3i', rate=Kfi)
    # G3i -> GAI (G3 degradation while in G3i form)
    sim.addReaction(reactant='G3i',product='GAI',rate=kdp_gal3)
    # G80Cd + G3i -> G80G3i (Sequestration of G80Cd in the cytoplasm)
    sim.addReaction(reactant=('G80Cd', 'G3i'), product='G80G3i', rate=Kfd3i80)

    # G80G3i -> G3i + G80Cd (Unbinding of G80 from G3i)
    sim.addReaction(reactant='G80G3i', product=('G80Cd', 'G3i'), rate=Kdr3i80)

    # This reaction may be incorrect in ODE Model
    # Maybe should be:
    # G80G3i -> G80 + GAI (G3 degradation while in G80G3i form)
    # Implemented as:
    # G80G3i -> Null (G3 degradation while in G80G3i form)
    sim.addReaction(reactant='G80G3i',product='',rate=0.5*kdp_gal3)