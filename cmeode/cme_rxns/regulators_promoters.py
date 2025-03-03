"""
The DNA-promoter Reactions.

The reaction rate constants are scaled to represent binding probabilities
for a system with multiple binding sites, even though this is a single 
binding site representation.

The rate parameters were optimized to match experimental results in

Ramsey, S., Orell, D., Smith, J.J, Marelli, M., Petersen, T.W.,
de Atuari, P., Bolouri, H., Aitchison, J.D., Dual feedback loops
in the GAL regulon supress cellular heterogeneity in yeast,
NATURE GENETICS. 435 (2005) pp. 228-232

@param sim The simulation object to which the reactions will be passed
"""
def getDNAPromoterReactions(sim):

    # Rate constants for genes with single binding sites, proteins: G3, G80
    Kp = 0.0248
    Kq = 0.1885
    kf1 = 0.1
    kr1 = kf1/Kp
    kf2 = 0.1
    kr2 = kf2/Kq

    # Rate constants for genes with 4 binding sites, proteins: G1, reporter
    Kp4 = 0.2600
    Kq4 = 1.1721
    kf1_4 = 0.1
    kf2_4 = 0.1
    kr1_4 = kf1_4/Kp4
    kr2_4 = kf2_4/Kq4

    # Rate constants for genes with 5 binding sites, proteins: G2
    Kp5 = 0.0099
    Kq5 = 0.7408
    kf1_5 = 0.1
    kf2_5 = 0.1
    kr1_5 = kf1_5/Kp5
    kr2_5 = kf2_5/Kq5


    """ Reactions for DG1 """
    # DG1 + G4d -> DG1_G4d (G1 DNA-promoter binding)
    sim.addReaction(reactant=('DG1','G4d'),product='DG1_G4d',rate=kf1_4)

    # DG1_G4d -> G4d + DG1 (G1 DNA-promoter unbinding)
    sim.addReaction(reactant='DG1_G4d',product=('DG1','G4d'),rate=kr1_4)

    # DG1_G4d + G80d -> DG1_G4d_G80d (binding with the repressor)
    sim.addReaction(reactant=('DG1_G4d','G80d'),product='DG1_G4d_G80d',rate=kf2_4)

    # DG1_G4d_G80d -> DG1_G4d + G80d (unbinding of the repressor)
    sim.addReaction(reactant='DG1_G4d_G80d',product=('DG1_G4d','G80d'),rate=kr2_4)


    """ Reactions for DG2 """
    # DG2 + G4d -> DG2_G4d (DG2 DNA-promoter binding)
    sim.addReaction(reactant=('DG2','G4d'),product='DG2_G4d',rate=kf1_5)

    # DG2_G4d -> G4d + DG2 (G2 DNA-promoter unbinding)
    sim.addReaction(reactant='DG2_G4d',product=('DG2','G4d'),rate=kr1_5)

    # DG2_G4d + G80d -> DG2_G4d_G80d (binding with the repressor)
    sim.addReaction(reactant=('DG2_G4d','G80d'),product='DG2_G4d_G80d',rate=kf2_5)

    # DG2_G4d_G80d -> DG2_G4d + G80d (unbinding of the repressor)
    sim.addReaction(reactant='DG2_G4d_G80d',product=('DG2_G4d','G80d'),rate=kr2_5)


    """ Reactions for DG3 """
    # DG3 + G4d -> DG3_G4d DG3 DNA-promoter binding)
    sim.addReaction(reactant=('DG3','G4d'),product='DG3_G4d',rate=kf1)

    # DG3_G4d -> G4d + DG3 (G3 DNA-promoter unbinding)
    sim.addReaction(reactant='DG3_G4d',product=('DG3','G4d'),rate=kr1)

    # DG3_G4d + G80d -> DG3_G4d_G80d (binding with the repressor)
    sim.addReaction(reactant=('DG3_G4d','G80d'),product='DG3_G4d_G80d',rate=kf2)

    # DG3_G4d_G80d -> DG3_G4d + G80d (unbinding of the repressor)
    sim.addReaction(reactant='DG3_G4d_G80d',product=('DG3_G4d','G80d'),rate=kr2)

    """ Reactions for DGrep """
    # DGrep + G4d -> DGrep_G4d DGrep DNA-promoter binding)
    sim.addReaction(reactant=('DGrep','G4d'),product='DGrep_G4d',rate=kf1_4)

    # DGrep_G4d -> G4d + DGrep (G1 DNA-promoter unbinding)
    sim.addReaction(reactant='DGrep_G4d',product=('DGrep','G4d'),rate=kr1_4)

    # DGrep_G4d + G80d -> DGrep_G4d_G80d (binding with the repressor)
    sim.addReaction(reactant=('DGrep_G4d','G80d'),product='DGrep_G4d_G80d',rate=kf2_4)

    # DGrep_G4d_G80d -> DGrep_G4d + G80d (unbinding of the repressor)
    sim.addReaction(reactant='DGrep_G4d_G80d',product=('DGrep_G4d','G80d'),rate=kr2_4)


    """ Reactions for DG80 """
    # DG80 + G4d -> DG80_G4d DG80 DNA-promoter binding)
    sim.addReaction(reactant=('DG80','G4d'),product='DG80_G4d',rate=kf1)

    # DG80_G4d -> G4d + DG80 (G1 DNA-promoter unbinding)
    sim.addReaction(reactant='DG80_G4d',product=('DG80','G4d'),rate=kr1)

    # DG80_G4d + G80d -> DG80_G4d_G80d (binding with the repressor)
    sim.addReaction(reactant=('DG80_G4d','G80d'),product='DG80_G4d_G80d',rate=kf2)

    # DG80_G4d_G80d -> DG80_G4d + G80d (unbinding of the repressor)
    sim.addReaction(reactant='DG80_G4d_G80d',product=('DG80_G4d','G80d'),rate=kr2)


def getDNAPromoterReactions_non_reg(sim):

    # Rate constants for genes with single binding sites, proteins: G3, G80
    Kp = 0.0248
    Kq = 0.1885
    kf1 = 0.1
    kr1 = kf1/Kp
    kf2 = 0.1
    kr2 = kf2/Kq

    # Rate constants for genes with 4 binding sites, proteins: G1, reporter
    Kp4 = 0.2600
    Kq4 = 1.1721
    kf1_4 = 0.1
    kf2_4 = 0.1
    kr1_4 = kf1_4/Kp4
    kr2_4 = kf2_4/Kq4

    # Rate constants for genes with 5 binding sites, proteins: G2
    Kp5 = 0.0099
    Kq5 = 0.7408
    kf1_5 = 0.1
    kf2_5 = 0.1
    kr1_5 = kf1_5/Kp5
    kr2_5 = kf2_5/Kq5


    """ Reactions for DG1 """
    # DG1 + G4d -> DG1_G4d (G1 DNA-promoter binding)
    sim.addReaction(reactant=('DG1','G4d'),product='DG1_G4d',rate=kf1_4)

    # DG1_G4d -> G4d + DG1 (G1 DNA-promoter unbinding)
    sim.addReaction(reactant='DG1_G4d',product=('DG1','G4d'),rate=kr1_4)

    # DG1_G4d + G80d -> DG1_G4d_G80d (binding with the repressor)
    sim.addReaction(reactant=('DG1_G4d','G80d'),product='DG1_G4d_G80d',rate=kf2_4)

    # DG1_G4d_G80d -> DG1_G4d + G80d (unbinding of the repressor)
    sim.addReaction(reactant='DG1_G4d_G80d',product=('DG1_G4d','G80d'),rate=kr2_4)


    """ Reactions for DG2 """
    # DG2 + G4d -> DG2_G4d (DG2 DNA-promoter binding)
    sim.addReaction(reactant=('DG2','G4d'),product='DG2_G4d',rate=kf1_5)

    # DG2_G4d -> G4d + DG2 (G2 DNA-promoter unbinding)
    sim.addReaction(reactant='DG2_G4d',product=('DG2','G4d'),rate=kr1_5)

    # DG2_G4d + G80d -> DG2_G4d_G80d (binding with the repressor)
    sim.addReaction(reactant=('DG2_G4d','G80d'),product='DG2_G4d_G80d',rate=kf2_5)

    # DG2_G4d_G80d -> DG2_G4d + G80d (unbinding of the repressor)
    sim.addReaction(reactant='DG2_G4d_G80d',product=('DG2_G4d','G80d'),rate=kr2_5)


    """ Reactions for DG3 """
    # # DG3 + G4d -> DG3_G4d DG3 DNA-promoter binding)
    # sim.addReaction(reactant=('DG3','G4d'),product='DG3_G4d',rate=kf1)

    # # DG3_G4d -> G4d + DG3 (G3 DNA-promoter unbinding)
    # sim.addReaction(reactant='DG3_G4d',product=('DG3','G4d'),rate=kr1)

    # # DG3_G4d + G80d -> DG3_G4d_G80d (binding with the repressor)
    # sim.addReaction(reactant=('DG3_G4d','G80d'),product='DG3_G4d_G80d',rate=kf2)

    # # DG3_G4d_G80d -> DG3_G4d + G80d (unbinding of the repressor)
    # sim.addReaction(reactant='DG3_G4d_G80d',product=('DG3_G4d','G80d'),rate=kr2)

    """ Reactions for DGrep """
    # DGrep + G4d -> DGrep_G4d DGrep DNA-promoter binding)
    sim.addReaction(reactant=('DGrep','G4d'),product='DGrep_G4d',rate=kf1_4)

    # DGrep_G4d -> G4d + DGrep (G1 DNA-promoter unbinding)
    sim.addReaction(reactant='DGrep_G4d',product=('DGrep','G4d'),rate=kr1_4)

    # DGrep_G4d + G80d -> DGrep_G4d_G80d (binding with the repressor)
    sim.addReaction(reactant=('DGrep_G4d','G80d'),product='DGrep_G4d_G80d',rate=kf2_4)

    # DGrep_G4d_G80d -> DGrep_G4d + G80d (unbinding of the repressor)
    sim.addReaction(reactant='DGrep_G4d_G80d',product=('DGrep_G4d','G80d'),rate=kr2_4)


    """ Reactions for DG80 """
    # DG80 + G4d -> DG80_G4d DG80 DNA-promoter binding)
    # sim.addReaction(reactant=('DG80','G4d'),product='DG80_G4d',rate=kf1)

    # # DG80_G4d -> G4d + DG80 (G1 DNA-promoter unbinding)
    # sim.addReaction(reactant='DG80_G4d',product=('DG80','G4d'),rate=kr1)

    # # DG80_G4d + G80d -> DG80_G4d_G80d (binding with the repressor)
    # sim.addReaction(reactant=('DG80_G4d','G80d'),product='DG80_G4d_G80d',rate=kf2)

    # # DG80_G4d_G80d -> DG80_G4d + G80d (unbinding of the repressor)
    # sim.addReaction(reactant='DG80_G4d_G80d',product=('DG80_G4d','G80d'),rate=kr2)


