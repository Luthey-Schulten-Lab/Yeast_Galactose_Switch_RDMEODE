"""
Galactose Operon Transcription 

In this case degradation refers not only to degradation but also to 
the dilution that will occur as a yeast cell divides over a cell cycle.

@param sim The simulation object to which the reactions will be added
"""



def getTranscriptionReactions(sim):
    
    # Transcription to form R1
    kalpha1 = 0.7379 # min^-1
    sim.addReaction(reactant='DG1_G4d', product=('R1','DG1_G4d'), rate=kalpha1)

    # Degradation of R1
    kdr_gal1 = 0.02236 # min^-1
    sim.addReaction(reactant='R1', product='', rate = kdr_gal1)

    # Transcription to form R2 
    kalpha2 = 2.542 # min^-1
    sim.addReaction(reactant='DG2_G4d', product=('R2','DG2_G4d'), rate=kalpha2)

    # Degradation of R2
    kdr_gal2 = 0.07702 # min^-1
    sim.addReaction(reactant='R2', product='', rate = kdr_gal2)

    # Transcription to from R3
    kalpha3 = 0.7465 # min^-1
    
    sim.addReaction(reactant='DG3_G4d', product=('R3','DG3_G4d'),rate=kalpha3*0.571429)

    # Degradation of R3
    kdr_gal3 = 0.02666 # min^-1
    sim.addReaction(reactant='R3', product='',rate=kdr_gal3)

    # Transcription of R4
    # R4 is constitutively expressed in this model
    kir_gal4 = 0.009902 # min^-1
    sim.addReaction(reactant='',product='R4',rate=kir_gal4)

    # Degradation of R4
    kdr_gal4 = 0.02476 # min^-1
    sim.addReaction(reactant='R4', product='', rate =kdr_gal4)

    # Transcription of reporter
    kalpha_rep = 1.1440 # min^-1
    sim.addReaction(reactant='DGrep_G4d',product=('reporter_rna','DGrep_G4d'), rate = kalpha_rep)

    # Degradation of reporter
    kdr_rep = 0.03466 # min^-1
    sim.addReaction(reactant='reporter_rna', product='', rate = kdr_rep)
    
    # Transcription of R80
    kalpha80 = 0.6065 # min^-1
    sim.addReaction(reactant='DG80_G4d', product=('R80','DG80_G4d'), rate = kalpha80)

    # Degradation of R80
    kdr_gal80 = 0.02888 # min^-1
    sim.addReaction(reactant='R80', product='', rate = kdr_gal80)

def getTranscriptionReactions_non_reg(sim, mutant_q=1.0,f_mut=0.04):
    
    # Transcription to form R1
    kalpha1 = 0.7379 # min^-1
    sim.addReaction(reactant='DG1_G4d', product=('R1','DG1_G4d'), rate=kalpha1)

    # Degradation of R1
    kdr_gal1 = 0.02236 # min^-1
    sim.addReaction(reactant='R1', product='', rate = kdr_gal1)

    # Transcription to form R2 
    kalpha2 = 2.542 # min^-1
    sim.addReaction(reactant='DG2_G4d', product=('R2','DG2_G4d'), rate=kalpha2)

    # Degradation of R2
    kdr_gal2 = 0.07702 # min^-1
    sim.addReaction(reactant='R2', product='', rate = kdr_gal2)

    # Transcription to form R3( we dont need regualtion here), G4d removed 
    kalpha3 = 0.7465 # min^-1
    # mutant_q = 1
    # f_mut = 0.04 # 0.04 ~0.06
    sim.addReaction(reactant='DG3', product=('R3','DG3'),rate=kalpha3*mutant_q*f_mut)
    # sim.addReaction(reactant='', product=('R3'),rate=kalpha3*mutant_q*f_mut)
    # Degradation of R3
    kdr_gal3 = 0.02666 # min^-1
    sim.addReaction(reactant='R3', product='',rate=kdr_gal3)

    # Transcription of R4
    # R4 is constitutively expressed in this model
    kir_gal4 = 0.009902 # min^-1
    sim.addReaction(reactant='',product='R4',rate=kir_gal4)

    # Degradation of R4
    kdr_gal4 = 0.02476 # min^-1
    sim.addReaction(reactant='R4', product='', rate =kdr_gal4)

    # Transcription of reporter
    kalpha_rep = 1.1440 # min^-1
    sim.addReaction(reactant='DGrep_G4d',product=('reporter_rna','DGrep_G4d'), rate = kalpha_rep)

    # Degradation of reporter
    kdr_rep = 0.03466 # min^-1
    sim.addReaction(reactant='reporter_rna', product='', rate = kdr_rep)
    
    # Transcription to form R80( we dont need regualtion here), G4d removed  
    kalpha80 = 0.6065 # min^-1
    sim.addReaction(reactant='DG80', product=('R80','DG80'), rate = kalpha80* f_mut)
    # sim.addReaction(reactant='', product=('R80'), rate = kalpha80 * f_mut)
    # Degradation of R80
    kdr_gal80 = 0.02888 # min^-1
    sim.addReaction(reactant='R80', product='', rate = kdr_gal80)
