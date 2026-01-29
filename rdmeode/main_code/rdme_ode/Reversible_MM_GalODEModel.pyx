#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False
#cython: initializedcheck=False
#cython: embedsignature=True


## MODIFICATION notes
# The first 5 lines are directives which make the cython code more dangerous: i.e. segfaults possible if you mess up.
# If you are going to change the way things are indexed, it's good to remove the directives during testing and
# add them back later.
# The size of the species concentration vector is hardcoded in Reversible_MM_GalODEModel.__init__, if the species
# types change, this must be updated.

from libc.math cimport *
import numpy as np
cimport numpy as cnp

cdef class Reversible_MM_GalODEModel:
    cdef double[:] dxdt 
    cdef double[:] ys 
    cdef cnp.ndarray np_dxdt
    cdef cnp.ndarray np_ys
    cdef double GAE_mM

    def __init__(self, GAE_mM=0.5):
        self.GAE_mM = GAE_mM
        self.np_dxdt = np.zeros(37, dtype=np.float64)
        self.dxdt = self.np_dxdt
        self.np_ys = np.zeros(37, dtype=np.float64)
        self.ys = self.np_ys

    def __call__(self, double[:] y,  double t):
        self._rhs(y)
        return self.np_dxdt

    cdef void _rhs(self, double[:] x):
        cdef double R1 = x[0]
        cdef double R2 = x[1]
        cdef double R3 = x[2]
        cdef double R4 = x[3]
        cdef double reporter_rna = x[4]
        cdef double R80 = x[5]
        cdef double G1 = x[6]
        cdef double G2 = x[7]
        cdef double G3 = x[8]
        cdef double G3i = x[9]
        cdef double G4 = x[10]
        cdef double G4d = x[11]
        cdef double reporter = x[12]
        cdef double G80 = x[13]
        cdef double G80C = x[14]
        cdef double G80d = x[15]
        cdef double G80Cd = x[16]
        cdef double G80G3i = x[17]
        cdef double GAI = x[18]
        cdef double DG1 = x[19]
        cdef double DG1_G4d = x[20]
        cdef double DG1_G4d_G80d = x[21]
        cdef double DG2 = x[22]
        cdef double DG2_G4d = x[23]
        cdef double DG2_G4d_G80d = x[24]
        cdef double DG3 = x[25]
        cdef double DG3_G4d = x[26]
        cdef double DG3_G4d_G80d = x[27]
        cdef double DGrep = x[28]
        cdef double DGrep_G4d = x[29]
        cdef double DGrep_G4d_G80d = x[30]
        cdef double DG80 = x[31]
        cdef double DG80_G4d = x[32]
        cdef double DG80_G4d_G80d = x[33]
        cdef double G2GAE = x[34]
        cdef double G2GAI = x[35]
        cdef double G1GAI = x[36]


        # Now let's try everything in molecs

        #cdef double GAE = 16.6
        #cdef double GAE = 4.0
        # cdef double GAE_mM = 11.1 #mM # TME: moved to toplevel
        #cdef double GAE = 0.0

        cdef double molecTomM = 4.65e-8       # mM/molecules

        cdef double GAE = self.GAE_mM / molecTomM # GAE in molecs

        #cdef double GAI = GAI*molecTomM

        #cdef double G2GAE = G2GAE*molecTomM
        #cdef double G2GAI = G2GAI*molecTomM
        #cdef double G1GAI = G1GAI*molecTomM

        cdef double max_R1 = 33  # Iyer & Struhl

        cdef double max_R2 = 33
        cdef double max_R80 = 21 # total amount of gal3p is 5 times the total amount of gal80p (Verma et al.)
        cdef double max_R4 = 0.4 
        cdef double max_R3 = 28
        cdef double max_RRep = 33

        cdef double kdr_gal80 = log(2)/24 # young lab data says 16 minutes wang/liu/story say 32 minutes
        cdef double kdr_gal3 = log(2)/26
        cdef double kdr_gal2 = log(2)/9
        cdef double kdr_gal1 = log(2)/31
        # DB: Try change kdr_gal1
        #cdef double kdr_gal1 = log(2)/1000
        cdef double kdr_gal4 = log(2)/28 # Wang/Liu/Storey
        cdef double kdr_rep = log(2)/20
        # DB: Try change kdr_rep
        # No real effect
        #cdef double kdr_rep = log(2)/1000

        cdef double cellDoublingTime = 180 # cell doubling time, in minutes

        cdef double prot_to_mrna_gal3 = 4800 # Beyer
        cdef double prot_to_mrna_gal1 = 500 # "metabolism" genes=5000, but we have no spec. data for Gal1p
        cdef double prot_to_mrna_gal2 = 3500 # IEE paper
        cdef double prot_to_mrna_gal80 = 530 # Beyer
        cdef double prot_to_mrna_gal4 = 1545 # Beyer
        cdef double prot_to_mrna_rep = 500  # IEE paper

        cdef double max_G4d = max_R4*prot_to_mrna_gal4/2

        cdef double kir_gal80 = max_R80 * kdr_gal80
        cdef double kir_gal3 = max_R3 * kdr_gal3
        cdef double kir_gal2 = max_R2 * kdr_gal2
        cdef double kir_gal1 = max_R1 * kdr_gal1
        cdef double kir_gal4 = max_R4 * kdr_gal4
        cdef double kir_rep = max_RRep * kdr_rep

        cdef double kdp_gal80 = log(2)/100
        cdef double kdp_gal3 = log(2)/60
        cdef double kdp_gal2 = log(2)/cellDoublingTime
        cdef double kdp_gal1 = log(2)/cellDoublingTime
        cdef double kdp_gal4 = log(2)/100	 		# Salghetti et al., PNAS 97(7):3118-3123 (2000)
        cdef double kdp_rep = log(2)/60

        cdef double kip_gal80 = prot_to_mrna_gal80 * kdp_gal80
        cdef double kip_gal3 = prot_to_mrna_gal3 * kdp_gal3
        cdef double kip_gal2 = prot_to_mrna_gal2 * kdp_gal2
        cdef double kip_gal1 = prot_to_mrna_gal1 * kdp_gal1
        cdef double kip_gal4 = prot_to_mrna_gal4 * kdp_gal4
        cdef double kip_rep = prot_to_mrna_rep * kdp_rep

        # Kinetic Parameters:

        cdef double alpha_TR = 1                   # dimensionless
        #cdef double Km_TR = 1.0                      # millimolar
        cdef double Km_TR = 1.0 / molecTomM          # molecules
        #cdef double Km_TR = 21505000                  # molec 
        cdef double k_TR = 4350.0                    # min^-1

        cdef double kcat_GK = 3350.0                 # min^-1   
        #cdef double Km_GK = 0.6 / molecTomM          # molecules
        cdef double Km_GK = 12903000                  # molec

        #cdef double q = 30  # cooperativity for multiple repressor (G80d) binding

        # cdef double kfp = 6.5/max_G4d  # from Verma et al 2003 (frac sat in gal80 deletion mutant is 70% for one binding site)
        # cdef double kfr = 5*kfp 
        # cdef double krp = 1.0   
        # cdef double krr = 1.0 

        cdef double Kfi = 0.000000745  # molec^-1 min^-1


        cdef double Kri = 890
        cdef double Kfd3i80 = 0.025716 #0.01596 
        cdef double Kdr3i80 = 0.0159616 #0.02572 
        cdef double Kfd = 100 #0.001
        cdef double Krd = 0.001 #100
        cdef double Kf80 = 50
        cdef double Kr80 = 50 

        # Transcriptional Regulation 

        # DB: Change Kp and Kq to try to increase sensitivity
        # How much play room do we have




        # David CHANGE: Here is where I enter the new Kp, Kq values for different numbers of binding sites
        # These values were determined by an optimization using MATLAB fmincon
        # Protein counts are still undershot relative to original matlab model at 0.0 mM Gex
        # Good agreement at all other concentrations

        #cdef double Kp = 5.0
        #cdef double Kp = 0.02104

        #cdef double Kq = 0.1052
        # How do they arrive at Kq? Is this # open for negotiation.
        #cdef double Kq = 1.5


        # The Kp values, relating binding strength of the transcription factor

        # Kp for genes with single binding sites, proteins: G3, G80
        cdef double Kp = 0.0248

        # Kp for genes with 4 binding sites, proteins: G1, reporter
        cdef double Kp4 = 0.2600

        # Kp for genes with 5 binding sites, proteins: G2
        cdef double Kp5 = 0.0099


        # The Kq values, relating to binding strength of the repressor

        # Kq for genes with 1 binding site, proteins: G3, G80
        cdef double Kq = 0.1885

        # Kq for genes with 4 binding sites, proteins: G1, reporter
        cdef double Kq4 = 1.1721

        # Kq for genes with 5 binding sites, proteins: G2
        cdef double Kq5 = 0.7408



        # Values for single binding site genes, proteins: G3, G80
        cdef double kf1 = 0.1
        cdef double kf2 = 0.1
        cdef double kr1 = kf1/Kp   # kf1 and kf2 values from Atauri et al Syst. Biol. 2004
        cdef double kr2 = kf2/Kq

        # 4 binding site genes, proteins: G1, reporter
        cdef double kf1_4 = 0.1
        cdef double kf2_4 = 0.1
        cdef double kr1_4 = kf1_4/Kp4
        cdef double kr2_4 = kf2_4/Kq4

        # 5 binding site genes, proteins: G2
        cdef double kf1_5 = 0.1
        cdef double kf2_5 = 0.1
        cdef double kr1_5 = kf1_5/Kp5
        cdef double kr2_5 = kf2_5/Kq5



        cdef double Co = 1  # units in molec

        # Steady state: kalpha/[DGal4p]=kdr/[R1] (Rate of production = Rate of
        # degradation) do this for all cases and while doing this change kalpha

        # Derived value for DGal4p from the fracSat fn. 
        #cdef double kalpha1 = (kdr_gal1*R1)/(DGal4p*Co)
        #cdef double kalpha1 = (kdr_gal1*0.2646)/(0.0565*Co)
        cdef double kalpha1 = kir_gal1/Co
        #cdef double kalpha1 = (kdr_gal1*0.1646)/(0.0565*Co)

        # cdef double kalpha2=(kdr_gal2*R2)/(DGal4p*Co)
        cdef double kalpha2 = kir_gal2/Co
        #cdef double kalpha2 = (kdr_gal2*0.3305)/(0.0565*Co)

        # cdef double kalpha3=(kdr_gal3*R3)/(DGal4p*Co*q(wildtype))
        # See GalNoiseSuppNote Aitchison et. al.
        # for details about qwildtype = 0.571
        cdef double kalpha3 = kir_gal3/Co
        #cdef double kalpha3 = (kdr_gal3*0.9044)/(0.0565*Co*0.571429)

        cdef double kalpha_rep = kir_rep/Co
        # cdef double kalpharep=(kdr_rep*Rrep)/(DGal4p*Co)
        #cdef double kalpha_rep = (kdr_rep*0.2646)/(0.0565*Co)

        # Steady state: kalpha80=(kdr_gal80*R80)/(DGal4p*Co)
        cdef double kalpha80 = kir_gal80/Co
        #cdef double kalpha80 = (kdr_gal80*1.1870)/(0.0565*Co)


        # Reactions:

        # vTR = k_TR * G2 * (GAE - GAI) / (Km_TR + GAE + GAI + (alpha_TR*GAE*GAI/Km_TR))

        # Updated Reactions for transport
        # Should be removing 3 and adding 9 reactions (total = +6)

        # Now in reversible Michaelis Menten form

        # G2 + GAE--> G2GAE   kf_TR
        # G2GAE --> GAE + G2  kr_TR
        # G2GAE --> G2GAI  k_TR
        # Km_TR = (kr_TR + k_TR)/kf_TR


        #cdef double kr_TR = 0.01 * k_TR # reverse rate of Gal2 transporter binding to galactose 1/min
        cdef double kr_TR = 0.55 * k_TR

        cdef double kf_TR = (kr_TR + k_TR)/Km_TR # forward rate of Gal2 transporter binding to galactose 1/(min.molec)

        #kf_TR * GAE * G2 - kr_TR*GalG2 - k_TR*GalG2 - kf_TR * GalG2 + kr_TR * GAE*G2 + k_TR*GAI*G2 # 35 GalG2 and bckwd??
        #kf_TR*GAE*G2 - kr_TR*G2GAE - k_TR*G2GAE - kf_TR*GAE + kr_TR*G2*GAE #35 G2GAE

        #kf_TR*G2*GAI - k_TR*G2GAI - kr_TR*G2GAI + kr_GK*G1GAI - kf_GK*G1*GAI #36 G2GAI

        # And add the reverse of these 3 reactions
        # G2 + GAI  --> G2GAI kf_TR
        # G2GAI --> G2 + GAI kr_TR
        # G2GAI --> G2GAE k_TR
        # Km_TR = (kr_TR + k_TR)/kf_TR

        # New expression for vTR
        # cdef double vTR = k_TR * (GalG2 - (GAI + G2)) - kr_TR * (GalG2-(GAE + G2))
        # k_TR*G2GAE + kr_TR*G2GAI - kf_TR*GAI*G2 #19 GAI

        #cdef double vGK = kcat_GK * G1 * GAI / (Km_GK + GAI)

        # Updated Reactions for Galactokinase
        # G1 + GAI --> G1GAI kf_GK : Association to form ES complex
        # G1GAI --> G1 + GAI  kr_GK : The dissociation of the ES complex
        # G1GAI --> G1   kcat_GK : The enzymatic Reaction (Not keeping track of product)
        # cdef double Km = (kr + kcat_GK)/kf

        cdef double kr_GK = 0.55 * kcat_GK # reverse rate of G1 binding to galactose (set as 1% fwd rate) 1/min


        cdef double kf_GK = (kr_GK + kcat_GK)/Km_GK # forward rate of G1 binding to galactose 1/(min.molec)

        # kf_GK*G1*GAI - kr_GK*G1GAI - kcat_GK*G1GAI # 37 G1GAI 

        # kr_GK*G1GAI - kf_GK*G1*GAI # add to GAI



        # cdef double Kirgal1 = par(1)
        # cdef double Kirgal2 = par(2)

        # cdef double FoneSite = fracSatThreeStatesOneSite(kfp, krp, kfr, krr, G4d, G80d)
        # cdef double FtwoSites = fracSatThreeStatesTwoSites(kfp, krp, kfr, krr, q, G4d, G80d)
        # cdef double FfourSites = fracSatThreeStatesFourSites(kfp, krp, kfr, krr, q, G4d, G80d)
        # cdef double FfiveSites = fracSatThreeStatesFiveSites(kfp, krp, kfr, krr, q, G4d, G80d)

        self.dxdt[0] =   kalpha1*DG1_G4d-kdr_gal1*R1 #1 R1
        self.dxdt[1] =   kalpha2*DG2_G4d-kdr_gal2*R2 #2 R2
        self.dxdt[2] =   0.571429*kalpha3*DG3_G4d-kdr_gal3*R3 #3 R3
        self.dxdt[3] =   kir_gal4-kdr_gal4*R4 #4 R4
        self.dxdt[4] =   kalpha_rep*DGrep_G4d-kdr_rep*reporter_rna #5 reporter_rna
        self.dxdt[5] =   kalpha80*DG80_G4d-kdr_gal80*R80 #6 R80
        self.dxdt[6] =   kip_gal1*R1 - kdp_gal1*G1 - kf_GK*G1*GAI + kr_GK*G1GAI + kcat_GK*G1GAI #7 G1
                    #kip_gal2*R2-kdp_gal2*G2 #8 G2
        self.dxdt[7] =   -kf_TR*G2*GAE + kr_TR*G2GAE - kf_TR*G2*GAI + kr_TR*G2GAI+kip_gal2*R2-kdp_gal2*G2 # 8 G2
        self.dxdt[8] =   kip_gal3*R3-kdp_gal3*G3-Kfi*G3*GAI+Kri*G3i                       #9 G3
        self.dxdt[9] =   Kfi*G3*GAI-Kri*G3i-kdp_gal3*G3i-Kfd3i80*G80Cd*G3i+Kdr3i80*G80G3i #10 G3i
        self.dxdt[10] =   kip_gal4*R4-kdp_gal4*G4-2*Kfd*G4**2+2*Krd*G4d #11 G4
        self.dxdt[11] =   Kfd*G4**2-Krd*G4d-kdp_gal4*G4d+kr1_4*DG1_G4d+kr1_5*DG2_G4d+kr1*DG3_G4d+kr1_4*DGrep_G4d+kr1*DG80_G4d-G4d*(kf1_4*DG1+kf1_5*DG2+kf1*DG3+kf1_4*DGrep+kf1*DG80) #12 G4d
        self.dxdt[12] =   kip_rep*reporter_rna-kdp_rep*reporter #13 reporter
        # DB: CHANGE
        #self.dxdt[13] =   kip_gal80*R80-kdp_gal80*G80-Kf80*G80+Kr80*G80C-2*Kfd*G80**2+2*Krd*G80d #14 G80
        self.dxdt[13] =   kip_gal80*R80-kdp_gal80*G80-Kf80*G80+Kr80*G80C-2*Kfd*G80**2+2*Krd*G80d #14 G80
        self.dxdt[14] =   Kf80*G80-Kr80*G80C-2*Kfd*G80C**2+2*Krd*G80Cd-kdp_gal80*G80C # 15 G80C
        self.dxdt[15] =   Kfd*G80**2-Krd*G80d-kdp_gal80*G80d+Kf80*G80Cd-Kr80*G80d+kr2_4*DG1_G4d_G80d+kr2_5*DG2_G4d_G80d+kr2*DG3_G4d_G80d+kr2_4*DGrep_G4d_G80d+kr2*DG80_G4d_G80d-G80d*(kf2_4*DG1_G4d+kf2_5*DG2_G4d+kf2*DG3_G4d+kf2_4*DGrep_G4d+kf2*DG80_G4d) #16 G80d
        self.dxdt[16] =   Kfd*G80C**2-Krd*G80Cd-kdp_gal80*G80Cd+Kf80*G80d-Kr80*G80Cd-Kfd3i80*G80Cd*G3i+Kdr3i80*G80G3i #17 G80Cd
        self.dxdt[17] =   Kfd3i80*G80Cd*G3i-Kdr3i80*G80G3i-0.5*kdp_gal3*G80G3i #18 G80G3i
                     #vTR-vGK #19 GAI
                    
                    # vTR = k_TR * G2 * (GAE - GAI) / (Km_TR + GAE + GAI + (alpha_TR*GAE*GAI/Km_TR))
                    
                    # Updated dx/dt GAI
                    #(k_TR*G2GAE + kr_TR*G2GAI - kf_TR*GAI*G2 + kr_GK*G1GAI - kf_GK*G1*GAI)/molecTomM #19 GAI
                    #(k_TR*G2GAE + kr_TR*G2GAI  + kr_GK*G1GAI - kf_GK*G1*GAI - kf_TR*GAI*G2) - Kfi*G3*GAI + Kri*G3i #19 GAI
        self.dxdt[18] =   kr_TR*G2GAI - kf_TR*GAI*G2 + kr_GK*G1GAI - kf_GK*G1*GAI - Kfi*G3*GAI + Kri*G3i + kdp_gal1*G1GAI + kdp_gal2*G2GAI + kdp_gal3*G3i #19 GAI
                    
                    # In above why kf_TR*GAI*G2 twice?
                    #kr_TR*G2GAI - kf_TR*GAI*G2 + kr_GK*G1GAI - kf_GK*G1*GAI - Kfi*G3*GAI + Kri*G3i #19 GAI
                    
                    # G3 + GAI --> G3i Kfi
                    # G3i --> G3 + GAI Kri
                    # G3i + G80 --> G80G3i kcat = Kfd3i80
                    
                    #########DG1#################
        self.dxdt[19] =   kr1_4*DG1_G4d-kf1_4*DG1*G4d #20 
        self.dxdt[20] =   kr2_4*DG1_G4d_G80d-kf2_4*DG1_G4d*G80d+kf1_4*DG1*G4d-kr1_4*DG1_G4d #21 
        self.dxdt[21] =   kf2_4*DG1_G4d*G80d-kr2_4*DG1_G4d_G80d #22 
                     #########DG2#################
        self.dxdt[22] =   kr1_5*DG2_G4d-kf1_5*DG2*G4d #23 
        self.dxdt[23] =   kr2_5*DG2_G4d_G80d-kf2_5*DG2_G4d*G80d+kf1_5*DG2*G4d-kr1_5*DG2_G4d #24 
        self.dxdt[24] =   kf2_5*DG2_G4d*G80d-kr2_5*DG2_G4d_G80d #25 
                     #########DG3#################
        self.dxdt[25] =   kr1*DG3_G4d-kf1*DG3*G4d #26 
        self.dxdt[26] =   kr2*DG3_G4d_G80d-kf2*DG3_G4d*G80d+kf1*DG3*G4d-kr1*DG3_G4d #27 
        self.dxdt[27] =   kf2*DG3_G4d*G80d-kr2*DG3_G4d_G80d #28 
                     #########DGrep#################
        self.dxdt[28] =   kr1_4*DGrep_G4d-kf1_4*DGrep*G4d #29 
        self.dxdt[29] =   kr2_4*DGrep_G4d_G80d-kf2_4*DGrep_G4d*G80d+kf1_4*DGrep*G4d-kr1_4*DGrep_G4d #30 
        self.dxdt[30] =   kf2_4*DGrep_G4d*G80d-kr2_4*DGrep_G4d_G80d #31 
                     #########DG80#################
        self.dxdt[31] =   kr1*DG80_G4d-kf1*DG80*G4d #32 
        self.dxdt[32] =   kr2*DG80_G4d_G80d-kf2*DG80_G4d*G80d+kf1*DG80*G4d-kr1*DG80_G4d #33 
        self.dxdt[33] =   kf2*DG80_G4d*G80d-kr2*DG80_G4d_G80d #34 
                     #########Additional DB##########
                     #(kf_TR*GAE*G2 - kr_TR*G2GAE - k_TR*G2GAE - kf_TR*GAE + kr_TR*G2*GAE)/molecTomM #35 G2GAE
        self.dxdt[34] =   k_TR*G2GAI - k_TR*G2GAE - kr_TR*G2GAE + kf_TR*GAE*G2 - kdp_gal2*G2GAE  #35 G2GAE
                    # kf*G2*Gex - kr*G2Gex - kcat*G2Gex + kcat*G2Gic # G2Gex
                    
                    #(kf_TR*G2*GAI - k_TR*G2GAI - kr_TR*G2GAI + kr_GK*G1GAI - kf_GK*G1*GAI)/molecTomM #36 G2GAI
        self.dxdt[35] =   -k_TR*G2GAI + kf_TR*GAI*G2 - kr_TR*G2GAI + k_TR*G2GAE - kdp_gal2*G2GAI #36 G2GAI
                    #kcat*G2Gex - kcat*G2Gic - kr*G2Gic + kf*G2*Gic # G2Gic

        self.dxdt[36] =   kf_GK*G1*GAI - kr_GK*G1GAI - kcat_GK*G1GAI-kdp_gal1*G1GAI # 37 G1GAI

