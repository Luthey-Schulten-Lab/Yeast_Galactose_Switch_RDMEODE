import numpy as np
import pyximport
pyximport.install(setup_args={ "include_dirs":np.get_include()})
from Reversible_MM_GalODEModel import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt

y0 = np.array([0.264635551766987,0.330544405508794,0.904521515742709,0.400000000000000,
               0.264635551677896,1.18715948592467,132.318563460887,1156.91017704601,
               4341.70321120979,0,0.156531275667963,308.921734355756,132.317774287091,
               0.125764433780103,0.0955338910599898,157.246650776274,157.239961338382,
               0,0,0.00869385904938387,0.0565075912119200,0.934768219347246,
               0.00869385904938387,0.0565075912119200,0.934768219347246,
               0.00869385904938387,0.0565075912119200,0.934768219347246,
               0.00869385904938387,0.0565075912119200,0.934768219347246,
               0.00869385904938387,0.0565075912119200,0.934768219347246,0,0,0])

speciesNames = [ "R1", "R2", "R3", "R4", "reporter_rna", "R80", "G1", "G2", "G3",
                 "G3i", "G4", "G4d", "reporter", "G80", "G80C", "G80d", "G80Cd",
                 "G80G3i", "GAI", "DG1", "DG1_G4d", "DG1_G4d_G80d", "DG2",
                 "DG2_G4d", "DG2_G4d_G80d", "DG3", "DG3_G4d", "DG3_G4d_G80d",
                 "DGrep", "DGrep_G4d", "DGrep_G4d_G80d", "DG80", "DG80_G4d",
                 "DG80_G4d_G80d", "G2GAE", "G2GAI", "G1GAI"]

ts = np.logspace(0,4,300)
model = Reversible_MM_GalODEModel()
ys = odeint(model, y0, ts)

ncols=5
nrows=int(np.ceil(len(speciesNames)/ncols))
fig, axs = plt.subplots(nrows=nrows,ncols=ncols,figsize=(15,3*nrows))
ylim = (1e-3, 10**np.ceil(np.log10(ys.max())))
xlim = (ts[0],ts[-1])

for i,ax in enumerate(axs.ravel()):
    try:
        ax.loglog(ts, ys[:,i])
        ax.set(ylim=ylim, xlim=xlim, title=speciesNames[i])
    except:
        ax.remove()
fig.tight_layout()

fig.savefig("gal.pdf")
