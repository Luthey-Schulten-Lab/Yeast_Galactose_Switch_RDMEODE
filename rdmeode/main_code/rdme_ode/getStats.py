import numpy as np
import time
import pickle
from jLM.RDME import File as RDMEFile

traj = RDMEFile("yeast-dummy-mrna.lm")

def latticeGen(r):
    alpha = 0.1
    def lattice(x):
        return traj.h5['Simulations/{:07d}/Lattice/{:010d}'.format(r,x)]
    pl = lattice(0)
    buf = np.zeros(pl.shape, dtype=pl.dtype)
    ns = traj.h5['Simulations/{:07d}/LatticeTimes'.format(r)].shape[0]
    avgTime = None
    elapsedSeconds=0
    for i in range(ns):
        tStart = time.time()
        lattice(i).read_direct(buf)
        yield buf
        dt = time.time() - tStart
        elapsedSeconds += dt
        if not avgTime:
            avgTime = dt
        else:
            avgTime = avgTime*(1-alpha) + alpha*dt
        projected = (ns-i)*avgTime/60
        elapsed = elapsedSeconds/60
        print("{:>6d}/{:<6d} {:.1f} min remain (elapsed: {:.1f} min, projected total: {:.1f} min)".format(i+1,ns, projected, elapsed, projected+elapsed))
    
pstats = {x.name:{y.name: dict(number=[], conc=[]) for y in traj.regionList} for x in traj.speciesList}

for pl in latticeGen(1):
    d = traj.particleStatistics(pl)
    for s in traj.speciesList:
        for r in traj.regionList:
            pstats[s.name][r.name]['number'].append(d['countBySpeciesRegion'][s][r])
            pstats[s.name][r.name]['conc'].append(d['concBySpeciesRegion'][s][r])

for s in traj.speciesList:
    for r in traj.regionList:
        for k in pstats[s.name][r.name]:
            pstats[s.name][r.name][k] = np.array(pstats[s.name][r.name][k])

pickle.dump(pstats, open("pstats-dummy-mrna.pkl", "wb"))

