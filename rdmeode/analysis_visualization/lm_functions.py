import numpy as np
def deleteParticle(particles, x, y, z, pid):
    """
    Inputs:
    Returns:
    Called by:
    Description:
    """

    ps = [p for p_ in particles[:,z,y,x,:] for p in p_ if p != 0]
    if pid in ps:
        ps.remove(pid)
        ps = np.array(ps)
        pps = 16 # Particles Per Site
        ps.resize(pps)
        ps = ps.reshape((particles.shape[0], particles.shape[4]))
        particles[:,z,y,x,:] = ps
    
    return None


def checkParticle(particles, x, y, z, pid):
    """
    Inputs:
    Returns:
    Called by:
    Description:
    """

    #     print(x,y,z)
    ps = np.array([p for p_ in particles[:,z,y,x,:] for p in p_ if p == pid])
    if pid in ps:
    #         print(ps)
        return True
    else:
        return False
    
    
def getParticlesInSite(particles, x, y, z):
    """
    Inputs:
    Returns:
    Called by:
    Description:
    """
    
    ps = np.array([p for p_ in particles[:,z,y,x,:] for p in p_ if p != 0])
    
#     print(ps)
    
    return ps