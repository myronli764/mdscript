import numpy as np
import numba as nb
from scipy.stats import circmean



# x: ndarray[frames,n_particles,3]
# box: ndarray[frames,3]
@nb.jit(nopython=True,nogil=True)
def pbc(x,l):
    return x - l*np.rint(x/l)

def Rg_Ree(x,box,cl):
    frames, n, _ = x.shape
    pos = x.reshape(frames,-1,cl,_)
    bl = box.mean()
    chainCM = pos.mean(axis=-2)
    _,nmols,__ = chainCM.shape
    rg2 = []
    ree2 = []
    for f in range(frames):
        for i in range(nmols):
            rg2.append(np.mean(((pos[f][i] - chainCM[f][i])**2).sum(axis=-1)))
            ree2.append(((pos[f][i][0]-pos[f][i][-1])**2).sum(axis=-1))
    return np.mean(np.array(rg2)),np.mean(np.array(ree2))

