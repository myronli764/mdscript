import numpy as np
import numba as nb
import networkx as nx

@nb.jit(nopython=True,nogil=True)
def pbc(x,l):
    return x - l * np.rint(x/l)

def bonded(x,types,box,cl):
    frames, nbeads, _ = x.shape
    nmol = int(nbeads/cl)
    bl = box.mean()
    top = nx.Graph()
    for ic in range(nmol):
        top.add_edges_from([(i + ic*cl,(i+1) + ic*cl) for i in range(cl-1)])
    for i,t in enumerate(types):
        top.nodes[i]['type'] = t
    pos = x.reshape(frames,nmol,cl,3)
    
    # bond
    bond_ = np.diff(pos,axis=-2)
    bond_ = pbc(bond_,bl)
    hist_b, b = np.histogram((bond_**2).sum(axis=-1)**0.5,bins=1000,range=(0,100),density=True)

    # angle
    angle_first = bond_[:,:-1,:]
    angle_second = bond_[:,1:,:]
    cos = (angle_first*angle_second).sum(axis=-1)/np.linalg.norm(angle_first,axis=-1)/np.linalg.norm(angle_second,axis=-1)
    angle = np.arccos(-cos)
    hist_a, a = np.histogram(angle,bins=1000,range=(0,np.pi),density=True)
    
    # dihedral
    cross = np.cross(angle_first,angle_second)
    dihedral_first = cross[:,:,:-1,:]
    dihedral_second = cross[:,:,1:,:]
    cos = (dihedral_first*dihedral_second).sum(axis=-1)/np.linalg.norm(dihedral_first,axis=-1)/np.linalg.norm(dihedral_second,axis=-1)
    di = np.arccos(-cos)
    hist_d, d = np.histogram(di,bins=1000,range=(0,np.pi),density=True)
    return np.vstack((b[:-1],hist_b)), np.vstack((a[:-1],hist_a)), np.vstack((d[:-1],hist_d))


