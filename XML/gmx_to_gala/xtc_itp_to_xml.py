import MDAnalysis as mda
import numpy as np
import networkx as nx
import pickle
from itp_parser import itp_parser
from XML import xml
import tqdm

def pbc(x,l):
    return x - l*np.rint(x/l)

fn = 't_6000.gro'
u = mda.Universe('topol.itp','eq_uw.xtc')
print(u.trajectory.n_frames)
#raise
for t in tqdm.tqdm(u.trajectory,total=u.trajectory.n_frames):
#for idx in np.arange(11)*600:
    #fn = f't_{idx:d}.gro'
    #top = itp_parser('topol.top')
    #u = mda.Universe('anneal.tpr', fn)
    atoms = u.atoms
    positions = u.atoms.positions 
    #print(positions[0])
    velocities = np.zeros_like(positions)
    mass = u.atoms.masses
    bonds_dict = {}
    for bond in u.bonds:
        i = bond.atoms[0].index
        j = bond.atoms[1].index
        si = atoms[i].element
        sj = atoms[j].element
        bonds_dict[(i,j)] = (f'{si}-{sj}',0,0)
    angles_dict = {}
    for angle in u.angles:
        i = angle.atoms[0].index
        j = angle.atoms[1].index
        k = angle.atoms[2].index
        si = atoms[i].type
        sj = atoms[j].type
        sk = atoms[k].type
        angles_dict[(i,j,k)] = (f'{si}-{sj}-{sk}',0,0)
    dihedrals_dict = {}
    for dihedral in u.dihedrals:
        i = dihedral.atoms[0].index
        j = dihedral.atoms[1].index
        k = dihedral.atoms[2].index
        l = dihedral.atoms[3].index
        si = atoms[i].type
        sj = atoms[j].type
        sk = atoms[k].type
        sl = atoms[l].type
        dihedrals_dict[(i,j,k,l)] = (f'{si}-{sj}-{sk}-{sl}',0,0,0)
    atomtypes = u.atoms.elements
    #image = np.rint((positions - u.dimensions[:3]*0.1*0.5)/u.dimensions[:3]*0.1)
    image = np.rint((t.positions - t.dimensions[:3]*0.5)/(t.dimensions[:3])).astype(int)
    XML = xml(p=pbc(t.positions-t.dimensions[:3]/2,t.dimensions[:3]), v=velocities, bond=bonds_dict, angle={}, dihedral={}, atomtype=atomtypes, box=t.dimensions[:3], mass=mass,image=image)
    XML.writer(f't_{t.frame}_m.xml', need=(1,1,1), program='galamost')
    #pickle.dump(params_hash, open('params_hash.pkl','wb'))
    #print(params_hash)
    # charge, sigma, epsilon
