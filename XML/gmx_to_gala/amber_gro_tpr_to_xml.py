import MDAnalysis as mda
import numpy as np
import networkx as nx
import pickle
from itp_parser import itp_parser
from XML import xml


fn = 't_6000.gro'
#top = itp_parser('topol.top')
u = mda.Universe('output.itp', 'output.gro')
atoms = u.atoms
positions = u.atoms.positions * 0.1
velocities = np.zeros_like(positions)
mass = u.atoms.masses
bonds_dict = {}
for bond in u.bonds:
    i = bond.atoms[0].index
    j = bond.atoms[1].index
    si = atoms[i].type
    sj = atoms[j].type
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
atomtypes = u.atoms.types

XML = xml(p=positions, v=velocities, bond=bonds_dict, angle=angles_dict, dihedral=dihedrals_dict, atomtype=atomtypes, box=u.dimensions[:3]*0.1, mass=mass)
XML.writer('output.xml', need=(1,1,1), program='galamost')
#pickle.dump(params_hash, open('params_hash.pkl','wb'))
#print(params_hash)
# charge, sigma, epsilon
