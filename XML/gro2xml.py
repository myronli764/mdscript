import MDAnalysis as mda
import numpy as np
import networkx as nx
from sys import argv
import argparse

class xml:
    def __init__(self,p:np.ndarray ,v: np.ndarray ,atomtype:list, bond: dict ,angle : dict,dihedral: dict,box: np.ndarray):
        r'''

        :param p: position of all particles
        :param v: velocity of all particles
        :param atomtype : cg type list , eg. ['A','B','A', ... ]
        :param bond: cg bond dict , eg. {(i,j):(bondtype,r0,k0), ... }
        :param angle: cg angle dict , eg. {(i,j,k): (angletype,th0,k0), ... }
        :param dihedral: cg dihedral dict , eg. {(i,j,k,l):(dihedraltype,di_parameter),  ... }
        :param box: [lx,ly,lz]
        '''
        self.p = p
        self.v = v
        self.bond = bond
        self.box = box
        self.angle = angle
        self.dihedral = dihedral
        self.type = atomtype
    def writer(self,file:str,need=None,program='hoomd',gala_init=None):
        if need is not None:
            IsAngle = need[0]
            IsDihed = need[1]
            IsBond = need[2]
        else :
            IsAngle = 1
            IsDihed = 1
            IsBond = 1
        with open(file,'w') as f:
            natoms = len(self.p)
            nbonds = len(self.bond.keys())
            nangle = len(self.angle.keys())
            ndihed = len(self.dihedral.keys())
            string = f'<?xml version="1.0" encoding="UTF-8"?>\n<{program}_xml version="1.6">\n'
            string += f'<configuration time_step="0" dimensions="3" natoms="{natoms}" >\n'
            try:
                string += f'<box lx="{self.box[0]}" ly="{self.box[1]}" lz="{self.box[2]}" xy="{self.box[3]}" xz="{self.box[4]}" yz="{self.box[5]}"/>\n'
            except:
                string += f'<box lx="{self.box[0]}" ly="{self.box[1]}" lz="{self.box[2]}" xy="0" xz="0" yz="0"/>\n'
            string += f'<position num="{natoms}">\n'
            for p_ in self.p:
                string += f'{int(p_[0]*10**3)*10**(-3):8.3f}  {int(p_[1]*10**3)*10**(-3):8.3f}  {int(p_[2]*10**3)*10**(-3):8.3f}\n'
            string += '</position>\n'
            string += f'<velocity num="{natoms}">\n'
            for v_ in self.v:
                string += f'{v_[0]:8.3f}  {v_[1]:8.3f}  {v_[2]:8.3f}\n'
            string += '</velocity>\n'
            string += f'<type num="{natoms}">\n'
            for t_ in self.type:
                string += f'{t_}\n'
            string += '</type>\n'
            if program == 'galamost' and gala_init is not None:
                string += f'<h_init num="{natoms}">\n'
                for h in gala_init:
                    #if t_ in gala_init:
                    #    string += '1\n'
                    #else:
                    string += f'{h}\n'
                string += f'</h_init>\n'
                string += f'<h_cris num="{natoms}">\n'
                for t_ in self.type:
                    string += '0\n'
                string += f'</h_cris>\n'
            #string += f'<bond num="{nbonds}">\n'
            if nbonds != 0 and IsBond:
                string += f'<bond num="{nbonds}">\n'
                for b in self.bond:
                    string += f'{self.bond[b][0]} {b[0]} {b[1]}\n'
            else:
                string += f'<bond num="{0}">\n'
            string += '</bond>\n'
            #string += f'<angle num="{nangle}">\n'
            if nangle != 0 and IsAngle:
                string += f'<angle num="{nangle}">\n'
                for a in self.angle:
                    string += f'{self.angle[a][0]} {a[0]} {a[1]} {a[2]}\n'
            else:
                string += f'<angle num="{0}">\n'
            string += '</angle>\n'
            string += f'<dihedral num="{ndihed}">\n'
            if ndihed != 0 and IsDihed:
                for d in self.dihedral:
                    string += f'{self.dihedral[d][0]} {d[0]} {d[1]} {d[2]} {d[3]}\n'
            string += '</dihedral>\n'
            string += f'</configuration>\n</{program}_xml>'
            f.write(string)
            f.close()

parser = argparse.ArgumentParser(description='Turn gromacs file into ovito readable .xml file')
parser.add_argument('-gro',dest='gro',type=str,help='The .gro file')
parser.add_argument('-tpr',dest='tpr',type=str,help='The .tpr file')
parser.add_argument('-xml',dest='xml',type=str,help='Name of the output .xml file',default='out.xml')

args = parser.parse_args()
gro = args.gro
tpr = args.tpr
xml_name = args.xml

u = mda.Universe(tpr,gro)
atoms = u.atoms
pos = []
types = []
for a in atoms:
    types.append(a.element)
    pos.append(a.position * 0.1)
pos = np.array(pos)
vel = np.zeros_like(pos)
bonds = {}
for b in u.bonds:
    ai, aj = b.atoms
    bonds[(ai.id,aj.id)] = [f'{ai.element}-{aj.element}',0,0]
box = u.dimensions[:3]*0.1
xmlw = xml(pos,vel,types,bonds,{},{},box)
xmlw.writer(xml_name,program='galamost')

