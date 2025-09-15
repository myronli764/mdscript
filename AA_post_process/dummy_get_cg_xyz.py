import MDAnalysis as mda
import numpy as np
import tqdm

from misc import TCMol, Atom, Bond
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from read_tpr_connection import GetConnectedMolFromTpr
import networkx as nx

class xml:
    def __init__(self,p:np.ndarray ,v: np.ndarray ,atomtype:list, bond: dict ,angle : dict,dihedral: dict,box: np.ndarray):
        r'''

        :param p: position of all particles
        :param v: velocity of all particles
        :param atomtype : cg type list , eg. ['A','B','A', ... ]
        :param bond: cg bond list , eg. [((i,j),(bondtype,r0,k0)), ... ]
        :param angle: cg angle list , eg. [((i,j,k),(angletype,th0,k0)), ... ]
        :param dihedral: cg dihedral list , eg. [((i,j,k,l),(dihedraltype,di_parameter))]
        :param box: [lx,ly,lz]
        '''
        self.p = p
        self.v = v
        self.bond = bond
        self.box = box
        self.angle = angle
        self.dihedral = dihedral
        self.type = atomtype

    def writer(self,file:str,need=None,program='hoomd'):
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
                string += f'{p_[0]:8.3f}  {p_[1]:8.3f}  {p_[2]:8.3f}\n'
            string += '</position>\n'
            string += f'<velocity num="{natoms}">\n'
            for v_ in self.v:
                string += f'{v_[0]:8.3f}  {v_[1]:8.3f}  {v_[2]:8.3f}\n'
            string += '</velocity>\n'
            string += f'<type num="{natoms}">\n'
            for t_ in self.type:
                string += f'{t_}\n'
            string += '</type>\n'
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

aa_mols, mol_graphs, (global_local_idx_map,local_global_idx_maps) = GetConnectedMolFromTpr('eq10.tpr','eq10.xtc')
u = mda.Universe('eq10.tpr','eq10.xtc')
atoms   = u.atoms
positions = atoms.positions
bonds = u.bonds
imps = u.impropers

rings = []
for g in mol_graphs:
    cb = nx.cycle_basis(g)
    for cycle in cb:
        rings.append(cycle)
rings_6 = [r for r in rings if len(r)==6]
rings_6 = np.array(rings_6)
atom_to_ringidx = {}
for i, r in enumerate(rings_6):
    for aidx in r:
        if aidx not in atom_to_ringidx:
            atom_to_ringidx[aidx] = []
        atom_to_ringidx[aidx].append(i)
aromatic_atoms = []
for imp in imps:
    #print(imp)
    bonded_atoms = imp.atoms
    num_in_ring_6 = 0
    a1 = bonded_atoms[0]
    a2 = bonded_atoms[1]
    a3 = bonded_atoms[2]
    a4 = bonded_atoms[3]
    for a in [a1, a2, a3, a4]:
        if a.index in atom_to_ringidx:
            num_in_ring_6 += 1
    if num_in_ring_6 >=3 :
        for a in [a1, a2, a3, a4]:
            if a.mass > 2 and a.index in atom_to_ringidx:
                aromatic_atoms.append(a.index)
aromatic_atoms = set(aromatic_atoms)
for g in mol_graphs:
    #print(g.nodes)
    for n in g.nodes:
        if n in aromatic_atoms:
            #print(n, g.nodes[n], atom_to_ringidx[n])
            g.nodes[n]['is_aromatic'] = True

aa_mols = []
for i in tqdm.tqdm(range(len(mol_graphs))):
    tCMol = TCMol(mol_graphs[i])
    new_mol = tCMol.TCMolToMol()
    pattern = '[#6][#8]'
    smarts = Chem.MolFromSmarts(pattern)
    matches = new_mol.GetSubstructMatches(smarts)
    #print(len(matches))
    for match in matches:
        #print(match)
        #for i in match:
        #    atom = new_mol.GetAtomWithIdx(i)
        #    if
        i, j = match
        bond = new_mol.GetBondBetweenAtoms(i, j)
        for idx in match:
            atom = new_mol.GetAtomWithIdx(idx)
            if atom.GetAtomicNum() == 8:
                if atom.GetDegree() == 1:
                    bond.SetBondType(Chem.BondType.DOUBLE)
    Chem.SanitizeMol(new_mol)
    aa_mols.append(new_mol)
#print(len(aa_mols))
## get CG bead positions and type
def get_backbone_A_connected_atoms(molecules, graphs):
    # backbond A is the dianhydride in polyimide monomer, O1C(=O)CCC1(=O)-ph-C2C(=O)OC(=O)C2
    pattern = Chem.MolFromSmarts('C(=O)NC(=O)')
    backbone_Ac_atoms = []
    for (mol, aid_idx_map), g in zip(molecules,graphs):
        matches = mol.GetSubstructMatches(pattern)
        #print("Number of matches found:", len(matches))
        if matches:
            for match in matches:
                #local_idx = set([idx for idx in match if mol.GetAtomWithIdx(idx).GetIsAromatic()])
                #global_idxs = []
                for idx in match:
                    if mol.GetAtomWithIdx(idx).GetSymbol() == 'N':
                        atom = mol.GetAtomWithIdx(idx)
                        nbrs = [_.GetIdx() for _ in atom.GetNeighbors()]
                        s = 0
                        for nbr in nbrs:
                            if nbr in match:
                                s += 1
                        if s != 2:
                            print(match, nbrs)

                global_idx = set([aid_idx_map[idx] for idx in match if not mol.GetAtomWithIdx(idx).GetIsAromatic()])
                #if 661 in global_idx:
                #    nbrs = [_.GetIdx() for _ in mol.GetAtomWithIdx(661).GetNeighbors()]
                #    print(match, [aid_idx_map[idx] for idx in match], "00000000000000000000000000000", nbrs)
                #print("Match found at global indices:", global_idx)
                backbone_Ac_atoms.append(global_idx)
                #print("Simple cycles in subgraph:", global_cycles)
    return backbone_Ac_atoms

def get_backbone_A_atoms(molecules,graphs):
    # backbond A is the dianhydride in polyimide monomer, O1C(=O)CCC1(=O)-ph-C2C(=O)OC(=O)C2
    pattern = Chem.MolFromSmarts('Cc1ccc(c2cc(C)c(C)cc2)cc1C')
    backbone_A_atoms = []
    for (mol, aid_idx_map), g in zip(molecules,graphs):
        matches = mol.GetSubstructMatches(pattern)
        #print("Number of matches found:", len(matches))
        if matches:
            for match in matches:
                local_idx = set([idx for idx in match if mol.GetAtomWithIdx(idx).GetIsAromatic()])
                global_idx = set([aid_idx_map[idx] for idx in match if mol.GetAtomWithIdx(idx).GetIsAromatic()])
                #print("Match found at global indices:", global_idx)
                subg = g.subgraph(global_idx)
                simple_cycles = list(nx.simple_cycles(subg))
                global_cycles = []
                for cycle in simple_cycles:
                    global_cycle = set([idx for idx in cycle])
                    global_cycles.append(global_cycle)
                backbone_A_atoms.append(global_cycles)
                #print("Simple cycles in subgraph:", global_cycles)
    return backbone_A_atoms

def get_backbone_B_atoms(molecules, graphs):
    # backbond B is the dianhydride in polyimide monomer, Nc1cc(N)ccc1
    pattern = Chem.MolFromSmarts('Nc1cc(N)ccc1')
    backbone_B_atoms = []
    for (mol, aid_idx_map), g in zip(molecules,graphs):
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            for match in matches:
                local_idx = set([idx for idx in match if mol.GetAtomWithIdx(idx).GetIsAromatic()])
                global_idx = set([aid_idx_map[idx] for idx in match if mol.GetAtomWithIdx(idx).GetIsAromatic()])
                #print("Match found at global indices:", global_idx)
                backbone_B_atoms.append(global_idx)
                #print("Simple cycles in subgraph:", global_cycles)
    return backbone_B_atoms

def get_sideLC_atoms(molecules, graphs):
    # sideLC is the liquid crystal group in the side chain of monomer, Oc1ccc(c2ccccc2)cc1
    pattern1 = Chem.MolFromSmarts('Oc1ccc(c2ccccc2)cc1')
    pattern2 = Chem.MolFromSmarts('Oc1ccccc1')
    sideLC_atoms = []
    for (mol, aid_idx_map), g in zip(molecules,graphs):
        matches = mol.GetSubstructMatches(pattern1)
        matches2 = mol.GetSubstructMatches(pattern2)
        if matches:
            for match, match2 in zip(matches, matches2):
                global_idx = set([aid_idx_map[idx] for idx in match if mol.GetAtomWithIdx(idx).GetIsAromatic()])
                global_idx1 = set([aid_idx_map[idx] for idx in match2 if mol.GetAtomWithIdx(idx).GetIsAromatic()])
                global_idx2 = global_idx - global_idx1
                sideLC_atoms.append([global_idx1, global_idx2])
                #sideLC_atoms.append(global_idx2)
    return sideLC_atoms

def get_sidechainC_atoms(molecules, graphs):
    # sidechain C is the cyano group in the side chain of monomer, C#N
    pattern = Chem.MolFromSmarts('OCCCCCCO')
    sidechainC_atoms = []
    for (mol, aid_idx_map), g in zip(molecules,graphs):
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            for match in matches:
                local_idx = set([idx for idx in match])
                global_idx_all = set([aid_idx_map[idx] for idx in local_idx])
                subg = g.subgraph(global_idx_all)
                path = []
                for i in subg.nodes:
                    if subg.degree[i] == 1:
                        path.append(i)
                        #print("Node with degree 1 found:", i)
                path = nx.shortest_path(subg, source=path[0], target=path[-1])
                global_idx = [idx for idx in path]
                #atoms_label = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in path]
                global_idx1 = global_idx[:4]
                global_idx2 = global_idx[4:]
                #global_idx3 = global_idx[7:]
                #print("Match found at global indices:", global_idx, "with atoms:", atoms_label)
                sidechainC_atoms.append([global_idx1, global_idx2])
                #sidechainC_atoms.append(global_idx2)
                #sidechainC_atoms.append(global_idx3)
    return sidechainC_atoms

def get_sidechainC_connected_atoms(molecules, graphs):
    # sidechain C is the cyano group in the side chain of monomer, C=O
    pattern = Chem.MolFromSmarts('C(=O)OCC')
    sidechainCc_atoms = []
    for (mol, aid_idx_map), g in zip(molecules,graphs):
        matches = mol.GetSubstructMatches(pattern)
        #print("Number of matches found:", len(matches))
        if matches:
            for match in matches:
                #local_idx = set([idx for idx in match])
                local_idx = []
                for idx in match:
                    if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6:
                        neis = mol.GetAtomWithIdx(idx).GetNeighbors()
                        neis_an = [nei.GetAtomicNum() for nei in neis if not nei.GetIsAromatic()]
                        if 6 not in neis_an:
                            local_idx.append(idx)
                #print(local_idx)
                for idx in local_idx:
                    neis = mol.GetAtomWithIdx(idx).GetNeighbors()
                    for nei in neis:
                        if nei.GetAtomicNum() == 8 and nei.GetDegree() == 1:
                            local_idx.append(nei.GetIdx())
                global_idx = set([aid_idx_map[idx] for idx in local_idx])
                #print("Match found at global indices:", global_idx)
                sidechainCc_atoms.append(global_idx)
    return sidechainCc_atoms

meta = []
for aa_mol, local_global_idx_map in zip(aa_mols, local_global_idx_maps):
    #for a in aa_mol.GetAtoms():
    #    print(a.GetIdx(), a.GetSymbol(), a.GetIsAromatic())
    #print(local_global_idx_map.keys())
    meta.append((aa_mol, local_global_idx_map))
#raise
import random
# O(=C)NC(=O)
backbone_Ac_atoms = get_backbone_A_connected_atoms(meta, mol_graphs)
# [O(=C)NC(=O)]-c1ccccc1-c2ccccc2-[O(=C)NC(=O)]
backbone_A_atoms = get_backbone_A_atoms(meta, mol_graphs)
# Nc1cc(N)ccc1
backbone_B_atoms = get_backbone_B_atoms(meta, mol_graphs)
# [CCCO]-c1ccccc1-c2ccccc2-[C#N]
sideLC_atoms = get_sideLC_atoms(meta, mol_graphs)
# OCCC-CCC-CCCO
sidechainC_atoms = get_sidechainC_atoms(meta, mol_graphs)
# [c1ccccc1]-C(=O)-[OCC]
sidechainCc_atoms = get_sidechainC_connected_atoms(meta, mol_graphs)


#n_cg_nodes = len(backbone_A_atoms) + len(backbone_Ac_atoms) + len(backbone_B_atoms) + len(sideLC_atoms) + len(sidechainC_atoms) + len(sidechain_Cc_atoms)
#all_cg_atoms = (backbone_A_atoms + backbone_Ac_atoms + backbone_B_atoms + sideLC_atoms + sidechainC_atoms + sidechain_Cc_atoms)
#print("Number of coarse-grained nodes:", n_cg_nodes)
#print("Number of backbone_A atoms:", len(backbone_A_atoms))
#print("Number of backbone_Ac atoms:", len(backbone_Ac_atoms))
#print("Number of backbone_B atoms:", len(backbone_B_atoms))
#print("Number of sideLC atoms:", len(sideLC_atoms))
#print("Number of sidechainC atoms:", len(sidechainC_atoms))
#print("Number of sidechainCc atoms:", len(sidechainCc_atoms))

cg_types = (['A'] * len(backbone_A_atoms) * 2 + ['Ac'] * len(backbone_Ac_atoms) +
            ['B'] * len(backbone_B_atoms) + ['LC'] * len(sideLC_atoms) * 2 +
            ['C'] * len(sidechainC_atoms) * len(sidechainC_atoms[0]) + ['Cc'] * len(sidechainCc_atoms))
n_cg_nodes = len(cg_types)
print("Total number of coarse-grained nodes:", n_cg_nodes)
num_mol = len(aa_mols)
num_monomer = 20
bead_per_monomer = 11
cg_graphs = nx.Graph()
offset = 0
for i,_ in enumerate(backbone_A_atoms):
    cg_graphs.add_node(offset, type='A', aa_idx=_[0])
    offset += 1
    cg_graphs.add_node(offset, type='A', aa_idx=_[1])
    offset += 1
for _ in backbone_Ac_atoms:
    cg_graphs.add_node(offset, type='Ac', aa_idx=list(_))
    offset += 1
for _ in backbone_B_atoms:
    cg_graphs.add_node(offset, type='B', aa_idx=list(_))
    offset += 1
for _ in sideLC_atoms:
    cg_graphs.add_node(offset, type='LC', aa_idx=list(_[0]))
    offset += 1
    cg_graphs.add_node(offset, type='LC', aa_idx=list(_[1]))
    offset += 1
for _ in sidechainC_atoms:
    #s = 0
    for __ in _:
        cg_graphs.add_node(offset, type='C', aa_idx=list(__))
        offset += 1
for _ in sidechainCc_atoms:
    cg_graphs.add_node(offset, type='Cc', aa_idx=list(_))
    offset += 1
#for imol in range(num_mol):
#    for imono in range(num_monomer):
#        offset = imol * num_monomer * bead_per_monomer + imono * bead_per_monomer
#        #for itype, cg_atom in zip(cg_types, (backbone_A_atoms + backbone_Ac_atoms + backbone_B_atoms + sideLC_atoms + sidechainC_atoms + sidechainCc_atoms)):
#        #    s = ''
#        #    for i in cg_atom:
#        #        s += f'index == {i + offset} or '
#        #    print(f'({s[:-4]})  # {itype}, mol {imol}, mono {imono}')
#        cg_graphs.add_node(offset + 0, type='A', aa_idx=backbone_A_atoms[imol*num_monomer + imono][0])
#        cg_graphs.add_node(offset + 1, type='A', aa_idx=backbone_A_atoms[imol*num_monomer + imono][1])
#        cg_graphs.add_node(offset + 2, type='Ac', aa_idx=list(backbone_Ac_atoms[imol*num_monomer + imono][0]))
#        cg_graphs.add_node(offset + 3, type='Ac', aa_idx=list(backbone_Ac_atoms[imol*num_monomer + imono][1]))
#        cg_graphs.add_node(offset + 4, type='B', aa_idx=list(backbone_B_atoms[imol*num_monomer + imono]))
#        cg_graphs.add_node(offset + 5, type='LC', aa_idx=list(sideLC_atoms[imol*num_monomer + imono][0]))
#        cg_graphs.add_node(offset + 6, type='LC', aa_idx=list(sideLC_atoms[imol*num_monomer + imono][1]))
#        cg_graphs.add_node(offset + 7, type='C', aa_idx=list(sidechainC_atoms[imol*num_monomer + imono][0]))
#        cg_graphs.add_node(offset + 8, type='C', aa_idx=list(sidechainC_atoms[imol*num_monomer + imono][1]))
#        cg_graphs.add_node(offset + 9, type='C', aa_idx=list(sidechainC_atoms[imol*num_monomer + imono][2]))
#        cg_graphs.add_node(offset +10, type='Cc', aa_idx=list(sidechainCc_atoms[imol*num_monomer + imono]))
print("Total number of CG nodes:", cg_graphs.number_of_nodes())
aid_to_cgid = {}
for n in cg_graphs.nodes:
    for aid in cg_graphs.nodes[n]['aa_idx']:
        aid_to_cgid[aid] = n
for b in bonds:
    bonded_atoms = b.atoms
    a1 = bonded_atoms[0]
    a2 = bonded_atoms[1]
    cg1 = aid_to_cgid.get(a1.index, -1)
    cg2 = aid_to_cgid.get(a2.index, -1)
    if cg1 != -1 and cg2 != -1 and cg1 != cg2:
        cg_graphs.add_edge(cg1, cg2, type=f'{cg_graphs.nodes[cg1]["type"]}-{cg_graphs.nodes[cg2]["type"]}')
cg_mols = [cg_graphs.subgraph(c).copy() for c in nx.connected_components(cg_graphs)]

print("Number of CG molecules:", len(cg_mols))
trajectory = u.trajectory
#print(trajectory.n_frames)
c = 0
for t in tqdm.tqdm(trajectory,total=trajectory.n_frames,desc="Processing frames"):
    frame = trajectory.frame
    #if frame < 1000:
    #    continue
    #frame = c
    cg_positions = []
    for n in cg_graphs:#tqdm.tqdm(cg_graphs.nodes, desc="Calculating CG positions", total=n_cg_nodes):
        aa_idxs = cg_graphs.nodes[n]['aa_idx']
        #pos = np.array([atoms[aa_idx].center_of_mass(unwrap=True) for aa_idx in aa_idxs]) * 0.1
        pos = u.atoms[list(aa_idxs)].center_of_mass(unwrap=True) * 0.1
        cg_graphs.nodes[n]['x'] = pos
        cg_positions.append(pos)
    cg_positions = np.array(cg_positions)
    bond_dict = {(u, v): (cg_graphs.edges[u, v]['type'], 0.47, 1000) for u, v in cg_graphs.edges}
    ves = np.zeros_like(cg_positions)
    XML = xml(p=cg_positions, v=ves, atomtype=cg_types, bond=bond_dict, angle={}, dihedral={}, box=u.dimensions[0:3]*0.1)
    XML.writer(f'cg/dummy_cg_{frame:0>8d}.xml',program='galamost')
    c += 1