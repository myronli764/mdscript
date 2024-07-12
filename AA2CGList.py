import networkx as nx
import numpy as np
import MDAnalysis as mda
from typing import List,Union
import re
from io import StringIO
from xml.etree import cElementTree

r'''
 This program is functionated for providing nonbond list with exclusion,
 bond list, angle list and dihedral list.
 IBI users are suffering from asignating the AA atoms with CG index for a
 long time. This program is used for reducing the human operation on complex
 network system as well as corse-grained the AA system.
 It needs .xml .gro .tpr for inputs.
 TODO
 ....
'''

def control_in(control_file):
    pass


class Box(object):
    def __init__(self):
        self.xy = 0
        self.xz = 0
        self.yz = 0
        return

    def update(self, dic):
        self.__dict__.update(dic)


class XmlParser(object):
    def __init__(self, filename, needed=None):
        tree = cElementTree.ElementTree(file=filename)
        root = tree.getroot()
        self.box = Box()
        self.data = {}
        needed = [] if needed is None else needed
        for key in root[0].attrib:
            self.__dict__[key] = int(root[0].attrib[key])
        for element in root[0]:
            if element.tag == 'box':
                self.box.update(element.attrib)
                continue
            if (len(needed) > 0) and (element.tag not in needed):
                continue
            if element.tag == 'reaction':
                self.data['reaction'] = []
                reaction_list = element.text.strip().split('\n')
                while '' in reaction_list:
                    reaction_list.remove('')
                for l in reaction_list:
                    r = re.split(r'\s+', l)
                    while '' in r:
                        r.remove('')
                    r[1:] = [int(_) for _ in r[1:]]
                    self.data['reaction'].append(r)
                continue
            if element.tag == 'template':
                self.data['template'] = eval('{%s}' % element.text)
                continue
            if len(element.text.strip()) > 0:
                self.data[element.tag] = np.genfromtxt(StringIO(element.text), dtype=None, encoding=None)


fgToaaidx_hash = {
    'Z':{
        #    [ 1  ,  2   ,  3   ,  4   ,  5   ,  6   ,  7   ,  8   ,  9   , 10   , 11   , 12   , 13   , 14   , 15   , 16   , 17   ,  18   , 19   , 20   , 21   , 22   , 23   , 24   , 25   , 26, 27   , 28  ]
        #    [True, False, False, False, False, False, False, False, False,False, False, True , True , True , True , False, False, False,False, False, False, False, False, False, False, False, False,False]
        'Z': np.array([1,12,13,14,15,26,27,28],dtype=int) - 1,
        'B1':np.array([21,22,23,24,25],dtype=int) - 1 ,
        'B2':np.array([16,17,18,19,20],dtype=int) - 1,
        'B3':np.array([7,8,9,10,11],dtype=int) - 1,
        'B4':np.array([2,3,4,5,6],dtype=int) - 1,
    },
    'A':{
        'A': np.array([1,2,3,4,5,6,7,8,9,10,11,12,13],dtype=int) - 1,
    },
    'X0':{
        'X1': np.array([1163,1164,1165,1166,1187,1188,1189],dtype=int) - 1163,
        'Y':  np.array([1167,1168,1169,1170,1171,1184,1185,1186], dtype=int) - 1163,
        'X2': np.array([1172,1173,1174,1175,1181,1182,1183], dtype=int) - 1163,
        'B':  np.array([1176,1177,1178,1179,1180], dtype=int) - 1163,
    },
    'X':{
        'X1': np.array([122,121,120,94,93,92,81],dtype=int) - 81,
        'Y':  np.array([95,96,97,98,99,117,118,119], dtype=int) - 81,
        'X2': np.array([81,92,93,94,120,121,122], dtype=int) - 81,
        'B1': np.array([109,110,111,112,113], dtype=int) - 81,
        'B2': np.array([104,105,106,107,108], dtype=int) - 81,
        'B3': np.array([87,88,89,90,91], dtype=int) - 81,
        'B4': np.array([82,83,84,85,86], dtype=int) - 81,
    },
    'Z0':{
        'Z0': np.array([167,168,169,170,171,177,178,179], dtype=int) - 167,
        'B': np.array([172,173,174,175,176], dtype=int) - 167,
    },
    'A0': {
        'A0': np.array([128,129,130,131,132,133,134,135,136,137,138,139,140], dtype=int) - 123,
        'B' : np.array([123,124,125,126,127], dtype=int) - 123,
    },
}

def cg_info(top: nx.Graph) -> Union[nx.Graph,list,list]: ## Union[bond:nx.Graph, angle: Dict, dihedral: Dict]
    bond = top
    angle = []
    for i in bond.nodes():
        nei = list(bond.neighbors(i))
        if len(nei) >= 2:
            for ci,_i in enumerate(nei):
                for _j in nei[ci+1:]:
                    angle.append((_i,i,_j))
    dihedral = []
    for edge in bond.edges():
        degree = dict(bond.degree(edge))
        keys = list(degree.keys())
        i = edge[0]
        j = edge[1]
        nei_i = list(bond.neighbors(edge[0]))
        nei_j = list(bond.neighbors(edge[1]))
        nei_i.remove(j)
        nei_j.remove(i)
        if bond.degree(keys[0]) >= 2 and bond.degree(keys[1]) >= 2 :
            for ni in nei_i:
                for nj in nei_j:
                    dihedral.append((ni,i,j,nj))
    dihedral_flag = 1
    if dihedral_flag == 0:
        dihedral = []
    return (bond,angle,dihedral)

def cgTofg1(G:nx.Graph):
    fg1 = nx.Graph()
    fgrid = 0
    cgid2fgid_hash = {}
    for n in G.nodes:
        cgrestype = G.nodes[n]['type']
        if cgrestype == 'Z' :
            r1,r2,r3,r4,r5 = (fgrid,fgrid+1,fgrid+2,fgrid+3,fgrid+4)
            cgid2fgid_hash[n] = [r1,r2,r3,r4,r5]
            fg1.add_node(r1, type=f'{cgrestype}.Z',cgid=n)
            fg1.add_node(r2, type=f'{cgrestype}.B1',cgid=n)
            fg1.add_node(r3, type=f'{cgrestype}.B2',cgid=n)
            fg1.add_node(r4, type=f'{cgrestype}.B3',cgid=n)
            fg1.add_node(r5, type=f'{cgrestype}.B4',cgid=n)
            fg1.add_edge(r1, r2, bt =f'{cgrestype[0]}-B')
            fg1.add_edge(r1, r3, bt =f'{cgrestype[0]}-B')
            fg1.add_edge(r1, r4, bt =f'{cgrestype[0]}-B')
            fg1.add_edge(r1, r5, bt =f'{cgrestype[0]}-B')
            fgrid = r5 +1
        elif cgrestype == 'X0':
            r1, r2, r3, r4 = (fgrid, fgrid + 1, fgrid + 2, fgrid + 3)
            cgid2fgid_hash[n] = [r1, r2, r3, r4]
            fg1.add_node(r1, type=f'{cgrestype}.X1', cgid=n)
            fg1.add_node(r2, type=f'{cgrestype}.Y', cgid=n)
            fg1.add_node(r3, type=f'{cgrestype}.X2', cgid=n)
            fg1.add_node(r4, type=f'{cgrestype}.B', cgid=n)
            fg1.add_edge(r1, r2, bt=f'{cgrestype[0]}-Y')
            fg1.add_edge(r2, r3, bt=f'{cgrestype[0]}-Y')
            fg1.add_edge(r4, r3, bt=f'{cgrestype[0]}-B')
            fgrid = r4 + 1
        elif cgrestype == 'X':
            r1,r2,r3,r4,r5,r6,r7 = (fgrid, fgrid + 1, fgrid + 2, fgrid + 3, fgrid + 4, fgrid + 5, fgrid + 6)
            cgid2fgid_hash[n] = [r1, r2, r3, r4, r5, r6, r7]
            fg1.add_node(r1, type=f'{cgrestype}.X1',cgid=n)
            fg1.add_node(r2, type=f'{cgrestype}.Y',cgid=n)
            fg1.add_node(r3, type=f'{cgrestype}.X2',cgid=n)
            fg1.add_node(r4, type=f'{cgrestype}.B1',cgid=n)
            fg1.add_node(r5, type=f'{cgrestype}.B2',cgid=n)
            fg1.add_node(r6, type=f'{cgrestype}.B3', cgid=n)
            fg1.add_node(r7, type=f'{cgrestype}.B4', cgid=n)
            fg1.add_edge(r1, r2, bt =f'{cgrestype[0]}-Y')
            fg1.add_edge(r2, r3, bt =f'{cgrestype[0]}-Y')
            fg1.add_edge(r3, r4, bt =f'{cgrestype[0]}-B')
            fg1.add_edge(r3, r5, bt =f'{cgrestype[0]}-B')
            fg1.add_edge(r1, r6, bt =f'{cgrestype[0]}-B')
            fg1.add_edge(r1, r7, bt =f'{cgrestype[0]}-B')
            fgrid = r7 +1
        elif cgrestype in {'Z0','A0'}:
            r1,r2 = (fgrid,fgrid+1)
            cgid2fgid_hash[n] = [r1, r2]
            fg1.add_node(r1, type=f'{cgrestype}.{cgrestype}', cgid=n)
            fg1.add_node(r2, type=f'{cgrestype}.B', cgid=n)
            fg1.add_edge(r1, r2, bt=f'{cgrestype[0]}-B')
            fgrid = r2 +1
        elif cgrestype == 'A':
            r1 = fgrid
            cgid2fgid_hash[n] = [r1,]
            fg1.add_node(r1, type=f'{cgrestype}.A', cgid=n)
            fgrid = r1 +1
    return fg1,cgid2fgid_hash

def get_aa_bond(u:mda.Universe,ag_res_nh:List[mda.core.groups.AtomGroup],cgtop:nx.Graph,mda2xml_hash):
    atoms = u.atoms
    cgbondmap = {} ## cgbond i,j -> aai, aaj
    for ag in ag_res_nh:
        tg = ag.bonds
        tg: mda.core.groups.topologyobjects.TopologyGroup
        allbonds = tg.indices
        for a1,a2 in allbonds:
            if atoms[a1].mass < 2 or atoms[a2].mass < 2:
                continue
            if atoms[a1].resid != atoms[a2].resid:
                r1 = atoms[a1].resid
                r2 = atoms[a2].resid
                cgbondmap[(mda2xml_hash[r1],mda2xml_hash[r2])] = (a1,a2)
                #print(r1,r2,a1,a2)
                cgtop.edges[(mda2xml_hash[r1],mda2xml_hash[r2])]['aa_map'] = (a1,a2)
    return cgbondmap,cgtop

def add_ag_to_fg1(fg1:nx.Graph,ag_res_nh:List[mda.core.groups.AtomGroup,],xml2mda_hash,res_pos_hash):
    for n in fg1.nodes:
        xmlrid = fg1.nodes[n]['cgid']
        mdarid = xml2mda_hash[xmlrid]
        mdarpos = res_pos_hash[mdarid]
        xmlres_ag = ag_res_nh[mdarpos]
        fgtype = fg1.nodes[n]['type']
        cgtype, fgtype = fgtype.split('.')
        fgres_ag = xmlres_ag[fgToaaidx_hash[cgtype][fgtype]]
        fg1.nodes[n]['aa_group'] = fgres_ag
        fg1.nodes[n]['aa_ids'] = fgres_ag.ids
        fg1.nodes[n]['fgtype'] = fgtype[0]
    return fg1

def add_agbond_to_fg1(fg1:nx.Graph,cg:nx.Graph,cgid2fgid_hash):
    for e in cg.edges:
        a1,a2 = cg.edges[e]['aa_map']
        fgis, fgjs = cgid2fgid_hash[e[0]], cgid2fgid_hash[e[1]]
        right_fgiid = None
        right_fgjid = None
        for i in fgis:
            iaa_ids = fg1.nodes[i]['aa_ids']
            if a1 in iaa_ids:
                right_fgiid = i
        for j in fgjs:
            jaa_ids = fg1.nodes[j]['aa_ids']
            if a2 in jaa_ids:
                right_fgjid = j
        right_fgit = fg1.nodes[right_fgiid]['fgtype']
        right_fgjt = fg1.nodes[right_fgjid]['fgtype']
        fg1.add_edge(right_fgiid,right_fgjid,bt=f'{right_fgit}-{right_fgjt}')
        #print(e,(right_fgiid,right_fgjid),(a1,a2))
    return fg1
def mdaResTogroRes(gro:str,res_hash):
    f = open(gro,'r')
    lines = f.readlines()
    lines = lines[2:-1]
    mdaRes = list(res_hash.keys())
    groRes = []
    for l in lines:
        rid = re.split(r'[A-Z]',l)[0]
        rid = int(rid)
        if l.split()[1][0] == 'H':
            continue
        if groRes == []:
            groRes.append(rid)
            continue
        if rid == groRes[-1]:
            continue
        else:
            groRes.append(rid)
    mda2xml_Hash = {}
    xml2mda_Hash = {}
    xmlRes = []
    for i,grorid in enumerate(groRes):
        xmlRes.append(grorid + 110*int(i/110))
    for xmlrid,mdarid in zip(xmlRes,mdaRes):
        mda2xml_Hash[mdarid] = xmlrid
        xml2mda_Hash[xmlrid] = mdarid
    return mda2xml_Hash,xml2mda_Hash
        #print(rid)

def get_exclude(fg1:nx.Graph,exclude:str,dihedrals:List,bondstop:nx.Graph,angles:List):
    excludelist = {}
    for n in fg1.nodes:
        if excludelist.get(n) is None:
            excludelist[n] = set()
    if exclude == 'bond':
        for e in bondstop.edges:
            excludelist[e[0]] = {e[1]}.union(excludelist[e[0]])
            excludelist[e[1]] = {e[0]}.union(excludelist[e[1]])
    if exclude == 'angle':
        for i,j,k in angles:
            excludelist[i] = {j,k}.union(excludelist[i])
            excludelist[j] = {i,k}.union(excludelist[j])
            excludelist[k] = {i,j}.union(excludelist[k])
    if exclude == 'dihedral':
        for i,j,k,l in dihedrals:
            excludelist[i] = {j,k,l}.union(excludelist[i])
            excludelist[j] = {i,k,l}.union(excludelist[j])
            excludelist[k] = {i,j,l}.union(excludelist[k])
            excludelist[l] = {i,j,k}.union(excludelist[l])
    return excludelist

def GetTypesList(xmlfile,grofile,tprfile,xtcfile=None,exclude=None):
    xml = XmlParser(xmlfile)
    gro = grofile
    if xtcfile is None:
        u = mda.Universe(tprfile)
    else:
        u = mda.Universe(tprfile,xtcfile)

    ag_res = u.atoms.split('residue')
    ag_res_nh = []
    for ag in ag_res:
        if ag.atoms[0].mass < 2:
            continue
        ag_res_nh.append(ag)



    res_hash = {}
    res_pos_hash = {}
    for i,ag in enumerate(ag_res_nh):
        res_hash[ag.resids[0]] = ag.resnames[0]
        res_pos_hash[ag.resids[0]] = i
    mda2xml_hash , xml2mda_hash = mdaResTogroRes(gro,res_hash)
    bonds = xml.data['bond']
    cgtop = nx.Graph()
    cgtop.add_nodes_from([(k+1,dict(type=xml.data['type'][k])) for k in range(len(xml.data['type']))])
    cgtop.add_edges_from([(b[1]+1,b[2]+1) for b in bonds])

    alltypes = ['Z','Z0','A','A0','X','X0']

    fg1, cgid2fgid_hash = cgTofg1(cgtop)

    cgbondmap,cgtop = get_aa_bond(u,ag_res_nh,cgtop,mda2xml_hash)
    fg1 = add_ag_to_fg1(fg1,ag_res_nh,xml2mda_hash,res_pos_hash)
    fg1 = add_agbond_to_fg1(fg1,cgtop,cgid2fgid_hash)
    bondstop,angles,dihedrals = cg_info(fg1)
    bonstop : nx.Graph
    bondstype_hash = {}
    anglestype_hash = {}
    dihedralstype_hash = {}
    bondtypes = set()
    angletypes = set()
    dihedraltypes = set()
    for e in bondstop.edges:
        bt_ = fg1.edges[e]['bt']
        first,second = bt_.split('-')
        order_ = ((first,second),(second,first))
        order = None
        for _ in order_:
            bt_ = f'{_[0]}-{_[1]}'
            if bt_ in bondtypes:
                order = _
        if order is None:
            order = order_[0]
            bondtypes.add(f'{order[0]}-{order[1]}')
        bt = f'{order[0]}-{order[1]}'
        bondstype_hash[e] = bt
    for angle in angles:
        i,j,k = angle
        it, jt ,kt = fg1.nodes[i]['fgtype'],fg1.nodes[j]['fgtype'],fg1.nodes[k]['fgtype']
        order_ = ((it,jt,kt),(kt,jt,it))
        order = None
        for _ in order_:
            at_ = f'{_[0]}-{_[1]}-{_[2]}'
            if at_ in angletypes:
                order = _
        if order is None:
            order = order_[0]
            angletypes.add(f'{order[0]}-{order[1]}-{order[2]}')
        at = f'{order[0]}-{order[1]}-{order[2]}'
        anglestype_hash[(i,j,k)] = at
    for di in dihedrals:
        i,j,k,l = di
        it, jt, kt, lt = fg1.nodes[i]['fgtype'],fg1.nodes[j]['fgtype'],fg1.nodes[k]['fgtype'],fg1.nodes[l]['fgtype']
        order_ = ((it,jt,kt,lt),(lt,kt,jt,it))
        order = None
        for _ in order_:
            dt_ = f'{_[0]}-{_[1]}-{_[2]}-{_[3]}'
            if dt_ in dihedraltypes:
                order = _
        if order is None:
            order = order_[0]
            dihedraltypes.add(f'{order[0]}-{order[1]}-{order[2]}-{order[3]}')
        dt = f'{order[0]}-{order[1]}-{order[2]}-{order[3]}'
        dihedralstype_hash[(i,j,k,l)] = dt

    nonbondslist = {}
    bondslist = {}
    angleslist = {}
    dihedralslist = {}
    exclusions = {}
    ### use types found to get the analyze list:
    ### example:
    ### bondslist['Z-B'] -> [<AtomGroup with 8 atoms>, <AtomGroup with 5 atoms>, <AtomGroup with 8 atoms>,...,]
    excludelist = get_exclude(fg1, exclude, dihedrals, bondstop, angles)
    for n in fg1.nodes:
        fgtype = fg1.nodes[n]['fgtype']
        if nonbondslist.get(fgtype) is None:
            nonbondslist[fgtype] = []
        if exclusions.get(fgtype) is None:
            exclusions[fgtype] = []
        nonbondslist[fgtype].append(fg1.nodes[n]['aa_group'])
        exclusions[fgtype].append(excludelist[n])

    for type_hash, type_list in zip((bondstype_hash,anglestype_hash,dihedralstype_hash),(bondslist,angleslist,dihedralslist)):
        for k in type_hash:
            if type_list.get(type_hash[k]) is None:
                type_list[type_hash[k]] = []
            for i in k:
                type_list[type_hash[k]].append(fg1.nodes[i]['aa_group'])
    print('find types:',set(nonbondslist.keys()))
    print('find bondtypes:', bondtypes)
    print('find angletypes:', angletypes)
    print('find dihedraltypes:', dihedraltypes)
    return u,exclusions, nonbondslist, bondslist, angleslist, dihedralslist



if __name__ == '__main__':
    # from cg2fg import GetTypesList
    ret = GetTypesList(xmlfile='z-5tiao-modified.xml',tprfile='zz-npt-PR-eq.tpr',grofile='z-npt-tuihuo-re-eq.gro',exclude='bond') ## exclude = ['bond','angle','dihedral']
    u, exclusions, nonbondslist, bondslist, angleslist, dihedralslist = ret
    #print(exclusions)
