import MDAnalysis as mda
import numpy as np
from sys import argv
import argparse
from numba import jit,cuda,float64
import networkx as nx
import re
from io import StringIO
from xml.etree import cElementTree
from scipy.stats import circmean
import math
def control_in(control_file):
    pass
r'''
 this analysic program provide pairs, bonds and angle distribution 
 for the given system. It needs .gro .xtc .xml .tpr files for input, 
 used for the CG bead index determination, AA trajectory, CG topology
 and AA topology.
 TODO
 1. modify the logic of the program
 2. use jit for subjecting excluded calculation
'''
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
def XmlWriter(xml:XmlParser,filename='modified.xml'):
    lx,ly,lz,xy,xz,yz = np.array((xml.box.lx, xml.box.ly, xml.box.lz,xml.box.xy, xml.box.xz,xml.box.yz),dtype=float)
    program = 'galamost'
    pos_ = xml.data['position']
    image_ = xml.data['image']
    mass_ = xml.data['mass']
    types_ = xml.data['type']
    n_atoms = len(pos_)
    position = ''
    image = ''
    mass = ''
    types = ''
    for p,i,m,t in zip(pos_,image_,mass_,types_):
        position += '%.6f %.6f %.6f\n' % (p[0],p[1],p[2])
        image += '%d %d %d\n' % (i[0],i[1],i[2])
        mass += '%.6f\n' % (m)
        types += '%s\n' % (t)
    bonds = ''
    bonds_ = xml.data['bond']
    n_bonds = len(bonds_)
    for b in bonds_:
        bonds += '%s %d %d\n' % (b[0],b[1],b[2])
    f = open(filename,'w')
    f.write(
        f'<?xml version ="1.0" encoding ="UTF-8" ?>' +
        f'\n<{program}_xml version="1.3">' +
        f'\n<configuration time_step="0" dimensions="3" natoms="{n_atoms:d}" >' +
        f'\n<box lx="{lx:.8f}" ly="{ly:.8f}" lz="{lz:.8f}" xy="{xy:8f}" xz="{xz:8f}" yz="{yz:8f}"/>' +
        f'\n<position num="{n_atoms:d}">\n{position}</position>\n<type num="{n_atoms:d}">\n{types}</type>' +
        f'\n<image num="{n_atoms:d}">' +
        f'\n{image}</image>' +
        f'\n<mass num="{n_atoms:d}">' +
        f'\n{mass}</mass>' +
        f'\n<bond num="{n_bonds:d}">' +
        f'\n{bonds}' +
        f'\n</bond>' +
        f'\n</configuration>' +
        f'\n</{program}_xml>'
        )
    f.close()

parser = argparse.ArgumentParser(description='Bonded distribution calculation by bead')

parser.add_argument('-xtc',dest='xtc',type=str,help='XTC file')
parser.add_argument('-tpr',dest='tpr',type=str,help='TPR file')
parser.add_argument('-xml',dest='xml',type=str,help='XML file')
parser.add_argument('-gro',dest='gro',type=str,help='GRO file')
args = parser.parse_args()

xtc = args.xtc
tpr = args.tpr
xml = args.xml
gro = args.gro

def GetBondNeiBead(ag):
    nodes = set()
    edges = set()
    for a in ag:
        tg = a.bonds
        rid = a.resid
        for b in tg:
            for neia in b.atoms:
                if neia.name[:1] == 'H':
                    continue
                if rid == neia.resid:
                    continue
                edges.add((rid,neia.resid))
                nodes.add(rid)
                nodes.add(neia.resid)
    return rid, nodes, edges

@jit(nopython=True)
def pbc(x,box):
    return x - box * np.rint(x/box)


# the rdf calculated here is provided by calculation of all pairs and subjection of the neighbors later
@cuda.jit(device=True)
def rint(x):
    return float64(int(x - 0.5)) if x < 0.0 else float64(int(x + 0.5))


@cuda.jit
def cu_rdf_kernel(x,y,box,out,nbin,bins,frames):
    i,j = cuda.grid(2)
    if i >= x.shape[1] or j >= y.shape[1]:
        return
    #if i == j:
    #    return
    for f in range(frames):
        dx = x[f][i][0] - y[f][j][0] - box[f][0] * rint((x[f][i][0] - y[f][j][0]) / box[f][0] )
        dy = x[f][i][1] - y[f][j][1] - box[f][1] * rint((x[f][i][1] - y[f][j][1]) / box[f][1] )
        dz = x[f][i][2] - y[f][j][2] - box[f][2] * rint((x[f][i][2] - y[f][j][2]) / box[f][2] )
        dr = (dx**2 + dy**2 + dz**2)**0.5
        ibin = int(dr/bins)
        if ibin < nbin:
            cuda.atomic.add(out,ibin,1) 

def cu_call(x,y,box,binr,bins=0.01):
    frames = x.shape[0]
    Nx = x.shape[1]
    Ny = y.shape[1]
    out = np.zeros((int(binr[1]/bins),))
    nbin = len(out)
    d_x = cuda.to_device(x)
    d_y = cuda.to_device(y)
    d_box = cuda.to_device(box)
    d_out = cuda.to_device(out)
    tpb = 32
    blocksize = (tpb,tpb)
    gridsize = (math.ceil(Nx/blocksize[0]),
                math.ceil(Ny/blocksize[1]))
    cu_rdf_kernel[gridsize,blocksize](d_x,d_y,d_box,d_out,nbin,bins,frames)
    out_host = d_out.copy_to_host()
    return out_host/frames

    

xml = XmlParser(xml)
types_set = set()
cgtop = nx.Graph()
types = xml.data['type']
bonds = xml.data['bond']
for bt,i,j in bonds:
    cgtop.add_node(i,type=types[i])
    cgtop.add_node(j,type=types[j])
    cgtop.add_edge(i,j)
    types_set.add(types[i][0])
    types_set.add(types[j][0])
    
position_types = {}
position_types_idx = {}
for t in types_set:
    position_types[t] = []
    position_types_idx[t] = []

u = mda.Universe(tpr,xtc)
idx = []

for a in u.atoms:
    if a.name[:1] == 'H':
        continue
    idx.append(a.id)
atoms = u.atoms[idx]
ags = atoms.split('residue')
res_cgid_hash = {}
for i,ag in enumerate(ags):
    res_cgid_hash[ag[0].resid] = i


top = nx.Graph()
trajectory = u.trajectory
cgp = np.zeros((trajectory.n_frames,len(ags),3))
box = []#np.zeros((trajectory.n_frames,3))
meta = {}
fgro = open(gro,'r')
grodata = fgro.readlines()[2:-1]
resid = {}
for i,r in enumerate(grodata):
    resid[i] = int(r[:5].split()[0])

#checklog = open('check.log','w')
import tqdm
import time
for i,ag in enumerate(ags):
    ibead = resid[ag[0].id] - 1
    ty = types[ibead][0]
    position_types_idx[ty].append(ibead)
idx_hash = {}
for k in (position_types_idx):
    for i,idx in enumerate(position_types_idx[k]):
        idx_hash[idx] = i
#print(idx_hash)
#raise

for t in tqdm.tqdm(trajectory,total=trajectory.n_frames):
    bl = t.dimensions[:3]*0.1
    box.append(bl)
    for ty in types_set:
        position_types[ty].append([])
    for i,ag in enumerate(ags):
        s = time.time()
        #cm = ag.center_of_mass(unwrap=True)*0.1
        cm = circmean(ag.positions*0.1,high=bl[0]/2,low=-bl[0]/2,axis=0)
        ibead = resid[ag[0].id]-1 #res_cgid_hash[ag[0].resid]
        #checklog.write(f'{ag[0].resid} {ag[0].resname} ||| {resid[ag[0].id]} {types[resid[ag[0].id]-1]} ||| {ibead+1} {ibead} {types[ibead]}\n')
        nodes = list(cgtop.neighbors(ibead))
        edges = [(ibead,n) for n in nodes]
        top.add_nodes_from(nodes)
        top.add_edges_from(edges)
        if top.nodes[ibead].get('position') is None:
            top.nodes[ibead]['position'] = []
        top.nodes[ibead]['position'].append(cm)
        position_types[types[ibead][0]][t.frame].append(cm)
        if ag[0].resname != types[ibead]:
            print(ag[0].resname,types[ibead])
        top.nodes[ibead]['type'] = ag[0].resname[0]
    #if t.frame > 1000:
    #    break


for k in types_set:
    position_types[k] = np.array(position_types[k])
N_types = {}
exclude = {}
for i,k in enumerate(types_set):
    N_types[k] = position_types[k].shape[1]
    print(k,position_types[k].shape)
    for j,k_ in enumerate(types_set):
        if i > j:
            continue
        if exclude.get((k,k_)) is not None:
            continue
        if exclude.get((k_,k)) is not None:
            continue
        exclude[(k,k_)] = {}
pairs = exclude.keys()
box = np.array(box)
bins = 0.01
boxl = box[0][0]
nbin = int(boxl/2/bins)
r = np.arange(0,nbin) * bins + bins/2
#exclude = {}
for i,j in tqdm.tqdm(top.edges,total=top.number_of_edges(),desc='Getting exclude'):
    ni = top.nodes[i]
    nj = top.nodes[j]
    ti, tj = top.nodes[i]['type'], top.nodes[j]['type']
    if (ti,tj) in pairs:
        order = (ti,tj)
    elif (tj,ti) in pairs:
        order = (tj,ti)
    if exclude[order].get(i) is None:
        exclude[order][i] = set()
    if exclude[order].get(j) is None:
        exclude[order][j] = set()
    exclude[order][i].add(j)
    exclude[order][j].add(i)
    if exclude[(ti,ti)].get(i) is None:
        exclude[(ti,ti)][i] = set()
    if exclude[(tj,tj)].get(j) is None:
        exclude[(tj,tj)][j] = set()
    exclude[(ti,ti)][i].add(i)
    exclude[(tj,tj)][j].add(j)
    for neii in top.neighbors(i):
        if neii == j:
            continue
        #
        tneii = top.nodes[neii]['type']
        if (ti,tneii) in pairs:
            neiorder = (ti,tneii)
        elif (tneii,ti) in pairs:
            neiorder = (tneii,ti)
        if exclude[neiorder].get(i) is None:
            exclude[neiorder][i] = set()
        exclude[neiorder][i].add(neii)
        # exclude angle of j
        if (tj,tneii) in pairs:
            neiorder = (tj,tneii)
        elif (tneii,tj) in pairs:
            neiorder = (tneii,tj)
        if exclude[neiorder].get(j) is None:
            exclude[neiorder][j] = set()
        exclude[neiorder][j].add(neii)
    for neij in top.neighbors(j):
        if neij == i:
            continue
        #
        tneij = top.nodes[neij]['type']
        if (tj,tneij) in pairs:
            neiorder = (tj,tneij)
        elif (tneij,tj) in pairs:
            neiorder = (tneij,tj)
        if exclude[neiorder].get(j) is None:
            exclude[neiorder][j] = set()
        exclude[neiorder][j].add(neij)
        # exclude angle of i
        if (ti,tneij) in pairs:
            neiorder = (ti,tneij)
        elif (tneij,ti) in pairs:
            neiorder = (tneij,ti)
        if exclude[neiorder].get(i) is None:
            exclude[neiorder][i] = set()
        exclude[neiorder][i].add(neij)
for k in exclude:
    if exclude[k] == {}:
        print(k)
## TODO
# use numpy.ndarray to remove the frame cycle
#@jit(nopython=True,nogil=True)
def jit_rdf(x,y,box,exclude_hash,x_types_idx,idx_hash,binsize,binr,nbin):
    y_ex_ = []
    for i,xi in enumerate(x[0]):
        idx = x_types_idx[i]
        y_ex_.append([])
        if exclude_hash.get(idx) is None:
            #y_ex_[i].append([])
            continue
        exclude = list(exclude_hash[idx])
        for e in exclude:
            y_ex_[i].append(idx_hash[e])
    #for t in range(len(box)):
    for t in tqdm.tqdm(range(len(box)),total=len(box),desc='subjecting exclude'):
        bl = box[t]
        for i,xi in enumerate(x[t]):
            y_ = y[t][y_ex_[i]]
            for yj in y_:
                r = np.sum((pbc(xi - yj,bl))**2)**0.5
                if int(r/binsize) < nbin:
                    binr[int(r/binsize)] += 1
    return binr/len(box)


for i,ti in enumerate(types_set):
    for j,tj in enumerate(types_set):
        if i > j:
            continue
        if (ti,tj) in pairs:
            pair = (ti,tj)
        elif (tj,ti) in pairs:
            pair = (tj,ti)
        hist_ = cu_call(position_types[ti],position_types[tj],box,(0,r[-1]),bins)
        out = np.zeros((int(r[-1]/bins),))
        out = jit_rdf(position_types[ti],position_types[tj],box,exclude[pair],position_types_idx[ti],idx_hash,bins,out,nbin)
        #out[0] = 0
        #print(pair)
        #if set(pair) in [set(['Y','B']),set(['A','Y']),set(['X','A'])]:
            #print(pair,exclude[pair])
            #print(ti,tj,hist_[0],out[0])
        #print(ti,tj,hist_[0],out[0])
        hist = hist_ - out
        V = (box[:,:1]*box[:,1:2]*box[:,2:3]).mean()
        dV = np.diff(4/3.0*np.pi*r**3)
        rho = N_types[ti]*N_types[tj]/V
        gr = hist/dV/rho
        np.savetxt(f'rdf_{ti}-{tj}.txt',np.vstack((r[:-1],gr)).T)
        #import matplotlib.pyplot as plt
        #plt.plot(r[:-1],hist_/dV/rho+0.1,label=f'{pair} no exclude')
        #plt.plot(r[:-1],out/dV/rho+0.2,label=f'{pair} to be exclude')
        #plt.plot(r[:-1],gr,label=pair)
        #plt.legend()
        #plt.show()

bonds = {}
angles = {}
for i,j in tqdm.tqdm(top.edges,total=top.number_of_edges()):
    ni = top.nodes[i]
    nj = top.nodes[j]
    ni['position'] = np.array(top.nodes[i]['position'])
    nj['position'] = np.array(top.nodes[j]['position'])
    bt = (ni['type'],nj['type'])
    bt_ = (nj['type'],ni['type'])
    if bonds.get(bt) is not None:
        order = bt
    elif bonds.get(bt_) is not None:
        order = bt_
    else:
        order = bt
        bonds[order] = []
    bonds[order].append((pbc(ni['position']-nj['position'],box)**2).sum(axis=-1)**0.5)
    for neii in top.neighbors(i):
        if neii == j:
            continue
        nneii = top.nodes[neii]
        agt = (nneii['type'],ni['type'],nj['type'])
        agt_ = (nj['type'],ni['type'],nneii['type'])
        if angles.get(agt) is not None:
            order = agt
        elif angles.get(agt_) is not None:
            order = agt_
        else:
            order = agt
            angles[order] = []
        a = pbc(nj['position']-ni['position'],box) # vij
        b = pbc(nneii['position']-ni['position'],box) # vinneii
        #import pickle
        #pickle.dump(a,open('a.pkl','wb'))
        #pickle.dump(b,open('b.pkl','wb'))
        angles[order].append(np.arccos((a*b).sum(axis=-1) / (np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)))*(180/np.pi))
    for neij in top.neighbors(j):
        if neij == i:
            continue
        nneij = top.nodes[neij]
        agt = (nneij['type'],nj['type'],ni['type'])
        agt_ = (ni['type'],nj['type'],nneij['type'])
        if angles.get(agt) is not None:
            order = agt
        elif angles.get(agt_) is not None:
            order = agt_
        else:
            order = agt
            angles[order] = []
        a = pbc(ni['position']-nj['position'],box) # vji
        b = pbc(nneij['position']-nj['position'],box) # vjnneij
        angles[order].append(np.arccos((a*b).sum(axis=-1) / (np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)))*(180/np.pi))

#root = 'result'
bonds_dis = {}
angles_dis = {}

for bt in bonds:
    print(bt)
    hist, r = np.histogram(bonds[bt],bins=1000,density=True)
    bonds_dis[bt] = np.vstack((r[:-1],hist)).T
    np.savetxt(f'{bt[0]}-{bt[1]}.txt',bonds_dis[bt])
for agt in angles:
    hist, r = np.histogram(angles[agt],bins=360,range=(0,180),density=True)
    angles_dis[agt] = np.vstack((r[:-1],hist)).T
    np.savetxt(f'{agt[0]}-{agt[1]}-{agt[2]}.txt',angles_dis[agt])

