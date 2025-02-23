import os
import numpy as np 
import sys
sys.path.append('/home/lmy/HTSP/FPSG')
from CG.GenXYZ import xml as XML
from utils.io.xml_parser import XmlParser
import networkx as nx
import tqdm

if not os.path.exists('chunk'):
    os.mkdir('chunk')
xml = XmlParser('edit.xml')
box = np.array([xml.box.lx, xml.box.ly, xml.box.lz],dtype=float)
cg_sys = nx.Graph()
for i,t,p in zip(range(len(xml.data['type'])),xml.data['type'],xml.data['position']):
    cg_sys.add_node(i,t=t,x=p)
for b in xml.data['bond']:
    bt, i, j = b
    cg_sys.add_edge(i, j, bt=bt)
anglehash = {}
for a in xml.data['angle']:
    at, i, j, k = a
    anglehash[(cg_sys.nodes[i]['t'],cg_sys.nodes[j]['t'],cg_sys.nodes[k]['t'])] = at
    anglehash[(cg_sys.nodes[k]['t'],cg_sys.nodes[j]['t'],cg_sys.nodes[i]['t'])] = at

chunk_per_d = 2 # 1, 2, 3, 4, ..
#chunk_num = chunk_per_d ** 3

chunk_box = box/chunk_per_d
x = xml.data['position']
chunk_box_idx = x//chunk_box

chunk_idx = []
chunk_idx_hash = {}
chunk_particle_idx = {}
cnum = 0
for cid in chunk_box_idx:
    #if chunk_particle_idx.get(tuple(cid)) is None:
    #    chunk_particle_idx[tuple(cid)] = []
    if chunk_idx_hash.get(tuple(cid)) is None:
        chunk_idx_hash[tuple(cid)] = cnum
        cnum += 1
    chunk_idx.append(chunk_idx_hash[tuple(cid)])
chunk_idx = np.array(chunk_idx)
for cid in chunk_idx_hash:
    chunk_particle_idx[chunk_idx_hash[cid]] = (np.arange(len(chunk_idx))[chunk_idx == chunk_idx_hash[cid]])
chunk_num = len(chunk_particle_idx.keys())

for i in tqdm.tqdm(range(chunk_num),total=chunk_num,desc=f'Generating chunk {i:0>3d}'):
    boxi = box
    ps = x[chunk_particle_idx[i]]
    vs = np.zeros_like(ps)
    local_idx_hash = {}
    cg_i_sys = nx.Graph()
    local_nodes = set()
    ts = []
    for c,ip in enumerate(chunk_particle_idx[i]):
        local_idx_hash[ip] = c
        cg_i_sys.add_node(c,t=cg_sys.nodes[ip]['t'])
        local_nodes.add(ip)
        ts.append(cg_sys.nodes[ip]['t'])
    #else_idx = set()
    for ip in chunk_particle_idx[i]:
        neis = cg_sys.neighbors(ip)
        for inei in neis:
            if inei in local_nodes:
                cg_i_sys.add_edge(local_idx_hash[inei],local_idx_hash[ip],bt=cg_sys.edges[inei,ip]['bt'])
    bonds = {}
    angles = {}
    for ei, ej in cg_i_sys.edges:
        aeit = cg_i_sys.nodes[ei]['t']
        aejt = cg_i_sys.nodes[ej]['t']
        bonds[(ei, ej)] = [cg_i_sys.edges[ei, ej]['bt'], ei, ej]
        for neii in cg_i_sys.neighbors(ei):
            if neii == ej:
                continue
            aneiit = cg_i_sys.nodes[neii]['t']
            at = anglehash[aneiit,aeit,aejt]
            if angles.get((neii,ei,ej)) is not None or angles.get((ej,ei,neii)) is not None:
                continue
            else:
                angles[(neii,ei,ej)] = [at, neii, ei, ej]

        for neji in cg_i_sys.neighbors(ej):
            if neji == ei:
                continue
            anejit = cg_i_sys.nodes[neji]['t']
            at = anglehash[anejit,aejt,aeit]
            if angles.get((neji,ej,ei)) is not None or angles.get((ei,ej,neji)) is not None:
                continue
            else:
                angles[(neji,ej,ei)] = [at, neji, ej, ei]
        xmlw = XML(ps,vs,ts,bonds,angles,{},boxi)
        xmlw.writer(f'chunk/{i:0>3d}.xml',program='galamost')
                
        

