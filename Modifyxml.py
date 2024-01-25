from utils.io.xml_parser import XmlParser
import networkx as nx
import numpy as np

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


xml = XmlParser('z-top-new-1tiao.xml')
XmlWriter(xml)
bonds = xml.data['bond']
types = xml.data['type']
N = len(types)
top = nx.Graph()
top.add_nodes_from([(i,dict(type=types[i])) for i in range(N)])
top.add_edges_from([(b[1],b[2],dict(bt=b[0])) for b in bonds])
endA = []
endAnei = {}
nonendA = []
endXZ = []
endXZnei = {}
for n in top.nodes:
    node = top.nodes[n]
    degree = nx.degree(top,n)
    if node['type'] == 'A':
        if degree == 1:
            nei_n = list(top.neighbors(n))[0]
            endA.append(n)
            endAnei[n] = nei_n
        else :
            nonendA.append(n)
            #print(degree)
    if node['type'] in ['X','Z']:
        if degree == 1:
            endXZ.append(n)
            nei_n = list(top.neighbors(n))[0]
            endXZnei[n] = nei_n

#print(endA)
#print(endAnei)
for n in endXZ:
    node = top.nodes[n]
    if node['type'] == 'Z':
        top.nodes[n]['type'] = 'Z0'
        top.edges[(endXZnei[n], n)]['bt'] = f'Z0-{top.nodes[endXZnei[n]]["type"]}'
    if node['type'] == 'X':
        top.nodes[n]['type'] = 'X0'
        top.edges[(endXZnei[n], n)]['bt'] = f'X0-{top.nodes[endXZnei[n]]["type"]}'

for n in endA:
    #print(top.has_edge(n,endAnei[n]))
    top.nodes[n]['type'] = 'A0'
    top.edges[(endAnei[n],n)]['bt'] = f'A0-{top.nodes[endAnei[n]]["type"]}'
    print(nx.degree(top,endAnei[n]))
new_bonds = []
bts = []
for e in top.edges:
    if top.edges[e]['bt'] == 'Z-A':
        new_bonds.append(('A-Z', e[0], e[1]))
        continue
    new_bonds.append((top.edges[e]['bt'],e[0],e[1]))
    if top.edges[e]['bt'] not in bts:
        bts.append(top.edges[e]['bt'])
print(bts)
xml.data['bond'] = new_bonds
new_types = []
for n in top.nodes:
    new_types.append(top.nodes[n]['type'])

xml.data['type'] = new_types
#print(new_types[16],new_types[17])
XmlWriter(xml)






#print(types)
#print(bonds)