import networkx as nx
import numpy as np
import re
from io import StringIO
from xml.etree import cElementTree

import numpy as np


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

from sys import argv
import networkx as nx
import numpy as np
top = nx.Graph()
xml = XmlParser(argv[1])
position_ = xml.data['position']
mass_ = xml.data['mass']
types_ = xml.data['type']
for i,t in enumerate(xml.data['type']):
    top.add_node(i)
for b in xml.data['bond']:
    top.add_edge(b[1],b[2],bondtype=b[0])

nlist = list(top.nodes())
for n in top.nodes:
    if top.degree(n) == 1:
        nlist.remove(n)
n_hash = {}
for i,n in enumerate(nlist):
    n_hash[n] = i

position = position_[nlist]
types = types_[nlist]
weight = mass_[nlist]
bond = []
for i,j in top.edges:
    bond.append((top.edges[(i,j)]['bondtype'],n_hash[i],n_hash[j]))
xml.data['position'] = position
xml.data['mass'] = mass
xml.data['type'] = types
xml.data['bond'] = bond
xml.data['image'] = xml.data['image'][nlist]
XmlWriter(xml)
