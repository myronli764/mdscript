import XmlParser
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
