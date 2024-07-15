from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import os
from sys import argv
import argparse

parser = argparse.ArgumentParser(description='Transition State Search')
parser.add_argument('-lammda',dest='lammda',type=float,help='Lammda: Coefficient of the distance between two reactants.')
parser.add_argument('-lNH',dest='lNH',type=float,help='Times relationship of length of the initial NH bond.')
parser.add_argument('-theta',dest='theta',type=float,help='Theta (degree) of CO bond and CC bond, elongate the broken bond CO, higher the longer.')

args = parser.parse_args()
lammda = args.lammda
lNH = args.lNH
theta = args.theta

def cos(a,b):
    return a.dot(b)/np.linalg.norm(a)/np.linalg.norm(b)

def sin(a,b):
    return np.linalg.norm(np.cross(a,b))/np.linalg.norm(a)/np.linalg.norm(b)

fndir = '1th'
molA = Chem.MolFromSmiles('Nc1cc(N)ccc1')
molB = Chem.MolFromSmiles('C1OC1COc2cc(Cc3cc(NCC4OC4)ccc3)ccc2')

molAh = AllChem.AddHs(molA)
molBh = AllChem.AddHs(molB)
AllChem.EmbedMolecule(molAh)
AllChem.EmbedMolecule(molBh)
#Chem.MolToPDBFile(molAh,os.path.join(fndir,'Nucls_1th.pdb'))
#Chem.MolToPDBFile(molBh,os.path.join(fndir,'epoxys_1th.pdb'))
c_A = AllChem.UFFOptimizeMolecule(molAh,maxIters=1000)
c_B = AllChem.UFFOptimizeMolecule(molBh,maxIters=1000)

Nucls = molAh.GetSubstructMatch(Chem.MolFromSmiles('N'))
epoxys = molBh.GetSubstructMatch(Chem.MolFromSmiles('C1OC1'))

Nid = Nucls[0]
for i in epoxys:
    a = molBh.GetAtomWithIdx(i)
    if a.GetSymbol() == 'O':
        Oid = i
    elif a.GetSymbol() == 'C' :
        flag = 1
        for nei in a.GetNeighbors():
            if nei.GetSymbol() == 'C' and nei.GetIdx() not in epoxys:
                flag = 0
            #elif nei.GetSymbol() == 'C':
            #    RCid = i
        if flag:
            Cid = i
for i in epoxys:
    if i != Oid and i != Cid:
        RCid = i
Nclus_neis = set()
for b in molAh.GetAtomWithIdx(Nid).GetBonds():
    i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
    Nclus_neis.add(i)
    Nclus_neis.add(j)
    if i == Nid:
        neiid = j
    else:
        neiid = i
    if molAh.GetAtomWithIdx(neiid).GetSymbol() == 'H':
        Hid = neiid
for i in Nclus_neis:
    if i != Hid and molAh.GetAtomWithIdx(i).GetSymbol() == 'H':
        HHid = i 
conformer_A = molAh.GetConformer(c_A)
conformer_B = molBh.GetConformer(c_B)
position_A = conformer_A.GetPositions()
position_B = conformer_B.GetPositions()
position_A -= position_A[Nid]
position_B -= position_B[Cid]
vNH = position_A[Hid]
vOC = position_B[Oid]
vCC = position_B[RCid]
vOC_ = vOC - vCC
uNH = vNH/np.linalg.norm(vNH)
uOC = vOC/np.linalg.norm(vOC)
u = np.cross(uNH,uOC)/np.linalg.norm(np.cross(uNH,uOC))
K = np.array([ [ 0    ,-u[2], u[1] ],
               [ u[2] ,    0,-u[0] ],
               [ -u[1], u[0],   0] ])
R =  cos(uNH,uOC) * np.eye(3) + sin(uNH,uOC) * K + (1-cos(uNH,uOC))*(np.outer(u,u))
R_180 = -np.eye(3) + 2 * np.outer(u,u)
R_180_NH = -np.eye(3) + 2 * np.outer(uNH,uNH)
delta = uNH * np.linalg.norm(vOC @ R - vNH)/2

cross = np.cross(vOC,np.cross(vOC,vCC))
up = cross/np.linalg.norm(cross)

uv = np.cross(vOC,vCC)/np.linalg.norm(np.cross(vOC,vCC))
dt = (360-theta)/180*np.pi
tR = np.cos(dt) * np.eye(3) + np.sin(dt) * np.array([[0,-uv[2],uv[1]],[uv[2],0,-uv[0]],[-uv[1],uv[0],0]]) + (1-np.cos(dt))*(np.outer(uv,uv))

#
d = lammda*np.linalg.norm(vOC)
dr = (360-60)/180*np.pi
dR =  np.cos(dr) * np.eye(3) + np.sin(dr) * np.array([[0,-uNH[2],uNH[1]],[uNH[2],0,-uNH[0]],[-uNH[1],uNH[0],0]]) + (1-np.cos(dr))*(np.outer(uNH,uNH))
el = 3
for i,p in enumerate(position_B):
    if i == Oid:
        #p_ = p - d*up + R @ delta * el
        #conformer_B.SetAtomPosition(i, p_ @ R + 0.4* (p_ - vCC) @ R)
        conformer_B.SetAtomPosition(i,((p - vCC) @ tR + vCC - d*up) @ R  )
        continue
    conformer_B.SetAtomPosition(i, p @ R - d*up @ R)
for i,p in enumerate(position_A):
    if i == HHid:
        #print(p/np.linalg.norm(p) @ R_180_NH)
        conformer_A.SetAtomPosition(i,p @ dR @ R_180_NH + 2*delta)
        continue
    if i == Hid:
        conformer_A.SetAtomPosition(i,p @ R_180_NH + 2*delta + uNH * np.linalg.norm(vNH) * (lNH - 1.0))
        continue
    conformer_A.SetAtomPosition(i, p @ R_180_NH + 2*delta )
    

#Chem.MolToPDBFile(molAh,os.path.join(fndir,'RNucls_1th.pdb'))
#Chem.MolToPDBFile(molBh,os.path.join(fndir,'Repoxys_1th.pdb'))
option = r'''%chk=ts.chk
%nprocshared=24
%mem=528MW
#p freq M062X/3-21G fopt=(ts,calcfc) iop(1/11=1,5/13=1) optcyc=300 scfcon=8 temperature=493

Title Card Required

0 1
'''
tdir = f'{theta:.2f}_{lNH:.2f}_{lammda:.2f}'
if os.path.exists(os.path.join(fndir,tdir)) is not True:
    os.mkdir(os.path.join(fndir,tdir))
gid = 0
fidx = open(os.path.join(fndir,tdir,f'fidx_1th_{theta:.2f}_{lNH:.2f}_{lammda:.2f}.txt'),'w')
f = open(os.path.join(fndir,tdir,f'g_1th_{theta:.2f}_{lNH:.2f}_{lammda:.2f}.com'),'w')
f.write(option)
for i,p in enumerate(conformer_A.GetPositions()):
    if i in [Nid,Hid]:
        fidx.write(f'{gid} ')
    a = molAh.GetAtomWithIdx(i)
    f.write(f' {a.GetSymbol()}    {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n')
    gid += 1
for i,p in enumerate(conformer_B.GetPositions()):
    if i in [Oid,Cid]:
        fidx.write(f'{gid} ')
    a = molBh.GetAtomWithIdx(i)
    f.write(f' {a.GetSymbol()}    {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n')
    gid += 1
f.write('\n')
f.close()
fidx.close()
