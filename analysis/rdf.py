import numpy as np
import numba as nb
import networkx as nx
from numba import types as Type
from numba.typed import List, Dict
from numba import cuda,float64

nested_list_type = nb.types.ListType(nb.types.ListType(nb.types.int64))

@nb.jit(nopython=True,nogil=True)
def pbc(x,l):
    return x - l * np.rint(x/l)

@nb.jit(nopython=True)
#def rdf_calc(x,y,xid_hash,yid_hash,xex_hash,y_type,bl,nbin,bins,binr):
def rdf_calc(x,y,xid_hash,yid_hash,xex_hash,bl,nbin,bins,binr):
    #for i in xex_hash[ix]:
    #    xex_hash[ix] = set(xex_hash[ix])
    for t in range(len(x)):
        for ix,_x in enumerate(x[t]):
            for iy,_y in enumerate(y[t]):
                #if yid_hash[iy] in xex_hash[xid_hash[ix]][y_type]:
                if yid_hash[iy] in xex_hash[ix]:
                    continue
                r = (pbc(_x - _y,bl)**2).sum(axis=-1)**0.5
                nb = np.rint(r/bins)
                nb = int(nb)
                if nb < nbin:
                    binr[nb] += 1
    return binr/len(x)

def rdf_calc_ex(x,y,xid_hash,yid_hash,xex_hash,bl,nbin,bins,binr):
    #for i in xex_hash[ix]:
    #    xex_hash[ix] = set(xex_hash[ix])
    for t in range(len(x)):
        for ix,_x in enumerate(x[t]):
            for iy,_y in enumerate(y[t]):
                #if yid_hash[iy] in xex_hash[xid_hash[ix]][y_type]:
                if yid_hash[iy] not in xex_hash[ix]:
                    continue
                r = (pbc(_x - _y,bl)**2).sum(axis=-1)**0.5
                nb = np.rint(r/bins)
                nb = int(nb)
                if nb < nbin:
                    binr[nb] += 1
    return binr/len(x)

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

def cu_call(x,y,box,out,bins=0.01):
    frames = x.shape[0]
    Nx = x.shape[1]
    Ny = y.shape[1]
    #out = np.zeros((int(binr[1]/bins),))
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

def GetExclude(top,types):
    bondex = {t : {} for t in types}
    angleex = {t : {} for t in types}
    dihedralex = {t : {} for t in types}
    for i in top.nodes:
        t = top.nodes[i]['type']
        bondex[t][i] = {t : set() for t in types}
        angleex[t][i] = {t : set() for t in types}
        dihedralex[t][i] = {t : set() for t in types}
        for nei in top.neighbors(i):
            neit = top.nodes[nei]['type']
            bondex[t][i][neit].add(nei)
        bondex[t][i][t].add(i)
        angleex[t][i][t].add(i)
        dihedralex[t][i][t].add(i)
    for e in top.edges:
        i,j = e
        ti = top.nodes[i]['type']
        tj = top.nodes[j]['type']
        neii = set(top.neighbors(i))
        neij = set(top.neighbors(j))
        for nei in neii:
            neit = top.nodes[nei]['type']
            angleex[tj][j][neit].add(nei)
        for nei in neij:
            neit = top.nodes[nei]['type']
            angleex[ti][i][neit].add(nei)

        for inei in neii:
            neis = set(top.neighbors(inei))
            for iinei in neis:
                tiinei = top.nodes[iinei]['type']
                dihedralex[tj][j][tiinei].add(iinei)
        for inei in neij:
            neis = set(top.neighbors(inei))
            for iinei in neis:
                tiinei = top.nodes[iinei]['type']
                dihedralex[ti][i][tiinei].add(iinei)
        exclude = {}
        nodes = top.nodes
        for t in dihedralex:
            exclude[t] = {}
            for i in dihedralex[t]:
                exs = dihedralex[t][i]
                exclude[t][i] = {t_: set() for t_ in types}
                exclude[t][i][t].add(i)
                for et in exs:
                    for e in exs[et]:
                        exclude[t][i][et].add(e)
    return exclude

def ex_list(ex,ex_type):
    exlist = List()
    idx = sorted(list(ex.keys()))
    for i in idx:
        exlist.append(List(ex[i][ex_type]))
    return exlist
def ex_dict(ex,ex_type):
    exdict = Dict.empty(key_type=nb.types.int64,value_type=nb.types.ListType(nb.types.int64))
    idx = sorted(list(ex.keys()))
    for i in idx:
        #exdict[i] = Set.empty(types.int64)
        #for _ in ex[i][ex_type]:
        #    exdict[i].add(_)
        exdict[i] = List(ex[i][ex_type])
    return exdict

def rdf(x,types,box,cl):
    types_ = set()
    for t in types:
        types_.add(t)
    frames, nbeads, _ = x.shape
    nmol = int(nbeads/cl)
    xA = x[:,types == 'A',:]
    _, nA, __ = xA.shape
    xAid_hash = np.arange(nbeads)[types == 'A']
    xB = x[:,types == 'B',:]
    _, nB, __ = xB.shape
    xBid_hash = np.arange(nbeads)[types == 'B']
    bins = 0.1
    bl = box.mean()
    nbin = int(bl/2/bins)
    binrAB = np.zeros((nbin,))
    binrAA = np.zeros((nbin,))
    binrBB = np.zeros((nbin,))
    r = np.arange(nbin)*bins
    dV = np.diff(4/3*np.pi*r**3)
    top = nx.Graph()
    for ic in range(nmol):
        top.add_edges_from([(i + ic*cl,(i+1) + ic*cl) for i in range(cl-1)])
    for i,t in enumerate(types):
        top.nodes[i]['type'] = t
    exclude = GetExclude(top,types_)
    # cuda
    #binr_AB_ = cu_call(xA,xB,box,binr,bins) 
    #exout = rdf_calc_ex(xA,xB,xAid_hash,xBid_hash,ex_list(exclude['A'],'B'),bl,nbin,bins,binr) 
    #binr_AB = binr_AB_ - exout 

    #binr_AA_ = cu_call(xA,xA,box,binr,bins) 
    #exout = rdf_calc_ex(xA,xA,xAid_hash,xBid_hash,ex_list(exclude['A'],'A'),bl,nbin,bins,binr)
    #binr_AA = binr_AA_ - exout

    #binr_BB_ = cu_call(xB,xB,box,binr,bins) 
    #exout = rdf_calc_ex(xB,xB,xAid_hash,xBid_hash,ex_list(exclude['B'],'B'),bl,nbin,bins,binr)
    #binr_BB = binr_BB_ - exout
    
    # jit
    binr_AB = rdf_calc(xA,xB,xAid_hash,xBid_hash,ex_list(exclude['A'],'B'),bl,nbin,bins,binrAB)
    binr_AA = rdf_calc(xA,xA,xAid_hash,xAid_hash,ex_list(exclude['A'],'A'),bl,nbin,bins,binrAA)
    binr_BB = rdf_calc(xB,xB,xBid_hash,xBid_hash,ex_list(exclude['B'],'B'),bl,nbin,bins,binrBB)

    # nojit
    #binr_AB = rdf_calc(xA,xB,xAid_hash,xBid_hash,exclude['A'],'B',bl,nbin,bins,binr) 
    #binr_AA = rdf_calc(xA,xA,xAid_hash,xAid_hash,exclude['A'],'A',bl,nbin,bins,binr)
    #binr_BB = rdf_calc(xB,xB,xBid_hash,xBid_hash,exclude['B'],'B',bl,nbin,bins,binr)

    rdf_AB = binr_AB[:-1]/nA/(nB/bl**3)/dV
    rdf_AA = binr_AA[:-1]/nA/((nA-1)/bl**3)/dV
    rdf_BB = binr_BB[:-1]/nB/((nB-1)/bl**3)/dV
    return np.vstack((r[:-1],rdf_AB)).T , np.vstack((r[:-1],rdf_AA)).T , np.vstack((r[:-1],rdf_BB)).T
    

    


