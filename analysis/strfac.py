import numpy as np
import numba as nb
import tqdm
import time

@nb.jit(nopython=True)
def hist_by_r_3d(hist,rx,ry,rz,dr,rmax):
    r0 = np.arange(dr,rmax,dr)
    count = np.zeros_like(r0)
    ret = np.zeros_like(r0)
    nrs = len(r0)
    D = len(rx)
    for i in range(D):
        for j in range(D):
            for k in range(D):
                r = np.sum((np.array((rx[i],ry[j],rz[k])))**2)**0.5
                nr = int(r/dr)
                if nr < nrs:
                    count[nr] += 1
                    ret[nr] += hist[i,j,k]
    count[ count == 0] = 1
    #print(count)
    return ret/count

@nb.jit(nopython=True,nogil=True)
def pbc(x,l):
    return x - l * np.rint(x/l)

def pad_by_pbc(x,box,npad=1):
    box = box.reshape(-1,1,3)
    total = int((2*npad+1)**3-1)
    k = int(2*npad+1)
    all_x = {}
    for i in range(k):
        for j in range(k):
            for l in range(k):
                pad_order = np.array([i,j,l])
                all_x[(i,j,l)] = x + pad_order*box
    px = x
    for order in all_x:
        if order == (0,0,0):
            continue
        px = np.concatenate((px,all_x[order]),axis=1)
    return px

def Strucf_Sq(x,types,box,npad=1):
    frames, nbeads, _ = x.shape
    xM = pbc(x.reshape(frames,-1,2,3).mean(axis=-2).reshape(frames,-1,3),box.reshape(-1,1,3))
    x = pbc(x,box.reshape(-1,1,3))
    bl = box.mean()
    xA = x[:,types == 'A',:]
    xB = x[:, types == 'B',:]
    
    _, nA, __ = xA.shape
    _, nB, __ = xB.shape
    _, nM, __ = xM.shape

    #pxA = pad_by_pbc(xA,box,npad)
    #pxB = pad_by_pbc(xB,box,npad)
    #pxM = pad_by_pbc(xM,box,npad)
    

    bins = 1
    k = 1#int(2*npad+1)
    nbin = int(bl/bins)
    kbl = k*bl
    #pxA = pbc(pxA,kbl)
    #pxB = pbc(pxB,kbl)
    #pxM = pbc(pxM,kbl)
    binrange = (-(k*bl)/2,bl*k/2)
    freq = np.fft.fftfreq(nbin) * 2 * np.pi
    #qx, qy, qz = np.meshgrid(freq,freq,freq)
    qx, qy, qz = (freq,freq,freq)
    dq = freq[1] - freq[0]
    qmax = 1.*np.pi/bins
    q0 = np.arange(dq,qmax,dq)
    SqA_q = np.zeros_like(q0)
    SqB_q = np.zeros_like(q0)
    SqM_q = np.zeros_like(q0)
    cut = frames
    for f in tqdm.tqdm(range(frames),total=frames):
        if f == cut:
            break
        s = time.time()
        rhoA, qA = np.histogramdd(pbc(xA[f], box[f]),bins=(nbin, )*3, range=[(-_/2., _/2) for _ in box[f]])
        fft_rhoA = np.fft.fftn(rhoA)
        SqA = fft_rhoA * np.conjugate(fft_rhoA)
        SqA = SqA / (nA) 
        SqA_q += hist_by_r_3d(SqA.real,freq,freq,freq,dq,qmax)
        
        rhoB, qB = np.histogramdd(xB[f],bins=nbin,range=(binrange,binrange,binrange))
        fft_rhoB = np.fft.fftn(rhoB)
        SqB = fft_rhoB*np.conjugate(fft_rhoB)/ nB
        SqB_q += hist_by_r_3d(SqB.real,qx,qy,qz,dq,qmax)

        rhoM, qM = np.histogramdd(xM[f],bins=nbin,range=(binrange,binrange,binrange))
        fft_rhoM = np.fft.fftn(rhoM)
        SqM = fft_rhoM*np.conjugate(fft_rhoM)/ nM
        SqM_q += hist_by_r_3d(SqM.real,qx,qy,qz,dq,qmax)
        
        #_rhoM, _qM = np.histogramdd(-pxM,bins=nbin,range=binrange)
        #print(f'Calculate frame {f}, taking time {time.time()-s:.3f}')
    return np.vstack((q0,SqA_q/cut)), np.vstack((q0,SqB_q/cut)), np.vstack((q0,SqM_q/cut))
    
    


