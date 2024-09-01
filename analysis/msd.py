import numpy as np
import numba as nb

def msd_calc_fft(X,nmols):
    n_frames = len(X)
    S2 = np.sum((np.fft.ifft(np.abs(np.fft.fft(X,n=2*n_frames,axis=0))**2,axis=0,n=2*n_frames).real)[:n_frames],axis=-1)/np.arange(n_frames,0,-1).reshape(n_frames,1)
    Q = np.sum(X**2,axis=-1)
    ling = np.zeros((1,nmols))
    S1 = 2*np.sum(Q,axis=0) - np.cumsum(np.vstack((ling,Q[:-1]))+np.flip(np.vstack((Q[1:],ling)),axis=0),axis=0)
    S1 /= np.arange(n_frames,0,-1).reshape(n_frames,1)
    return np.mean(S1 - 2*S2,axis=-1)

def MSD(x,types,cl,dt):
    n_frames, nbeads, _ = x.shape
    xA = x[:,types=='A',:]
    _, nA, __ = xA.shape
    xB = x[:,types=='B',:]
    _, nB, __ = xB.shape
    CM = (x.reshape(n_frames,-1,cl,3).mean(axis=-2))
    _, nmols, __ = CM.shape
    t = np.arange(n_frames) * dt
    msdA = msd_calc_fft(xA,nA)
    msdB = msd_calc_fft(xB,nB)
    msdCM = msd_calc_fft(CM,nmols)
    return np.vstack((t,msdA)), np.vstack((t,msdB)), np.vstack((t,msdCM))
