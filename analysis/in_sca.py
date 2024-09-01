import time
from cmath import exp
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import tqdm
from numba import float64, complex128


@nb.jit(nopython=True,nogil=True)
def pbc(x,l):
    return x - l*np.rint(x/l)

@nb.jit(nopython=True, nogil=True)
def _q_vec(n_q, mid, q, dq, rtol=1e-3):
    ret = []
    for q_ary in np.ndindex(n_q):
        q_tmp = 0
        i = 0
        for qi in q_ary:
            q_tmp += ((qi - mid[i]) * dq[i]) ** 2
            i += 1
        q_tmp = q_tmp ** 0.5
        if abs(q_tmp - q) / q < rtol:
            ret.append(q_ary)
    return ret


@nb.guvectorize([(float64[:, :], float64[:, :], complex128[:])],
                '(n, p),(m, p)->(n)', target='parallel')
def exp_iqr(a, b, ret):
    for i in range(a.shape[0]):
        tmp = 0
        for j in range(b.shape[0]):
            prod = 0
            for k in range(b.shape[1]):
                prod += a[i, k] * b[j, k]
            tmp += exp(-1j * prod)
        ret[i] = tmp


def incoherrent_scattering(traj, q=None, dq=None, rtol=1e-3, q_vectors=None):
    n_dim = traj.shape[-1]
    s = time.time()
    q_vecs = q_vectors
    if q_vecs is None:
        if q is None or dq is None:
            raise ValueError("q, dq or q_vectors should be given!")
        if isinstance(dq, float):
            dq = np.asarray([dq,] * n_dim)
        dq = np.asarray(dq)
        n_q = np.asarray(q / dq + 0.5, dtype=np.int64) * 2 # nyquist theorem
        mid = n_q // 2
        n_q = tuple(n_q)
        last_len = 1
        while True:
            qvi = _q_vec(n_q, mid, q, dq, rtol) # f**k one more time, ignore it
            this_len = len(qvi)
            if (0 < this_len < 1000) or (this_len > 0 and last_len == 0):
                break
            if this_len > 1000:
                rtol = rtol / 1.1
            elif this_len == 0:
                rtol = rtol * 1.1
            last_len = this_len
        q_vecs = (np.asarray(qvi, dtype=np.float64) - mid) * dq
        #print("Generating q vecs in %.6fs, processing with num of q vectors: %d" % (time.time() - s, q_vecs.shape[0]))
    n_frames = traj.shape[0]
    corr = np.zeros((n_frames, traj.shape[1]), dtype=np.complex128)
    #if q_vecs.shape[0] > 1500:
    #    raise ValueError(f"Too many q vectors! {q_vecs.shape[0]} > 1500")
    for q_vec in tqdm.tqdm(q_vecs, desc="Processing with Q vectors", unit=r"Q vectors", ncols=100):
        traj_tmp = exp_iqr(traj, [q_vec])
        corr = corr + np.fft.ifft(np.abs(np.fft.fft(traj_tmp, axis=0, n=2 * n_frames)) ** 2,
                                  axis=0, n=2 * n_frames)[:n_frames]
    # traj = exp_iqr(traj, q_vecs) / q_vecs.shape[0]
    # corr = np.fft.ifft(np.abs(np.fft.fft(traj, axis=0, n=2*n_frames))**2, axis=0, n=2*n_frames)[:n_frames]
    corr = corr / np.arange(n_frames, 0, -1)[:, None] / q_vecs.shape[0]
    return corr, q_vecs

def DynamicStrfac_Fq(x,box,dt,qrange=(5,12),dq=1,types=[]):
    bl = box.mean()
    frames, nbeads, _ = x.shape
    #xM = pbc(x.reshape(frames,-1,2,3).mean(axis=-2).reshape(frames,-1,3),box.reshape(-1,1,3))
    x = pbc(x,box.reshape(-1,1,3))
    #xA = x[:,types=='A',:]
    #xB = x[:,types=='B',:]

    #_, nA, __ = xA.shape
    #_, nB, __ = xB.shape
    #_, nM, __ = xM.shape
    _, nM, __ = x.shape
    
    #FqA = {}
    #FqB = {}
    FqM = {}

    qr = np.arange(qrange[0],qrange[1],dq)
    #q = 3
    t = np.arange(frames)*dt
    #FqA[q] = np.vstack((t,np.abs(np.sum(incoherrent_scattering(xA,q,[2*np.pi/bl,2*np.pi/bl,2*np.pi/bl],rtol=1e-2)[0],axis=-1) / nA)))
    for q in tqdm.tqdm(qr,total=len(qr)):
        #FqA[q] = np.vstack((t,np.abs(np.sum(incoherrent_scattering(xA,q,[2*np.pi/bl,2*np.pi/bl,2*np.pi/bl],rtol=1e-2)[0],axis=-1) / nA)))
        #FqB[q] = np.vstack((t,np.abs(np.sum(incoherrent_scattering(xB,q,[2*np.pi/bl,2*np.pi/bl,2*np.pi/bl],rtol=1e-2)[0],axis=-1) / nB)))
        FqM[q] = np.vstack((t,np.abs(np.sum(incoherrent_scattering(x,q,[2*np.pi/bl,2*np.pi/bl,2*np.pi/bl],rtol=1e-2)[0],axis=-1) / nM)))
    return FqM

