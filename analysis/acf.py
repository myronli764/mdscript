import numpy as np
import os
def ACF(input):
    pos, posAA, posBB = input
    cl = 16
    nframes = len(pos)
    t = np.arange(0,nframes) * 10
    ## pos AB
    posAB = pos.reshape(nframes,-1,cl,3)
    rAB = np.diff(posAB,axis=2)
    rAB = rAB.reshape((nframes, -1, 3))
    acf_AB = np.mean(
        np.sum(((np.fft.ifft(np.abs(np.fft.fft(rAB, n=2 * nframes, axis=0) ** 2), n=2 * nframes, axis=0)).real)[:nframes],
               axis=-1), axis=-1) / np.arange(nframes, 0, -1)
    acf_AB =  acf_AB / np.sum(rAB ** 2, axis=-1).mean()
    ## pos AA
    posAA = posAA.reshape(nframes, -1, int(cl/2), 3)
    rAA = np.diff(posAA, axis=2)
    rAA = rAA.reshape((nframes, -1, 3))
    acf_AA = np.mean(
        np.sum(
            ((np.fft.ifft(np.abs(np.fft.fft(rAA, n=2 * nframes, axis=0) ** 2), n=2 * nframes, axis=0)).real)[:nframes],
            axis=-1), axis=-1) / np.arange(nframes, 0, -1)
    acf_AA = acf_AA / np.sum(rAA ** 2, axis=-1).mean()
    ## pos BB
    posBB = posBB.reshape(nframes, -1, int(cl/2), 3)
    rBB = np.diff(posBB, axis=2)
    rBB = rBB.reshape((nframes, -1, 3))
    acf_BB = np.mean(
        np.sum(
            ((np.fft.ifft(np.abs(np.fft.fft(rBB, n=2 * nframes, axis=0) ** 2), n=2 * nframes, axis=0)).real)[:nframes],
            axis=-1), axis=-1) / np.arange(nframes, 0, -1)
    acf_BB = acf_BB / np.sum(rBB ** 2, axis=-1).mean()
    return (np.vstack((t,acf_AB)), np.vstack((t,acf_AA)), np.vstack((t,acf_BB)))
