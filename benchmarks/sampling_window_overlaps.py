#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from itertools import combinations,starmap

def overlap(r, R1, R2):
    """Calculates the overlap area between two disks with radius r and centers with positions
    at R1 and R2."""

    R = np.linalg.norm(R1 - R2)

    if R >= 2*r: 
        return 0
    
    else:
        return 2 * ( (r**2)*np.arccos(R/(2*r)) - R*0.25*np.sqrt(4*(r**2) - (R**2)) ) / (np.pi*(r**2))


def plot_centers(rdata):
    """Scatter plot of sampling window centers, whose colour is defined by the radius of the 
    sampling window."""

    if rdata.ndim == 2:
        radii = rdata[:,0]
        centers = rdata[:,1:]

    elif rdata.ndim == 3:
        radii = rdata[:,:,0].reshape((rdata.shape[0]*rdata.shape[1]))
        centers = rdata[:,:,1:].reshape((rdata.shape[0]*rdata.shape[1],2))

    else:
        print('ERROR -- plot_centers: Unexpected shape for rdata')
        return None

    fig, ax1 = plt.subplots()

    ye = ax1.scatter(centers[:,0],centers[:,1],c=radii,s=5.0,cmap='plasma')
    cbar = fig.colorbar(ye,ax=ax1)#,orientation='horizontal')
    ax1.set_aspect('equal')
    plt.show()


def overlap_stats(rdata, nwindows_per_r=10):
    """Computes average overlap of sampling windows of different radii for a 2D density fluctuations
    calculation run."""

    if rdata.ndim == 2:
        radii = rdata[:,0]
        centers = rdata[:,1:]

    elif rdata.ndim == 3:
        radii = rdata[:,:,0].reshape((rdata.shape[0]*rdata.shape[1]))
        centers = rdata[:,:,1:].reshape((rdata.shape[0]*rdata.shape[1],2))

    else:
        print('ERROR -- plot_centers: Unexpected shape for rdata')
        return None

    # Partition window data into sets with equal radii
    N = radii.shape[0]
    if N % nwindows_per_r != 0:
        print('ERROR: Number of data points (%d) is not divisible by number of samples per radius \
                (%d).'%(N,nwindows_per_r))
        return None

    avg_overlaps = np.zeros(N // nwindows_per_r)
    var_overlaps = np.zeros(N // nwindows_per_r)

    for n in range(N // nwindows_per_r):
        print(n)
        print(n*nwindows_per_r)
        print((n+1)*nwindows_per_r)
        rvals = radii[n*nwindows_per_r:(n+1)*nwindows_per_r].reshape(nwindows_per_r,1)
        print(np.all(rvals==rvals[0]))
        Xs = centers[n*nwindows_per_r:(n+1)*nwindows_per_r]
        center_combinations = combinations(Xs,2) #list of all pairs of centers
        args = [(rvals[0], *xs) for xs in center_combinations]
        #print(args)
        overlaps = np.array(list(starmap(overlap,args))).flatten()
        if np.any(overlaps < 0): print('ye')
        #print(overlaps)
        avg_overlaps[n] = np.mean(overlaps)
        var_overlaps[n] = np.var(overlaps)

    return avg_overlaps, var_overlaps



def moving_average(x, w):
    """Compute the moving average of array x with a sampling window of width w.
    Copied and pasted from: 
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy"""
    
    return np.convolve(x, np.ones(w), 'valid') / w


# ******* MAIN *******

rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

nsamples_arr = np.array([50,100,200,300])

deltaN = 1.1*(nsamples_arr[-1] - nsamples_arr[0])
rescaled_Ns = (nsamples_arr - nsamples_arr[0])/deltaN
clrs = [cm.plasma(x) for x in rescaled_Ns]

for n, clr in zip(nsamples_arr,clrs):

    rdata = np.load('../../simulation_outputs/hyperuniformity/random_overlap_tests/nsamples/rdata-%dsamples.npy'%n)

    print(rdata.shape)

    assert n == rdata.shape[1], 'rdata has unexpected shape'

    #plot_centers(rdata)

    mus, sigmas = overlap_stats(rdata,nwindows_per_r=n)

    print(mus.shape)

    ls = rdata[:,0,0]

    # Compute moving averages to smooth out the data
    nsample = 50
    smooth_mus = moving_average(mus, nsample)
    smooth_ls = moving_average(ls, nsample)

    plt.plot(ls,mus,'-',c=clr,lw=0.8,label='%d samples')
    #plt.plot(smooth_ls, smooth_mus,'k--',lw=0.8)
    #plt.plot(ls,sigmas,'b-',lw=0.8,label='$\sigma_{\\alpha}(\ell)$')
plt.xlabel('$\ell$')
plt.ylabel('$\langle \\alpha(\ell)\\rangle$')
plt.suptitle('Statistics of scaled intersection area $\\alpha$ of sampling windows with radius $\ell$')
plt.legend()
plt.show()

r1 = np.zeros(2)
r2 = np.array([500,700])
print(overlap(500,r1,r2))


