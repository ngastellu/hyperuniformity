#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from os import path
from glob import glob

def moving_average(x, w):
    """Compute the moving average of array x with a sampling window of width w.
    Copied and pasted from: 
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy"""
    
    return np.convolve(x, np.ones(w), 'valid') / w

datadir = '/Users/nico/Desktop/simulation_outputs/hyperuniformity/random_overlap_tests/nsamples/'

#rfiles = glob(path.join(datadir,'radii_*.npy'))
#sys_names = [('_').join(path.basename(rf).split('_')[1:]).split('.')[0] for rf in rfiles]

rcParams['text.usetex'] = True
rcParams['font.size'] = 16
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


rfile = path.join(datadir,'radii_random2d.npy')
flucfiles = glob(path.join(datadir,'dfs-*.npy'))
#npts_arr = np.array([int(path.basename(f).split('-')[1].split('_')[0]) for f in flucfiles])
nsamples_arr = np.array([int(path.basename(f).split('-')[1].split('s')[0]) for f in flucfiles])
print(nsamples_arr)
 

#sort data in order of increasing phi (=fraction of occ sites)
#order = np.argsort(npts_arr)
#npts_arr = npts_arr[order]
#flucfiles = [flucfiles[i] for i in order]
#phis = npts_arr/(2000*2000)
#print(phis)
#clrs = [cm.plasma(x) for x in phis]

#sort data in order of number of samples
order = np.argsort(nsamples_arr)
nsamples_arr = nsamples_arr[order]
print(nsamples_arr)
flucfiles = [flucfiles[i] for i in order]
delta = 1.2*(nsamples_arr[-1] - nsamples_arr[0])
shifted_nsamples = (nsamples_arr - nsamples_arr[0])/delta
clrs = [cm.plasma(x) for x in shifted_nsamples]



fig = plt.figure()
ax = fig.add_subplot(111)

for flucfile, n, clr in zip(flucfiles,nsamples_arr,clrs):
#for s in sys_names:

   # rfile = path.join(datadir,'radii_%s.npy'%s)
    #flucfile = path.join(datadir,'subsampled_%s_avg_dfs.npy'%s)
    #flucfile = path.join(datadir,'%s_avg_dfs.npy'%s)
    ls = np.load(rfile)
    fluctuations = np.load(flucfile)
    #smooth_ls = moving_average(ls,40)
    #smooth_fluctuations = moving_average(fluctuations,40)
    #_, flucs2 = np.load('fluctuations_data.npy')

    #fig = plt.figure()
    #ax = fig.add_subplot(111)

    ax.set_xscale('log')
    ax.set_yscale('log')

    #npts = int(path.basename(flucfile).split('_')[1]) #nb of occupied sites in random structure
    #L = 2000 #size of grid
    #phi = npts/(L*L) #fraction of occupied sites

    #ax.plot(ls,fluctuations,'ro',ms=1)
    ax.plot(ls,fluctuations,'o',c=clr,ms=1,label='$N_s = %4.3f$'%n)
    #ax.plot(smooth_ls,smooth_fluctuations,'k-',lw=0.8)


ax.plot(ls,ls**(-2),'k--',lw=0.7,label='$\ell^{-d}$')
#ax.plot(ls,ls**(-3),'k-.',lw=0.7,label='$\ell^{-(d+1)}$')
#ax.plot(ls,fluctuations[0]*ls**(-3),'k-.',lw=0.7,label='$\ell^{-(d+1)}$')
plt.suptitle('Density fluctuations in random 2D point configurations with fraction of occupied sites $\phi$')
#plt.suptitle(' '.join(s.split('_')))
plt.legend()
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')
#ax.set_xlim(ls[0],ls[-1]+50)

plt.show()
