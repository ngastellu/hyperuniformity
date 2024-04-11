#!/usr/bin/env pythonw

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from os import path
from glob import glob

def moving_average(x, w):
    """Compute the moving average of array x with a sampling window of width w.
    Copied and pasted from: 
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy"""

datadir = '/Users/nico/Desktop/McGill/Research/simulation_outputs/hyperuniformity/'

#rfiles = glob(path.join(datadir,'radii_*.npy'))
#sys_names = [('_').join(path.basename(rf).split('_')[1:]).split('.')[0] for rf in rfiles]

rcParams['text.usetex'] = True
rcParams['font.size'] = 16
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

angstroms_per_pixel = 0.2

small_r_rfile = path.join(datadir,'small-r_radii_MAC_40x40.npy')
small_r_flucfile = path.join(datadir,'small-r_dfs_MAC_40x40.npy')

rfile = path.join(datadir,'large-r_radii_MAC_40x40.npy')
flucfile = path.join(datadir,'large-r_dfs_MAC_40x40.npy')

small_r_ls = np.load(small_r_rfile)
small_r_dfs = np.load(small_r_flucfile)


large_r_ls = np.load(rfile)[1:]
large_r_dfs = np.load(flucfile)[1:]

# get rid of unphysical scaling region
upper_bound = 400
upper_bound_index = np.max((large_r_ls <= upper_bound).nonzero()[0])

ls = np.hstack((small_r_ls,large_r_ls[:upper_bound_index]))
dfs = np.hstack((small_r_dfs,large_r_dfs[:upper_bound_index]))

test_bounds = np.arange(2,np.max(ls)-300,2)

lams = np.zeros(test_bounds.shape[0])
bs = np.zeros(test_bounds.shape[0])
rvals = np.zeros(test_bounds.shape[0])
inds = np.zeros(test_bounds.shape[0],dtype=int)

print(ls[2000])


for k, sboundary_l in enumerate(test_bounds):

    sboundary_index = np.max((ls <= sboundary_l).nonzero()[0])
    inds[k] = sboundary_index

    fit_radii = ls[sboundary_index:]
    fit_dfs = dfs[sboundary_index:]

    lams[k], bs[k], rvals[k], *_ = linregress(np.log(fit_radii),np.log(fit_dfs))

print(inds)
plt.plot(test_bounds*0.2,lams,'r-o',ms=3.0,lw=0.8)
plt.xlabel('Fit boundary [\AA]')
plt.ylabel('Scaling exponent')
plt.show()

plt.plot(test_bounds*0.2,rvals,'r-o',ms=3.0,lw=0.8)
plt.xlabel('Fit boundary [\AA]')
plt.ylabel('$r^2$ of fit')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xscale('log')
ax.set_yscale('log')

ax.plot(ls,dfs,'o',c='r',ms=1)
ax.plot(ls[inds[0]:], np.exp(bs[0])*(ls[inds[0]:]**lams[0]), 'b-', label='Fit: $\lambda = %f$'%lams[0],lw=1.0)

plt.suptitle('Density fluctuations in MAC (40nm $\\times$ 40nm)')
plt.legend()
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')
#ax.set_xlim(1,100)
plt.legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xscale('log')
ax.set_yscale('log')

ax.plot(ls,dfs,'o',c='r',ms=1)
ax.plot(ls[inds[-1]:], np.exp(bs[-1])*(ls[inds[-1]:]**lams[-1]), 'b-', label='Fit: $\lambda = %f$'%lams[-1],lw=1.0)

plt.suptitle('Density fluctuations in MAC (40nm $\\times$ 40nm)')
plt.legend()
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')
#ax.set_xlim(1,100)
plt.legend()
plt.show()
