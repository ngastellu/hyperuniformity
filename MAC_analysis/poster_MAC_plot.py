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
    
    return np.convolve(x, np.ones(w), 'valid') / w

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

crystal_rfile = path.join(datadir, 'crystal2d_radii.npy')
crystal_flucfile = path.join(datadir, 'dfs_crystal2d.npy')

#crystal_rfile = path.join(datadir, 'radii_graphene.npy')
#crystal_flucfile = path.join(datadir, 'dfs_graphene.npy')

cls = np.load(crystal_rfile)[2:]
cdfs = np.load(crystal_flucfile)[2:]


small_r_ls = np.load(small_r_rfile)
small_r_dfs = np.load(small_r_flucfile)


ls = np.load(rfile)[1:]
fluctuations = np.load(flucfile)[1:]

# only keep the 'large l' part of the original dataset
ending_index = np.max((ls < 400).nonzero()[0])
ls = ls[:ending_index]
dfs = fluctuations[:ending_index]

print(ls.shape)
print(dfs.shape)

full_ls = np.hstack((small_r_ls,ls))
full_dfs = np.hstack((small_r_dfs,dfs))


# Define fitting regions do fit

sboundary_l = 40
sboundary_index = np.max((small_r_ls <= sboundary_l).nonzero()[0])

fit_radii = small_r_ls[:sboundary_index]
fit_dfs = small_r_dfs[:sboundary_index]

pfrs=small_r_ls[:sboundary_index + 300]

lboundary_l = 80
lboundary_index = np.max((full_ls <= lboundary_l).nonzero()[0])

lboundary_l2 = 400
lboundary_index2 = np.max((full_ls <= lboundary_l2).nonzero()[0])

fit_radii2 = full_ls[lboundary_index:lboundary_index2]
fit_dfs2 = full_dfs[lboundary_index:lboundary_index2]

pfrs2 = full_ls[lboundary_index-300:lboundary_index2]


a, b, rval, *_ = linregress(np.log(fit_radii),np.log(fit_dfs))
print('For l <= %f:\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        '%(small_r_ls[99], a, b, rval))

a2, b2, rval2, *_ = linregress(np.log(fit_radii2),np.log(fit_dfs2))
print('For l > %f:\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        '%(lboundary_l, a2, b2, rval2))



fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xscale('log')
ax.set_yscale('log')

ax.plot(cls*0.08,cdfs,'o',c='#07f09d',ms=1,label='graphene')
ax.plot(full_ls*0.2,full_dfs,'o',c='#f0075a',ms=1,label='MAC')
ax.plot(full_ls*0.2,cdfs[0]*(full_ls)**(-3),'k-.',lw=0.7,label='$\ell^{-3}$ (Crystal)')
ax.plot(full_ls*0.2,(small_r_dfs[1])*full_ls**(-2),'k--',lw=0.7,label='$\ell^{-2}$ (Random)')
ax.plot(pfrs*0.2, np.exp(b)*(pfrs**a),'-', c='#075af0',lw=1.8)# label='Fit: $\ell^{%3.2f}$'%a)
ax.plot(pfrs2*0.2, np.exp(b2+0.1)*(pfrs2**a2),'-', c='#51d706',lw=1.8)#, label='Fit: $\ell^{%3.2f}$'%a2)

plt.suptitle('Density fluctuations in MAC (40nm $\\times$ 40nm)')
plt.legend()
ax.set_xlabel(r'$\ell$ [\AA]')
ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')
textstr='$\ell^{%3.2f}$'%a2
#ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=16,
                #verticalalignment='top', color='#51d706')
#ax.set_xlim(1,100)
plt.legend()
plt.show()

