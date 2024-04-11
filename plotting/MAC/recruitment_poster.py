#!/usr/bin/env pythonw

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from os import path
from glob import glob


datadir = '/Volumes/D-Rive2/Research/simulation_outputs/hyperuniformity/'
flucfile = path.join(datadir,'avg_dfs_bigMAC.npy')

rcParams['text.usetex'] = True
rcParams['font.size'] = 25
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

angstroms_per_pixel = 0.2

dfs = np.load(flucfile)
ls = np.linspace(1,1000,7000)*angstroms_per_pixel

crystal_rfile = path.join(datadir, 'crystal2d_radii.npy')
crystal_flucfile = path.join(datadir, 'dfs_crystal2d.npy')

cls = np.load(crystal_rfile)[2:]
cdfs = np.load(crystal_flucfile)[2:]

boundary_l1 = 10
boundary_l2 = 85# boundary value of l separating the two scaling regions

boundary_index1 = np.max((ls <= boundary_l1).nonzero()[0])
boundary_index2 = np.max((ls <= boundary_l2).nonzero()[0])

fit_radii2 = ls[boundary_index1:boundary_index2]
fit_dfs2 = dfs[boundary_index1:boundary_index2]

a2, b2, rval2, *_ = linregress(np.log(fit_radii2),np.log(fit_dfs2))
print('For l > %f:\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        '%(boundary_l2, a2, b2, rval2))


fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xscale('log')
ax.set_yscale('log')

plt_radii = np.logspace(-1, 2,1000)

ax.plot(cls*0.08,cdfs,'o',c='#07f09d',ms=2,label='Graphene')
ax.plot(ls,dfs,'o',c='#f0075a',ms=2,label='MAC')
ax.plot(cls*0.08,cdfs[9]*(cls*0.08)**(-3),'k-.',lw=0.7,label='$\ell^{-3}$ (Crystal)')
ax.plot(ls,(dfs[1])*ls**(-2),'k--',lw=0.7,label='$\ell^{-2}$ (Random)')
#ax.plot(plt_radii, np.exp(b2)*(plt_radii**a2), 'b-', label='Fit: $\ell^{%3.2f}$'%a2,lw=1.0)
ax.plot(plt_radii, np.exp(b2)*(plt_radii**a2), 'b-', lw=1.0)
#ax.plot(full_ls*0.2,full_dfs,'o',c='#f0075a',ms=1,label='MAC')
#ax.plot(full_ls*0.2,cdfs[0]*(full_ls)**(-3),'k-.',lw=0.7,label='$\ell^{-3}$ (Crystal)')
#ax.plot(full_ls*0.2,(small_r_dfs[1])*full_ls**(-2),'k--',lw=0.7,label='$\ell^{-2}$ (Random)')
#ax.plot(plt_radii, np.exp(b2)*(plt_radii**a2), 'b-', label='Fit: $\sigma_{\\rho}^2(\ell)\sim\ell^{%3.2f}$'%a2,lw=1.0)

plt.legend()
ax.set_xlabel(r'$\ell$ [\AA]')
ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')
textstr='$\ell^{%3.2f}$'%a2
#ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=16,
                #verticalalignment='top', color='#51d706')
#ax.set_xlim(1,100)
plt.legend()
plt.show()

