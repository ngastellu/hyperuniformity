#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import linregress
from density_fluctuations import fit_dfs
from qcnico.plt_utils import setup_tex

ddir = '/Users/nico/Desktop/simulation_outputs/hyperuniformity/crystal_v_random/'

datafiles = [ddir + f for f in ['avg_dfs_radii_random_pbc.npy', 'graphene_dfs.npy']]


fit_l1 = 20
fit_l2 = 60

clrs = ['#d62d20', '#0057e7']
lbls = ['Random', 'Crystal']

setup_tex(fontsize=50)
fig = plt.figure()

ax = fig.add_subplot(111)

fit_l1 = 20
fit_l2 = 60


ax.set_xscale('log')
ax.set_yscale('log')

for datafile, c, lbl in zip(datafiles,clrs,lbls):
    dat = np.load(datafile)
    r = dat[:,0]
    dfs = dat[:,1]
    a, b, r2 = fit_dfs(r, dfs,lbounds=[fit_l1,fit_l2])
    print('Fit for %s found. (%s)\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\n\
        '%(lbl,datafile.split('/')[-1],a, b, r2))
    
    
    ax.plot(r,dfs*(r**2),'o',c=c,ms=4,alpha=0.7,label=lbl)
    # ax.plot(r, np.exp(b)*np.power(r,a),'--',lw=2.0,c=c,label=f'$\ell^{{{a:.3f}}}$')
    ax.plot(r, np.exp(b)*np.power(r,a+2),'--',lw=2.0,c=c,label=f'$\ell^{{{a+2:.3f}}}$')

ax.axvline(x=fit_l1,ymin=0,ymax=1,c='k',ls='--',lw=0.8)
ax.axvline(x=fit_l2,ymin=0,ymax=1,c='k',ls='--',lw=0.8)

ax.set_xlim([0,70])

ax.set_xlabel('$\ell$')
# ax.set_ylabel('$\sigma_{\\rho}^2(\ell)$')
ax.set_ylabel('$\sigma_{N}^2(\ell)/\ell^2$')

plt.legend()
plt.show()
