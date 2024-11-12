#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours
from density_fluctuations import fit_dfs


datadir = '/Users/nico/Desktop/simulation_outputs/hyperuniformity/bigMAC/'
datafile_1 = datadir + 'avg_dfs_radii_40x40_pbc.npy'
datafile_2 = datadir + 'avg_dfs_radii_tempdot6_pbc.npy'
datafile_3 = datadir + 'avg_dfs_radii_tempdot5_pbc.npy'

clrs = MAC_ensemble_colours()
lbls = ['sAMC-500', 'sAMC-q400', 'sAMC-300']

datafiles = [datafile_1, datafile_2, datafile_3]

slopes = np.zeros(len(datafiles))
intercepts = np.zeros(len(datafiles))
fit_r2 = np.zeros(len(datafiles))

# datafile_1 = datadir + 'ata_structures/avg_dfs_radii_tempdot6_relaxed_263structures.npy'
# datafile_2 = datadir + 'avg_dfs_radii_pCNN_relaxed.npy'

setup_tex(fontsize=20)
fig = plt.figure()

ax = fig.add_subplot(111)

fit_l1 = 20
fit_l2 = 60


ax.set_xscale('log')
ax.set_yscale('log')

# r = np.linspace(1,700,700)
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
    
    
    ax.plot(r,dfs,'o',c=c,ms=4,alpha=0.7,label=lbl)
    ax.plot(r, np.exp(b)*np.power(r,a),'--',lw=2.0,c=c,label=f'$\ell^{{{a:.3f}}}$')


ax.axvline(x=fit_l1,ymin=0,ymax=1,c='k',ls='--',lw=0.8)
ax.axvline(x=fit_l2,ymin=0,ymax=1,c='k',ls='--',lw=0.8)

ax.set_xlim([0,100])

ax.set_xlabel('$\ell$')
ax.set_ylabel('$\sigma_{\\rho}^2(\ell)$')
ax.set_title('Density fluctuation scaling for equilibrated sAMC-300 and sAMC-q400')

plt.legend()
plt.show()

