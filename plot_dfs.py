#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex
from density_fluctuations import fit_dfs


datafile = '/Users/nico/Desktop/simulation_outputs/hyperuniformity/ata_structures/avg_dfs_radii_tempdot6_relaxed_263structures.npy'
# datafile = '/Users/nico/Desktop/simulation_outputs/hyperuniformity/avg_dfs_radii_pCNN_relaxed.npy'


dfs = np.load(datafile)[:,1]
r = np.load(datafile)[:,0]

l1 = 3
l2 = 20


a, b, r2 = fit_dfs(r, dfs,lbounds=[l1,l2])
print('Fit found.\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        '%(a, b, r2))



fig = plt.figure()

ax = fig.add_subplot(111)


ax.set_xscale('log')
ax.set_yscale('log')


ax.plot(r,dfs,'ro',ms=1,alpha=0.7)
ax.plot(r, np.exp(b)*np.power(r,a),'k--',lw=1.0,label=f'$\ell^{{-{a}}}$')
ax.axvline(x=l1,ymin=0,ymax=1,c='k',ls='-',lw=0.8)
ax.axvline(x=l2,ymin=0,ymax=1,c='k',ls='-',lw=0.8)
ax.set_xlabel('$\ell$')
ax.set_ylabel('$\sigma_{\\rho}^2(\ell)$')

plt.legend()
plt.show()
