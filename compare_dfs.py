#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex
from density_fluctuations import fit_dfs


datadir = '/Users/nico/Desktop/simulation_outputs/hyperuniformity/'

r_pCNN, dfs_pCNN = np.load(datadir + 'dfs_radii_bigMAC_unsymm_RS.npy').T
r_tdot6, dfs_tdot6 = np.load(datadir + 'ata_structures/avg_dfs_tempdot6.npy')



a_pCNN, b_pCNN, r2 = fit_dfs(r_pCNN, dfs_pCNN,lbounds=[3,50])
print('PixelCNN fit found.\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        '%(a_pCNN, b_pCNN, r2))


a_tdot6, b_tdot6, r2 = fit_dfs(r_tdot6, dfs_tdot6,lbounds=[3,50])
print('PixelCNN fit found.\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        '%(a_tdot6, b_tdot6, r2))



fig = plt.figure()

ax = fig.add_subplot(111)


ax.set_xscale('log')
ax.set_yscale('log')


ax.plot(r_pCNN,dfs_pCNN,'ro',ms=1,alpha=0.7,label="Michael's model")
ax.plot(r_tdot6,dfs_tdot6,'bo',ms=1,alpha=0.7,label="Ata's conditional model ($T = 0.6$)")
ax.plot(r_pCNN, np.exp(b_pCNN)*np.power(r_pCNN,a_pCNN),'r--',lw=1.0,label=f'$\ell^{{-{a_pCNN}}}$')
ax.plot(r_tdot6, np.exp(b_tdot6)*np.power(r_tdot6,a_tdot6),'b--',lw=1.0,label=f'$\ell^{{-{a_tdot6}}}$')
ax.set_xlabel('$\ell$')
ax.set_ylabel('$\sigma_{\\rho}^2(\ell)$')

plt.legend()
plt.show()
