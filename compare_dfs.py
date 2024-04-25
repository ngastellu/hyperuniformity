#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex
from density_fluctuations import fit_dfs


datadir = '/Users/nico/Desktop/simulation_outputs/hyperuniformity/'
datafile_1 = datadir + 'ata_structures/avg_dfs_radii_tempdot6_relaxed_263structures.npy'
datafile_2 = datadir + 'avg_dfs_radii_pCNN_relaxed.npy'

# r = np.linspace(1,700,700)

dat1 = np.load(datafile_1)
dat2 = np.load(datafile_2)

r1 = dat1[:,0]
dfs_1 = dat1[:,1]

r2 = dat2[:,0]
dfs_2 = dat2[:,1]

print(r1)
print(r2)

a_1, b_1, r21 = fit_dfs(r1, dfs_1,lbounds=[5,50])
print('Fit 1 found. (%s)\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        '%(datafile_1.split('/')[-1],a_1, b_1, r21))


a_2, b_2, r22 = fit_dfs(r2, dfs_2,lbounds=[5,50])
print('Fit 2 found. (%s)\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        '%(datafile_2.split('/')[-1],a_2, b_2, r22))



fig = plt.figure()

ax = fig.add_subplot(111)


ax.set_xscale('log')
ax.set_yscale('log')


ax.plot(r1,dfs_1,'ro',ms=1,alpha=0.7,label="Ata's conditional model ($T = 0.6$)")
ax.plot(r2,dfs_2,'bo',ms=1,alpha=0.7,label="Michael's model")
ax.plot(r1, np.exp(b_1)*np.power(r1,a_1),'r--',lw=1.0,label=f'$\ell^{{-{a_1}}}$')
ax.plot(r2, np.exp(b_2)*np.power(r2,a_2),'b--',lw=1.0,label=f'$\ell^{{-{a_2}}}$')
ax.set_xlabel('$\ell$')
ax.set_ylabel('$\sigma_{\\rho}^2(\ell)$')

plt.legend()
plt.show()
