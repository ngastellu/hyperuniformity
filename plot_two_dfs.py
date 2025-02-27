#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex
from density_fluctuations import fit_fluctuations

datafile1 = f'/Users/nico/Desktop/simulation_outputs/hyperuniformity/poisson_tests/avg_nfs_radii_random_grid_L_1500.npy'
datafile2 = f'/Users/nico/Desktop/simulation_outputs/hyperuniformity/poisson_tests/avg_nfs_radii_random_grid_L_1500rho_mult_0.50.npy'

label1 = '$\\rho = \\rho_{\\text{MAC}}$'
label2 = '$\\rho = \\rho_{\\text{MAC}}/2$'


data1 = np.load(datafile1)

if data1.shape[0] == 2: # 1 data point (r, fluctuation) <----> 1 column
        fluctuations1 = data1[1,:]
        r1 = data1[0,:]
elif data1.shape[1] == 2: # 1 data point (r, fluctuation) <----> 1 row
        fluctuations1 = data1[:,1]
        r1 = data1[:,0]

data2 = np.load(datafile2)
if data2.shape[0] == 2: # 2 data point (r, fluctuation) <----> 2 column
        fluctuations2 = data2[1,:]
        r2 = data2[0,:]
elif data2.shape[1] == 2: # 2 data point (r, fluctuation) <----> 2 row
        fluctuations2 = data2[:,1]
        r2 = data2[:,0]

setup_tex()

fig = plt.figure()

ax = fig.add_subplot(111)


ax.set_xscale('log')
ax.set_yscale('log')


ax.plot(r1,fluctuations1/(np.pi*r1**2),'ro',ms=3,alpha=0.7,label=label1)
ax.plot(r2,fluctuations2/(np.pi*r2**2),'bo',ms=3,alpha=0.7,label=label2)
# ax.plot(r, np.exp(b)*np.power(r,a),'k--',lw=1.0,label=f'$\ell^{{-{a}}}$')
# ax.axvline(x=l1,ymin=0,ymax=1,c='k',ls='-',lw=0.8)
# ax.axvline(x=l2,ymin=0,ymax=1,c='k',ls='-',lw=0.8)
ax.set_xlabel('$\ell$')
ax.set_ylabel('$\sigma_{N}(\ell)^2/\pi\ell^2$')
# ax.set_title(f'MCMC Ising with $L={L}$ at $T={T}$')

plt.legend()
plt.show()
