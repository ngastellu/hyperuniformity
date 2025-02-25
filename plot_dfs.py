#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex
from density_fluctuations import fit_fluctuations

T = 3.1
L = 50
datafile = f'/Users/nico/Desktop/simulation_outputs/hyperuniformity/ising/MCMC-generated/nfs_radii_T_{T}_L_{L}.npy'

print(f'Reading from file: ', datafile.split('/')[-1])

data = np.load(datafile)

if data.shape[0] == 2: # 1 data point (r, fluctuation) <----> 1 column
        fluctuations = data[1,:]
        r = data[0,:]
elif data.shape[1] == 2: # 1 data point (r, fluctuation) <----> 1 row
        fluctuations = data[:,1]
        r = data[:,0]


print()

l1 = 1
l2 = 20


a, b, r2 = fit_fluctuations(r, fluctuations/r**2,lbounds=[l1,l2])
print('Fit found.\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        '%(a, b, r2))


setup_tex()

fig = plt.figure()

ax = fig.add_subplot(111)


ax.set_xscale('log')
ax.set_yscale('log')


ax.plot(r,fluctuations/r**2,'ro',ms=1,alpha=0.7)
ax.plot(r, np.exp(b)*np.power(r,a),'k--',lw=1.0,label=f'$\ell^{{-{a}}}$')
ax.axvline(x=l1,ymin=0,ymax=1,c='k',ls='-',lw=0.8)
ax.axvline(x=l2,ymin=0,ymax=1,c='k',ls='-',lw=0.8)
ax.set_xlabel('$\ell$')
ax.set_ylabel('$\sigma_{\\rho}^2(\ell)$')
ax.set_title(f'MCMC Ising with $L={L}$ at $T={T}$')

# plt.legend()
plt.show()
