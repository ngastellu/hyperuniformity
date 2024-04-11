#!/usr/bin/env pythonw

from os import path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress


datadir = '../../../simulation_outputs/hyperuniformity'
flucfile = path.join(datadir,'avg_dfs_MAC_16x16.npy')
radfile = path.join(datadir, 'radii_16x16_MAC.npy')

fluctuations = np.load(flucfile)[:-2]
ls = np.load(radfile)[:-2]

boundary_l = 100
boundary_index = np.max((ls <= boundary_l).nonzero()[0])

slope, intercept, rval, pval, stderr = linregress(np.log(ls[:boundary_index]),np.log(fluctuations[:boundary_index]))

print('Fit found.\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        pval = %f\n\
        stderr = %f\n\
        '%(slope, intercept, rval, pval,stderr))

fig = plt.figure()

ax = fig.add_subplot(111)

plt.rc('text',usetex=True)    

ax.set_xscale('log')
ax.set_yscale('log')


ax.plot(ls,fluctuations,'ro',ms=1)
#ax.plot(ls,flucs2,'bo',ms=1)
ax.plot(ls,fluctuations[0]*ls**(-2),'k--',lw=0.8,label=r'$\ell^{-2}$')
plt.plot(ls, (ls**slope)*np.exp(intercept),'k-',lw=0.8,label='$\ell^{%2.4f}$'%slope)
#ax.plot(ls,(ls[0]/ls)**(2.45)+fluctuations[0]-1,'k--',lw=0.8)
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')
ax.set_xlim(ls[0],110)

plt.legend()
plt.show()
