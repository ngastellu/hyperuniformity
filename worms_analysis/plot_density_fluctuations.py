#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

fluctuations = np.load('dfs_worms_avg.npy')
ls = np.load('radii.npy')

slope, intercept, rval, pval, stderr = linregress(np.log(ls[:124]),np.log(fluctuations[:124]))

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
ax.plot(ls,ls**(-2),'k--',lw=0.8,label='$\ell^{-2}$')
plt.plot(ls, (ls**slope)*np.exp(intercept),'k-',lw=0.8,label='$\ell^{%2.4f}$'%slope)
#ax.plot(ls,(ls[0]/ls)**(2.45)+fluctuations[0]-1,'k--',lw=0.8)
ax.set_xlabel('$\ell$')
ax.set_ylabel('$\sigma_{\\rho}^2(\ell)$')

plt.legend()
plt.show()
