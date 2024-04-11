#!/usr/bin/env pythonw

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt



# *** UNCOMMENT FOR MANNA ***
#fluctuations = np.load('avg_density_fluctuations_manna_1d.npy')
#radii = np.arange(1,30000)
fluctuations = np.load('avg_density_fluctuations_manna_1d.npy')
radii = np.arange(1,30000)
densities = np.load('rhos_manna_1d.npy')

#fit most critical dataset to a line in log-log space
a, b, rval, *_ = linregress(np.log(radii[:1000]),np.log(fluctuations[-1,:1000]))
print('Fit found.\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        '%(a, b, rval))

fig = plt.figure()

ax = fig.add_subplot(111)
plt.rc('text',usetex = True)
plt.rc('font', size=20 )
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')
#ax.set_xlim(radii[0],radii[-1]/2.0)

ax.plot(radii,fluctuations[2],'o',ms=0.5,c='#00cb3e',fillstyle='full')
ax.plot(radii,1.0/radii,'k--',lw=0.8,label=r'$\ell^{-1}$')
ax.plot(radii,np.exp(b)*(radii**a),'k-',lw=0.8,label='$\ell^{%2.4f}$'%a)
#ax.plot(radii, np.exp(b)*np.power(radii,a),'k-',lw=0.8,label=r'$\ell^{%f}$'%a)
plt.xlim([0.5,20000])
plt.legend()
plt.show()

