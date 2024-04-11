#!/usr/bin/env pythonw

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt



fluctuations = np.load('avg_density_fluctuations_manna_2d.npy')
radii = np.linspace(1,750,800)
densities = np.load('rhos_manna_2d.npy')
colours = ['#67cd00','#6600cd','#ff005e','#005eff','#ffa100','#c00000']


#fit most critical dataset to a line in log-log space
#a, b, rval, *_ = linregress(np.log(radii),np.log(fluctuations[-1]))
#print('Fit found.\n\
#        Slope = %f\n\
#        Intercept = %f\n\
#        rval = %f\n\
#        '%(a, b, rval))

fig = plt.figure()

ax = fig.add_subplot(111)
plt.rc('text',usetex = True)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\sigma_{\ell}^2(\rho)$')
#ax.set_xlim(radii[0],radii[-1]/2.0)

for k,rho in enumerate(densities):
    ax.plot(radii,fluctuations[k],'o',ms=0.5,label=r'$\rho={}$'.format(rho),c=colours[k],fillstyle='full')
ax.plot(radii,1.0/(radii**2),'k--',lw=0.8,label=r'$\ell^{-1}$')
#ax.plot(radii, np.exp(b)*np.power(radii,a),'k-',lw=0.8,label=r'$\ell^{%f}$'%a)
#plt.xlim([0.5,25000])
plt.legend()
plt.show()

