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
colours = ['#67cd00','#6600cd','#ff005e','#005eff','#ffa100','#c00000']

# *** UNCOMMENT FOR RAND ORG ***
#fluctuations = np.load('avg_density_fluctuations_rand_org_2d.npy')
#radii = np.linspace(1,300,300)
#densities = [0.8,0.83,0.85,0.86,0.865,0.868,0.869]
#colours = ['#67cd00','#6600cd','#ff005e','#005eff','#ffa100','#ff00ff','#c00000']

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
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')
#ax.set_xlim(radii[0],radii[-1]/2.0)

for k,rho in enumerate(densities):
    ax.plot(radii,fluctuations[k],'o',ms=0.5,label=r'$\rho={}$'.format(rho),c=colours[k],fillstyle='full')
ax.plot(radii,1.0/radii,'k--',lw=0.8,label=r'$\ell^{-1}$')
ax.plot(radii,np.exp(b)*(radii**a),'k-',lw=0.8,label='$\ell^{%2.4f}$'%a)
#ax.plot(radii, np.exp(b)*np.power(radii,a),'k-',lw=0.8,label=r'$\ell^{%f}$'%a)
plt.xlim([0.5,20000])
plt.legend()
plt.show()

