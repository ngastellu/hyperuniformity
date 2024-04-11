#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import linregress

fluctuations_random = np.load('dfs_random.npy')
fluctuations_crystal = np.load('dfs_crystal.npy')
ls = np.load('radii.npy')



a_c, b_c, rval, *_ = linregress(np.log(ls[:479]),np.log(fluctuations_crystal[:479]))
print('Crystal fit found.\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        '%(a_c, b_c, rval))

a_r, b_r, rval, *_ = linregress(np.log(ls[:479]),np.log(fluctuations_random[:479]))
print('Random fit found.\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        '%(a_r, b_r, rval))


rcParams['text.usetex'] = True
rcParams['font.size'] = 15
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

fig = plt.figure()

ax = fig.add_subplot(111)

#plt.rc('text',usetex=True)    
#plt.rc('font',size=20)

ax.set_xscale('log')
ax.set_yscale('log')


ax.plot(ls,fluctuations_random,'ro',ms=1,label='Random')
ax.plot(ls,fluctuations_crystal,'bo',ms=1,label='Crystal')
ax.plot(ls, np.exp(b_c)*np.power(ls,-3),'k--',lw=1.0,label=r'$\ell^{-3}$')
ax.plot(ls, np.exp(b_r)*np.power(ls,-2),'k-.',lw=1.0,label=r'$\ell^{-2}$')
ax.set_xlim([0,1000])
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')

plt.legend()
plt.show()
