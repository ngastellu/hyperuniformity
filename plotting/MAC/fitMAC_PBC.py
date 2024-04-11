#!/usr/bin/env pythonw

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from os import path
from glob import glob

def moving_average(x, w):
    """Compute the moving average of array x with a sampling window of width w.
    Copied and pasted from: 
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy"""
    
    return np.convolve(x, np.ones(w), 'valid') / w


pixels2angstroms = 0.2 # 5 pixels/angstrom
datadir = '/Users/nico/Desktop/McGill/Research/simulation_outputs/hyperuniformity/bigMAC'


rcParams['text.usetex'] = True
rcParams['font.size'] = 25
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


rfile = path.join(datadir,'radii-78.npy')
flucfile = path.join(datadir,'avg_dfs_bigMAC_PBC.npy')

fig = plt.figure()
ax = fig.add_subplot(111)

ls = np.load(rfile)
#ls = np.linspace(1,1000,7000)
dfs = np.load(flucfile)

boundary_l = 800

boundary_index = np.max((ls <= boundary_l).nonzero()[0])


ax.set_xscale('log')
ax.set_yscale('log')


#ax.errorbar(ls[:boundary_index],dfs[:boundary_index],yerr=error[:boundary_index],ms=1,marker='o',c='r',errorevery=20)
ax.plot(ls[:boundary_index],dfs[:boundary_index],'ro', ms=1)
#ax.plot(small_ls*0.2, small_ls_dfs, 'bo', ms=1)
#ax.plot(smooth_ls,smooth_fluctuations,'k-',lw=0.8)


ax.plot(ls,dfs[0]*ls**(-2),'k--',lw=0.7,label='$\ell^{-2}$')
#ax.plot(ls,np.log(ls)/ls,'k--',lw=0.7,label='$\\text{ln}(\ell)/\ell$')
#ax.plot(ls*0.2,dfs[0]*ls**(-3),'k-.',lw=0.7,label='$\ell^{-(d+1)}$')
plt.suptitle('Density fluctuations in MAC (40nm $\\times$ 40nm)')
plt.legend()
#ax.set_xlim(0.2,110)
#ax.set_xlabel(r'$\ell$ [pixels]')
ax.set_xlabel(r'$\ell$ [\AA]')
ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')

plt.show()

# Define fitting regions do fit
boundary_l1 = 10
boundary_l2 = 200# boundary value of l separating the two scaling regions

boundary_index1 = np.max((ls <= boundary_l1).nonzero()[0])
boundary_index2 = np.max((ls <= boundary_l2).nonzero()[0])

#fit_radii = small_r_ls[:100]
#fit_dfs = small_r_dfs[:100]

fit_radii2 = ls[boundary_index1:boundary_index2]
fit_dfs2 = dfs[boundary_index1:boundary_index2]

#a, b, rval, *_ = linregress(np.log(fit_radii),np.log(fit_dfs))
#print('For l <= %f:\n\
#        Slope = %f\n\
#        Intercept = %f\n\
#        rval = %f\n\
#        '%(small_r_ls[99], a, b, rval))

a2, b2, rval2, *_ = linregress(np.log(fit_radii2),np.log(fit_dfs2))
print('For l > %f:\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        '%(boundary_l, a2, b2, rval2))


plt_radii = np.logspace(-1, 2,1000)
print(plt_radii.shape)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xscale('log')
ax.set_yscale('log')

#ax.plot(small_r_ls,small_r_dfs,'o',c='r',ms=1)
ax.plot(ls,dfs,'o',c='r',ms=1)
#ax.plot(fit_radii2,dfs[0]*fit_radii2**(-2),'k--',lw=0.7,label='$\ell^{-2}$')
#ax.plot(fit_radii, np.exp(b)*(fit_radii**a), 'b-', label='Fit: $\lambda = %f$'%a,lw=2.0)
ax.plot(plt_radii, np.exp(b2)*(plt_radii**a2), 'b-', label='Fit: $\sigma_{\\rho}^2(\ell)\sim\ell^{%3.2f}$'%a2,lw=1.0)

plt.suptitle('Density fluctuations in MAC (40nm $\\times$ 40nm)')
plt.legend()
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')
ax.set_xlim(0.5,400)
plt.legend()
plt.show()
