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

#datadir = '/Users/nico/Desktop/McGill/Research/simulation_outputs/hyperuniformity/bigMAC/'
datadir = '/Volumes/D-Rive2/Research/simulation_outputs/hyperuniformity/bigMAC/'

#rfiles = glob(path.join(datadir,'radii_*.npy'))
#sys_names = [('_').join(path.basename(rf).split('_')[1:]).split('.')[0] for rf in rfiles]

rcParams['text.usetex'] = True
rcParams['font.size'] = 14
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


#rfile = path.join(datadir,'avg_dfs_40x40_MAC.npy')
flucfile = path.join(datadir,'avg_dfs_bigMAC.npy')
small_l_flucfile = path.join(datadir,'small-r_avg_dfs_bigMAC.npy')

fig = plt.figure()
ax = fig.add_subplot(111)

#ls = np.load(rfile)
ls = np.linspace(1,1000,7000)
dfs = np.load(flucfile)
error = np.load(path.join(datadir,'std_dfs_bigMAC.npy'))

small_ls = np.linspace(1,100,700)
small_ls_dfs = np.load(small_l_flucfile)

print(ls.shape)
print(dfs.shape)

boundary_l = 100*5

boundary_index = np.max((ls <= boundary_l).nonzero()[0])


ax.set_xscale('log')
ax.set_yscale('log')


#ax.errorbar(ls[:boundary_index],dfs[:boundary_index],yerr=error[:boundary_index],ms=1,marker='o',c='r',errorevery=20)
ax.plot(ls[:boundary_index]*0.2,dfs[:boundary_index],'ro', ms=1)
#ax.plot(small_ls*0.2, small_ls_dfs, 'bo', ms=1)
#ax.plot(smooth_ls,smooth_fluctuations,'k-',lw=0.8)


ax.plot(ls*0.2,dfs[0]*ls**(-2),'k--',lw=0.7,label='$\ell^{-2}$')
#ax.plot(ls*0.2,dfs[0]*ls**(-3),'k-.',lw=0.7,label='$\ell^{-(d+1)}$')
plt.suptitle('Density fluctuations in MAC (40nm $\\times$ 40nm)')
plt.legend()
ax.set_xlim(0.2,110)
#ax.set_xlabel(r'$\ell$ [pixels]')
ax.set_xlabel(r'$\ell$ [\AA]')
ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')

plt.show()



# Define fitting regions do fit

#boundary_l = 400# boundary value of l separating the two scaling regions
#
#boundary_index = np.max((ls <= boundary_l).nonzero()[0])
#
#fit_radii = small_r_ls[:100]
#fit_dfs = small_r_dfs[:100]
#
#fit_radii2 = ls[:boundary_index]
#fit_dfs2 = dfs[:boundary_index]
#
#a, b, rval, *_ = linregress(np.log(fit_radii),np.log(fit_dfs))
#print('For l <= %f:\n\
#        Slope = %f\n\
#        Intercept = %f\n\
#        rval = %f\n\
#        '%(small_r_ls[99], a, b, rval))
#
#a2, b2, rval2, *_ = linregress(np.log(fit_radii2),np.log(fit_dfs2))
#print('For l > %f:\n\
#        Slope = %f\n\
#        Intercept = %f\n\
#        rval = %f\n\
#        '%(boundary_l, a2, b2, rval2))
#
#
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#
#ax.set_xscale('log')
#ax.set_yscale('log')
#
#ax.plot(small_r_ls,small_r_dfs,'o',c='r',ms=1)
#ax.plot(ls,dfs,'o',c='r',ms=1)
#ax.plot(full_ls,small_r_dfs[1]*full_ls**(-2),'k--',lw=0.7,label='$\ell^{-d}$')
#ax.plot(fit_radii, np.exp(b)*(fit_radii**a), 'b-', label='Fit: $\lambda = %f$'%a,lw=2.0)
#ax.plot(fit_radii2, np.exp(b2)*(fit_radii2**a2), 'g-', label='Fit: $\lambda = %f$'%a2,lw=2.0)
#
#plt.suptitle('Density fluctuations in MAC (40nm $\\times$ 40nm)')
#plt.legend()
#ax.set_xlabel(r'$\ell$')
#ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')
##ax.set_xlim(1,100)
#plt.legend()
#plt.show()


all_ls = np.hstack((small_ls,ls))
all_dfs = np.hstack((small_ls_dfs,dfs))

a3, b3, rval3, *_ = linregress(np.log(all_ls),np.log(all_dfs))
print('For all ls:\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        '%(a3, b3, rval3))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xscale('log')
ax.set_yscale('log')

ax.plot(all_ls,all_dfs,'o',c='r',ms=1)
ax.plot(all_ls,all_dfs,'o',c='r',ms=1)
ax.plot(all_ls,all_dfs[1]*all_ls**(-2),'k--',lw=0.7,label='$\ell^{-2}$ (Random)')
ax.plot(all_ls, np.exp(b3)*(all_ls**a3), 'b-', label='Fit: $\lambda = %f$'%a3,lw=1.0)

plt.suptitle('Density fluctuations in MAC (40nm $\\times$ 40nm)')
plt.legend()
ax.set_xlabel(r'$\ell$')
ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')
#ax.set_xlim(1,100)
plt.legend()
plt.show()

boundary_index = np.max((all_ls <= boundary_l).nonzero()[0])

fit_ls = all_ls[:boundary_index]
fit_dfs = all_dfs[:boundary_index]

a, b, rval, *_ = linregress(np.log(fit_ls),np.log(fit_dfs))
print('For l <= %f angstroms:\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        '%(boundary_l*0.2, a, b, rval))


fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xscale('log')
ax.set_yscale('log')

ax.plot(all_ls*0.2,all_dfs,'o',c='r',ms=1)
ax.plot(all_ls*0.2,all_dfs[1]*all_ls**(-2),'k--',lw=0.7,label='$\ell^{-2}$ (Random)')
ax.plot(fit_ls*0.2, np.exp(b)*(fit_ls**a), 'b-', label='Fit: $\lambda = %f$'%a,lw=1.0)

plt.suptitle('Density fluctuations in MAC (40nm $\\times$ 40nm)')
plt.legend()
ax.set_xlabel(r'$\ell$ [\AA]')
ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')
ax.set_xlim(0.2,boundary_l*0.2)
#ax.set_xlim(1,100)
plt.legend()
plt.show()
