#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex , get_cm
from density_fluctuations import fit_fluctuations


datadir = '/Users/nico/Desktop/simulation_outputs/hyperuniformity/ising/MAP_generated/last_100x100_subsample/'
temps_str = ['6', '65', '7', '8', '9']
temps = [float(f'0.{T}') for T in temps_str]

datafiles = [datadir + f'avg_nfs_tdot{T}.npy' for T in temps_str]
clrs = get_cm(temps,'inferno',min_val=0.1,max_val=0.8)
lbls = [f'$\\tilde{{T}} = {T}$' for T in temps]


slopes = np.zeros(len(datafiles))
intercepts = np.zeros(len(datafiles))
fit_r2 = np.zeros(len(datafiles))

setup_tex(fontsize=30)
fig = plt.figure()

ax = fig.add_subplot(111)

fit_l1 = 1
fit_l2 = 17


ax.set_xscale('log')
ax.set_yscale('log')

alphas = np.zeros(len(datafiles))

r = np.load(datadir + 'radii.npy')

for k, datafile, c, lbl in zip(range(len(datafiles)),datafiles,clrs,lbls):
    nfs = np.load(datafile)
    a, b, r2 = fit_fluctuations(r, nfs/(np.pi*r**2),lbounds=[fit_l1,fit_l2])
    alphas[k] = a
    print('Fit for %s found. (%s)\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\n\
        '%(lbl,datafile.split('/')[-1],a, b, r2))
    
    
    ax.plot(r,nfs/(np.pi*r**2),'o',c=c,ms=5,alpha=0.7,label=lbl)
    ax.plot(r, np.exp(b)*np.power(r,a),'--',lw=2.0,c=c)#,label=f'$\ell^{{{a+2:.3f}}}$')

ax.axvline(x=fit_l1,ymin=0,ymax=1,c='k',ls='--',lw=0.8)
ax.axvline(x=fit_l2,ymin=0,ymax=1,c='k',ls='--',lw=0.8)

ax.set_xlim([1,19])

ax.set_xlabel('$\ell$ [\AA]')
# ax.set_ylabel('$\sigma_{\\rho}^2(\ell)$')
ax.set_ylabel('$\sigma_{N}^2(\ell)/\ell^2$ [\AA$^{-2}$]')

plt.legend(ncol=2)
plt.show()

# temps = np.linspace(0.5,1.1,6)

fig, ax = plt.subplots()
for t, a, c in zip(temps,alphas,clrs):
    ax.plot(t,a,c=c,marker='^',markersize=20,zorder=10)

ax.plot(temps,alphas,color='gray',ls='--',lw=1.0,zorder=1)

ax.axhline(y=0,xmin=0,xmax=1,c='k',ls='--',lw=3.0)

ax.set_xlabel('$\\tilde{T}$')
ax.set_ylabel('Fluctuations exponent $\\alpha$')
plt.show()

