#!/usr/bin/env python


import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex, MAC_ensemble_colours, get_cm
from density_fluctuations import fit_dfs


datadir = '/Users/nico/Desktop/simulation_outputs/hyperuniformity/bigMAC/'
datafiles = [
# datadir + 'avg_dfs_radii_40x40_pbc.npy',
datadir + 'avg_dfs_radii_tempdot5_pbc.npy',
datadir + 'avg_dfs_radii_tempdot6_pbc.npy',
datadir + 'avg_dfs_radii_tempdot7_pbc.npy',
datadir + 'avg_dfs_radii_tempdot8_pbc.npy',
datadir + 'avg_dfs_radii_tempdot9_pbc.npy',
datadir + 'avg_dfs_radii_t1_pbc.npy']

# clrs = MAC_ensemble_colours() #+ ['red']
# lbls = ['sAMC-500', 'sAMC-q400', 'sAMC-300'] #+ ['$\\tilde{T} = 1$']

# lbls = ['$\\tilde{T} = 1\'$'] + [f'$\\tilde{{T}} = {t}$' for t in np.linspace(0.5,1.0,6)]
lbls = [f'$\\tilde{{T}} = {t}$' for t in np.linspace(0.5,1.0,6)]
clrs = get_cm(np.arange(5,11),'inferno',min_val=0.1,max_val=0.8)


slopes = np.zeros(len(datafiles))
intercepts = np.zeros(len(datafiles))
fit_r2 = np.zeros(len(datafiles))

# datafile_1 = datadir + 'ata_structures/avg_dfs_radii_tempdot6_relaxed_263structures.npy'
# datafile_2 = datadir + 'avg_dfs_radii_pCNN_relaxed.npy'

setup_tex(fontsize=50)
fig = plt.figure()

ax = fig.add_subplot(111)

fit_l1 = 20
fit_l2 = 60


ax.set_xscale('log')
ax.set_yscale('log')

alphas = np.zeros(len(datafiles))

# r = np.linspace(1,700,700)
for k, datafile, c, lbl in zip(range(len(datafiles)),datafiles,clrs,lbls):
# for datafile, c, lbl in zip(datafiles,clrs,lbls):
    dat = np.load(datafile)
    r = dat[:,0]
    dfs = dat[:,1]
    a, b, r2 = fit_dfs(r, dfs,lbounds=[fit_l1,fit_l2])
    alphas[k] = a+2
    print('Fit for %s found. (%s)\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\n\
        '%(lbl,datafile.split('/')[-1],a, b, r2))
    
    
    ax.plot(r,dfs*(r**2),'o',c=c,ms=10,alpha=0.7,label=lbl)
    # ax.plot(r, np.exp(b)*np.power(r,a),'--',lw=2.0,c=c,label=f'$\ell^{{{a:.3f}}}$')
    ax.plot(r, np.exp(b)*np.power(r,a+2),'--',lw=2.0,c=c)#,label=f'$\ell^{{{a+2:.3f}}}$')


# dat = np.load(datafiles[0])
# r = dat[:,0]
# dfs = dat[:,1]
# a, b, r2 = fit_dfs(r, dfs,lbounds=[fit_l1,fit_l2])
# print('Fit for %s found. (%s)\n\
#     Slope = %f\n\
#     Intercept = %f\n\
#     rval = %f\n\n\
#     '%(lbl,datafile.split('/')[-1],a, b, r2))
 
# ax.plot(r,dfs*(r**2),'o',c='r',ms=4,alpha=0.7,label=lbls[0])
# ax.plot(r, np.exp(b)*np.power(r,a),'--',lw=2.0,c=c,label=f'$\ell^{{{a:.3f}}}$')
# ax.plot(r, np.exp(b)*np.power(r,a+2),'--',lw=2.0,c='r',label=f'$\ell^{{{a+2:.3f}}}$')


ax.axvline(x=fit_l1,ymin=0,ymax=1,c='k',ls='--',lw=0.8)
ax.axvline(x=fit_l2,ymin=0,ymax=1,c='k',ls='--',lw=0.8)

ax.set_xlim([15,70])

ax.set_xlabel('$\ell$ [\AA]')
# ax.set_ylabel('$\sigma_{\\rho}^2(\ell)$')
ax.set_ylabel('$\sigma_{N}^2(\ell)/\ell^2$ [\AA$^{-2}$]')

plt.legend(ncol=3)
plt.show()

temps = np.linspace(0.5,1.1,6)

fig, ax = plt.subplots()
for t, a, c in zip(temps,alphas,clrs):
    ax.plot(t,a,c=c,marker='^',markersize=20,zorder=10)

ax.plot(temps,alphas,color='gray',ls='--',lw=1.0,zorder=1)

ax.axhline(y=0,xmin=0,xmax=1,c='k',ls='--',lw=3.0)

ax.set_xlabel('$\\tilde{T}$')
ax.set_ylabel('Fluctuations exponent $\\alpha$')
plt.show()

