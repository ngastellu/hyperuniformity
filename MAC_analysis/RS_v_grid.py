#!/usr/bin/env pythonw

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from os import path
from glob import glob



pixels2angstroms = 0.2 # 5 pixels/angstrom
datadir = '/Users/nico/Desktop/simulation_outputs/hyperuniformity/bigMAC'

#rfiles = glob(path.join(datadir,'radii_*.npy'))
#sys_names = [('_').join(path.basename(rf).split('_')[1:]).split('.')[0] for rf in rfiles]

rcParams['text.usetex'] = True
rcParams['font.size'] = 21
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


#rfile = path.join(datadir,'avg_dfs_40x40_MAC.npy')
flucfile = path.join(datadir,'avg_dfs_bigMAC.npy')
small_l_flucfile = path.join(datadir,'small-r_avg_dfs_bigMAC.npy')

grid_flucfile = path.join(datadir, 'dfs_radii_bigMAC_unsymm_RS.npy')
RS_ls, RS_dfs = np.load(grid_flucfile).T

ls = np.linspace(1,1000,7000)
dfs = np.load(flucfile)

ls *= pixels2angstroms
dfs *= 1.0/(pixels2angstroms**4)
boundary_l = 800*pixels2angstroms# boundary value of l separating the two scaling regions
boundary_index = np.max((ls <= boundary_l).nonzero()[0])

fig0, ax0 = plt.subplots()
ax0.plot(ls[:boundary_index],dfs[:boundary_index],'ro', ms=1, label='pixel')
ax0.plot(RS_ls[:boundary_index],RS_dfs[:boundary_index],'bo', ms=1, label='real-space')

ax0.set_xscale('log')
ax0.set_yscale('log')

plt.show()



# Define fitting regions do fit
boundary_l1 = 10
boundary_l2 = 85# boundary value of l separating the two scaling regions

for ldat, dfdat, m in zip([ls,RS_ls],[dfs,RS_dfs],['Grid', 'Real space']):

    boundary_index1 = np.max((ldat <= boundary_l1).nonzero()[0])
    boundary_index2 = np.max((ldat <= boundary_l2).nonzero()[0])


    fit_radii2 = ldat[boundary_index1:boundary_index2]
    fit_dfs2 = dfdat[boundary_index1:boundary_index2]


    a2, b2, rval2, *_ = linregress(np.log(fit_radii2),np.log(fit_dfs2))
    print('For %f <= l <= %f:\n\
            Slope = %f\n\
            Intercept = %f\n\
            rval = %f\n\
            '%(boundary_l1, boundary_l2, a2, b2, rval2))


    plt_radii = np.logspace(-1, 2,1000)
    print(plt_radii.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xscale('log')
    ax.set_yscale('log')

    #ax.plot(small_r_ls,small_r_dfs,'o',c='r',ms=1)
    ax.plot(ldat,dfdat,'o',c='r',ms=1)
    #ax.plot(fit_radii2,dfs[0]*fit_radii2**(-2),'k--',lw=0.7,label='$\ell^{-2}$')
    #ax.plot(fit_radii, np.exp(b)*(fit_radii**a), 'b-', label='Fit: $\lambda = %f$'%a,lw=2.0)
    ax.plot(plt_radii, np.exp(b2)*(plt_radii**a2), 'b-', label='Fit: $\sigma_{\\rho}^2(\ell)\sim\ell^{%3.2f}$'%a2,lw=1.0)

    plt.suptitle('%s method fit'%m)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')
    ax.set_xlim(0.2,85)
    plt.legend()
    plt.show()
