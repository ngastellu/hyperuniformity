#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import get_cm, setup_tex


temps = [2.35, 2.625, 3.1]
clrs = get_cm(temps,min_val=0.2,max_val=0.8,cmap_str='inferno')
L = 50

setup_tex(fontsize=20)

fig, ax = plt.subplots()

for T, c in zip(temps,clrs):
    r, nfs = np.load(f'/Users/nico/Desktop/simulation_outputs/hyperuniformity/ising/MCMC-generated/nfs_radii_T_{T}_L_{L}.npy')
    ax.plot(r,nfs/(np.pi*r**2),'o',c=c,ms=5,alpha=0.7,label=f'$T = {T}$')

ax.set_xlabel('Window radius $\ell$')
ax.set_ylabel('$\sigma_N(\ell)^2/\pi\ell^2$')
ax.set_title(f'MCMC Ising realizations with $L={L}$')
ax.legend()
plt.show()