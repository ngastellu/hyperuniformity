#!/usr/bin/env python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex



structype = 't1'

l1 = 20
l2 = 60

if structype == '40x40':    
    official_name = 'sAMC-500'
elif structype == 'tempdot5':
    official_name = 'sAMC-300'
elif structype == 'tempdot6':
    official_name = 'sAMC-q400'
elif structype == 't1':
    official_name = '$\\tilde{T} = 1$'
else:
    print(f'{structype} is not a valid ensemble name.')
    sys.exit(1)

rhosc_datadir = f'/Users/nico/Desktop/simulation_outputs/hyperuniformity/bigMAC/rho_v_sample_crystallinity/{structype}/'
r_datadir = f'/Users/nico/Desktop/simulation_outputs/hyperuniformity/bigMAC/sample_radii/{structype}/'
rhosc_npys = os.listdir(rhosc_datadir)

setup_tex(fontsize=25)

fig,ax = plt.subplots()

for npy in rhosc_npys:
    n = npy.split('.')[0].split('-')[1] 
    radii = np.load(os.path.join(r_datadir,f'radii-{n}.npy'))
    rho, sample_cryst = np.load(os.path.join(rhosc_datadir,npy))
    filter = (radii >= l1) * (radii <= l2)
    ye = ax.scatter(rho[filter], sample_cryst[filter],c=radii[filter],s=3.0)

ax.set_xlabel('Density $\\rho(\ell)$ [\AA$^{-2}$]')
ax.set_ylabel('Crystallinity')
cbar = fig.colorbar(ye,ax=ax,orientation='vertical',label='Sample radius $\ell$ [\AA]')
ax.set_title(official_name)
plt.show()
