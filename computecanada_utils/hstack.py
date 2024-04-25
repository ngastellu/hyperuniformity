#!/usr/bin/env python

import numpy as np
import os

for n in range(132,264):
    print('\n')
    print(n)
    os.chdir('job-%d'%n)
    try:
        dfs_list = [np.load('dfs_unrelaxed-%d.npy'%j) for j in range(2)]
        radii_list = [np.load('radii-%d.npy'%j) for j in range(2)]
    except FileNotFoundError:
        print('file not found!')
        os.chdir('..')
        continue
    dfs = np.hstack(dfs_list)
    radii = np.hstack(radii_list)
    dat = np.vstack((radii,dfs)).T
    os.chdir('..')
    np.save('hstacked_data/dfs_radii_unrelaxed_sample-%d.npy'%n, dat)
    
