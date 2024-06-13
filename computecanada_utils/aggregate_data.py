#!/usr/bin/env python

import numpy as np
import os
from sys import exit


savedirs = ['hstacked_data', 'vstacked_rdata']

for d in savedirs:
    if not os.path.exists(d):
        os.mkdir(d)
    else:
        proceed_input = input(f'!!! Caution directory {d} already exists and will be overwritten !!! Proceed? [y/n] ')
        if proceed_input.strip().lower() == 'y':
            proceed = True
        else:
            proceed = False
        if not proceed:
            print('Exiting.')
            exit()
        

for n in range(1,301):
    print('\n')
    print(n)
    os.chdir('job-%d'%n)
    try:
        dfs_list = [np.load('dfs-%d.npy'%j) for j in range(2)]
        radii_list = [np.load('radii-%d.npy'%j) for j in range(2)]
        rdata_list = [np.load(f'rdata-{j}.npy') for j in range(2)]
    except FileNotFoundError:
        print('file not found!')
        os.chdir('..')
        continue
    dfs = np.hstack(dfs_list)
    radii = np.hstack(radii_list)
    rdata = np.vstack(rdata_list)
    dat = np.vstack((radii,dfs)).T
    os.chdir('..')
    np.save('hstacked_data/dfs_radii_relaxed_sample-%d.npy'%n, dat)
    np.save('vstacked_rdata/rdata_relaxed_sample-%d.npy'%n, rdata) 
