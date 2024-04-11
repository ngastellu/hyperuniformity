#!/usr/bin/env python

import numpy as np
import os

for n in range(1,301):
    os.chdir('job-%d'%n)
    dfs_list = [np.load('dfs-%d.npy'%j) for j in range(40)]
    dfs = np.hstack(dfs_list)
    os.chdir('..')
    np.save('hstacked_data/dfs_job-%d.npy'%n)
    
