#!/usr/bin/env python

import numpy as np
from glob import glob


npys = glob('hstacked_data/dfs_radii_unrelaxed_sample-*.npy')


nfiles = len(npys)


dat = np.load(npys[0])
r = dat[:,0]
avg_dfs = dat[:,1]

all_radii = np.zeros((nfiles,r.shape[0]))
all_radii[0,:] = r

for k, f in enumerate(npys):
    dat = np.load(f)
    all_radii[k,:] = dat[:,0]
    avg_dfs += dat[:,1]


print('Radii match: ', np.all(all_radii == all_radii[0,:]))

avg_dfs /= nfiles
np.save('avg_dfs_radii_tempdot6_relaxed_263structures.npy', np.vstack((r,avg_dfs)).T)
