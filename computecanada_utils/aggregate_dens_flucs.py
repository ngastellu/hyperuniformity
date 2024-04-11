#!/usr/bin/env python

import numpy as np

radii0 = np.load('job-1/radii.npy')

dens_flucs = np.zeros((12,radii0.shape[0]),dtype=float)
avg_dens_flucs = np.zeros((12,radii0.shape[0]),dtype=float)
w = np.ones(12)*3
w[-2:] = 5

for i in range(1,7):
    radii = np.load('job-%d/radii.npy'%i)
    assert np.all(radii0 == radii), 'Mismatched radius arrays.'
    
    for j, df in enumerate(dens_flucs):
        df = np.load('job-%d/fluctuations_data_1d_%d.npy'%(i,j+1))

    avg_dens_flucs[i-1] = np.average(dens_flucs,axis=0,weights=w)

np.save('avg_density_fluctuations_1d_10k.npy',avg_density_fluctuations)
