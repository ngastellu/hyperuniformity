#!/usr/bin/env python

import numpy as np
from manna_1d import DensityFluctuations
import matplotlib.pyplot as plt


nb_realisations = 100
L = 100000
rho = 1.66
N = rho*L

radii = np.arange(30000)
density_fluctuations = np.zeros(radii.shape[0],dtype=float)
grid = np.zeros(L,dtype=int)

for i in range(nb_realisations):
    inds = np.random.randint(L,size=N)
    grid[inds] += np.ones(N,dtype=int)

    for r in radii:
        density_fluctuations[r] += DensityFluctuations(grid,L,r,100)

density_fluctuations /= nb_realisations


