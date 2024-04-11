#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

def DensityFluctuations(grid,L,l,sample_size):
    """Measures the density fluctuations of a 1-dimensional Manna grid, where the density is measured over a volume V=2*l.
    
    Parameters
    ----------
    
    grid: `numpy.ndarray`
        Manna model grid where element i (for a 1D grid) corresponds to the number of particles at the site located at the ith row and jth column. 
    L: `int`
        Size of the grid; i.e. number of sites for a 1D grid.
    l: `float`
        Lengthscale used for the densities. For a n-dimensional model density ~ (nb. of particles)/(l**n).
    sample_size: `int`
        Number of data points used to estimate the density standard deviation.

    Output
    ------
    
    variance: `float`
        Variance of the density in the input grid."""

    if l >= size/2.0:
        print('DensityFluctuations error: sample radius l={} too large for a grid of size {}. Setting it to its maximum allowed value: {}.'.format(l,size,(size/2.0) - 1))
        l = (size/2.0) - 1
    
    Ns = np.zeros(sample_size,dtype=np.float)
    area = 2*l

    sample_times = np.zeros(sample_size,dtype=np.float)

    for k in range(sample_size):
        start = perf_counter()
        
        ### Uncomment this section for integer sampling (i.e. center of window is always a lattice site)
        #center = np.random.randint(0,L)
        #upper_bound = center + l
        #lower_bound = center - l

        center = np.random.rand()*L
        upper_bound = int(center + l)
        lower_bound = int(center - l)

        if upper_bound < size and lower_bound >= 0:
            Ns[k] = np.sum(grid[lower_bound:upper_bound+1])
        
        #if limits of sampled array exit the array, apply PBC
        elif upper_bound >= size:
            Ns[k] = np.sum(grid[lower_bound:]) + np.sum(grid[:upper_bound - size + 1])
        else: # lower_bound < 0
            Ns[k] = np.sum(grid[:upper_bound+1]) + np.sum(grid[size + lower_bound:])
        end = perf_counter()
        sample_times[k] = end - start


    variance = np.var(Ns/area)
    return variance, np.mean(sample_times)

size = 100000
crystal = np.zeros(size,dtype=int)
for i in range(size):
    if i % 2: crystal[i] += 1

#crystal = np.ones(size)

radii = np.arange(1,int(size/3.0)+10)
fluctuations = np.zeros(radii.shape)
for k,r in enumerate(radii):
    fluctuations[k], _  = DensityFluctuations(crystal,size,r,100)

print((fluctuations==0).nonzero())

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')

plt.plot(radii,fluctuations,'ro',ms=0.5)
plt.plot(radii,1.0/(radii**2),'k--',lw=0.8)
plt.plot(radii,1.0/radii,'k-.',lw=0.8)
plt.show()


