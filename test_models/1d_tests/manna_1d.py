#!/usr/bin/env python

import numpy as np
from scipy import spatial
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
        center = np.random.randint(0,L)
        upper_bound = center + l
        lower_bound = center - l

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

        

### Set up grid and randomly occupy certain sites ###

global_start = perf_counter()

nb_realisations = 50 #number of grids generated and sampled at each density

size = 100000

densities = np.array([1.66,1.665,1.668,1.671,1.6715,1.6718])

rmax = 25000
radii = np.arange(1,rmax)
fluctuations = np.zeros((len(densities),len(radii)),dtype=np.float)
avg_times = np.zeros((len(densities),len(radii)),dtype=np.float)

for j,rho in enumerate(densities):
    print(rho,flush=True)
    for n in range(nb_realisations):
        grid = np.zeros(size,dtype=int)
        N = int(rho*size)
        occupied_sites = np.random.randint(0,size,N)

        #occupy the grid
        for site in occupied_sites:
            grid[site] += 1

        init_end = perf_counter()


        ### Dynamics ###

        activity_threshold = 3 #minimum number of particles in an active site
        active_bools = grid >= activity_threshold
        not_absorbed = np.any(active_bools)
        nb_iterations = 0

        dynamics_start = perf_counter()

        while not_absorbed:
            active_sites = active_bools.nonzero()[0]
            #occupancies = grid[active_sites]
            #nb_shifted_right = np.random.binomial(occupancies,0.5)
            #nb_shifted_left = occupancies - nb_shifted_right
            for i in active_sites:
                occ = grid[i]
                nb_shifted_right = np.random.binomial(occ,0.5)
                nb_shifted_left = occ - nb_shifted_right
                grid[i] = 0
                if i != size - 1:
                    grid[i+1] += nb_shifted_right
                    grid[i-1] += nb_shifted_left
                else:
                    grid[0] += nb_shifted_right
                    grid[i-1] += nb_shifted_left
            
            #check which sites are active
            active_bools = grid >= activity_threshold
            not_absorbed = np.any(active_bools)

            nb_iterations += 1

        dynamics_end = perf_counter()
            


        for k,l in enumerate(radii):
            rho_variance, avg_time = DensityFluctuations(grid,size,l,100)
            #fluctuations[j,k], avg_times[j,k] += DensityFluctuations(grid,size,l,250)
            fluctuations[j,k] += rho_variance
            avg_times[j,k] += avg_time

#average over each realisation
fluctuations /= nb_realisations
avg_times /= nb_realisations

np.save('radii.npy',radii)
np.save('fluctuations_data_1d.npy',fluctuations)
np.save('avg_times_data_1d.npy',avg_times)

global_end = perf_counter()
print('Running this script took {}s.'.format(global_end - global_start))
