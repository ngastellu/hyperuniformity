#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from time import perf_counter

def RandomNeighbour(site, n, L):
    """Returns the indices labelling a randomly chosen nearest neighbour to grid point with indices `site`.
    
    Parameters
    ----------
    
    site: `tuple` or `list` or `numpy.ndarray`
        Two integers labelling a site on a discrete `n`-dimensional grid.
    n: `int`
        Dimension of the grid.
    L: `int`
        Size of the grid (i.e. the grid contains L*L sites).
    
    Output
    ------
    
    neighbour: `tuple`
        Two integers labelling a randomly chosen grid point that is directly adjacent to the point labelled by the input indices."""

    if len(site) != n:
        print('RandomNeighbour error: Size of input does not match the dimension of the grid.\nReturning all zeros.')
        neighbour = np.zeros(n)

    else:
        neighbour = np.array(site)
        nb_neighbours = 2*n #2 nearest neighbours per direction
        dice = np.random.randint(0,nb_neighbours) 
        k = dice // 2 #index that will be updated
        if dice % 2: 
            neighbour[k] += 1
            if neighbour[k] >= L: neighbour[k] = 0 #apply PBC
        else:
            neighbour[k] -= 1
            if neighbour[k] < 0: neighbour[k] = L-1 #apply PBC
    
    return tuple(neighbour)


def DensityFluctuations(grid,grid_points,grid_tree,L,l,sample_size):
    """Measures the density fluctuations of a 2-dimensional Manna grid, where the density is measured over a volume V=l**2.
    
    Parameters
    ----------
    
    grid: `numpy.ndarray`
        Manna model grid where element (i,j) (for a 2D grid) corresponds to the number of particles at the site located at the ith row and jth column. 
    L: `int`
        Size of the grid; e.g. a 2D grid of size L has L*L sites.
    l: `float`
        Lengthscale used for the densities. For a n-dimensional model density = (nb. of particles)/(l**n).
    sample_size: `int`
        Number of data points used to estimate the density standard deviation.

    Output
    ------
    
    variance: `float`
        Variance of the density in the input grid."""


    Ns = np.zeros(sample_size,dtype=np.float)
    area = np.pi*l*l

    sample_times = np.zeros(sample_size,dtype=np.float)
    query_times = np.zeros(sample_size,dtype=np.float)

    for k in range(sample_size):
        start = perf_counter()
        center = np.random.rand(2)*L
        query_start = perf_counter()
        index_list = grid_tree.query_ball_point(center,l)
        query_end = perf_counter()
        sampled_indices = grid_points[index_list]
        sampled_grid = grid[sampled_indices[:,0],sampled_indices[:,1]]
        Ns[k] = np.sum(sampled_grid)
        end = perf_counter()
        query_times[k] = query_end - query_start
        sample_times[k] = end - start

    variance = np.var(Ns/area)
    return variance, np.mean(sample_times), np.mean(query_times)

#--------MAIN---------#

#np.random.seed(0)        

### Set up grid and randomly occupy certain sites ###

size = 100
grid = np.zeros((size,size),dtype=int)
grid_points = np.array(list(np.ndindex(grid.shape))) #list of all integer pairs [i,j] that index points on the grid
grid_tree = spatial.cKDTree(grid_points,boxsize=size)

density = 1.295 #critical density (obtained from https://doi.org/10.1073/pnas.1619260114)
volume = size**2 #for 2D model


N = int(density*volume)
occupied_sites = np.random.randint(0,size,(N,2))

#occupy the grid
for site in occupied_sites:
    i,j = site
    grid[i,j] += 1


plt.imshow(grid,cmap='hot',interpolation='gaussian')
plt.colorbar()
plt.show()


### Dynamics ###

activity_threshold = 3 #minimum number of particles in an active site
active_bools = grid >= activity_threshold
not_absorbed = np.any(active_bools)
nb_iterations = 0

while not_absorbed:
    start = perf_counter()
    active_sites = np.vstack(active_bools.nonzero()).T 
    for site in active_sites:
        i,j = site
        occ = grid[i,j] #number of particles at active site
        grid[i,j] = 0
        for k in range(occ):
            neighbour = RandomNeighbour(site,2,size)
            grid[neighbour] += 1

    
    #check which sites are active
    active_bools = grid >= activity_threshold
    not_absorbed = np.any(active_bools)

    nb_iterations += 1
    
print('Absorbing state reached after {} iterations.\nTotal number of particles conserved: {}.'.format(nb_iterations,N==np.sum(grid)))

rmax = 400
radii = np.linspace(1,rmax,500)
fluctuations = np.zeros(len(radii),dtype=np.float)
avg_times = np.zeros(len(radii),dtype=np.float)
query_times = np.zeros(len(radii),dtype=np.float)

#for k,l in enumerate(np.arange(1,num_ls+1)):
for k,l in enumerate(radii):
    print(k)
    fluctuations[k], avg_times[k], query_times[k] = DensityFluctuations(grid,grid_points,grid_tree,size,l,100)

fluctuations_data = np.vstack((radii,fluctuations))
avg_times_data = np.vstack((radii,avg_times))
query_times_data = np.vstack((radii,query_times))

#np.save('fluctuations_data_kdtree.npy',fluctuations_data)
#np.save('avg_times_data_kdtree.npy',avg_times_data)
#np.save('query_times_data_kdtree.npy',query_times_data)
