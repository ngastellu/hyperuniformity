#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

def DensityFluctuations(grid,L,l,sample_size):
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

    if l%2 != 0: l += 1 #makes it easier if l is even
    
    #check whether the sampled volume is smaller than the entire grid
    if l >= L and L%2==1:
        print('Sampled volume l**2 = {}**2 is too large. Setting l={}-1, its maximum allowed even value.'.format(l,L))
        l = L-1
    elif l >= L and L%2==0:
        print('Sampled volume l**2 = {}**2 is too large. Setting l={}-2, its maximum allowed even value.'.format(l,L))
        l = L-2

    volume = l**2
    Ns = np.zeros(sample_size,dtype=np.float)

    for k in range(sample_size):
        grid_to_sample = np.copy(grid)
        i,j = np.random.randint(0,L,2) #select random site around which the grid will be sampled 
        radius = l // 2
        
        #check that the grid can be correctly sampled around the grid point
        if i < radius:
            shift = radius - i
            grid_to_sample = np.roll(grid_to_sample,shift,axis=0)
            i += shift
        elif i > L - radius - 1:
            shift = L - radius - 1 - i
            grid_to_sample = np.roll(grid_to_sample,shift,axis=0)
            i += shift
        if j < radius:
            shift = radius - j
            grid_to_sample = np.roll(grid_to_sample,shift,axis=1)
            j += shift
        elif j > L - radius - 1:
            shift = L - radius - 1 - j
            grid_to_sample = np.roll(grid_to_sample,shift,axis=1)
            j += shift

        sampled_grid = grid_to_sample[i-(l//2):i+(l//2)+1, j-(l//2):j+(l//2)+1]
        
        Ns[k] = np.sum(sampled_grid)

    variance = np.var(Ns/volume)

    return variance


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

    if len(np.array(site)) != n:
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

### Set up grid and randomly occupy certain sites ###

size = 50
grid = np.zeros((size,size),dtype=int)

N = 3000 #number of particles
occupied_sites = np.random.randint(0,size,(N,2))

#occupy the grid
for site in occupied_sites:
    i,j = site
    grid[i,j] += 1

#plot initial state of grid
#plt.ion() #turn interactive mode on
plt.imshow(grid,cmap='hot',interpolation='gaussian')
plt.colorbar()
plt.show()

### Dynamics ###

activity_threshold = 3 #minimum number of particles in an active site
active_bools = grid >= activity_threshold
not_absorbed = np.any(active_bools)
nb_iterations = 0
iteration_times = np.zeros(100000)

while not_absorbed or nb_iterations < 100000:
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
    
    end = perf_counter()
    if nb_iterations < 100000:
        iteration_times[nb_iterations] = end - start
    nb_iterations += 1


    #plot current state of grid
    #plt.imshow(grid,cmap='hot')
    #plt.draw()
    #plt.pause(1e-3)

#plt.ioff()

print('Absorbing state reached after {} iterations.'.format(nb_iterations))

#plot final state of the grid
plt.imshow(grid,cmap='hot',interpolation='gaussian')
plt.colorbar()
plt.show()

plt.close()

plt.plot(iteration_times,'r-')
plt.show()
