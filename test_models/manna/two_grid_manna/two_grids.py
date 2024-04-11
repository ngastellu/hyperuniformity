#!/usr/bin/env pythonw

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

np.random.seed(0)        

### Set up grid and randomly occupy certain sites ###

size = 200 #1000
grid = np.zeros((size,size),dtype=int)
intermediate_grid = np.zeros((size,size),dtype=int)
initial_grid = np.zeros((size,size),dtype=int)
#grid_points = np.array([[i,j] for i in range(size) for j in range(size)])
grid_points = np.array(list(np.ndindex(grid.shape))) #list of all integer pairs [i,j] that index points on the grid
grid_tree = spatial.cKDTree(grid_points,boxsize=size)

density = 1.295 #from https://doi.org/10.1073/pnas.1619260114
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
    if nb_iterations == 480:
        intermediate_grid[:] = grid 

    
    #check which sites are active
    active_bools = grid >= activity_threshold
    not_absorbed = np.any(active_bools)

    nb_iterations += 1
    
print('Absorbing state reached after {} iterations.'.format(nb_iterations))

vmin = np.min(intermediate_grid)
vmax = np.max(intermediate_grid)

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(121)
mesh = ax1.imshow(intermediate_grid,cmap='hot',interpolation='gaussian')
mesh.set_clim(vmin,vmax)
ax1.set_axis_off()
ax2 = fig.add_subplot(122)
mesh2 = ax2.imshow(grid,cmap='hot',interpolation='gaussian')
mesh2.set_clim(vmin,vmax)
ax2.set_axis_off()

#visualise colorbar
fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                            wspace=0.4, hspace=0.1)
cb_ax = fig.add_axes([0.88, 0.1, 0.02, 0.6])
cbar = fig.colorbar(mesh, cax=cb_ax)

#fig.tight_layout()
plt.show()

#plt.imshow(grid,cmap='hot',interpolation='gaussian')
#plt.imshow(intermediate_grid,cmap='hot',interpolation='gaussian')
#plt.colorbar()
#plt.show()

#num_ls = 400
rmax = 99
radii = np.linspace(1,rmax,500)
fluctuations = np.zeros((2,len(radii)),dtype=np.float)

#for k,l in enumerate(np.arange(1,num_ls+1)):
for j,g in enumerate([initial_grid,grid]):
    for k,l in enumerate(radii):
        print(k)
        fluctuations[j,k], *_ = DensityFluctuations(g,grid_points,grid_tree,size,l,750)


np.save('fluctuations_data_2grid.npy',fluctuations)
np.save('radii_2grid.npy',radii)
#np.save('avg_times_data_kdtree.npy',avg_times_data)
#np.save('query_times_data_kdtree.npy',query_times_data)
