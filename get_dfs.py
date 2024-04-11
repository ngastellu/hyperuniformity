#!/usr/bin/env python

import numpy as np
from scipy.spatial import cKDTree
from time import perf_counter
from mpi4py import MPI

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


    for k, N in enumerate(Ns):
        #center = np.random.randint(L,size=2)
        center = np.random.random(2)*(L-2*l) + l
        index_list = grid_tree.query_ball_point(center,l)
        sampled_indices = grid_points[index_list]
        sampled_grid = grid[sampled_indices[:,0],sampled_indices[:,1]]
        Ns[k] = np.sum(sampled_grid)

    variance = np.var(Ns/area)
    return variance


# ******* MAIN ********

start = perf_counter()

rank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()

#print('Hello from process %d of %d.'%(rank+1,nprocs)) #check all processes are running

full_arr = np.load(sys.argv[1])

nstrucs_per_proc = full_arr.shape[0] // nprocs

if full_arr.ndim == 3:
    if rank < nprocs - 1: 
        structures = full_arr[rank*nstrucs_per_proc:(rank+1)*nstrucs_per_proc,:,:]
    else: 
        structures = full_arr[rank*nstrucs_per_proc:-1,:,:]

if full_arr.ndim == 4:
    if rank < nprocs - 1: 
        structures = full_arr[rank*nstrucs_per_proc:(rank+1)*nstrucs_per_proc,0,:,:]
    else: 
        structures = full_arr[rank*nstrucs_per_proc:-1,0,:,:]


grid_points = np.array(list(np.ndindex(structures[0])))
grid_tree = cKDTree(grid_points)

L = structures.shape[-1]

#resolution of 3 radii values per grid spacing
radii = np.linspace(1,L/2,L*3)

dfs = np.zeros(radii.shape[0],dtype=float)

for struc in structures:
    for k,r in enumerate(radii):
        dfs[k] += DensityFluctuations(struc,grid_points,grid_tree,L,r,10)

dfs /= structures.shape[0]

np.save('dfs-%d.npy'%(rank+1),dfs) #distinguish the output of each process by its number

if rank == 0:
    np.save('radii.npy',radii)

end = perf_counter()
print('Process %d took %f seconds to execute.'%(rank+1,end-start))
