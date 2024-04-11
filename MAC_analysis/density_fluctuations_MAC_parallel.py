#!/usr/bin/env python

import numpy as np
from scipy.spatial import cKDTree
from time import perf_counter
import mpi4py import MPI

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

    for k, N in enumerate(Ns):
        center = np.random.randint(L,size=2)
        index_list = grid_tree.query_ball_point(center,l)
        sampled_indices = grid_points[index_list]
        sampled_grid = grid[sampled_indices[:,0],sampled_indices[:,1]]
        Ns[k] = np.sum(sampled_grid)

    variance = np.var(Ns/area)
    return variance


# ******* MAIN ********

rank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()

print('Hello from process %d of %d.'%(rank+1,size)) #check all processes are running

MAC_structure = np.load('28_16x16_MACs.npy')[rank] #analyse each MAC structure in parallel using MPI
grid_points = np.array(list(np.ndindex(MAC_structure.shape)))
grid_tree = cKDTree(grid_points)

L = MAC_structure.shape[0]
radii = np.linspace(1,L,1000)

dfs = np.zeros(radii.shape[0],dtype=float)
for k,r in enumerate(radii):
    dfs[k], *_ = DensityFluctuations(MAC_structure,grid_points,grid_tree,L,r,25)

np.save('dfs-%d.npy'%(rank+1),dfs) #distinguish the output of each process by its number
np.save('radii.npy',radii)
