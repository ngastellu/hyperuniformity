#!/usr/bin/env python

import numpy as np
from time import perf_counter
import os
import sys
from density_fluctuations import FluctuationsGrid_vectorised, get_grid_indices
from scipy.spatial import KDTree
from mpi4py import MPI


npy_name = sys.argv[1]
temp = float(npy_name.split('_')[2])

all_ising_samples = np.load(os.path.expanduser(f'~/scratch/hyperuniformity/ising/MCMC-generated/samples/{npy_name}'))


rank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()


grid0 = all_ising_samples[0,:,:] # all Ising realizations have the same shape, so obtain `grid_points` and `grid_tree` from the first one, burn the first 300 pixels to match Ata's susceptibility analysis

start = perf_counter()
grid_points = get_grid_indices(grid0)
end1 = perf_counter()
print(f'Getting `grid_points` took {end1-start} secs.', flush=True)

grid_tree = KDTree(grid_points,boxsize=grid0.shape) # assume PBC
end2 = perf_counter()
print(f'Getting `grid_tree` took {end2-end1} secs.', flush=True)



nsamples = all_ising_samples.shape[0]
M = nsamples // nprocs

L = grid0.shape[0] #assume square grid i.e. grid0.shape[0] = grid0.shape[1]

rmax = 0.15 * L # sample up to 40% of lattice per radii
nradii = 400
nwindows = 20

radii = np.linspace(0,rmax,nradii)


# Partition Ising samples over different MPI processes
if rank < nprocs - 1:
    ising_samples = all_ising_samples[rank*M:(rank+1)*M,:,:]
else:
    ising_samples = all_ising_samples[rank*M:,:,:]

nsamples = ising_samples.shape[0] # usually equal to M, except if rank = nprocs-1

nfs = np.zeros((nsamples,nradii))

for n, grid in enumerate(ising_samples):
    print(n, grid.shape, grid.sum())
    igrid = rank*M + n
    if np.all(grid == 0) or np.all(grid==1) or np.all(grid==-1):
        print(f'--- Skipping grid {igrid} (completely uniform) ---')
        continue
    start = perf_counter()
    for k, r in enumerate(radii):
        nfs[n,k] = FluctuationsGrid_vectorised(grid,grid_points,grid_tree,L,r,nwindows,fluctuations_type='number')
    end = perf_counter()
    print(f'[proc. {rank}] Grid {igrid} done! [{end-start} seconds]',flush=True)

np.save(f'T_{temp}_L_{L}/nfs-{rank}.npy', np.mean(nfs,axis=0))
np.save(f'T_{temp}_L_{L}/radii-{rank}.npy', radii)
