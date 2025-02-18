#!/usr/bin/env python

import numpy as np
from time import perf_counter
import os
import sys
from density_fluctuations import FluctuationsGrid_vectorised, get_grid_indices
from scipy.spatial import KDTree
from mpi4py import MPI


gen_temp = sys.argv[1]
all_ising_samples = np.load(os.path.expanduser(f'~/scratch/hyperuniformity/ising/samples/run_0_loadedmodelising2dot6dot{gen_temp}.npy')).astype('bool')


rank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()


grid0 = all_ising_samples[0,0,:,:] # all Ising realizations have the same shape, so obtain `grid_points` and `grid_tree` from the first one

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

rmax = 0.2 * L # sample up to 40% of lattice per radii
nradii = 400
nwindows = 25

radii = np.linspace(0,rmax,nradii)


# Partition Ising samples over different MPI processes
if rank < nprocs - 1:
    ising_samples = all_ising_samples[rank*M:(rank+1)*M,0,:,:]
else:
    ising_samples = all_ising_samples[rank*M:,0,:,:]

nsamples = ising_samples.shape[0] # usually equal to M, except if rank = nprocs-1

nfs = np.zeros((nsamples,nradii))

for n, grid in enumerate(ising_samples):
    start = perf_counter()
    for k, r in enumerate(radii):
        nfs[n,k] = FluctuationsGrid_vectorised(grid,grid_points,grid_tree,L,r,nwindows,fluctuations_type='number')
    end = perf_counter()
    print(f'[proc. {rank}] Grid {rank*M + n} done! [{end-start} seconds]',flush=True)

np.save(f'tdot{gen_temp}/nfs-{rank}.npy', np.mean(nfs,axis=0))
np.save(f'tdot{gen_temp}/radii-{rank}.npy', radii)
