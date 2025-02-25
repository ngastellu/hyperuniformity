#!/usr/bin/env python

import numpy as np

def npoints_poisson(lam, area, rng):
    intensity = lam*area
    npoints = int(rng.poisson(intensity,1))
    return npoints

def populate_random_grid(grid, grid_indices, lam, seed=None):
    rng = np.random.default_rng(seed)

    area = np.prod(grid.shape)
    npoints = npoints_poisson(lam,area,rng) #number of occupied grid points
    iocc = rng.choice(grid_indices,size=npoints,replace=False) # indices of occupied grid points
    grid[iocc[:,0], iocc[:,1]] = True
    
    return grid


if __name__ == "__main__":
    import sys
    from density_fluctuations import FluctuationsGrid_vectorised
    from qcnico.lattice import cartesian_product
    from scipy.spatial import KDTree
    from mpi4py import MPI
    from time import perf_counter


    rank = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()

    seed = int(sys.argv[1])
    nsamples = 50
    lambda_mac = 64000/(400*400) #density of MAC
    lx = ly = 0
    a = 500 #angstroms
    Lx = Ly = a
    eps_pbc= 1.0

    grid = np.zeros((Lx, Ly), dtype=bool)
    grid_indices = cartesian_product(np.arange(Lx), np.arange(Ly))

    sample = populate_random_grid(grid, grid_indices, lambda_mac, seed=seed)
    tree = KDTree(grid_indices, boxsize=(a,a))

    nradii = 1200
    rmax = Lx / 5

    M = int(nradii/nprocs)

    if rank < nprocs - 1:
        radii = np.logspace(0,np.log10(rmax),nradii,base=10)[rank*M:(rank+1)*M]
    else:
        radii = np.logspace(0,np.log10(rmax),nradii,base=10)[rank*M:]
    
    print(f'[proc {rank}] radii = [{np.min(radii)} ; {np.max(radii)}] ({radii.shape} points)')
    nfs = np.zeros(radii.shape[0],dtype=float)
    rdata = np.zeros((radii.shape[0]*nsamples,3))
    for k,r in enumerate(radii):
        start = perf_counter()
        nfs[k] = FluctuationsGrid_vectorised(grid, grid_indices, tree, Lx, r, nsamples, fluctuations_type='number')
        end = perf_counter()
        print(f'[proc {rank}] r = {r} took {end - start} seconds', flush=True)

    np.save('nfs-%d_pbc.npy'%(rank),nfs) #distinguish the output of each process by its number
    np.save('radii-%d_pbc.npy'%rank,radii)

