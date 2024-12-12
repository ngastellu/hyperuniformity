#!/usr/bin/env python

import numpy as np

def npoints_poisson(lam, area, rng):
    intensity = lam*area
    npoints = int(rng.poisson(intensity,1))
    return npoints

def generate_square_ppp(lam,a,seed=64):
    rng = np.random.default_rng(seed)
    area = a * a
    npoints = npoints_poisson(lam,area,rng)
    points = rng.random((npoints,2)) * a
    return points


if __name__ == "__main__":
    import sys
    from density_fluctuations import DensityFluctuationsRS
    from scipy.spatial import KDTree
    from mpi4py import MPI


    rank = MPI.COMM_WORLD.Get_rank()
    nprocs = MPI.COMM_WORLD.Get_size()

    seed = int(sys.argv[1])
    nsamples = int(sys.argv[2])
    lambda_mac = 64000/(400*400) #density of MAC
    lx = ly = 0
    a = 400 #angstroms
    Lx = Ly = a
    eps_pbc= 1.0

    sample = generate_square_ppp(lambda_mac,a,seed=seed)
    tree = KDTree(sample, boxsize=(a,a))

    nradii = 500
    rmax = 100

    M = int(nradii/nprocs)

    if rank < nprocs - 1:
        radii = np.linspace(1,rmax,nradii)[rank*M:(rank+1)*M]

    else:
        radii = np.linspace(1,rmax,nradii)[rank*M:]

    dfs = np.zeros(radii.shape[0],dtype=float)
    rdata = np.zeros((radii.shape[0]*nsamples,3))
    for k,r in enumerate(radii):
        dflucs, (in_samp, rhos, rdat) = DensityFluctuationsRS(tree,r,[lx,Lx],[ly,Ly],nsamples,return_rdata=True,return_insample=True,return_densities=True)
        dfs[k] = dflucs
        rdata[nsamples*k:nsamples*(k+1)] = rdat

    np.save('dfs-%d_pbc.npy'%(rank),dfs) #distinguish the output of each process by its number
    np.save('radii-%d_pbc.npy'%rank,radii)
    np.save(f'rdata-{rank}_pbc.npy', rdata)

