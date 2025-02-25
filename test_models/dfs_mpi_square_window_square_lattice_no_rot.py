#!/usr/bin/env python

import sys
import os
from glob import glob
import numpy as np
from scipy.spatial import KDTree
from density_fluctuations import NumberFluctuationsSquareWindow
from mpi4py import MPI
from qcnico.lattice import cartesian_product

def square_lattice(a, n):
    points_1d = np.arange(n) * a
    return cartesian_product(points_1d,points_1d)


rank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()

# rCC = 1.8 # max C-C 'bond length' (i.e. nearest-neigbour distance)
eps = 1.0 # additional 'whitespace' between cell and its periodic image

nsamples = int(sys.argv[1])

a = 1.0 # lattice constant
n = 400 # nb. of points in x and y direction

pos = square_lattice(a,n)

n_side_lengths = 10000
lmax = 100
print(f'Max sample window size = {lmax}', flush=True)
print(f'Number of radii = {n_side_lengths} --> dr = {(lmax-1)/n_side_lengths}',flush=True)

all_side_lengths = np.linspace(1,lmax,n_side_lengths) 

M = all_side_lengths.shape[0] // nprocs

if rank < nprocs:
    side_lengths = all_side_lengths[rank*M:(rank+1)*M]
else:
    side_lengths = all_side_lengths[rank*M:]


nfs = np.zeros(M,dtype=float) 
for n,l in enumerate(side_lengths):

    lx = np.min(pos[:,0])
    ly = np.min(pos[:,1])

    Lx = np.max(pos[:,0])
    Ly = np.max(pos[:,1])

    nflucs = NumberFluctuationsSquareWindow(pos,l,[lx,Lx],[ly,Ly],nsamples,restrict_centres=False)
    nfs[n] = nflucs


# avg_nfs = np.mean(nfs,axis=0) # perform average over sampled orientations
np.save('nfs_theta0-%d_pbc.npy'%(rank), nfs) #distinguish the output of each process by its number
np.save('side_lengths_theta0-%d_pbc.npy'%rank,side_lengths)