#!/usr/bin/env python

import sys
import os
from glob import glob
import numpy as np
from scipy.spatial import KDTree
from density_fluctuations import NumberFluctuationsSquareWindow
from mpi4py import MPI
from qcnico.coords_io import read_xyz


rank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()

# rCC = 1.8 # max C-C 'bond length' (i.e. nearest-neigbour distance)
eps = 1.0 # additional 'whitespace' between cell and its periodic image

strucindex = int(sys.argv[1])
structype = os.getcwd().split('/')[-2]
nsamples = int(sys.argv[2])


if structype == '40x40':
    xyz_prefix = 'bigMAC-'
else:
    xyz_prefix = structype + 'n'

pos = read_xyz(os.path.expanduser(f'~/scratch/clean_bigMAC/{structype}/relaxed_structures_no_dangle/{xyz_prefix}{strucindex}_relaxed_no-dangle.xyz'))
pos = pos[:,:2]

lx = np.min(pos[:,0])
ly = np.min(pos[:,1])

# Ensure no negative coords (necessary for periodic k-d tree to work)
if lx < 0:
    pos[:,0] -= lx
if ly < 0:
    pos[:,1] -= ly

lx = np.min(pos[:,0])
ly = np.min(pos[:,1])

Lx = np.max(pos[:,0])
Ly = np.max(pos[:,1])


print(f'Coord bounds along x-direction: {[lx,Lx]}',flush=True)
print(f'Coord bounds along y-direction: {[ly,Ly]}',flush=True)

n_side_lengths = 500
#rmax = np.min(tree.maxes)/5.0
lmax = 100
print(f'Max sample window size = {lmax}', flush=True)
print(f'Number of radii = {n_side_lengths} --> dr = {(lmax-1)/n_side_lengths}',flush=True)



M = int(n_side_lengths/nprocs)

if rank < nprocs - 1:
    side_lengths = np.linspace(1,lmax,n_side_lengths)[rank*M:(rank+1)*M]

else:
    side_lengths = np.linspace(1,lmax,n_side_lengths)[rank*M:]

nfs = np.zeros(side_lengths.shape[0],dtype=float)
for k,l in enumerate(side_lengths):
    nflucs = NumberFluctuationsSquareWindow(pos,l,[lx,Lx],[ly,Ly],nsamples)
    nfs[k] = nflucs

np.save('nfs-%d_pbc.npy'%(rank),nfs) #distinguish the output of each process by its number
np.save('side_lengths-%d_pbc.npy'%rank,side_lengths)