#!/usr/bin/env python

import sys
import os
from glob import glob
import numpy as np
from scipy.spatial import KDTree
from density_fluctuations import DensityFluctuationsRS
from mpi4py import MPI
from qcnico.coords_io import read_xyz


rank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()

rCC = 1.8 # max C-C 'bond length' (i.e. nearest-neigbour distance)
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

tree = KDTree(pos,boxsize=[Lx+eps,Ly+eps])

nradii = 500
#rmax = np.min(tree.maxes)/5.0
rmax = 100
print(f'Max sample window size = {rmax}', flush=True)
print(f'Number of radii = {nradii} --> dr = {(rmax-1)/nradii}',flush=True)



M = int(nradii/nprocs)

if rank < nprocs - 1:
    radii = np.linspace(1,rmax,nradii)[rank*M:(rank+1)*M]

else:
    radii = np.linspace(1,rmax,nradii)[rank*M:]


dfs = np.zeros(radii.shape[0],dtype=float)
rdata = np.zeros((radii.shape[0]*nsamples,3))
all_rhos = np.zeros((radii.shape[0],nsamples))
all_sample_masks = np.zeros(radii.shape[0]*nsamples,pos.shape[0],dtype='bool')
for k,r in enumerate(radii):
    dflucs, in_samp, rhos, rdat = DensityFluctuationsRS(tree,r,[lx,Lx],[ly,Ly],nsamples,return_rdata=True,return_insample=True,return_densities=True)
    all_sample_masks[nsamples*k:nsamples*(k+1),:] = in_samp
    dfs[k] = dflucs
    all_rhos[k,:] = rhos
    rdata[nsamples*k:nsamples*(k+1)] = rdat

np.save('dfs-%d_pbc.npy'%(rank),dfs) #distinguish the output of each process by its number
np.save('radii-%d_pbc.npy'%rank,radii)
np.save(f'rdata-{rank}_pbc.npy', rdata)
np.save(f'densities-{rank}.npy',all_rhos)
np.save(f'sample_masks-{rank}.npy', all_sample_masks)