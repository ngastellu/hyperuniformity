#!/usr/bin/env python

import sys
from os import path
from glob import glob
import numpy as np
from scipy.spatial import cKDTree
from density_fluctuations import DensityFluctuationsRS
from mpi4py import MPI
from qcnico.coords_io import read_xsf, read_xyz
from qcnico.remove_dangling_carbons import remove_dangling_carbons


rank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()

rCC = 1.8

strucindex = int(sys.argv[1])
structype = sys.argv[2]
nsamples = int(sys.argv[3])


if structype == '40x40':
    posfile = path.expanduser(f'~/scratch/clean_bigMAC/40x40/relax/no_PBC/relaxed_structures/bigMAC-{strucindex}_relaxed.xsf')
else:
    if '_old_model' == structype[-10:]:
        posfile = path.expanduser(f'~/scratch/clean_bigMAC/{structype}/sample-{strucindex}/{structype[:-10]}n{strucindex}_relaxed.xsf')
    else:
        #posfile = path.expanduser(f'~/scratch/clean_bigMAC/{structype}/sample-{strucindex}/{structype}n{strucindex}_relaxed.xsf')
        posfile = path.expanduser(f'~/scratch/clean_bigMAC/{structype}/relaxed_structures/{structype}n{strucindex}_relaxed.xsf')
        #posfile = path.expanduser(f'~/scratch/ata_structures/{structype}/{structype}n{strucindex}.xyz')

pos, _ = read_xsf(posfile)
#pos  = read_xyz(posfile)
pos = pos[:,:2]
pos = remove_dangling_carbons(pos, rCC)

lx = np.min(pos[:,0])
Lx = np.max(pos[:,0])

ly = np.min(pos[:,1])
Ly = np.max(pos[:,1])

print(f'Coord bounds along x-direction: {[lx,Lx]}',flush=True)
print(f'Coord bounds along y-direction: {[ly,Ly]}',flush=True)

tree = cKDTree(pos)

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
for k,r in enumerate(radii):
    dflucs, rdat = DensityFluctuationsRS(tree,r,[lx,Lx],[ly,Ly],nsamples)
    dfs[k] = dflucs
    rdata[nsamples*k:nsamples*(k+1)] = rdat


np.save('dfs-%d.npy'%(rank),dfs) #distinguish the output of each process by its number
np.save('radii-%d.npy'%rank,radii)
