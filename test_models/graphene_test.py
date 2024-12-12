#!/usr/bin/env python

import sys
import numpy as np
from density_fluctuations import DensityFluctuationsRS
from scipy.spatial import KDTree
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()

pos = np.load('square_ase_graphene_cel_big.npy')
Lx = 442.712186414605
Ly = 357.84

lx = ly = 0
tree = KDTree(pos,boxsize=[Lx,Ly])
nradii = 500
rmax = 100
nsamples = 1000


M = int(nradii/nprocs)

if rank < nprocs - 1:
    radii = np.linspace(1,rmax,nradii)[rank*M:(rank+1)*M]

else:
    radii = np.linspace(1,rmax,nradii)[rank*M:]


dfs = np.zeros(radii.shape[0],dtype=float)

for k,r in enumerate(radii):
    dflucs, (in_samp, rhos, rdat) = DensityFluctuationsRS(tree,r,[lx,Lx],[ly,Ly],nsamples,return_rdata=True,return_insample=True,return_densities=True)
    dfs[k] = dflucs

np.save('dfs-%d_pbc.npy'%(rank),dfs) #distinguish the output of each process by its number
np.save('radii-%d_pbc.npy'%rank,radii)