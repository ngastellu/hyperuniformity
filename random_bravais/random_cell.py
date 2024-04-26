#!/usr/bin/env python

import sys
import numpy as np
from scipy.spatial import cKDTree
from qcnico.lattice import make_supercell
from density_fluctuations import DensityFluctuationsRS
from mpi4py import MPI


nn = int(sys.argv[1])

np.random.seed(nn)

rank = MPI.COMM_WORLD.Get_rank()
nprocs = MPI.COMM_WORLD.Get_size()


a = 20 #size of unit cell
l = 18 #span of random coords in unot cell
n = 40 #number of atoms per cell
nsamples = 50


latt_vecs = np.eye(2) * a

cell = np.random.rand(2,n) * l

supercell = make_supercell(cell,latt_vecs,50,50)

supercell = supercell.T

np.save(f'random_cell_n{n}_l{l}_a{a}_seed{nn}.npy', supercell)


lx = np.min(supercell[:,0])
Lx = np.max(supercell[:,0])

ly = np.min(supercell[:,1])
Ly = np.max(supercell[:,1])

print(f'Coord bounds along x-direction: {[lx,Lx]}',flush=True)
print(f'Coord bounds along y-direction: {[ly,Ly]}',flush=True)

tree = cKDTree(supercell)

nradii = 1000
rmax = 350
print(f'Max sample window size = {rmax}', flush=True)
print(f'Number of radii = {nradii} --> dr = {(rmax-1)/nradii}',flush=True)



M = int(nradii/nprocs)

if rank < nprocs - 1:
    radii = np.linspace(1,rmax,nradii)[rank*M:(rank+1)*M]

else:
    radii = np.linspace(1,rmax,nradii)[rank*M:]


dfs = np.zeros(radii.shape[0],dtype=float)
for k,r in enumerate(radii):
    dfs[k] = DensityFluctuationsRS(tree,r,[lx,Lx],[ly,Ly],nsamples)

np.save('dfs-%d.npy'%(rank),dfs) #distinguish the output of each process by its number
np.save('radii-%d.npy'%rank,radii)
