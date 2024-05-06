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

rho = 65000 / (400*400) #approximate nb of atoms per angstrom^2 in MAC

a = int(sys.argv[2]) #size of unit cell
l = int(sys.argv[3]) #span of random coords in unot cell
#n = int(rho * a * a) #number of atoms per cell
n = int(rho * l * l) #number of atoms per cell

print(f'******* Points per cell = {n} *******')

nsamples = 50
ncells = 100


latt_vecs = np.eye(2) * a

cell = np.random.rand(2,n) * l

supercell = make_supercell(cell,latt_vecs,ncells,ncells)

supercell = supercell.T

np.save(f'random_cell_n{n}_l{l}_a{a}_seed{nn}.npy', supercell)


lx = np.min(supercell[:,0])
Lx = np.max(supercell[:,0])

ly = np.min(supercell[:,1])
Ly = np.max(supercell[:,1])

print(f'Coord bounds along x-direction: {[lx,Lx]}',flush=True)
print(f'Coord bounds along y-direction: {[ly,Ly]}',flush=True)

tree = cKDTree(supercell)

rmax = (ncells * a) / 3
nradii = 5 * rmax
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

np.save('dfs_a%d_l%d_n%d-%d.npy'%(a,l,n,rank),dfs) #distinguish the output of each process by its number
np.save('radii_a%d_l%d_n%d-%d.npy'%(a,l,n,rank),radii)
