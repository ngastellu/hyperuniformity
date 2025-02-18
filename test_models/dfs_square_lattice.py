#!/usr/bin/env python

import sys
import os
from glob import glob
import numpy as np
from scipy.spatial import KDTree
from density_fluctuations import NumberFluctuationsRS
from qcnico.rotate_pos import rotate_pos
from qcnico.lattice import cartesian_product

def square_lattice(a, n):
    points_1d = np.arange(n) * a
    return cartesian_product(points_1d,points_1d)



# rCC = 1.8 # max C-C 'bond length' (i.e. nearest-neigbour distance)
eps = 1.0 # additional 'whitespace' between cell and its periodic image

nsamples = int(sys.argv[1])

a = 1.0 # lattice constant
n = 400 # nb. of points in x and y direction

pos = square_lattice(a,n)


nradii = 500
lmax = 100

radii = np.linspace(1,lmax,nradii) 

nfs = np.zeros(nradii,dtype=float) # 1 row <---> 1 angle; 1 column <---> 1 side length


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

tree = KDTree(pos,boxsize=[Lx+eps,Ly+eps])

print(f'Coord bounds along x-direction: {[lx,Lx]}',flush=True)
print(f'Coord bounds along y-direction: {[ly,Ly]}',flush=True)


for k,l in enumerate(radii):
    nflucs = NumberFluctuationsRS(tree,l,[lx,Lx],[ly,Ly],nsamples)
    nfs[n,k] = nflucs

# avg_nfs = np.mean(nfs,axis=0) # perform average over sampled orientations
np.save('nfs_pbc.npy', nfs) #distinguish the output of each process by its number
np.save('radii_pbc.npy',radii)