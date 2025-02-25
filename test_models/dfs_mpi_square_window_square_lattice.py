#!/usr/bin/env python

import sys
import os
from glob import glob
import numpy as np
from scipy.spatial import KDTree
from density_fluctuations import NumberFluctuationsSquareWindow
from mpi4py import MPI
from qcnico.rotate_pos import rotate_pos
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

all_thetas_deg = np.arange(360)
M = all_thetas_deg.shape[0] // nprocs

if rank < nprocs:
    thetas_deg = all_thetas_deg[rank*M:(rank+1)*M]
else:
    thetas_deg = all_thetas_deg[rank*M:]


n_side_lengths = 500
lmax = 100
print(f'Max sample window size = {lmax}', flush=True)
print(f'Number of radii = {n_side_lengths} --> dr = {(lmax-1)/n_side_lengths}',flush=True)

side_lengths = np.linspace(1,lmax,n_side_lengths) 

nfs = np.zeros((M,side_lengths.shape[0]),dtype=float) # 1 row <---> 1 angle; 1 column <---> 1 side length

for n,theta in enumerate(thetas_deg):
    theta_rad = theta * np.pi/180
    pos = rotate_pos(pos,theta_rad)

    lx = np.min(pos[:,0])
    ly = np.min(pos[:,1])

    Lx = np.max(pos[:,0])
    Ly = np.max(pos[:,1])


    print(f'Coord bounds along x-direction: {[lx,Lx]}',flush=True)
    print(f'Coord bounds along y-direction: {[ly,Ly]}',flush=True)


    for k,l in enumerate(side_lengths):
        nflucs = NumberFluctuationsSquareWindow(pos,l,[lx,Lx],[ly,Ly],nsamples,restrict_centres=False)
        nfs[n,k] = nflucs


# avg_nfs = np.mean(nfs,axis=0) # perform average over sampled orientations
np.save('rotated_sq_window/nfs-%d_pbc.npy'%(rank), nfs) #distinguish the output of each process by its number
np.save('rotated_sq_window/side_lengths-%d_pbc.npy'%rank,side_lengths)
np.save('rotated_sq_window/rotation_angles_degrees-%d.npy'%rank,thetas_deg)