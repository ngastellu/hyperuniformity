#!/usr/bin/env python

import sys
import os
from glob import glob
import numpy as np
from scipy.spatial import KDTree
from density_fluctuations import NumberFluctuationsRS

def grid2realspace(grid,occ_value,latt_vecs=None):
    """Given a 2D array `grid` , return the positions of the elements whose value is `occ_value`,
    rescaled by `latt_vecs` (matrix of column vectors). For instance, if `grid[i,j] = occ_value`, then 
    `i * latt_vecs[:,0] + j * latt_vecs[:,1]` will be in the output array."""

    iocc = np.vstack((grid == occ_value).nonzero())
    
    if latt_vecs is not None:
        return (latt_vecs @ iocc).T # return as row vectors
    else:
        return iocc.T
    



ijob = int(sys.argv[1])
grid = np.load(f'random_grid-{ijob}.npy')
pos = grid2realspace(grid,1)

nsamples = 50

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

eps = 1.0 # additional 'whitespace' between cell and its periodic image

print(f'Coord bounds along x-direction: {[lx,Lx]}',flush=True)
print(f'Coord bounds along y-direction: {[ly,Ly]}',flush=True)

tree = KDTree(pos,boxsize=[Lx+eps,Ly+eps])

nradii = 1200
#rmax = np.min(tree.maxes)/5.0
rmax = 100

radii = np.logspace(0,np.log10(rmax),nradii,base=10)

print(f'Max sample window size = {rmax}', flush=True)
print(f'Number of radii = {nradii} --> dr = {(rmax-1)/nradii}',flush=True)


dfs = np.zeros(radii.shape[0],dtype=float)
rdata = np.zeros((radii.shape[0]*nsamples,3))
all_rhos = np.zeros((radii.shape[0],nsamples))
all_sample_masks = np.zeros((radii.shape[0]*nsamples,pos.shape[0]),dtype='bool')
for k,r in enumerate(radii):
    dflucs, (in_samp, rhos, rdat) = NumberFluctuationsRS(tree,r,[lx,Lx],[ly,Ly],nsamples,return_rdata=True,return_insample=True,return_densities=True)
    all_sample_masks[nsamples*k:nsamples*(k+1),:] = in_samp
    dfs[k] = dflucs
    all_rhos[k,:] = rhos
    rdata[nsamples*k:nsamples*(k+1)] = rdat

np.save('dfs_realspace_pbc.npy',dfs) #distinguish the output of each process by its number
np.save('radii_realspace_pbc.npy',radii)
np.save('rdata_realspace_pbc.npy', rdata)
np.save('densities_realspace.npy',all_rhos)
np.save('sample_masks_realspace.npy', all_sample_masks)
