#!/usr/bin/env python

import numpy as np
import os
import sys

def sample_crystallinity(samp_mask,cryst_mask):
    """Returns fraction of crystalline atoms in sampling window. Both `cryst_mask` and `samp_mask` 
    are boolean arrays so multiplying them amounts to a logical AND operation.
    Also works for an array of sample window masks, stacked row wise (i.e. `samp_mask[j]` contains
    the mask corresponding to the jth sampling window."""
    return (samp_mask * cryst_mask).sum(axis=1) / samp_mask.sum(axis=1)


structype = os.path.basename(os.getcwd())
nprocs_mpi = 2

if structype == '40x40':
    lbls = np.arange(1,300)
elif structype == 'tempdot6':
    lbls = np.arange(218)
elif structype == 'tempdot5':
    lbls = np.arange(217)
else:
    print(f'{structype} is not a valid ensemble name.')
    sys.exit(1)

cryst_dir = os.path.expanduser(f'~/scratch/structural_characteristics_MAC/labelled_ring_centers/{structype}/')

for n in lbls:
    print(n)
    cryst_mask = np.load(os.path.join(cryst_dir, f'sample-{n}/crystalline_atoms_mask-{n}.npy'))
    densities = np.hstack([np.load(f'job-{n}/densities-{k}.npy').ravel() for k in range(nprocs_mpi)])
    samp_masks = np.vstack([np.load(f'job-{n}/sample_masks-{k}.npy') for k in range(nprocs_mpi)])
    samp_crysts = sample_crystallinity(samp_masks,cryst_mask)
    out_data = np.vstack((densities,samp_crysts))
    np.save(f'job-{n}/rho_sampcryst-{n}.npy', out_data)