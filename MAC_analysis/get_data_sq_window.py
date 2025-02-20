#!/usr/bin/env python 

import numpy as np
import os
from glob import glob


def orientation_avg_single_sample(isample):
    print(f'\n{isample}', end= ' ', flush=True)
    try:
        nfs = np.vstack([np.load(f'job-{isample}/nfs-{n}_pbc.npy') for n in range(2)])
        return np.mean(nfs,axis=0)
    except Exception as e:
        print(e)
        return None

def get_rot45_nfs(isample, i45=45): # focus on 45 degree rotation beacause it gives marked hyperfluctuations in my square lattice test.
    try:
        nfs = np.load(f'job-{isample}/nfs-0_pbc.npy')
    except Exception as e:
        print(e)
        return None

    if i45 is None:
        angles = np.load(f'job-{isample}/rotation_angles_degrees-0.npy')
        i45 = (angles == 45).nonzero()[0][0]
    return nfs[i45,:]

ijobs = [d.split('-')[1] for d in glob('job-*')]

structype = os.path.basename(os.getcwd())

avg_nfs = orientation_avg_single_sample(ijobs[0])
avg_nfs_45deg = get_rot45_nfs(ijobs[0],i45=45)

N1 = 0
N2 = 0

for n in ijobs[1:]:
    avg_nfs_sample = orientation_avg_single_sample(n)
    if avg_nfs_sample is not None:
        avg_nfs += avg_nfs_sample
        N1 += 1
    avg_nfs_45deg_sample = get_rot45_nfs(n, i45=45)
    if avg_nfs_45deg_sample is not None:
        avg_nfs += avg_nfs_sample
        N2 += 1

avg_nfs /= N1
avg_nfs_45deg /= N2

np.save(f'avg_nfs_rot_sq_window_{structype}.npy', avg_nfs)
np.save(f'avg_nfs_rot_sq_window_45deg_{structype}.npy', avg_nfs_45deg)