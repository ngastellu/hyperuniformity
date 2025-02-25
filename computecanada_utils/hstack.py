#!/usr/bin/env python

import numpy as np
from glob import glob
import os


outdir = 'hstacked_data'
if not os.path.isdir(outdir):
    os.makedirs(outdir)

nprocs = 2

ijobs = np.sort([int(dd.split('-')[1]) for dd in glob('job-*')])

for n in ijobs:
    print(n,flush=True)
    flucs = np.hstack([np.load(os.path.join('job-%d'%n,'nfs-%d_pbc.npy'%k)) for k in range(nprocs)])
    np.save(os.path.join(outdir, 'nfs-%d.npy'%n), flucs)