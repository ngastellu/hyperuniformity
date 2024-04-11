#!/usr/bin/env python

import numpy as np
from os import path
from glob import glob


datadirs = glob('run-*')

for d in datadir:
    sys_name = d.split('-')[1]
    datafiles = glob(path.join(d,'*npy'))
    data = np.array([np.load(f) for f in datafiles])
    avg = np.mean(data,axis=0)
    print(sys_name)
    print(data.shape)
    print(avg.shape)
    np.save(path.join(d,'%s_avg_dfs.npy'%sys_name), avg)
