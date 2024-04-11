#!/usr/bin/env python

import subprocess as sbp
import numpy as np
from glob import glob

fs = glob('*.npy')

for f in fs:
    
    N = np.load(f).shape[0]

    if N in [10000, 10002]: nprocs = 40
    else: nprocs = 10

    sys_name = f.split('.')[0]

    fin = open('submit.template')
    fout = open('submit_%s.sh'%sys_name, 'w')
    p1 = sbp.Popen('sed -e \'s/##NPROCS##/%d/g\' -e \'s/##SYSNAME##/%s/g\''%(nprocs,sys_name),shell=True,stdin=fin,stdout=fout)
    p1.wait()

    sbp.run('sbatch submit_%s.sh'%(sys_name,f),shell=True)
