#!/usr/bin/env python

import numpy as np


rdatadir = '/Users/nico/Desktop/simulation_outputs/hyperuniformity/bigMAC/rdata/tempdot5/'
rdir = '/Users/nico/Desktop/simulation_outputs/hyperuniformity/bigMAC/sample_radii/tempdot5/'
for n in range(217):
    print(n)
    rdata = np.load(rdatadir + f'rdata-{n}.npy')
    radii = rdata[:,0]
    np.save(rdir + f'radii-{n}.npy', radii)


