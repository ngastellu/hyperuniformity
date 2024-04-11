#!/usr/bin/env pythonw

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from os import path
from glob import glob

def moving_average(x, w):
    """Compute the moving average of array x with a sampling window of width w.
    Copied and pasted from: 
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy"""
    
    return np.convolve(x, np.ones(w), 'valid') / w

datadir = '/Users/nico/Desktop/McGill/Research/simulation_outputs/hyperuniformity/bigMAC/'

#rfiles = glob(path.join(datadir,'radii_*.npy'))
#sys_names = [('_').join(path.basename(rf).split('_')[1:]).split('.')[0] for rf in rfiles]

rcParams['text.usetex'] = True
rcParams['font.size'] = 25
rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


#rfile = path.join(datadir,'avg_dfs_40x40_MAC.npy')
flucfile = path.join(datadir,'avg_dfs_bigMAC.npy')

fig = plt.figure()
ax = fig.add_subplot(111)

#ls = np.load(rfile)
ls = np.linspace(1,1000,7000)
dfs = np.load(flucfile)



ax.set_xscale('log')
ax.set_yscale('log')


ax.plot(ls*0.2,dfs,'ro', ms=5)


plt.suptitle('Density fluctuations in MAC (40nm $\\times$ 40nm)')
ax.set_xlabel(r'$\ell$ [\AA]')
ax.set_ylabel(r'$\sigma_{\rho}^2(\ell)$')

plt.show()

