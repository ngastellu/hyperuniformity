#!/usr/bin/env python

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
from qcnico.plt_utils import setup_tex



L = 1000
radii = np.arange(1,501)
avg_dfs = np.load('/Users/nico/Desktop/simulation_outputs/hyperuniformity/avg_dfs_random.npy')
ending_inds = np.arange(10,499)

slopes = np.zeros(ending_inds.shape)

for k, n in enumerate(ending_inds):
    slopes[k] = linregress(np.log(radii[:n]), np.log(avg_dfs[:n])).slope


setup_tex()
plt.plot(radii[ending_inds]/L, slopes, '-o', ms=0.5)
plt.axhline(y=-2,xmin=0,xmax=1,color='r',lw=0.8,label='$\lim_{L\\rightarrow\infty}\lambda = -2$ (exact)')
plt.xlabel('Upper bound of fit region $\ell_{\\text{max}}/L$')
plt.ylabel('Fit exponent $\lambda$')
plt.suptitle('Finite-size effect on the estimate of the $\sigma^2(\ell)\sim\ell^\lambda$ fit')
plt.legend()
plt.show()