#!/usr/bin/env pythonw

import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt

r = np.log(np.arange(1,30000))
dfs = np.log(np.load('avg_density_fluctuations_manna_1d.npy')[-1])

slope, intercept, rval, pval, stderr = linregress(r,dfs)

print('Fit found.\n\
        Slope = %f\n\
        Intercept = %f\n\
        rval = %f\n\
        pval = %f\n\
        stderr = %f\n\
        '%(slope, intercept, rval, pval,stderr))

plt.plot(r,dfs,'r-',ms=0.5)
plt.plot(r,intercept + slope*r,'k-',lw=0.8)
plt.show()

