#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt
from density_fluctuations import DensityFluctuationsRS
from scipy.spatial import cKDTree




L = 1000

points = np.random.random((10000, 2))*L
tree = cKDTree(points)

radii = np.linspace(1,2000,2000)
dfs = np.zeros(2000,dtype=np.float)

for k, r in enumerate(radii):
    print(k)
    dfs[k] = DensityFluctuationsRS(tree,r,[0,L],[0,L],10)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(radii,dfs,'ro',ms=3.0)
ax.plot(radii,radii**(-2),'k--',lw=0.7)
plt.show()

