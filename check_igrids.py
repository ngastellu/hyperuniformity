#!/usr/bin/env python

import numpy as np
from qcnico.lattice import cartesian_product
from density_fluctuations import get_grid_indices


N = 10
grid = np.zeros((N,N))

igrid1 = get_grid_indices(grid)
igrid2 = cartesian_product(np.arange(N),np.arange(N))

print('get_grid_indices: \n', igrid1)
print('cartesian_product: \n', igrid2)


print(np.all(igrid1 == igrid2))
