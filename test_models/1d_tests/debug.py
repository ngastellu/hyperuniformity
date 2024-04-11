#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt

### Set up grid and randomly occupy certain sites ###

np.random.seed(0)

size = 100000
rho = 1.67
grid = np.zeros(size,dtype=int)
N = int(rho*size)

num_iter = 964 #number of iterations before reaching absorbing state, when seed = 0

bad_indices = np.zeros((num_iter,size),dtype=int)
Ns = np.zeros(num_iter, dtype=int)
deltaNs = np.zeros((num_iter,size),dtype=int)
deltaGs = np.zeros((num_iter,size),dtype=int)
grid_record = np.zeros((num_iter+1,size),dtype=int)

print(N)
occupied_sites = np.random.randint(0,size,N)

#occupy the grid
for site in occupied_sites:
    grid[site] += 1
print(np.sum(grid))

grid_record[0,:] = grid


### Dynamics ###

activity_threshold = 3 #minimum number of particles in an active site
active_bools = grid >= activity_threshold
not_absorbed = np.any(active_bools)
nb_iterations = 0
bad_indices = np.zeros(size)

while not_absorbed:
    Ns[nb_iterations] = np.sum(grid)
    active_sites = active_bools.nonzero()[0]
    occupancies = grid[active_sites]
    nb_shifted_right = np.random.binomial(occupancies,0.5)
    nb_shifted_left = occupancies - nb_shifted_right
    print(nb_iterations)
    print(np.all(nb_shifted_right + nb_shifted_left == occupancies))
    print(np.all(nb_shifted_right + nb_shifted_left == grid[active_sites]))
    print(np.all(occupancies == grid[active_sites]))
    for k,i in enumerate(active_sites):
        conserved  = grid[i] == nb_shifted_right[k] + nb_shifted_left[k]
        if not conserved:
            #print('nah')
            #print(i, grid[i] - nb_shifted_right[k] - nb_shifted_left[k])
            #print('grid[i] = ', grid[i])
            #print('grid[active_sites][k] = ',grid[active_sites][k])
            #print('grid[active_sites[k]] = ',grid[active_sites[k]])
            #print('occupancies[k] = ', occupancies[k])
            #print('right[k] + left[k] = ', nb_shifted_right[k] + nb_shifted_left[k])
            #print('*******')
            #print('shifted to i+1 = ', nb_shifted_right[k])
            #print('shifted to i-1 = ', nb_shifted_left[k])
            deltaNs[nb_iterations,i] = grid[i] - nb_shifted_right[k] - nb_shifted_left[k]
            deltaGs[nb_iterations,i] = grid[i] - occupancies[k]
        #else: print('ye')
        grid[i] = 0
        if i != size - 1:
            grid[i+1] += nb_shifted_right[k]
            grid[i-1] += nb_shifted_left[k]
        else:
            grid[0] += nb_shifted_right[k]
            grid[i-1] += nb_shifted_left[k]
    
    #check which sites are active
    active_bools = grid >= activity_threshold
    not_absorbed = np.any(active_bools)

    #print(np.sum(grid))
    nb_iterations += 1
    grid_record[nb_iterations,:] = grid 

print(nb_iterations)
print('deltaGs == deltaNs: ', np.all(deltaGs==deltaNs))
#np.save('deltaNs.npy',deltaNs)
#np.save('deltaGs.npy',deltaGs)
#np.save('grid_record.npy',grid_record)

plt.plot(range(200),Ns[:200],'r-')
plt.show()
