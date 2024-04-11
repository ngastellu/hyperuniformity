#!/usr/bin/env pythonw

import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt

global_start = perf_counter()

np.random.seed(42)

size = 1000

rho = 1.66

grid = np.zeros(size,dtype=int)
N = int(rho*size)
occupied_sites = np.random.randint(0,size,N)

#occupy the grid
for site in occupied_sites:
    grid[site] += 1

init_end = perf_counter()
init_time = init_end - global_start

### Dynamics ###

activity_threshold = 3 #minimum number of particles in an active site
num_iter = 20444 #for rho=1.66 and seed(42)
active_bools = grid >= activity_threshold
not_absorbed = np.any(active_bools)
nb_iterations = 0

total_iter_times = np.zeros(num_iter,dtype=float)
tot_binom_times = np.zeros(num_iter,dtype=float)
tot_update_times = np.zeros(num_iter,dtype=float)
active_check_times = np.zeros(num_iter,dtype=float)
n_particles = np.zeros(num_iter+1,dtype=int)

n_particles[0] = np.sum(grid)

dynamics_start = perf_counter()

while not_absorbed: #and nb_iterations < max_iter:
    iter_start = perf_counter()
    active_sites = active_bools.nonzero()[0]
    update_time = 0
    binom_time = 0
    for i in active_sites:
        occ = grid[i]
        sample_start = perf_counter()
        nb_shifted_right = np.random.binomial(occ,0.5)
        sample_end = perf_counter()
        binom_time += sample_end - sample_start
        nb_shifted_left = occ - nb_shifted_right
        grid[i] = 0
        if i != size - 1:
            grid[i+1] += nb_shifted_right
            grid[i-1] += nb_shifted_left
        else:
            grid[0] += nb_shifted_right
            grid[i-1] += nb_shifted_left
    
        update_end = perf_counter()
        update_time += update_end - sample_end

    n_particles[nb_iterations+1] = np.sum(grid)

 
    #check which sites are active
    check_start = perf_counter()
    active_bools = grid >= activity_threshold
    not_absorbed = np.any(active_bools)
    check_end = perf_counter()

    iter_end = perf_counter()
    total_iter_times[nb_iterations] = iter_end - iter_start
    tot_binom_times[nb_iterations] = binom_time
    tot_update_times[nb_iterations] = update_time
    active_check_times[nb_iterations] = check_end-check_start

    nb_iterations += 1
    
    #if nb_iterations == max_iter:
    #    print('Max number of iterations (%d) reached. Terminating dynamics.'%max_iter,flush=True)

dynamics_end = perf_counter()

dyn_time = dynamics_end - dynamics_start
print('Dynamics ended; took %d iterations in %f seconds. Now computing density fluctuations.\n'%(nb_iterations,dyn_time),flush=True)

plt.plot(total_iter_times,'k-',label='tot')
plt.plot(tot_binom_times,'r-',lw=0.8,label='binom')
plt.plot(tot_update_times,'b-',lw=0.8,label='update')
plt.plot(active_check_times,'g-',lw=0.8,label='check')
plt.xlim([0,np.argmin(total_iter_times)])
plt.legend()
plt.show()

plt.plot(n_particles,'r-',lw=0.8)
plt.xlim([0,np.argmin(n_particles)])
plt.show()

global_end = perf_counter()
print('Running this script took {}s.'.format(global_end - global_start))
