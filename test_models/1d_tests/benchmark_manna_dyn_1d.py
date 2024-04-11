#!/usr/bin/env python

import numpy as np
from time import perf_counter

global_start = perf_counter()

size = 100000

rho = 1.671

rmax = 25000
radii = np.arange(1,rmax)

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
max_iter = 1000000
active_bools = grid >= activity_threshold
not_absorbed = np.any(active_bools)
nb_iterations = 0

total_iter_times = np.zeros(max_iter,dtype=float)
avg_binom_times = np.zeros(max_iter,dtype=float)
avg_update_times = np.zeros(max_iter,dtype=float)
active_check_times = np.zeros(max_iter,dtype=float)

dynamics_start = perf_counter()

while not_absorbed or nb_iterations < max_iter:
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
        update_time += sample_end - update_end

 
    #check which sites are active
    check_start = perf_counter()
    active_bools = grid >= activity_threshold
    not_absorbed = np.any(active_bools)
    check_end = perf_counter()

    iter_end = perf_counter()
    total_iter_times[nb_iterations] = iter_end - iter_start
    avg_binom_times[nb_iterations] = binom_time/len(active_sites)
    avg_update_times[nb_iterations] = update_time/len(active_sites)
    active_check_times[nb_iterations] = check_end-check_start

    nb_iterations += 1
    
    if nb_iterations == max_iter:
        fo.write('Max number of iterations (%d) reached. Terminating dynamics.'%max_iter)

dynamics_end = perf_counter()

dyn_time = dynamics_end - dynamics_start
print('Dynamics ended; took %f seconds. Now computing density fluctuations.\n'%dyn_time,flush=True)

np.save('iter_times_1d.npy',total_iter_times)
np.save('binom_avg_times_1d.npy',avg_binom_times)
np.save('update_avg_times_1d.npy',avg_update_times)
np.save('check_active_times_1d.npy',active_check_times)

global_end = perf_counter()
print('Running this script took {}s.'.format(global_end - global_start))
