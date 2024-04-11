#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt

#npy_filename = input('Name of .npy file to plot: ')
directory = 'beluga_outputs/'
npy_filename = directory + 'query_times_data_kdtree.npy'
npy_fn2 = directory + 'avg_times_data_kdtree.npy'

query_times = np.load(npy_filename)
iter_times = np.load(npy_fn2)


plt.plot(query_times[0],query_times[1],'r-',lw=0.8,label='query')
plt.plot(iter_times[0],iter_times[1],'b-',lw=0.8,label='full iteration')
plt.xlabel('Iteration number')
plt.ylabel('Time')
plt.legend()
plt.show()
