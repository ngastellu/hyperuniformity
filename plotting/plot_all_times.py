#!/usr/bin/env pythonw

import numpy as np
import matplotlib.pyplot as plt

prefixes = ['avg','query','l1','l2','sum']

data = []

for pref in prefixes:
    filename =  pref + '_times_data_kdtree_new.npy'
    data.append(np.load(filename))

colours = ['#ee2c2c','#2cee2c','#2c2cee','#8d2cee','#8dee2c']

for k,dat in enumerate(data):
    plt.plot(*dat,ls='-',c=colours[k],lw=0.8,label=prefixes[k])

#plt.plot(data[0][0],data[3][1]+data[4][1],c='#ff34b3',ls='-',lw=0.8,label='l2+sum')

plt.legend()
plt.show()
