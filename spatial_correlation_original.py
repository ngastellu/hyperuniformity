#!/usr/bin/env pythonw  

# compute the pair correlation and radial density functions for a given sample
import numpy as np
import matplotlib.pyplot as plt


#HERE IS WHERE YOU LOAD THE DATASET - 4D [SAMPLE #, 0, Y, X]


# example data - one image with rings
sample = np.zeros((1,1,256,256)).astype('uint8')
x0, y0 = [sample.shape[-2]//2 - 1, sample.shape[-1]//2 - 1]

for i in range(256):
    for j in range(256):
        radius = np.sqrt((i - y0) ** 2 + (j - x0) ** 2)
        if np.round(radius)%8 == 0:
            sample[0,0,i,j] = 1

sample[0,0,y0,x0] = 1


#preprocess sample
max_rad = sample.shape[2] // 2 - 1 # the radius to be explored is automatically set to the maximum possible for the sample image
nbins = max_rad * 10 # set number of bins for sorting
box_size = 2 * max_rad + 1 # size of box for radial searching
x0, y0 = [sample.shape[-2]//2 - 1, sample.shape[-1]//2 - 1] # pick a nice central pixel
sample = sample[sample[:,:,y0,x0]!=0] # delete samples with zero particles at centre (a waste, I know, but you can always just feed it more samples, or get rid of this if you don't need a central particle)
sample = sample[:,y0-max_rad:y0+max_rad+1, x0-max_rad:x0+max_rad+1] # adjust sample size

# prep radial bins
a, bins = np.histogram(1, bins = nbins, range = (.01, max_rad + 0.01)) # bin the possible radii
circle_square_ratio = np.pi/4  # radio of circle to square area with equal radius

# prep constants
dr = bins[1]-bins[0] # differential radius
N_i = sample.shape[0]  # number of samples
N_tot = np.sum(sample)*circle_square_ratio - N_i # total particle number adjusted for a circular radius and subtracting the centroid
rho = np.average(sample)  # particle density

# initialize outputs
radial_corr = np.zeros(nbins) # radial density
radial_corr2 = np.zeros(nbins) # radial pair-correlation
corr = np.zeros((box_size, box_size)) # 2D density
corr2 = np.zeros((box_size, box_size)) # 2D pair-correlation

# for each pixel within a square box of the appropriate size, assign a radius, coordinates and check its occupancy
for i in range(box_size): # for a box of radius max_rad around x0, y0
    for j in range(box_size):
        if (i != y0) or (j != x0):

            radius= np.sqrt((i - y0) **2 + (j - x0) ** 2)
            corr[i, j] = np.sum(sample[:, i, j]) # density distribution
            corr2[i, j] = corr[i, j] / (radius) # pair-correlation

            if radius <= max_rad: # if we are within the circle drawn over the square
                bin_num = np.digitize(radius, bins) - 1  # add to bin
                radial_corr[bin_num] += corr[i, j]
                radial_corr2[bin_num] += corr2[i, j]


bin_rad = np.zeros(len(bins)-1)
bin_rad = np.zeros(len(bins)-1)

for i in range(len(bins)-1):
    bin_rad[i] = (bins[i] + bins[i+1]) / 2 #assign a radius to each bin

radial_corr2 = radial_corr2 / (2 * np.pi * dr * rho * N_i) # normalize the pair-correlation function

#compute rolling means for correlation functions
rolling_mean = np.zeros(len(radial_corr2))
rolling_mean2 = np.zeros(len(radial_corr))
run = int(nbins // 10 * 2) # length of rolling mean
for i in range(run,len(radial_corr2)):
    rolling_mean[i] = np.average(radial_corr2[i-run:i])
    rolling_mean2[i] = np.average(radial_corr[i-run:i])

# average out the central points for easier graph viewing
corr[y0,x0] = -np.average(corr)
corr2[y0,x0] = -np.average(corr2)

plt.figure()
plt.subplot(2,3,1)
corr[y0,x0]=np.average(corr)
plt.imshow(corr)
plt.subplot(2,3,5)
plt.plot(bin_rad, radial_corr2 ,'-')
plt.subplot(2,3,2)
plt.plot(bin_rad, radial_corr , '-')
plt.subplot(2,3,4)
corr2[y0,x0]=np.average(corr2)
plt.imshow(corr2)
plt.subplot(2,3,6)
plt.plot(bin_rad, rolling_mean, '.')
plt.subplot(2,3,3)
plt.plot(bin_rad, rolling_mean2, '.')
plt.show()
