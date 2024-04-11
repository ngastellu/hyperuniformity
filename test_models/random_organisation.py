#!/usr/bin/env python

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gamma
from time import perf_counter

def nball_volume(r,dim):
    if dim == 1: return 2*r
    elif dim == 2: return np.pi*(r**2)
    elif dim == 3: return (4/3)*np.pi*(r**3)
    else: return (r**dim) * ( np.sqrt(np.pi**dim) / gamma(1+(dim/2)) )


class RandomOrgModel:

    def __init__(self,dimension,size,density,radius,max_displacement):

        assert dimension in [1,2,3], 'RandOrg only implemented for up to 3 dimensions.'

        self.L = size
        self.dim = dimension
        self.rho = density
        self.l = radius
        self.eps = max_displacement

        N = int(size * density)

        if dimension > 1:
            self.positions = np.random.random_sample((N,dimension))*size
            self.KDtree = cKDTree(self.positions,boxsize=size)
        else:
            self.positions = np.random.random_sample(N)*size


    def active_particle_indices(self,save_degeneracies=False):

        if self.dim > 1:
            if save_degeneracies:
                unique_active_indices, counts = np.unique(self.KDtree.query_pairs(self.l,output_type='ndarray'), return_counts=True)
                return unique_active_indices, counts

            else:
                unique_active_indices = np.unique(self.KDtree.query_pairs(self.l,'ndarray'))
                return unique_active_indices

        else:
            dists = np.array([np.abs(self.positions - x) for x in self.positions])
            bool_array = dists < self.l
            if save_degeneracies:
                counts_full = np.sum(bool_array,axis=1)
                unique_active_indices = counts_full.nonzero()[0]
                counts = counts_full[unique_active_indices]
                return unique_active_indices, counts


            else:
                unique_active_indices = np.any(bool_array,axis=1).nonzero()[0]
                return unique_active_indices


    def run_dynamics(self,accumulate_impacts=False,max_iter=np.inf,return_nsteps=False):
        
        if accumulate_impacts:
            active_particles, counts = self.active_particle_indices(save_degeneracies=True)
        else:
            active_particles = self.active_particle_indices()

        n_active = len(active_particles)
        absorbed = n_active == 0

        iter_counter = 0

        while (not absorbed) and iter_counter < max_iter:
            if self.dim > 1:
                rs = np.random.random(n_active)*self.eps
                thetas = np.random.random(n_active)*2*np.pi
                xs = rs * np.cos(thetas)
                ys = rs * np.sin(thetas)
                displacements = np.vstack([xs,ys]).T
                if self.dim == 3:
                    phis = np.random.random(n_active)*np.pi
                    displacements *= np.sin(phis)
                    zs = rs * np.cos(phis)
                    displacements = np.hstack([displacements,zs.reshape(rs.shape[0],1)])
            
            else:
                displacements = (2*np.random.random(n_active)-1)*self.eps
            
            if accumulate_impacts:
                self.positions[active_particles] += counts*displacements
                active_particles, counts = self.active_particle_indices(save_degeneracies=True)
            else:
                self.positions[active_particles] += displacements
                active_particles = self.active_particle_indices()

            if self.dim > 1:
                self.KDtree = cKDTree(self.positions,boxsize=self.L)


            n_active = len(active_particles)
            absorbed = n_active == 0
            iter_counter += 1

        if return_nsteps: return iter_counter


        def density_fluctuations_array(self,rmax,num_rs=int(rmax),num_samples=100,return_radii=False):

            radii = np.linspace(1,rmax,num_rs)
            rho_vars = np.zeros(radii.shape[0],dtype=float)

            # ****TEST IDEA: check if using boolean algebra on np.ndarrays is faster than cKDTree approach****
            for r, sigma in zip(radii,rho_vars):
                volume = nball_volume(r,self.dim)
                sample_centers = np.random.random(num_samples)*self.L
                if self.dim > 1:
                    rhos = self.KDtree.query_ball_point(sample_centers,r,return_length=True) / volume
                else:
                    rhos=np.array([np.sum(np.abs(self.positions-x) <= r) for x in sample_centers])/volume 
                sigma = np.var(rhos)

            if return_radii: return rho_vars, radii
            else: return rho_vars
