#!/usr/bin/env python

import numpy as np
from random_organisation import RandomOrgModel1D
from time import perf_counter

L = 10000
rho = 0.23
l = 1
eps = 0.25
rmax_sampling = 3000


np_realisations = 50

realisation_time = 0
dyn_time = 0
df_time = 0
rho_vars = np.zeros(rmax_sampling-1,dtype=float)
niter = 0

for i in nb_realisations:

    start_tot = perf_counter()
    
    model = RandomOrgModel1D(L,rho,l,eps)
    
    dyn_start = perf_counter()

    niter += model.run_dynamics(return_nsteps=True)

    dyn_end = perf_counter()

    rho_vars = model.density_fluctuations_array(rmax_sampling)

    end_tot = perf_counter()
    print('Done with realisation %d; ran in %f seconds.'%(i,end_tot-start_tot))
    realisation_time += end_tot - start_tot
    dyn_time += dyn_end - dyn_start
    df_time += end_tot - dyn_end


np.save('rand_org_1d_avg_density_fluctuations.npy',rho_vars)

end = perf_counter()

print('Script finished running in %f seconds.\n\
    Average number of iterations before absorbtion = %d\n\
    Average dynamics time = %f seconds\n\
    Average density fluctuations time = %f seconds\n\
    Average total realisation time = %f seconds.'
    %(end-start,
        niter/nb_realisations,
        dyn_time/nb_realisations,
        df_time/nb_realisations,
        reaslisation_time/nb_realisations))
