import numpy as np
from scipy.stats import linregress
from scipy.spatial import cKDTree

def FluctuationsGrid(grid,grid_points,grid_tree,L,l,sample_size,fluctuations_type='density'):
    """Measures the density fluctuations of a 2-dimensional square and discrete grid, where the density is sampled
    using a circular window of radius l.
    
    Parameters
    ----------
    
    grid: `numpy.ndarray`, shape=(N,M)
        Manna model grid where element (i,j) (for a 2D grid) corresponds to the number of particles at the site located at the ith row and jth column. 
    grid_points: `numpy.ndarray`, shape=(N*M,2)
        Array containing indices (i,j) labelling the different sites of `grid`.
    grid_tree: `scipy.spatial.cKDTree`
        K-d tree containing the grid points (i.e. `grid_indices`). This argument is passed to `DensityFluctuations`
        to avoid re-building the k-d tree for each `l`.
        *** Could be removed as an argument with judicious caching of the k-d tree (cf LRU cache) ***
    L: `int`
        Size of the grid; e.g. a 2D grid of size L has L*L sites.
    l: `float`
        Lengthscale used for the densities. For a n-dimensional model density = (nb. of particles)/(l**n).
    sample_size: `int`
        Number of data points used to estimate the density standard deviation.
    fluctuations_type: `str`
        Type of fluctuations to be computed. Two valid values: 'number' and 'density'.

    Output
    ------
    
    variance: `float`
        Variance of the density in the input grid."""


    Ns = np.zeros(sample_size,dtype=np.float)


    for k, N in enumerate(Ns):
        center = np.random.random(2)*(L-2*l) + l
        index_list = grid_tree.query_ball_point(center,l)
        sampled_indices = grid_points[index_list]
        sampled_grid = grid[sampled_indices[:,0],sampled_indices[:,1]]
        Ns[k] = np.sum(sampled_grid)

    if fluctuations_type == 'density':
        area = np.pi*l*l
        variance = np.var(Ns/area)
    else:
        variance = np.var(Ns)
    
    return variance


def DensityFluctuationsGrid_vectorised(grid,grid_points,grid_tree,L,l,sample_size,save_rdata=False):
    """Vectorised version of the DensityFluctuationsGrid function defined above.
    STILL IN PROGRESS; only partial vectorisation has been achieved; performance boost remain
    to be measured."""

    Ns = np.zeros(sample_size,dtype=np.float)
    area = np.pi*l*l

    r_data = np.zeros((sample_size,3))
    r_data[:,0] = r

    centers = np.random.rand(sample_size,2) * (L-2*l) + l
    inds_lists = grid_tree.query_ball_point(centers,l)

    r_data[:,1:] = centers

    neighbs_inds = [grid_points[l] for l in inds_lists]
    Ns = np.array([np.sum(M[n[:,0], n[:,1]]) for n in neighbs_inds])

    variance = np.var(Ns/area)

    if save_rdata: return variance, r_data
    else: return variance


def DensityFluctuationsRS(structure_tree,l,xbounds,ybounds,sample_size,return_rdata=False,seed=None, return_insample=False,return_densities=False):
    """Computes the fluctuations in density of a point process resolved in continuous 2D space (as opposed
    to a discrete grid). The point process is sampled using circular windows. 
    
    Parameters
    ----------
    
    structure_tree:`scipy.cKDTree`
        The k-d tree containing the coordinates of the point process under consideration.
    l: `float`
        Radius of the sampling window.
    x_bounds: `np.ndarray`, shape=(2,)
        Set of 2 floats whose first element is the minimum x coordinate of the point process and the second
        is its maximum x coordinate.
    y_bounds: `np.ndarray`, shape=(2,)
        Set of 2 floats whose first element is the minimum y coordinate of the point process and the second
        is its maximum y coordinate.
    sample_size: `int`
        Number of sampling windows used to estimate the density fluctuations.
    return_rdata: `bool`
        If `True`, the function also returns the positions and radii of the sampling windows.
    seed: `None` (default) or `int`
        Seed for RNG that determines window positions
    return_isample: `bool`
        If `True`, also return boolean array `in_sample` of shape (sample_size,Natoms), where `in_sample[n,k] = True` if the nth sample contatins the kth atom.
    return_densities: `bool`
        If `True`, also return the densities from each observation window.

    Output
    ------
    variance: `float`
        Density fluctuations between the different sampled regions of the structure.
    rdata: `np.ndarray`, shape = (`sample_size`,3)
        List of vectors [`l`,x,y] where (x,y) are the coordinates of the sampling widows of (radius `l`)
        used to calculate `variance.`
    """


    if seed is not None:
        np.random.seed(seed)
    
    area = np.pi*l*l

    lx, Lx = xbounds
    ly, Ly = ybounds

    # use vectorised sampling
    centers = np.random.random((sample_size,2))*np.array([(Lx-2*l),(Ly-2*l)]) + l + np.array([lx,ly])
    
    # Determines which optional arrays get returned by the function
    # 0: in_sample
    # 1: densities
    # 2: rdata

    out_save = np.zeros(3,dtype='bool') 


    if return_insample:
        out_save[0] = True
        Natoms = structure_tree.n
        in_sample = np.zeros((sample_size, Natoms),dtype='bool')
        densities = np.zeros(sample_size)
        samples = structure_tree.query_ball_point(centers,l)
        for n,s in enumerate(samples):
            in_sample[n,s] = True 
            densities[n] = len(s) / area
    else:
        densities = np.array([len(ll) for ll in structure_tree.query_ball_point(centers, l)]) / area
        in_sample = 0
    
    if return_densities:
        out_save[1] = True
 
    if return_rdata:
        out_save[2] = True

        radii = np.ones((sample_size,1))*l
        rdata = np.hstack((radii,centers))
    else:
        rdata = 0
    
    variance = np.var(densities)

    if np.any(out_save): 
        out = (in_sample,densities,rdata)
        return variance, tuple([out[k] for k in out_save.nonzero()[0]])
    else:
        return variance

def fit_dfs(radii,dfs,lbounds=None):
    if lbounds is not None:
        lmin = lbounds[0]
        lmax = lbounds[1]
        inds = ((radii >= lmin)*(radii <= lmax)).nonzero()[0]
        radii = radii[inds]
        dfs = dfs[inds]

    lr_obj = linregress(np.log(radii),np.log(dfs))
    a = lr_obj.slope
    b = lr_obj.intercept
    r2 = lr_obj.rvalue**2
    return a, b, r2

def NumberFluctuationsSquareWindow(pos, l, xbounds, ybounds, sample_size):
    """Computes the number fluctuations in 2D system described by positions `pos` (`shape = (N,2)`), when using a square sampling window of side `l`, and collecting n=`sample_sizes` samples. The `bounds` arguments describes the extremal x- and y-coords spanned by the system; `bounds = [xmin, xmax, ymin, ymax]`, and are computed and returned if set to `None` (this behaviour avoids computing the min/max coords along both axes every time this function is called)."""

    xmin, xmax = xbounds
    ymin, ymax = ybounds
    
    # use vectorised sampling
    centers = np.random.random((sample_size,2))*np.array([(xmax-2*l),(ymax-2*l)]) + l + np.array([xmin,ymin])

    nb_in_window = np.zeros(sample_size,dtype=int)

    X = pos[:,0]
    Y = pos[:,1]

    for k, r0 in enumerate(centers):
        x0, y0 = r0
        mask = (np.abs(X-x0) <= l/2) * (np.abs(Y-y0) <= l/2)
        nb_in_window[k] = mask.sum()
    
    return np.var(nb_in_window)
    