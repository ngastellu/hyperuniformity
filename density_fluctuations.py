import numpy as np
from scipy.stats import linregress
from scipy.spatial import cKDTree

def DensityFluctuationsGrid(grid,grid_points,grid_tree,L,l,sample_size):
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

    Output
    ------
    
    variance: `float`
        Variance of the density in the input grid."""


    Ns = np.zeros(sample_size,dtype=np.float)
    area = np.pi*l*l


    for k, N in enumerate(Ns):
        #center = np.random.randint(L,size=2)
        center = np.random.random(2)*(L-2*l) + l
        index_list = grid_tree.query_ball_point(center,l)
        sampled_indices = grid_points[index_list]
        sampled_grid = grid[sampled_indices[:,0],sampled_indices[:,1]]
        Ns[k] = np.sum(sampled_grid)

    variance = np.var(Ns/area)
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


def DensityFluctuationsRS(structure_tree,l,xbounds,ybounds,sample_size,save_rdata=False):
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
    save_rdata: `bool`
        If `True`, the function also returns the positions and radii of the sampling windows.

    Output
    ------
    variance: `float`
        Density fluctuations between the different sampled regions of the structure.
    rdata: `np.ndarray`, shape = (`sample_size`,3)
        List of vectors [`l`,x,y] where (x,y) are the coordinates of the sampling widows of (radius `l`)
        used to calculate `variance.`
    """

    area = np.pi*l*l

    lx, Lx = xbounds
    ly, Ly = ybounds

    # use vectorised sampling
    centers = np.random.random((sample_size,2))*np.array([(Lx-2*l),(Ly-2*l)]) + l + np.array([lx,ly])
    densities = np.array([len(ll) for ll in structure_tree.query_ball_point(centers, l)]) / area

    variance = np.var(densities)
    
    if save_rdata:
        radii = np.ones((sample_size,1))*l
        rdata = np.hstack((radii,centers))
        return variance, rdata
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