"""Extract some features.
"""

import math
import numpy as np

from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree


def _pca(data):
    """Carry out a PCA on a set of 3D points

    Parameters
    ----------
    data : np.array
        Raw data to which a PCA must be applied

    Returns
    -------
    sklearn.decomposition.PCA
        Principle component analysis done on input data
    """
    return PCA(n_components=3).fit(data)


def standard_normalization(sample):
    """Normalize a set of points regarding mean and standard deviation

    Parameters
    ----------
    sample : np.array
        Set of points to normalized; must be a 2D-shaped np.array

    Returns
    -------
    np.array
        Normalized (2D-shaped) set of points
    """
    return (sample - sample.mean(axis=0)) / sample.std(axis=0)

def normalize_over_1(l):
    """Normalized a list of values so as to get new values comprised between 0
    and 1, and such that `sum(new_values)==1`

    Parameters
    ----------
    l : list
        Values that must be normalized

    Returns
    -------
    list
        Normalized values
    """
    return [item / sum(l) for item in l]

def features3d(eigenvalues):
    """Compute barycentric coordinates of a point within the explained variance
    space, knowing the PCA eigenvalues

    See Brodu, N., Lague D., 2011. 3D Terrestrial lidar data classification of
    complex natural scenes using a multi-scale dimensionality criterion:
    applications in geomorphology. arXiV:1107.0550v3.

    Extract of C++ code by N. Brodu:

        // Use barycentric coordinates : a for 1D, b for 2D and c for 3D
        // Formula on wikipedia page for barycentric coordinates
        // using directly the triangle in %variance space, they simplify a lot
        //FloatType c = 1 - a - b; // they sum to 1
        a = svalues[0] - svalues[1];
        b = 2 * svalues[0] + 4 * svalues[1] - 2;

    Parameters
    ----------
    eigenvalues : list
        Normalized eigenvalues given by the neighborhood PCA

    Returns
    -------
    list
        3D features that express the point within the variance space
    """
    a = eigenvalues[0] - eigenvalues[1]
    b = 2 * eigenvalues[0] + 4 * eigenvalues[1] - 2
    return [a, b]


def build_neighborhood(point, nb_neighbors, kd_tree):
    """Extract a point neighborhood by the way of a KDTree method

    Parameters
    ----------
    point : np.array
        Coordinates of the reference point (x, y, z)
    nb_neighborhood : int
        Number of neighboring points to consider
    tree : sklearn.neighbors.kd_tree.KDTree
        Tree representation of the point cloud

    Returns
    -------
    dict
        Neighborhood, decomposed as a mean distance between the reference point
    and its neighbors and an array of neighbor indices within the point cloud

    """
    dist, ind = kd_tree.query(np.expand_dims(point, 0), k=nb_neighbors)
    return {"distance": dist.squeeze(), "indices": ind.squeeze()}


def compute_3D_properties(neighboring_distances):
    """Compute some geometric properties of a local point cloud

    See: Martin Weinmann, Boris Jutzi, Stefan Hinz, Clément Mallet,
    2015. Semantic point cloud interpretation based on optimal neighborhoods,
    relevant features and efficient classifiers. ISPRS Journal of
    Photogrammetry and Remote Sensing, vol 105, pp 286-304.

    Parameters
    ----------
    neighboring_distances : np.array
        Distance of each neighboring points to the reference point

    Returns
    -------
    list
        3D geometric properties
    """
    radius = max(neighboring_distances)
    max_difference = np.nan
    std_deviation = np.nan
    density = (len(neighboring_distances) + 1) / ((4/3) * math.pi * radius)
    verticality = np.nan
    return [radius, max_difference, std_deviation, density, verticality]


def compute_3D_features(lbda):
    """Build the set of 3D features for a typical 3D point within a local
    neighborhood represented through PCA eigenvalues

    Parameters
    ----------
    lbda : np.array
        Eigenvalues of a point neighborhood

    """
    e = normalize_over_1(lbda)
    curvature_change = e[2]
    linearity = (e[0] - e[1]) / e[0]
    planarity = (e[1] - e[2]) / e[0]
    scattering = e[2] / e[0]
    omnivariance = (e[0] * e[1] * e[2]) ** (1/3)
    anisotropy = (e[0] - e[2]) / e[0]
    eigenentropy = -1 * np.sum([i * math.log(i) for i in e])
    eigenvalue_sum = np.sum(lbda)
    return [curvature_change, linearity, planarity, scattering,
            omnivariance, anisotropy, eigenentropy, eigenvalue_sum]


def generate_features(point_cloud, nb_neighbors, nb_points=None,
                      kdtree_leaf_size=1000):
    """Build the point features for all (or for a sample of) points within
    the point cloud

    Parameters
    ----------
    point_cloud : np.array
        Coordinates of all points within the point cloud; must be a 2D-shaped
    array with `point_cloud.shape[1] == 3`
    nb_neighbors : int
        Number of points that must be consider within a neighborhod
    nb_points : int
        Number of sample points to consider; if None, all the points are
    considered
    kdtree_leaf_size : int
        Size of each kd-tree leaf (in number of points)

    """
    if nb_points is None:
        sample_mask = range(point_cloud.shape[0])
    else:
        sample_mask = np.random.choice(np.arange(point_cloud.shape[0]),
                                       size=nb_points,
                                       replace=False)
    sample = (point_cloud[idx] for idx in sample_mask)
    kd_tree = KDTree(point_cloud[:, :3], leaf_size=kdtree_leaf_size)
    for point in sample:
        xyz_data, rgb_data = point[:3], point[3:]
        neighborhood = build_neighborhood(xyz_data, nb_neighbors, kd_tree)
        neighbors = point_cloud[neighborhood["indices"], :3]
        lbda = _pca(neighbors).singular_values_
        yield (features3d(lbda)
               + compute_3D_properties(neighbors[:,2], neighborhood["distance"])
               + compute_3D_features(lbda)
               + (rgb_data/255).tolist())
