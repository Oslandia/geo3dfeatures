"""Extract some geometric features associated to 3D point clouds.

Based on two scientific papers:
  - Nicolas Brodu, Dimitri Lague, 2011. 3D Terrestrial lidar data
classification of complex natural scenes using a multi-scale dimensionality
criterion: applications in geomorphology. arXiV:1107.0550.
  - Martin Weinmann, Boris Jutzi, Stefan Hinz, Clément Mallet, 2015. Semantic
point cloud interpretation based on optimal neighborhoods, relevant features
and efficient classifiers. ISPRS Journal of Photogrammetry and Remote Sensing,
vol 105, pp 286-304.
"""


import math
from collections import OrderedDict

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree


def _pca(data, k=3):
    """Carry out a PCA on a set of 2D or 3D points. The number of components
    depends on the point cloud dimensionality

    Parameters
    ----------
    data : numpy.array
        Raw data to which a PCA must be applied
    k : int
        Number of PCA components

    Returns
    -------
    sklearn.decomposition.PCA
        Principle component analysis done on input data
    """
    return PCA(n_components=k).fit(data)


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
    point : numpy.array
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


def compute_3D_properties(z_neighbors, distances):
    """Compute some geometric properties of a local point cloud

    See: Martin Weinmann, Boris Jutzi, Stefan Hinz, Clément Mallet,
    2015. Semantic point cloud interpretation based on optimal neighborhoods,
    relevant features and efficient classifiers. ISPRS Journal of
    Photogrammetry and Remote Sensing, vol 105, pp 286-304.

    Parameters
    ----------
    z_neighbors : numpy.array
        Neighboring point z-coordinates
    distances : numpy.array
        Distance of each neighboring points to the reference point

    Returns
    -------
    list
        3D geometric properties
    """
    radius = np.max(distances)
    z_range = np.ptp(z_neighbors)
    std_deviation = np.std(z_neighbors)
    density = (len(distances) + 1) / ((4/3) * math.pi * radius ** 3)
    verticality = np.nan
    return [radius, z_range, std_deviation, density, verticality]


def compute_3D_features(lbda):
    """Build the set of 3D features for a typical 3D point within a local
    neighborhood represented through PCA eigenvalues

    Parameters
    ----------
    lbda : numpy.array
        Eigenvalues of a point neighborhood

    """
    e = [item / sum(lbda) for item in lbda]
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


def compute_2D_properties(point, neighbors):
    """Compute 2D geometric features according to (Lari & Habib, 2012) quoted
    by (Weinmann *et al.*, 2015)

    For sake of consistency, (Weinmann *et al.*, 2015) uses 3D neighborhood to
    compute these 2D metrics. We apply this hypothesis here.

    Parameters
    ----------
    point : numpy.array
        Reference point 2D-coordinates
    neighbors : numpy.array
        Neighboring point 2D-coordinates (x, y)

    """
    xs, ys = neighbors[:, 0], neighbors[:, 1]
    distances = [math.sqrt((x-point[0])**2 + (y-point[1])**2)
                 for x, y in zip(xs, ys)]
    radius_2D = max(distances)
    density_2D = (len(distances) + 1) / (math.pi * radius_2D ** 2)
    return [radius_2D, density_2D]


def compute_2D_features(lbda):
    """Build the set of 2D features for a typical 2D point within a local
    neighborhood represented through PCA eigenvalues

    Parameters
    ----------
    lbda : numpy.array
        Eigenvalues of a point neighborhood

    """
    eigenvalues_sum_2D = sum(lbda)
    eigenvalues_ratio_2D = lbda[0] / lbda[1]
    return [eigenvalues_sum_2D, eigenvalues_ratio_2D]


def build_accumulation_features(point_cloud, bin_size=0.25, buf=1e-3):
    """Compute accumulation features as a new way of designing a 2D-neighborhood,
    following the description of (Weinmann *et al.*, 2015): such features are
    built by binning the 2D-space, and evaluating the number of points
    contained, the Z-range and the Z-standard deviation in each bin. The
    features are then assigned to the points regarding the bin that they belong
    to.

    Parameters
    ----------
    point_cloud : numpy.array
        Coordinates of all points within the point cloud; must be a 3D-shaped
    bin_size : float
        Size of each squared bin edge
    buf : float
        Epsilon quantity used for expanding the largest bins and consider max
    values

    Returns
    -------
    pandas.DataFrame
        Set of features built through binning process, for each point within
    the cloud

    """
    assert point_cloud.shape[1] == 3
    df = pd.DataFrame(point_cloud, columns=["x", "y", "z"])
    xmin, xmax = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    ymin, ymax = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    xbins = np.arange(xmin, xmax+bin_size+buf, bin_size)
    df["xbin"] = pd.cut(df.x, xbins, right=False)
    ybins = np.arange(ymin, ymax+bin_size+buf, bin_size)
    df["ybin"] = pd.cut(df.y, ybins, right=False)
    aggdf = (df
             .groupby(["xbin", "ybin"])["z"]
             .agg(["count", "min", "max", "std"])
             .reset_index())
    aggdf["z_range"] = aggdf["max"] - aggdf["min"]
    aggdf.drop(["min", "max"], axis=1, inplace=True)
    return (df
            .merge(aggdf, on=["xbin", "ybin"], how="left")
            .drop(["xbin", "ybin"], axis=1))


def retrieve_accumulation_features(point, features):
    """Get the accumulation features for given `point`, by querying the pandas
    dataframe containing the information for every point cloud item.

    Parameters
    ----------
    point : numpy.array
        Coordinates of the point of interest, for identification purpose
    features : pandas.DataFrame
        Collection of point features

    Returns
    -------
    list
        Accumulation features
    """
    point_x, point_y, point_z = point
    point_features = features.query("x==@point_x & y==@point_y & z==@point_z")
    assert point_features.shape[0] == 1
    acc_density = point_features.iloc[0]["count"]
    z_range = point_features.iloc[0]["z_range"]
    z_std = point_features.iloc[0]["std"]
    return [acc_density, z_range, z_std]


def generate_features(point_cloud, nb_neighbors, kdtree_leaf_size=1000):
    """Build the point features for all (or for a sample of) points within
    the point cloud

    Parameters
    ----------
    point_cloud : numpy.array
        Coordinates of all points within the point cloud; must be a 2D-shaped
        array with `point_cloud.shape[1] == 3`
    nb_neighbors : int
        Number of points that must be consider within a neighborhod
    considered
    kdtree_leaf_size : int
        Size of each kd-tree leaf (in number of points)

    Returns
    ------
    list, OrderedDict generator (features for each point)
    """
    acc_features = build_accumulation_features(point_cloud[:, :3])
    kd_tree = KDTree(point_cloud[:, :3], leaf_size=kdtree_leaf_size)
    for point in point_cloud:
        xyz_data, rgb_data = point[:3], point[3:]
        neighborhood = build_neighborhood(xyz_data, nb_neighbors, kd_tree)
        neighbors = point_cloud[neighborhood["indices"], :3]
        lbda_3D = _pca(neighbors, k=3).singular_values_
        lbda_2D = _pca(neighbors[:, :2], k=2).singular_values_
        alpha, beta = features3d(lbda_3D)
        radius, z_range, std_deviation, density, verticality =\
            compute_3D_properties(neighbors[:, 2], neighborhood["distance"])
        curvature_change, linearity, planarity, scattering, omnivariance, anisotropy, eigenentropy, eigenvalue_sum = compute_3D_features(lbda_3D)
        radius_2D, density_2D = compute_2D_properties(xyz_data[:2], neighbors[:, :2])
        eigenvalue_sum_2D, eigenvalue_ratio_2D = compute_2D_features(lbda_2D)
        bin_density, bin_z_range, bin_z_std =\
            retrieve_accumulation_features(xyz_data, acc_features)
        yield OrderedDict([("x", xyz_data[0]),
                           ("y", xyz_data[1]),
                           ("z", xyz_data[2]),
                           ("alpha", alpha),
                           ("beta", beta),
                           ("radius", radius),
                           ("z_range", z_range),
                           ("std_dev", std_deviation),
                           ("density", density),
                           ("verticality", verticality),
                           ("curvature_change", curvature_change),
                           ("linearity", linearity),
                           ("planarity", planarity),
                           ("scattering", scattering),
                           ("omnivariance", omnivariance),
                           ("anisotropy", anisotropy),
                           ("eigenentropy", eigenentropy),
                           ("eigenvalue_sum", eigenvalue_sum),
                           ("radius_2D", radius_2D),
                           ("density_2D", density_2D),
                           ("eigenvalue_sum_2D", eigenvalue_sum_2D),
                           ("eigenvalue_ratio_2D", eigenvalue_ratio_2D),
                           ("bin_density", bin_density),
                           ("bin_z_range", bin_z_range),
                           ("bin_z_std", bin_z_std),
                           ("r", rgb_data[0]),
                           ("g", rgb_data[1]),
                           ("b", rgb_data[2])])
