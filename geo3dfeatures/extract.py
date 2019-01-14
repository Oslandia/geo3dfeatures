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

from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree

from geo3dfeatures.features import (accumulation_2d_neighborhood, triangle_variance_space,
                                    compute_3D_features, compute_2D_features)


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
        Neighborhood, decomposed as a mean distance between the reference point and
        its neighbors and an array of neighbor indices within the point cloud. Get
        `nb_neighborhood + 1` in order to have the reference point and its k neighbors.
    """
    dist, ind = kd_tree.query(np.expand_dims(point, 0), k=nb_neighbors + 1)
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
    density = (len(distances) + 1) / ((4 / 3) * math.pi * radius ** 3)
    verticality = np.nan
    return [radius, z_range, std_deviation, density, verticality]



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
    distances = [
        math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2) for x, y in zip(xs, ys)
    ]
    radius_2D = max(distances)
    density_2D = (len(distances) + 1) / (math.pi * radius_2D ** 2)
    return [radius_2D, density_2D]


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
    acc_features = accumulation_2d_neighborhood(point_cloud[:, :3])
    kd_tree = KDTree(point_cloud[:, :3], leaf_size=kdtree_leaf_size)
    for point in point_cloud:
        xyz_data, rgb_data = point[:3], point[3:]
        neighborhood = build_neighborhood(xyz_data, nb_neighbors, kd_tree)
        neighbors = point_cloud[neighborhood["indices"], :3]
        pca = PCA().fit(neighbors)  # PCA on the x,y,z coords
        pca_2d = PCA().fit(neighbors[:, :2])  # PCA just on the x,y coords
        eigenvalue_sum = (pca.singular_values_ ** 2).sum()
        alpha, beta = triangle_variance_space(pca)
        radius, z_range, std_deviation, density, verticality = compute_3D_properties(
            neighbors[:, 2], neighborhood["distance"]
        )
        curvature_change, linearity, planarity, scattering, omnivariance, anisotropy, eigenentropy = compute_3D_features(
            pca
        )
        radius_2D, density_2D = compute_2D_properties(xyz_data[:2], neighbors[:, :2])
        eigenvalue_sum_2D, eigenvalue_ratio_2D = compute_2D_features(pca_2d)
        bin_density, bin_z_range, bin_z_std = retrieve_accumulation_features(
            xyz_data, acc_features
        )
        yield OrderedDict(
            [
                ("x", xyz_data[0]),
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
                ("b", rgb_data[2]),
            ]
        )
