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
from scipy.spatial import cKDTree as KDTree

from geo3dfeatures.features import (
    accumulation_2d_neighborhood,
    triangle_variance_space,
    normalize, val_sum, val_range, std_deviation,
    curvature_change, linearity, planarity, scattering,
    omnivariance, anisotropy, eigenentropy, verticality_coefficient,
    eigenvalue_ratio_2D,
    radius_2D, radius_3D, density_2D, density_3D
)


def compute_tree(point_cloud, leaf_size):
    """Compute the KDTree structure

    Parameters
    ----------
    point_cloud : numpy.array
    leaf_size : int

    Returns
    -------
    sklearn.neighbors.KDTree (or scipy.spatial.KDTree)
    """
    return KDTree(point_cloud, leaf_size)


def request_tree(point, nb_neighbors, kd_tree):
    """Extract a point neighborhood by the way of a KDTree method

    Parameters
    ----------
    point : numpy.array
        Coordinates of the reference point (x, y, z)
    nb_neighborhood : int
        Number of neighboring points to consider
    tree : scipy.spatial.KDTree
        Tree representation of the point cloud

    Returns
    -------
    tuple
        Neighborhood, decomposed as a mean distance between the reference point and
        its neighbors and an array of neighbor indices within the point cloud. Get
        `nb_neighborhood + 1` in order to have the reference point and its k neighbors.
    """
    return kd_tree.query(point, k=nb_neighbors + 1)


def fit_pca(point_cloud):
    """Fit a PCA model on a 3D point cloud characterized by x-, y- and
    z-coordinates

    Parameters
    ----------
    point_cloud : numpy.array
        Array of (x, y, z) points

    Returns
    -------
    sklearn.decomposition.PCA
        Principle Component Analysis model fitted to the input data
    """
    return PCA().fit(point_cloud)


def alphabeta_features(
        point_cloud, nb_neighbors, 
        input_columns=["x", "y", "z"], kdtree_leaf_size=1000
):
    """Compute 'alpha' and 'beta' features within 'point_cloud', a set of 3D
    points, according to Brodu et al (2012)

    Apart from the point cloud base features (x, y, z, r, g, b), one has two
    additional features ('alpha' and 'beta').

    Parameters
    ----------
    point_cloud : numpy.array
        Coordinates of all points within the point cloud; must be a 2D-shaped
        array with `point_cloud.shape[1] == 3`
    nb_neighbors : int
        Number of points that must be consider within a neighborhod
    considered
    input_columns : list
        List of input column names, that must begin "x", "y" and "z" columns
    at least; its length must correspond to "point_cloud" number of columns
    kdtree_leaf_size : int
        Size of each kd-tree leaf (in number of points)

    Returns
    ------
    list, OrderedDict generator (features for each point)
    """
    assert input_columns[:3] == ["x", "y", "z"]
    assert len(input_columns) == point_cloud.shape[1]
    kd_tree = compute_tree(point_cloud[:, :3], leaf_size=kdtree_leaf_size)
    for point in point_cloud:
        _, neighbor_idx = request_tree(point[:3], nb_neighbors, kd_tree)
        neighbors = point_cloud[neighbor_idx, :3]
        pca = fit_pca(neighbors)  # PCA on the x,y,z coords
        norm_eigenvalues_3D = normalize(pca.singular_values_ ** 2)
        alpha, beta = triangle_variance_space(norm_eigenvalues_3D)
        yield OrderedDict(
            [
                (name, data)
                for name, data in zip(input_columns, point)
            ]
            +
            [
                ("alpha", alpha),
                ("beta", beta),
            ]
        )


def eigen_features(
        point_cloud, nb_neighbors, 
        input_columns=["x", "y", "z"], kdtree_leaf_size=1000
):
    """Compute 'alpha' and 'beta' features within 'point_cloud', a set of 3D
    points, according to Brodu et al (2012), as well as neighborhood attributes
    based on eigenvalues (see Weinmann et al, 2015, for details about such
    metrics)

    Apart from the point cloud base features (x, y, z, r, g, b), one has eight
    additional features: 'alpha' and 'beta', plus 'curvature_change',
    'linearity', 'planarity', 'scattering', 'omnivariance', 'anisotropy',
    'eigenentropy' and 'eigenvalue_sum'.

    Parameters
    ----------
    point_cloud : numpy.array
        Coordinates of all points within the point cloud; must be a 2D-shaped
        array with `point_cloud.shape[1] == 3`
    nb_neighbors : int
        Number of points that must be consider within a neighborhod
    considered
    input_columns : list
        List of input column names, that must begin "x", "y" and "z" columns
    at least; its length must correspond to "point_cloud" number of columns
    kdtree_leaf_size : int
        Size of each kd-tree leaf (in number of points)

    Returns
    ------
    list, OrderedDict generator (features for each point)
    """
    assert input_columns[:3] == ["x", "y", "z"]
    assert len(input_columns) == point_cloud.shape[1]
    kd_tree = compute_tree(point_cloud[:, :3], leaf_size=kdtree_leaf_size)
    for point in point_cloud:
        _, neighbor_idx = request_tree(point[:3], nb_neighbors, kd_tree)
        neighbors = point_cloud[neighbor_idx, :3]
        pca = fit_pca(neighbors)  # PCA on the x,y,z coords
        eigenvalues_3D = pca.singular_values_ ** 2
        norm_eigenvalues_3D = normalize(eigenvalues_3D)
        alpha, beta = triangle_variance_space(norm_eigenvalues_3D)
        yield OrderedDict(
            [
                (name, data)
                for name, data in zip(input_columns, point)
            ]
            +
            [
                ("alpha", alpha),
                ("beta", beta),
                ("curvature_change", curvature_change(norm_eigenvalues_3D)),
                ("linearity", linearity(norm_eigenvalues_3D)),
                ("planarity", planarity(norm_eigenvalues_3D)),
                ("scattering", scattering(norm_eigenvalues_3D)),
                ("omnivariance", omnivariance(norm_eigenvalues_3D)),
                ("anisotropy", anisotropy(norm_eigenvalues_3D)),
                ("eigenentropy", eigenentropy(norm_eigenvalues_3D)),
                ("eigenvalue_sum", val_sum(eigenvalues_3D)),
            ]
        )


def all_features(
        point_cloud, nb_neighbors,
        input_columns=["x", "y", "z"], kdtree_leaf_size=1000
):
    """Build the full feature set for all points within the point cloud

    Parameters
    ----------
    point_cloud : numpy.array
        Coordinates of all points within the point cloud; must be a 2D-shaped
        array with `point_cloud.shape[1] == 3`
    nb_neighbors : int
        Number of points that must be consider within a neighborhod
    considered
    input_columns : list
        List of input column names, that must begin "x", "y" and "z" columns
    at least; its length must correspond to "point_cloud" number of columns
    kdtree_leaf_size : int
        Size of each kd-tree leaf (in number of points)

    Returns
    ------
    list, OrderedDict generator (features for each point)

    """
    assert input_columns[:3] == ["x", "y", "z"]
    assert len(input_columns) == point_cloud.shape[1]
    acc_features = accumulation_2d_neighborhood(point_cloud, input_columns)
    kd_tree = compute_tree(point_cloud[:, :3], leaf_size=kdtree_leaf_size)
    for point in acc_features.values:
        distance, neighbor_idx = request_tree(point[:3], nb_neighbors, kd_tree)
        neighbors = point_cloud[neighbor_idx, :3]
        pca = fit_pca(neighbors)  # PCA on the x,y,z coords
        eigenvalues_3D = pca.singular_values_ ** 2
        norm_eigenvalues_3D = normalize(eigenvalues_3D)
        alpha, beta = triangle_variance_space(norm_eigenvalues_3D)
        rad_3D = radius_3D(distance)
        pca_2d = fit_pca(neighbors[:, :2])  # PCA just on the x,y coords
        eigenvalues_2D = pca_2d.singular_values_ ** 2
        rad_2D = radius_2D(point[:2], neighbors[:, :2])
        yield OrderedDict(
            [
                (name, data)
                for name, data in zip(input_columns, point[:-3])
            ]
            +
            [
                ("alpha", alpha),
                ("beta", beta),
                ("radius", rad_3D),
                ("z_range", val_range(neighbors[:, 2])),
                ("std_dev", std_deviation(neighbors[:, 2])),
                ("density", density_3D(rad_3D, nb_neighbors)),
                ("verticality", verticality_coefficient(pca)),
                ("curvature_change", curvature_change(norm_eigenvalues_3D)),
                ("linearity", linearity(norm_eigenvalues_3D)),
                ("planarity", planarity(norm_eigenvalues_3D)),
                ("scattering", scattering(norm_eigenvalues_3D)),
                ("omnivariance", omnivariance(norm_eigenvalues_3D)),
                ("anisotropy", anisotropy(norm_eigenvalues_3D)),
                ("eigenentropy", eigenentropy(norm_eigenvalues_3D)),
                ("eigenvalue_sum", val_sum(eigenvalues_3D)),
                ("radius_2D", rad_2D),
                ("density_2D", density_2D(rad_2D, nb_neighbors)),
                ("eigenvalue_sum_2D", val_sum(eigenvalues_2D)),
                ("eigenvalue_ratio_2D", eigenvalue_ratio_2D(eigenvalues_2D)),
                ("bin_density", point[-3]),
                ("bin_z_range", point[-2]),
                ("bin_z_std", point[-1])
            ]
        )
