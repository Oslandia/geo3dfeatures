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
from collections import defaultdict
from multiprocessing import Pool
from timeit import default_timer as timer
from typing import NamedTuple, Tuple

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from scipy.spatial import cKDTree as KDTree

from tqdm import tqdm

import daiquiri

from geo3dfeatures.features import (
    triangle_variance_space,
    sum_normalize, val_sum, val_range, std_deviation,
    curvature_change, linearity, planarity, scattering,
    omnivariance, anisotropy, eigenentropy, verticality_coefficient,
    eigenvalue_ratio_2D,
    radius_2D, radius_3D, density_2D, density_3D
)


logger = daiquiri.getLogger(__name__)


class NeighborFeatures(NamedTuple):
    """List of features for neighbor-based neighborhood
    """
    x: float
    y: float
    z: float
    num_neighbors: int
    alpha: float
    beta: float
    radius: float
    z_range: float
    std_dev: float
    density: float
    verticality: float
    curvature_change: float
    linearity: float
    planarity: float
    scattering: float
    omnivariance: float
    anisotropy: float
    eigenentropy: float
    eigenvalue_sum: float
    radius_2D: float
    density_2D: float
    eigenvalue_sum_2D: float
    eigenvalue_ratio_2D: float


class RadiusFeatures(NamedTuple):
    """List of features for radius-based neighborhood
    """
    x: float
    y: float
    z: float
    alpha: float
    beta: float
    neighbor_nb: int
    z_range: float
    std_dev: float
    density: float
    verticality: float
    curvature_change: float
    linearity: float
    planarity: float
    scattering: float
    omnivariance: float
    anisotropy: float
    eigenentropy: float
    eigenvalue_sum: float
    radius_2D: float
    density_2D: float
    eigenvalue_sum_2D: float
    eigenvalue_ratio_2D: float


class ExtraFeatures(NamedTuple):
    """List of extra features (from the original input data file)
    """
    names: Tuple[str]
    values: Tuple[float]


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


def request_tree(point, kd_tree, nb_neighbors=None, radius=None):
    """Extract a point neighborhood by the way of a KDTree method

    At least one parameter amongst 'nb_neighbors' and 'radius' much be
    valid. 'nb_neighbors' is considered first.

    Parameters
    ----------
    point : numpy.array
        Coordinates of the reference point (x, y, z)
    tree : scipy.spatial.KDTree
        Tree representation of the point cloud
    nb_neighbors : int
        Number of neighboring points to consider
    radius : float
        Radius that defines the neighboring ball around a given point

    Returns
    -------
    tuple
        Neighborhood, decomposed as a mean distance between the reference point
    and its neighbors and an array of neighbor indices within the point
    cloud. Get `nb_neighborhood + 1` in order to have the reference point and
    its k neighbors. If the neighborhood is recovered through radius, distances
    are not computed.
    """
    if nb_neighbors is not None:
        return kd_tree.query(point, k=nb_neighbors + 1)
    if radius is not None:
        return None, kd_tree.query_ball_point(point, r=radius)
    raise ValueError(
        "Kd-tree request error: nb_neighbors and radius can't be both None."
        )


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


def process_full(neighbors, neighborhood_extra, extra, mode="neighbors"):
    """Build the full feature set for a single point

    Parameters
    ----------
    neighbors : numpy.array
        Coordinates of all points within the point neighborhood; must be a
    2D-shaped array with `point_cloud.shape[1] == 3`
    neighborhood_extra : numpy.array or int
        If mode == "neighbors", gives distance between the point and all its
    neighbors; otherwise (mode == "radius") gives the number of points in the
    local sphere
    extra : ExtraFeatures
        Names and values of extra input features, e.g. the RGB color
    mode : str
        Gives the neighborhood definition, i.e. "neighbors" or "radius"

    Returns
    ------
    list, OrderedDict generator (features for each point)

    """
    num_neighbors = neighbors.shape[0] - 1
    x, y, z = neighbors[0]
    extra_2D = radius_2D(neighbors[0, :2], neighbors[:, :2])
    if mode == "neighbors":
        FeatureTuple = NeighborFeatures
        extra_3D = radius_3D(neighborhood_extra)
        dens_3D = density_3D(extra_3D, len(neighbors))
        dens_2D = density_2D(extra_2D, len(neighbors))
    elif mode == "radius":
        FeatureTuple = RadiusFeatures
        extra_3D = len(neighbors)
        dens_3D = density_3D(neighborhood_extra, extra_3D)
        dens_2D = density_2D(neighborhood_extra, extra_2D)
    else:
        raise ValueError("Unknown neighboring mode.")
    if len(neighbors) <= 2:
        return (
            FeatureTuple(
                x, y, z,
                np.nan, np.nan,  # alpha, beta
                extra_3D,  # radius3D
                val_range(neighbors[:, 2]),  # z_range
                std_deviation(neighbors[:, 2]),  # std_dev
                dens_3D,  # density3D
                np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan,
                extra_2D,  # radius2D
                dens_2D,  # density2D
                np.nan,  # eigenvalue_sum_2D
                np.nan,  # eigenvalue_ratio_2D
            ),
            extra
        )
    else:
        pca = fit_pca(neighbors)  # PCA on the x,y,z coords
        eigenvalues_3D = pca.singular_values_ ** 2
        norm_eigenvalues_3D = sum_normalize(eigenvalues_3D)
        alpha, beta = triangle_variance_space(norm_eigenvalues_3D)
        pca_2d = fit_pca(neighbors[:, :2])  # PCA just on the x,y coords
        eigenvalues_2D = pca_2d.singular_values_ ** 2
        return (FeatureTuple(x, y, z,
                             num_neighbors,
                             alpha,
                             beta,
                             extra_3D,
                             val_range(neighbors[:, 2]),  # z_range
                             std_deviation(neighbors[:, 2]),  # std_dev
                             dens_3D,
                             verticality_coefficient(pca),
                             curvature_change(norm_eigenvalues_3D),
                             linearity(norm_eigenvalues_3D),
                             planarity(norm_eigenvalues_3D),
                             scattering(norm_eigenvalues_3D),
                             omnivariance(norm_eigenvalues_3D),
                             anisotropy(norm_eigenvalues_3D),
                             eigenentropy(norm_eigenvalues_3D),
                             val_sum(eigenvalues_3D),  # eigenvalue sum
                             extra_2D,  # radius 2D
                             dens_2D,
                             val_sum(eigenvalues_2D),  # eigenvalue_sum_2D
                             eigenvalue_ratio_2D(eigenvalues_2D)),
                extra)


def _wrap_full_process(args):
    """Wrap the 'process_full' function in order to use 'pool.imap_unordered' which takes
    only one argument
    """
    return process_full(*args)


def sequence_full(
        scene, tree, nb_neighbors, radius=None, extra_columns=None
):
    """Build a data generator for getting neighborhoods, distances and
        accumulation features for each point

    Parameters
    ----------
    scene : np.array
        point cloud + extra columns
    tree : scipy.spatial.ckdtree.CKDTree
        Point cloud kd-tree for computing nearest neighborhoods
    nb_neighbors : list
        List of neighbors numbers in each point neighborhood
    radius : float
        Radius that defines the neighboring ball around a given point
    extra_columns : tuple
        Extra input data column names, reused for output dataframe

    Yields
    ------
    int
        Number of neighbors (or radius)
    numpy.array
        Geometric coordinates of neighbors
    numpy.array
        Euclidian distance between the reference point and its neighbors
    ExtraFeature
        Extra input data column names, reused for output dataframe
    str
        radius or neigbors mode
    """
    if extra_columns is None:
        if scene.shape[1] != 3:
            raise ValueError("No extra column declared.")
    else:
        if scene.shape[1] - 3 != len(extra_columns):
            raise ValueError("Extra column lengths does not match data.")
    num_max_neighbors = list(reversed(nb_neighbors))[0]
    for point in scene:
        neighborhood_extra, neighbor_idx = request_tree(
            point[:3], tree, num_max_neighbors, radius
        )
        mode = "neighbors"
        extra_features = (
            ExtraFeatures(extra_columns, tuple(point[3:]))
            if extra_columns else ExtraFeatures(tuple(), tuple())
        )
        # XXX we have some issues with radius
        # https://git.oslandia.net/Oslandia-data/geo3dfeatures/issues/46
        # if neighborhood_extra is None:  # True if radius is not None
        #     neighborhood_extra = radius
        #     mode = "radius"
        #     yield radius, tree.data[neighbor_idx], neighborhood_extra, extra_features, mode
        # else:
        for num_neighbors in reversed(nb_neighbors):
            # add 1 neighbor because we have the reference point
            index = neighbor_idx[:num_neighbors + 1]
            neighbors = tree.data[index]
            yield neighbors, neighborhood_extra, extra_features, mode


def _dump_results_by_chunk(iterable, h5path, chunksize, progress_bar):
    """Write result in a hdf5 file by chunk.
    """
    def chunkgenerator(iterable):
        group = defaultdict(list)
        features, extra = next(iterable)
        names = features._fields + extra.names
        group[features.num_neighbors].append(features + extra.values)
        for num, (features, extra) in enumerate(iterable, start=1):
            group[features.num_neighbors].append(features + extra.values)
            if (num+1) % chunksize == 0:
                names = features._fields + extra.names
                yield names, group
                group = defaultdict(list)
        yield names, group

    with pd.HDFStore(h5path) as store:
        for names, chunk in chunkgenerator(iterable):
            for num, data in chunk.items():
                df = pd.DataFrame(data, columns=names)
                key = "/num_{:04d}".format(num)
                store.append(key, df)
            progress_bar.update()


def extract(
        point_cloud, tree, h5path, nb_neighbors, radius=None,
        nb_processes=2, extra_columns=None,
        chunksize=20000
):
    """Extract geometric features from a 3D point cloud.

    Write the results in a CSV file.

    Parameters
    ----------
    point_cloud : numpy.array
        3D point cloud
    tree : scipy.spatial.ckdtree.CKDTree
        Point cloud kd-tree for computing nearest neighborhoods
    h5path : Path
        hdf5 output path (extracted features)
    nb_neighbors : list of ints
        Number of neighbors in each point neighborhood
    radius : float
        Radius that defines the neighboring ball around a given point
    nb_processes : int
        Number of parallel cores
    extra_columns : list
        Extra input data column names, reused for output (None by default)
    """
    logger.info("Computation begins!")
    if nb_neighbors is None and radius is None:
        logger.error("nb_neighbors and radius can't be both None.")
        raise ValueError(
            "Error in input neighborhood definition: "
            "nb_neighbors and radius can't be both None."
        )
    if radius is not None:
        raise NotImplementedError("we have some issues with radius")
    if isinstance(nb_neighbors, int):
        nb_neighbors = [nb_neighbors]
    start = timer()
    gen = sequence_full(
        point_cloud, tree, nb_neighbors, radius, extra_columns
    )
    with Pool(processes=nb_processes) as pool:
        logger.info("Total number of points: %s", point_cloud.shape[0])
        steps = math.ceil(point_cloud.shape[0] / chunksize * len(nb_neighbors))
        result_it = pool.imap_unordered(
            _wrap_full_process, gen, chunksize=chunksize
        )
        with tqdm(total=steps) as pbar:
            _dump_results_by_chunk(
                result_it, h5path, chunksize, progress_bar=pbar
            )
    stop = timer()
    logger.info("Time spent: %.2fs", stop - start)
