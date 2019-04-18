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

from csv import writer as CSVWriter
import math
from multiprocessing import Pool
from timeit import default_timer as timer
from typing import NamedTuple, Tuple

import daiquiri
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm

from geo3dfeatures import FEATURE_SETS
from geo3dfeatures.features import (
    accumulation_2d_neighborhood,
    triangle_variance_space,
    normalize, val_sum, val_range, std_deviation,
    curvature_change, linearity, planarity, scattering,
    omnivariance, anisotropy, eigenentropy, verticality_coefficient,
    eigenvalue_ratio_2D,
    radius_2D, radius_3D, density_2D, density_3D
)


logger = daiquiri.getLogger(__name__)


class AlphaBetaFeatures(NamedTuple):
    """Alpha & Beta features (barycentric coordinates from PCA eigenvalues)
    """
    x: float
    y: float
    z: float
    alpha: float
    beta: float


class EigenvaluesFeatures(NamedTuple):
    """List of features (eigenvalues)
    """
    x: float
    y: float
    z: float
    alpha: float
    beta: float
    curvature_change: float
    linearity: float
    planarity: float
    scattering: float
    omnivariance: float
    anisotropy: float
    eigenentropy: float
    eigenvalue_sum: float


class Features(NamedTuple):
    """List of features
    """
    x: float
    y: float
    z: float
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
    bin_density: float
    bin_z_range: float
    bin_z_std: float


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
        Neighborhood, decomposed as a mean distance between the reference point
    and its neighbors and an array of neighbor indices within the point
    cloud. Get `nb_neighborhood + 1` in order to have the reference point and
    its k neighbors.
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


def process_alphabeta(neighbors, extra):
    """Compute 'alpha' and 'beta' features for a single point, according to
    Brodu et al (2012)

    Apart from the point cloud base features (x, y, z, r, g, b), one has two
    additional features ('alpha' and 'beta').

    Parameters
    ----------
    neighbors : numpy.array
        Coordinates of all points within the point neighborhood; must be a
        2D-shaped array with `point_cloud.shape[1] == 3`
    extra : ExtraFeatures
        Names and values of extra input features, e.g. the RGB color

    Returns
    ------
    list, OrderedDict generator (features for each point)
    """
    pca = fit_pca(neighbors[:, :3])  # PCA on the x,y,z coords
    eigenvalues_3D = pca.singular_values_ ** 2
    norm_eigenvalues_3D = normalize(eigenvalues_3D)
    alpha, beta = triangle_variance_space(norm_eigenvalues_3D)
    x, y, z = neighbors[0]
    return (AlphaBetaFeatures(x, y, z,
                              alpha,
                              beta),
            extra)


def process_eigenvalues(neighbors, extra):
    """Compute 'alpha' and 'beta' features for a single point, according to
    Brodu et al (2012), as well as neighborhood attributes based on eigenvalues
    (see Weinmann et al, 2015, for details about such metrics)

    Apart from the point cloud base features (x, y, z, r, g, b), one has eight
    additional features: 'alpha' and 'beta', plus 'curvature_change',
    'linearity', 'planarity', 'scattering', 'omnivariance', 'anisotropy',
    'eigenentropy' and 'eigenvalue_sum'.

    Parameters
    ---------
    neighbors : numpy.array
        Coordinates of all points within the point neighborhood; must be a
    2D-shaped array with `point_cloud.shape[1] == 3`
    extra : ExtraFeatures
        Names and values of extra input features, e.g. the RGB color

    Returns
    ------
    list, OrderedDict generator (features for each point)

    """
    pca = fit_pca(neighbors[:, :3])  # PCA on the x,y,z coords
    eigenvalues_3D = pca.singular_values_ ** 2
    norm_eigenvalues_3D = normalize(eigenvalues_3D)
    alpha, beta = triangle_variance_space(norm_eigenvalues_3D)
    x, y, z = neighbors[0]
    return (EigenvaluesFeatures(x, y, z,
                                alpha,
                                beta,
                                curvature_change(norm_eigenvalues_3D),
                                linearity(norm_eigenvalues_3D),
                                planarity(norm_eigenvalues_3D),
                                scattering(norm_eigenvalues_3D),
                                omnivariance(norm_eigenvalues_3D),
                                anisotropy(norm_eigenvalues_3D),
                                eigenentropy(norm_eigenvalues_3D),
                                val_sum(eigenvalues_3D)),  # eigenvalue sum
            extra)


def process_full(neighbors, distance, z_acc, extra):
    """Build the full feature set for a single point

    Parameters
    ----------
    neighbors : numpy.array
        Coordinates of all points within the point neighborhood; must be a
    2D-shaped array with `point_cloud.shape[1] == 3`
    distance : numpy.array
        Distance between the point and all its neighbors
    z_acc : numpy.array
        Accumulation features associated to the point
    extra : ExtraFeatures
        Names and values of extra input features, e.g. the RGB color

    Returns
    ------
    list, OrderedDict generator (features for each point)

    """
    pca = fit_pca(neighbors)  # PCA on the x,y,z coords
    eigenvalues_3D = pca.singular_values_ ** 2
    norm_eigenvalues_3D = normalize(eigenvalues_3D)
    alpha, beta = triangle_variance_space(norm_eigenvalues_3D)
    rad_3D = radius_3D(distance)
    pca_2d = fit_pca(neighbors[:, :2])  # PCA just on the x,y coords
    eigenvalues_2D = pca_2d.singular_values_ ** 2
    rad_2D = radius_2D(neighbors[0, :2], neighbors[:, :2])
    x, y, z = neighbors[0]
    return (Features(x, y, z,
                     alpha,
                     beta,
                     rad_3D,
                     val_range(neighbors[:, 2]),  # z_range
                     std_deviation(neighbors[:, 2]),  # std_dev
                     density_3D(rad_3D, len(neighbors)),
                     verticality_coefficient(pca),
                     curvature_change(norm_eigenvalues_3D),
                     linearity(norm_eigenvalues_3D),
                     planarity(norm_eigenvalues_3D),
                     scattering(norm_eigenvalues_3D),
                     omnivariance(norm_eigenvalues_3D),
                     anisotropy(norm_eigenvalues_3D),
                     eigenentropy(norm_eigenvalues_3D),
                     val_sum(eigenvalues_3D),  # eigenvalue sum
                     rad_2D,  # radius 2D
                     density_2D(rad_2D, len(neighbors)),
                     val_sum(eigenvalues_2D),  # eigenvalue_sum_2D
                     eigenvalue_ratio_2D(eigenvalues_2D),
                     z_acc[-3],    # bin_density
                     z_acc[-2],    # bin_z_range
                     z_acc[-1]),   # bin_z_std
            extra)


def _wrap_alphabeta_process(args):
    """Wrap the 'process_alphabeta' function in order to use 'pool.imap_unordered' which takes
    only one argument
    """
    return process_alphabeta(*args)


def _wrap_eigenvalues_process(args):
    """Wrap the 'process_eigenvalues' function in order to use 'pool.imap_unordered' which takes
    only one argument
    """
    return process_eigenvalues(*args)


def _wrap_full_process(args):
    """Wrap the 'process_full' function in order to use 'pool.imap_unordered' which takes
    only one argument
    """
    return process_full(*args)


def sequence_light(point_cloud, tree, nb_neighbors, extra_columns):
    """Build a data generator for getting neighborhoods and distances for each
    point

    Parameters
    ----------
    point_cloud : numpy.array
        3D point cloud
    tree : scipy.spatial.ckdtree.CKDTree
        Point cloud kd-tree for computing nearest neighborhoods
    nb_neighbors : int
        Number of neighbors in each point neighborhood
    extra_columns : tuple
        Extra input data column names, reused for output dataframe

    Yields
    ------
    numpy.array
        Geometric coordinates of neighbors
    numpy.array
        Euclidian distance between the reference point and its neighbors
    """
    for point in point_cloud:
        distance, neighbor_idx = request_tree(point[:3], nb_neighbors, tree)
        extra_features = ExtraFeatures(extra_columns, tuple(point[3:])) if extra_columns else ExtraFeatures(tuple(), tuple())
        yield tree.data[neighbor_idx], extra_features


def sequence_full(
        acc_features, tree, nb_neighbors, extra_columns
):
    """Build a data generator for getting neighborhoods, distances and
        accumulation features for each point

    Parameters
    ----------
    acc_features : pd.DataFrame
        point cloud + extra columns + z-accumulation features
    tree : scipy.spatial.ckdtree.CKDTree
        Point cloud kd-tree for computing nearest neighborhoods
    nb_neighbors : int
        Number of neighbors in each point neighborhood
    extra_columns : tuple
        Extra input data column names, reused for output dataframe

    Yields
    ------
    numpy.array
        Geometric coordinates of neighbors
    numpy.array
        Euclidian distance between the reference point and its neighbors
    numpy.array
        Reference point accumulation features
    """
    for point in acc_features.values:
        distance, neighbor_idx = request_tree(point[:3], nb_neighbors, tree)
        z_acc = point[-3:]
        extra_features = ExtraFeatures(extra_columns, tuple(point[3:-3])) if extra_columns else ExtraFeatures(tuple(), tuple())
        yield tree.data[neighbor_idx], distance, z_acc, extra_features


def _dump_results_by_chunk(iterable, csvpath, chunksize, progress_bar):
    """Write result in a CSV file by chunk.
    """
    def chunkgenerator(iterable):
        group = []
        for num, (features, extra) in enumerate(iterable):
            group.append(features + extra.values)
            if (num+1) % chunksize == 0:
                yield group
                group = []
        yield group

    features, extra = next(iterable)
    with open(csvpath, 'w') as fobj:
        writer = CSVWriter(fobj)
        # write the headers. 'extra.names' can be an empty tuple
        writer.writerow(features._fields + extra.names)
        # write the first line
        writer.writerow(features + extra.values)
        # write results by chunk
        num_processed_points = chunksize
        for chunk in chunkgenerator(iterable):
            writer.writerows(chunk)
            num_processed_points += chunksize
            progress_bar.update()


def extract(
        point_cloud, tree, nb_neighbors, csvpath, sample_points=None,
        feature_set="full", nb_processes=2, extra_columns=None, chunksize=20000
):
    """Extract geometric features from a 3D point cloud.

    Write the results in a CSV file.

    Parameters
    ----------
    point_cloud : numpy.array
        3D point cloud
    tree : scipy.spatial.ckdtree.CKDTree
        Point cloud kd-tree for computing nearest neighborhoods
    nb_neighbors : int
        Number of neighbors in each point neighborhood
    csvpath : Path
        CSV output path (extracted features)
    sample_points : int
        Sampling size (if None, this is the full point cloud which is taking into account)
    nb_processes : int
        Number of parallel cores
    extra_columns : list
        Extra input data column names, reused for output (None by default)
    """
    logger.info("Computation begins!")
    acc_features = accumulation_2d_neighborhood(point_cloud, extra_columns)
    if sample_points is not None:
        acc_features = acc_features.sample(sample_points)
    start = timer()
    if feature_set == "full":
        gen = sequence_full(acc_features, tree, nb_neighbors, extra_columns)
    else:
        gen = sequence_light(acc_features.values[:, :-3], tree, nb_neighbors, extra_columns)
    with Pool(processes=nb_processes) as pool:
        logger.info("Total number of points: %s", point_cloud.shape[0])
        steps = math.ceil(point_cloud.shape[0] / chunksize)
        with tqdm(total=steps) as pbar:
            if feature_set == "full":
                result_it = pool.imap_unordered(
                    _wrap_full_process, gen, chunksize=chunksize
                )
            elif feature_set == "eigenvalues":
                result_it = pool.imap_unordered(
                    _wrap_eigenvalues_process, gen, chunksize=chunksize
                )
            elif feature_set == "alphabeta":
                result_it = pool.imap_unordered(
                    _wrap_alphabeta_process, gen, chunksize=chunksize
                )
            else:
                raise ValueError(
                    "Unknown feature set, choose amongst {}"
                    .format(FEATURE_SETS)
                )
            _dump_results_by_chunk(
                result_it, csvpath, chunksize, progress_bar=pbar
            )
    stop = timer()
    logger.info("Time spent: %s", stop - start)
