"""K-means output post-processing

Compute our own mode method inspired from scipy.stats source code:
- https://github.com/scipy/scipy/blob/master/scipy/stats/stats.py#L609

Compute the mode in an alternative pure-numpy way:
- https://stackoverflow.com/questions/12297016/how-to-find-most-frequent-values-in-numpy-ndarray
"""

import argparse
from csv import writer as CSVWriter
import math
from multiprocessing import Pool
from pathlib import Path
import pickle
from timeit import default_timer as timer
from typing import NamedTuple

import daiquiri
import pandas as pd
from scipy import stats
from tqdm import tqdm

from geo3dfeatures import extract, io


logger = daiquiri.getLogger(__name__)


def postprocess_labels(
        point_cloud, labels, tree, n_neighbors=None, radius=None
):
    """

    Parameters
    ----------
    point_cloud : pd.DataFrame
    labels : np.array
    tree : scipy.spatial.ckdtree.cKDTree
    nb_neighbors : int
        Number of neighbors in each point neighborhood
    radius : float
        Radius that defines the neighboring ball around a given point
    """
    point_cloud_array = point_cloud.values
    _, neighbors = extract.request_tree(
        point_cloud_array, tree, n_neighbors, radius
    )
    point_neighborhoods = labels[neighbors]
    u, indices = np.unique(point_neighborhoods, return_inverse=True)
    new_clusters = u[
        np.argmax(
            np.apply_along_axis(
                np.bincount, 1, indices.reshape(point_neighborhoods.shape),
                None, np.max(indices) + 1),
            axis=1)
    ]
    return new_clusters


class Row(NamedTuple):
    x: float
    y: float
    z: float
    r: float
    g: float
    b: float
    updated: bool


def yield_main_clusters(point_cloud, tree, nb_neighbors, radius=None):
    """Compute the most encountered cluster amongst the point neighborhood

    Parameters
    ----------
    point_cloud : pandas.DataFrame
        3D point cloud
    tree : scipy.spatial.ckdtree.CKDTree
        Point cloud kd-tree for computing nearest neighborhoods
    nb_neighbors : int
        Number of neighbors in each point neighborhood
    radius : float
        Radius that defines the neighboring ball around a given point

    Yields
    ------
    int
    """
    points = point_cloud.values
    for point in points:
        _, neighbors = extract.request_tree(
            point[:3], tree, nb_neighbors, radius
        )
        new_cluster = stats.mode(points[neighbors, 3:]).mode
        updated = (new_cluster == point[3:]).all()
        yield (
            point[:3], new_cluster[0], updated
            )


def process_point(point, new_cluster, updated):
    """
    """
    return Row(
        point[0],
        point[1],
        point[2],
        new_cluster[0],
        new_cluster[1],
        new_cluster[2],
        updated
    )

def _wrap_process(args):
    """Wrap the 'process_point' function in order to use 'pool.imap_unordered'
    which takes only one argument

    """
    return process_point(*args)


def _dump_results_by_chunk(iterable, csvpath, chunksize, progress_bar):
    """Write result in a CSV file by chunk.
    """
    def chunkgenerator(iterable):
        group = []
        for num, item in enumerate(iterable):
            group.append(item)
            if (num + 1) % chunksize == 0:
                yield group
                group = []
        yield group

    item = next(iterable)
    with open(csvpath, 'w') as fobj:
        writer = CSVWriter(fobj)
        # write the headers. 'extra.names' can be an empty tuple
        writer.writerow(item._fields)
        # write the first line
        writer.writerow(item)
        # write results by chunk
        num_processed_points = chunksize
        for chunk in chunkgenerator(iterable):
            writer.writerows(chunk)
            num_processed_points += chunksize
            progress_bar.update()


def postprocess_multi(
        point_cloud, tree, csvpath, nb_neighbors=None, radius=None,
        nb_processes=2, chunksize=10000
):
    """Postprocess cluster outputs with multiprocessing tools

    Write the results in a CSV file.

    Parameters
    ----------
    point_cloud : numpy.array
        3D point cloud
    tree : scipy.spatial.ckdtree.CKDTree
        Point cloud kd-tree for computing nearest neighborhoods
    csvpath : Path
        CSV output path (extracted features)
    nb_neighbors : int
        Number of neighbors in each point neighborhood
    radius : float
        Radius that defines the neighboring ball around a given point
    nb_processes : int
        Number of parallel cores
    """
    logger.info("Computation begins!")
    if nb_neighbors is None and radius is None:
        logger.error("nb_neighbors and radius can't be both None.")
        raise ValueError(
            "Error in input neighborhood definition: "
            "nb_neighbors and radius can't be both None."
        )
    start = timer()
    gen = yield_main_clusters(
        point_cloud, tree, nb_neighbors, radius
    )
    with Pool(processes=nb_processes) as pool:
        logger.info("Total number of points: %s", point_cloud.shape[0])
        steps = math.ceil(point_cloud.shape[0] / chunksize)
        result_it = pool.imap_unordered(
            _wrap_process, gen, chunksize=chunksize
        )
        with tqdm(total=steps) as pbar:
            _dump_results_by_chunk(
                result_it, csvpath, chunksize, progress_bar=pbar
            )
    stop = timer()
    logger.info("Time spent: %.2fs", stop - start)


import numpy as np
def mode(a, axis=0):
    """
    """
    scores = np.unique(np.reshape(a, (-1, a.shape[-1])), axis=0)
    testshape = list(a.shape)
    testshape[axis] = 1
    mostfrequent = np.squeeze(np.zeros(testshape))
    oldcounts = np.zeros(a.shape[0])
    for score in scores:
        template = np.all(a == score, axis=2)
        counts = np.sum(template, axis)
        mostfrequent[counts > oldcounts] = score
        oldcounts = np.maximum(counts, oldcounts)
    return mostfrequent

def modpy(a, axis=1):
    """
    """
    u, indices = np.unique(a, return_inverse=True)
    return u[np.argmax(np.apply_along_axis(np.bincount, axis,
                                           indices.reshape(a.shape), None,
                                           np.max(indices) + 1), axis=axis)]


def postprocess(
    point_cloud, tree, csvpath, nb_neighbors=None, radius=None

):
    """Postprocess cluster outputs

    Write the results in a CSV file.

    Parameters
    ----------
    point_cloud : numpy.array
        3D point cloud
    tree : scipy.spatial.ckdtree.CKDTree
        Point cloud kd-tree for computing nearest neighborhoods
    csvpath : Path
        CSV output path (extracted features)
    nb_neighbors : int
        Number of neighbors in each point neighborhood
    radius : float
        Radius that defines the neighboring ball around a given point
    """
    start = timer()
    point_cloud_array = point_cloud.values
    _, neighbors = extract.request_tree(
        point_cloud_array[:,:3], tree, nb_neighbors, None
    )
    new_clusters = modpy(point_cloud_array[neighbors, 3:])
    df = point_cloud[["x", "y", "z"]].copy()
    df.assign(r=new_clusters[:, 0], g=new_clusters[:, 1], b=new_clusters[:, 2])
    # Write results
    df.to_csv(csvpath, index=False)
    stop = timer()
    logger.info("Time spent: %.2fs", stop - start)


def main(args):

    datapath = Path(args.datapath)
    experiment = Path(args.input_file).stem

    cluster_file_name = (
        "kmeans-n" + str(args.neighbors) + "-" + args.config_name +
        "-" + str(args.n_clusters) + "." + args.extension
        )
    print(cluster_file_name)
    output_path = datapath / "output" / experiment / "postprocessed"
    output_path.mkdir(exist_ok=True)
    cluster_file_path = (
        datapath / "output" / experiment / "clustering" / cluster_file_name
        )
    if args.extension == "las":
        point_cloud = io.las(cluster_file_path)
    elif args.extension == "xyz":
        point_cloud = io.xyz(cluster_file_path)
    point_cloud = pd.DataFrame(
        point_cloud,
        columns=("x", "y", "z", "r", "g", "b")
    )

    tree_file = Path(
        "data", "output", experiment,
        "kd-tree-leaf-" + str(args.tree_leaf_size) + ".pkl"
    )
    with open(tree_file, 'rb') as fobj:
        logger.info("Load kd-tree from file...")
        tree = pickle.load(fobj)

    output_filename = (
        "kmeans-n" + str(args.neighbors) + "-" + args.config_name + "-" +
        str(args.n_clusters) + ".xyz"
        )
    output_file_path = Path(
        args.datapath, "output", experiment, "postprocessed", output_filename
        )
    if args.n_processes > 1:
        postprocess_multi(
            point_cloud,
            tree,
            output_file_path,
            nb_neighbors=50,
            radius=None,
            nb_processes=args.n_processes,
            chunksize=args.chunksize
        )
    else:
        postprocess(
            point_cloud,
            tree,
            output_file_path,
            nb_neighbors=50,
            radius=None,
        )


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config-name', help="name of the k-mean configuration"
        )
    parser.add_argument(
        '-d', '--datapath', default="./data",
        help="path of data on file system"
    )
    parser.add_argument(
        '-e', '--extension', default="las",
        help="clustering output file extension"
    )
    parser.add_argument(
        '-i', '--input-file', help="input file"
    )
    parser.add_argument(
        '-k', '--n-clusters', type=int, help="number of clusters"
    )
    parser.add_argument(
        '-m', '--n-processes', type=int, default=1,
        help="Number of process for multiprocessing"
        )
    parser.add_argument(
        '-n', '--neighbors', type=int, help="number neighboring points"
    )
    parser.add_argument(
        '-s', '--chunksize', type=int, help="Size of writing chunks"
        )
    parser.add_argument(
        '-t', '--tree-leaf-size', type=int, help="KD-tree leaf size"
        )

    args = parser.parse_args()
    main(args)
