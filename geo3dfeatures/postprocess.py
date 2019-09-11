"""K-means output post-processing

Compute our own mode method inspired from scipy.stats source code:
- https://github.com/scipy/scipy/blob/master/scipy/stats/stats.py#L609

Compute the mode in an alternative pure-numpy way:
- https://stackoverflow.com/questions/12297016/how-to-find-most-frequent-values-in-numpy-ndarray
"""

import daiquiri
import numpy as np

from geo3dfeatures import extract


logger = daiquiri.getLogger(__name__)


def batch_points(points, batch_size):
    """Batch the point structure so as to split the postprocessing phase

    Parameters
    ----------
    points : np.array
        Full point structure
    batch_size : int
        Number of points to consider in each subsample

    Yields
    ------
    np.array
        Point subsample
    """
    for value in range(0, points.shape[0], batch_size):
        yield points[value:(value+batch_size)]


def postprocess_batch_labels(
        point_generator, batch_size, labels, tree, n_neighbors=None, radius=None
):
    """Postprocess the clustered labels by considering a batched point cloud
        for memory-saving purpose

    Parameters
    ----------
    point_generator : iterator
        Generator of points, built to reduce the memory footprint
    batch_size : int
        Number of points in each batch, by definition (one passes this argument
    in order to avoid confusion for the last item)
    labels : np.array
        Set of output labels, after k-mean algorithm
    tree : scipy.spatial.ckdtree.cKDTree
        Spatial kd-tree designed to identify the clustered point neighbors
    nb_neighbors : int
        Number of neighbors in each point neighborhood
    radius : float
        Radius that defines the neighboring ball around a given point
    """
    new_labels = np.zeros_like(labels)
    for idx, item in enumerate(point_generator):
        _, neighbors = extract.request_tree(item, tree, n_neighbors, radius)
        point_neighborhoods = labels[neighbors]
        u, indices = np.unique(point_neighborhoods, return_inverse=True)
        new_clusters = u[
            np.argmax(
                np.apply_along_axis(
                    np.bincount, 1, indices.reshape(point_neighborhoods.shape),
                    None, np.max(indices) + 1),
                axis=1)
        ]
        new_labels[idx*batch_size:(idx+1)*batch_size] = new_clusters
    return new_labels
