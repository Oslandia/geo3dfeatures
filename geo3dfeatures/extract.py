"""Extract some features.
"""

import numpy as np

from sklearn.decomposition import PCA


def _pca(data):
    """Carry out a PCA on a set of 3D points
    """
    return PCA(n_components=3).fit(data)


def norm(sample):
    return (sample - sample.mean(axis=0)) / sample.std(axis=0)


def features3d(sing_values):
    r = sing_values / sing_values.sum()
    # canupo refers to a barycentric coords.
    # Formula on wikipedia page for barycentric coordinates
    # using directly the triangle in %variance space, they simplify a lot
    # c = 1 - a - b
    return {'a': r[0] - r[1],
            'b': 2 * r[0] + 4 * r[1] - 2}


def featuregen(points, scene, tree, neighbors_num):
    """
    Parameters
    ----------
    points : generator
    scene : numpy.array
    tree : sklearn.KDTree
    neighbors_num : int
    """
    for row in points:
        point = row[[0, 1, 2]][np.newaxis, :]  # tree.query needs a 2D array
        # find nearest neighbors
        dist, ind = tree.query(point, k=neighbors_num)
        dist, ind = dist.squeeze(), ind.squeeze()
        sample = norm(scene[ind][: ,[0, 1, 2]])
        lbda = _pca(sample).singular_values_
        f3d = features3d(lbda)
        a = f3d['a']
        b = f3d['b']
        C = lbda[-1] / lbda.sum()
        L = (lbda[0] - lbda[1]) / lbda[0]
        P = (lbda[1] - lbda[2]) / lbda[1]
        S = lbda[-1] / lbda[0]
        r = [lbda[0], lbda[1], lbda[2], a, b, C, L, P, S, dist.mean()] + row[3:].tolist()
        yield r
