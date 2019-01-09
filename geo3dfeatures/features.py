"""Geometry features.

List of functions which extracts 1D, 2D or 3D geometry features from point clouds.
"""

import numpy as np
import pandas as pd


def accumulation_2d_neighborhood(point_cloud, bin_size=0.25, buf=1e-3):
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
        Size of each squared bin edge (in meter)
    buf : float
        Epsilon quantity used for expanding the largest bins and consider max values

    Returns
    -------
    pandas.DataFrame
        Set of features built through binning process, for each point within the cloud
    """
    assert point_cloud.shape[1] == 3
    df = pd.DataFrame(point_cloud, columns=["x", "y", "z"])
    xmin, xmax = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    ymin, ymax = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    xbins = np.arange(xmin, xmax + bin_size + buf, bin_size)
    df["xbin"] = pd.cut(df.x, xbins, right=False)
    ybins = np.arange(ymin, ymax + bin_size + buf, bin_size)
    df["ybin"] = pd.cut(df.y, ybins, right=False)
    aggdf = (
        df.groupby(["xbin", "ybin"])["z"]
        .agg(["count", "min", "max", "std"])
        .reset_index()
    )
    aggdf["z_range"] = aggdf["max"] - aggdf["min"]
    aggdf.drop(columns=["min", "max"], inplace=True)
    return df.merge(aggdf, on=["xbin", "ybin"], how="left").drop(
        columns=["xbin", "ybin"]
    )


def normalized_eigenvalues(pca):
    """Compute and normalized the eigenvalues from a fitted PCA.

    The singular values stored in a PCA instance are the squarred root of
    eigenvalues.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA

    Returns
    -------
    np.ndarray
    """
    eigenvalues = pca.singular_values_ * pca.singular_values_
    return eigenvalues / eigenvalues.sum()


def triangle_variance_space(pca):
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
    pca : sklearn.decomposition.PCA

    Returns
    -------
    list
        First two barycentric coordinates in the variance space (triangle)
    """
    eigenvalues = normalized_eigenvalues(pca)
    a = eigenvalues[0] - eigenvalues[1]
    b = 2 * eigenvalues[0] + 4 * eigenvalues[1] - 2
    return [a, b]
