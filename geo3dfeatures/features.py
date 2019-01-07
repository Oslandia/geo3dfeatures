"""List of functions which extracts 1D, 2D or 3D geometry features from point clouds.
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
