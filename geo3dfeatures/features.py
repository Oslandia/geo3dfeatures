"""Geometry features.

List of functions which extracts 1D, 2D or 3D geometry features from point clouds.
"""

import math

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
        Coordinates of all points within the point cloud
    bin_size : float
        Size of each squared bin edge (in meter)
    buf : float
        Epsilon quantity used for expanding the largest bins and consider max values

    Returns
    -------
    pandas.DataFrame
        Set of features built through binning process, for each point within the cloud
    """
    df = pd.DataFrame(point_cloud, columns=["x", "y", "z", "r", "g", "b"])
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

    See the wikipedia page
    https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Conversion_between_barycentric_and_Cartesian_coordinates

    If you project the three normalized eigenvalues on a 2D plane, i.e. (λ1, λ2), you
    get the triangle with these following coordinates:

    (1/3, 1/3), (1/2, 1/2), (1, 0)

    Thus you can build the T matrix and get the barycentric coordinates with the
    T^{-1}(r - r_3) formula.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA

    Returns
    -------
    list
        First two barycentric coordinates in the variance space (triangle)
    """
    eigenvalues = normalized_eigenvalues(pca)
    alpha = eigenvalues[0] - eigenvalues[1]
    beta = 2 * eigenvalues[0] + 4 * eigenvalues[1] - 2
    return [alpha, beta]


def compute_3D_features(pca):
    """Build the set of 3D features for a typical 3D point within a local
    neighborhood represented through PCA eigenvalues

    Parameters
    ----------
    pca : sklearn.decompositions.PCA
        PCA computed on the x,y,z coords

    Returns
    -------
    list
    """
    assert pca.n_components_ == 3
    e = normalized_eigenvalues(pca)
    curvature_change = e[2]
    linearity = (e[0] - e[1]) / e[0]
    planarity = (e[1] - e[2]) / e[0]
    scattering = e[2] / e[0]
    omnivariance = e.prod() ** (1 / 3)
    anisotropy = (e[0] - e[2]) / e[0]
    eigenentropy = -1 * np.sum(e * np.log(e))
    return [
        curvature_change,
        linearity,
        planarity,
        scattering,
        omnivariance,
        anisotropy,
        eigenentropy
    ]


def compute_2D_features(pca):
    """Build the set of 2D features for a typical 2D point within a local
    neighborhood represented through PCA eigenvalues

    Parameters
    ----------
    pca : sklearn.decompositions.PCA
        PCA computed on the x,y coords.

    Returns
    -------
    list
    """
    assert pca.n_components_ == 2
    eigenvalues = pca.singular_values_ ** 2
    eigenvalue_sum_2D = sum(eigenvalues)
    eigenvalue_ratio_2D = eigenvalues[1] / eigenvalues[0]
    return [eigenvalue_sum_2D, eigenvalue_ratio_2D]


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
    return [radius, z_range, std_deviation, density]


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


def verticality_coefficient(pca):
    """Verticality score aiming at evaluating how vertical a 3D-point cloud is,
    by considering its decomposition through a PCA.

    See:
    - Martin Weinmann, Boris Jutzi, Stefan Hinz, Clément Mallet,
    2015. Semantic point cloud interpretation based on optimal neighborhoods,
    relevant features and efficient classifiers. ISPRS Journal of
    Photogrammetry and Remote Sensing, vol 105, pp 286-304.
    - Jerome Demantke, Bruno Vallet, Nicolas Paparotidis, 2012. Streamed
    Vertical Rectangle Detection in Terrestrial Laser Scans for Facade Database
    Productions. In ISPRS Annals of the Photogrammetry, Remote Sensing and
    Spatial Information Sciences, Volume 1-3, pp99-104.

    Parameters
    ----------
    pca : sklearn.decomposition.PCA
        Principle Component Analysis output; must have a `.components_`
    attribute that contains decomposition axes in 3D space

    Returns
    -------
    float
        Verticality coefficient, as defined in (Demantke et al, 2012) and
    (Weinmann et al, 2015)
    """
    decomposition_axes = pca.components_
    assert decomposition_axes.shape[1] == 3
    return 1 - abs(decomposition_axes[2, 2])
