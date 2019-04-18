"""Geometry features.

List of functions which extracts 1D, 2D or 3D geometry features from point
clouds.

See:
- Martin Weinmann, Boris Jutzi, Stefan Hinz, Clément Mallet,
2015. Semantic point cloud interpretation based on optimal neighborhoods,
relevant features and efficient classifiers. ISPRS Journal of
Photogrammetry and Remote Sensing, vol 105, pp 286-304.
- Brodu, N., Lague D., 2011. 3D Terrestrial lidar data classification of
complex natural scenes using a multi-scale dimensionality criterion:
applications in geomorphology. arXiV:1107.0550v3.


"""

import math

import numpy as np
import pandas as pd


def accumulation_2d_neighborhood(
        point_cloud, extra_columns=None, bin_size=1
):
    """Compute accumulation features as a new way of designing a
        2D-neighborhood, following the description of (Weinmann *et al.*,
        2015): such features are built by binning the 2D-space, and evaluating
        the number of points contained, the Z-range and the Z-standard
        deviation in each bin. The features are then assigned to the points
        regarding the bin that they belong to.

    Parameters
    ----------
    point_cloud : numpy.array
        Coordinates of all points within the point cloud
    extra_columns : tuple
        List of extra input column names; its length must correspond to "point_cloud"
        number of columns minus 3 (x, y, z are supposed to be the first three
        columns)
    bin_size : int
        Size of each squared bin edge (in meter)

    Returns
    -------
    pandas.DataFrame
        Set of features built through binning process, for each point within
    the cloud

    """
    input_columns = ("x", "y", "z")
    if extra_columns is not None:
        input_columns += extra_columns
    else:
        # you only take into account the x, y, z coordinates from the point cloud data
        point_cloud = point_cloud[:, :3]
    if len(input_columns) != point_cloud.shape[1]:
        raise ValueError("Column names does not match the point cloud shape.")
    df = pd.DataFrame(point_cloud, columns=input_columns)
    xmin, xmax = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
    ymin, ymax = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
    xbins = np.arange(xmin, xmax + bin_size + 1e-3, bin_size)
    df["xbin"] = pd.cut(df.x, xbins, right=False)
    ybins = np.arange(ymin, ymax + bin_size + 1e-3, bin_size)
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


def normalize(a):
    """Compute and normalize values in "a".

    Parameters
    ----------
    a : numpy.array

    Returns
    -------
    numpy.array
    """
    return a / np.sum(a)


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


def triangle_variance_space(eigenvalues):
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
    https://en.wikipedia.org/wiki/Barycentric_coordinate_system\
    #Conversion_between_barycentric_and_Cartesian_coordinates

    If you project the three normalized eigenvalues on a 2D plane, i.e. (λ1,
    λ2), you get the triangle with these following coordinates:

    (1/3, 1/3), (1/2, 1/2), (1, 0)

    Thus you can build the T matrix and get the barycentric coordinates with
    the T^{-1}(r - r_3) formula.

    Parameters
    ----------
    eigenvalues : numpy.array
        Set of normalized eigenvalues; must be (1, 3)-shaped

    Returns
    -------
    list
        First two barycentric coordinates in the variance space (triangle)
    """
    alpha = eigenvalues[0] - eigenvalues[1]
    beta = 2 * eigenvalues[0] + 4 * eigenvalues[1] - 2
    return [alpha, beta]


def curvature_change(eigenvalues):
    """Compute the curvature change in the local dataset

    See (Weinmann et al, 2015)

    Parameters
    ----------
    eigenvalues : numpy.array
        Set of normalized eigenvalues; must be (1, 3)-shaped

    Returns
    -------
    float
        Curvature change
    """
    return eigenvalues[2]


def linearity(eigenvalues):
    """Compute the linearity of the local dataset

    See (Weinmann et al, 2015)

    Parameters
    ----------
    eigenvalues : numpy.array
        Set of normalized eigenvalues; must be (1, 3)-shaped

    Returns
    -------
    float
        Linearity
    """
    return (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0]


def planarity(eigenvalues):
    """Compute the planarity of the local dataset

    See (Weinmann et al, 2015)

    Parameters
    ----------
    eigenvalues : numpy.array
        Set of normalized eigenvalues; must be (1, 3)-shaped

    Returns
    -------
    float
        Planarity
    """
    return (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]


def scattering(eigenvalues):
    """Compute the scattering of the local dataset

    See (Weinmann et al, 2015)

    Parameters
    ----------
    eigenvalues : numpy.array
        Set of normalized eigenvalues; must be (1, 3)-shaped

    Returns
    -------
    float
        Scattering
    """
    return eigenvalues[2] / eigenvalues[0]


def omnivariance(eigenvalues):
    """Compute the omnivariance of the local dataset

    See (Weinmann et al, 2015)

    Parameters
    ----------
    eigenvalues : numpy.array
        Set of normalized eigenvalues; must be (1, 3)-shaped

    Returns
    -------
    float
        Omnivariance
    """
    return np.prod(eigenvalues) ** (1 / 3)


def anisotropy(eigenvalues):
    """Compute the anisotropy of the local dataset

    See (Weinmann et al, 2015)

    Parameters
    ----------
    eigenvalues : numpy.array
        Set of normalized eigenvalues; must be (1, 3)-shaped

    Returns
    -------
    float
        Anisotropy
    """
    return (eigenvalues[0] - eigenvalues[2]) / eigenvalues[0]


def eigenentropy(eigenvalues):
    """Compute the eigenentropy

    Parameters
    ----------
    eigenvalues : numpy.array
        Set of normalized eigenvalues; must be (1, 3)-shaped

    Returns
    -------
    float
        Entropy of the dataset
    """
    return -1 * np.sum(eigenvalues * np.log(eigenvalues))


def val_sum(a):
    """Compute the sum of items in "a"

    Parameters
    ----------
    a : numpy.array
        Data

    Returns
    -------
    float
        Sum of "a" items
    """
    return np.sum(a)


def eigenvalue_ratio_2D(eigenvalues):
    """Compute the 2D eigenvalue ratio

    Parameters
    ----------
    eigenvalues : numpy.array
        Array of eigenvalues; must be (1, 2)-shaped
    """
    return eigenvalues[1] / eigenvalues[0]


def val_range(a):
    """Compute the range between minimal and maximal values of "a"

    Parameters
    ----------
    a : numpy.array
        Data

    Returns
    -------
    float
        Range of "a"
    """
    return np.ptp(a)


def std_deviation(a):
    """Compute the standard deviation of values contained in "a"

    Parameters
    ----------
    a : numpy.array
        Data

    Returns
    -------
    float
        Standard deviation of "a"
    """
    return np.std(a)


def radius_3D(distances):
    """Compute the max distance between a point and its neighbors, assuming
    that "distances" contains the euclidian distances

    Parameters
    ----------
    distances : numpy.array

    Returns
    -------
        3D radius associated to the point of interest
    """
    return np.max(distances)


def radius_2D(point, neighbors_2D):
    """Compute the max distance between 'point' and its neighbors, measured as
    the squared euclidian distance between 'point' and the furthest neighbor

    Parameters
    ----------
    point : numpy.array
        (x, y) coordinates of the point of interest
    2D_neighbors : numpy.array
        Set of 2D neighboring points

    Returns
    -------
    float
        2D radius associated to "point"
    """
    return np.max(np.sum((neighbors_2D - point) ** 2, axis=1))


def density_3D(radius, nb_neighbors):
    """Compute the density in a 3D space, as the ratio between point quantity
    and 3D volume

    Parameters
    ----------
    radius : float
        Radius of the sphere of interest around the considered point
    nb_neighbors : int
        Number of points in the neighborhood

    Returns
    -------
    float
        3D point density in the considered volume
    """
    return (nb_neighbors + 1) / ((4 / 3) * math.pi * radius ** 3)


def density_2D(radius, nb_neighbors):
    """Compute the density in a 2D space, as the ratio between point quantity
    and 2D area

    Parameters
    ----------
    radius : float
        Radius of the area of interest around the considered point
    nb_neighbors : int
        Number of points in the neighborhood

    Returns
    -------
    float
        2D point density in the considered area
    """
    return (nb_neighbors + 1) / (math.pi * radius ** 2)


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
