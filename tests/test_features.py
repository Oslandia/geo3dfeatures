import pytest

import numpy as np

from sklearn.decomposition import PCA

from geo3dfeatures.features import (accumulation_2d_neighborhood, triangle_variance_space,
                                    compute_3D_features)


SEED = 1337
np.random.seed(SEED)
SIZE = 5000


@pytest.fixture
def line(size=SIZE):
    """Build x,y,z point clouds as a line for 1D features
    """
    data = np.zeros((size, 3))
    data[:, 0] = 2 * np.random.uniform(size=size) - 1  # x
    # y close to zero
    data[:, 1] = np.random.normal(loc=0, scale=0.002, size=size)
    # z close to zero
    data[:, 2] = np.random.normal(loc=0, scale=0.002, size=size)
    return data


@pytest.fixture
def plane(size=SIZE):
    """Build x,y,z point clouds as a plane for 2D features
    """
    data = np.zeros((size, 3))
    data[:, 0] = 20 + 2 * np.random.uniform(size=size) - 1  # x
    data[:, 1] = 40 + 2 * np.random.uniform(size=size) - 1  # y
    data[:, 2] = 5 + np.random.normal(loc=0, scale=0.002, size=size)  # z close to zero
    return data


@pytest.fixture
def sphere(size=SIZE):
    """Build x,y,z point clouds as a sphere for 3D features
    """
    maxsize = size * 2
    data = np.zeros((maxsize, 3))
    data[:, 0] = 2 * np.random.uniform(size=maxsize) - 1  # x
    data[:, 1] = 2 * np.random.uniform(size=maxsize) - 1  # y
    data[:, 2] = 2 * np.random.uniform(size=maxsize) - 1  # z
    d = data[:, 0] * data[:, 0] + data[:, 1] * data[:, 1] + data[:, 2] * data[:, 2]
    # keep just point inside the sphere
    return data[d <= 1.0][:size]


def test_accumulation_2d_features(line, plane, sphere):
    acc1D = accumulation_2d_neighborhood(line)
    acc2D = accumulation_2d_neighborhood(plane)
    acc3D = accumulation_2d_neighborhood(sphere)
    # a large density of points for the line
    assert acc1D["count"].mean() > acc2D["count"].mean()
    assert acc1D["count"].mean() > acc3D["count"].mean()
    # low z-std and z-range for line and plane
    assert abs(acc1D["std"].mean() - acc2D["std"].mean()) < 1e-4
    # higher z-std and z-range for sphere
    assert acc3D["std"].mean() > acc1D["std"].mean()
    assert acc3D["std"].mean() > acc2D["std"].mean()
    assert acc3D["z_range"].mean() > acc1D["z_range"].mean()
    assert acc3D["z_range"].mean() > acc2D["z_range"].mean()


def test_sum_triangle_variance_space(plane):
    """Test the values of the barycentric coordinates (variance space).

    The function returns the first two barycentric coordinates but you should have
    `alpha + beta + gamma = 1`
    """
    pca = PCA().fit(plane)
    alpha, beta = triangle_variance_space(pca)
    assert alpha + beta <= 1.0


def test_triangle_variance_space_1D_case(line):
    """Test the values of the barycentric coordinates (variance space).

    'alpha' must be >> to 'beta'
    """
    pca = PCA().fit(line)
    alpha, beta = triangle_variance_space(pca)
    assert alpha > beta
    assert abs(alpha - 1) < 1e-3


def test_triangle_variance_space_2D_case(plane):
    """Test the values of the barycentric coordinates (variance space).

    beta must be > alpha and close to 1.0
    """
    pca = PCA().fit(plane)
    alpha, beta = triangle_variance_space(pca)
    assert alpha < beta
    assert beta >= 0.95


def test_triangle_variance_space_3D_case(sphere):
    """Test the values of the barycentric coordinates (variance space).

    alpha and beta must be close to 0. (so gamme close to 1.)
    """
    pca = PCA().fit(sphere)
    alpha, beta = triangle_variance_space(pca)
    assert 1 - (alpha + beta) >= 0.95


def test_3d_features_line(line):
    """Test curvature change, linearity, planarity and scattering for a line.
    """
    pca = PCA().fit(line)
    curvature_change, linearity, planarity, scattering, *rest = compute_3D_features(
        pca
        )
    # close to 1
    assert linearity >= 0.95
    # close to zero
    assert curvature_change <= 5e-3
    assert planarity <= 0.05
    scattering <= 0.05


def test_volume_features_plane(plane):
    """Test curvature change, linearity, planarity and scattering for a plane.
    """
    pca = PCA().fit(plane)
    curvature_change, linearity, planarity, scattering, *rest = compute_3D_features(
        pca
        )
    # close to zero
    assert curvature_change <= 5e-3
    # close to 1
    assert planarity >= 0.95
    # close to 0
    linearity <= 0.05
    scattering <= 0.05


def test_3d_features_sphere(sphere):
    """Test curvature change, linearity, planarity and scattering for a sphere.
    """
    pca = PCA().fit(sphere)
    curvature_change, linearity, planarity, scattering, *rest = compute_3D_features(
        pca
        )
    # close to 1/3
    assert abs(curvature_change - 1/3) <= 5e-3
    # close to 1
    assert scattering >= 0.95
    # close to 0
    linearity <= 0.05
    planarity <= 0.05
