import pytest

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from scipy.spatial import cKDTree as KDTree

from geo3dfeatures.features import (accumulation_2d_neighborhood,
                                    triangle_variance_space,
                                    sum_normalize,
                                    curvature_change,
                                    linearity,
                                    planarity,
                                    scattering,
                                    verticality_coefficient,
                                    val_sum, eigenvalue_ratio_2D,
                                    val_range, std_deviation,
                                    radius_2D, radius_3D,
                                    density_2D, density_3D,
                                    eigenentropy)
from geo3dfeatures.extract import request_tree


def _neighbors(data, reference_point, neighbors_size=50, leaf_size=100):
    """Compute the closest neighbors with a kd-tree.

    Return the neighborhood and the distance
    """
    kd_tree = KDTree(data, leaf_size)
    return request_tree(reference_point, kd_tree, nb_neighbors=neighbors_size)


def test_accumulation_2d_features(line, plane, sphere):
    coords = list("xyz")
    acc1D = accumulation_2d_neighborhood(pd.DataFrame(line, columns=coords))
    acc2D = accumulation_2d_neighborhood(pd.DataFrame(plane, columns=coords))
    acc3D = accumulation_2d_neighborhood(pd.DataFrame(sphere, columns=coords))
    # a large density of points for the line
    assert acc1D["bin_density"].mean() > acc2D["bin_density"].mean()
    assert acc1D["bin_density"].mean() > acc3D["bin_density"].mean()
    # low z-std and z-range for line and plane
    assert abs(acc1D["bin_z_std"].mean() - acc2D["bin_z_std"].mean()) < 1e-4
    # higher z-std and z-range for sphere
    assert acc3D["bin_z_std"].mean() > acc1D["bin_z_std"].mean()
    assert acc3D["bin_z_std"].mean() > acc2D["bin_z_std"].mean()
    assert acc3D["bin_z_range"].mean() > acc1D["bin_z_range"].mean()
    assert acc3D["bin_z_range"].mean() > acc2D["bin_z_range"].mean()


def test_sum_triangle_variance_space(plane):
    """Test the values of the barycentric coordinates (variance space).

    The function returns the first two barycentric coordinates but you should have
    `alpha + beta + gamma = 1`
    """
    pca = PCA().fit(plane)
    norm_eigenvalues = sum_normalize(pca.singular_values_ ** 2)
    print(pca.singular_values_)
    print(pca.singular_values_ ** 2)
    print(norm_eigenvalues)
    alpha, beta = triangle_variance_space(norm_eigenvalues)
    assert alpha + beta <= 1.0


def test_triangle_variance_space_1D_case(line):
    """Test the values of the barycentric coordinates (variance space).

    'alpha' must be >> to 'beta'
    """
    pca = PCA().fit(line)
    norm_eigenvalues = sum_normalize(pca.singular_values_ ** 2)
    alpha, beta = triangle_variance_space(norm_eigenvalues)
    assert alpha > beta
    assert abs(alpha - 1) < 1e-3


def test_triangle_variance_space_2D_case(plane):
    """Test the values of the barycentric coordinates (variance space).

    beta must be > alpha and close to 1.0
    """
    pca = PCA().fit(plane)
    norm_eigenvalues = sum_normalize(pca.singular_values_ ** 2)
    alpha, beta = triangle_variance_space(norm_eigenvalues)
    assert alpha < beta
    assert beta >= 0.95


def test_triangle_variance_space_3D_case(sphere):
    """Test the values of the barycentric coordinates (variance space).

    alpha and beta must be close to 0. (so gamme close to 1.)
    """
    pca = PCA().fit(sphere)
    norm_eigenvalues = sum_normalize(pca.singular_values_ ** 2)
    alpha, beta = triangle_variance_space(norm_eigenvalues)
    assert 1 - (alpha + beta) >= 0.95


def test_3d_features_line(line):
    """Test curvature change, linearity, planarity and scattering for a line.
    """
    pca = PCA().fit(line)
    norm_eigenvalues = sum_normalize(pca.singular_values_ ** 2)
    # close to 1
    assert linearity(norm_eigenvalues) >= 0.9
    # close to zero
    assert curvature_change(norm_eigenvalues) <= 5e-3
    assert planarity(norm_eigenvalues) <= 0.05
    assert scattering(norm_eigenvalues) <= 0.05


def test_3d_features_plane(plane):
    """Test curvature change, linearity, planarity and scattering for a plane.
    """
    pca = PCA().fit(plane)
    norm_eigenvalues = sum_normalize(pca.singular_values_ ** 2)
    # close to zero
    assert curvature_change(norm_eigenvalues) <= 5e-3
    # close to 1
    assert planarity(norm_eigenvalues) >= 0.9
    # close to 0
    assert linearity(norm_eigenvalues) <= 0.1
    assert scattering(norm_eigenvalues) <= 0.1


def test_3d_features_sphere(sphere):
    """Test curvature change, linearity, planarity and scattering for a sphere.
    """
    pca = PCA().fit(sphere)
    norm_eigenvalues = sum_normalize(pca.singular_values_ ** 2)
    # close to 1/3
    assert abs(curvature_change(norm_eigenvalues) - 1/3) <= 0.05
    # close to 1
    assert scattering(norm_eigenvalues) >= 0.9
    # close to 0
    assert linearity(norm_eigenvalues) <= 0.05
    assert planarity(norm_eigenvalues) <= 0.05


def test_2d_features_ztube(ztube):
    """Projection on (x,y) look like a tiny noisy circle

    Low and close eigenvalues
    """
    pca = PCA().fit(ztube[:, :2])
    eigenvalues = pca.singular_values_ ** 2
    assert val_sum(eigenvalues) < 1.0
    assert abs(eigenvalue_ratio_2D(eigenvalues) - 1) <= 0.1


def test_2d_features_wall(wall):
    """Projection on (x,y) look like a straight line

    - High first eigen values. A very low lambda_2 / lambda_1 ratio.
    - The value of the sum is quite close to the first eigen value.
    """
    pca = PCA().fit(wall[:, :2])
    eigenvalues = pca.singular_values_ ** 2
    assert eigenvalue_ratio_2D(eigenvalues) <= 1e-5
    assert abs(val_sum(eigenvalues) - eigenvalues[0]) <= 0.05


def test_2d_features_roof(roof):
    """Projection on (x,y) look like a plane

    - The sum of eigen values is quite high
    - The values of the eigenvalues are almost equal. The ratio is close to 1
    """
    pca = PCA().fit(roof[:, :2])
    eigenvalues = pca.singular_values_ ** 2
    assert abs(eigenvalue_ratio_2D(eigenvalues) - 1) <= 0.1
    assert val_sum(eigenvalues) > 1000


def test_3D_properties_plane_and_sphere_comparison(plane, sphere):
    """Compare all 3D properties between a sphere and a plane.

    """
    index = 42
    reference_point_plane = plane[index]
    reference_point_sphere = sphere[index]
    dist_plane, neighbors_plane = _neighbors(plane, reference_point_plane)
    dist_sphere, neighbors_sphere  = _neighbors(sphere, reference_point_sphere)
    radius_plane = radius_3D(dist_plane)
    radius_sphere = radius_3D(dist_sphere)
    assert radius_sphere > radius_plane
    density_plane = density_3D(radius_plane, len(neighbors_plane))
    density_sphere = density_3D(radius_sphere, len(neighbors_sphere))
    assert density_plane > density_sphere
    z_range_plane = val_range(plane[neighbors_plane, 2])
    z_range_sphere = val_range(sphere[neighbors_sphere, 2])
    assert z_range_sphere > z_range_plane
    std_deviation_plane = std_deviation(plane[neighbors_plane, 2])
    std_deviation_sphere = std_deviation(sphere[neighbors_sphere, 2])
    assert std_deviation_sphere > std_deviation_plane
    vcoef_plane = verticality_coefficient(PCA().fit(plane))
    vcoef_sphere = verticality_coefficient(PCA().fit(sphere))
    assert vcoef_plane < vcoef_sphere


def test_verticality_coefficient_line(line):
    """Test verticality coefficient for a line-shaped point cloud

    As the point cloud is constrained only over the x-axis, the first
    eigenvector is strongly influenced by the x-components. However the other
    eigenvectors may have any shape, hence the verticality coefficient may take
    any value between 0 and 1. The test scope is limited to the definition
    domain.
    """
    pca = PCA().fit(line)
    vcoef = verticality_coefficient(pca)
    assert vcoef >= 0 and vcoef <= 1


def test_verticality_coefficient_plane(plane):
    """Test verticality coefficient for a plane-shaped point cloud

    As data are spread over x and y-axis, the two first eigenvectors are
    supposed to have small z-components, hence the third eigenvector has a high
    z-component. So the verticality coefficient must be small (let say, smaller
    than 0.01).
    """
    pca = PCA().fit(plane)
    vcoef = verticality_coefficient(pca)
    assert vcoef >= 0 and vcoef < 0.01


def test_verticality_coefficient_sphere(sphere):
    """Test verticality coefficient for a sphere-shaped point cloud

    As the point cloud is not constrained over x-, y- or z- axis, the
    verticality coefficient may take any value between 0 and 1. The test scope
    is limited to the definition domain.
    """
    pca = PCA().fit(sphere)
    vcoef = verticality_coefficient(pca)
    assert vcoef >= 0 and vcoef <= 1


def test_verticality_coefficient_ztube(ztube):
    """Test verticality coefficient for a ztube-shaped point cloud

    As the point cloud is not constrained over x-, y- or z- axis, the
    verticality coefficient may take any value between 0 and 1. The test scope
    is limited to the definition domain.
    """
    pca = PCA().fit(ztube)
    vcoef = verticality_coefficient(pca)
    assert vcoef <= 1 and vcoef > 1 - 0.01


def test_verticality_coefficient_wall(wall):
    """Test verticality coefficient for a sphere-shaped point cloud

    As the point cloud is not constrained over x-, y- or z- axis, the
    verticality coefficient may take any value between 0 and 1. The test scope
    is limited to the definition domain.
    """
    pca = PCA().fit(wall)
    vcoef = verticality_coefficient(pca)
    assert vcoef <= 1 and vcoef > 1 - 0.01


def test_verticality_coefficient_roof(roof):
    """Test verticality coefficient for a sphere-shaped point cloud

    As the point cloud is not constrained over x-, y- or z- axis, the
    verticality coefficient may take any value between 0 and 1. The test scope
    is limited to the definition domain.
    """
    pca = PCA().fit(roof)
    vcoef = verticality_coefficient(pca)
    assert vcoef >= 0 and vcoef <= 0.01


def test_empty_density_2D():
    """Test density_2D function with empty neighborhood, i.e. when radius is
    equal to 0
    """
    dummy_neighbor_number = 1
    assert density_2D(0, dummy_neighbor_number) is None


def test_empty_density_3D():
    """Test density_3D function with empty neighborhood, i.e. when radius is
    equal to 0
    """
    dummy_neighbor_number = 1
    assert density_3D(0, dummy_neighbor_number) is None


def test_eigenentropy_with_null_eigenvalue():
    """Test eigenentropy computation when at least one eigenvalue is equal to 0
    """

    with pytest.warns(None) as warning_record:
        eigen1 = eigenentropy(np.array([2.43, 0.96, 0.0]))
    eigen2 = eigenentropy(np.array([2.43, 0.96]))
    assert len(warning_record) == 0  # Test that no warning is raised
    assert eigen1 == eigen2


def test_radius_2D():
    x = np.array([[0., 0., 0.],
                  [1., 0., 1.],
                  [0., 0., 1.],
                  [1., 1., 0.],
                  [2., 2., 2.],
                  [0., 1., 10.],
                  [-0.05, 1., 2.],
                  [4., -3., 0.]])
    point = x[0]
    expected = 5.0
    result = radius_2D(point[:2], x[:, :2])
    assert abs(result - expected) <= 1e-4
