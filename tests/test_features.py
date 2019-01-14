import pytest

import numpy as np

from geo3dfeatures.extract import build_accumulation_features


SEED = 1337
np.random.seed(SEED)
SIZE = 5_000


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
    data[:, 0] = 2 * np.random.uniform(size=size) - 1  # x
    data[:, 1] = 2 * np.random.uniform(size=size) - 1  # y
    data[:, 2] = np.random.normal(loc=0, scale=0.002, size=size)  # z close to zero
    return data


@pytest.fixture
def sphere(size=SIZE):
    """Build x,y,z point clouds as a sphere for 3D features
    """
    size = size * 2
    data = np.zeros((size, 3))
    data[:, 0] = 2 * np.random.uniform(size=size) - 1  # x
    data[:, 1] = 2 * np.random.uniform(size=size) - 1  # y
    data[:, 2] = 2 * np.random.uniform(size=size) - 1  # z
    d = data[:, 0] * data[:, 0] + data[:, 1] * data[:, 1] + data[:, 2] * data[:, 2]
    # keep just point inside the sphere
    return data[d <= 1.0][:size]


def test_accumulation_features(line, plane, sphere):
    acc1D = build_accumulation_features(line)
    acc2D = build_accumulation_features(plane)
    acc3D = build_accumulation_features(sphere)
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