"""Test config setup
"""

import pytest

import numpy as np

from geo3dfeatures.extract import compute_tree


SEED = 1337
np.random.seed(SEED)


@pytest.fixture
def size():
    return 5000


@pytest.fixture
def line(size):
    """Build x,y,z point clouds as a line for 1D features
    """
    data = np.zeros((size, 3))
    data[:, 0] = np.random.uniform(low=1, high=2, size=size)  # x
    # y close to zero
    data[:, 1] = np.random.normal(loc=0, scale=0.002, size=size)
    # z close to zero
    data[:, 2] = np.random.normal(loc=0, scale=0.002, size=size)
    return data


@pytest.fixture
def plane(size):
    """Build x,y,z point clouds as a plane for 2D features
    """
    data = np.zeros((size, 3))
    # data[:, 0] = 20 + 2 * np.random.uniform(size=size) - 1  # x
    data[:, 0] = np.random.uniform(low=20, high=22, size=size)  # x
    # data[:, 1] = 40 + 2 * np.random.uniform(size=size) - 1  # y
    data[:, 1] = np.random.uniform(low=40, high=42, size=size)  # y
    data[:, 2] = np.random.normal(loc=5, scale=0.002, size=size)  # z with std close to zero
    return data


@pytest.fixture
def sphere(size):
    """Build x,y,z point clouds as a sphere for 3D features
    """
    maxsize = size * 2
    data = np.zeros((maxsize, 3))
    data[:, 0] = np.random.uniform(low=-1, high=1, size=maxsize)  # x
    data[:, 1] = np.random.uniform(low=-1, high=1, size=maxsize)  # y
    data[:, 2] = np.random.uniform(low=-1, high=1, size=maxsize)  # z
    d = data[:, 0] * data[:, 0] + data[:, 1] * data[:, 1] + data[:, 2] * data[:, 2]
    # keep just point inside the sphere
    return data[d <= 1.0][:size]


@pytest.fixture
def ztube(size):
    """small x,y variations along z-axis
    """
    data = np.zeros((size, 3))
    # small x variations
    data[:, 0] = np.random.normal(loc=5, scale=0.002, size=size)
    # small y variations
    data[:, 1] = np.random.normal(loc=10, scale=0.002, size=size)
    data[:, 2] = np.random.uniform(low=10, high=14, size=size)  # z
    return data


@pytest.fixture
def wall(size):
    """High verticality. Plane projection on (x,y) should look like to a straight line.
    """
    data = np.zeros((size, 3))
    data[:, 0] = np.random.uniform(low=1, high=2, size=size)
    data[:, 1] = 2 + 0.5 * data[:, 0]
    data[:, 2] = np.random.uniform(low=10, high=20, size=size)
    return data


@pytest.fixture
def roof(size):
    """Looks like a roof.

    - high elevation
    - plane projection on (x,y) looks like a plane (even a square since max-min are the same of x and y)
    """
    data = np.zeros((size, 3))
    data[:size//2, 0] = np.random.uniform(low=10, high=16, size=size // 2)  # x
    data[:size//2, 1] = np.random.uniform(low=20, high=23, size=size // 2)  # y
    # increase according to y (linearly)
    z0 = 2.
    data[:size//2, 2] = z0 + 1/3 * data[:size//2, 1]
    # translation over y and minus z
    data[size//2:, 0] = np.random.uniform(low=10, high=16, size=size // 2)
    # data[size//2:, 0] = data[:size//2, 0]
    data[size//2:, 1] = 3 + data[:size//2, 1]
    zmax = data[:size//2, 2].max()
    data[size//2:, 2] = 2 * zmax - z0 - 1/3 * data[size//2:, 1]
    return data


@pytest.fixture
def sphere_tree(sphere):
    """Compute a kd-tree for the sphere dataset
    """
    return compute_tree(sphere, leaf_size=100)
