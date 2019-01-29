"""Draft module for documenting and plotting test fixtures

'line', 'plane', 'sphere', 'ztube', 'wall' and 'roof' are test fixtures, as
denoted in 'tests/test_fixtures.py'.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

SIZE = 5000

def line(size=SIZE):
    """Build x,y,z point clouds as a line for 1D features
    """
    data = np.zeros((size, 3))
    data[:, 0] = np.random.uniform(low=1, high=2, size=size)  # x
    # y close to zero
    data[:, 1] = np.random.normal(loc=0, scale=0.002, size=size)
    # z close to zero
    data[:, 2] = np.random.normal(loc=0, scale=0.002, size=size)
    return data


def plane(size=SIZE):
    """Build x,y,z point clouds as a plane for 2D features
    """
    data = np.zeros((size, 3))
    # data[:, 0] = 20 + 2 * np.random.uniform(size=size) - 1  # x
    data[:, 0] = np.random.uniform(low=20, high=22, size=size)  # x
    # data[:, 1] = 40 + 2 * np.random.uniform(size=size) - 1  # y
    data[:, 1] = np.random.uniform(low=40, high=42, size=size)  # y
    data[:, 2] = np.random.normal(loc=5, scale=0.002, size=size)  # z with std close to zero
    return data


def sphere(size=SIZE):
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


def ztube(size=SIZE):
    """small x,y variations along z-axis
    """
    data = np.zeros((size, 3))
    # small x variations
    data[:, 0] = np.random.normal(loc=5, scale=0.002, size=size)
    # small y variations
    data[:, 1] = np.random.normal(loc=10, scale=0.002, size=size)
    data[:, 2] = np.random.uniform(low=10, high=14, size=size)  # z
    return data


def wall(size=SIZE):
    """High verticality. Plane projection on (x,y) should look like to a straight line.
    """
    data = np.zeros((size, 3))
    data[:, 0] = np.random.uniform(low=1, high=2, size=size)
    data[:, 1] = 2 + 0.5 * data[:, 0]
    data[:, 2] = np.random.uniform(low=10, high=20, size=size)
    return data


def roof(size=SIZE):
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


def plot_fixture(fixture):
    """Plot a scatter plot that illustrates the fixture representation
    as a 3d point cloud

    The three eigenvectors are materialized on the plot by three corresponding
    lines that start from the point cloud mean. The direction of the vectors is
    given by definition of the PCA, whilst their length is artificially given
    by the component explained variance

    See:
    https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

    Maybe a smarter arrow implementation (does not use sklearn PCA) at:
    https://sebastianraschka.com/Articles/2014_pca_step_by_step.html

    Parameters
    ----------
    fixture : str
        Name of the fixture, amongst 'line', 'plane', 'sphere', 'ztube',
    'wall' or 'roof'
    """
    if fixture == "line":
        data = line()
    elif fixture == "plane":
        data = plane()
    elif fixture == "sphere":
        data = sphere()
    elif fixture == "ztube":
        data = ztube()
    elif fixture == "wall":
        data = wall()
    elif fixture == "roof":
        data = roof()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    zord = 1
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], zorder=zord)
    if fixture == "line":
        ax.set_ylim((-0.5, 0.5))
        ax.set_zlim((-0.5, 0.5))
    elif fixture == "plane":
        ax.set_zlim((4.5, 5.5))
    elif fixture == "ztube":
        ax.set_xlim((4.5, 5.5))
        ax.set_ylim((9.5, 10.5))
    ax.set_title(fixture)
    ax.set_xlabel("x_values")
    ax.set_ylabel("y_values")
    ax.set_zlabel("z_values")
    pca = PCA().fit(data)
    mean_x, mean_y, mean_z = np.mean(data, axis=0)
    zord = 10
    ax.scatter(mean_x, mean_y, mean_z, 'k', zorder=zord)
    colors = ['r', 'g', 'y']
    for length, eigvec, c in zip(
            pca.explained_variance_, pca.components_, colors
    ):
        zord += 10
        origin = pca.mean_
        dest = origin + eigvec * 3 * np.sqrt(length)
        ax.plot(
            [origin[0], dest[0]],
            [origin[1], dest[1]],
            [origin[2], dest[2]],
            color=c,
            lw=3,
            zorder=zord
        )
    ax.legend(["eigenvector 1", "eigenvector 2", "eigenvector 3"])
    fig.tight_layout()
    fixture_path = Path("docs", "images", fixture + ".png")
    fig.savefig(fixture_path)
