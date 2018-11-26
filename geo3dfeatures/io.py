"""I/O module to load/write some point clouds files
"""

import numpy as np


def xyz(fpath, names=None):
    """Read a .xyz file

    This is a ASCII text file with 3D coordinates and some extra columns

    Parameters
    ----------
    fpath : str
    names : list (None by default)

    Returns
    -------
    np.array
    """
    return np.loadtxt(fpath, delimiter=' ', skiprows=1)
