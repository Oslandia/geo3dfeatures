"""I/O module to load/write some point clouds files
"""

import numpy as np


def xyz(fpath, names=None, header=True):
    """Read a .xyz file

    This is a ASCII text file with 3D coordinates and some extra columns

    Parameters
    ----------
    fpath : str
    names : list (None by default)
    header : bool
        If True, there is a header line that must be skipped

    Returns
    -------
    np.array
    """
    return np.loadtxt(fpath, delimiter=' ', skiprows=header)
