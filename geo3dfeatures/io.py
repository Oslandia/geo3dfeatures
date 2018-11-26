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


def write_features(fpath, gen, columns, sep=","):
    """Write the data contained into generator in a .csv file

    Parameters
    ----------
    fpath : str
        Path of the file that must be written on the file system
    gen : generator
        Data stored as a generator
    columns : list
        Header of the .csv file
    sep : str
        Separator between data items used in the .csv file

    """
    with open(fpath, 'w') as fobj:
        fobj.write(sep.join(columns))
        fobj.write("\n")
        for row in gen:
            fobj.write(sep.join(str(x) for x in row))
            fobj.write("\n")
