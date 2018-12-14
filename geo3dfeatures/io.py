"""I/O module to load/write some point clouds files
"""


import csv

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


def write_features(fpath, gen):
    """Write the fields from a data generator into a .csv file

    Parameters
    ----------
    fpath : str
        Path of the output file
    gen : generator
        Data stored as an ordered dict
    """
    with open(fpath, 'w') as fobj:
        # get the first data to get the field names
        first = next(gen)
        writer = csv.DictWriter(fobj, first.keys())
        writer.writeheader()
        writer.writerow(first)
        for row in gen:
            writer.writerow(row)
