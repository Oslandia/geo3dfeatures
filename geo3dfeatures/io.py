"""I/O module to load/write some point clouds files
"""


import csv
import laspy

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


def las(fpath):
    """Read a .las file with ̀laspy` package

    Parameters
    ----------
    fpath : str
        Path of the input file

    Returns
    -------
    numpy.array
        x, y, z point coordinates as well as r, g, b color features stored in
    an array

    """
    input_file = laspy.file.File(str(fpath), mode="r")
    data = np.array([input_file.x, input_file.y, input_file.z,
                     input_file.red, input_file.green, input_file.blue]).T
    return data


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
