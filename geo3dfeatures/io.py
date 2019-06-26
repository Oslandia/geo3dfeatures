"""I/O module to load/write some point clouds files
"""


import csv

import laspy

from plyfile import PlyData

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
    return np.loadtxt(fpath, delimiter=" ", skiprows=header)


def las(fpath):
    """Read a .las file with `laspy` package

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
    input_file = laspy.file.File(fpath, mode="r")
    # in case the RGB were uint16 encoded.
    # note : a color channel on uint16 can have valurs from 0 to 65535, see np.iinfo('uint16').max
    if input_file.red.dtype == np.dtype('uint16'):
        factor = 256
    else:
        factor = 1
    data = np.vstack(
        (
            input_file.x,
            input_file.y,
            input_file.z,
            input_file.red / factor,
            input_file.green / factor,
            input_file.blue / factor,
        )
    )
    return data.transpose()


def ply(fpath):
    """Read .ply file with 'plyfile'.

    For now, just read the (x, y, z) coordinates.

    Parameters
    ----------
    fpath : str
        Path of the input file

    Returns
    -------
    numpy.array
        x, y, z point coordinates
    """
    reader = PlyData.read(fpath)
    vertex = reader["vertex"]
    result = np.array([vertex["x"], vertex["y"], vertex["z"]])
    return np.transpose(result)


def write_features(fpath, gen):
    """Write the fields from a data generator into a .csv file

    Parameters
    ----------
    fpath : str
        Path of the output file
    gen : generator
        Data stored as an ordered dict
    """
    with open(fpath, "w") as fobj:
        # get the first data to get the field names
        first = next(gen)
        writer = csv.DictWriter(fobj, first.keys())
        writer.writeheader()
        writer.writerow(first)
        for row in gen:
            writer.writerow(row)
