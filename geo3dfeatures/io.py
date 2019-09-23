"""I/O module to load/write some point clouds files
"""

from configparser import ConfigParser
import csv
from pathlib import Path
import sys

import daiquiri
import laspy
import numpy as np
from plyfile import PlyData
import pandas as pd


KEY_H5_FORMAT = "/num_{:04d}"

logger = daiquiri.getLogger(__name__)


def read_xyz(fpath, names=None, header=True):
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


def write_xyz(data, filepath):
    """Write a .xyz file, which is a simple csv-like format

    Parameters
    ----------
    data : pd.DataFrame
        Data to save
    filepath : pathlib.Path
        Path where to save the data
    """
    data.to_csv(filepath, sep=" ", index=False, header=True)


def read_las(fpath):
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


def write_las(data, input_filepath, output_filepath):
    """Write a .las file with the "laspy" package, by reusing the input file
    features

    Parameters
    ----------
    data : pd.DataFrame
        Data to save
    input_filepath : pathlib.Path
        Path of the input file, from which header metadata is extracted
    output_filepath : pathlib.Path
        Path where to save the data
    """
    if not input_filepath.is_file():
        logger.error("%s is not a valid file.", input_filepath)
        sys.exit(1)
    with laspy.file.File(input_filepath, mode="r") as input_las:
        with laspy.file.File(
                output_filepath, mode="w", header=input_las.header
        ) as output_las:
            output_las.x = data.x
            output_las.y = data.y
            output_las.z = data.z
            output_las.red = data.r
            output_las.green = data.g
            output_las.blue = data.b


def read_ply(fpath):
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


def load_features(datapath, experiment, neighbors, sample=None):
    """Read the featurized data stored in a h5 file, and aggregate all the
    neighborhood sizes by suffixing the feature column names

    The dataframe rows are stored with respect to XYZ columns to assign the
    feature values to the relevant points, as featurization does not
    guarantee it.

    Parameters
    ----------
    datapath : str
        Root of the data folder
    experiment : str
        Name of the experiment, used for identifying the accurate subfolder
    neighbors : list
        List of number of neighbors
    sample : str
        If None, consider the whole scene, otherwise load the features only for
    the relevant class sample

    Returns
    -------
    pandas.DataFrame
        Feature set, each record refering to a point; columns correspond to
    geometric features
    """
    sample_suffix = "" if sample is None else "_" + sample
    filename = "features" + sample_suffix + ".h5"
    filepath = Path(datapath, "output", experiment, "features", filename)
    if not filepath.is_file():
        logger.warning(
            "File %s does not exist, verify the 'sample' argument!", filepath
        )
        return None
    logger.info("Recover features stored in %s", filepath)
    no_rename = ["x", "y", "z", "r", "g", "b"]
    with pd.HDFStore(filepath, mode="r") as store:
        # loop on the possible number of neighbors and concatenate features
        # we have to sort each dataframe in order to align each point x,y,z
        num_neighbor = neighbors[0]
        key = KEY_H5_FORMAT.format(num_neighbor)
        df = store[key]
        df.sort_values(by=list("xyz"), inplace=True)
        df.drop(columns=["num_neighbors"], inplace=True)
        cols = [x for x in df if x not in no_rename]
        df.rename(columns={key: key + "_" + str(num_neighbor) for key in cols}, inplace=True)
        df.index = pd.Index(range(df.shape[0]))
        dataframes = [df]
        for num_neighbor in neighbors[1:]:
            key = KEY_H5_FORMAT.format(num_neighbor)
            newdf = store[key]
            newdf.drop(
                columns=["num_neighbors", "r", "g", "b"],
                errors="ignore", inplace=True
            )
            newdf.sort_values(by=list("xyz"), inplace=True)
            newdf.drop(columns=["x", "y", "z"], inplace=True)
            newdf.rename(columns={key: key + "_" + str(num_neighbor) for key in cols}, inplace=True)
            newdf.index = pd.Index(range(newdf.shape[0]))
            dataframes.append(newdf)

    return pd.concat(dataframes, axis="columns")


def read_config(config_path):
    """Create a config object starting from a configuration file in the
    "config" folder

    Parameters
    ----------
    config_path : str
        Path of the configuration file on the file system; should end with
    ".ini" extension

    Returns
    -------
    configparser.ConfigParser
        Feature coefficient configuration for the clustering process
    """
    feature_config = ConfigParser()
    feature_config.optionxform = str  # Preserve case in feature names
    if config_path.is_file():
        feature_config.read(config_path)
    else:
        logger.error(f"{config_path} is not a valid file.")
        sys.exit(1)
    if not feature_config.has_section("clustering"):
        logger.error(
            f"{config_path} is not a valid configuration file "
            "(no 'clustering' section)."
        )
        sys.exit(1)
    return feature_config


def instance(neighbors, radius):
    """Build the instance name, depending on the input parameters

    Parameters
    ----------
    neighbors : int
        Number of neighbors used to compute the feature set
    radius : float
        Threshold that define the neighborhood, in order to compute the feature
        set; used if neighbors is None

    Returns
    -------
    str
        Name of the instance
    """
    if neighbors is not None:
        return "-".join(str(x) for x in neighbors)
    elif radius is not None:
        neighborhood = "r" + str(radius)
    else:
        raise ValueError(
            "Error in input neighborhood definition: "
            "neighbors and radius arguments can't be both undefined"
            )
    return neighborhood
