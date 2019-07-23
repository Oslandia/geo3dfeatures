"""Compute k-means clustering on 3D point cloud with geometric features
"""

import argparse
from configparser import ConfigParser
import os
from pathlib import Path
import sys

import daiquiri
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import laspy

from geo3dfeatures.features import max_normalize

logger = daiquiri.getLogger(__name__)

SEED = 1337


def instance(neighbors, radius, feature_set, bin_size):
    """Build the instance name, depending on the input parameters

    Parameters
    ----------
    neighbors : int
        Number of neighbors used to compute the feature set
    radius : float
        Threshold that define the neighborhood, in order to compute the feature
    set; used if neighbors is None
    feature_set : str
        Set of features, i.e. alphabeta, eigenvalues or full
    bin_size : float
        Bin size used to compute accumulation features

    Returns
    -------
    str
        Name of the instance
    """
    if neighbors is not None:
        neighborhood = "n" + str(neighbors)
    elif radius is not None:
        neighborhood = "r" + str(radius)
    else:
        raise ValueError(
            "Error in input neighborhood definition: "
            "neighbors and radius arguments can't be both undefined"
            )
    return (
        neighborhood + "-" + feature_set + "-binsize-" + str(bin_size)
    )


def load_features(
        datapath, experiment, neighbors, radius, feature_set, bin_size
):
    """Read feature set from the file system, starting from the input
        parameters

    Parameters
    ----------
    datapath : str
        Root of the data folder
    experiment : str
        Name of the experiment, used for identifying the accurate subfolder
    neighbors : int
        Number of neighbors used to compute the feature set
    radius : float
        Threshold that define the neighborhood, in order to compute the feature
    set; used if neighbors is None
    feature_set : str
        Set of features, i.e. alphabeta, eigenvalues or full
    bin_size : float
        Bin size used to compute accumulation features

    Returns
    -------
    pandas.DataFrame
        Feature set, each record refering to a point; columns correspond to
    geometric features
    """
    filepath = Path(
        datapath, "output", experiment, "features",
        "features-" + instance(neighbors, radius, feature_set, bin_size)
        + ".csv"
        )
    logger.info(f"Recover features stored in {filepath}")
    return pd.read_csv(filepath)


def update_features(df, config_path):
    """Modify data features with the help of multiplier coefficients associated
    to each data feature, before to fit the k-mean model

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    config_path : str
        Path of the file where the coefficients are stored

    Returns
    -------
    np.array
        Array of feature coefficient; sorted in the same order than feature_list
    """
    coefs = np.ones(shape=df.shape[1])
    if os.path.isfile(config_path):
        config = ConfigParser()
        config.optionxform = str  # Preserve case in feature names
        config.read(config_path)
        df_coefs = pd.DataFrame(np.expand_dims(coefs, 0), columns=df.columns)
        for key in config["clustering"]:
            df_coefs[key] = float(config["clustering"][key])
        coefs = np.squeeze(np.array(df_coefs))
    for idx, column in enumerate(df.columns):
        if coefs[idx] != 1:
            logger.info(f"Multiply {column} feature by {coefs[idx]}")
            df[column] = coefs[idx] * df[column]


def colorize_clusters(points, clusters):
    """Associated a (r, g, b) color for each record of a dataframe, depending
    on the cluster id

    Parameters
    ----------
    points : pandas.DataFrame
        Set of (x, y, z) points
    clusters : list
        Resulting cluster id; must be of the same length than points

    Returns
    -------
    pandas.DataFrame
        Colorized points, i.e. set of (x, y, z, r, g, b) clustered points
    """
    palette = sns.color_palette("colorblind", len(np.unique(clusters)))
    colors = np.array([palette[l] for l in clusters]) * 256
    colors = pd.DataFrame(colors, columns=["r", "g", "b"], dtype=np.uint8)
    return points.join(colors)


def save_clusters(
        results, datapath, experiment, neighbors, radius,
        feature_set, bin_size, nb_clusters, config_name, xyz=False
):
    """Save the resulting dataframe into the accurate folder on the file system

    Parameters
    ----------
    results : pandas.DataFrame
        Data to save
    datapath : str
        Root of the data folder
    experiment : str
        Name of the experiment, used for identifying the accurate subfolder
    neighbors : int
        Number of neighbors used to compute the feature set
    radius : float
        Threshold that define the neighborhood, in order to compute the feature
    set; used if neighbors is None
    feature_set : str
        Set of features, i.e. alphabeta, eigenvalues or full
    bin_size : float
        Bin size used to compute accumulation features
    nb_clusters : int
        Number of cluster, used for identifying the resulting data
    config_name : str
        Cluster configuration filename
    xyz : boolean
        If true, the output file is a .xyz, otherwise a .las file will be
    produced
    """
    output_path = Path(
        datapath, "output", experiment, "clustering",
    )
    os.makedirs(output_path, exist_ok=True)
    extension = "xyz" if xyz else "las"
    output_file = Path(
        output_path,
        "kmeans-"
        + instance(neighbors, radius, feature_set, bin_size)
        + "-" + config_name + "-" + str(nb_clusters) + "." + extension
    )
    if xyz:
        results.to_csv(
            str(output_file), sep=" ", index=False, header=False
        )
    else:
        input_file_path = Path(datapath, "input", experiment + ".las")
        with laspy.file.File(input_file_path, mode="r") as input_las:
            outfile = laspy.file.File(
                output_file, mode="w", header=input_las.header
            )
            outfile.x = results.x
            outfile.y = results.y
            outfile.z = results.z
            outfile.red = results.r
            outfile.green = results.g
            outfile.blue = results.b
            outfile.close()
    logger.info(f"Clusters saved into {output_file}")


def main(opts):
    config_path = Path("config", opts.config_file)
    data = load_features(
        opts.datapath, opts.experiment, opts.neighbors, opts.radius,
        opts.feature_set, opts.bin_size
    )

    for c in data.columns[3:]:
        data[c] = max_normalize(data[c])

    if "bin_z_range" in data.columns:
        data["bin_z_range"].fillna(0, inplace=True)

    points = data[["x", "y", "z"]].copy()
    data.drop(["x", "y", "z"], axis=1, inplace=True)

    update_features(data, config_path)

    logger.info(f"Compute {opts.nb_clusters} clusters...")
    model = KMeans(opts.nb_clusters, random_state=SEED)
    model.fit(data)

    colored_results = colorize_clusters(points, model.labels_)
    save_clusters(
        colored_results, opts.datapath, opts.experiment, opts.neighbors,
        opts.radius, opts.feature_set, opts.bin_size, opts.nb_clusters,
        config_path.stem, opts.xyz
    )
