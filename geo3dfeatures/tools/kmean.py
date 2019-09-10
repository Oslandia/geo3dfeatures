"""Compute k-means clustering on 3D point cloud with geometric features
"""

from configparser import ConfigParser
import os
from pathlib import Path
import sys

import daiquiri
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import MiniBatchKMeans
import laspy

from geo3dfeatures.features import accumulation_2d_neighborhood, max_normalize
from geo3dfeatures.extract import compute_tree

from geo3dfeatures import postprocess

logger = daiquiri.getLogger(__name__)

SEED = 1337
KMEAN_BATCH = 10_000
KEY_H5_FORMAT = "/num_{:04d}"


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


def load_features(datapath, experiment, neighbors):
    """Read feature set from the file system, starting from the input
        parameters

    Parameters
    ----------
    datapath : str
        Root of the data folder
    experiment : str
        Name of the experiment, used for identifying the accurate subfolder
    neighbors : list
        List of number of neigbors

    Returns
    -------
    pandas.DataFrame
        Feature set, each record refering to a point; columns correspond to geometric features
    """
    filepath = Path(
        datapath, "output", experiment, "features", "features.h5"
    )
    logger.info(f"Recover features stored in {filepath}")
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
    if os.path.isfile(config_path):
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


def update_features(df, config):
    """Modify data features with the help of multiplier coefficients associated
    to each data feature, before to fit the k-mean model

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    config : configparser.ConfigParser
        Feature coefficient configuration

    Returns
    -------
    np.array
        Array of feature coefficient; sorted in the same order than feature_list
    """
    coefs = np.ones(shape=df.shape[1])
    df_coefs = pd.DataFrame(np.expand_dims(coefs, 0), columns=df.columns)
    for key in config["clustering"]:
        if key not in df.columns:
            logger.warning("%s is not a known feature, skipping.", key)
            continue
        df_coefs[key] = float(config["clustering"][key])
    coefs = np.squeeze(np.array(df_coefs))
    for idx, column in enumerate(df.columns):
        if coefs[idx] != 1:
            logger.info("Multiply %s feature by %s", column, coefs[idx])
            df[column] = coefs[idx] * df[column]
    if "bin" in config["clustering"]:
        bin_size = float(config["clustering"]["bin"])
        logger.info("Found a 'bin' in the config file. Computation of the accumulation features")
        logger.info("Bin size %s", bin_size)
        df = accumulation_2d_neighborhood(df, bin_size)
    return df


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
        nb_clusters, config_name, postprocess=False, xyz=False
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
    nb_clusters : int
        Number of cluster, used for identifying the resulting data
    config_name : str
        Cluster configuration filename
    postprocess : boolean
        If true, the output clusters are postprocessed, otherwise they are
    k-mean algorithm outputs
    xyz : boolean
        If true, the output file is a .xyz, otherwise a .las file will be
    produced
    """
    output_path = Path(
        datapath, "output", experiment, "clustering",
    )
    os.makedirs(output_path, exist_ok=True)
    extension = "xyz" if xyz else "las"
    postprocess_suffix = "-pp" if postprocess else ""
    output_file = Path(
        output_path,
        "kmeans-"
        + instance(neighbors, radius)
        + "-" + config_name + "-" + str(nb_clusters) + postprocess_suffix
        + "." + extension
    )
    if xyz:
        results.to_csv(
            str(output_file), sep=" ", index=False, header=True
        )
    else:
        input_file_path = Path(datapath, "input", experiment + ".las")
        if not input_file_path.is_file():
            logger.error(f"{input_file_path} is not a valid file.")
            sys.exit(1)
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
    feature_config = read_config(config_path)

    experiment = opts.input_file.split(".")[0]
    data = load_features(opts.datapath, experiment, opts.neighbors)
    points = data[["x", "y", "z"]].copy()

    for c in data.drop(columns=["x", "y"]):
        data[c] = max_normalize(data[c])

    data = update_features(data, feature_config)

    data.drop(columns=["x", "y"], inplace=True)
    if "bin_density" in data.columns:  # There are accumulation features
        for c in ("bin_z_range", "bin_z_std", "bin_density"):
            data[c] = max_normalize(data[c])
            data[c].fillna(0, inplace=True)

    logger.info("Compute %s clusters...", opts.nb_clusters)
    model = MiniBatchKMeans(
        n_clusters=opts.nb_clusters,
        batch_size=KMEAN_BATCH,
        random_state=SEED
    )
    model.fit(data)
    labels = model.labels_

    # Postprocessing
    if opts.post_processing:
        logger.info(f"Post-process point labels by batches of {KMEAN_BATCH}")
        tree = compute_tree(points, opts.kdtree_leafs)
        gen = postprocess.batch_points(points, KMEAN_BATCH)
        pp_neighbors = (
            opts.postprocessing_neighbors
            if opts.postprocessing_neighbors is not None
            else max(opts.neighbors)
            )
        labels = postprocess.postprocess_batch_labels(
            gen, labels, tree, pp_neighbors
        )

    colored_results = colorize_clusters(points, labels)
    save_clusters(
        colored_results, opts.datapath, experiment, opts.neighbors,
        opts.radius, opts.nb_clusters,
        config_path.stem, opts.post_processing, opts.xyz
    )
