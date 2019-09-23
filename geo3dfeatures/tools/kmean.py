"""Compute k-means clustering on 3D point cloud with geometric features
"""

from pathlib import Path

import daiquiri
import numpy as np
import pandas as pd

from geo3dfeatures.classification import colorize_labels, compute_clusters
from geo3dfeatures.features import accumulation_2d_neighborhood, max_normalize
from geo3dfeatures.extract import compute_tree
from geo3dfeatures import io
from geo3dfeatures import postprocess


logger = daiquiri.getLogger(__name__)

SEED = 1337
KMEAN_BATCH = 10_000
POSTPROCESSING_BATCH = 10_000
KEY_H5_FORMAT = "/num_{:04d}"


def add_accumulation_features(df, config):
    """Add the accumulation features to the set of existing features before to
    compute k-means

    The bin definition is provided within the specified configuration object.

    Parameters
    ----------
    df : pd.DataFrame
        Set of features
    config : configparser.ConfigParser
        Configuration parameters, as provided by the chosen configuration file;
    contains a "bin" section within its "clustering" main section

    Returns
    -------
    pd.DataFrame
        Updated set of features with bin-related information
    """
    bin_size = float(config["clustering"]["bin"])
    logger.info(
        "Computation of the accumulation features with bin_size=%s", bin_size
    )
    df = accumulation_2d_neighborhood(df, bin_size)
    for c in ("bin_z_range", "bin_z_std", "bin_density"):
        df[c] = max_normalize(df[c])
        df[c].fillna(0, inplace=True)
    return df


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
        if key in ("r", "g", "b", "z"):
            df_coefs[key] = float(config["clustering"][key])
        else:
            key_columns = df.columns.str.startswith(key)
            coefs[key_columns] = float(config["clustering"][key])
    coefs = np.squeeze(np.array(df_coefs))
    for idx, column in enumerate(df.columns):
        if coefs[idx] != 1:
            logger.info("Multiply %s feature by %s", column, coefs[idx])
            df[column] = coefs[idx] * df[column]


def save_clusters(
        results, datapath, experiment, neighbors, radius,
        nb_clusters, config_name, pp_neighbors, xyz=False
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
    pp_neighbors : int
        If >0, the output clusters are postprocessed, otherwise they are k-mean
    algorithm outputs
    xyz : boolean
        If true, the output file is a .xyz, otherwise a .las file will be
    produced
    """
    output_path = Path(
        datapath, "output", experiment, "prediction",
    )
    output_path.mkdir(exist_ok=True)
    extension = "xyz" if xyz else "las"
    postprocess_suffix = (
        "-pp" + str(pp_neighbors) if pp_neighbors > 0 else ""
        )
    output_file_path = Path(
        output_path,
        "kmeans-"
        + io.instance(neighbors, radius)
        + "-" + config_name + "-" + str(nb_clusters) + postprocess_suffix
        + "." + extension
    )
    if xyz:
        io.write_xyz(results, output_file_path)
    else:
        input_file_path = Path(datapath, "input", experiment + ".las")
        io.write_las(results, input_file_path, output_file_path)
    logger.info("Clusters saved into %s", output_file_path)


def main(opts):
    config_path = Path("config", opts.config_file)
    feature_config = io.read_config(config_path)

    experiment = opts.input_file.split(".")[0]
    data = io.load_features(opts.datapath, experiment, opts.neighbors)
    points = data[["x", "y", "z"]].copy()

    for c in data.drop(columns=["x", "y"]):
        data[c] = max_normalize(data[c])

    if "bin" in feature_config["clustering"]:
        data = add_accumulation_features(data, feature_config)

    update_features(data, feature_config)
    data.drop(columns=["x", "y"], inplace=True)

    logger.info("Compute %s clusters...", opts.nb_clusters)
    labels = compute_clusters(
        data, n_clusters=opts.nb_clusters, batch_size=KMEAN_BATCH, seed=SEED
    )

    # Postprocessing
    if opts.postprocessing_neighbors > 0:
        logger.info(f"Post-process point labels by batches of {KMEAN_BATCH}")
        tree = compute_tree(points, opts.kdtree_leafs)
        gen = postprocess.batch_points(points, POSTPROCESSING_BATCH)
        labels = postprocess.postprocess_batch_labels(
            gen, POSTPROCESSING_BATCH, labels, tree, opts.postprocessing_neighbors
        )

    colored_results = colorize_labels(points, labels)
    save_clusters(
        colored_results, opts.datapath, experiment, opts.neighbors,
        opts.radius, opts.nb_clusters,
        config_path.stem, opts.postprocessing_neighbors, opts.xyz
    )
