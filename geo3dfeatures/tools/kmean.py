"""Compute k-means clustering on 3D point cloud with geometric features
"""

from pathlib import Path

import daiquiri
import numpy as np
import pandas as pd

from geo3dfeatures.classification import (
    colorize_labels, compute_clusters, save_labels
    )
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
    bin_size = float(config.get("clustering", "bin"))
    logger.info(
        "Computation of the accumulation features with bin_size=%s", bin_size
    )
    df = accumulation_2d_neighborhood(df, bin_size)
    for c in ("bin_z_range", "bin_z_std", "bin_density"):
        df[c] = max_normalize(df[c])
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


def main(opts):
    config_path = Path("config", opts.config_file)
    feature_config = io.read_config(config_path)

    experiment = opts.input_file.split(".")[0]
    data = io.load_features(opts.datapath, experiment, opts.neighbors)
    points = data[["x", "y", "z"]].copy()

    for c in data.drop(columns=["x", "y"]):
        data[c] = max_normalize(data[c])

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
    save_labels(
        colored_results, opts.datapath, experiment, opts.neighbors,
        opts.radius, "kmeans", opts.nb_clusters,
        config_path.stem, opts.postprocessing_neighbors, opts.xyz
    )
