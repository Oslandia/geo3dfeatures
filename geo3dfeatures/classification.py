"""3D point clouds classification.

This module aims at classifying points that compose a 3D point cloud
regarding their spatial organization, and most specifically, the geometric
structure of their neighborhood.
"""

from pathlib import Path

import daiquiri
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from geo3dfeatures import io


logger = daiquiri.getLogger(__name__)


def standard_normalization(sample):
    """Normalize a set of points regarding mean and standard deviation

    Parameters
    ----------
    sample : numpy.array
        Set of points to normalized; must be a 2D-shaped np.array

    Returns
    -------
    np.array
        Normalized (2D-shaped) set of points
    """
    return (sample - sample.mean(axis=0)) / sample.std(axis=0)


def normalize_features(df_features):
    """Normalize geometric features in order to push data towards a machine
    learning procedure

    Parameters
    ----------
    df_features : pandas.DataFrame
        Input features

    Returns
    -------
    pandas.DataFrame
        Output normalized features
    """
    df_out = df_features.copy()
    for column in df_out.columns:
        df_out[column] = standard_normalization(df_out[column])
    return df_out


def compute_clusters(data, n_clusters, batch_size, seed=1337):
    """Predict k-mean labels on the provided dataset

    If batch_size > 0, the MiniBatchKMeans is used instead of KMeans

    Parameters
    ----------
    df_features : pandas.DataFrame
        Input geometric features on a sample of 3D points
    n_clusters : int
        Number of clusters to consider
    batch_size : int
        Batch size for MiniBatchKMeans; if 0 a simple KMeans algorithm is run
    seed : int
        Random seed for k-mean algorithm initialization

    Returns
    -------
    list
        Predicted clusters according to the k-mean clustering
    """
    if batch_size == 0:
        model = KMeans(n_clusters=n_clusters, random_state=seed)
    else:
        model = MiniBatchKMeans(
            n_clusters=n_clusters, batch_size=batch_size, random_state=seed
        )
    model.fit(data)
    return model.labels_


def colorize_labels(points, labels, glossary=None):
    """Associated a (r, g, b) color for each record of a dataframe, depending
    on the label id

    Parameters
    ----------
    points : pandas.DataFrame
        Set of (x, y, z) points
    labels : list
        Resulting label id; must be of the same length than points
    glossary : dict
        Label glossary; if None the label color is defined by a default
    'colorblind' palette

    Returns
    -------
    pandas.DataFrame
        Colorized points, i.e. set of (x, y, z, r, g, b) clustered points
    """
    if glossary is None:
        palette = sns.color_palette("colorblind", len(np.unique(labels)))
    else:
        palette = [label["color"] for _, label in glossary.items()]
    colors = np.array([palette[l] for l in labels]) * 255
    colors = pd.DataFrame(colors, columns=["r", "g", "b"], dtype=np.uint8)
    return points.join(colors)


def split_dataset(df, test_part=0.25):
    """Split the dataset in four parts, namely a training and a testing sets,
    and explicative variables and explained labels for each set.

    Based on scikit-learn API.

    Parameters
    ----------
    df : pd.DataFrame
        Data to split; must contain a "label" feature
    test_part : float
        Testing set proportion (0 < p < 1)

    Returns
    -------
    tuple of four pd.DataFrame
        Splitted dataset
    """
    if "label" not in df.columns:
        raise ValueError("No 'label' column in the provided dataset.")
    train_dataset, test_dataset = train_test_split(
        df, test_size=test_part, shuffle=True
    )
    return (
        train_dataset.drop(columns=["label"]),
        train_dataset["label"],
        test_dataset.drop(columns=["label"]),
        test_dataset["label"]
        )


def train_predictive_model(data, labels, seed=1337):
    """Normalize the data and apply a logistic regression on input data in
    order to train a model able to predict 3D point labels with regards to
    their geometric properties

    Parameters
    ----------
    data : pd.DataFrame
        Input geometric features on a sample of 3D points
    labels : pd.Series
        3D point expected output, targetted by the predictive model
    seed : int
        Random seed for algorithm initialization

    Returns
    -------
    sklearn.pipeline.Pipeline
        Classifier, as a combination of a scaler and a predictive model
    """
    if len(data) != len(labels):
        raise ValueError(
            "Data and label shapes do not correspond: there are "
            f"{len(data)} records in data, and {len(labels)} labels",
        )
    classifier = make_pipeline(
        MinMaxScaler(),
        LogisticRegression(
            solver="lbfgs", multi_class="multinomial",
            max_iter=200, random_state=seed
        )
    )
    classifier.fit(data, labels)
    return classifier


def save_labels(
        results, datapath, experiment, neighbors, radius, algorithm,
        nb_clusters=None, config_name="full", pp_neighbors=0, xyz=False
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
    algorithm : str
        Algorithm used in label prediction purpose ("kmeans", "logreg", etc)
    nb_clusters : int
        Number of cluster, used for identifying the resulting data (used only
    if algorithm is "kmeans")
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
    output_path.mkdir(exist_ok=True, parents=True)
    extension = "xyz" if xyz else "las"
    postprocess_suffix = (
        "-pp" + str(pp_neighbors) if pp_neighbors > 0 else ""
        )
    algo_str = algorithm + (
        "-" + str(nb_clusters) if algorithm == "kmeans" else ""
    )
    output_file_path = Path(
        output_path,
        algo_str + "-"
        + io.instance(neighbors, radius)
        + "-" + config_name + postprocess_suffix
        + "." + extension
    )
    if xyz:
        io.write_xyz(results, output_file_path)
    else:
        input_file_path = Path(datapath, "input", experiment + ".las")
        io.write_las(results, input_file_path, output_file_path)
    logger.info("Predicted labels saved into %s", output_file_path)
