"""3D point clouds classification.

This module aims at classifying points that compose a 3D point cloud
regarding their spatial organization, and most specifically, the geometric
structure of their neighborhood.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler


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


def colorize_labels(points, labels, palette=None):
    """Associated a (r, g, b) color for each record of a dataframe, depending
    on the label id

    Parameters
    ----------
    points : pandas.DataFrame
        Set of (x, y, z) points
    labels : list
        Resulting label id; must be of the same length than points
    palette : seaborn.palettes._ColorPalette
        Colorization palette; if None a default 'colorblind' palette is applied

    Returns
    -------
    pandas.DataFrame
        Colorized points, i.e. set of (x, y, z, r, g, b) clustered points
    """
    if palette is None:
        palette = sns.color_palette("colorblind", len(np.unique(labels)))
    colors = np.array([palette[l] for l in labels]) * 256
    colors = pd.DataFrame(colors, columns=["r", "g", "b"], dtype=np.uint8)
    return points.join(colors)


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
    classifier = make_pipeline(
        MinMaxScaler(),
        LogisticRegression(
            solver="lbfgs", multi_class="multinomial",
            max_iter=200, random_state=seed
        )
    )
    classifier.fit(data, labels)
    return classifier
