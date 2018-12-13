"""This module aims at classifying points that compose a 3D point cloud
regarding their spatial organization, and most specifically, the geometric
structure of their neighborhood.

"""

import math
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans


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


def compute_cluster(df_features, nb_clusters=2, seed=1337):
    """Compute a KMeans clustering on `df_features`

    Parameters
    ----------
    df_features : pandas.DataFrame
        Input geometric features on a sample of 3D points
    nb_clusters : int
        Number of clusters to consider

    Returns
    -------
    sklearn.cluster.KMeans
        Clustering model applied on input features
    """
    model = KMeans(nb_clusters, random_state=seed)
    model.fit(df_features)
    return model
