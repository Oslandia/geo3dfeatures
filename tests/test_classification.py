
from pathlib import Path
import pytest
from shutil import rmtree

import numpy as np
import pandas as pd

from geo3dfeatures.classification import (
    standard_normalization, normalize_features, compute_clusters,
    colorize_labels, train_predictive_model, save_labels
    )
from geo3dfeatures.io import load_features


_here = Path(__file__).absolute().parent
DATADIR = _here / "data"
NEIGHBORHOOD_SIZE = [10, 50, 200]
EXPERIMENT = "b9"


def test_standard_normalization():
    """Test the standard scaler function.
    """
    a = np.random.randint(0, 50, 100)
    na = standard_normalization(a)
    assert a.shape == na.shape
    assert na.mean() < 1e-3
    assert na.std() - 1 < 1e-3


def test_normalize_features():
    """Test the feature normalization process; it normalize each feature of the
    input dataframe.
    """
    features = load_features(DATADIR, EXPERIMENT, NEIGHBORHOOD_SIZE)
    norm_features = normalize_features(features)
    assert norm_features.shape == features.shape
    for feature in norm_features:
        assert norm_features[feature].mean() < 1e-3
        assert norm_features[feature].std() - 1 < 1e-3


def test_compute_clusters():
    """Test the k-mean clustering procedure: it must give as many labels as one
    has individuals in the dataset; plus, the labels must be between 0 and
    N_CLUSTERS.
    """
    features = load_features(DATADIR, EXPERIMENT, NEIGHBORHOOD_SIZE)
    N_CLUSTERS = 4
    labels = compute_clusters(features, n_clusters=N_CLUSTERS, batch_size=0)
    assert labels.shape == (features.shape[0],)
    assert set(np.unique(labels)) == set(range(N_CLUSTERS))
    b_labels = compute_clusters(features, n_clusters=N_CLUSTERS, batch_size=50)
    assert b_labels.shape == (features.shape[0],)
    assert set(np.unique(b_labels)) == set(range(N_CLUSTERS))


def test_colorize_labels():
    """Test the label colorization procedure: it must return a pandas dataframe
    with XYZ and RGB features, and the number of computed RGB triplets must
    correspond to the cluster quantity.

    The user may choose its own color palette. In such a case, one must
    retrieve corresponding RGB triplets at the end of the process.
    """
    features = load_features(DATADIR, EXPERIMENT, NEIGHBORHOOD_SIZE)
    N_CLUSTERS = 4
    labels = np.random.randint(0, N_CLUSTERS, features.shape[0])
    df_color = colorize_labels(features[["x", "y", "z"]], labels)
    assert set(df_color.columns) == set("xyzrgb")
    assert len(df_color[["r", "g", "b"]].drop_duplicates()) == N_CLUSTERS
    COLORS = [(0, 0, 255), (51, 102, 153), (0, 255, 51), (255, 102, 204)]
    GLOSSARY = {
        "foo": {"id": 0, "color": (0.0, 0.0, 1.0)},
        "bar": {"id": 1, "color": (0.2, 0.4, 0.6)},
        "dummy": {"id": 2, "color": (0.0, 1.0, 0.2)},
        "doe": {"id": 3, "color": (1.0, 0.4, 0.8)}
    }
    df_color = colorize_labels(features[["x", "y", "z"]], labels, GLOSSARY)
    assert set(df_color.columns) == set("xyzrgb")
    assert len(df_color[["r", "g", "b"]].drop_duplicates()) == N_CLUSTERS
    unique_output_colors = df_color[["r", "g", "b"]].drop_duplicates().values
    assert np.all([c in unique_output_colors for c in COLORS])


def test_train_predictive_model():
    """Test the predictive model creation. One only verify the data and label
    shape relevance.
    """
    data = np.random.rand(10, 3)
    wrong_labels = np.random.randint(3, size=8)
    with pytest.raises(ValueError):
        train_predictive_model(data, wrong_labels)
    labels = np.random.randint(3, size=10)
    train_predictive_model(data, labels)


def test_save_labels():
    """Test the predicted label serialization:
    - .las file output
    - .xyz file output
    - output with post-processing
    """
    DATADIR = Path(_here, "data")
    EXPERIMENT = "test"
    NEIGHBORS = [10, 50, 200]
    results = pd.DataFrame(columns=list("xyzrgb"))
    PRED_DIR = DATADIR / "output" / EXPERIMENT / "prediction"
    save_labels(
        results, DATADIR, EXPERIMENT, NEIGHBORS, None,
        algorithm="logreg", config_name="full", pp_neighbors=0, xyz=False
    )
    pred_path = PRED_DIR / "logreg-10-50-200-full.las"
    assert pred_path.is_file()
    save_labels(
        results, DATADIR, EXPERIMENT, NEIGHBORS, None,
        algorithm="kmeans", nb_clusters=4,
        config_name="full", pp_neighbors=0, xyz=True
    )
    pred_path = PRED_DIR / "kmeans-4-10-50-200-full.xyz"
    assert pred_path.is_file()
    save_labels(
        results, DATADIR, EXPERIMENT, NEIGHBORS, None,
        algorithm="logreg", config_name="full", pp_neighbors=100, xyz=False
    )
    pred_path = PRED_DIR / "logreg-10-50-200-full-pp100.las"
    assert pred_path.is_file()
    rmtree(PRED_DIR.parent)
