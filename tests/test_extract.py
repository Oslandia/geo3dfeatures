import pytest

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree as KDTree

from geo3dfeatures.extract import (
    compute_tree, request_tree, sequence_full, process_full, extract,
    ExtraFeatures
    )

_here = Path(__file__).resolve().parent
DATADIR = _here / "data"


def test_extract(sphere):
    """Test the feature set extraction
    """
    kd_tree = KDTree(sphere, leafsize=100)
    h5path = DATADIR / "test_extract.h5"
    extract(
        sphere, kd_tree, nb_neighbors=[10], h5path=h5path, nb_processes=2
    )
    features = pd.read_hdf(h5path, "/num_0010")
    h5path.unlink()
    assert features.shape[0] == sphere.shape[0]


def test_extract_extra_features(sphere):
    """Test the feature set extraction when there is more than only "x", "y"
    and "z" features in the input data
    """
    kd_tree = KDTree(sphere, leafsize=100)
    colors = np.random.randint(0, 255, sphere.shape)
    data = np.concatenate((sphere, colors), axis=1)
    h5path = DATADIR / "test_extract.h5"
    extract(
        data, kd_tree, nb_neighbors=[10], h5path=h5path,
        extra_columns=("r", "g", "b"), nb_processes=2
    )
    nb_extra_features = 20
    nb_output_features = sphere.shape[1] + colors.shape[1] + nb_extra_features
    features = pd.read_hdf(h5path, "/num_0010")
    h5path.unlink()
    assert features.shape[0] == sphere.shape[0]
    assert features.shape[1] == nb_output_features


def test_sequence_full(sphere):
    """Test the sequence building in the case of "full" feature set
    """
    NB_NEIGHBORS = 10
    tree = compute_tree(sphere, leaf_size=500)
    gen = sequence_full(sphere, tree, nb_neighbors=[NB_NEIGHBORS])
    first_item = next(gen)
    assert len(first_item) == 3
    assert first_item[0].shape == (NB_NEIGHBORS + 1, 3)
    assert first_item[1].shape == (NB_NEIGHBORS + 1,)
    assert isinstance(first_item[2], ExtraFeatures)
    assert len(list(gen)) == sphere.shape[0] - 1


def test_process_full(sphere):
    """Test the full feature set processing for the first "sphere" point
    """
    additional_features = [
        "num_neighbors", "alpha", "beta", "radius",
        "z_range", "std_dev", "density", "verticality",
        "curvature_change", "linearity", "planarity",
        "scattering", "omnivariance", "anisotropy",
        "eigenentropy", "eigenvalue_sum",
        "radius_2D", "density_2D", "eigenvalue_sum_2D", "eigenvalue_ratio_2D",
    ]
    tree = compute_tree(sphere, leaf_size=500)
    gen = sequence_full(sphere, tree, nb_neighbors=[10])
    item = next(gen)
    features, _ = process_full(
        item[0], item[1], extra=None
    )
    assert len(features) == 3 + len(additional_features)
    assert features.x == sphere[0, 0]
    assert features.y == sphere[0, 1]
    assert features.z == sphere[0, 2]
    dfeatures = features._asdict()
    assert list(dfeatures.keys()) == ["x", "y", "z"] + additional_features


def test_request_tree(sphere):
    """Test a kd-tree request, depending on the neighborhood definition, either
    starting from a number of neighbors or from a ball radius
    """
    tree = compute_tree(sphere, leaf_size=500)
    with pytest.raises(ValueError):
        request_tree(sphere[0], tree)
    result_radius = request_tree(sphere[0], tree, radius=1)
    assert len(result_radius) == 2
    assert result_radius[0] is None
    result_neighbors = request_tree(sphere[0], tree, nb_neighbors=10)
    assert len(result_neighbors) == 2
