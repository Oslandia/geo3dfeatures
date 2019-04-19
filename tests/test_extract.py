import pytest

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree as KDTree

from geo3dfeatures.extract import (
    compute_tree, sequence_light, sequence_full,
    process_alphabeta, process_eigenvalues, process_full, extract
    )
from geo3dfeatures.features import accumulation_2d_neighborhood

def test_extract(sphere):
    """Test the feature set extraction
    """
    kd_tree = KDTree(sphere, leafsize=100)
    csvpath = "tests/data/test_extract.csv"
    # with pytest.raises(ValueError):
    #     extra_columns = ("dummy-column")
    #     extract(
    #         sphere, kd_tree, nb_neighbors=10, csvpath=csvpath,
    #         extra_columns=extra_columns, feature_set="full", nb_processes=2
    #     )
    extract(
        sphere, kd_tree, nb_neighbors=10, csvpath=csvpath,
        feature_set="full", nb_processes=2
    )
    features = pd.read_csv(csvpath)
    assert features.shape[0] == sphere.shape[0]


def test_extract_extra_features(sphere):
    """Test the feature set extraction when there is more than only "x", "y"
    and "z" features in the input data
    """
    kd_tree = KDTree(sphere, leafsize=100)
    colors = np.random.randint(0, 255, sphere.shape)
    data = np.concatenate((sphere, colors), axis=1)
    csvpath = "tests/data/test_extract.csv"
    extract(
        data, kd_tree, nb_neighbors=10, csvpath=csvpath,
        extra_columns=("r", "g", "b"), feature_set="alphabeta", nb_processes=2
    )
    nb_extra_features = 2
    nb_output_features = sphere.shape[1] + colors.shape[1] + nb_extra_features
    features = pd.read_csv(csvpath)
    assert features.shape[0] == sphere.shape[0]
    assert features.shape[1] == nb_output_features


def test_sequence_light(sphere):
    """Test the sequence building in the case of "alphabeta" and "eigenvalues"
    feature set
    """
    NB_NEIGHBORS = 10
    tree = compute_tree(sphere, leaf_size=500)
    gen = sequence_light(sphere, tree, nb_neighbors=NB_NEIGHBORS)
    first_item = next(gen)
    assert len(first_item) == 2
    assert first_item[0].shape == (NB_NEIGHBORS + 1, 3)
    assert len(first_item[1]) == 2
    assert first_item[1].names == first_item[1].values == ()
    assert len(list(gen)) == sphere.shape[0] - 1


def test_sequence_full(sphere):
    """Test the sequence building in the case of "full" feature set
    """
    NB_NEIGHBORS = 10
    tree = compute_tree(sphere, leaf_size=500)
    acc_features = accumulation_2d_neighborhood(sphere)
    gen = sequence_full(acc_features, tree, nb_neighbors=NB_NEIGHBORS)
    first_item = next(gen)
    assert len(first_item) == 4
    assert first_item[0].shape == (NB_NEIGHBORS + 1, 3)
    assert first_item[1].shape == (NB_NEIGHBORS + 1,)
    assert first_item[2].shape == (3,)
    assert len(first_item[3]) == 2
    assert first_item[3].names == first_item[3].values == ()
    assert len(list(gen)) == sphere.shape[0] - 1


def test_process_alphabeta(sphere):
    """Test the alphabeta feature set processing for the first "sphere" point
    """
    additional_features = ["alpha", "beta"]
    tree = compute_tree(sphere, leaf_size=500)
    gen = sequence_light(sphere, tree, nb_neighbors=10)
    features, _ = process_alphabeta(next(gen)[0], extra=None)
    assert len(features) == 3 + len(additional_features)
    assert features.x == sphere[0, 0]
    assert features.y == sphere[0, 1]
    assert features.z == sphere[0, 2]
    dfeatures = features._asdict()
    assert list(dfeatures.keys()) == ["x", "y", "z"] + additional_features


def test_process_eigenvalues(sphere):
    """Test the eigenvalues feature set processing for the first "sphere" point
    """
    additional_features = [
        "alpha", "beta",
        "curvature_change", "linearity", "planarity",
        "scattering", "omnivariance", "anisotropy",
        "eigenentropy", "eigenvalue_sum"
    ]
    tree = compute_tree(sphere, leaf_size=500)
    gen = sequence_light(sphere, tree, nb_neighbors=10)
    features, _ = process_eigenvalues(next(gen)[0], extra=None)
    assert len(features) == 3 + len(additional_features)
    assert features.x == sphere[0, 0]
    assert features.y == sphere[0, 1]
    assert features.z == sphere[0, 2]
    dfeatures = features._asdict()
    assert list(dfeatures.keys()) == ["x", "y", "z"] + additional_features


def test_process_full(sphere):
    """Test the full feature set processing for the first "sphere" point
    """
    additional_features = [
        "alpha", "beta", "radius",
        "z_range", "std_dev", "density", "verticality",
        "curvature_change", "linearity", "planarity",
        "scattering", "omnivariance", "anisotropy",
        "eigenentropy", "eigenvalue_sum",
        "radius_2D", "density_2D", "eigenvalue_sum_2D", "eigenvalue_ratio_2D",
        "bin_density", "bin_z_range", "bin_z_std"
    ]
    tree = compute_tree(sphere, leaf_size=500)
    acc_features = accumulation_2d_neighborhood(sphere)
    gen = sequence_full(acc_features, tree, nb_neighbors=10)
    item = next(gen)
    features, _ = process_full(
        item[0], item[1], item[2], extra=None
    )
    assert len(features) == 3 + len(additional_features)
    assert features.x == sphere[0, 0]
    assert features.y == sphere[0, 1]
    assert features.z == sphere[0, 2]
    dfeatures = features._asdict()
    assert list(dfeatures.keys()) == ["x", "y", "z"] + additional_features
