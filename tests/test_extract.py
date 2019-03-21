import pytest

from geo3dfeatures.extract import (
    compute_tree, sequence_light, sequence_full,
    process_alphabeta, process_eigenvalues, process_full, extract
    )


def test_extract(sphere):
    """Test the feature set extraction
    """
    with pytest.raises(ValueError):
        input_columns = ["x", "y"]
        features = extract(
            sphere, nb_neighbors=10, input_columns=input_columns,
            kdtree_leaf_size=1000, feature_set="full", nb_processes=2
        )
    with pytest.raises(ValueError):
        input_columns = ["dummy1", "dummy2", "dummy3"]
        features = extract(
            sphere, nb_neighbors=10, input_columns=input_columns,
            kdtree_leaf_size=1000, feature_set="full", nb_processes=2
        )
    features = extract(
        sphere, nb_neighbors=10, input_columns=["x", "y", "z"],
        kdtree_leaf_size=1000, feature_set="full", nb_processes=2
    )
    assert features.shape[0] == sphere.shape[0]


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
    assert first_item[1].shape == (NB_NEIGHBORS + 1,)
    assert len(list(gen)) == sphere.shape[0] - 1


def test_sequence_full(sphere):
    """Test the sequence building in the case of "full" feature set
    """
    NB_NEIGHBORS = 10
    tree = compute_tree(sphere, leaf_size=500)
    gen = sequence_full(sphere, tree, nb_neighbors=NB_NEIGHBORS)
    first_item = next(gen)
    assert len(first_item) == 3
    assert first_item[0].shape == (NB_NEIGHBORS + 1, 3)
    assert first_item[1].shape == (NB_NEIGHBORS + 1,)
    assert first_item[2].shape == (3,)
    assert len(list(gen)) == sphere.shape[0] - 1


def test_process_alphabeta(sphere):
    """Test the alphabeta feature set processing for the first "sphere" point
    """
    input_columns = ["x", "y", "z"]
    additional_features = ["alpha", "beta"]
    tree = compute_tree(sphere, leaf_size=500)
    gen = sequence_light(sphere, tree, nb_neighbors=10)
    features = process_alphabeta(next(gen)[0], input_columns=input_columns)
    assert len(features) == len(input_columns) + 2
    assert features["x"] == sphere[0, 0]
    assert features["y"] == sphere[0, 1]
    assert features["z"] == sphere[0, 2]
    assert list(features.keys()) == input_columns + additional_features


def test_process_eigenvalues(sphere):
    """Test the eigenvalues feature set processing for the first "sphere" point
    """
    input_columns = ["x", "y", "z"]
    additional_features = [
        "alpha", "beta",
        "curvature_change", "linearity", "planarity",
        "scattering", "omnivariance", "anisotropy",
        "eigenentropy", "eigenvalue_sum"
    ]
    tree = compute_tree(sphere, leaf_size=500)
    gen = sequence_light(sphere, tree, nb_neighbors=10)
    features = process_eigenvalues(next(gen)[0], input_columns=input_columns)
    assert len(features) == len(input_columns) + 10
    assert features["x"] == sphere[0, 0]
    assert features["y"] == sphere[0, 1]
    assert features["z"] == sphere[0, 2]
    assert list(features.keys()) == input_columns + additional_features


def test_process_full(sphere):
    """Test the full feature set processing for the first "sphere" point
    """
    input_columns = ["x", "y", "z"]
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
    gen = sequence_full(sphere, tree, nb_neighbors=10)
    item = next(gen)
    features = process_full(
        item[0], item[1], item[2], input_columns=input_columns
    )
    assert len(features) == len(input_columns) + 22
    assert features["x"] == sphere[0, 0]
    assert features["y"] == sphere[0, 1]
    assert features["z"] == sphere[0, 2]
    assert list(features.keys()) == input_columns + additional_features
