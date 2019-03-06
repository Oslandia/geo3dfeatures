import pytest

from geo3dfeatures.extract import (
    alphabeta_features, eigen_features, all_features
    )

def test_alphabeta_features(sphere):
    """Test the alphabeta feature set generation
    """
    with pytest.raises(ValueError):
        input_columns = ["x", "y"]
        ab_gen = alphabeta_features(
            sphere, input_columns=input_columns, nb_neighbors=50
        )
        first_item = next(ab_gen)
    with pytest.raises(ValueError):
        input_columns = ["dummy1", "dummy2", "dummy3"]
        ab_gen = alphabeta_features(
            sphere, input_columns=input_columns, nb_neighbors=50
        )
        first_item = next(ab_gen)
    input_columns = ["x", "y", "z"]
    ab_gen = alphabeta_features(
        sphere, input_columns=input_columns, nb_neighbors=50
    )
    first_item = next(ab_gen)
    assert len(first_item) == len(input_columns) + 2
    assert list(first_item.keys()) == input_columns + ["alpha", "beta"]
    assert len(list(ab_gen)) == sphere.shape[0] - 1


def test_eigen_features(sphere):
    """Test the eigenvalue feature set generation
    """
    with pytest.raises(ValueError):
        input_columns = ["x", "y"]
        eigen_gen = eigen_features(
            sphere, input_columns=input_columns, nb_neighbors=50
        )
        first_item = next(eigen_gen)
    with pytest.raises(ValueError):
        input_columns = ["dummy1", "dummy2", "dummy3"]
        eigen_gen = eigen_features(
            sphere, input_columns=input_columns, nb_neighbors=50
        )
        first_item = next(eigen_gen)
    input_columns = ["x", "y", "z"]
    eigen_gen = eigen_features(
        sphere, input_columns=input_columns, nb_neighbors=50
    )
    first_item = next(eigen_gen)
    assert len(first_item) == len(input_columns) + 10
    additional_features = [
        "alpha", "beta", "curvature_change",
        "linearity", "planarity", "scattering", "omnivariance",
        "anisotropy", "eigenentropy", "eigenvalue_sum"
    ]
    assert list(first_item.keys()) == input_columns + additional_features
    assert len(list(eigen_gen)) == sphere.shape[0] - 1


def test_full_features(sphere):
    """Test the full feature set generation
    """
    with pytest.raises(ValueError):
        input_columns = ["x", "y"]
        full_gen = all_features(
            sphere, input_columns=input_columns, nb_neighbors=50
        )
        first_item = next(full_gen)
    with pytest.raises(ValueError):
        input_columns = ["dummy1", "dummy2", "dummy3"]
        full_gen = all_features(
            sphere, input_columns=input_columns, nb_neighbors=50
        )
        first_item = next(full_gen)
    input_columns = ["x", "y", "z"]
    full_gen = all_features(
        sphere, input_columns=input_columns, nb_neighbors=50
    )
    first_item = next(full_gen)
    additional_features = [
        "alpha", "beta", "radius",
        "z_range", "std_dev", "density", "verticality",
        "curvature_change", "linearity", "planarity",
        "scattering", "omnivariance", "anisotropy",
        "eigenentropy", "eigenvalue_sum",
        "radius_2D", "density_2D", "eigenvalue_sum_2D", "eigenvalue_ratio_2D",
        "bin_density", "bin_z_range", "bin_z_std"
    ]
    assert len(first_item) == len(input_columns) + 22
    assert list(first_item.keys()) == input_columns + additional_features
    assert len(list(full_gen)) == sphere.shape[0] - 1
