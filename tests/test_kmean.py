
from configparser import NoOptionError
from pathlib import Path
import pytest

import numpy as np
import pandas as pd

from geo3dfeatures.tools.kmean import (
    add_accumulation_features, update_features
    )

from geo3dfeatures import io

_here = Path(__file__).absolute().parent
CONFIG_PATH = Path(_here, "config")


def test_add_accumulation_features():
    """Test the addition of accumulation features in a 3D point cloud
    dataframe. For the test purpose, one considers two spaces:
    - the first one is defined on [0,1]^3, with bin size equal to 1. In such a
    scenario, the dataset is decomposed in one single bin, and bin feature are
    equal to 0 as they do not discriminate the points.
    - the second space is defined on [0, 2]^3, with bin size equal to 1. Here
    there are four bins.

    As a side test, one controls the existence of the "bin" key in the
    configuration file.
    """
    data = np.random.rand(100, 3)  # First scenario
    df = pd.DataFrame(data, columns=list("xyz"))
    nobin_config = io.read_config(CONFIG_PATH / "base.ini")
    with pytest.raises(NoOptionError):  # No-bin-key side scenario
        df_acc = add_accumulation_features(df.copy(), nobin_config)
    valid_config = io.read_config(CONFIG_PATH / "bin.ini")
    df_acc = add_accumulation_features(df.copy(), valid_config)
    assert float(valid_config.get("clustering", "bin")) == 1.
    assert df_acc.shape[1] == df.shape[1] + 3
    bins = df_acc[["bin_density", "bin_z_range", "bin_z_std"]].drop_duplicates()
    assert len(bins) == 1
    assert np.all(bins.values == [0, 0, 0])
    df *= 2  # Second scenario
    df_acc = add_accumulation_features(df.copy(), valid_config)
    bins = df_acc[["bin_density", "bin_z_range", "bin_z_std"]].drop_duplicates()
    assert len(bins) == 4


def test_update_features():
    """Test the feature update procedure, based on a given configuration file.

    One tests:
    - empty configuration file (no modification)
    - unvalid key, which means unknown feature names (key 'foo' in "bin.ini")
    - valid key with value != 1 and one-lettered feature name, which modifies
    the feature values (key 'z' in "bin.ini")
    - valid key with value != 1, which modifies the feature values (key
    "bin_density" in "bin.ini")
    """
    data = np.random.rand(10, 3) * 2
    df = pd.DataFrame(data, columns=list("xyz"))
    up_df = df.copy()
    base_config = io.read_config(CONFIG_PATH / "base.ini")
    update_features(up_df, base_config)
    for feature in up_df:
        assert np.all(up_df[feature] == df[feature])
    bin_config = io.read_config(CONFIG_PATH / "bin.ini")
    df = add_accumulation_features(df, bin_config)
    up_df = df.copy()
    update_features(up_df, bin_config)
    z_coef = float(bin_config.get("clustering", "z"))
    assert np.all(df["z"] * z_coef == up_df["z"])
    bin_density_coef = float(bin_config.get("clustering", "bin_density"))
    assert np.all(df["bin_density"] * bin_density_coef == up_df["bin_density"])
