
from pathlib import Path
import pytest

from geo3dfeatures.tools.train import create_training_dataset

from geo3dfeatures import io


_here = Path(__file__).absolute().parent
DATADIR = _here / "data"
LABELS = {"vegetation": 0, "roof": 1, "ground": 2}
NEIGHBORS = [10, 50, 200]
N_FEATURES = 19


def test_create_training_dataset():
    """Test the creation of a training dataset for implementing a supervised
    learning algorithm.

    One tests the case where there is no sampled dataset to concatenate (pandas
    raises a ValueError), and a classic case where some label-related samples
    are available.
    """
    EMPTY_EXPERIMENT = "test"
    with pytest.raises(ValueError):
        dataset = create_training_dataset(
            DATADIR, EMPTY_EXPERIMENT, NEIGHBORS, LABELS
        )
    VALID_EXPERIMENT = "b9"
    ground_data = io.read_ply(DATADIR / "input" / "b9_ground.ply")
    vegetation_data = io.read_ply(DATADIR / "input" / "b9_vegetation.ply")
    roof_data = io.read_ply(DATADIR / "input" / "b9_roof.ply")
    dataset = create_training_dataset(
        DATADIR, VALID_EXPERIMENT, NEIGHBORS, LABELS
    )
    assert dataset.shape[0] == (
        len(ground_data) + len(vegetation_data) + len(roof_data)
        )
    assert dataset.shape[1] == (
        1  # labels
        + len(NEIGHBORS) * N_FEATURES  # geometric features
        )
