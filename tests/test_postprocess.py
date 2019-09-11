"""Run test on functions within the postprocess.py module
"""

from pathlib import Path
import pytest

import numpy as np

from geo3dfeatures import postprocess


_here = Path(__file__).resolve().parent
DATADIR = _here / "data"


def test_batch_points(sphere):
    """Test the batching process: one may produce several tables of length
    "batch_size"
    """
    BATCH_SIZE = 1_000
    nb_points = sphere.shape[0]
    batched_sphere = postprocess.batch_points(sphere, BATCH_SIZE)
    sphere_item = next(batched_sphere)
    assert sphere_item.shape == (BATCH_SIZE, sphere.shape[1])
    expected_items = nb_points // BATCH_SIZE
    assert expected_items == len(list(batched_sphere)) + 1


def test_postprocess_batch_labels(sphere, sphere_tree):
    """Test the postprocessing itself by fixing a bunch of labels

    Let define a k-mean output with 1 as the first label, and the remaining
    values are 0. The postprocessing must output an array of 0.
    """
    BATCH_SIZE = 1_500
    NB_NEIGHBORS = 100
    LABELS = np.zeros((sphere.shape[0],), dtype=int)
    LABELS[0] = 1
    batched_sphere = postprocess.batch_points(sphere, BATCH_SIZE)
    new_labels = postprocess.postprocess_batch_labels(
        batched_sphere, BATCH_SIZE, LABELS, sphere_tree, n_neighbors=NB_NEIGHBORS
    )
    assert np.all([new_labels == 0])
