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
    """
    BATCH_SIZE = 1_000
    NB_NEIGHBORS = 100
    LABELS = np.zeros((sphere.shape[0],), dtype=int)
    LABELS[0] = 1
    batched_sphere = postprocess.batch_points(sphere, BATCH_SIZE)
    new_labels = postprocess.postprocess_batch_labels(
        batched_sphere, LABELS, sphere_tree, n_neighbors=NB_NEIGHBORS
    )
    assert new_labels[0] != LABELS[0]
    assert new_labels[0] == 0
    assert sum(new_labels) == 0
