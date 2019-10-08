"""Predict point semantic classes in 3D point cloud starting from a trained
model.
"""

from pathlib import Path
import pickle

import daiquiri

from geo3dfeatures.extract import compute_tree
from geo3dfeatures import classification
from geo3dfeatures import io
from geo3dfeatures import postprocess
from geo3dfeatures.tools import GLOSSARY


logger = daiquiri.getLogger(__name__)

POSTPROCESSING_BATCH = 10_000


def main(opts):
    experiment = opts.input_file.split(".")[0]
    logger.info("Load data from %s dataset...", experiment)
    df = io.load_features(opts.datapath, experiment, opts.neighbors)

    logger.info("Load the trained classifier...")
    model_dir = Path(opts.datapath, "trained_models")
    if opts.generalized_model:
        model_filename = "logreg-" + io.instance(opts.neighbors, None) + ".pkl"
    else:
        model_filename = experiment + "-" + io.instance(opts.neighbors, None) + ".pkl"
    with open(model_dir / model_filename, "rb") as fobj:
        clf = pickle.load(fobj)

    logger.info("Predict labels...")
    points = df[["x", "y", "z"]].copy()
    df.drop(columns=["x", "y", "z"], inplace=True)
    labels = clf.predict(df)

    # Postprocessing
    if opts.postprocessing_neighbors > 0:
        logger.info(
            "Post-process point labels by batches of %s", POSTPROCESSING_BATCH
        )
        tree = compute_tree(points, opts.kdtree_leafs)
        gen = postprocess.batch_points(points, POSTPROCESSING_BATCH)
        labels = postprocess.postprocess_batch_labels(
            gen, POSTPROCESSING_BATCH, labels, tree, opts.postprocessing_neighbors
        )

    logger.info("Save predictions on disk...")
    outdf = classification.colorize_labels(points, labels, GLOSSARY)
    classification.save_labels(
        outdf, opts.datapath, experiment, opts.neighbors, opts.radius,
        algorithm="logreg", config_name="full",
        pp_neighbors=opts.postprocessing_neighbors, xyz=False
    )
