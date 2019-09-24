"""Predict point semantic classes in 3D point cloud starting from a trained
model.
"""

from pathlib import Path
import pickle

import daiquiri
import seaborn as sns

from geo3dfeatures.extract import compute_tree
from geo3dfeatures import classification
from geo3dfeatures import io
from geo3dfeatures import postprocess


logger = daiquiri.getLogger(__name__)

POSTPROCESSING_BATCH = 10_000
PALETTE = sns.color_palette("colorblind")
GLOSSARY = {
    "vegetation": {"id": 0, "color": PALETTE[2]},  # Vegetation: green
    "falaise": {"id": 1, "color": PALETTE[7]},  # Vegetation: grey
    "eboulis": {"id": 2, "color": PALETTE[5]},  # Vegetation: marron
    "route": {"id": 3, "color": PALETTE[0]},  # Vegetation: blue
    "structure": {"id": 4, "color": PALETTE[1]},  # Vegetation: orange
    }


def save_predictions(
        results, datapath, experiment, neighbors, radius,
        config_name, pp_neighbors, xyz=False
):
    """Save the resulting dataframe into the accurate folder on the file system

    Parameters
    ----------
    results : pandas.DataFrame
        Data to save
    datapath : str
        Root of the data folder
    experiment : str
        Name of the experiment, used for identifying the accurate subfolder
    neighbors : int
        Number of neighbors used to compute the feature set
    radius : float
        Threshold that define the neighborhood, in order to compute the feature
    set; used if neighbors is None
    config_name : str
        Feature configuration filename
    pp_neighbors : int
        If >0, the predicted labels are postprocessed, otherwise they are
    outputs of the supervised learning algorithm
    xyz : boolean
        If true, the output file is a .xyz, otherwise a .las file will be
    produced
    """
    output_path = Path(
        datapath, "output", experiment, "prediction",
    )
    output_path.mkdir(exist_ok=True)
    extension = "xyz" if xyz else "las"
    postprocess_suffix = (
        "-pp" + str(pp_neighbors) if pp_neighbors > 0 else ""
        )
    output_file_path = Path(
        output_path,
        "logreg-" + io.instance(neighbors, radius)
        + "-" + config_name + postprocess_suffix + "." + extension
    )
    if xyz:
        io.write_xyz(results, output_file_path)
    else:
        input_file_path = Path(datapath, "input", experiment + ".las")
        io.write_las(results, input_file_path, output_file_path)
    logger.info("Predictions saved into %s", output_file_path)


def main(opts):
    experiment = opts.input_file.split(".")[0]
    df = io.load_features(opts.datapath, experiment, opts.neighbors)
    points = df[["x", "y", "z"]].copy()

    logger.info("Load the trained classifier...")
    model_dir = Path(opts.datapath, "trained_models")
    model_filename = experiment + ".pkl"
    with open(model_dir / model_filename, "rb") as fobj:
        clf = pickle.load(fobj)

    logger.info("Predict labels...")
    labels = clf.predict(df.drop(columns=["x", "y", "z"]))

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
