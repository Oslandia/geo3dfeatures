"""Train a logistic regression model to predict 3D point semantic class
"""

from pathlib import Path
import pickle

import daiquiri
import pandas as pd

from geo3dfeatures import classification
from geo3dfeatures import io


logger = daiquiri.getLogger(__name__)

SEED = 1337
LABELS = {
    "vegetation": 0,
    "falaise": 1,
    "eboulis": 2,
    "route": 3,
    "structure": 4
    }


def create_training_dataset(datapath, experiment, neighborhood_sizes):
    """Create a training dataset that will feed the classifier in the training
    step

    Parameters
    ----------
    datapath : str
        Root of the data folder
    experiment : str
        Name of the experiment, used for identifying the accurate subfolder
    neighbors : list
        List of number of neighbors

    Returns
    -------
    pd.DataFrame
        Shuffled training dataset, without point coordinates
    """
    dfs = []
    for label in LABELS.keys():
        df = io.load_features(datapath, experiment, neighborhood_sizes, label)
        if df is not None:
            df["label"] = LABELS[label]
            dfs.append(df)
    df = pd.concat(dfs, axis=0)
    return df.sample(frac=1.).drop(columns=["x", "y", "z"])


def main(opts):
    logger.info("Prepare the training dataset...")
    experiment = opts.input_file.split(".")[0]
    dataset = create_training_dataset(
        opts.datapath, experiment, opts.neighbors
    )
    print(dataset.shape)

    logger.info("Train the classifier...")
    clf = classification.train_predictive_model(
        dataset.drop(columns=["label"]), dataset["label"], SEED
        )

    logger.info("Serialize the classifier...")
    model_dir = Path(opts.datapath, "trained_models")
    model_dir.mkdir(exist_ok=True)
    model_filename = experiment + ".pkl"
    with open(model_dir / model_filename, "wb") as fobj:
        pickle.dump(clf, fobj)
