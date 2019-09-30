"""Train a logistic regression model to predict 3D point semantic class
"""

import json
from pathlib import Path
import pickle

import daiquiri
import pandas as pd
from sklearn.metrics import confusion_matrix

from geo3dfeatures import classification
from geo3dfeatures import io
from geo3dfeatures.tools import GLOSSARY, EXPERIMENTS


logger = daiquiri.getLogger(__name__)

SEED = 1337


def create_training_dataset(datapath, experiment, neighborhood_sizes, labels):
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
    labels : dict
        Dataset glossary

    Returns
    -------
    pd.DataFrame
        Shuffled training dataset, without point coordinates
    """
    dfs = []
    for label in labels.keys():
        df = io.load_features(datapath, experiment, neighborhood_sizes, label)
        if df is not None:
            df["label"] = labels[label]["id"]
            dfs.append(df)
    df = pd.concat(dfs, axis=0)
    return df.sample(frac=1.).drop(columns=["x", "y", "z"])


def main(opts):
    logger.info("Prepare the training dataset...")
    if opts.input_file is None:
        logger.info("Train a model with all the available samples...")
        experiment = "logreg"
        dfs = []
        for expe in EXPERIMENTS:
            dfs.append(
                create_training_dataset(
                    opts.datapath, expe, opts.neighbors, GLOSSARY
                    )
                )
        df = pd.concat(dfs, axis=0)
        X_train, Y_train, X_test, Y_test = classification.split_dataset(df)
    else:
        experiment = opts.input_file.split(".")[0]
        logger.info("Train a model with %s data...", experiment)
        df = create_training_dataset(
            opts.datapath, experiment, opts.neighbors, GLOSSARY
        )
        X_train, Y_train, X_test, Y_test = classification.split_dataset(df)

    logger.info("Train the classifier...")
    clf = classification.train_predictive_model(X_train, Y_train, SEED)

    logger.info("Evaluate the model and store the results...")
    accuracy_score = clf.score(X_test, Y_test)
    Y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    evaluation = {
        "score": accuracy_score,
        "confusion_matrix": conf_matrix.tolist()
        }
    model_dir = Path(opts.datapath, "trained_models")
    model_dir.mkdir(exist_ok=True)
    eval_filename = (
        experiment + "-" + io.instance(opts.neighbors, None) + ".json"
        )
    with open(model_dir / eval_filename, "w") as fobj:
        json.dump(evaluation, fobj)

    logger.info("Serialize the classifier...")
    model_filename = eval_filename.replace(".json", ".pkl")
    with open(model_dir / model_filename, "wb") as fobj:
        pickle.dump(clf, fobj)
