"""CLI for Geo3Dfeatures

Available choices::
    - geo3d sample [args]
    - geo3d info [args]
    - geo3d index [args]
    - geo3d featurize [args]
    - geo3d kmean [args]
    - geo3d train [args]
    - geo3d predict [args]
"""

import argparse
from pathlib import Path

from geo3dfeatures.tools import (
    info, sample, featurize, kmean, index, train, predict
    )


DATADIR = Path("data")
INPUT_DIR = DATADIR / "input"
KD_TREE_LEAF_SIZE = 500
N_JOBS = 2
CHUNKSIZE = 20_000
PP_NEIGHBORS = 0
TRAIN_CONFIG_FILENAME = "base.ini"
KMEAN_CONFIG_FILENAME = "base.ini"


def sample_parser(subparser, reference_func):
    """Add arguments focused on data generation process

    Parameters
    ----------
    subparser : argparser.parser.SubParsersAction
    reference_func : function
    """
    parser = subparser.add_parser(
        "sample", help="Extract a sample of a .las file"
    )
    add_data_args(parser, by_dataset=True)
    parser.add_argument(
        '-s', '--sample-points', type=int, required=True,
        help="Number of sample points to evaluate"
    )
    parser.set_defaults(func=reference_func)


def info_parser(subparser, reference_func):
    """Add arguments for describing a point cloud file

    Parameters
    ----------
    subparser : argparser.parser.SubParsersAction
    reference_func : function
    """
    parser = subparser.add_parser(
        "info", help="Describe an input .las file"
    )
    add_data_args(parser, by_dataset=True)
    parser.set_defaults(func=reference_func)


def index_parser(subparser, reference_func):
    """Index a point cloud scene.

    Parameters
    ----------
    subparser : argparser.parser.SubParsersAction
    reference_func : function
    """
    parser = subparser.add_parser(
        "index", help="Index a point cloud file and serialize it"
    )
    add_data_args(parser, by_dataset=True)
    add_kdtree_args(parser)
    parser.set_defaults(func=reference_func)


def featurize_parser(subparser, reference_func):
    """Add instance-specific arguments from the command line

    Parameters
    ----------
    subparser : argparser.parser.SubParsersAction
    reference_func : function
    """
    parser = subparser.add_parser(
        "featurize",
        help="Extract the geometric feature associated to 3D points"
    )
    add_data_args(parser, by_dataset=True)
    add_kdtree_args(parser)
    add_neighborhood_args(parser)
    parser.add_argument(
        "-m", "--nb-process", type=int, default=N_JOBS,
        help="Number of process used during the computation"
    )
    parser.add_argument(
        '-c', '--extra-columns', nargs="+",
        help="Extra point cloud feature names (other than x,y,z)"
    )
    parser.add_argument(
        '--label-scene', action="store_true",
        help=(
            "If the scene is a sample of an existing "
            "scene with the '_label.{las, xyz}' suffix?"
        )
    )
    parser.add_argument(
        "--chunksize", default=CHUNKSIZE, type=int,
        help="Number of points in each writing chunk"
    )
    parser.set_defaults(func=reference_func)


def kmean_parser(subparser, reference_func):
    """Add instance-specific arguments from the command line

    Build a high-resolution image labelling by predicting semantic segmentation
    labels on image patches, and by postprocessing resulting arrays so as to
    get geographic entities.

    Parameters
    ----------
    subparser : argparser.parser.SubParsersAction
    reference_func : function
    """
    parser = subparser.add_parser(
        "cluster", help="Cluster a set of 3D points with a k-means algorithm"
    )
    add_data_args(parser, by_dataset=True)
    add_kdtree_args(parser)
    add_neighborhood_args(parser)
    add_label_prediction_args(parser)
    parser.add_argument(
        "-k", "--nb-clusters", type=int, required=True,
        help="Desired amount of clusters"
    )
    parser.add_argument(
        "-c", "--config-file", default=KMEAN_CONFIG_FILENAME,
        help="Clustering analysis config file; summarizes feature coefficients"
    )
    parser.set_defaults(func=reference_func)


def train_parser(subparser, reference_func):
    """Add instance-specific arguments from the command line

    Train a supervised learning model for predicting 3D point semantic
    classes.

    Parameters
    ----------
    subparser : argparser.parser.SubParsersAction
    reference_func : function
    """
    parser = subparser.add_parser(
        "train", help="Train a semantic segmentation model"
    )
    add_data_args(parser, by_dataset=False)
    add_neighborhood_args(parser)
    parser.add_argument(
        "-c", "--config-file", default=TRAIN_CONFIG_FILENAME,
        help="Classification config file; summarizes used feature"
    )
    parser.set_defaults(func=reference_func)


def predict_parser(subparser, reference_func):
    """Add instance-specific arguments from the command line

    Use a logistic regression trained model in order to predict 3D point
    semantic classes in a point cloud.

    Parameters
    ----------
    subparser : argparser.parser.SubParsersAction
    reference_func : function
    """
    parser = subparser.add_parser(
        "predict",
        help="Predict 3D point semantic class starting from a trained model"
    )
    add_data_args(parser, by_dataset=True)
    add_kdtree_args(parser)
    add_neighborhood_args(parser)
    add_label_prediction_args(parser)
    parser.add_argument(
        "-c", "--config-file", default=TRAIN_CONFIG_FILENAME,
        help="Classification config file; summarizes used feature"
    )
    parser.add_argument(
        "-g", "--generalized-model", action="store_true",
        help=(
            "If true, one considers a generalized model, "
            "otherwise it is a dataset-specific classifier."
            )
        )
    parser.set_defaults(func=reference_func)


def add_data_args(parser, by_dataset):
    """Add a bunch of command arguments that permits to identify the instance
    of interest

    Parameters
    ----------
    parser : argparse.ArgumentParser
    by_dataset : bool
        True if the "input-file" argument is required; false otherwise
    ("featurize", "kmean" and "predict" programs)
    """
    parser.add_argument(
        "-d", "--datapath", default=DATADIR, type=Path,
        help="Data folder on the file system"
    )
    parser.add_argument(
        "-i", "--input-file", required=by_dataset,
        help="Input point cloud file"
    )


def add_kdtree_args(parser):
    """Add arguments related to kd-tree serialization

    Parameters
    ----------
    parser : argparse.ArgumentParser
    """
    kdtree_group = parser.add_mutually_exclusive_group()
    kdtree_group.add_argument(
        "--tree-file", type=Path,
        help=(
            "kd-tree serialized filepath (alternative definition: "
            "-t/-kdtree-leafs for using kd-trees in the "
            "'datapath/output/dataset/' folder)"
            )
    )
    kdtree_group.add_argument(
        '-t', '--kdtree-leafs', type=int, default=KD_TREE_LEAF_SIZE,
        help=(
            "Number of leafs in the KD-tree, stored in "
            "'<datapath>/output/<dataset>/' folder (alternative definition: "
            "--tree-file for specifying directly the kd-tree filepath)"
            )
    )


def add_neighborhood_args(parser):
    """
    """
    neighbor_group = parser.add_mutually_exclusive_group()
    neighbor_group.add_argument(
        '-n', '--neighbors', nargs="+", type=int,
        help=(
            "List of neighborhood sizes to consider. "
            "Alternative neighborhood definition: -r/--radius."
        )
    )
    neighbor_group.add_argument(
        '-r', '--radius', type=float,
        help=(
            "Radius that defines the neighboring ball. "
            "Alternative neighborhood definition: -n/--neighbors."
        )
    )


def add_label_prediction_args(parser):
    """Add command arguments that describe the label prediction process:
    - -p/--postprocessing-neighbors denotes the neighborhood size to consider
    when running the post-processing step
    - -xyz means the dataset must be saved as a .xyz file (instead of a .las
    file)

    Parameters
    ----------
    parser : argparse.ArgumentParser
    """
    parser.add_argument(
        "-p", "--postprocessing-neighbors", type=int, default=PP_NEIGHBORS,
        help=(
            "Clean the prediction output by postprocessing the result. "
            "The parameter gives the postprocessing neighborhood definition, "
            "as a neighboring point quantity. If 0, no postprocessing."
            )
        )
    parser.add_argument(
        "-xyz", action="store_true",
        help=(
            "Output file extension: .xyz if true, similar to input otherwise."
        )
    )


def main():
    """Main method of the module
    """
    parser = argparse.ArgumentParser(
        prog="geo3d",
        description="Geo3dfeatures framework for 3D semantic analysis",
    )
    sub_parsers = parser.add_subparsers(dest="command")
    sample_parser(sub_parsers, reference_func=sample.main)
    info_parser(sub_parsers, reference_func=info.main)
    index_parser(sub_parsers, reference_func=index.main)
    featurize_parser(sub_parsers, reference_func=featurize.main)
    kmean_parser(sub_parsers, reference_func=kmean.main)
    train_parser(sub_parsers, reference_func=train.main)
    predict_parser(sub_parsers, reference_func=predict.main)
    args = parser.parse_args()

    if args.func:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
