"""Deeposlandia central command-line interface

Available choices::
    - deepo datagen [args]
    - deepo train [args]
    - deepo infer [args]
    - deepo postprocess [args]
"""

import argparse
import os

from geo3dfeatures.tools import sample, featurize, profiling, kmean
from geo3dfeatures import FEATURE_SETS


def sample_parser(subparser, reference_func):
    """Add arguments focused on data generation process

    Parameters
    ----------
    subparser : argparser.parser.SubParsersAction
    reference_func : function
    """
    parser = subparser.add_parser(
        "sample",
        help="Extract a sample of a .las file"
    )
    parser.add_argument("-d", "--datapath",
                        default="./data",
                        help="Data folder on the file system")
    parser.add_argument("-i", "--input-file",
                        required=True,
                        help="Input point cloud file")
    parser.add_argument('-p', '--sample-points',
                        type=int, required=True,
                        help="Number of sample points to evaluate")
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
    add_instance_args(parser, featurized=False)
    parser.add_argument('-c', '--input-columns',
                        default=["x", "y", "z"], nargs="+",
                        help="Input point cloud feature names")
    parser.add_argument("-i", "--input-file",
                        required=True,
                        help="Input point cloud file")
    parser.add_argument('-t', '--kdtree-leafs',
                        type=int, default=1000,
                        help="Number of leafs in KD-tree")
    parser.add_argument("-m", "--nb-process",
                        type=int, default=1,
                        help="")
    parser.set_defaults(func=reference_func)


def profiling_parser(subparser, reference_func):
    """Add instance-specific arguments from the command line

    Parameters
    ----------
    subparser : argparser.parser.SubParsersAction
    reference_func : function
    """
    parser = subparser.add_parser(
        "profile",
        help=(
            "Extract in a human-readable format the "
            "geometric feature extraction time measurements"
        )
    )
    parser.add_argument("-e", "--experiment",
                        required=True,
                        help="Name of the feature extraction experiment")
    parser.add_argument("-F", "--file-format",
                        choices=["csv", "json"], default="csv",
                        help="Timer file format")
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
        "cluster",
        help="Cluster a set of 3D points with a k-means algorithm"
    )
    add_instance_args(parser, featurized=True)
    parser.add_argument("-k", "--nb-clusters",
                        type=int, required=True,
                        help="Desired amount of clusters")
    parser.set_defaults(func=reference_func)


def add_instance_args(parser, featurized=True):
    """Add a bunch of command arguments that permits to identify the instance
    of interest

    Parameters
    ----------
    parser : argparse.ArgumentParser
    """
    parser.add_argument("-d", "--datapath",
                        default="./data",
                        help="Data folder on the file system")
    parser.add_argument("-e", "--experiment",
                        required=featurized,
                        help="Name of the feature extraction experiment")
    parser.add_argument('-f', '--feature-set',
                        choices=FEATURE_SETS, default="full",
                        help="Set of computed features")
    parser.add_argument('-n', '--neighbors',
                        type=int, default=50, required=featurized,
                        help="Number of neighbors to consider")
    parser.add_argument('-p', '--sample-points',
                        type=int,
                        required=featurized,
                        help="Number of sample points to evaluate")


def main():
    """Main method of the module
    """
    parser = argparse.ArgumentParser(
        prog="geo3d",
        description="Geo3dfeatures framework for 3D semantic analysis",
    )
    sub_parsers = parser.add_subparsers(dest="command")
    sample_parser(sub_parsers, reference_func=sample.main)
    featurize_parser(sub_parsers, reference_func=featurize.main)
    profiling_parser(sub_parsers, reference_func=profiling.main)
    kmean_parser(sub_parsers, reference_func=kmean.main)
    args = parser.parse_args()

    if args.func:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
