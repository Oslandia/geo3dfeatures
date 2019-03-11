import os
import sys
import argparse
from pathlib import Path

import numpy as np

from geo3dfeatures.io import xyz as read_xyz, las as read_las, write_features
from geo3dfeatures.extract import (
    alphabeta_features, eigen_features, all_features
)
from geo3dfeatures import FEATURE_SETS

def _parse_args(args):
    parser = argparse.ArgumentParser(description=("3D point cloud geometric"
                                                  " feature extraction"))
    parser.add_argument('-c', '--input-columns',
                        default=["x", "y", "z"], nargs="+",
                        help="Input point cloud feature names")
    parser.add_argument("-d", "--datapath", default="./data",
                        help="Data folder on the file system")
    parser.add_argument("-e", "--experiment",
                        help="Name of the feature extraction experiment")
    parser.add_argument("-i", "--input-file",
                        help="Input point cloud file")
    parser.add_argument('-f', '--feature-set', choices=FEATURE_SETS,
                        help="Set of computed features")
    parser.add_argument('-n', '--neighbors',
                        type=int, default=50,
                        help="Number of neighbors to consider")
    parser.add_argument('-p', '--sample-points',
                        type=int,
                        help="Number of sample points to evaluate")
    parser.add_argument('-t', '--kdtree-leafs',
                        type=int, default=1000,
                        help="Number of leafs in KD-tree")
    return parser.parse_args(args)


def main(argv=sys.argv[1:]):
    opts = _parse_args(argv)

    input_path = Path(opts.datapath, "input", opts.input_file)
    if input_path.suffix == ".xyz":
        data = read_xyz(str(input_path))
    elif input_path.suffix == ".las":
        data = read_las(str(input_path))
    else:
        raise ValueError("Wrong file extension, please send xyz or las file.")

    if len(opts.input_columns) != data.shape[1]:
        raise ValueError("The given input columns does not match data shape.")

    if opts.sample_points is not None:
        sample_mask = np.random.choice(np.arange(data.shape[0]),
                                       size=opts.sample_points,
                                       replace=False)
        data = data[sample_mask]

    if opts.feature_set == "alphabeta":
        gen = alphabeta_features(
            data, opts.neighbors, opts.input_columns, opts.kdtree_leafs
        )
    elif opts.feature_set == "eigenvalues":
        gen = eigen_features(
            data, opts.neighbors, opts.input_columns, opts.kdtree_leafs
        )
    elif opts.feature_set == "full":
        gen = all_features(
            data, opts.neighbors, opts.input_columns, opts.kdtree_leafs
        )
    else:
        raise ValueError("Choose a valid feature set amongst %s", FEATURE_SETS)

    experiment = (
        opts.experiment
        if opts.experiment is not None
        else opts.input_file.split(".")[0]
        )
    instance = (
        "features-" + str(len(data)) + "-"
        + str(opts.neighbors) + "-" + str(opts.feature_set)
        )
    print(instance)
    output_path = Path(opts.datapath, "output", experiment, "features")
    os.makedirs(output_path, exist_ok=True)
    output_file = Path(output_path, instance + ".csv")
    print(output_file)
    write_features(output_file, gen)

if __name__ == '__main__':
    main()
