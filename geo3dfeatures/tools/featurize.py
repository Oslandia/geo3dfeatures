import sys
import pickle
from pathlib import Path

import numpy as np

from geo3dfeatures.io import (
    xyz as read_xyz,
    las as read_las,
    ply as read_ply,
    )
from geo3dfeatures.extract import extract


def main(opts):
    input_path = Path(opts.datapath, "input", opts.input_file)
    if input_path.suffix == ".xyz":
        data = read_xyz(str(input_path))
    elif input_path.suffix == ".las":
        data = read_las(str(input_path))
    elif input_path.suffix == ".ply":
        data = read_ply(str(input_path))
    else:
        raise ValueError("Wrong file extension, please send xyz or las file.")

    if len(opts.input_columns) != data.shape[1]:
        raise ValueError("The given input columns does not match data shape.")

    if opts.sample_points is not None:
        sample_mask = np.random.choice(np.arange(data.shape[0]),
                                       size=opts.sample_points,
                                       replace=False)
        data = data[sample_mask]

    experiment = (
        opts.experiment
        if opts.experiment is not None
        else opts.input_file.split(".")[0]
        )
    tree_file = opts.tree_file
    if not tree_file:
        tree_file = Path(
            opts.datapath, "output", experiment,
            "kd-tree-leaf-" + str(opts.kdtree_leafs) + ".pkl"
        )
        if not tree_file.exists():
            print("No serialized kd-tree with",
                  f"leaf size = {opts.kdtree_leafs}.",
                  "Please index your point cloud with the 'index' command,",
                  "then use the --tree-file or the -t/--kdtree-leafs option.")
            sys.exit(0)

    with open(tree_file, 'rb') as fobj:
        print("load kd-tree from file")
        tree = pickle.load(fobj)

    instance = (
        "features-" + str(len(data)) + "-" + str(opts.neighbors) + "-"
        + str(opts.feature_set) + "-" + str(opts.nb_process)
        )
    output_path = Path(opts.datapath, "output", experiment, "features")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = Path(output_path, instance + ".csv")

    # XXX fix the number of arbitrary columns
    extract(
        data[:, :3], tree, opts.neighbors, output_file, opts.feature_set, opts.nb_process)
    print("Results in {}".format(output_file))
